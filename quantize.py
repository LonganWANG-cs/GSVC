from vector_quantize_pytorch import VectorQuantize, ResidualVQ
import inspect
import os
import torch
from torch import nn
import torch.nn.functional as F
import constriction
import numpy as np


# vector-quantize-pytorch >= 1.31 enables the rotation trick by default for
# vector dimensions greater than one.  Older releases do not expose this
# option and use the standard straight-through estimator.  Keep that behavior
# explicit while remaining compatible with both package versions.
_STANDARD_STE_KWARGS = (
    {"rotation_trick": False}
    if "rotation_trick" in inspect.signature(VectorQuantize.__init__).parameters
    else {}
)



def grad_scale(x, scale):
    return (x - x * scale).detach() + x * scale

def ste(x):
    return (x.round() - x).detach() + x

def fake_quantize_fp16_ste(x):
    """FP16 fake quantization with a straight-through gradient, without a custom
    autograd Function.

    Bit-exact with ``FakeQuantizationFP16.apply``: ``h = x.half().float()`` lies
    within a factor of two of ``x``, so by Sterbenz's lemma ``h - x`` is exact in
    float32 and ``x + (h - x)`` is exactly ``h``.  The subtraction is detached,
    so the derivative is the identity -- the same straight-through rule the
    Function implements.

    The reason to prefer this form is capture, not speed: a custom
    ``autograd.Function``'s backward is dispatched by the autograd engine, which
    is the mechanism that made stage 1's step impossible to record as a CUDA
    graph.  Written in plain ATen the backward stays on the capture stream.
    ``FakeQuantizationFP16`` is kept for its bit-accounting helpers, which the
    decoder and the bpp path still use.
    """
    return x + (x.half().float() - x).detach()


class FakeQuantizationFP16(torch.autograd.Function):
    """performs fake quantization for half precision"""

    @staticmethod
    def forward(_, x):
        return x.half().float()

    @staticmethod
    def backward(_, grad_output):
        return grad_output

    @staticmethod
    @torch.no_grad()
    def _encode_to_uint16_code(x: torch.Tensor):
        """
        Convert float Tensor to FP16 bit-pattern symbols.

        Each FP16 value is represented by one uint16 symbol.
        These symbols are then entropy-coded by size().
        """
        if not torch.is_floating_point(x):
            x = x.float()

        x_half_np = (
            x.detach()
             .to(torch.float16)
             .contiguous()
             .cpu()
             .numpy()
        )

        code = x_half_np.view(np.uint16).astype(np.int32)

        return code

    @staticmethod
    @torch.no_grad()
    def size_raw(x: torch.Tensor):
        """Raw FP16 storage size without entropy coding."""
        return x.numel() * 16

    @staticmethod
    @torch.no_grad()
    def size(x: torch.Tensor):
        code = FakeQuantizationFP16._encode_to_uint16_code(x).reshape(-1)
        code = np.asarray(code, dtype=np.uint16)

        if code.size == 0:
            return 0

        low_byte = (code & 0x00FF).astype(np.uint8)
        high_byte = ((code >> 8) & 0x00FF).astype(np.uint8)

        compressed_low, histogram_low = compress_matrix_flatten_categorical_hardcode_unique(
            low_byte,
            qmin=0,
            qmax=255,
        )

        compressed_high, histogram_high = compress_matrix_flatten_categorical_hardcode_unique(
            high_byte,
            qmin=0,
            qmax=255,
        )

        bits = 0
        bits += get_np_size(compressed_low) * 8
        bits += get_np_size(histogram_low) * 8
        bits += get_np_size(compressed_high) * 8
        bits += get_np_size(histogram_high) * 8

        return int(bits)
 

class FakeQuantizationFP12(nn.Module):
    """Capture-safe FP12 fake quantization with a straight-through gradient.

    The fixed format is ``sign:1 | exponent:5 | mantissa:6`` with exponent
    bias 15. It has no learned scale/offset and always uses 12 raw bits per
    scalar; no entropy coding is applied.
    """

    def __init__(self):
        super().__init__()
        self.exponent_bits = 5
        self.mantissa_bits = 6
        self.exponent_bias = 15

    def _quantize_to_float(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_floating_point(x):
            x = x.float()
        abs_x = x.abs()
        zero_mask = abs_x == 0
        mantissa, exponent = torch.frexp(abs_x)
        mantissa = mantissa * 2.0
        exponent = (exponent - 1).clamp(
            1 - self.exponent_bias, self.exponent_bias)
        levels = float(1 << self.mantissa_bits)
        mantissa_code = torch.round((mantissa - 1.0) * levels)
        mantissa_code = mantissa_code.clamp(0, levels - 1)
        mantissa_q = 1.0 + mantissa_code / levels
        quantized = torch.ldexp(mantissa_q, exponent) * torch.sign(x)
        return torch.where(zero_mask, torch.zeros_like(quantized), quantized)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        quantized = self._quantize_to_float(x)
        # FP12 in the forward pass and identity gradient in the backward pass.
        return x + (quantized - x).detach()

    @torch.no_grad()
    def _encode_to_int16_code(self, x: torch.Tensor) -> torch.Tensor:
        """Return FP12 symbols in an int16 container (low 12 bits are used)."""
        if not torch.is_floating_point(x):
            x = x.float()
        abs_x = x.abs()
        zero_mask = abs_x == 0
        mantissa, exponent = torch.frexp(abs_x)
        mantissa = mantissa * 2.0
        exponent = (exponent - 1).clamp(
            1 - self.exponent_bias, self.exponent_bias)
        levels = float(1 << self.mantissa_bits)
        mantissa_code = torch.round((mantissa - 1.0) * levels)
        mantissa_code = mantissa_code.clamp(0, levels - 1).to(torch.int32)
        exponent_code = (exponent + self.exponent_bias).to(torch.int32)
        exponent_code.clamp_(0, (1 << self.exponent_bits) - 1)
        sign_code = (x < 0).to(torch.int32)
        code = (sign_code << 11) | (exponent_code << 6) | mantissa_code
        code = torch.where(zero_mask, torch.zeros_like(code), code)
        return (code & 0xFFF).to(torch.int16)

    @staticmethod
    @torch.no_grad()
    def _pack_12bit(code: torch.Tensor) -> torch.Tensor:
        """Pack two raw 12-bit symbols into three bytes without entropy coding."""
        code = code.to(torch.int32).flatten() & 0xFFF
        if code.numel() % 2:
            code = torch.cat((code, code.new_zeros(1)))
        first = code[0::2]
        second = code[1::2]
        byte0 = first & 0xFF
        byte1 = ((first >> 8) & 0x0F) | ((second & 0x0F) << 4)
        byte2 = (second >> 4) & 0xFF
        return torch.stack((byte0, byte1, byte2), dim=1).flatten().to(torch.uint8)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self._pack_12bit(self._encode_to_int16_code(x))

    @staticmethod
    @torch.no_grad()
    def decompress(packed: torch.Tensor, num_values: int) -> torch.Tensor:
        """Unpack the raw FP12 byte stream emitted by ``encode``."""
        packed = packed.to(torch.int32).flatten()
        expected_bytes = ((num_values + 1) // 2) * 3
        if packed.numel() != expected_bytes:
            raise ValueError(f"FP12 payload has {packed.numel()} bytes; expected {expected_bytes}")
        triples = packed.reshape(-1, 3)
        first = triples[:, 0] | ((triples[:, 1] & 0x0F) << 8)
        second = (triples[:, 1] >> 4) | (triples[:, 2] << 4)
        code = torch.stack((first, second), dim=1).flatten()[:num_values]
        mantissa = 1.0 + (code & 0x3F).float() / 64.0
        exponent = ((code >> 6) & 0x1F) - 15
        value = torch.ldexp(mantissa, exponent)
        value = torch.where((code & 0x800) != 0, -value, value)
        return torch.where(code == 0, torch.zeros_like(value), value)

    @staticmethod
    @torch.no_grad()
    def size(x: torch.Tensor) -> int:
        """Raw storage size; deliberately does not entropy-code FP12 symbols."""
        return int(x.numel()) * 12


class UniformQuantizer_one(nn.Module):
    def __init__(self, signed=False, bits=8, learned=False, num_channels=1, entropy_type="none", weight=0.001):
        super().__init__()

        if isinstance(signed, (list, tuple)):
            if len(signed) != num_channels:
                raise ValueError(f"len(signed)={len(signed)} must equal num_channels={num_channels}")
            signed_list = [bool(v) for v in signed]
        else:
            signed_list = [bool(signed)] * num_channels

        if isinstance(bits, (list, tuple)):
            if len(bits) != num_channels:
                raise ValueError(f"len(bits)={len(bits)} must equal num_channels={num_channels}")
            bits_list = [int(v) for v in bits]
        else:
            bits_list = [int(bits)] * num_channels

        qmin, qmax = [], []
        for is_signed, bit in zip(signed_list, bits_list):
            if is_signed:
                qmin.append(-2 ** (bit - 1))
                qmax.append(2 ** (bit - 1) - 1)
            else:
                qmin.append(0)
                qmax.append(2 ** bit - 1)

        self.register_buffer("qmin", torch.tensor(qmin, dtype=torch.float32).view(1, num_channels))
        self.register_buffer("qmax", torch.tensor(qmax, dtype=torch.float32).view(1, num_channels))
        # Constant integer alphabet bounds, cached so size() need not sync.
        self._qmin_int = int(min(qmin))
        self._qmax_int = int(max(qmax))

        self.learned = learned
        self.entropy_type = entropy_type
        if self.learned:
            self.scale = nn.Parameter(torch.ones(num_channels), requires_grad=True)
            self.beta = nn.Parameter(torch.zeros(num_channels), requires_grad=True)

        self.weight = weight

    def _clamp_code(self, code):
        # qmin/qmax are registered buffers, so they already follow the module to
        # its device -- the per-call .to() was pure Python overhead. One clamp
        # with tensor bounds replaces maximum+minimum (two kernels -> one).
        return torch.clamp(code, self.qmin, self.qmax)

    def _entropy_proxy(self, quant):
        if self.entropy_type is None or self.entropy_type == "none" or self.weight <= 0:
            # A python float keeps the disabled case off the GPU entirely;
            # new_tensor(0.0) was a host->device copy plus a no-op graph node
            # on every iteration.
            return 0.0

        if self.entropy_type == "l1":
            return quant.abs().mean()

        if self.entropy_type == "l2":
            return (quant ** 2).mean()

        raise ValueError(f"Unknown entropy_type: {self.entropy_type}")

    def forward(self, x):
        if self.learned:
            # The LSQ-style grad_scale() rescaling that used to be computed here
            # was immediately overwritten by the assignment below, so it never
            # influenced the result -- but the float(...item()) it needed forced
            # a GPU->CPU sync on every training iteration. Removed; bit-exact.
            s_scale, beta_scale = self.scale, self.beta

            code = self._clamp_code((x - beta_scale) / s_scale)
            quant = ste(code)
            dequant = quant * s_scale + beta_scale
        else:
            code = self._clamp_code(x * self.qmax.to(device=x.device, dtype=x.dtype))
            quant = ste(code)
            dequant = quant / self.qmax.to(device=x.device, dtype=x.dtype)

        bits=0
        entropy_loss = self._entropy_proxy(quant)
        if not self.training:
            bits = self.size(quant)

        return dequant, entropy_loss * self.weight, bits

    def compress(self, x):
        code = self._clamp_code((x - self.beta) / self.scale)
        code = code.round()
        return code, code * self.scale + self.beta

    def decompress(self, x):
        return x * self.scale + self.beta

    def size(self, quant):
        # _qmin_int/_qmax_int are cached at construction: qmin/qmax are constant
        # buffers, so re-reducing them with .item() was two syncs per call.
        # Going through .cpu().numpy() instead of .tolist() avoids materialising
        # a multi-100k-element python list of ints for numpy to re-parse.
        compressed, histogram_table = compress_matrix_flatten_categorical_hardcode_unique(
            quant.detach().to(torch.int32).reshape(-1).cpu().numpy(),
            qmin=self._qmin_int,
            qmax=self._qmax_int,
        )
        bits = 0
        bits += get_np_size(compressed) * 8
        bits += get_np_size(histogram_table) * 8
        bits += self.scale.numel() * torch.finfo(self.scale.dtype).bits
        bits += self.beta.numel() * torch.finfo(self.beta.dtype).bits
        return bits

class UniformQuantizer(nn.Module):
    def __init__(self, signed=False, bits=8, learned=False, num_channels=1, entropy_type="none", weight=0.001):
        super().__init__()

        if isinstance(signed, (list, tuple)):
            if len(signed) != num_channels:
                raise ValueError(f"len(signed)={len(signed)} must equal num_channels={num_channels}")
            signed_list = [bool(v) for v in signed]
        else:
            signed_list = [bool(signed)] * num_channels

        if isinstance(bits, (list, tuple)):
            if len(bits) != num_channels:
                raise ValueError(f"len(bits)={len(bits)} must equal num_channels={num_channels}")
            bits_list = [int(v) for v in bits]
        else:
            bits_list = [int(bits)] * num_channels

        qmin, qmax = [], []
        for is_signed, bit in zip(signed_list, bits_list):
            if is_signed:
                qmin.append(-2 ** (bit - 1))
                qmax.append(2 ** (bit - 1) - 1)
            else:
                qmin.append(0)
                qmax.append(2 ** bit - 1)

        self.register_buffer("qmin", torch.tensor(qmin, dtype=torch.float32).view(1, num_channels))
        self.register_buffer("qmax", torch.tensor(qmax, dtype=torch.float32).view(1, num_channels))
        # Constant integer alphabet bounds, cached so size() need not sync.
        self._qmin_int = int(min(qmin))
        self._qmax_int = int(max(qmax))

        self.learned = learned
        self.entropy_type = entropy_type
        if self.learned:
            self.scale = nn.Parameter(torch.ones(num_channels)/self.qmax, requires_grad=True)
            self.beta = nn.Parameter(torch.ones(num_channels)/self.qmax, requires_grad=True)

        self.weight = weight
        self.bits=bits

    def _clamp_code(self, code):
        # qmin/qmax are registered buffers, so they already follow the module to
        # its device -- the per-call .to() was pure Python overhead. One clamp
        # with tensor bounds replaces maximum+minimum (two kernels -> one).
        return torch.clamp(code, self.qmin, self.qmax)

    def _entropy_proxy(self, quant):
        if self.entropy_type is None or self.entropy_type == "none" or self.weight <= 0:
            # A python float keeps the disabled case off the GPU entirely;
            # new_tensor(0.0) was a host->device copy plus a no-op graph node
            # on every iteration.
            return 0.0

        if self.entropy_type == "l1":
            return quant.abs().mean()

        if self.entropy_type == "l2":
            return (quant ** 2).mean()

        raise ValueError(f"Unknown entropy_type: {self.entropy_type}")

    def forward(self, x):
        if self.learned:
            # Same dead grad_scale() block as in UniformQuantizer_one: its result
            # was overwritten on the next line, but the float(...item()) forced a
            # GPU->CPU sync every iteration. Removed; bit-exact.
            s_scale, beta_scale = self.scale, self.beta

            code = self._clamp_code((x - beta_scale) / s_scale)
            quant = ste(code)
            dequant = quant * s_scale + beta_scale
        else:
            code = self._clamp_code(x * self.qmax.to(device=x.device, dtype=x.dtype))
            quant = ste(code)
            dequant = quant / self.qmax.to(device=x.device, dtype=x.dtype)

        bits = 0
        entropy_loss = self._entropy_proxy(quant)

        if not self.training:
            bits = self.size(quant)

        return dequant, entropy_loss * self.weight, bits

    def compress(self, x):
        code = self._clamp_code((x - self.beta) / self.scale)
        code = code.round()
        return code, code * self.scale + self.beta

    def decompress(self, x):
        return x * self.scale + self.beta
    
    def size(self, quant):
        # _qmin_int/_qmax_int are cached at construction: qmin/qmax are constant
        # buffers, so re-reducing them with .item() was two syncs per call.
        # Going through .cpu().numpy() instead of .tolist() avoids materialising
        # a multi-100k-element python list of ints for numpy to re-parse.
        compressed, histogram_table = compress_matrix_flatten_categorical_hardcode_unique(
            quant.detach().to(torch.int32).reshape(-1).cpu().numpy(),
            qmin=self._qmin_int,
            qmax=self._qmax_int,
        )
        bits = 0
        bits += get_np_size(compressed) * 8
        bits += get_np_size(histogram_table) * 8
        bits += self.scale.numel() * torch.finfo(self.scale.dtype).bits
        bits += self.beta.numel() * torch.finfo(self.beta.dtype).bits
        return bits

def _vq_noop(*args, **kwargs):
    """Stands in for an already-completed ``init_embed_`` (see
    ``VectorQuantizer.prepare_for_capture``)."""
    return None


class _CaptureSafeTensorCtor:
    """``torch.tensor`` replacement that builds device zeros without an H2D copy.

    ``vector_quantize_pytorch`` calls ``tensor(0., device=cuda, requires_grad=..)``
    once per forward to seed its loss accumulator.  Constructing a CUDA tensor
    from a python scalar copies from pageable host memory, which is illegal
    during CUDA-graph capture.  Cloning a cached device zero produces the same
    value with a device-side kernel.  Anything that is not a zero scalar with an
    explicit device falls through to the real constructor.
    """

    def __init__(self, original):
        self._original = original
        self._zeros = {}

    def __call__(self, data, *args, **kwargs):
        device = kwargs.get("device")
        scalar_zero = isinstance(data, (int, float)) and data == 0
        vector_zero = (
            isinstance(data, (list, tuple))
            and len(data) == 1
            and isinstance(data[0], (int, float))
            and data[0] == 0
        )
        if not args and device is not None and (scalar_zero or vector_zero):
            key = (str(device), str(kwargs.get("dtype")), vector_zero)
            zero = self._zeros.get(key)
            if zero is None:
                create_kwargs = dict(kwargs)
                create_kwargs.pop("requires_grad", None)
                zero = self._original(data, **create_kwargs)
                self._zeros[key] = zero
            out = zero.clone()
            if kwargs.get("requires_grad", False):
                out.requires_grad_(True)
            return out
        return self._original(data, *args, **kwargs)


class _CaptureSafeTorchProxy:
    """Module-local ``torch`` proxy for old vector-quantize-pytorch releases.

    Older releases call ``torch.tensor([0.], device=...)`` instead of importing
    ``tensor`` into the VQ module namespace.  Replacing ``torch.tensor`` itself
    would monkey-patch PyTorch process-wide, so replace only that module's
    ``torch`` global with this delegating proxy while a graph is captured.
    """

    def __init__(self, module):
        self._module = module
        self.tensor = _CaptureSafeTensorCtor(module.tensor)

    def __getattr__(self, name):
        return getattr(self._module, name)


# ---------------------------------------------------------------------------
# Specialised residual vector quantizer
# ---------------------------------------------------------------------------
# `vector_quantize_pytorch` is written for the general case, and the generality
# is what costs: profiling the captured stage-2 step (RESEARCH_LOG 2026-08-30)
# put the colour VQ at 347 of the step's 540 kernels and 38% of its GPU kernel
# time -- to quantize N RGB triples against an 8-entry, 3-dimensional codebook,
# an operation that is 10 kernels written directly.  The two worst offenders are
# the library's general-purpose dispatch and the einsum used to gather from the
# codebook.  GSVC explicitly uses the standard straight-through estimator
# for RGB/SH colour, so the newer library's rotation trick is not reproduced.
#
# `_fast_residual_vq` reproduces the library's forward for exactly the
# configuration this project uses, reading and writing the library modules' own
# buffers so state dicts stay byte-compatible and `compress`/`decompress`/`size`
# are untouched.  It is a *forward* rewrite: autograd still supplies the
# gradients, including the standard STE below.
#
# The configuration it assumes is asserted at first use (`_fast_vq_eligible`);
# anything unexpected falls back to the library.  Set GSVC_FAST_VQ=0 to
# force the library path.
#
# The tensor ranks and einsum expressions intentionally match
# vector-quantize-pytorch 1.14.8.  Replacing them with superficially equivalent
# 2-D matmuls or gathers changes floating-point accumulation and eventually the
# training trajectory.  Deterministic LBG only replaces the old random k-means
# seed; the subsequent VQ forward and update follow the old library design.

_FAST_VQ_ENABLED = os.environ.get("GSVC_FAST_VQ", "1") == "1"
_FAST_VQ_STATE = {}


def _fast_vq_eligible(rvq):
    """True if `rvq` is a supported standard-STE residual VQ.

    Checked once per module and cached: every generic feature of the library
    that the fast path does not reproduce must be off.  Returns False (and the
    caller uses the library) rather than raising, so an upgrade that flips a
    default degrades to "slow" and never to "wrong".
    """
    cached = _FAST_VQ_STATE.get(id(rvq))
    if cached is not None:
        return cached

    ok = True
    try:
        ok &= isinstance(rvq, ResidualVQ)
        ok &= not getattr(rvq, "quantize_dropout", False)
        ok &= not getattr(rvq, "implicit_neural_codebook", False)
        ok &= float(getattr(rvq, "quant_grad_frac", 0.0)) == 0.0
        ok &= isinstance(rvq.project_in, nn.Identity)
        ok &= isinstance(rvq.project_out, nn.Identity)
        ok &= getattr(rvq, "beam_size", None) in (None, 1)
        ok &= getattr(rvq, "eval_beam_size", None) in (None, 1)
        for layer in rvq.layers:
            c = layer._codebook
            ok &= isinstance(layer.project_in, nn.Identity)
            ok &= isinstance(layer.project_out, nn.Identity)
            ok &= layer.heads == 1 and layer.channel_last
            ok &= not bool(getattr(layer, "rotation_trick", False))
            ok &= bool(getattr(layer, "route_gradients_to_input", True))
            ok &= not getattr(layer, "directional_reparam", False)
            ok &= not layer.freeze_codebook
            ok &= float(layer.sync_update_v) == 0.0
            ok &= bool(layer.has_commitment_loss)
            ok &= not layer.commitment_use_cross_entropy_loss
            ok &= not layer.has_codebook_diversity_loss
            ok &= not layer.has_codebook_orthogonal_loss
            ok &= layer.in_place_codebook_optimizer is None
            ok &= not getattr(layer, "accept_image_fmap", False)
            ok &= not getattr(layer, "accept_3d_fmap", False)
            ok &= not c.manual_ema_update
            # Original VQ updates a buffer codebook with internal EMA;
            # learnable VQ updates a Parameter through commitment loss.
            ok &= bool(c.ema_update) != bool(layer.learnable_codebook)
            ok &= not c.affine_param and not c.use_cosine_sim
            ok &= not c.has_dead_code_replacement
            ok &= not getattr(c, "stochastic_sample_codes", False)
            ok &= getattr(c, "vq_bridge", None) is None
            ok &= tuple(c.embed.shape)[0] == 1          # single head
    except AttributeError:
        ok = False

    _FAST_VQ_STATE[id(rvq)] = bool(ok)
    return bool(ok)


def _fast_vq_layer(layer, r):
    """One `VectorQuantize` layer on `r` of shape [N, D]; returns (q, idx, loss)."""
    c = layer._codebook
    training = layer.training

    # k-means on the first call, exactly as the library does.  `prepare_for_capture`
    # stubs this out for the captured region (it is a no-op once initialised, but
    # its `if self.initted:` guard reads a CUDA buffer back to the host).
    c.init_embed_(r.unsqueeze(0))

    embed = c.embed if c.learnable_codebook else c.embed.detach()  # [1, K, D]
    embed = embed.to(r.dtype)
    flatten = r.unsqueeze(0)                                   # [1, N, D]

    # Exact 1.14.8 cdist expression and gumbel_sample's deterministic branch.
    x2 = (flatten ** 2).sum(dim=-1)                            # [1, N]
    y2 = (embed ** 2).sum(dim=-1)                              # [1, K]
    xy = torch.einsum("b i d, b j d -> b i j", flatten, embed) * -2
    dist = (x2.unsqueeze(-1) + y2.unsqueeze(1) + xy).clamp(min=0).sqrt()
    idx_h = (-dist).argmax(dim=-1)                             # [1, N]
    onehot_h = F.one_hot(idx_h, embed.shape[1]).type(r.dtype)  # [1, N, K]

    idx = idx_h.squeeze(0)                                     # [N]
    if training:
        # Old VectorQuantize temporarily represents [N,D] as [N,1,D] and uses
        # a one-hot einsum during training.
        unpacked_onehot = onehot_h.unsqueeze(2)                # [1, N, 1, K]
        q = torch.einsum(
            "h b n c, h c d -> h b n d", unpacked_onehot, embed
        ).squeeze(0).squeeze(1)                                # [N, D]
    else:
        # 1.14.8 uses batched_embedding (indexed gather) in eval mode.
        q = embed[0][idx]

    if training and c.ema_update:
        # EMA codebook update, on the library's own buffers.  Both sums stay
        # matmuls: `index_add_` would reorder them and, being atomic, would make
        # the update nondeterministic.
        cluster_size = onehot_h.sum(dim=1)                     # [1, K]
        embed_sum = torch.einsum(
            "h n d, h n c -> h c d", flatten, onehot_h
        ).contiguous()                                         # [1, K, D]
        decay = c.decay
        c.cluster_size.data.lerp_(cluster_size, 1.0 - decay)
        c.embed_avg.data.lerp_(embed_sum, 1.0 - decay)
        # update_ema: laplace smoothing, renormalise, write back to `embed`.
        cs = c.cluster_size
        smoothed = (
            (cs + c.eps)
            / (cs.sum(dim=-1, keepdim=True) + embed.shape[1] * c.eps)
        ) * cs.sum(dim=-1, keepdim=True)
        c.embed.data.copy_(c.embed_avg / smoothed.unsqueeze(-1))

    if not training:
        return q, idx, r.new_zeros(())

    # Match vector-quantize-pytorch: an original/EMA codebook is detached from
    # commitment loss, while a learnable codebook receives its gradient here.
    commit_q = q if c.learnable_codebook else q.detach()
    commit = F.mse_loss(commit_q, r)

    # Standard straight-through estimator: forward value is q, while the
    # reconstruction gradient is passed back to the matching RGB/SH channel.
    q = r + (q - r).detach()

    return q, idx, commit * layer.commitment_weight


def _fast_residual_vq(rvq, x):
    """`ResidualVQ.forward` for the plain EMA configuration; same return shape."""
    quantized_out = torch.zeros_like(x)
    residual = x
    idxs, losses = [], []
    for layer in rvq.layers:
        q, idx, loss = _fast_vq_layer(layer, residual)
        residual = residual - q.detach()          # quant_grad_frac == 0
        quantized_out = quantized_out + q
        idxs.append(idx)
        losses.append(loss)
    return quantized_out, torch.stack(idxs, dim=-1), torch.stack(losses, dim=-1)


# ---- deterministic codebook design (GSVC_LBG_INIT) ---------------------
#
# The library initialises a codebook by k-means from a `torch.randperm` sample
# of the input rows, which is the only RNG left in stage 2.  LBG binary
# splitting (Linde-Buzo-Gray) is the classical deterministic alternative and is
# what VQ codec design has used for exactly this job: start from the centroid of
# the data, repeatedly split the highest-distortion cluster, run Lloyd after
# each split.  It takes any K (the P-frame delta codebook is 7, the I-frame's
# is 8) and consumes no randomness at all.
#
# Cost is irrelevant here: this runs ONCE per codebook per frame, against
# thousands of training iterations.  With K=8, D=3, N<=18000 one distance pass
# is ~400k FLOPs, so the library's frugal `kmeans_iters=5` buys nothing and the
# defaults below run Lloyd to convergence instead.
#
# Every reduction is a matmul rather than `index_add_`: index_add_ is atomic,
# and atomics would reorder the sums and make this init nondeterministic --
# defeating the entire purpose.  `_fast_vq_layer` avoids index_add_ for the
# same reason.

def _lbg_assign(x, c):
    """Nearest-centroid assignment and squared distances, library tie-break."""
    # Same expression as _fast_vq_layer's: x^2 + c^2 - 2xc.  Only the argmin is
    # used, but computing it identically keeps ties breaking the same way.
    d2 = (x * x).sum(-1, keepdim=True) + (c * c).sum(-1) - 2.0 * (x @ c.t())
    d2 = d2.clamp(min=0.0)
    return d2.argmin(dim=1), d2


def _lbg_lloyd(x, c, iters, tol):
    """Lloyd iterations. Returns (centroids, counts). Empty clusters are kept."""
    counts = None
    for _ in range(iters):
        assign, _ = _lbg_assign(x, c)
        onehot = F.one_hot(assign, c.shape[0]).to(x.dtype)      # [N, K]
        counts = onehot.sum(dim=0)                              # [K]
        sums = onehot.t() @ x                                   # [K, D]
        nz = (counts > 0).unsqueeze(1)
        new = torch.where(nz, sums / counts.clamp(min=1).unsqueeze(1), c)
        shift = (new - c).abs().max()
        c = new
        if float(shift) <= tol:      # host sync; once per frame, never captured
            break
    if counts is None:
        assign, _ = _lbg_assign(x, c)
        counts = F.one_hot(assign, c.shape[0]).to(x.dtype).sum(dim=0)
    return c, counts


@torch.no_grad()
def lbg_init(x, k, split_iters=3, final_iters=50, tol=1e-7):
    """Deterministic K-centroid codebook for x [N, D]. Returns (embed, counts).

    Greedy splitting rather than the textbook 1->2->4->8 doubling, so K need not
    be a power of two.  The split direction is the displacement to the chosen
    cluster's farthest member: deterministic, never degenerate for a cluster
    with any spread, and it needs no eigen-decomposition.
    """
    x = x.detach()
    if x.shape[0] == 0:
        return x.new_zeros((k, x.shape[1])), x.new_zeros(k)
    if x.shape[0] <= k:
        # Fewer points than codes: use the points, pad by repeating the last.
        c = torch.cat([x, x[-1:].expand(k - x.shape[0], -1)], dim=0).contiguous()
        return _lbg_lloyd(x, c, final_iters, tol)

    c = x.mean(dim=0, keepdim=True)                             # [1, D]
    while c.shape[0] < k:
        assign, d2 = _lbg_assign(x, c)
        onehot = F.one_hot(assign, c.shape[0]).to(x.dtype)
        own = d2.gather(1, assign.unsqueeze(1))                 # [N, 1]
        sse = (onehot.t() @ own).squeeze(1)                     # [K]
        j = int(sse.argmax())
        members = x[assign == j]
        if members.shape[0] >= 2:
            off = members - c[j]
            far = members[int(off.pow(2).sum(-1).argmax())]
            delta = 0.5 * (far - c[j])
        else:
            delta = None
        if delta is None or float(delta.abs().max()) == 0.0:
            # Degenerate cluster (a single point, or all members identical).
            # Perturb along one coordinate axis, chosen by the codebook index so
            # repeated degenerate splits do not all collapse onto each other.
            scale = x.std(dim=0).max().clamp(min=1e-6) * 1e-3
            delta = torch.zeros_like(c[j])
            delta[c.shape[0] % c.shape[1]] = scale
        c = torch.cat([c, (c[j] - delta).unsqueeze(0)], dim=0).contiguous()
        c[j] = c[j] + delta
        c, _ = _lbg_lloyd(x, c, split_iters, tol)
    return _lbg_lloyd(x, c, final_iters, tol)


class VectorQuantizer(nn.Module):
    def __init__(self, num_quantizers=1, codebook_dim=1, codebook_size=64, kmeans_iters=10, vector_type="vector"):
        super().__init__()
        self.num_quantizers = num_quantizers
        self.vector_type = vector_type
        if self.num_quantizers == 1:
            if self.vector_type == "vector":
                self.quantizer = VectorQuantize(dim=codebook_dim, codebook_size=codebook_size, decay = 0.8, commitment_weight = 1., kmeans_init = True,
                    **_STANDARD_STE_KWARGS,
                    kmeans_iters = kmeans_iters)#learnable_codebook=True, ema_update = False, orthogonal_reg_weight =1.)
        else:
            if self.vector_type == "vector":
                self.quantizer = ResidualVQ(dim=codebook_dim, codebook_size=codebook_size, num_quantizers=num_quantizers, decay = 0.8, commitment_weight = 1., kmeans_init = True,
                    **_STANDARD_STE_KWARGS,
                    kmeans_iters = kmeans_iters) #learnable_codebook=True, ema_update=False, orthogonal_reg_weight=0., in_place_codebook_optimizer=torch.optim.Adam)

    def _uses_fast_path(self):
        return _FAST_VQ_ENABLED and self.num_quantizers > 1 \
            and self.vector_type == "vector" and _fast_vq_eligible(self.quantizer)

    def _vq_call(self, x):
        """Dispatch to the specialised residual VQ when it applies."""
        if self._uses_fast_path():
            return _fast_residual_vq(self.quantizer, x)
        return self.quantizer(x)

    def forward(self, x):
        if self.training:
            x, _, l_vq = self._vq_call(x)
            l_vq = torch.sum(l_vq)
            return x, l_vq, 0
        else:
            num_points, num_channels = x.shape
            x, embed_index, l_vq = self._vq_call(x)
            l_vq = torch.sum(l_vq)
            bits = self.size(embed_index)
            # unit_bit = bits / num_points / num_channels
            return x, l_vq, bits

    def prepare_for_capture(self):
        """Make this quantizer's forward legal inside a CUDA-graph capture.

        ``vector_quantize_pytorch`` performs two device->host synchronizations on
        every forward.  Outside a graph they merely stall the host; inside a
        capture they are illegal ("operation not permitted when stream is
        capturing").  Both are semantic no-ops, and both are removed here:

        1. ``EuclideanCodebook.init_embed_`` opens with ``if self.initted:``,
           where ``initted`` is a CUDA bool buffer -- evaluating it in a python
           ``if`` copies a byte back to the host.  Once the codebook has been
           k-means initialised the method does nothing else, so it is replaced
           by a no-op on each already-initialised codebook.  This refuses to run
           on an uninitialised codebook rather than silently skipping the
           initialisation.
        2. ``VectorQuantize.forward`` seeds its loss accumulator with
           ``tensor(0., device=device)``, an H2D copy of a python scalar (the
           same hazard as the ``new_tensor(0.0)`` removed from the uniform
           quantizers).  ``tensor`` is a module-global in that file, so it is
           swapped for one that clones a cached device zero instead.

        Returns a callable that undoes both patches.  Leaving them in place is
        harmless -- they change no value -- but the caller should undo them so
        the library is not left monkey-patched beyond the captured region.
        """
        import vector_quantize_pytorch.vector_quantize_pytorch as vqm

        undo = []
        layers = getattr(self.quantizer, "layers", None) or [self.quantizer]
        for layer in layers:
            codebook = layer._codebook
            if not bool(codebook.initted):
                raise RuntimeError(
                    "VectorQuantizer.prepare_for_capture: codebook is not "
                    "k-means initialised yet -- run some eager forwards first"
                )
            codebook.init_embed_ = _vq_noop
            undo.append(lambda cb=codebook: cb.__dict__.pop("init_embed_", None))

        # The specialised fast forward creates its loss from existing device
        # tensors and never calls the library's tensor constructor.  Only patch
        # that constructor when falling back to the original library forward.
        if not self._uses_fast_path() and not getattr(
                vqm, "_fastgsvc_tensor_patched", False):
            original_tensor = getattr(vqm, "tensor", None)
            original_torch = getattr(vqm, "torch", None)

            if original_tensor is not None:
                vqm.tensor = _CaptureSafeTensorCtor(original_tensor)

                def _restore(module=vqm, fn=original_tensor):
                    module.tensor = fn
                    module._fastgsvc_tensor_patched = False
            elif original_torch is not None:
                vqm.torch = _CaptureSafeTorchProxy(original_torch)

                def _restore(module=vqm, torch_module=original_torch):
                    module.torch = torch_module
                    module._fastgsvc_tensor_patched = False
            else:
                raise RuntimeError(
                    "VectorQuantizer.prepare_for_capture: unsupported "
                    "vector-quantize-pytorch tensor constructor"
                )

            vqm._fastgsvc_tensor_patched = True
            undo.append(_restore)

        def undo_all():
            for fn in undo:
                fn()

        return undo_all

    # ---- warm-start across frames (GSVC_WARM_START_VQ) -----------------
    #
    # The library k-means-initialises each codebook on its first forward, from a
    # `torch.randperm` sample of the input rows.  On a P-frame that input is the
    # colour *delta*, a near-degenerate cloud around zero, so the sampled start
    # is both seed-sensitive and a poor guess.  Consecutive frames of a video
    # have near-identical colour statistics, so the previous frame's *converged*
    # codebook is a strictly better starting point and carries no RNG at all.
    #
    # The full EMA state is carried, not just `embed`: `_fast_vq_layer` updates
    # `embed` by dividing `embed_avg` by the Laplace-smoothed `cluster_size`, so
    # seeding `embed` alone would leave the first EMA step to divide by zeros.
    #
    # This changes only the *initial value* of a trained parameter.  The
    # codebook is still transmitted per frame (`features_codebooks`), so the
    # bitstream format and the decoder are untouched.

    def _codebook_layers(self):
        return list(getattr(self.quantizer, "layers", None) or [self.quantizer])

    @torch.no_grad()
    def export_state(self):
        """Full codebook EMA state, one entry per residual layer, on the CPU.

        Returns None if any layer is still uninitialised -- a codebook that
        never ran a forward (e.g. `features_dc_quantizer_I` on a P-frame with
        no new Gaussians) holds library defaults, not a trained codebook, and
        must never be handed on as one.
        """
        out = []
        for layer in self._codebook_layers():
            c = layer._codebook
            if not bool(c.initted):
                return None
            out.append({
                "embed": c.embed.detach().clone().cpu(),
                "embed_avg": c.embed_avg.detach().clone().cpu(),
                "cluster_size": c.cluster_size.detach().clone().cpu(),
            })
        return out

    @torch.no_grad()
    def warm_start_(self, state):
        """Seed the codebooks from `export_state` output; True if it took.

        Every layer is validated before any is written, so a rejected warm start
        leaves the module exactly as it was and the first forward runs the
        library's k-means as usual.  Like `_fast_vq_eligible`, a shape or
        attribute mismatch degrades to the stock path rather than raising: the
        I-frame codebook is size 8 and the P-frame delta codebook size 7, and
        that mismatch is a routine "no warm start available", not an error.
        """
        layers = self._codebook_layers()
        if not state or len(state) != len(layers):
            return False
        try:
            for layer, s in zip(layers, state):
                c = layer._codebook
                for k in ("embed", "embed_avg", "cluster_size"):
                    if tuple(getattr(c, k).shape) != tuple(s[k].shape):
                        return False
        except (AttributeError, KeyError, TypeError):
            return False
        for layer, s in zip(layers, state):
            c = layer._codebook
            c.embed.data.copy_(s["embed"].to(c.embed.device))
            c.embed_avg.data.copy_(s["embed_avg"].to(c.embed_avg.device))
            c.cluster_size.data.copy_(s["cluster_size"].to(c.cluster_size.device))
            c.initted.data.fill_(True)
        return True

    @staticmethod
    @torch.no_grad()
    def _seed_codebook_(codebook, embed, counts):
        """Write one layer's codebook and mark it initialised.

        `embed_avg` must be seeded as embed*cluster_size and `cluster_size` as
        the real assignment counts: the EMA update recovers the codebook as
        embed_avg / laplace_smoothed(cluster_size), so any other pairing makes
        the first EMA step jump.  This is the same contract `warm_start_`
        honours by carrying all three buffers, and the same one the library's
        own `init_embed_` follows after its k-means.
        """
        embed = embed.to(codebook.embed.dtype)
        counts = counts.to(codebook.cluster_size.dtype)
        codebook.embed.data.copy_(embed.unsqueeze(0))
        codebook.embed_avg.data.copy_((embed * counts.unsqueeze(1)).unsqueeze(0))
        codebook.cluster_size.data.copy_(counts.unsqueeze(0))
        codebook.initted.data.fill_(True)

    @torch.no_grad()
    def lbg_init_(self, x):
        """Deterministically design every layer's codebook from `x` [N, D].

        Replaces the library's random-sample k-means, so the first forward finds
        `initted` already true and `init_embed_` becomes a no-op.  Residual
        layers are designed in cascade on the running residual, matching how
        `_fast_residual_vq` feeds them at training time.

        Returns True on success, False (leaving the module untouched, so the
        stock k-means still runs) if the codebook geometry is not what this
        expects -- the same degrade-to-stock contract as `warm_start_`.
        """
        layers = self._codebook_layers()
        try:
            for layer in layers:
                e = layer._codebook.embed
                if e.dim() != 3 or e.shape[0] != 1 or e.shape[-1] != x.shape[-1]:
                    return False
        except (AttributeError, IndexError):
            return False

        residual = x.detach()
        for layer in layers:
            c = layer._codebook
            embed, counts = lbg_init(residual, c.embed.shape[-2])
            self._seed_codebook_(c, embed, counts)
            idx, _ = _lbg_assign(residual, embed.to(residual.dtype))
            residual = residual - embed.to(residual.dtype).index_select(0, idx)
        return True

    def compress(self, x):
        x, embed_index, _ = self._vq_call(x)
        return x, embed_index

    def decompress(self, embed_index):
        recon = 0
        for i,layer in enumerate(self.quantizer.layers):
            recon += layer._codebook.embed[0, embed_index[:, i]]
        return recon
    
    def size(self, embed_index):
        if self.num_quantizers == 1:
            if self.vector_type == "vector":
                codebook_bits = self.quantizer._codebook.embed.numel() * torch.finfo(
                    self.quantizer._codebook.embed.dtype
                ).bits
                codebook_size = self.quantizer._codebook.embed.shape[-2]
            elif self.vector_type == "ste":
                codebook_bits = self.quantizer.embedding.weight.data.numel() * torch.finfo(
                    self.quantizer.embedding.weight.data.dtype
                ).bits
                codebook_size = self.quantizer.embedding.weight.data.shape[0]
            else:
                raise NotImplementedError
        else:
            codebook_bits = 0
            codebook_size = None

            for layer in self.quantizer.layers:
                if self.vector_type == "vector":
                    codebook_bits += layer._codebook.embed.numel() * torch.finfo(
                        layer._codebook.embed.dtype
                    ).bits
                    codebook_size = layer._codebook.embed.shape[-2]
                elif self.vector_type == "ste":
                    codebook_bits += layer.embedding.weight.data.numel() * torch.finfo(
                        layer.embedding.weight.data.dtype
                    ).bits
                    codebook_size = layer.embedding.weight.data.shape[0]
                else:
                    raise NotImplementedError

        compressed, histogram_table = compress_matrix_flatten_categorical_hardcode_unique(
            embed_index.detach().to(torch.int32).reshape(-1).cpu().numpy(),
            qmin=0,
            qmax=codebook_size - 1,
        )

        index_bits = 0
        index_bits += get_np_size(compressed) * 8
        index_bits += get_np_size(histogram_table) * 8

        total_bits = codebook_bits + index_bits
        return total_bits


class LearnableVectorQuantizer(VectorQuantizer):
    """Gradient-trained codebook with deterministic init and standard STE.

    Internal codebook EMA is disabled.  The codebook Parameter is optimized
    through commitment loss and is included in the model-level EMA together
    with the Gaussian and LSQ parameters.
    """

    def __init__(
        self,
        num_quantizers=1,
        codebook_dim=1,
        codebook_size=64,
        kmeans_iters=10,
        vector_type="vector",
    ):
        nn.Module.__init__(self)
        self.num_quantizers = num_quantizers
        self.vector_type = vector_type
        if vector_type != "vector":
            raise NotImplementedError("Learnable VQ supports vector mode only")

        common = dict(
            dim=codebook_dim,
            codebook_size=codebook_size,
            decay=0.8,
            commitment_weight=1.0,
            kmeans_init=True,
            kmeans_iters=kmeans_iters,
            learnable_codebook=True,
            ema_update=False,
            **_STANDARD_STE_KWARGS,
        )
        if num_quantizers == 1:
            self.quantizer = VectorQuantize(**common)
        else:
            self.quantizer = ResidualVQ(num_quantizers=num_quantizers, **common)

def _make_categorical_model(probabilities):
    """Build the same exact model on constriction 0.3.5 and 0.5+.

    Version 0.3.5 implements the behavior later named ``perfect=True`` but
    does not accept that keyword.  Version 0.5 changes the default, so request
    it explicitly when the installed API supports it.
    """
    try:
        return constriction.stream.model.Categorical(
            probabilities, perfect=True)
    except TypeError as exc:
        if "perfect" not in str(exc):
            raise
        return constriction.stream.model.Categorical(probabilities)


def compress_matrix_flatten_categorical_hardcode_unique(matrix, qmin, qmax):
    symbols = np.asarray(matrix, dtype=np.int32).reshape(-1)
    if symbols.size == 0:
        alphabet_size = int(qmax - qmin + 1)
        return (
            np.array([], dtype=np.uint8),
            np.zeros(alphabet_size, dtype=np.int32),
        )
    qmin = int(qmin)
    qmax = int(qmax)
    alphabet_size = qmax - qmin + 1
    message = symbols - qmin
    if message.min() < 0 or message.max() >= alphabet_size:
        raise ValueError(
            f"Symbol out of fixed alphabet range: "
            f"symbol range=({symbols.min()}, {symbols.max()}), "
            f"alphabet=[{qmin}, {qmax}]"
        )
    histogram_table = np.bincount(
        message,
        minlength=alphabet_size
    ).astype(np.int32)
    probabilities = histogram_table.astype(np.float64) / np.sum(histogram_table).astype(np.float64)
    # Keep the bitstream compatible with constriction <= 0.3.5 and make the
    # choice explicit before constriction 0.5 changes its default.
    entropy_model = _make_categorical_model(probabilities)
    encoder = constriction.stream.stack.AnsCoder()
    encoder.encode_reverse(message.astype(np.int32), entropy_model)
    compressed = encoder.get_compressed()
    return compressed, histogram_table

def decompress_matrix_flatten_categorical_hardcode_unique(
    compressed,
    histogram_table,
    qmin,
    symbol_length,
    symbol_shape,
):
    histogram_table = np.asarray(histogram_table, dtype=np.int32)
    if symbol_length == 0:
        return np.array([], dtype=np.int32).reshape(symbol_shape)
    probabilities = histogram_table.astype(np.float64) / np.sum(histogram_table).astype(np.float64)
    entropy_model = _make_categorical_model(probabilities)
    decoder = constriction.stream.stack.AnsCoder(compressed)
    message = decoder.decode(entropy_model, symbol_length)
    symbols = message.astype(np.int32) + int(qmin)
    return symbols.reshape(symbol_shape)


def judege_type(min, max):
    if min>=0:
        if max<=256:
            return np.uint8
        elif max<=65535:
            return np.uint16
        else:
            return np.uint32
    else:
        if max<128 and min>=-128:
            return np.int8
        elif max<32768 and min>=-32768:
            return np.int16
        else:
            return np.int32
        
def get_np_size(x):
    return x.size * x.itemsize


def entropy_coded_symbol_bits(symbols):
    symbols = np.asarray(symbols).reshape(-1)
    compressed, histogram_table, unique = compress_matrix_flatten_categorical(symbols)
    bits = 0
    bits += get_np_size(compressed) * 8
    bits += get_np_size(histogram_table) * 8
    bits += get_np_size(unique) * 8
    return bits

def entropy_coded_symbol_bits(symbols, name=""):
    symbols = np.asarray(symbols).reshape(-1)
    compressed, histogram_table, unique = compress_matrix_flatten_categorical(symbols)
    compressed_bits = get_np_size(compressed) * 8
    hist_bits = get_np_size(histogram_table) * 8
    unique_bits = get_np_size(unique) * 8
    total_bits = compressed_bits + hist_bits + unique_bits
    values, counts = np.unique(symbols, return_counts=True)
    prob = counts.astype(np.float64) / counts.sum()
    entropy = -np.sum(prob * np.log2(prob))
    ideal_bits = entropy * symbols.size
    print(f"[Entropy debug] {name}")
    print(f"  num symbols      : {symbols.size}")
    print(f"  num unique       : {len(unique)}")
    print(f"  empirical entropy: {entropy:.4f} bits/symbol")
    print(f"  ideal bits       : {ideal_bits:.0f}")
    print(f"  compressed bits  : {compressed_bits}")
    print(f"  histogram bits   : {hist_bits}")
    print(f"  unique bits      : {unique_bits}")
    print(f"  total ec bits    : {total_bits}")
    print(f"  bits/symbol      : {total_bits / symbols.size:.4f}")
    print()
    return total_bits
