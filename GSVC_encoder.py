# Ablation: P-frame inter positions use raw FP12 instead of 8-bit LSQ+.
from gsplat.project_gaussians_2d import project_gaussians_2d
from gsplat.rasterize_sum import rasterize_gaussians_sum
from utils import loss_fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from quantize import (
    FakeQuantizationFP12,
    FakeQuantizationFP16,
    LearnableVectorQuantizer,
    UniformQuantizer,
    UniformQuantizer_one,
    VectorQuantizer,
    fake_quantize_fp16_ste,
)
from optimizer import Adan
import math
import os

class _CudaGraphQuantStep:
    """Makes a stage-2 quantized training step capturable as a CUDA graph.

    Stage 1 (`GSVC_overfitting.manual_gsplat_train_step`) had to abandon
    autograd entirely: PyTorch's engine drives gsplat's custom-C++ rasterizer
    backward on the legacy CUDA stream, which cannot be captured.  Stage 2 puts
    quantizers in front of that rasterizer, and one of them is a vector
    quantizer using standard STE.  The fast path preserves that estimator for
    both EMA-updated and learnable codebooks.

    So the step is split rather than rewritten:

        quantizers      -> ordinary autograd  (capturable; measured identical)
        project/raster  -> native kernels, manual backward (as in stage 1)

    The rasterizer's gradients with respect to the quantizer *outputs* come out
    of `rasterize_sum_backward` / `project_gaussians_2d_backward`, and are then
    handed to `torch.autograd.backward` as the seed for one ordinary backward
    through the quantizers only.  Nothing about the quantizers' mathematics is
    reimplemented, so their gradients are by construction the ones the eager
    path produces (verified: max relative difference ~1e-7 across every
    parameter, i.e. float32 re-association noise).

    Three things had to be cleared before this would capture; each was found
    with `torch.cuda.set_sync_debug_mode("error")` and a bisection, and each is
    recorded in RESEARCH_LOG.md:

    * the VQ's two per-forward device syncs -- see
      `quantize.VectorQuantizer.prepare_for_capture`;
    * `forward_quantize` stashes `self.xys` / `self.radii` on the module, which
      keeps the *previous* iteration's autograd graph alive.  That graph is
      rooted at gsplat's custom Functions and recorded on the default stream,
      and its mere existence made an unrelated backward on any parameter of the
      model fail to capture with the legacy-stream error.  `release_graph_refs`
      drops them;
    * the dead `.item()` in the uniform quantizers (already removed).
    """

    _graph_ema = None
    _graph_undo_vq_patch = None
    _graph_ones = None
    _graph_workspace_key = None

    def release_graph_refs(self):
        """Drop the retained autograd graph from the previous forward.

        `forward` / `forward_quantize` assign `self.xys` and `self.radii`, so
        after they return the module still references the whole graph they
        built, including gsplat's custom-Function nodes bound to the stream they
        ran on.  While those references live, capturing *any* backward involving
        this model's parameters fails with "operation would make the legacy
        stream depend on a capturing blocking stream".
        """
        for attr in ("xys", "radii"):
            if getattr(self, attr, None) is not None:
                setattr(self, attr, None)

    def prepare_for_capture(self):
        """Put the model into a state where the manual step can be captured."""
        self.release_graph_refs()
        undo = []
        for vq in self._vector_quantizers():
            undo.append(vq.prepare_for_capture())
        self._graph_undo_vq_patch = undo

    def reset_intersect_capacity_for_capture(self):
        """Rebuild gsplat's fixed intersection capacity at the capture point.

        Eager training before the first CUDA Graph may already have populated
        the process-global cache using much earlier Gaussian footprints.  Clear
        it once at the delayed capture boundary and return the number of eager
        forwards needed before capture can safely use the fixed capacity.
        """
        import gsplat.utils as gsplat_utils

        warmup = max(0, int(os.environ.get("GSPLAT_ISECT_WARMUP", "8")))
        gsplat_utils._ISECT_CAP_WARMUP = warmup
        gsplat_utils._ISECT_CAP_STATE.clear()
        return warmup

    def use_dynamic_intersect_capacity(self):
        """Keep intersection allocation exact until the delayed capture."""
        import gsplat.utils as gsplat_utils

        gsplat_utils._ISECT_CAP_WARMUP = 0
        gsplat_utils._ISECT_CAP_STATE.clear()

    def end_capture(self):
        """Undo `prepare_for_capture`, leaving the library unpatched."""
        for undo in (self._graph_undo_vq_patch or []):
            undo()
        self._graph_undo_vq_patch = None
        self._graph_ema = None

    def bind_graph_ema(self, ema_mgr, decay: float):
        """Fold `utils.ModelEMA.update` into the captured step.

        `ModelEMA.update` cannot run inside a graph: it walks `state_dict()` on
        the host, re-derives a point-id alignment with `searchsorted`, and may
        rebind the shadow's tensors.  Over a fixed-shape region every point id
        matches, so the whole procedure collapses to, per key, either
        `ema = decay*ema + (1-decay)*param` (the keys `ModelEMA` selects) or a
        straight copy (everything else, e.g. the codebook's `cluster_size`).
        Binding those tensor lists once lets both run as a few `_foreach_`
        kernels inside the graph.

        The classification is taken from the live `ModelEMA` instance rather
        than re-derived, so the two cannot drift apart.  Pass `ema_mgr=None` to
        unbind, which is required before any shape may change again.
        """
        if ema_mgr is None:
            self._graph_ema = None
            return
        model_state = self.state_dict()
        shadow_state = ema_mgr.ema.state_dict()
        if model_state.keys() != shadow_state.keys():
            raise RuntimeError(
                "bind_graph_ema: model and EMA shadow have different state keys; "
                "bind only after the last eager ModelEMA.reset()/update()"
            )
        ema_dst, ema_src = [], []
        copy_dst, copy_src = {}, {}
        for key, value in model_state.items():
            shadow = shadow_state[key]
            if shadow.shape != value.shape:
                raise RuntimeError(
                    f"bind_graph_ema: shape mismatch for '{key}': model "
                    f"{tuple(value.shape)} vs EMA {tuple(shadow.shape)}"
                )
            if ema_mgr._should_ema(key) and torch.is_floating_point(value):
                ema_dst.append(shadow)
                ema_src.append(value)
            else:
                copy_dst.setdefault(value.dtype, []).append(shadow)
                copy_src.setdefault(value.dtype, []).append(value)
        if not ema_dst:
            raise RuntimeError("bind_graph_ema: nothing selected for averaging")
        self._graph_ema = (ema_dst, ema_src, copy_dst, copy_src, float(decay))

    def _apply_graph_ema(self):
        ema_dst, ema_src, copy_dst, copy_src, decay = self._graph_ema
        for dtype, dsts in copy_dst.items():
            torch._foreach_copy_(dsts, copy_src[dtype])
        torch._foreach_mul_(ema_dst, decay)
        torch._foreach_add_(ema_dst, ema_src, alpha=1.0 - decay)

    def manual_gsplat_train_step_quantize(self, gt_image, *, step=True):
        """One quantization-aware L2 step, capturable as a CUDA graph.

        Returns the same scalar the eager `train_iter_quantize` returns (the L2
        reconstruction loss plus the detached VQ loss), so an early-stopping
        rule reading the captured static tensor sees the same quantity.

        Requires a fixed image size, Gaussian count and optimizer learning rate:
        the LR is a python scalar baked into the captured optimizer kernels, so
        the driver must re-capture whenever the scheduler or the EMA-convergence
        decay changes it.
        """
        if self.loss_type != "L2":
            raise ValueError("the captured stage-2 step supports L2 only")
        if tuple(gt_image.shape) != (1, 3, self.H, self.W):
            raise ValueError(
                f"expected gt_image shape (1, 3, {self.H}, {self.W}), "
                f"got {tuple(gt_image.shape)}"
            )

        import gsplat.cuda as _C
        from gsplat.utils import bin_and_sort_gaussians, resolve_intersect_capacity

        self.optimizer.zero_grad(set_to_none=False)
        # Persistent unit gradient seed for the vq_loss branch.  Built here on
        # first use rather than per iteration: torch.ones(()) inside a capture
        # would allocate from the graph's private pool every replay.
        if getattr(self, "_graph_ones", None) is None:
            self._graph_ones = torch.ones((), device=self._xyz.device)

        # --- quantizers, under autograd -------------------------------------
        means, cholesky, colors, vq_loss = self._quantized_parts()

        n = means.shape[0]
        workspace = (n, self.H, self.W, means.device, means.dtype)
        if getattr(self, "_graph_workspace_key", None) != workspace:
            self._graph_workspace_key = workspace
            self._graph_v_depth = torch.zeros(n, device=means.device, dtype=torch.float32)
            self._graph_v_alpha = torch.zeros(self.H, self.W, device=means.device,
                                              dtype=torch.float32)

        # --- gsplat forward + loss + backward, no autograd -------------------
        means_d = means.detach().contiguous()
        cholesky_d = cholesky.detach().contiguous()
        colors_d = colors.detach().contiguous()

        xys, depths, radii, conics, num_tiles_hit = _C.project_gaussians_2d_forward(
            n, means_d, cholesky_d, self.H, self.W, self.tile_bounds, 0.01,
        )
        cumulative = torch.cumsum(num_tiles_hit, dim=0, dtype=torch.int32)
        num_intersects = resolve_intersect_capacity(cumulative, n, self.tile_bounds)
        (_, _, _, gaussian_ids_sorted, tile_bins) = bin_and_sort_gaussians(
            n, int(num_intersects), xys, depths, radii, cumulative, self.tile_bounds
        )
        raw_hwc, final_ts, final_idx = _C.rasterize_sum_forward(
            self.tile_bounds, (self.BLOCK_W, self.BLOCK_H, 1), (self.W, self.H, 1),
            gaussian_ids_sorted, tile_bins, xys, conics, colors_d, self._opacity,
            self.background,
        )

        # F.mse_loss(clamp(raw), target); the mask is torch.clamp's derivative.
        target_hwc = gt_image[0].permute(1, 2, 0)
        clipped = raw_hwc.clamp(0.0, 1.0)
        residual = clipped - target_hwc
        l2 = residual.square().mean()
        v_out = residual.mul(2.0 / residual.numel())
        v_out.mul_((raw_hwc >= 0.0) & (raw_hwc <= 1.0))

        v_xy, v_conic, v_colors, _ = _C.rasterize_sum_backward(
            self.H, self.W, self.BLOCK_H, self.BLOCK_W,
            gaussian_ids_sorted, tile_bins, xys, conics, colors_d,
            self._opacity, self.background, final_ts, final_idx,
            v_out.contiguous(), self._graph_v_alpha,
        )
        _, v_means, v_cholesky = _C.project_gaussians_2d_backward(
            n, means_d, cholesky_d, self.H, self.W, radii, conics,
            v_xy.contiguous(), self._graph_v_depth, v_conic.contiguous(),
        )

        # --- one ordinary backward through the quantizers only ---------------
        outputs = [means, cholesky, colors]
        grads = [v_means, v_cholesky, v_colors]
        if torch.is_tensor(vq_loss) and vq_loss.requires_grad:
            outputs.append(vq_loss)
            grads.append(self._graph_ones)
        torch.autograd.backward(tuple(outputs), tuple(grads))

        total = l2 + (vq_loss.detach() if torch.is_tensor(vq_loss) else vq_loss)

        if step:
            self.optimizer.step()
            if self._graph_ema is not None:
                # After the step, matching the eager loop's
                # `train_iter_quantize(); ema.update()` order.
                self._apply_graph_ema()
        return total


class GaussianVideo_frame(_CudaGraphQuantStep, nn.Module):
    def __init__(self, loss_type="L2", **kwargs):
        super().__init__()
        self.loss_type = loss_type
        self.init_num_points = kwargs["num_points"]
        self.H, self.W = kwargs["H"], kwargs["W"]
        self.BLOCK_W, self.BLOCK_H = kwargs["BLOCK_W"], kwargs["BLOCK_H"]
        self.tile_bounds = (
            (self.W + self.BLOCK_W - 1) // self.BLOCK_W,
            (self.H + self.BLOCK_H - 1) // self.BLOCK_H,
            1,
        ) # 
        self.device = kwargs["device"]
        self._xyz = nn.Parameter(torch.atanh(2 * (torch.rand(self.init_num_points, 2) - 0.5)))
        self._cholesky = nn.Parameter(torch.rand(self.init_num_points, 3))
        self._features_dc = nn.Parameter(torch.rand(self.init_num_points, 3))
        self.last_size = (self.H, self.W)
        self.register_buffer('_opacity', torch.ones((self.init_num_points, 1)))
        self.register_buffer('background', torch.ones(3))
        self.register_buffer('bound', torch.tensor([0.5, 0.5]).view(1, 2))
        self.register_buffer('cholesky_bound', torch.tensor([0.5, 0, 0.5]).view(1, 3))
        self.quantize = kwargs["quantize"]

        self.register_buffer('point_ids', torch.arange(self.init_num_points, dtype=torch.long))
        
        if self.quantize:
            cholesky_quantizer_cls = (
                UniformQuantizer_one
                if kwargs.get("is_ci", False)
                else UniformQuantizer
            )
            self.xyz_quantizer = fake_quantize_fp16_ste  # bit-exact, no custom autograd Function
            self.features_dc_quantizer = VectorQuantizer(codebook_dim=3, codebook_size=8, num_quantizers=2, vector_type="vector", kmeans_iters=5) 
            # self.cholesky_quantizer = cholesky_quantizer_cls(
            #     signed=[False, True, False], bits=6, learned=True,
            #     num_channels=3)
            self.cholesky_quantizer = UniformQuantizer_one(signed=[False, True, False], bits=6, learned=True, num_channels=3)
            
        if kwargs["opt_type"] == "adam":
            self.optimizer = torch.optim.Adam(self.parameters(), lr=kwargs["lr"])
        else:
            self.optimizer = Adan(self.parameters(), lr=kwargs["lr"])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20000, gamma=0.5)


    def _init_data(self):
        self.cholesky_quantizer._init_data(self._cholesky)

    @property
    def get_xyz(self):
        return torch.tanh(self._xyz)
    
    @property
    def get_features(self):
        return self._features_dc
    
    @property
    def get_cholesky_elements(self):
        return self._cholesky+self.cholesky_bound

    def compress_index_only(self):
        point_ids_list = self.point_ids.detach().view(-1).cpu().tolist()
        compressed_point_ids = self.IntegerArrayCompressor.compress(point_ids_list)
        pid_bit = self.IntegerArrayCompressor.bits_of_compressed(compressed_point_ids)
        return {"point_ids":compressed_point_ids, "pid_bit":pid_bit}

    def forward(self):
        self.xys, depths, self.radii, conics, num_tiles_hit = project_gaussians_2d(self.get_xyz, self.get_cholesky_elements, self.H, self.W, self.tile_bounds)
        out_img = rasterize_gaussians_sum(self.xys, depths, self.radii, conics, num_tiles_hit,
                self.get_features, self._opacity, self.H, self.W, self.BLOCK_H, self.BLOCK_W, background=self.background, return_alpha=False)
        out_img = torch.clamp(out_img, 0, 1) #[H, W, 3]
        out_img = out_img.view(-1, self.H, self.W, 3).permute(0, 3, 1, 2).contiguous()
        return {"render": out_img}
    
    def forward_quantize(self):
        l_vqm= 0
        means = torch.tanh(self.xyz_quantizer(self._xyz))
        cholesky_elements, l_vqs, _ = self.cholesky_quantizer(self._cholesky)
        cholesky_elements = cholesky_elements + self.cholesky_bound
        l_vqr = 0
        colors, l_vqc, _ = self.features_dc_quantizer(self.get_features)
        self.xys, depths, self.radii, conics, num_tiles_hit = project_gaussians_2d(means, cholesky_elements, self.H, self.W, self.tile_bounds)
        out_img = rasterize_gaussians_sum(self.xys, depths, self.radii, conics, num_tiles_hit,
                colors, self._opacity, self.H, self.W, self.BLOCK_H, self.BLOCK_W, background=self.background, return_alpha=False)
        out_img = torch.clamp(out_img, 0, 1)
        out_img = out_img.view(-1, self.H, self.W, 3).permute(0, 3, 1, 2).contiguous()
        vq_loss = l_vqm + l_vqs + l_vqr + l_vqc
        return {"render": out_img, "vq_loss": vq_loss}

    def forward_quantize_bpp(self):
        l_vqm, m_bit = 0,0
        m_bit = FakeQuantizationFP16.size(self._xyz)
        means = torch.tanh(self.xyz_quantizer(self._xyz))
        cholesky_elements, l_vqs, s_bit = self.cholesky_quantizer(self._cholesky)
        cholesky_elements = cholesky_elements + self.cholesky_bound
        l_vqr, r_bit = 0, 0
        colors, l_vqc, c_bit = self.features_dc_quantizer(self.get_features)
        self.xys, depths, self.radii, conics, num_tiles_hit = project_gaussians_2d(means, cholesky_elements, self.H, self.W, self.tile_bounds)
        out_img = rasterize_gaussians_sum(self.xys, depths, self.radii, conics, num_tiles_hit,
                colors, self._opacity, self.H, self.W, self.BLOCK_H, self.BLOCK_W, background=self.background, return_alpha=False)
        out_img = torch.clamp(out_img, 0, 1)
        out_img = out_img.view(-1, self.H, self.W, 3).permute(0, 3, 1, 2).contiguous()
        vq_loss = l_vqm + l_vqs + l_vqr + l_vqc
        return {"render": out_img, "vq_loss": vq_loss, "unit_bit":[m_bit, s_bit, r_bit, c_bit]}

    def train_iter_quantize(self, gt_image):
        render_pkg = self.forward_quantize()
        image = render_pkg["render"]
        loss = loss_fn(image.squeeze(0), gt_image.squeeze(0), self.loss_type, lambda_value=0)+ render_pkg["vq_loss"]
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        # Detached: backward() has already consumed the graph, and a live
        # reference to a loss rooted at gsplat's custom Functions (recorded
        # on the default stream) makes a later CUDA-graph capture of *any*
        # backward touching this model fail with the legacy-stream error.
        self.scheduler.step()
        return loss.detach()


    def train_iter_quantize_log(self, gt_image):
        render_pkg = self.forward_quantize()
        image = render_pkg["render"]
        loss = loss_fn(image.squeeze(0), gt_image.squeeze(0), self.loss_type, lambda_value=0)+ render_pkg["vq_loss"]
        loss.backward()
        with torch.no_grad():
            mse_loss = F.mse_loss(image, gt_image)
            psnr = 10 * math.log10(1.0 / mse_loss.item())
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        # Detached: backward() has already consumed the graph, and a live
        # reference to a loss rooted at gsplat's custom Functions (recorded
        # on the default stream) makes a later CUDA-graph capture of *any*
        # backward touching this model fail with the legacy-stream error.
        self.scheduler.step()
        return loss.detach(), psnr

    def _vector_quantizers(self):
        return (self.features_dc_quantizer,)

    def _quantized_parts(self):
        """The quantization half of `forward_quantize`, term for term.

        Kept separate so the captured step can differentiate exactly this much
        with autograd and drive the rasterizer by hand; any change here must be
        mirrored in `forward_quantize`.
        """
        means = torch.tanh(self.xyz_quantizer(self._xyz))
        cholesky_elements, l_vqs, _ = self.cholesky_quantizer(self._cholesky)
        cholesky_elements = cholesky_elements + self.cholesky_bound
        colors, l_vqc, _ = self.features_dc_quantizer(self.get_features)
        return means, cholesky_elements, colors, l_vqs + l_vqc

    def compress_wo_ec(self):
        quant_cholesky_elements, _ = self.cholesky_quantizer.compress(self._cholesky)
        _, feature_dc_index = self.features_dc_quantizer.compress(self._features_dc)
        return {"_xyz":self._xyz.half(), "_features_dc": feature_dc_index, "_cholesky": quant_cholesky_elements,}

    
    


class GaussianVideo_delta(_CudaGraphQuantStep, nn.Module):
    def __init__(self, loss_type="L2", **kwargs):
        super().__init__()
        self.loss_type = loss_type
        self.init_num_points = kwargs["num_points"]
        self.H, self.W = kwargs["H"], kwargs["W"]
        self.BLOCK_W, self.BLOCK_H = kwargs["BLOCK_W"], kwargs["BLOCK_H"]
        self.tile_bounds = (
            (self.W + self.BLOCK_W - 1) // self.BLOCK_W,
            (self.H + self.BLOCK_H - 1) // self.BLOCK_H,
            1,
        ) # 
        self.device = kwargs["device"]
        self._xyz = nn.Parameter(torch.atanh(2 * (torch.rand(self.init_num_points, 2) - 0.5)))
        self._cholesky = nn.Parameter(torch.rand(self.init_num_points, 3))
        self._features_dc = nn.Parameter(torch.rand(self.init_num_points, 3))
        self.last_size = (self.H, self.W)
        self.register_buffer('_xyz_mask',torch.ones((self.init_num_points, 2), dtype=torch.bool), persistent=True)
        self.register_buffer('p_xyz', torch.atanh(2 * (torch.rand(self.init_num_points, 2) - 0.5)))
        self.register_buffer('p_cholesky', torch.rand(self.init_num_points, 3))
        self.register_buffer('p_features_dc', torch.rand(self.init_num_points, 3))

        self.register_buffer('_opacity', torch.ones((self.init_num_points, 1)))
        self.register_buffer('background', torch.ones(3))
        self.register_buffer('bound', torch.tensor([0.5, 0.5]).view(1, 2))
        self.register_buffer('cholesky_bound', torch.tensor([0.5, 0, 0.5]).view(1, 3))

        self.register_buffer('point_ids', torch.arange(self.init_num_points, dtype=torch.long))
        
        self.quantize = kwargs["quantize"]

        self.add_num_Gaussians=kwargs["add_num_Gaussians"]
        self.is_first_p_frame=kwargs["is_first_p_frame"]

        

        if self.quantize:
            cholesky_quantizer_cls = (
                UniformQuantizer_one
                if kwargs.get("is_ci", False)
                else UniformQuantizer
            )
            pframe_vq_cls = (
                LearnableVectorQuantizer
                if os.environ.get("GSVC_PFRAME_LEARNABLE_VQ", "0") == "1"
                else VectorQuantizer
            )
            self.xyz_quantizer_I = fake_quantize_fp16_ste  # bit-exact, no custom autograd Function
            self.features_dc_quantizer_I = pframe_vq_cls(codebook_dim=3, codebook_size=8, num_quantizers=2, vector_type="vector", kmeans_iters=5)
            # self.cholesky_quantizer_I = cholesky_quantizer_cls(
            #     signed=[False, True, False], bits=6, learned=True,
            #     num_channels=3)
            self.cholesky_quantizer_I = UniformQuantizer_one(signed=[False, True, False], bits=6, learned=True, num_channels=3)
            # Fixed FP12 fake quantization: no learned LSQ scale/beta and no
            # position entropy loss or entropy coding.
            self.xyz_quantizer = FakeQuantizationFP12()
            # self.cholesky_quantizer = cholesky_quantizer_cls(
            #     signed=[True, True, True], bits=5, learned=True,
            #     num_channels=3)
            self.cholesky_quantizer = UniformQuantizer_one(signed=[True, True, True], bits=5, learned=True, num_channels=3)
            self.features_dc_quantizer = pframe_vq_cls(codebook_dim=3, codebook_size=7, num_quantizers=2, vector_type="vector", kmeans_iters=5)
            



        if kwargs["opt_type"] == "adam":
            self.optimizer = torch.optim.Adam(self.parameters(), lr=kwargs["lr"])
        else:
            self.optimizer = Adan(self.parameters(), lr=kwargs["lr"])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20000, gamma=0.5)


    def _init_data(self):
        self.cholesky_quantizer._init_data(self._cholesky)

    @property
    def get_xyz(self):
        return torch.tanh(self._xyz+self.p_xyz)
    
    @property
    def get_features(self):
        return self._features_dc+self.p_features_dc
    
    @property
    def get_cholesky_elements(self):
        return self._cholesky+self.p_cholesky+self.cholesky_bound
    
    @property
    def num_delta_Gaussians(self):
        return self.init_num_points - self.add_num_Gaussians

    def features_dc_quantizer_inputs(self):
        """The exact tensors `forward_quantize` feeds each colour quantizer.

        Returns (delta_input, abs_input_or_None).  Kept here beside
        `forward_quantize` rather than in the trainer so the deterministic
        codebook design cannot drift from the slicing it has to match: any
        change to the `add_num_Gaussians > 0` split below must be mirrored
        here, the same contract `_quantized_parts` already carries.
        """
        if self.add_num_Gaussians > 0:
            return (self._features_dc[:self.num_delta_Gaussians],
                    self._features_dc[self.num_delta_Gaussians:])
        return self._features_dc, None
 
    def forward(self):
        self.xys, depths, self.radii, conics, num_tiles_hit = project_gaussians_2d(self.get_xyz, self.get_cholesky_elements, self.H, self.W, self.tile_bounds)
        out_img = rasterize_gaussians_sum(self.xys, depths, self.radii, conics, num_tiles_hit,
                self.get_features, self._opacity, self.H, self.W, self.BLOCK_H, self.BLOCK_W, background=self.background, return_alpha=False)
        out_img = torch.clamp(out_img, 0, 1) #[H, W, 3]
        out_img = out_img.view(-1, self.H, self.W, 3).permute(0, 3, 1, 2).contiguous()
        return {"render": out_img}

    def compress_index_only(self):
        point_ids_list = self.point_ids.detach().view(-1).cpu().tolist()
        compressed_point_ids = self.IntegerArrayCompressor.compress(point_ids_list)
        pid_bit = self.IntegerArrayCompressor.bits_of_compressed(compressed_point_ids)
        return {"point_ids":compressed_point_ids, "pid_bit":pid_bit}
    
    def forward_quantize(self):
        l_vqr = 0
        l_vqm= 0
        if self.add_num_Gaussians>0:
            q_xy_front = self.xyz_quantizer(
                self._xyz[:self.num_delta_Gaussians])
            q_xy_back  = self.xyz_quantizer_I(self._xyz[self.num_delta_Gaussians:])
            q_xy = torch.cat([q_xy_front, q_xy_back], dim=0)

            cholesky_elements_front, l_vqs_front, _ = self.cholesky_quantizer(self._cholesky[:self.num_delta_Gaussians])
            cholesky_elements_back, l_vqs_back, _ = self.cholesky_quantizer_I(self._cholesky[self.num_delta_Gaussians:])
            cholesky_elements = torch.cat([cholesky_elements_front, cholesky_elements_back], dim=0)
            l_vqs = l_vqs_front + l_vqs_back
          

            colors_front, l_vqc_front, _ = self.features_dc_quantizer(self._features_dc[:self.num_delta_Gaussians])
            colors_back, l_vqc_back, _ = self.features_dc_quantizer_I(self._features_dc[self.num_delta_Gaussians:])
            colors = torch.cat([colors_front, colors_back], dim=0)
            l_vqc = l_vqc_front + l_vqc_back
       
        else:
            q_xy = self.xyz_quantizer(self._xyz)
            cholesky_elements, l_vqs, _ = self.cholesky_quantizer(self._cholesky)
            colors, l_vqc, _ = self.features_dc_quantizer(self._features_dc)

        means = torch.tanh(q_xy+ self.p_xyz)
        cholesky_elements = cholesky_elements + self.cholesky_bound+self.p_cholesky
        colors = colors+self.p_features_dc

        self.xys, depths, self.radii, conics, num_tiles_hit = project_gaussians_2d(means, cholesky_elements, self.H, self.W, self.tile_bounds)
        out_img = rasterize_gaussians_sum(self.xys, depths, self.radii, conics, num_tiles_hit,
                colors, self._opacity, self.H, self.W, self.BLOCK_H, self.BLOCK_W, background=self.background, return_alpha=False)
        out_img = torch.clamp(out_img, 0, 1)
        out_img = out_img.view(-1, self.H, self.W, 3).permute(0, 3, 1, 2).contiguous()
        vq_loss = l_vqm + l_vqs + l_vqr + l_vqc


        return {"render": out_img, "vq_loss": vq_loss}

    def forward_quantize_bpp(self):
        l_vqm, l_vqr, r_bit = 0, 0, 0
        if self.add_num_Gaussians>0:
            inter_xyz = self._xyz[:self.num_delta_Gaussians]
            q_xy_front = self.xyz_quantizer(inter_xyz)
            m_bit_front = self.xyz_quantizer.size(inter_xyz)
            q_xy_back = self.xyz_quantizer_I(self._xyz[self.num_delta_Gaussians:])
            m_bit_back = FakeQuantizationFP16.size(
                self._xyz[self.num_delta_Gaussians:]
            )
            q_xy = torch.cat([q_xy_front, q_xy_back], dim=0)
            m_bit= m_bit_front + m_bit_back


            cholesky_elements_front, l_vqs_front, s_bit_front = self.cholesky_quantizer(self._cholesky[:self.num_delta_Gaussians])
            cholesky_elements_back, l_vqs_back, s_bit_back = self.cholesky_quantizer_I(self._cholesky[self.num_delta_Gaussians:])
            cholesky_elements = torch.cat([cholesky_elements_front, cholesky_elements_back], dim=0)
            l_vqs = l_vqs_front + l_vqs_back
            s_bit = s_bit_front + s_bit_back

            colors_front, l_vqc_front, c_bit_front = self.features_dc_quantizer(self._features_dc[:self.num_delta_Gaussians])
            colors_back, l_vqc_back, c_bit_back = self.features_dc_quantizer_I(self._features_dc[self.num_delta_Gaussians:])
            colors = torch.cat([colors_front, colors_back], dim=0)
            l_vqc = l_vqc_front + l_vqc_back
            c_bit = c_bit_front + c_bit_back
        else:
            q_xy = self.xyz_quantizer(self._xyz)
            m_bit = self.xyz_quantizer.size(self._xyz)
            cholesky_elements, l_vqs, s_bit = self.cholesky_quantizer(self._cholesky)
            colors, l_vqc, c_bit = self.features_dc_quantizer(self._features_dc)

        means = torch.tanh(q_xy+ self.p_xyz)
        cholesky_elements = cholesky_elements + self.cholesky_bound+self.p_cholesky
        colors = colors+self.p_features_dc
        self.xys, depths, self.radii, conics, num_tiles_hit = project_gaussians_2d(means, cholesky_elements, self.H, self.W, self.tile_bounds)
        out_img = rasterize_gaussians_sum(self.xys, depths, self.radii, conics, num_tiles_hit,
                colors, self._opacity, self.H, self.W, self.BLOCK_H, self.BLOCK_W, background=self.background, return_alpha=False)
        out_img = torch.clamp(out_img, 0, 1)
        out_img = out_img.view(-1, self.H, self.W, 3).permute(0, 3, 1, 2).contiguous()
        vq_loss = l_vqm + l_vqs + l_vqr + l_vqc

        return {"render": out_img, "vq_loss": vq_loss, "unit_bit":[m_bit, s_bit, r_bit, c_bit]}

    def train_iter_quantize(self, gt_image):
        render_pkg = self.forward_quantize()
        image = render_pkg["render"]
        loss = loss_fn(image.squeeze(0), gt_image.squeeze(0), self.loss_type, lambda_value=0)+ render_pkg["vq_loss"]
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        # Detached: backward() has already consumed the graph, and a live
        # reference to a loss rooted at gsplat's custom Functions (recorded
        # on the default stream) makes a later CUDA-graph capture of *any*
        # backward touching this model fail with the legacy-stream error.
        self.scheduler.step()
        return loss.detach()
    
    def train_iter_quantize_log(self, gt_image):
        render_pkg = self.forward_quantize()
        image = render_pkg["render"]
        loss = loss_fn(image.squeeze(0), gt_image.squeeze(0), self.loss_type, lambda_value=0)+ render_pkg["vq_loss"]
        loss.backward()
        with torch.no_grad():
            mse_loss = F.mse_loss(image, gt_image)
            psnr = 10 * math.log10(1.0 / mse_loss.item())
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        # Detached: backward() has already consumed the graph, and a live
        # reference to a loss rooted at gsplat's custom Functions (recorded
        # on the default stream) makes a later CUDA-graph capture of *any*
        # backward touching this model fail with the legacy-stream error.
        self.scheduler.step()
        return loss.detach(), psnr

    def _vector_quantizers(self):
        if self.add_num_Gaussians > 0:
            return (self.features_dc_quantizer, self.features_dc_quantizer_I)
        return (self.features_dc_quantizer,)

    def _quantized_parts(self):
        """The quantization half of `forward_quantize`, term for term.

        Kept separate so the captured step can differentiate exactly this much
        with autograd and drive the rasterizer by hand; any change here must be
        mirrored in `forward_quantize`.
        """
        l_vqr = 0
        if self.add_num_Gaussians > 0:
            l_vqm = 0
            q_xy_front = self.xyz_quantizer(
                self._xyz[:self.num_delta_Gaussians])
            q_xy_back = self.xyz_quantizer_I(self._xyz[self.num_delta_Gaussians:])
            q_xy = torch.cat([q_xy_front, q_xy_back], dim=0)

            cholesky_front, l_vqs_front, _ = self.cholesky_quantizer(self._cholesky[:self.num_delta_Gaussians])
            cholesky_back, l_vqs_back, _ = self.cholesky_quantizer_I(self._cholesky[self.num_delta_Gaussians:])
            cholesky_elements = torch.cat([cholesky_front, cholesky_back], dim=0)
            l_vqs = l_vqs_front + l_vqs_back

            colors_front, l_vqc_front, _ = self.features_dc_quantizer(self._features_dc[:self.num_delta_Gaussians])
            colors_back, l_vqc_back, _ = self.features_dc_quantizer_I(self._features_dc[self.num_delta_Gaussians:])
            colors = torch.cat([colors_front, colors_back], dim=0)
            l_vqc = l_vqc_front + l_vqc_back
        else:
            l_vqm = 0
            q_xy = self.xyz_quantizer(self._xyz)
            cholesky_elements, l_vqs, _ = self.cholesky_quantizer(self._cholesky)
            colors, l_vqc, _ = self.features_dc_quantizer(self._features_dc)

        means = torch.tanh(q_xy + self.p_xyz)
        cholesky_elements = cholesky_elements + self.cholesky_bound + self.p_cholesky
        colors = colors + self.p_features_dc
        return means, cholesky_elements, colors, l_vqm + l_vqs + l_vqr + l_vqc

    def compress_wo_ec(self):
        if self.add_num_Gaussians>0:
            # Packed raw FP12 symbols; bit packing is not entropy coding.
            quant_xyz_front = self.xyz_quantizer.encode(
                self._xyz[:self.num_delta_Gaussians])
            quant_xyz_back  = self.xyz_quantizer_I(self._xyz[self.num_delta_Gaussians:]).half()
           
            quant_cholesky_elements_front, _ = self.cholesky_quantizer.compress(self._cholesky[:self.num_delta_Gaussians])
            quant_cholesky_elements_back, _ = self.cholesky_quantizer_I.compress(self._cholesky[self.num_delta_Gaussians:])
            quant_cholesky_elements = torch.cat([quant_cholesky_elements_front, quant_cholesky_elements_back], dim=0)

            _, feature_dc_index_front = self.features_dc_quantizer.compress(self._features_dc[:self.num_delta_Gaussians])
            _, feature_dc_index_back = self.features_dc_quantizer_I.compress(self._features_dc[self.num_delta_Gaussians:])
            feature_dc_index = torch.cat([feature_dc_index_front, feature_dc_index_back], dim=0)

            return {"_xyz_front":quant_xyz_front,"_xyz_back":quant_xyz_back, "_features_dc": feature_dc_index, "_cholesky": quant_cholesky_elements,}
        else:
            quant_xyz = self.xyz_quantizer.encode(self._xyz)
            quant_cholesky_elements, _ = self.cholesky_quantizer.compress(self._cholesky)
            _, feature_dc_index = self.features_dc_quantizer.compress(self._features_dc)
            return {"_xyz":quant_xyz, "_features_dc": feature_dc_index, "_cholesky": quant_cholesky_elements}




