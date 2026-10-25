from gsplat.project_gaussians_2d import project_gaussians_2d
from gsplat.rasterize_sum import rasterize_gaussians_sum
from utils import IterStageWallAverager, loss_fn, sobol_block, sobol_cholesky, sobol_raw_xyz
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from optimizer import Adan
from typing import Optional

class GaussianVideo_frame(nn.Module):
    def __init__(self, loss_type="L2", **kwargs):
        super().__init__()
        self.loss_type = loss_type
        self.init_num_points = kwargs["num_points"]
        self.max_num_points = kwargs["max_num_points"]
        self.alpha = float(kwargs.get("alpha", 1.0))
        self.densification_interval=kwargs["densification_interval"]
        self.iterations=kwargs["iterations"]
        self.H, self.W = kwargs["H"], kwargs["W"]
        self.BLOCK_W, self.BLOCK_H = kwargs["BLOCK_W"], kwargs["BLOCK_H"]
        self.tile_bounds = (
            (self.W + self.BLOCK_W - 1) // self.BLOCK_W,
            (self.H + self.BLOCK_H - 1) // self.BLOCK_H,
            1,
        ) # 
        self.device = kwargs["device"]
        self.removal_rate=kwargs["removal_rate"]
        self.iscover=kwargs["iscover"]
        if self.iscover:
            # Keep the deterministic coverage initialization unchanged.
            _u = sobol_block(0, self.init_num_points)
            self._xyz = nn.Parameter(sobol_raw_xyz(_u))
            self._cholesky = nn.Parameter(sobol_cholesky(_u, True))
            self._features_dc = nn.Parameter(torch.zeros(self.init_num_points, 3))
        else:
            # Ablation without coverage: position, shape, and color are all
            # independently initialized from the original random distributions.
            # self._xyz = nn.Parameter(torch.atanh(
            #     2 * (torch.rand(self.init_num_points, 2) - 0.5)))
            # self._cholesky = nn.Parameter(torch.rand(self.init_num_points, 3))
            # self._features_dc = nn.Parameter(torch.rand(self.init_num_points, 3))
            _u = sobol_block(0, self.init_num_points)
            self._xyz = nn.Parameter(sobol_raw_xyz(_u))
            self._cholesky = nn.Parameter(sobol_cholesky(_u, True))
            self._features_dc = nn.Parameter(torch.zeros(self.init_num_points, 3))
        self.isdensity=kwargs["isdensity"]
        self.isremoval=kwargs["isremoval"]
        self.register_buffer('point_ids', torch.arange(self.init_num_points, dtype=torch.long))
        self.register_buffer('next_point_id', torch.tensor(self.init_num_points, dtype=torch.long))
        self.enhanced=kwargs["enhanced"]
        
        if self.isremoval and not self.enhanced:
            self.rgb_W = nn.Parameter(0.01 * torch.ones(self.init_num_points, 1))
        elif self.isdensity or self.enhanced:
            self.rgb_W = nn.Parameter(torch.ones(self.init_num_points, 1))
        else:
           self.register_buffer('rgb_W', torch.ones((self.init_num_points, 1))) 
        
        self.last_size = (self.H, self.W)
        self.quantize = kwargs["quantize"]
        self.register_buffer('background', torch.ones(3))
        self.rgbW_activation = torch.sigmoid
        self.rgb_activation = torch.sigmoid
        self.register_buffer('bound', torch.tensor([0.5, 0.5]).view(1, 2))
        self.register_buffer('cholesky_bound', torch.tensor([0.5, 0, 0.5]).view(1, 3))
        self.lr = kwargs["lr"]
        self.lr_decay_start_iter = int(kwargs.get("lr_decay_start_iter", 1))
        self.opt_type =kwargs["opt_type"]
        if kwargs["opt_type"] == "adam":
            self.optimizer = torch.optim.Adam(self.parameters(), lr=kwargs["lr"])
        else:
            self.optimizer = Adan(self.parameters(), lr=kwargs["lr"])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20000, gamma=0.5)
        self.timer = IterStageWallAverager()
        self._vis_added_raw_xyz = None
        self._vis_removed_raw_xyz = None
        self._vis_added_iter = None
        self._vis_removed_iter = None
        self.flow_active = False
        # (ema_tensors, model_tensors, decay) when an EMA shadow is folded into
        # manual_gsplat_train_step; see bind_graph_ema().
        self._graph_ema = None

    def _get_opacity_cache(self):
        """Return a valid opacity tensor after Gaussian count changes."""
        cache = getattr(self, "_opacity_cache", None)
        if (cache is None or cache.shape[0] != self._xyz.shape[0]
                or cache.device != self._xyz.device
                or cache.dtype != self._xyz.dtype):
            cache = self._xyz.new_ones((self._xyz.shape[0], 1))
            self._opacity_cache = cache
        return cache

    def set_lr(self, lr: float):
        lr = float(lr)
        self.lr = lr
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        if hasattr(self, "scheduler") and (self.scheduler is not None) and hasattr(self.scheduler, "base_lrs"):
            self.scheduler.base_lrs = [lr for _ in self.scheduler.base_lrs]

    def _cholesky_init_scale(self, num_points: int) -> float:
        num_points = max(int(num_points), 1)
        alpha = float(self.alpha)
        return float(self.H * self.H) / (alpha * float(num_points))

    def apply_scale_to_initial_cholesky(self: int):
        if self.iscover:
            ratio = float(self._cholesky_init_scale(int(self._xyz.shape[0])))
            with torch.no_grad():
                self._cholesky.data.mul_(ratio)
        else:
            return

    
    def grow_points_to(self, target_num_points: int) -> int:
        target_num_points = int(target_num_points)
        if target_num_points <= 0:
            return self._xyz.shape[0]
        target_num_points = min(target_num_points, int(self.max_num_points))
        
        current = int(self._xyz.shape[0])
        if target_num_points <= current:
            return current

        add_n = target_num_points - current
        device = self._xyz.device
        with torch.no_grad():
            if self.iscover:
                # Keep progressive coverage growth deterministic.
                u = sobol_block(
                    int(self.next_point_id.item()), add_n, device=device)
                new_xyz = sobol_raw_xyz(u)
                new_cholesky = sobol_cholesky(u, True)
                new_cholesky = self._cholesky_init_scale(target_num_points) * new_cholesky
                new_features_dc = 0.01 * torch.ones(add_n, 3, device=device)
            else:
                u = sobol_block(
                    int(self.next_point_id.item()), add_n, device=device)
                new_xyz = sobol_raw_xyz(u)
                new_cholesky = sobol_cholesky(u, True)
                new_features_dc = 0.01 * torch.ones(add_n, 3, device=device)

            rgb_w_is_parameter = isinstance(
                getattr(self, "rgb_W", None), torch.nn.Parameter)
            if rgb_w_is_parameter:
                if self.isremoval and not self.enhanced:
                    new_rgb_W = 0.01 * torch.ones(add_n, 1, device=device)
                else:
                    new_rgb_W = torch.ones(add_n, 1, device=device)
            else:
                new_rgb_W = torch.ones(add_n, 1, device=device)

            self._xyz = nn.Parameter(
                torch.cat((self._xyz, new_xyz), dim=0))
            self._cholesky = nn.Parameter(
                torch.cat((self._cholesky, new_cholesky), dim=0))
            self._features_dc = nn.Parameter(
                torch.cat((self._features_dc, new_features_dc), dim=0))
            if rgb_w_is_parameter:
                self.rgb_W = nn.Parameter(
                    torch.cat((self.rgb_W, new_rgb_W), dim=0))
            else:
                self.rgb_W = torch.cat((self.rgb_W, new_rgb_W), dim=0)

            new_ids = torch.arange(
                self.next_point_id.item(),
                self.next_point_id.item() + add_n,
                dtype=torch.long,
                device=self.point_ids.device,
            )
            self.point_ids = torch.cat((self.point_ids, new_ids), dim=0)
            self.next_point_id = torch.tensor(
                self.next_point_id.item() + add_n,
                dtype=torch.long,
                device=self.point_ids.device,
            )

        for param_group in self.optimizer.param_groups:
            param_group["params"] = [
                p for p in self.parameters() if p.requires_grad
            ]
        active_params = {
            param
            for group in self.optimizer.param_groups
            for param in group["params"]
        }
        for param in list(self.optimizer.state.keys()):
            if param not in active_params:
                del self.optimizer.state[param]

        self._opacity_cache = None
        self._manual_gsplat_workspace_key = None
        self._graph_ema = None
        return int(self._xyz.shape[0])


    def _init_data(self):
        self.cholesky_quantizer._init_data(self._cholesky)
    
    def _build_tile_bounds(self, H: int, W: int):
        return (
            (W + self.BLOCK_W - 1) // self.BLOCK_W,
            (H + self.BLOCK_H - 1) // self.BLOCK_H,
            1,
        )


    @property
    def get_xyz(self):
        return torch.tanh(self._xyz)
    
    @property
    def get_features(self):
        return self._features_dc*self.get_rgb_W
    
    @property
    def get_rgb_W(self):
        return self.rgb_W

    @property
    def get_cholesky_elements(self):
        return self._cholesky+self.cholesky_bound

    
    def forward(self):
        # _opacity = torch.ones(self._xyz.shape[0], 1).to(self.device)
        self._get_opacity_cache()
        self.xys, depths, self.radii, conics, num_tiles_hit = project_gaussians_2d(self.get_xyz, self.get_cholesky_elements, self.H, self.W, self.tile_bounds)
        out_img = rasterize_gaussians_sum(self.xys, depths, self.radii, conics, num_tiles_hit,
                self.get_features, self._opacity_cache, self.H, self.W, self.BLOCK_H, self.BLOCK_W, background=self.background, return_alpha=False)
        out_img = torch.clamp(out_img, 0, 1) #[H, W, 3]
        out_img = out_img.view(-1, self.H, self.W, 3).permute(0, 3, 1, 2).contiguous()
        return {"render": out_img}

    # Gaussian tensors that utils.ModelEMA averages point-wise; in the graphed
    # tail the point ids are frozen, so its id-aligned gather/scatter collapses
    # to a plain elementwise average over exactly these.
    _GRAPH_EMA_KEYS = ("_xyz", "_cholesky", "_features_dc", "rgb_W")

    def bind_graph_ema(self, ema_model, decay: float):
        """Fold an EMA shadow model's update into ``manual_gsplat_train_step``.

        ``utils.ModelEMA.update`` cannot run inside a CUDA graph: it walks
        ``state_dict()`` on the host, re-derives the point-id alignment, and
        rebinds the shadow tensors (``_set_tensor_by_key``) on every call, so
        both the control flow and the tensor addresses change from step to step.
        While the Gaussian count is fixed -- which is exactly the region the
        graph covers -- the point ids are constant, every id matches, and that
        whole procedure reduces to ``ema = decay*ema + (1-decay)*param`` over
        the four Gaussian tensors.  Binding those tensor pairs once lets the
        update be recorded as two foreach kernels inside the captured step, so a
        replay advances the model and its EMA together for no extra CPU launches.

        Pass ``ema_model=None`` to unbind (required before the shape can change
        again, since the bound addresses would go stale).
        """
        if ema_model is None:
            self._graph_ema = None
            return
        src, dst = [], []
        mine = dict(self.named_parameters())
        mine.update(dict(self.named_buffers()))
        theirs = dict(ema_model.named_parameters())
        theirs.update(dict(ema_model.named_buffers()))
        for k in self._GRAPH_EMA_KEYS:
            s, d = mine.get(k), theirs.get(k)
            if s is None or d is None:
                raise RuntimeError(f"bind_graph_ema: '{k}' missing from model or EMA shadow")
            if s.shape != d.shape:
                raise RuntimeError(
                    f"bind_graph_ema: shape mismatch for '{k}': model {tuple(s.shape)} "
                    f"vs EMA {tuple(d.shape)} -- bind only after the last eager "
                    "densify/prune and the last ModelEMA.update()"
                )
            if not torch.is_floating_point(s):
                continue
            src.append(s.detach())
            dst.append(d.detach())
        self._graph_ema = (dst, src, float(decay))

    def manual_gsplat_train_step(self, gt_image, *, step=True):
        """One L2 training step through gsplat's native CUDA kernels WITHOUT the
        PyTorch autograd engine.

        The forward and backward extension kernels are invoked directly and their
        results propagated by hand into persistent parameter ``.grad`` buffers,
        then the ordinary optimizer step is applied.  This exists solely so the
        whole step (forward + loss + backward + optimizer) can be captured as a
        CUDA graph: the autograd engine drives a custom-Function backward on the
        legacy CUDA stream, which is illegal to capture, whereas these kernel
        launches all run on the current stream.  Numerically it is identical to
        the autograd path (validated bit-for-bit on xyz/cholesky gradients).

        Requires a fixed image size and Gaussian count.  Run a few eager warmup
        steps before capture so the intersect-capacity cache, optimizer state,
        work buffers, and ``.grad`` storage all exist and are frozen.
        """
        if self.loss_type != "L2":
            raise ValueError("manual gsplat differentiation currently supports L2 only")
        if gt_image.shape != (1, 3, self.H, self.W):
            raise ValueError(
                f"expected gt_image shape (1, 3, {self.H}, {self.W}), got {tuple(gt_image.shape)}"
            )

        import gsplat.cuda as _C
        from gsplat.utils import bin_and_sort_gaussians, resolve_intersect_capacity

        self.optimizer.zero_grad(set_to_none=False)
        n = self._xyz.shape[0]
        self._get_opacity_cache()

        # Persistent (non-autograd) work buffers, reallocated only on shape change.
        workspace_key = (n, self.H, self.W, self._xyz.device, self._xyz.dtype)
        if getattr(self, "_manual_gsplat_workspace_key", None) != workspace_key:
            self._manual_gsplat_workspace_key = workspace_key
            self._manual_gsplat_v_depth = torch.zeros(n, device=self._xyz.device, dtype=torch.float32)
            self._manual_gsplat_v_alpha = torch.zeros(self.H, self.W, device=self._xyz.device, dtype=torch.float32)

        # Parameter transforms (same as forward()): xyz=tanh(_xyz),
        # cholesky=_cholesky+bound, features=_features_dc*rgb_W.
        xyz = self.get_xyz
        cholesky = self.get_cholesky_elements
        features = self.get_features

        xys, depths, radii, conics, num_tiles_hit = _C.project_gaussians_2d_forward(
            n, xyz.contiguous(), cholesky.contiguous(), self.H, self.W, self.tile_bounds, 0.01,
        )
        cumulative = torch.cumsum(num_tiles_hit, dim=0, dtype=torch.int32)
        # Capture-safe intersect capacity (no per-forward device->host sync).
        num_intersects = resolve_intersect_capacity(cumulative, n, self.tile_bounds)
        (_, _, _, gaussian_ids_sorted, tile_bins) = bin_and_sort_gaussians(
            n, int(num_intersects), xys, depths, radii, cumulative, self.tile_bounds
        )
        raw_hwc, final_ts, final_idx = _C.rasterize_sum_forward(
            self.tile_bounds, (self.BLOCK_W, self.BLOCK_H, 1), (self.W, self.H, 1),
            gaussian_ids_sorted, tile_bins, xys, conics, features, self._opacity_cache, self.background,
        )

        # F.mse_loss(clamp(raw).permute, target); mask is torch.clamp's derivative.
        target_hwc = gt_image[0].permute(1, 2, 0)
        clipped = raw_hwc.clamp(0.0, 1.0)
        residual = clipped - target_hwc
        loss = residual.square().mean()
        v_out = residual.mul(2.0 / residual.numel())
        v_out.mul_((raw_hwc >= 0.0) & (raw_hwc <= 1.0))

        v_xy, v_conic, v_colors, _ = _C.rasterize_sum_backward(
            self.H, self.W, self.BLOCK_H, self.BLOCK_W,
            gaussian_ids_sorted, tile_bins, xys, conics, features,
            self._opacity_cache, self.background, final_ts, final_idx,
            v_out.contiguous(), self._manual_gsplat_v_alpha,
        )
        _, v_mean2d, v_cholesky = _C.project_gaussians_2d_backward(
            n, xyz.contiguous(), cholesky.contiguous(), self.H, self.W,
            radii, conics, v_xy.contiguous(), self._manual_gsplat_v_depth, v_conic.contiguous(),
        )

        gxyz = v_mean2d * (1.0 - xyz.square())               # d tanh
        gfeatures = v_colors * self.get_rgb_W

        def copy_grad(param, value):
            if param.requires_grad:
                if param.grad is None:
                    param.grad = torch.zeros_like(param)
                param.grad.copy_(value)

        copy_grad(self._xyz, gxyz)
        copy_grad(self._cholesky, v_cholesky)
        copy_grad(self._features_dc, gfeatures)
        if isinstance(self.rgb_W, nn.Parameter):
            copy_grad(self.rgb_W, (v_colors * self._features_dc).sum(dim=1, keepdim=True))

        if step:
            self.optimizer.step()
            if self._graph_ema is not None:
                # After the step, matching eager's `train_iter(); update_ema()`
                # order.  mul_ + add_(alpha=...) reproduces ModelEMA's
                # `old.mul(decay).add(cur, alpha=1-decay)` term for term.
                ema_t, model_t, decay = self._graph_ema
                torch._foreach_mul_(ema_t, decay)
                torch._foreach_add_(ema_t, model_t, alpha=1.0 - decay)
        return loss

    def adaptive_control_done_iter(self):
        """Last iteration at which adaptive densify/prune changes the Gaussian
        count; after this the shape is fixed and a CUDA graph can be captured.
        Tracks the thresholds in adaptive_control (final prune at iter 1000) and
        removal_control (acts through iter 4000)."""
        done = 0
        if getattr(self, "isdensity", False):
            done = max(done, 1000)
        if getattr(self, "isremoval", False):
            done = max(done, 4000)
        if getattr(self, "flow_active", False):
            done = max(done, int(getattr(self, "flow_max_iterations", 0)))
        return done

    def pop_visualization_events(self):
        if self._vis_added_raw_xyz is None and self._vis_removed_raw_xyz is None:
            return None
        event = {
            "added_raw_xyz": self._vis_added_raw_xyz,
            "removed_raw_xyz": self._vis_removed_raw_xyz,
            "added_iter": self._vis_added_iter,
            "removed_iter": self._vis_removed_iter,
        }
        self._vis_added_raw_xyz = None
        self._vis_removed_raw_xyz = None
        self._vis_added_iter = None
        self._vis_removed_iter = None
        return event

    def set_flow_prior(self, previous_xy, target_flow, weight, max_iterations):
        dtype, device = self._xyz.dtype, self._xyz.device
        self.flow_previous_xy = previous_xy.detach().to(device=device, dtype=dtype)
        self.flow_target = target_flow.detach().to(device=device, dtype=dtype)
        self.flow_num_points = int(previous_xy.shape[0])
        self.flow_weight = float(weight)
        self.flow_max_iterations = max(0, min(int(max_iterations), 499))
        self.flow_ema = None
        self.flow_best = None
        self.flow_bad_checks = 0
        self.flow_active = self.flow_num_points > 0 and self.flow_max_iterations > 0

    def _flow_prior_loss(self):
        if not self.flow_active:
            return self._xyz.new_zeros(())
        n = min(self.flow_num_points, int(self._xyz.shape[0]))
        normalized_xy = self.get_xyz[:n]
        current_xy = torch.stack(
            ((normalized_xy[:, 0] + 1.0) * (self.W * 0.5),
             (normalized_xy[:, 1] + 1.0) * (self.H * 0.5)), dim=1)
        actual_flow = current_xy - self.flow_previous_xy[:n]
        return (actual_flow - self.flow_target[:n]).square().sum(dim=1).mean()

    @torch.no_grad()
    def _update_flow_prior(self, flow_loss, iteration):
        if not self.flow_active or flow_loss is None:
            return
        value = flow_loss.detach()
        self.flow_ema = value if self.flow_ema is None else 0.9 * self.flow_ema + 0.1 * value
        if iteration >= self.flow_max_iterations:
            self.flow_active = False
            return
        if iteration < 40 or iteration % 20 != 0:
            return
        current = float(self.flow_ema.item())
        if self.flow_best is None:
            self.flow_best = current
        elif (self.flow_best - current) / max(self.flow_best, 1e-8) > 0.01:
            self.flow_best = current
            self.flow_bad_checks = 0
        else:
            self.flow_bad_checks += 1
            if self.flow_bad_checks >= 3:
                self.flow_active = False

    def _training_loss(self, image, gt_image):
        reconstruction_loss = loss_fn(image, gt_image, self.loss_type, lambda_value=0)
        if not self.flow_active:
            return reconstruction_loss, None
        flow_loss = self._flow_prior_loss()
        return reconstruction_loss + self.flow_weight * flow_loss, flow_loss

    def prepare_scale_cache(self, scales):
        cache = {}
        H0, W0 = int(self.H), int(self.W)
        for s in set(int(x) for x in scales):
            if s < 1:
                raise ValueError(f"scale must be >= 1, got {s}")
            Hs = max(1, H0 // s)
            Ws = max(1, W0 // s)
            tile_bounds_s = self._build_tile_bounds(Hs, Ws)
            inv_scale = 1.0 / float(s)  # Python float
            cache[s] = (Hs, Ws, tile_bounds_s, inv_scale)

        self._scale_cache = cache
    

    def forward_scale(self, scale):
        scale = int(scale)
        if scale < 1:
            raise ValueError(f"scale must be >= 1, got {scale}")
        if hasattr(self, "_scale_cache") and (scale in self._scale_cache):
            Hs, Ws, tile_bounds_s, inv_scale = self._scale_cache[scale]
        else:
            # fallback (still no GPU transfer)
            H0, W0 = int(self.H), int(self.W)
            Hs = max(1, H0 // scale)
            Ws = max(1, W0 // scale)
            tile_bounds_s = self._build_tile_bounds(Hs, Ws)
            inv_scale = 1.0 / float(scale)

        self._get_opacity_cache()
        tile_bounds_s = self._build_tile_bounds(Hs, Ws)
        xys, depths, radii, conics, num_tiles_hit = project_gaussians_2d(
            self.get_xyz,
            self.get_cholesky_elements*inv_scale,
            Hs, Ws,
            tile_bounds_s
        )
        out_img = rasterize_gaussians_sum(
            xys, depths, radii, conics, num_tiles_hit,
            self.get_features, self._opacity_cache,
            Hs, Ws,
            self.BLOCK_H, self.BLOCK_W,
            background=self.background,
            return_alpha=False
        )
        out_img = torch.clamp(out_img, 0, 1)
        out_img = out_img.view(-1, Hs, Ws, 3).permute(0, 3, 1, 2).contiguous()
        return {"render": out_img}

    def forward_contribution(self):
        self._get_opacity_cache()

        self.xys, depths, self.radii, conics, num_tiles_hit = project_gaussians_2d(
            self.get_xyz, self.get_cholesky_elements, self.H, self.W, self.tile_bounds
        )

        # w: [N]
        w = torch.norm(self.rgb_W, dim=1).to(self._xyz.device).flatten()

        # ------------------------------------------------------
        # Rank-based normalization:
        # sort w, then assign uniformly spaced values in [0, 1]
        # ------------------------------------------------------
        N = w.shape[0]
        if N <= 1:
            t = torch.zeros_like(w)
        else:
            sort_idx = torch.argsort(w)                 # ascending
            t = torch.empty_like(w, dtype=torch.float32)
            t_uniform = torch.linspace(
                0.0, 1.0, steps=N, device=w.device, dtype=torch.float32
            )
            t[sort_idx] = t_uniform

        # ------------------------------------------------------
        # Color mapping: blue -> cyan -> green -> yellow -> red
        # since w is non-negative, this is more suitable than blue-white-red
        # ------------------------------------------------------
        t2 = 2.0 * t
        r = torch.clamp(2.0 * t - 0.5, 0.0, 1.0)
        g = torch.clamp(1.0 - torch.abs(2.0 * t - 1.0), 0.0, 1.0)
        b = torch.clamp(1.5 - 2.0 * t, 0.0, 1.0)

        features_dc_w = torch.stack((r, g, b), dim=1)

        out_img = rasterize_gaussians_sum(
            self.xys,
            depths,
            self.radii,
            conics,
            num_tiles_hit,
            features_dc_w,
            self._opacity_cache,
            self.H,
            self.W,
            self.BLOCK_H,
            self.BLOCK_W,
            background=self.background,
            return_alpha=False,
        )

        out_img = torch.clamp(out_img, 0, 1)  # [H, W, 3]
        out_img = out_img.view(-1, self.H, self.W, 3).permute(0, 3, 1, 2).contiguous()
        return {"render": out_img}

    def forward_scale_contribution(self, scale):
        scale = int(scale)
        if scale < 1:
            raise ValueError(f"scale must be >= 1, got {scale}")
        if hasattr(self, "_scale_cache") and (scale in self._scale_cache):
            Hs, Ws, tile_bounds_s, inv_scale = self._scale_cache[scale]
        else:
            # fallback (still no GPU transfer)
            H0, W0 = int(self.H), int(self.W)
            Hs = max(1, H0 // scale)
            Ws = max(1, W0 // scale)
            tile_bounds_s = self._build_tile_bounds(Hs, Ws)
            inv_scale = 1.0 / float(scale)

        self._get_opacity_cache()
        tile_bounds_s = self._build_tile_bounds(Hs, Ws)
        xys, depths, radii, conics, num_tiles_hit = project_gaussians_2d(
            self.get_xyz,
            self.get_cholesky_elements*inv_scale,
            Hs, Ws,
            tile_bounds_s
        )


        # w: [N]
        w = torch.norm(self.rgb_W, dim=1).to(self._xyz.device).flatten()
        


        # ------------------------------------------------------
        # Rank-based normalization:
        # sort w, then assign uniformly spaced values in [0, 1]
        # ------------------------------------------------------
        N = w.shape[0]
        if N <= 1:
            t = torch.zeros_like(w)
        else:
            sort_idx = torch.argsort(w)                 # ascending
            t = torch.empty_like(w, dtype=torch.float32)
            t_uniform = torch.linspace(
                0.0, 1.0, steps=N, device=w.device, dtype=torch.float32
            )
            t[sort_idx] = t_uniform

        # ------------------------------------------------------
        # Color mapping: blue -> cyan -> green -> yellow -> red
        # since w is non-negative, this is more suitable than blue-white-red
        # ------------------------------------------------------
        t2 = 2.0 * t
        r = torch.clamp(2.0 * t - 0.5, 0.0, 1.0)
        g = torch.clamp(1.0 - torch.abs(2.0 * t - 1.0), 0.0, 1.0)
        b = torch.clamp(1.5 - 2.0 * t, 0.0, 1.0)

        features_dc_w = torch.stack((r, g, b), dim=1)

        out_img = rasterize_gaussians_sum(
            xys, depths, radii, conics, num_tiles_hit,
            features_dc_w, self._opacity_cache,
            Hs, Ws,
            self.BLOCK_H, self.BLOCK_W,
            background=self.background,
            return_alpha=False
        )
        out_img = torch.clamp(out_img, 0, 1)
        out_img = out_img.view(-1, Hs, Ws, 3).permute(0, 3, 1, 2).contiguous()
        return {"render": out_img}

    def update_optimizer(self):
        current_lr = self.optimizer.param_groups[0]["lr"]
        if self.opt_type == "adam":
            self.optimizer = torch.optim.Adam(self.parameters(), lr=current_lr)
        else:
            self.optimizer = Adan(self.parameters(), lr=current_lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=20000, gamma=0.5)
        self.lr = current_lr

    def _collect_grad_norms(self, grad_ema_state: Optional[dict] = None, grad_ema_decay: float = 0.99):
        raw_sq_sum = 0.0
        ema_sq_sum = 0.0 if grad_ema_state is not None else None
        for name, p in self.named_parameters():
            if p.grad is None or not p.requires_grad:
                continue
            g = p.grad.detach()
            if g.numel() == 0:
                continue
            raw_sq_sum += float(torch.sum(g * g).item())
            if grad_ema_state is None:
                continue
            prev = grad_ema_state.get(name)
            if prev is None or prev.shape != g.shape or prev.device != g.device or prev.dtype != g.dtype:
                new_ema = g.clone()
            else:
                new_ema = prev.mul(float(grad_ema_decay)).add(
                    g, alpha=1.0 - float(grad_ema_decay))
            grad_ema_state[name] = new_ema
            ema_sq_sum += float(torch.sum(new_ema * new_ema).item())
        return math.sqrt(raw_sq_sum), (
            math.sqrt(ema_sq_sum) if ema_sq_sum is not None else None)

    def _get_gaussian_importance_score(self):
        with torch.no_grad():
            rgb_importance = torch.linalg.vector_norm(self.rgb_W.detach(), dim=1)
            cholesky = self.get_cholesky_elements.detach()
            gaussian_area = cholesky[:, 0].abs() * cholesky[:, 2].abs()
            return rgb_importance * gaussian_area

    @torch.no_grad()
    def _prune_parameter_with_optimizer_state(self, param_name, keep_indices):
        old_param = getattr(self, param_name)
        keep_indices = keep_indices.to(device=old_param.device, dtype=torch.bool)
        new_param = torch.nn.Parameter(
            old_param.detach()[keep_indices].clone(),
            requires_grad=old_param.requires_grad)
        if old_param.grad is not None:
            new_param.grad = old_param.grad.detach()[keep_indices].clone()

        for param_group in self.optimizer.param_groups:
            for index, param in enumerate(param_group["params"]):
                if param is old_param:
                    param_group["params"][index] = new_param
                    break

        old_state = self.optimizer.state.pop(old_param, {})
        new_state = {}
        for state_name, state_value in old_state.items():
            if (torch.is_tensor(state_value) and state_value.ndim > 0
                    and state_value.shape[0] == old_param.shape[0]):
                state_mask = keep_indices.to(state_value.device)
                new_state[state_name] = state_value[state_mask].clone()
            elif torch.is_tensor(state_value):
                new_state[state_name] = state_value.clone()
            else:
                new_state[state_name] = state_value
        if new_state:
            self.optimizer.state[new_param] = new_state
        setattr(self, param_name, new_param)

    @torch.no_grad()
    def _prune_gaussians(self, keep_indices):
        keep_indices = keep_indices.to(device=self._xyz.device, dtype=torch.bool)
        self._prune_parameter_with_optimizer_state("_xyz", keep_indices)
        self._prune_parameter_with_optimizer_state("_cholesky", keep_indices)
        self._prune_parameter_with_optimizer_state("_features_dc", keep_indices)
        if isinstance(self.rgb_W, torch.nn.Parameter):
            self._prune_parameter_with_optimizer_state("rgb_W", keep_indices)
        else:
            self.rgb_W = self.rgb_W[keep_indices].clone()
        self.point_ids = self.point_ids[keep_indices].clone()
        self._opacity_cache = None
        self._graph_ema = None

    def remove_Gaussians(self, iteration, removal_start=0, removal_end=4000):
        if iteration < removal_start or iteration > removal_end:
            return
        densification_num = int(self.max_num_points * self.removal_rate)
        steps = max(1, int((removal_end - removal_start)
                           / self.densification_interval))
        if iteration < removal_end:
            remove_count = int(densification_num / steps)
        else:
            remove_count = int(self._xyz.shape[0]) - int(
                self.max_num_points * (1.0 - self.removal_rate))
        remove_count = min(max(remove_count, 0), max(0, int(self._xyz.shape[0]) - 1))
        if remove_count == 0:
            return
        sorted_indices = torch.argsort(self._get_gaussian_importance_score())
        remove_indices = sorted_indices[:remove_count]
        self._vis_removed_raw_xyz = self._xyz[remove_indices].detach().clone()
        self._vis_removed_iter = int(iteration)
        keep_indices = torch.ones(
            self._xyz.shape[0], dtype=torch.bool, device=self._xyz.device)
        keep_indices[remove_indices] = False
        self._prune_gaussians(keep_indices)

    def add_Gaussians(self, iteration):
        densification_num = int(self.max_num_points * self.removal_rate)
        if densification_num <= 0:
            return
        if self.iscover:
            # Coverage mode retains deterministic Sobol initialization and may
            # use the optical-flow-guided positions prepared by the trainer.
            u = sobol_block(int(self.next_point_id.item()), densification_num,
                            device=self._xyz.device)
            guided_new_xyz = getattr(self, "_guided_new_raw_xyz", None)
            if guided_new_xyz is not None:
                new_xyz = guided_new_xyz.detach().to(
                    device=self._xyz.device, dtype=self._xyz.dtype)
                if new_xyz.shape[0] != densification_num:
                    raise ValueError(
                        f"guided Gaussian count {new_xyz.shape[0]} != {densification_num}")
            else:
                new_xyz = sobol_raw_xyz(u)
            self._guided_new_raw_xyz = None
            new_cholesky = sobol_cholesky(u, True)
            new_cholesky.mul_(self._cholesky_init_scale(
                self._xyz.shape[0] + densification_num))
            new_features_dc = 0.01 * torch.ones(
                densification_num, 3, device=self._xyz.device)
        else:
            # Without coverage, P-frame additions use exactly the same random
            # position/shape/color distributions as an anchor-frame Gaussian.
            u = sobol_block(int(self.next_point_id.item()), densification_num,
                            device=self._xyz.device)
            self._guided_new_raw_xyz = None
            new_xyz = sobol_raw_xyz(u)
            new_cholesky = sobol_cholesky(u, True)
            new_features_dc = 0.01 * torch.ones(
                densification_num, 3, device=self._xyz.device)
        self._vis_added_raw_xyz = new_xyz.detach().clone()
        self._vis_added_iter = int(iteration)
        new_rgb_W = 0.01 * torch.ones(
            densification_num, 1, device=self._xyz.device)

        self._xyz = torch.nn.Parameter(torch.cat((self._xyz, new_xyz), dim=0))
        self._cholesky = torch.nn.Parameter(
            torch.cat((self._cholesky, new_cholesky), dim=0))
        self._features_dc = torch.nn.Parameter(
            torch.cat((self._features_dc, new_features_dc), dim=0))
        self.rgb_W = torch.nn.Parameter(torch.cat((self.rgb_W, new_rgb_W), dim=0))
        new_ids = torch.arange(
            self.next_point_id.item(),
            self.next_point_id.item() + densification_num,
            dtype=torch.long, device=self.point_ids.device)
        self.point_ids = torch.cat((self.point_ids, new_ids), dim=0)
        self.next_point_id = self.next_point_id.new_tensor(
            self.next_point_id.item() + densification_num)
        self._opacity_cache = None
        self._graph_ema = None
        self.update_optimizer()

    def removal_control(self, iteration):
        if iteration % self.densification_interval == 0:
            self.remove_Gaussians(iteration, removal_start=0, removal_end=4000)

    def adaptive_control(self, iteration):
        if iteration == 1:
            self.add_Gaussians(iteration)
        if iteration % self.densification_interval == 0:
            self.remove_Gaussians(iteration, removal_start=500, removal_end=1000)

    def train_iter_scale(self, gt_image,scale,iter):
        # Match the original propagation-frame order: newly added Gaussians
        # must participate in iteration 1's forward/backward/optimizer step.
        if iter == 1 and self.isdensity:
            self.add_Gaussians(iter)
        render_pkg = self.forward_scale(scale)
        image = render_pkg["render"]
        loss = loss_fn(image.squeeze(0), gt_image.squeeze(0), self.loss_type, lambda_value=0)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none = True)
        if iter >= self.lr_decay_start_iter:
            self.scheduler.step()
        # Rank and prune using the parameters updated by this iteration, as in
        # the original main-branch overfitting implementation.
        if iter % self.densification_interval == 0:
            if self.isdensity:
                self.remove_Gaussians(iter, removal_start=500, removal_end=1000)
            elif self.isremoval:
                self.remove_Gaussians(iter, removal_start=0, removal_end=4000)
        return loss

    def train_iter(self,gt_image,iter):
        if iter == 1 and self.isdensity:
            self.add_Gaussians(iter)
        render_pkg = self.forward()
        image = render_pkg["render"]
        loss, flow_loss = self._training_loss(image.squeeze(0), gt_image.squeeze(0))
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none = True)
        self.scheduler.step()
        if iter % self.densification_interval == 0:
            if self.isdensity:
                self.remove_Gaussians(iter, removal_start=500, removal_end=1000)
            elif self.isremoval:
                self.remove_Gaussians(iter, removal_start=0, removal_end=4000)
        self._update_flow_prior(flow_loss, iter)
        return loss

    def train_iter_scale_log(self, gt_image,scale,iter, grad_ema_state: Optional[dict] = None, grad_ema_decay: float = 0.99):
        if iter == 1 and self.isdensity:
            self.add_Gaussians(iter)
        render_pkg = self.forward_scale(scale)
        image = render_pkg["render"]
        loss = loss_fn(image.squeeze(0), gt_image.squeeze(0), self.loss_type, lambda_value=0)
        loss.backward()
        raw_grad_norm, ema_grad_norm = self._collect_grad_norms(
            grad_ema_state=grad_ema_state,
            grad_ema_decay=grad_ema_decay,
        )
        with torch.no_grad():
           mse_loss = F.mse_loss(image, gt_image)
           psnr = -10.0 * torch.log10(mse_loss)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none = True)
        if iter >= self.lr_decay_start_iter:
            self.scheduler.step()
        if iter % self.densification_interval == 0:
            if self.isdensity:
                self.remove_Gaussians(iter, removal_start=500, removal_end=1000)
            elif self.isremoval:
                self.remove_Gaussians(iter, removal_start=0, removal_end=4000)
        return loss, psnr, raw_grad_norm, ema_grad_norm

    def train_iter_log(self,gt_image,iter, grad_ema_state: Optional[dict] = None, grad_ema_decay: float = 0.99):
        if iter == 1 and self.isdensity:
            self.add_Gaussians(iter)
        render_pkg = self.forward()
        image = render_pkg["render"]
        loss, flow_loss = self._training_loss(image.squeeze(0), gt_image.squeeze(0))
        loss.backward()
        raw_grad_norm, ema_grad_norm = self._collect_grad_norms(
            grad_ema_state=grad_ema_state,
            grad_ema_decay=grad_ema_decay,
        )
        with torch.no_grad():
           mse_loss = F.mse_loss(image, gt_image)
           psnr = -10.0 * torch.log10(mse_loss)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none = True)
        self.scheduler.step()
        if iter % self.densification_interval == 0:
            if self.isdensity:
                self.remove_Gaussians(iter, removal_start=500, removal_end=1000)
            elif self.isremoval:
                self.remove_Gaussians(iter, removal_start=0, removal_end=4000)
        self._update_flow_prior(flow_loss, iter)
        return loss, psnr, raw_grad_norm, ema_grad_norm
    
    def pre_train_iter(self, gt_image, iter):
        render_pkg = self.forward()
        image = render_pkg["render"]
        loss = loss_fn(image, gt_image, self.loss_type, lambda_value=0.7)
        loss.backward()
        with torch.no_grad():
            mse_loss = F.mse_loss(image, gt_image)
            psnr = 10 * math.log10(1.0 / mse_loss.item())
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none = True)
        if iter >= self.lr_decay_start_iter:
            self.scheduler.step()
        return loss, psnr
    


    
