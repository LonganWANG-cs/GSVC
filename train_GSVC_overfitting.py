from gsvc_runtime import configure_runtime_environment
configure_runtime_environment()
import os

import math
import time
from pathlib import Path
import argparse
import numpy as np
import torch
import sys
from PIL import Image, ImageDraw
import torch.nn.functional as F
from pytorch_msssim import ms_ssim
from utils import (
    EarlyStopperEMA,
    LogWriter,
    MetricEarlyStopper,
    ModelEMA,
    ResolutionScheduler,
    apply_torch_determinism,
    detect_outliers_mean_diff,
    generate_video,
    log_run_config,
    loss_fn,
    process_yuv_video_frame_start_end,
)
from tqdm import tqdm
import random
import torchvision.transforms as transforms
import csv
from GSVC_overfitting import GaussianVideo_frame
class SimpleTrainer2d:
    """Trains random 2d gaussians to fit an image."""
    def __init__(
        self,
        image,
        frame_num,
        savdir,
        loss_type,
        model_name:str = "GaussianVideo",
        num_points: int = 10000,
        max_num_points: int = 10000,
        iterations:int = 50000,
        removal_rate=0.1,
        isdensity=False,
        isremoval=True,
        enhanced=False, 
        isschedule=False,
        iscover=False,
        model_path = None,
        Trained_Model=None,
        previous_image=None,
        is_propagation_frame=False,
        args = None,
    ):
        self.device = torch.device("cuda:0")
        self.args = args

        self.frame_num=frame_num
        self.previous_image = previous_image
        self.is_propagation_frame = bool(is_propagation_frame)

        self.gt_image = image_to_tensor(image).to(self.device)

        self.isschedule = isschedule
        self.iscover=iscover
        self.isdensity=isdensity
        self.isremoval=isremoval
        self.is_psnr_loss_log=args.is_log
        self.is_visual_log=args.is_log_visual
        self.vis_gaussian_events = bool(getattr(args, "vis_gaussian_events", False))
        self.vis_dot_radius = 4
        self.vis_dot_alpha = 120
        self.densification_interval=args.densification_interval

        if self.isschedule:
            self.num_points = num_points//2
        else:
            self.num_points = num_points
        self.max_num_points=max_num_points

        self.iterations = iterations

        self.model_name=model_name

        self.data_name=args.data_name

        BLOCK_H, BLOCK_W = 16, 16

        self.H, self.W = self.gt_image.shape[2], self.gt_image.shape[3]

        self.save_first_frame = args.save_first_frame
        self.save_everyframes = args.save_everyframes
        self.log_dir = Path(f"./checkpoints/{savdir}/{args.data_name}/{args.model_name}_{args.iterations}_{args.num_points}")

        self.loss_type = loss_type
        
        self.lr_warmup_iters = 2500
        self.lr_ema_I = args.lr_ema_I
        self.lr = args.lr
        self.use_ema = bool(getattr(self.args, "use_ema", False))
        self._ema_eval_graph = None  # (graph, static_loss) for the EMA early-stop probe
        if model_name == "GaussianVideo":
            self.gaussian_model = GaussianVideo_frame(loss_type=self.loss_type, opt_type="adan", num_points=self.num_points,max_num_points=self.max_num_points,densification_interval=self.densification_interval,iterations=self.iterations, H=self.H, W=self.W, BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W, 
            device=self.device, lr=self.lr, quantize=False,removal_rate=removal_rate,isdensity=self.isdensity,isremoval=self.isremoval,enhanced=enhanced, alpha=getattr(args, "alpha", 1.0),iscover=self.iscover).to(self.device)

        if model_path is not None:
            print(f"loading model path:{model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            model_dict = self.gaussian_model.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.gaussian_model.load_state_dict(model_dict)
                          
        if Trained_Model is not None:
            checkpoint = Trained_Model
            model_dict = self.gaussian_model.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.gaussian_model.load_state_dict(model_dict)
            self.is_P_frame=True
        else:
            self.is_P_frame=False

        # Use distinct stage-1 learning rates from the first optimization step:
        # anchors start at lr_ema_I (default 1e-2), while propagation frames
        # retain lr (default 1e-3).  set_lr also updates StepLR.base_lrs, so the
        # existing step_size=20000, gamma=0.5 decay remains active.
        self.gaussian_model.set_lr(
            self.lr if self.is_propagation_frame else self.lr_ema_I)

    def _raw_xyz_to_pixel_xy(self, raw_xyz):
        if raw_xyz is None or raw_xyz.numel() == 0:
            return None
        xy = torch.tanh(raw_xyz.detach()).to(torch.float32)
        pixel_xy = torch.stack((
            (xy[:, 0] + 1.0) * (float(self.W) * 0.5),
            (xy[:, 1] + 1.0) * (float(self.H) * 0.5)), dim=1)
        pixel_xy[:, 0].clamp_(0.0, float(self.W - 1))
        pixel_xy[:, 1].clamp_(0.0, float(self.H - 1))
        return pixel_xy.cpu().numpy()

    def _save_event_overlay(self, out_image, pixel_xy, save_name):
        if pixel_xy is None or len(pixel_xy) == 0:
            return
        base_img = transforms.ToPILImage()(
            out_image.float().squeeze(0).cpu()).convert("RGBA")
        overlay = Image.new("RGBA", base_img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay, "RGBA")
        radius = self.vis_dot_radius
        for x, y in pixel_xy:
            draw.ellipse((x-radius, y-radius, x+radius, y+radius),
                         fill=(255, 0, 0, self.vis_dot_alpha))
        save_dir = self.log_dir / "gaussian_event_vis"
        save_dir.mkdir(parents=True, exist_ok=True)
        Image.alpha_composite(base_img, overlay).convert("RGB").save(
            str(save_dir / save_name))

    def _maybe_save_gaussian_event_vis(self, iteration):
        if not self.vis_gaussian_events:
            return
        event = self.gaussian_model.pop_visualization_events()
        if event is None:
            return
        with torch.no_grad():
            out_image = self.gaussian_model()["render"].detach()
        for kind in ("added", "removed"):
            raw_xyz = event.get(f"{kind}_raw_xyz")
            if raw_xyz is not None and raw_xyz.numel() > 0:
                self._save_event_overlay(
                    out_image, self._raw_xyz_to_pixel_xy(raw_xyz),
                    f"{self.frame_num}_iter_{iteration:05d}_{kind}.png")
            

    def _capture_ema_eval_graph(self, ema_model):
        """Capture (EMA forward + L2 loss) as its own CUDA graph, if possible.

        The EMA early-stop rule does not read the training loss: it renders the
        *EMA weights* and scores that.  Left eager that render is ~50 kernel
        launches -- the very cost the training graph exists to remove -- so at
        one probe per ``early_stop_every`` iterations it would claw back a few
        percent of the win.  Captured, a probe costs one graph launch plus the
        single sync needed to read the scalar, the same price the non-EMA
        probe pays.  Returns True on success; on failure the caller silently
        falls back to the eager render (correct, just slower).  Set
        GSVC_CUDAGRAPH_EMA_EVAL=0 to force the eager probe.
        """
        if os.environ.get("GSVC_CUDAGRAPH_EMA_EVAL", "1") != "1":
            return False
        if self.loss_type != "L2":
            return False  # only the plain-MSE probe is worth graphing
        try:
            with torch.no_grad():
                ema_model.eval()
                _ = ema_model()  # materialize lazy buffers (_opacity_cache) on this stream
            torch.cuda.synchronize()
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.no_grad(), torch.cuda.stream(side):
                for _ in range(3):
                    out = ema_model()["render"]
                    _ = loss_fn(out.float(), self.gt_image.float(), self.loss_type, lambda_value=0)
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.no_grad(), torch.cuda.graph(graph):
                out = ema_model()["render"]
                static_loss = loss_fn(out.float(), self.gt_image.float(), self.loss_type, lambda_value=0)
            self._ema_eval_graph = (graph, static_loss)
            return True
        except Exception as exc:
            print(f"[cudagraph-train] EMA eval graph capture failed ({exc}); "
                  f"falling back to an eager EMA probe", flush=True)
            self._ema_eval_graph = None
            return False

    def _ema_eval_loss(self, ema_model):
        """Loss of the current EMA weights against the GT frame (one sync)."""
        cached = getattr(self, "_ema_eval_graph", None)
        if cached is not None:
            graph, static_loss = cached
            graph.replay()
            return float(static_loss.item())
        with torch.no_grad():
            ema_model.eval()
            out = ema_model()["render"]
            loss_v = loss_fn(out.float(), self.gt_image.float(), self.loss_type, lambda_value=0)
        return float(loss_v.item())

    def _train_static_cudagraph_manual_gsplat(self, early_stop_fn=None, early_stop_every=100,
                                              total_override=None, start_it=0,
                                              ema_mgr=None, ema_decay=0.999,
                                              reset_capacity=True, baseline_override=None):
        """Train a fixed-shape (sub)sequence with CUDA Graphs of the native-gsplat
        manual step (forward + native backward + optimizer).  A graph is
        re-captured whenever StepLR changes the optimizer LR. ``total_override`` runs only that many
        iterations (the stable tail after an eager densification/resolution
        ramp); ``start_it`` offsets early-stop bookkeeping.

        When ``ema_mgr`` is given, its shadow model is bound into the step
        (``bind_graph_ema``) so the EMA average is advanced *inside* the graph,
        and the EMA early-stop probe is captured as a second graph.

        Valid only when the Gaussian count, resolution, and optimizer parameter
        addresses stay constant across the captured region.
        """
        model = self.gaussian_model
        gt = self.gt_image
        total = int(self.iterations) if total_override is None else int(total_override)
        # Finish every configured exact-count sample before capture; otherwise
        # resolve_intersect_capacity would call .item() inside the graph.
        isect_warmup = max(0, int(os.environ.get("GSPLAT_ISECT_WARMUP", "8")))
        warmup = min(max(20, isect_warmup), total)
        if baseline_override is None:
            eager_baseline = min(200, max(0, total - warmup))
        else:
            eager_baseline = min(int(baseline_override), max(0, total - warmup))
        if total <= 0:
            return 0.0, None, 0

        # Per-(resolution, count) intersect capacity is a process-global cache;
        # reset it so THIS shape re-warms and freezes its own capacity in the
        # eager warmup below, before capture (a capacity frozen for another frame
        # can be too small and trip the device-side assert).  A follow-on segment
        # of the SAME frame must NOT reset: it has the same resolution and count,
        # so it should inherit the capacity the first segment froze rather than
        # re-warm and pick a different one.
        if reset_capacity:
            import gsplat.utils as gsplat_utils
            gsplat_utils._ISECT_CAP_WARMUP = max(
                0, int(os.environ.get("GSPLAT_ISECT_WARMUP", "8")))
            gsplat_utils._ISECT_CAP_STATE.clear()

        t0 = time.time()
        # Bind the EMA before the warmup, so those iterations average into the
        # shadow exactly as the eager loop's `train_iter(); update_ema()` would.
        # It must be bound only here: every eager ModelEMA.update() rebinds the
        # shadow tensors, so addresses are stable only once eager updates stop.
        model.bind_graph_ema(ema_mgr.ema if ema_mgr is not None else None, ema_decay)
        try:
            return self._run_static_cudagraph(model, gt, total, warmup, eager_baseline,
                                              early_stop_fn, early_stop_every, start_it,
                                              ema_mgr, t0)
        finally:
            model.bind_graph_ema(None, ema_decay)
            had_eval_graph = self._ema_eval_graph is not None
            self._ema_eval_graph = None
            if had_eval_graph and ema_mgr is not None:
                # forward() caches xys/radii on the module; captured, those point
                # into the eval graph's private pool.  Re-run it eagerly so the
                # module holds ordinary tensors once that pool is released.
                with torch.no_grad():
                    ema_mgr.ema()

    def _run_static_cudagraph(self, model, gt, total, warmup, eager_baseline,
                              early_stop_fn, early_stop_every, start_it, ema_mgr, t0):
        # Every real optimizer step must also advance StepLR.  CUDA Graph capture
        # records the optimizer's current scalar LR, so when StepLR changes it we
        # discard that graph and capture a new one with the updated LR.
        executed = 0
        loss = None

        initial_eager = min(total, warmup + eager_baseline)
        for _ in range(initial_eager):
            loss = model.manual_gsplat_train_step(gt)
            model.scheduler.step()
            executed += 1
        torch.cuda.synchronize()

        stopped = False
        while executed < total and not stopped:
            # Side-stream warmup is real training and therefore counts toward
            # both the iteration budget and the scheduler, just as eager steps.
            side_steps = min(3, total - executed)
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(side_steps):
                    loss = model.manual_gsplat_train_step(gt)
                    model.scheduler.step()
                    executed += 1
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            if executed >= total:
                break

            graph = torch.cuda.CUDAGraph()
            try:
                with torch.cuda.graph(graph):
                    static_loss = model.manual_gsplat_train_step(gt)
            except Exception as exc:
                raise RuntimeError("native-gsplat manual CUDA-graph capture failed") from exc

            self._ema_eval_graph = None
            if ema_mgr is not None:
                self._capture_ema_eval_graph(ema_mgr.ema)

            captured_lr = model.optimizer.param_groups[0]["lr"]
            while executed < total:
                graph.replay()
                model.scheduler.step()
                executed += 1
                loss = static_loss
                absolute_it = start_it + executed
                if (early_stop_fn is not None and early_stop_every > 0
                        and absolute_it % early_stop_every == 0
                        and early_stop_fn(absolute_it, static_loss)):
                    print(f"[cudagraph-train] early stop at iteration {absolute_it}", flush=True)
                    stopped = True
                    break
                if model.optimizer.param_groups[0]["lr"] != captured_lr:
                    # The replay just completed with captured_lr; StepLR's new
                    # value applies to the next iteration, which is re-captured.
                    break

        torch.cuda.synchronize()
        if loss is None:
            return time.time() - t0, None, executed
        # A captured loss belongs to its graph pool; clone before local graph
        # references are released at return.
        return time.time() - t0, loss.detach().clone(), executed

    # Segment 1 must not begin earlier than this even when the Gaussian count is
    # already final.  At this boundary the process-global intersection cache is
    # cleared, warmed again, and only then frozen for CUDA Graph capture.  Waiting
    # until iteration 5000 avoids sizing it from early, unstable footprints.
    _CG_MIN_CAPTURE_ITER = 5000

    @staticmethod
    def _use_dynamic_intersect_capacity():
        """Use exact intersection counts before the delayed graph boundary."""
        import gsplat.utils as gsplat_utils

        gsplat_utils._ISECT_CAP_WARMUP = 0
        gsplat_utils._ISECT_CAP_STATE.clear()

    def _plan_graph_segments(self, shape_ready, last_iter, ema_warmup, use_ema,
                             min_segment=260):
        """Plan how to cover ``shape_ready+1 .. last_iter`` with captured graphs.

        Returns a list of ``(start_iter, end_iter, fold_ema)``; empty when
        nothing left is long enough to pay for a capture.

        **Why two segments.** With ``--use_ema`` a single capture cannot begin
        until ``ModelEMA.reset()`` has seeded the shadow at ``ema_warmup``,
        because ``bind_graph_ema`` freezes the shadow's tensor addresses and
        every eager ``ModelEMA.update()`` rebinds them.  If the shape becomes
        final before the EMA boundary, a first graph can omit the EMA fold and a
        second graph can include it.  With the default 5000 capture boundary the
        EMA is normally already active, so the plan usually needs one segment.
        """
        start = max(int(shape_ready) + 1, self._CG_MIN_CAPTURE_ITER)
        if start > last_iter:
            return []
        if use_ema:
            boundary = int(ema_warmup)
            if boundary >= start:
                if (boundary - start + 1 > min_segment
                        and last_iter - boundary > min_segment):
                    return [(start, boundary, False), (boundary + 1, last_iter, True)]
                # One half is too short to earn its capture; fall back to the
                # single folded segment the one-segment code used.
                start = boundary + 1
        if last_iter - start + 1 <= min_segment:
            return []
        return [(start, last_iter, bool(use_ema))]

    def _run_graph_plan(self, plan, ema_mgr, ema_decay, maybe_update_ema,
                        early_stop_fn, early_stop_every):
        """Run the segments from :meth:`_plan_graph_segments` in order.

        Returns the absolute iteration number the last segment reached, so the
        caller can log how many iterations the frame actually ran (early
        stopping inside a segment cuts it short).
        """
        last_it = 0
        for idx, (seg_start, seg_end, fold_ema) in enumerate(plan):
            total = seg_end - seg_start + 1
            elapsed, gloss, ran = self._train_static_cudagraph_manual_gsplat(
                early_stop_fn, early_stop_every,
                total_override=total, start_it=seg_start - 1,
                ema_mgr=(ema_mgr if fold_ema else None), ema_decay=ema_decay,
                # A follow-on segment inherits the first segment's frozen
                # capacity and its already-materialised optimizer/grad/work
                # buffers, so it needs neither a capacity re-warm nor the long
                # eager baseline -- only the short side-stream warmup.
                reset_capacity=(idx == 0),
                baseline_override=(None if idx == 0 else 0))
            # CUDA Graph timing/statistics output disabled to avoid the
            # device-to-host synchronization caused by float(gloss).
            # print(f"[cudagraph-train] frame={self.frame_num} "
            #       f"seg={idx + 1}/{len(plan)} ema_folded={fold_ema} "
            #       f"from_iter={seg_start} seg_iters={total} seg_ran={ran} "
            #       f"seg_wall={elapsed:.3f} s "
            #       f"seg_per_iter={elapsed / max(1, ran) * 1e3:.3f} ms "
            #       f"loss={float(gloss):.6f}", flush=True)
            last_it = seg_start - 1 + ran
            if ran < total:
                return last_it  # early stopping broke out inside this segment
            if idx + 1 < len(plan):
                # The eager loop calls ModelEMA.reset() at exactly ema_warmup.
                # Do it here, between the segments, so the next capture binds an
                # already-seeded shadow whose addresses are then stable.
                maybe_update_ema(seg_end)
        return last_it

    def train(self,frame):
        # Iterations this frame actually executes.  Early stopping and the
        # CUDA-graph paths can both end a frame well before `self.iterations`,
        # so the configured budget is only an upper bound; main() logs this.
        self.iters_ran = int(self.iterations)
        self.early_stop_iter = None
        if os.environ.get("GSVC_CUDAGRAPH_TRAIN", "1") == "1":
            self._use_dynamic_intersect_capacity()
        ema_mgr = None
        early_metric = None
        if self.use_ema:
            ema_decay = float(getattr(self.args, "ema_decay", 0.99))
            ema_warmup = int(getattr(self.args, "ema_warmup", 2500))
            early_stop_every = int(getattr(self.args, "early_stop_every", 100))
            early_stop_warmup = int(getattr(self.args, "early_stop_warmup", 2500))
            early_stop_patience = int(getattr(self.args, "early_stop_patience", 100))
            early_stop_min_delta = float(getattr(self.args, "early_stop_min_delta", 1e-9))
            ema_mgr = ModelEMA(self.gaussian_model, decay=ema_decay)
            ema_mgr.to(self.device)
            early_metric = MetricEarlyStopper(
                patience=early_stop_patience,
                min_delta=early_stop_min_delta,
                mode="min",
            )
            ema_started = False
            def _maybe_update_ema(it: int):
                nonlocal ema_started
                if ema_mgr is None:
                    return
                if it < ema_warmup:
                    return
                if not ema_started:
                    ema_mgr.reset(self.gaussian_model)
                    ema_started = True
                    return
                ema_mgr.update(self.gaussian_model)
            def _maybe_early_stop(it,loss):
                if (ema_mgr is None) or (early_metric is None):
                    return False
                if not ema_started:
                    return False
                if it < early_stop_warmup:
                    return False
                if early_stop_every <= 0:
                    return False
                if it % early_stop_every != 0:
                    return False
                # Scores the EMA weights, not the training loss; under graph
                # replay this render is itself a captured graph (see
                # _capture_ema_eval_graph), otherwise it runs eagerly.
                return early_metric.step(self._ema_eval_loss(ema_mgr.ema))
        else:
            ema_decay = float(getattr(self.args, "ema_decay", 0.999))
            early_stop_every = int(getattr(self.args, "early_stop_every", 100))
            early_stop_warmup = int(getattr(self.args, "early_stop_warmup", 2500))
            early_stop_patience = int(getattr(self.args, "early_stop_patience", 100))
            early_stop_min_delta = float(getattr(self.args, "early_stop_min_delta", 1e-9))
            early = EarlyStopperEMA(
                patience=early_stop_patience,
                min_rel_improve=early_stop_min_delta,
                ema_beta=ema_decay,         # EMA smoothing factor
                check_every=early_stop_every,         # Perform a check every 10 iterations
                warmup=early_stop_warmup             # Disable early stopping during the first 2500 iterations
            )
            def _maybe_update_ema(it: int):
                return
            def _maybe_early_stop(it,loss):
                return early.step(loss,it)

        grad_ema_decay = float(getattr(self.args, "grad_ema_decay", 0.95))
        grad_ema_state = {}
        
        if not self.isschedule: #P-frame, I-frame(without schedule) 
            if not self.is_P_frame: #I-frame(without schedule) 
                self.gaussian_model.apply_scale_to_initial_cholesky() 
                if self.use_ema:
                    self.gaussian_model.set_lr(self.lr_ema_I)
            self.gaussian_model.train()
            if self.is_psnr_loss_log:
                torch.cuda.synchronize()
                start_time = time.time()
                psnr_list=[]
                loss_list=[]
                iter_list=[]
                grad_list=[]
                if self.is_visual_log:
                    self.render_iter(0)
                is_not_stop=True
                for iter in range(1, int(self.iterations)+1):
                    loss, psnr, raw_grad_norm, ema_grad_norm = self.gaussian_model.train_iter_log(
                        self.gt_image,
                        iter,
                        grad_ema_state=grad_ema_state if self.use_ema else None,
                        grad_ema_decay=grad_ema_decay,
                    )
                    self._maybe_save_gaussian_event_vis(iter)
                    _maybe_update_ema(iter)
                    if _maybe_early_stop(iter,loss) and is_not_stop:
                        print(f"Early stopping at iteration {iter}")
                        os.makedirs(self.log_dir, exist_ok=True)
                        stop_iter_path = os.path.join(self.log_dir, "stop_iter.txt")
                        with open(stop_iter_path, "w") as f:
                            f.write(str(iter))
                        self.early_stop_iter = iter
                        is_not_stop=False
                    if iter % 50 == 0:
                        if self.use_ema:
                            grad_list.append(float(ema_grad_norm if ema_grad_norm is not None else raw_grad_norm))
                        else:
                            grad_list.append(float(raw_grad_norm))
                        if (ema_mgr is not None) and ema_started:
                            with torch.no_grad():
                                ema_mgr.ema.eval()
                                out = ema_mgr.ema()
                                out_image = out["render"]
                                if self.is_visual_log:
                                    self.render_iter_ema(iter, out_image)
                                loss_ema = loss_fn(out_image.float(), self.gt_image.float(), self.loss_type, lambda_value=0)
                                mse_loss = F.mse_loss(out_image.float(), self.gt_image.float())
                                psnr_ema = 10 * math.log10(1.0 / mse_loss.item())
                            psnr_list.append(float(psnr_ema))
                            loss_list.append(float(loss_ema.item()))
                        else:
                            if self.is_visual_log:
                                self.render_iter(iter)
                            psnr_list.append(psnr.item())
                            loss_list.append(loss.item())
                        iter_list.append(iter)
                torch.cuda.synchronize()
                end_time = time.time() - start_time
                os.makedirs(self.log_dir, exist_ok=True)
                csv_path = os.path.join(self.log_dir, "training_log.csv")
                with open(csv_path, mode="w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["iter", "psnr", "loss", "grad_norm"])
                    for i, p, l, g in zip(iter_list, psnr_list, loss_list, grad_list):
                        writer.writerow([i, p, l, g])
            else:
                torch.cuda.synchronize()
                start_time = time.time()
                _cg_train = os.environ.get("GSVC_CUDAGRAPH_TRAIN", "1") == "1"
                _cg_es_every = int(getattr(self.args, "early_stop_every", 100))
                _cg_iters = int(self.iterations)
                # Plan the graphed region up front.  The shape is final after
                # adaptive_control_done_iter(); with EMA the region is split at
                # ema_warmup into an unfolded and a folded segment rather than
                # waiting for the shadow (see _plan_graph_segments).
                _cg_plan = self._plan_graph_segments(
                    shape_ready=self.gaussian_model.adaptive_control_done_iter(),
                    last_iter=_cg_iters,
                    ema_warmup=int(getattr(self.args, "ema_warmup", 2500)),
                    use_ema=self.use_ema) if _cg_train else []
                # Even a fully static frame remains eager until the delayed
                # intersection-capacity boundary represented by _cg_plan.
                continue_training = True
                if continue_training:
                    for iter in range(1, _cg_iters + 1):
                        # Hand off to the planned graph segments at the first
                        # iteration they cover; everything before that is the
                        # eager densify/prune ramp, which changes the count and
                        # so cannot be captured.
                        if _cg_plan and iter == _cg_plan[0][0]:
                            last_it = self._run_graph_plan(_cg_plan, ema_mgr, ema_decay,
                                                           _maybe_update_ema, _maybe_early_stop,
                                                           _cg_es_every)
                            self.iters_ran = int(last_it or iter)
                            break
                        loss = self.gaussian_model.train_iter(self.gt_image, iter)
                        self._maybe_save_gaussian_event_vis(iter)
                        _maybe_update_ema(iter)
                        if _maybe_early_stop(iter,loss):
                            print(f"Early stopping at iteration {iter}")
                            self.iters_ran = iter
                            self.early_stop_iter = iter
                            break
                torch.cuda.synchronize()
                end_time = time.time() - start_time

        else: #I-frame schedule
            sched = ResolutionScheduler(original_image=self.gt_image, iterations=self.iterations, resolution_mode="freq",gaussian_number=self.max_num_points)
            lr_decay_start_iter = int(sched.lr_decay_from_iter())
            self.gaussian_model.lr_decay_start_iter = max(lr_decay_start_iter, self.lr_warmup_iters + 1)
            ks = np.arange(self.iterations)
            sched_scales_int = np.array([sched.get_res_scale(k) for k in ks], dtype=np.int64)
            self.gaussian_model.prepare_scale_cache(sched_scales_int.tolist())
            sched_gauss_num = np.array([sched.get_gaussian_budget(k) for k in ks], dtype=np.int64)
            gt_cache = {}
            current_scale = int(sched_scales_int[0])
            gt_cache[current_scale] = sched.get_img_scale(0)
            current_gt = gt_cache[current_scale]
            current_budget = int(sched_gauss_num[0])
            self.gaussian_model.grow_points_to(current_budget)
            # self.render_iter(0,current_scale)
            self.gaussian_model.apply_scale_to_initial_cholesky() 
            self.gaussian_model.train()  
            if self.is_psnr_loss_log:
                torch.cuda.synchronize()
                start_time = time.time()
                psnr_list=[]
                loss_list=[]
                iter_list=[]
                grad_list=[]
                if self.is_visual_log:
                    self.render_iter(0)
                is_not_stop=True
                for iter in range(1, int(self.iterations)+1):
                    if iter == self.lr_warmup_iters + 1:
                        self.gaussian_model.set_lr(self.lr_ema_I)
                    scale_k = int(sched_scales_int[iter - 1])
                    if scale_k != current_scale:
                        current_scale = scale_k
                        if current_scale not in gt_cache:
                            gt_cache[current_scale] = sched.get_img_scale(iter - 1)
                        current_gt = gt_cache[current_scale]
                    target_budget = int(sched_gauss_num[iter - 1])
                    if target_budget > current_budget:
                        current_budget = target_budget
                        self.gaussian_model.grow_points_to(target_budget)
                    loss, psnr, raw_grad_norm, ema_grad_norm = self.gaussian_model.train_iter_scale_log(
                        current_gt,
                        current_scale,
                        iter,
                        grad_ema_state=grad_ema_state if self.use_ema else None,
                        grad_ema_decay=grad_ema_decay,
                    )
                    self._maybe_save_gaussian_event_vis(iter)
                    _maybe_update_ema(iter)
                    if _maybe_early_stop(iter,loss) and is_not_stop:
                        print(f"Early stopping at iteration {iter}")
                        os.makedirs(self.log_dir, exist_ok=True)
                        stop_iter_path = os.path.join(self.log_dir, "stop_iter.txt")
                        with open(stop_iter_path, "w") as f:
                            f.write(str(iter))
                        self.early_stop_iter = iter
                        is_not_stop=False
                    if iter % 50 == 0:
                        if self.use_ema:
                            grad_list.append(float(ema_grad_norm if ema_grad_norm is not None else raw_grad_norm))
                        else:
                            grad_list.append(float(raw_grad_norm))
                        if (ema_mgr is not None) and ema_started:
                            with torch.no_grad():
                                ema_mgr.ema.eval()
                                out = ema_mgr.ema()
                                out_image = out["render"]
                                if self.is_visual_log:
                                    self.render_iter_ema(iter, out_image)
                                loss_ema = loss_fn(out_image.float(), self.gt_image.float(), self.loss_type, lambda_value=0)
                                mse_loss = F.mse_loss(out_image.float(), self.gt_image.float())
                                psnr_ema = 10 * math.log10(1.0 / mse_loss.item())
                            psnr_list.append(float(psnr_ema))
                            loss_list.append(float(loss_ema.item()))
                        else:
                            psnr_list.append(psnr.item())
                            loss_list.append(loss.item())
                            if self.is_visual_log:
                                self.render_iter_scale(iter,current_scale)
                        iter_list.append(iter)
                torch.cuda.synchronize()
                end_time = time.time() - start_time
                os.makedirs(self.log_dir, exist_ok=True)
                csv_path = os.path.join(self.log_dir, "training_log.csv")
                with open(csv_path, mode="w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["iter", "psnr", "loss", "grad_norm"])
                    for i, p, l, g in zip(iter_list, psnr_list, loss_list, grad_list):
                        writer.writerow([i, p, l, g])
            else:
                torch.cuda.synchronize()
                start_time = time.time()
                _cg_train = os.environ.get("GSVC_CUDAGRAPH_TRAIN", "1") == "1"
                _cg_es_every = int(getattr(self.args, "early_stop_every", 100))
                _cg_ema_warmup = int(getattr(self.args, "ema_warmup", 2500)) if self.use_ema else 0
                _sched_final_scale = int(sched_scales_int[-1])
                _cg_iters = int(self.iterations)
                for iter in range(1, _cg_iters + 1):
                    if iter == self.lr_warmup_iters + 1:
                        self.gaussian_model.set_lr(self.lr_ema_I)
                    scale_k = int(sched_scales_int[iter - 1])
                    if scale_k != current_scale:
                        current_scale = scale_k
                        if current_scale not in gt_cache:
                            gt_cache[current_scale] = sched.get_img_scale(iter - 1)
                        current_gt = gt_cache[current_scale]
                    target_budget = int(sched_gauss_num[iter - 1])
                    if target_budget > current_budget:
                        current_budget = target_budget
                        self.gaussian_model.grow_points_to(target_budget)
                    # Graph the tail only once resolution is full AND adaptive
                    # densify/prune has finished, past LR warmup, so the graphed
                    # tail matches eager's final resolution and Gaussian count.
                    if (_cg_train and current_scale == 1 and _sched_final_scale == 1
                            and iter >= self._CG_MIN_CAPTURE_ITER
                            and iter > self.gaussian_model.adaptive_control_done_iter()
                            and iter > self.lr_warmup_iters + 1
                            and iter > _cg_ema_warmup):
                        remaining = _cg_iters - iter + 1
                        if remaining > 260:
                            elapsed_g, gloss, ran_g = self._train_static_cudagraph_manual_gsplat(
                                _maybe_early_stop, _cg_es_every, total_override=remaining, start_it=iter - 1,
                                ema_mgr=ema_mgr, ema_decay=ema_decay)
                            self.iters_ran = iter - 1 + int(ran_g)
                            # CUDA Graph tail timing/statistics output disabled
                            # to avoid synchronizing on float(gloss).
                            # print(f"[cudagraph-train] frame={self.frame_num} graph_tail_from_iter={iter} "
                            #       f"tail_iters={remaining} tail_ran={ran_g} tail_wall={elapsed_g:.3f} s "
                            #       f"tail_per_iter={elapsed_g / max(1, ran_g) * 1e3:.3f} ms "
                            #       f"loss={float(gloss):.6f}", flush=True)
                            break
                    loss = self.gaussian_model.train_iter_scale(current_gt, current_scale, iter)
                    self._maybe_save_gaussian_event_vis(iter)
                    _maybe_update_ema(iter)
                    if _maybe_early_stop(iter,loss):
                        print(f"Early stopping at iteration {iter}")
                        self.iters_ran = iter
                        self.early_stop_iter = iter
                        break
                torch.cuda.synchronize()
                end_time = time.time() - start_time
  
        num_gaussian_points =self.gaussian_model._xyz.size(0)
        if (ema_mgr is not None) and ema_started:
            orig_model = self.gaussian_model
            self.gaussian_model = ema_mgr.ema
            psnr_value, ms_ssim_value,img = self.test(frame,num_gaussian_points)
            self.gaussian_model = orig_model
            final_model_for_export = ema_mgr.ema
        else:
            psnr_value, ms_ssim_value,img = self.test(frame,num_gaussian_points)
            final_model_for_export = self.gaussian_model

        with torch.no_grad():
            self.gaussian_model.eval()
            torch.cuda.synchronize()
            test_start_time = time.time()
            for i in range(100):
                _ = self.gaussian_model()
            torch.cuda.synchronize()
            test_end_time = (time.time() - test_start_time) / 100
        Gmodel = final_model_for_export.state_dict()

        filtered_Gmodel = {
            k: v for k, v in Gmodel.items()
            if k in ['_xyz', '_cholesky','_features_dc','rgb_W']
        }
        # filtered_Gmodel['_features_dc']=final_model_for_export.get_features
        filtered_Gmodel['point_ids']=final_model_for_export.point_ids
        filtered_Gmodel['next_point_id']=final_model_for_export.next_point_id
        loss=0
        return psnr_value, ms_ssim_value, end_time, test_end_time, 1/test_end_time, \
         filtered_Gmodel, img, num_gaussian_points, loss

    def pre_train(self):     
        progress_bar = tqdm(range(1, int(self.iterations)+1), desc="Training progress")
        self.gaussian_model.train()
        for iter in range(1, int(self.iterations)+1):
            loss, psnr = self.gaussian_model.pre_train_iter(self.gt_image, iter)
            with torch.no_grad():
                if iter % 10 == 0:
                    num_gaussian_points = self.gaussian_model._xyz.size(0)
                    progress_bar.set_postfix({"Loss": f"{loss.item():.7f}", "PSNR": f"{psnr:.4f}", "NumGauss": f"{num_gaussian_points}"})
                    progress_bar.update(10)
        progress_bar.close()
        Gmodel =self.gaussian_model.state_dict()
        filtered_Gmodel = {
            k: v for k, v in Gmodel.items()
            if k in ['_xyz', '_cholesky','_features_dc','rgb_W']
        }
        # filtered_Gmodel['_features_dc']=final_model_for_export.get_features
        filtered_Gmodel['point_ids']=self.gaussian_model.point_ids
        return filtered_Gmodel, loss

    def render_iter_scale(self,iter,scale):
        with torch.no_grad():
            out = self.gaussian_model.forward_scale(scale)
            out_image = out["render"]
            save_path_img = self.log_dir / "img"
            save_path_img.mkdir(parents=True, exist_ok=True)
            transform = transforms.ToPILImage()
            img = transform(out_image.float().squeeze(0))
            name =str(self.frame_num) + f"_fitting_{iter}.png" 
            img.save(str(save_path_img / name))
    def render_iter(self,iter):
        with torch.no_grad():
            out = self.gaussian_model.forward()
            out_image = out["render"]
            save_path_img = self.log_dir / "img"
            save_path_img.mkdir(parents=True, exist_ok=True)
            transform = transforms.ToPILImage()
            img = transform(out_image.float().squeeze(0))
            name =str(self.frame_num) + f"_fitting_{iter}.png" 
            img.save(str(save_path_img / name))

    def render_iter_ema(self,iter,out_image):
        with torch.no_grad():
            save_path_img = self.log_dir / "img"
            save_path_img.mkdir(parents=True, exist_ok=True)
            transform = transforms.ToPILImage()
            img = transform(out_image.float().squeeze(0))
            name =str(self.frame_num) + f"_fitting_{iter}.png" 
            img.save(str(save_path_img / name))

    def test(self,frame,num_gaussian_points):
        self.gaussian_model.eval()
        with torch.no_grad():
            out = self.gaussian_model()
            out_image = out["render"]
        mse_loss = F.mse_loss(out_image.float(), self.gt_image.float())
        psnr = 10 * math.log10(1.0 / mse_loss.item())
        ms_ssim_value = ms_ssim(out_image.float(), self.gt_image.float(), data_range=1, size_average=True).item()
        transform = transforms.ToPILImage()
        img = transform(out_image.float().squeeze(0))
        if self.save_everyframes:
            save_path_img = self.log_dir / "img"
            save_path_img.mkdir(parents=True, exist_ok=True)
            name =str(self.frame_num) + "_fitting.png" 
            img.save(str(save_path_img / name))
        elif (frame==0 or (frame+1)%100==0 ) and self.save_first_frame:
            save_path_img = self.log_dir / "img"
            save_path_img.mkdir(parents=True, exist_ok=True)
            name =str(self.frame_num) + "_fitting.png" 
            img.save(str(save_path_img / name))       
        return psnr, ms_ssim_value,img

def image_to_tensor(img: Image.Image):
    transform = transforms.ToTensor()
    img_tensor = transform(img).unsqueeze(0)  # [1, C, H, W]
    return img_tensor

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-d", "--dataset", type=str, default='/home/e/e1344641/data/UVG/Beauty/Beauty_1920x1080_120fps_420_8bit_YUV.yuv', help="Training dataset"
    )
    parser.add_argument(
        "--data_name", type=str, default='Beauty', help="Training dataset"
    )
    parser.add_argument(
        "--model_name", type=str, default="GaussianVideo"
    )
    parser.add_argument(
        "--model_path", type=str, default=None, help="Path to a load models"
    )
    parser.add_argument(
        "--savdir", type=str, default="result", help="Path to save results"
    )
    parser.add_argument(
        "--savdir_m", type=str, default="models", help="Path to save models"
    )
    parser.add_argument(
        "--fps", type=int, default=24, help="number of frames per second (default: %(default)s)"
    )
    parser.add_argument(
        "--image_length", type=int, default=50, help="number of input frames (default: %(default)s)"
    )
    parser.add_argument(
        "--width", type=int, default=1920, help="width (default: %(default)s)"
    )
    parser.add_argument(
        "--height", type=int, default=1080, help="height (default: %(default)s)"
    )
    parser.add_argument(
        "--iterations", type=int, default=50000, help="number of training iterations (default: %(default)s)"
    )
    parser.add_argument(
        "--densification_interval",type=int,default=100,help="densification_interval (default: %(default)s)"
    )
    parser.add_argument(
        "--num_points", type=int, default=10000, help="2D GS points (default: %(default)s)"
    )
    parser.add_argument(
        "--loss_type", type=str, default='L2', help="Type of Loss"
    )
    parser.add_argument(
        "--seed", type=int, default=1, help="Set random seed for reproducibility"
    )
    parser.add_argument(
        "--removal_rate", type=float, default=0.1, help="Removal rate"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Initial Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--lr_ema_I", type=float, default=1e-2, help="Learning rate for ema model (default: %(default)s)",
    )
    parser.add_argument(
        "--alpha", type=float, default=32, help="Cholesky init scale alpha (scale = Hs*Ws/(alpha*num_points), Hs/Ws depend on scheduler scale)"
    )
    parser.add_argument(
        "--look_ahead", type=int, default=0, help="Frame enhancement referring frame number (default: %(default)s)"
    )
    parser.add_argument(
        "--save_first_frame", action="store_true", help="Save first frame"
    )
    parser.add_argument(
        "--save_everyframes", action="store_true", help="Save Every frame"
    )
    parser.add_argument(
        "--is_log", action="store_true", help="Save PSNR and loss log"
    )
    parser.add_argument(
        "--is_log_visual", action="store_true", help="Save PSNR and loss log"
    )
    parser.add_argument(
        "--is_ad", action="store_true", help="Adaptive control of gaussians setup (GSA+GSP)"
    )
    parser.add_argument(
        "--is_rm", action="store_true", help="Removal control of gaussians setup (GSP)"
    )
    parser.add_argument(
        "--is_cover", action="store_true", help="Coverage initalization"
    )
    parser.add_argument(
        "--is_sch", action="store_true", help="Progressive Training"
    )
    parser.add_argument(
        "--use_ema", action="store_true", help="Use weight-EMA model for validation/early-stop and final output"
    )
    parser.add_argument(
        "--ema_decay", type=float, default=0.99, help="EMA decay for model weights"
    )
    parser.add_argument(
        "--ema_warmup", type=int, default=2500, help="Do not maintain EMA before this iteration"
    )
    parser.add_argument(
        "--early_stop_every", type=int, default=100, help="Evaluate EMA model loss every N iterations for early stopping"
    )
    parser.add_argument(
        "--early_stop_warmup", type=int, default=2500, help="Do not early-stop before this iteration"
    )
    parser.add_argument(
        "--early_stop_patience", type=int, default=100, help="Stop after N EMA-evals without improvement"
    )
    parser.add_argument(
        "--early_stop_min_delta", type=float, default=1e-9, help="Minimum loss improvement to be considered better"
    )
    parser.add_argument(
        "--grad_ema_decay", type=float, default=0.99, help="EMA decay for stored gradient statistics"
    )
    parser.add_argument(
        "--vis_gaussian_events", action="store_true",
        help="Visualize Gaussians added by GSA and removed by GSP"
    )
    args = parser.parse_args(argv)
    # Kept for log_run_config(), which needs the parser to tell a value given on
    # the command line from one that came from its default.  Leading underscore
    # keeps it out of the dumped argument table.
    args._parser = parser
    return args

def make_save_state_from_init_state(state):
    save_state = {}

    for k, v in state.items():
        if torch.is_tensor(v):
            save_state[k] = v.detach().clone()
        else:
            save_state[k] = v

    if "_features_dc" in save_state and "rgb_W" in save_state:
        save_state["_features_dc"] = save_state["_features_dc"] * save_state["rgb_W"]
        del save_state["rgb_W"]

    return save_state

def main(argv):
    args = parse_args(argv)
    args.save_imgs=True
    loss_type=args.loss_type
    savdir=args.savdir
    savdir_m=args.savdir_m
    width = args.width
    height = args.height
    removal_rate=args.removal_rate
    look_ahead=args.look_ahead
    gmodel_save_path = Path(f"./checkpoints/{savdir_m}/{args.data_name}/{args.model_name}_{args.iterations}_{args.num_points}")
    gmodel_save_path.mkdir(parents=True, exist_ok=True)
    is_ad=args.is_ad
    is_rm=args.is_rm
    is_schedule=args.is_sch
    iscover=args.is_cover
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(args.seed)
    # Seeding and cudnn cover PyTorch's own choices; GSVC_DETERMINISTIC_TORCH
    # (with CUBLAS_WORKSPACE_CONFIG) covers the kernels that still add in a
    # run-dependent order.  Applied outside the seed block so it is honoured even
    # when --seed is not given.
    apply_torch_determinism()

    logwriter = LogWriter(Path(f"./checkpoints/{savdir}/{args.data_name}/{args.model_name}_{args.iterations}_{args.num_points}"))
    log_run_config(logwriter, args, argv=argv, parser=getattr(args, "_parser", None),
                   stage="stage 1: overfitting (train_GSVC_overfitting.py)")
    psnrs, ms_ssims, training_times, eval_times, eval_fpses, gaussian_number = [], [], [], [], [],[]
    iterations_run = []
    image_h, image_w = 0, 0
    print(width, height)
    start = 0
    end = start + args.image_length
    video_frames = process_yuv_video_frame_start_end(
        args.dataset, width, height, start, end)
    frame_length = len(video_frames)
    if args.image_length <= frame_length:
        image_length=args.image_length
    else:
        image_length=frame_length
    Gmodel=None
    img_list = []
    gmodels_state_dict = {}
    num_gaussian_points_dict={}

    output_path_K_frames = Path(f"./checkpoints/{savdir}/{args.data_name}/K_frames.txt")
    if output_path_K_frames.exists():
        with open(output_path_K_frames, "r") as f:
            K_frames = [int(line.strip()) for line in f.readlines()]
    elif image_length==1:
        K_frames = [1]
    else:
        loss_list=[]
        for i in range(start, start+image_length):
            frame_num=i+1
            if frame_num ==1:
                loss_extractor_K = SimpleTrainer2d(image=video_frames[i],frame_num=frame_num,savdir=savdir,loss_type=loss_type, num_points=5000, 
                    iterations=500, model_name=args.model_name, args=args, model_path=None,Trained_Model=None,isdensity=False,isremoval=False,removal_rate=removal_rate)
                Gmodel,_= loss_extractor_K.pre_train()
                loss=0
                loss_list.append(0)
            else:
                loss_extractor_K = SimpleTrainer2d(image=video_frames[i],frame_num=frame_num,savdir=savdir,loss_type=loss_type, num_points=5000, 
                    iterations=500, model_name=args.model_name, args=args, model_path=None,Trained_Model=None,isdensity=False,isremoval=False,removal_rate=removal_rate)
                loss_extractor_P = SimpleTrainer2d(image=video_frames[i],frame_num=frame_num,savdir=savdir,loss_type=loss_type, num_points=5000, 
                    iterations=100, model_name=args.model_name, args=args, model_path=None,Trained_Model=Gmodel,isdensity=False,isremoval=False,removal_rate=removal_rate)  
                Gmodel, loss_K = loss_extractor_K.pre_train()
                _, loss_P = loss_extractor_P.pre_train()
                loss_list.append(loss_P-loss_K)
        loss_list = np.array([
            v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v for v in loss_list
        ])
        values_to_normalize = loss_list[1:]
        min_value = np.min(values_to_normalize)
        max_value = np.max(values_to_normalize)
        # Normalized values in range [0, 1]
        loss_range = max_value - min_value
        normalized_loss_list = [loss_list[0]] + [
            (v - min_value) / loss_range if loss_range > 0 else 0.0
            for v in values_to_normalize
        ]
        output_path = Path(f"./checkpoints/{savdir}/{args.data_name}/{args.model_name}_{args.iterations}_{args.num_points}/loss_list.txt")
        with open(output_path, "w") as f:
            for index, loss in enumerate(normalized_loss_list, start=1):
                f.write(f"Frame {index}: {loss}\n")

        outlier_indices = detect_outliers_mean_diff(normalized_loss_list)
        K_frames = [int(x + 1) for x in outlier_indices]
        K_frames = np.insert(K_frames, 0, 1)
        K_frames = K_frames.astype(int)
        output_path_K_frames = Path(f"./checkpoints/{savdir}/{args.data_name}/K_frames.txt")
        with open(output_path_K_frames, "w") as f:
            for frame in K_frames:
                f.write(f"{frame}\n")
    K_frames = [1]
    print("K-frames:", K_frames)

    Gmodel=None

    for i in range(start, start+image_length):
        frame_num=i+1
        if frame_num in K_frames:
            if look_ahead>0:
                trainer = SimpleTrainer2d(image=video_frames[i],frame_num=frame_num,savdir=savdir,loss_type=loss_type, num_points=args.num_points,max_num_points=args.num_points,
                    iterations=args.iterations, model_name=args.model_name, args=args, model_path=None,Trained_Model=None,isdensity=False,isremoval=is_rm,removal_rate=removal_rate,isschedule=is_schedule,iscover=iscover)
                psnr, ms_ssim, training_time, _, _, Gmodel,_, num_gaussian_points, _ = trainer.train(i)
                end_f = min(frame_length, frame_num + args.look_ahead)
                logwriter.write("Two-pass Frame_{}: {}x{}, PSNR:{:.4f}, MS-SSIM:{:.4f}, Training:{:.4f}s, Gaussians:{}, Iterations:{}/{}".format(frame_num, trainer.H, trainer.W, psnr, ms_ssim, training_time, num_gaussian_points, trainer.iters_ran, args.iterations))
                for fnum in range(frame_num+1, end_f+1):
                    print(f"Frame {fnum} training")
                    trainer = SimpleTrainer2d(image=video_frames[fnum-1],frame_num=fnum,savdir=savdir,loss_type=loss_type, num_points=num_gaussian_points,max_num_points=args.num_points,
                        iterations=args.iterations, model_name=args.model_name, args=args, model_path=None,Trained_Model=Gmodel,isdensity=False,isremoval=False,removal_rate=removal_rate,enhanced=True,iscover=iscover,
                        previous_image=video_frames[fnum-2],is_propagation_frame=True)
                    psnr, ms_ssim, training_time, _, _, Gmodel,_, num_gaussian_points, _ = trainer.train(fnum-1)
                    logwriter.write("Two-pass Frame_{}: {}x{}, PSNR:{:.4f}, MS-SSIM:{:.4f}, Training:{:.4f}s, Gaussians:{}, Iterations:{}/{}".format(fnum, trainer.H, trainer.W, psnr, ms_ssim, training_time, num_gaussian_points, trainer.iters_ran, args.iterations))
                trainer = SimpleTrainer2d(image=video_frames[i],frame_num=frame_num,savdir=savdir,loss_type=loss_type, num_points=num_gaussian_points,max_num_points=args.num_points,
                    iterations=args.iterations, model_name=args.model_name, args=args, model_path=None,Trained_Model=Gmodel,isdensity=False,isremoval=False,removal_rate=removal_rate,enhanced=True,iscover=iscover,
                    previous_image=None,is_propagation_frame=False)
            else:
                trainer = SimpleTrainer2d(image=video_frames[i],frame_num=frame_num,savdir=savdir,loss_type=loss_type, num_points=args.num_points,max_num_points=args.num_points,
                    iterations=args.iterations, model_name=args.model_name, args=args, model_path=None,Trained_Model=None,isdensity=False,isremoval=is_rm,removal_rate=removal_rate,isschedule=is_schedule,iscover=iscover)
        else:
            trainer = SimpleTrainer2d(image=video_frames[i],frame_num=frame_num,savdir=savdir,loss_type=loss_type, num_points=num_gaussian_points,max_num_points=args.num_points,
                iterations=args.iterations, model_name=args.model_name, args=args, model_path=None,Trained_Model=Gmodel,isdensity=is_ad,isremoval=False,removal_rate=removal_rate,iscover=iscover,
                previous_image=video_frames[i-1],is_propagation_frame=True)
        psnr, ms_ssim, training_time, eval_time, eval_fps, Gmodel,img, num_gaussian_points, loss  = trainer.train(i)
        img_list.append(img)
        psnrs.append(psnr)
        ms_ssims.append(ms_ssim)
        training_times.append(training_time) 
        eval_times.append(eval_time)
        eval_fpses.append(eval_fps)
        gaussian_number.append(num_gaussian_points)
        iterations_run.append(trainer.iters_ran)
        image_h += trainer.H
        image_w += trainer.W
        # gmodels_state_dict[f"frame_{frame_num}"] = Gmodel
        gmodels_state_dict[f"frame_{frame_num}"] = make_save_state_from_init_state(Gmodel)
        num_gaussian_points_dict[f"frame_{frame_num}"]=num_gaussian_points
        torch.cuda.empty_cache()
        if i==0 or (i+1)%1==0:
            logwriter.write("Frame_{}: {}x{}, PSNR:{:.4f}, MS-SSIM:{:.4f}, Training:{:.4f}s, Eval:{:.8f}s, FPS:{:.4f}, Loss:{:.4f}, Gaussians:{}, Iterations:{}/{}".format(frame_num, trainer.H, trainer.W, psnr, ms_ssim, training_time, eval_time, eval_fps, loss, num_gaussian_points, trainer.iters_ran, args.iterations))
    torch.save(gmodels_state_dict, gmodel_save_path / "gmodels_state_dict.pth")
    with open(Path(f"./checkpoints/{savdir}/{args.data_name}/{args.model_name}_{args.iterations}_{args.num_points}") / "num_gaussian_points.txt", 'w') as f:
        for key, value in num_gaussian_points_dict.items():
            f.write(f'{key}: {value}\n')
    avg_psnr = torch.tensor(psnrs).mean().item()
    avg_ms_ssim = torch.tensor(ms_ssims).mean().item()
    avg_training_time = torch.tensor(training_times).mean().item()
    avg_eval_time = torch.tensor(eval_times).mean().item()
    avg_eval_fps = torch.tensor(eval_fpses).mean().item()
    avg_h = image_h//image_length
    avg_w = image_w//image_length
    gaussians = sum(gaussian_number) / len(gaussian_number)
    avg_iterations = sum(iterations_run) / len(iterations_run)
    logwriter.write("Average: {}x{}, PSNR:{:.4f}, MS-SSIM:{:.4f}, Training:{:.4f}s, Eval:{:.8f}s, FPS:{:.4f}, Gaussian_number:{:.4f}, Iterations:{:.1f}/{} (total {})".format(
        avg_h, avg_w, avg_psnr, avg_ms_ssim, avg_training_time, avg_eval_time, avg_eval_fps, gaussians, avg_iterations, args.iterations, sum(iterations_run)))
    generate_video(savdir,img_list, args.data_name, args.model_name,args.fps,args.iterations,args.num_points,origin=True)

if __name__ == "__main__":
    
    main(sys.argv[1:])

