# Ablation trainer: P-frame inter positions use raw FP12 without LSQ or EC.
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
from PIL import Image
import torch.nn.functional as F
from pytorch_msssim import ms_ssim
from utils import (
    LogWriter,
    MetricEarlyStopper,
    ModelEMA,
    apply_torch_determinism,
    generate_video_wq,
    log_run_config,
    loss_fn,
    process_yuv_video_frame_start_end,
)
import random
import torchvision.transforms as transforms
from GSVC_encoder import GaussianVideo_frame, GaussianVideo_delta
import csv
import traceback

# Warm-start each P-frame's colour codebook from the previous frame's converged
# one instead of k-means-ing it from a random sample (see
# `quantize.VectorQuantizer.warm_start_`).  Off by default so the two arms of
# the A/B are one commit apart in configuration only.
WARM_START_VQ = os.environ.get("GSVC_WARM_START_VQ", "0") == "1"

# Design the I-frame colour codebook deterministically (LBG binary splitting)
# instead of k-means-ing it from a random sample.  The I-frame is the last RNG
# source warm-start cannot reach: its codebook feeds every P-frame of the GOP
# through p_features_dc, so its init variance propagates down the whole chain.
LBG_INIT = os.environ.get("GSVC_LBG_INIT", "1") == "1"

#add the compressed index
class SimpleTrainer2d:
    """Trains random 2d gaussians to fit an image."""
    def __init__(
        self,
        image,
        frame_num,
        savdir,
        loss_type,
        num_points: int = 2000,
        model_name:str = "GaussianVideo",
        iterations:int = 50000,
        trained_model = None,
        p_trained_model=None,
        args = None,
        isremoval=False,
        removal_rate=0,
        is_first_p_frame=False,
    ):
        self.device = torch.device("cuda:0")
        self.gt_image = image_to_tensor(image).to(self.device)
        self.frame_num=frame_num
        self.num_points = num_points
        if isremoval:
            self.num_points = int(num_points*(1-removal_rate))
        self.model_name=model_name
        self.data_name=args.data_name
        BLOCK_H, BLOCK_W = 16, 16
        self.H, self.W = self.gt_image.shape[2], self.gt_image.shape[3]
        self.iterations = iterations
        self.save_imgs = args.save_imgs
        self.save_everyimgs = args.save_everyimgs
        self.log_dir = Path(f"./checkpoints_quant/{savdir}/{args.data_name}/{args.model_name}_{args.iterations}_{args.num_points}")
        self.loss_type = loss_type
        self.p_trained_model=p_trained_model
        self.is_psnr_loss_log=args.is_log
        self.args = args
        self.is_first_p_frame = is_first_p_frame
        # Codebook state handed down by the previous frame's `test()`.  Rides on
        # `p_trained_model` (== the previous frame's decoded_Gmodel), so no new
        # argument has to be threaded through main()'s frame loop.
        self.p_vq_state = (p_trained_model or {}).get('vq_state') if p_trained_model is not None else None
        self.warm_started = []
        if p_trained_model is not None:
            checkpoint = trained_model
            p_checkpoint = p_trained_model
            cur_len = checkpoint['_xyz'].shape[0]
            self.num_points = cur_len
            add_num_Gaussians = 0
            self.gaussian_model = GaussianVideo_delta(loss_type=self.loss_type, opt_type="adan", num_points=self.num_points,iterations=self.iterations, H=self.H, W=self.W, BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W, 
            device=self.device, lr=args.lr, quantize=True, add_num_Gaussians=add_num_Gaussians, is_first_p_frame=self.is_first_p_frame, is_ci=args.is_ci).to(self.device)
            model_dict = self.gaussian_model.state_dict()
            delta_model_dict = self.gaussian_model.state_dict()
            has_ids = 'point_ids' in checkpoint
            if has_ids:
                self.cur_ids = checkpoint['point_ids'].detach().cpu().numpy()
                prev_ids = p_checkpoint['point_ids'].detach().cpu().numpy()
                prev_index = {int(i): idx for idx, i in enumerate(prev_ids.tolist())}
                cur_len = checkpoint['_xyz'].shape[0]
                device = self.device

                def _count_unmatched(prev_index_map, cur_ids_list):
                    unmatched = 0
                    for pid in cur_ids_list:
                        if prev_index_map.get(int(pid), -1) == -1:
                            unmatched += 1
                    return unmatched

                # ---------- build one shared order ----------
                matched_idx = []
                unmatched_idx = []
                for ci, pid in enumerate(self.cur_ids):
                    if prev_index.get(int(pid), -1) != -1:
                        matched_idx.append(ci)
                    else:
                        unmatched_idx.append(ci)

                if len(matched_idx) == 0 or len(unmatched_idx) == 0:
                    order = None
                else:
                    order = torch.as_tensor(matched_idx + unmatched_idx, device=device, dtype=torch.long)

                def align_param(name):
                    cur = checkpoint[name].to(device)
                    dtype = cur.dtype
                    shape_tail = cur.shape[1:]

                    aligned_prev = torch.zeros((cur_len,) + shape_tail, dtype=dtype, device=device)
                    for ci, pid in enumerate(self.cur_ids):
                        pi = prev_index.get(int(pid), -1)
                        if pi != -1:
                            aligned_prev[ci] = p_checkpoint[name][pi].to(device)

                    if order is not None:
                        cur = cur.index_select(0, order)
                        aligned_prev = aligned_prev.index_select(0, order)

                    return cur, aligned_prev

                cur_xyz, prev_xyz = align_param('_xyz')
                cur_ch, prev_ch = align_param('_cholesky')
                cur_feat, prev_feat = align_param('_features_dc')

                # reorder point_ids with the SAME order
                point_ids_tensor = checkpoint['point_ids'].to(device)
                if order is not None:
                    point_ids_tensor = point_ids_tensor.index_select(0, order)

                # keep self.cur_ids consistent with the reordered parameters
                self.cur_ids = point_ids_tensor.detach().cpu().numpy()

                self.gaussian_model.add_num_Gaussians = _count_unmatched(prev_index, self.cur_ids)
                print(f"add_num_Gaussians: {self.gaussian_model.add_num_Gaussians}")
                delta_model_dict['_xyz'] = cur_xyz - prev_xyz
                delta_model_dict['_cholesky'] = cur_ch - prev_ch
                delta_model_dict['_features_dc'] = cur_feat - prev_feat
                delta_model_dict['p_xyz'] = prev_xyz
                delta_model_dict['p_cholesky'] = prev_ch
                delta_model_dict['p_features_dc'] = prev_feat
                delta_model_dict['point_ids'] = point_ids_tensor

                self.gaussian_model.load_state_dict(delta_model_dict, strict=False)
            else:
                # Fallback: original delta difference (no stable IDs available)
                pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
                p_pretrained_dict = {k: v for k, v in p_checkpoint.items() if k in model_dict}
                # Compute deltas current - prev where available
                for name in ['_xyz','_cholesky','_features_dc']:
                    if (name in pretrained_dict) and (name in p_pretrained_dict):
                        delta_model_dict[name] = pretrained_dict[name] - p_pretrained_dict[name]
                        delta_model_dict['p'+name] = p_pretrained_dict[name]
                self.gaussian_model.load_state_dict(delta_model_dict, strict=False)

            # Warm-start the colour codebooks from the previous frame.  Must run
            # here, after `add_num_Gaussians` has been resolved from the ID
            # match above: it decides whether `features_dc_quantizer_I` is used
            # at all, and warm-starting a quantizer that never runs would hand
            # an untrained codebook to the next frame.
            #
            #   delta codebook (size 7) <- previous P-frame's delta codebook.
            #     Rejected by shape after an I-frame (size 8), which is correct:
            #     the first P of a GOP k-means-inits as before.
            #   abs codebook (size 8)   <- the I-frame's codebook, or the
            #     previous P-frame's `_I`.  Both are fitted on absolute colour
            #     (`align_param` leaves p_features_dc = 0 for unmatched ids),
            #     so they are the same quantity.
            if WARM_START_VQ and self.p_vq_state is not None:
                if self.gaussian_model.features_dc_quantizer.warm_start_(self.p_vq_state.get('delta')):
                    self.warm_started.append('delta')
                if self.gaussian_model.add_num_Gaussians > 0 and \
                        self.gaussian_model.features_dc_quantizer_I.warm_start_(self.p_vq_state.get('abs')):
                    self.warm_started.append('abs')

            # Deterministic fallback for whatever warm-start could not fill.
            # On the FIRST P-frame of a GOP that is the size-7 delta codebook:
            # it has no size-7 predecessor to inherit from, so without this it
            # is the one codebook per GOP still k-means-initialised from a
            # random sample.  With warm-start off entirely it covers every
            # P-frame codebook.  Ordered after warm-start so an inherited
            # codebook always beats a freshly designed one.
            #
            # The input slices come from the model so they cannot drift from
            # what `forward_quantize` actually passes each quantizer.
            self.lbg_seeded = []
            if LBG_INIT:
                d_in, a_in = self.gaussian_model.features_dc_quantizer_inputs()
                if 'delta' not in self.warm_started and \
                        self.gaussian_model.features_dc_quantizer.lbg_init_(d_in):
                    self.lbg_seeded.append('delta')
                if a_in is not None and 'abs' not in self.warm_started and \
                        self.gaussian_model.features_dc_quantizer_I.lbg_init_(a_in):
                    self.lbg_seeded.append('abs')
            print(f"[codebook-init] frame {self.frame_num} (P): "
                  f"warm_start={self.warm_started or 'none'} "
                  f"lbg={self.lbg_seeded or 'none'} "
                  f"(ws={WARM_START_VQ}, lbg={LBG_INIT}, "
                  f"add_num_Gaussians={self.gaussian_model.add_num_Gaussians})",
                  flush=True)
        else:
            checkpoint = trained_model
            self.cur_ids = checkpoint['point_ids'].detach().cpu().numpy()
            cur_len = checkpoint['_xyz'].shape[0]
            self.num_points = cur_len
            self.gaussian_model = GaussianVideo_frame(loss_type=self.loss_type, opt_type="adan", num_points=self.num_points,iterations=self.iterations, H=self.H, W=self.W, BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W, 
            device=self.device, lr=args.lr, quantize=True,
            is_ci=args.is_ci).to(self.device)
            model_dict = self.gaussian_model.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.gaussian_model.load_state_dict(model_dict)

            # Design the codebook on exactly the tensor the forward will
            # quantize -- `GaussianVideo_frame.forward_quantize` passes
            # `get_features`, which for the I-frame class is `_features_dc`
            # itself.  Run here, before the eager warmup, so the first forward
            # finds `initted` already true and the library's k-means never runs.
            self.warm_started = []
            self.lbg_seeded = []
            if LBG_INIT and self.gaussian_model.features_dc_quantizer.lbg_init_(
                    self.gaussian_model.get_features):
                self.lbg_seeded.append('abs')
            print(f"[codebook-init] frame {self.frame_num} (I): "
                  f"warm_start=none lbg={self.lbg_seeded or 'none'} "
                  f"(ws={WARM_START_VQ}, lbg={LBG_INIT}, points={self.num_points})",
                  flush=True)
            

    
    # Keep training eager through iteration 5000 so Gaussian footprints have
    # settled before gsplat's intersection capacity is rebuilt and frozen.
    # The environment variable remains available for controlled experiments.
    _CG_EAGER_WARMUP = int(
        os.environ.get("GSVC_CG_EAGER_WARMUP", "5000")
    )
    # Below this many remaining iterations a capture cannot pay for itself.
    _CG_MIN_SEGMENT = 260

    def _capture_step_graph(self, ema_mgr, ema_decay, iter_now,
                            reset_intersect_capacity=False):
        """Capture one quantization-aware training step as a CUDA graph.

        Returns ``(graph, static_loss, consumed)``.  ``static_loss`` is the
        scalar the captured step writes each replay -- the same quantity the
        eager ``train_iter_quantize`` returns -- and ``consumed`` is the number
        of real training iterations the warmup and the capture itself performed,
        which the caller must add to its counter so the frame still trains the
        requested number of steps.

        The side-stream warmup is the standard CUDA-graph recipe: it binds the
        step's launches off the legacy stream before capture begins.
        """
        model = self.gaussian_model
        gt = self.gt_image
        # Clear before the warmup as well as after it: the eager steps that
        # ran up to here left this iteration's autograd graph attached to
        # the module, and it must not be alive at any point of the capture.
        model.release_graph_refs()
        capacity_warmup = 0
        if reset_intersect_capacity:
            capacity_warmup = model.reset_intersect_capacity_for_capture()
        model.prepare_for_capture()
        model.bind_graph_ema(ema_mgr, ema_decay)
        # gsplat performs a host-side exact-count query during these forwards.
        # Complete all of them before entering CUDA Graph capture; retain the
        # usual three side-stream warmups when the configured count is smaller.
        warmup_steps = max(3, capacity_warmup)
        try:
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(warmup_steps):
                    model.manual_gsplat_train_step_quantize(gt)
                    model.scheduler.step()
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            # The warmup steps leave this iteration's autograd graph attached to
            # the module; capturing with it alive fails on the legacy stream.
            model.release_graph_refs()

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                static_loss = model.manual_gsplat_train_step_quantize(gt)
            torch.cuda.synchronize()
            # CUDA Graph capture statistics output disabled.
            # print(f"[cudagraph-quant] captured at iter {iter_now} "
            #       f"ema_folded={ema_mgr is not None} "
            #       f"lr={self.gaussian_model.optimizer.param_groups[0]['lr']:.3e}",
            #       flush=True)
            # Capture RECORDS the kernels without executing them, so only the
            # warmup advanced training -- the capture itself counts for nothing,
            # and the schedule must not be stepped for it either.  (`static_loss`
            # holds undefined values until the first replay, which is fine: the
            # early-stop probe only reads it on a later iteration.)
            return graph, static_loss, warmup_steps
        except Exception:
            model.end_capture()
            raise

    def train(self):
        # Without EMA, encoder training always consumes the full configured
        # iteration budget.  EMA runs retain their convergence-based early stop.
        self.iters_ran = int(self.iterations)
        self.gaussian_model.train()
        use_ema = bool(getattr(self.args, "use_ema", False))
        ema_decay = float(getattr(self.args, "ema_decay", 0.999))
        ema_warmup = int(getattr(self.args, "ema_warmup", 2500))
        if use_ema:
            use_ema_decay = int(getattr(self.args, "use_ema_decay",0))
        else:
            use_ema_decay = 0
        early_stop_every = int(getattr(self.args, "early_stop_every", 100))
        early_stop_warmup = int(getattr(self.args, "early_stop_warmup", 2500))
        early_stop_patience = int(getattr(self.args, "early_stop_patience", 10))
        early_stop_min_delta = float(getattr(self.args, "early_stop_min_delta", 0.01))
        ema_mgr = None
        early_metric = None
        if use_ema:
            #if self.p_trained_model is  None:
                #early_stop_warmup = 20000
            ema_mgr = ModelEMA(self.gaussian_model, decay=ema_decay)
            ema_mgr.to(self.device)
            early_metric = MetricEarlyStopper(
                patience=early_stop_patience,
                min_delta=early_stop_min_delta,
                mode="min",
            )

        ema_started = False
        ema_decay_used = 0
        best_loss = None
        best_model_dict = None
        best_iter = None
        best_source = None
        enable_ema_lr_decay = use_ema and (self.p_trained_model is None) and (use_ema_decay > 0)

        def _evaluate_candidate(model):
            """Evaluate a checkpoint candidate outside CUDA Graph capture."""
            was_training = model.training
            model.eval()
            try:
                with torch.no_grad():
                    out = model.forward_quantize()
                    out_image = out["render"]
                    loss_v = loss_fn(
                        out_image.squeeze(0).float(),
                        self.gt_image.squeeze(0).float(),
                        self.loss_type,
                        lambda_value=0,
                    ) + out["vq_loss"]
                return float(loss_v.item())
            finally:
                # forward_quantize retains xys/radii.  Dropping those references
                # keeps a later graph capture or re-capture independent of this
                # eager validation forward.
                model.release_graph_refs()
                model.train(was_training)

        def _record_best(loss_value, candidate, it, source):
            nonlocal best_loss, best_model_dict, best_iter, best_source
            if loss_value is None or candidate is None:
                return
            if best_loss is None or loss_value < best_loss:
                best_loss = float(loss_value)
                # Keep the checkpoint outside CUDA graph-private memory and do
                # not retain another full model on the GPU.
                best_model_dict = {
                    key: value.detach().cpu().clone()
                    for key, value in candidate.state_dict().items()
                }
                best_iter = int(it)
                best_source = source

        def _halve_optimizer_lr():
            for param_group in self.gaussian_model.optimizer.param_groups:
                param_group["lr"] *= 0.5
        def _reset_ema_early_stop_state():
            nonlocal early_metric

            if early_metric is not None:
                early_metric = MetricEarlyStopper(
                    patience=early_stop_patience,
                    min_delta=early_stop_min_delta,
                    mode="min",
                )
        def _get_current_lr():
            if len(self.gaussian_model.optimizer.param_groups) == 0:
                return None
            return self.gaussian_model.optimizer.param_groups[0]["lr"]
        

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

        def _maybe_evaluate_candidate(it: int):
            if early_stop_every <= 0:
                return None, False, None, None
            if it % early_stop_every != 0:
                return None, False, None, None
            if use_ema:
                if not ema_started or it < early_stop_warmup:
                    return None, False, None, None
                candidate = ema_mgr.ema
                source = "ema"
            else:
                candidate = self.gaussian_model
                source = "model"
            loss_value = _evaluate_candidate(candidate)
            should_stop = (
                early_metric.step(loss_value)
                if early_metric is not None else False
            )
            return loss_value, should_stop, candidate, source

        if self.is_psnr_loss_log:
            psnr_list=[]
            loss_list=[]
            iter_list=[]
            #progress_bar = tqdm(range(1, int(self.iterations)+1), desc="Training progress")
            torch.cuda.synchronize()
            start_time = time.time()
            for iter in range(1, int(self.iterations)+1):
                loss, psnr = self.gaussian_model.train_iter_quantize_log(self.gt_image)
                #loss = self.gaussian_model.train_iter_quantize(self.gt_image)
                _maybe_update_ema(iter)
                candidate_loss, should_stop, candidate, source = _maybe_evaluate_candidate(iter)
                _record_best(candidate_loss, candidate, iter, source)
                if should_stop:
                    if enable_ema_lr_decay and (ema_decay_used < use_ema_decay):
                        _halve_optimizer_lr()
                        ema_decay_used += 1
                        early_stop_patience=early_stop_patience*2
                        early_stop_min_delta=early_stop_min_delta/2
                        _reset_ema_early_stop_state()
                    else:
                        print(f"Early stopping EMA training at iteration {iter}")
                        break
                if iter % 100 == 0:
                    if (ema_mgr is not None) and ema_started:
                        with torch.no_grad():
                            ema_mgr.ema.eval()
                            out = ema_mgr.ema.forward_quantize()
                            out_image = out["render"]
                            loss_ema = loss_fn(out_image.float(), self.gt_image.float(), self.loss_type, lambda_value=0)
                            mse_loss = F.mse_loss(out_image.float(), self.gt_image.float())
                            psnr_ema = 10 * math.log10(1.0 / mse_loss.item())
                        #print(psnr_ema)
                        psnr_list.append(float(psnr_ema))
                        loss_list.append(float(loss_ema.item()))
                    else:
                        psnr_list.append(psnr)
                        loss_list.append(loss.item())
                    iter_list.append(iter)
            self.iters_ran = iter
            torch.cuda.synchronize()
            end_time = time.time() - start_time
            os.makedirs(self.log_dir, exist_ok=True)
            csv_path = os.path.join(self.log_dir, f"{self.frame_num}_training_log.csv")
            with open(csv_path, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["iter", "psnr", "loss"])
                for i, p, l in zip(iter_list, psnr_list, loss_list):
                    writer.writerow([i, p, l])
        else:
            torch.cuda.synchronize()
            start_time = time.time()
            # CUDA-graph training (GSVC_CUDAGRAPH_TRAIN=1).  Stage 2 keeps
            # the point count fixed, but runs eagerly through iteration 5000 so
            # the intersection capacity is sampled from settled footprints.
            # The graph is
            # dropped and re-captured whenever the EMA shadow is seeded or the
            # learning rate changes; see _capture_step_graph.
            use_cudagraph = os.environ.get("GSVC_CUDAGRAPH_TRAIN", "1") == "1"
            if use_cudagraph:
                self.gaussian_model.use_dynamic_intersect_capacity()
            graph = static_loss = None
            captured_lr = None
            total_iters = int(self.iterations)
            replayed = 0
            iter = 0
            intersect_capacity_rebuilt = False
            while iter < total_iters:
                if (use_cudagraph and graph is None
                        and iter >= self._CG_EAGER_WARMUP
                        and (total_iters - iter) > self._CG_MIN_SEGMENT):
                    try:
                        graph, static_loss, consumed = self._capture_step_graph(
                            ema_mgr if ema_started else None, ema_decay, iter,
                            reset_intersect_capacity=not intersect_capacity_rebuilt)
                        intersect_capacity_rebuilt = True
                        captured_lr = _get_current_lr()
                        iter += consumed
                        replayed += consumed
                        if iter >= total_iters:
                            break
                    except Exception:
                        # Do NOT fall back to eager training here.  A failed
                        # capture leaves the CUDA context (and the RNG
                        # generator's capture state) broken for the rest of the
                        # process: a later k-means then dies with "Offset
                        # increment outside graph capture encountered
                        # unexpectedly" in code that has nothing to do with
                        # graphs.  Continuing would silently produce a corrupt
                        # run, so fail loudly with the real traceback instead.
                        self.gaussian_model.end_capture()
                        print(f"[cudagraph-quant] capture FAILED at iter {iter}; "
                              f"the CUDA context is no longer trustworthy, aborting. "
                              f"Re-run without GSVC_CUDAGRAPH_TRAIN=1 to train "
                              f"eagerly.", flush=True)
                        traceback.print_exc()
                        raise

                iter += 1
                if graph is not None:
                    graph.replay()
                    loss = static_loss
                    replayed += 1
                    # The captured optimizer kernels bake in the learning rate,
                    # so the schedule has to be advanced on the host and the
                    # graph thrown away the moment it actually moves.
                    self.gaussian_model.scheduler.step()
                    if _get_current_lr() != captured_lr:
                        graph = static_loss = None
                    elif use_ema and not ema_started:
                        # This segment runs before ModelEMA.reset() has seeded
                        # the shadow, so nothing is folded into it yet.  Keep
                        # calling the eager hook (it early-returns) and drop the
                        # graph on the iteration that seeds, so the next capture
                        # can fold the update in.
                        _maybe_update_ema(iter)
                        if ema_started:
                            graph = static_loss = None
                else:
                    loss = self.gaussian_model.train_iter_quantize(self.gt_image)
                    _maybe_update_ema(iter)
                candidate_loss, should_stop, candidate, source = _maybe_evaluate_candidate(iter)
                _record_best(candidate_loss, candidate, iter, source)
                if should_stop:
                    if enable_ema_lr_decay and (ema_decay_used < use_ema_decay):
                        old_lr = _get_current_lr()
                        _halve_optimizer_lr()
                        ema_decay_used += 1
                        new_lr = _get_current_lr()
                        print(
                            f"[I-frame][EMA converge] Iter {iter}: "
                            f"lr decay {ema_decay_used}/{use_ema_decay}, "
                            f"lr {old_lr} -> {new_lr}. Reset early-stop and continue training."
                        )
                        early_stop_patience=early_stop_patience*2
                        early_stop_min_delta=early_stop_min_delta/2
                        _reset_ema_early_stop_state()
                        # _halve_optimizer_lr just invalidated the captured
                        # optimizer kernels' baked-in learning rate.
                        graph = static_loss = None
                        continue
                    else:
                        print(f"Early stopping EMA training at iteration {iter}", flush=True)
                        break

            self.iters_ran = iter
            torch.cuda.synchronize()
            end_time = time.time() - start_time
            if use_cudagraph:
                self.gaussian_model.end_capture()
                graph = static_loss = None
                # CUDA Graph timing/statistics output disabled.
                # print(f"[cudagraph-quant] frame={self.frame_num} ran_iters={iter} "
                #       f"replayed={replayed} wall={end_time:.3f} s "
                #       f"per_iter={end_time / max(1, iter) * 1e3:.3f} ms", flush=True)

        # Always consider the final state, including runs shorter than the
        # periodic evaluation interval.  CUDA graph patches have been undone by
        # this point, so this is an ordinary eager validation forward.
        final_candidate = (
            ema_mgr.ema
            if (ema_mgr is not None and ema_started)
            else self.gaussian_model
        )
        final_source = "ema" if final_candidate is not self.gaussian_model else "model"
        final_loss = _evaluate_candidate(final_candidate)
        _record_best(final_loss, final_candidate, self.iters_ran, final_source)
        if best_model_dict is not None:
            self.gaussian_model.load_state_dict(best_model_dict)
            print(
                f"[best-model] frame={self.frame_num} source={best_source} "
                f"iter={best_iter} loss={best_loss:.8f}",
                flush=True,
            )
        psnr_value, ms_ssim_value, bpp, img, decoded_Gmodel,encoded_Gmodel = self.test()
        with torch.no_grad():
            self.gaussian_model.eval()
            torch.cuda.synchronize()
            test_start_time = time.time()
            for i in range(100):
                _ = self.gaussian_model.forward_quantize()
            torch.cuda.synchronize()
            test_end_time = (time.time() - test_start_time) / 100
        self.num_gaussian_points = int(self.gaussian_model._xyz.size(0))
        return psnr_value, ms_ssim_value, end_time, test_end_time, 1/test_end_time, bpp, decoded_Gmodel, img, encoded_Gmodel
        
    
    def test(self):
        self.gaussian_model.eval()
        with torch.no_grad():
            out = self.gaussian_model.forward_quantize_bpp()
            out_image = out["render"]
            mse_loss = F.mse_loss(out_image.float(), self.gt_image.float())
            psnr = 10 * math.log10(1.0 / mse_loss.item())
            ms_ssim_value = ms_ssim(out_image.float(), self.gt_image.float(), data_range=1, size_average=True).item()
            m_bit, s_bit, r_bit, c_bit = out["unit_bit"]
            # compress_index_pkg = self.gaussian_model.compress_index_only()
            # pid_bit = compress_index_pkg["pid_bit"]
            # print(f"m_bit:{m_bit/self.H/self.W}, s_bit:{s_bit/self.H/self.W}, r_bit:{r_bit/self.H/self.W}, c_bit:{c_bit/self.H/self.W}, pid_bit:{pid_bit/self.H/self.W}")
            # bpp = (m_bit + s_bit + r_bit + c_bit + pid_bit)/self.H/self.W
            bpp = (m_bit + s_bit + r_bit + c_bit)/self.H/self.W
            transform = transforms.ToPILImage()
            img = transform(out_image.float().squeeze(0))
            if self.save_everyimgs:
                save_path_img = self.log_dir / "img"
                save_path_img.mkdir(parents=True, exist_ok=True)
                name =str(self.frame_num) + "_fitting.png" 
                img.save(str(save_path_img / name))

            if self.p_trained_model is not None:
                if self.gaussian_model.add_num_Gaussians>0:
                    q_xyz_front = self.gaussian_model.xyz_quantizer(
                        self.gaussian_model._xyz[:self.gaussian_model.num_delta_Gaussians])
                    q_xyz_back = self.gaussian_model.xyz_quantizer_I(self.gaussian_model._xyz[self.gaussian_model.num_delta_Gaussians:])
                    q_xyz = torch.cat([q_xyz_front, q_xyz_back], dim=0) + self.gaussian_model.p_xyz
                    
                    q_ch_front, _, _ = self.gaussian_model.cholesky_quantizer(self.gaussian_model._cholesky[:self.gaussian_model.num_delta_Gaussians])
                    q_ch_back, _, _ = self.gaussian_model.cholesky_quantizer_I(self.gaussian_model._cholesky[self.gaussian_model.num_delta_Gaussians:])
                    q_ch = torch.cat([q_ch_front, q_ch_back], dim=0) + self.gaussian_model.p_cholesky
                    
                    q_feat_front, _, _ = self.gaussian_model.features_dc_quantizer(self.gaussian_model._features_dc[:self.gaussian_model.num_delta_Gaussians])
                    q_feat_back, _, _ = self.gaussian_model.features_dc_quantizer_I(self.gaussian_model._features_dc[self.gaussian_model.num_delta_Gaussians:])
                    q_feat = torch.cat([q_feat_front, q_feat_back], dim=0) + self.gaussian_model.p_features_dc

                else:
                    q_xyz = self.gaussian_model.xyz_quantizer(
                        self.gaussian_model._xyz)
                    q_xyz = q_xyz + self.gaussian_model.p_xyz
                    q_ch, _, _ = self.gaussian_model.cholesky_quantizer(self.gaussian_model._cholesky)
                    q_ch = q_ch + self.gaussian_model.p_cholesky
                    q_feat, _, _ = self.gaussian_model.features_dc_quantizer(self.gaussian_model._features_dc)
                    q_feat = q_feat + self.gaussian_model.p_features_dc
                current_point_ids = self.gaussian_model.point_ids.detach().clone()
                decoded_Gmodel = {
                    '_xyz': q_xyz.detach().clone(),
                    '_cholesky': q_ch.detach().clone(),
                    '_features_dc': q_feat.detach().clone(),
                    'point_ids': current_point_ids,
                }
            else:
                q_xyz = self.gaussian_model.xyz_quantizer(self.gaussian_model._xyz)
                q_ch, _, _ = self.gaussian_model.cholesky_quantizer(self.gaussian_model._cholesky)
                q_feat, _, _ = self.gaussian_model.features_dc_quantizer(self.gaussian_model._features_dc)
                current_point_ids = self.gaussian_model.point_ids.detach().clone()
                decoded_Gmodel = {
                    '_xyz': q_xyz.detach().clone(),
                    '_cholesky': q_ch.detach().clone(),
                    '_features_dc': q_feat.detach().clone(),
                    'point_ids': current_point_ids,
                }

            # Hand this frame's converged codebooks to the next one.  Taken
            # after the EMA/best-model restore above, so it is the codebook the
            # frame actually encoded with -- and after model.eval(), so the
            # export cannot itself perturb the EMA state.
            vq_state = {'delta': None, 'abs': None}
            if self.p_trained_model is None:
                # I-frame: its one codebook is fitted on absolute colour.
                vq_state['abs'] = self.gaussian_model.features_dc_quantizer.export_state()
            else:
                vq_state['delta'] = self.gaussian_model.features_dc_quantizer.export_state()
                if self.gaussian_model.add_num_Gaussians > 0:
                    vq_state['abs'] = self.gaussian_model.features_dc_quantizer_I.export_state()
                else:
                    # `_I` never ran this frame, so its codebook is untrained.
                    # Pass the inherited one along instead of breaking the chain.
                    vq_state['abs'] = (self.p_vq_state or {}).get('abs')
            decoded_Gmodel['vq_state'] = vq_state

            encoded_Gmodel = self.gaussian_model.compress_wo_ec()
            
            # attach quantizer parameters
            cholesky_q = self.gaussian_model.cholesky_quantizer
            encoded_Gmodel['cholesky_scale'] = cholesky_q.scale.detach().clone().cpu()
            encoded_Gmodel['cholesky_beta'] = cholesky_q.beta.detach().clone().cpu()
            fvq = self.gaussian_model.features_dc_quantizer
            if fvq.num_quantizers == 1:
                codebooks = [fvq.quantizer._codebook.embed.detach().clone().cpu()]
            else:
                codebooks = [layer._codebook.embed.detach().clone().cpu() for layer in fvq.quantizer.layers]
            encoded_Gmodel['features_codebooks'] = codebooks
            # attach current frame point ids if available
            encoded_Gmodel['point_ids'] = self.gaussian_model.point_ids.detach().clone().cpu()
            if self.p_trained_model is not None:
                # FP12 has no learned scale/beta metadata.
                encoded_Gmodel['xyz_format'] = 'fp12_e5m6_raw'
                cholesky_q_I = self.gaussian_model.cholesky_quantizer_I
                encoded_Gmodel['cholesky_scale_I'] = cholesky_q_I.scale.detach().clone().cpu()
                encoded_Gmodel['cholesky_beta_I'] = cholesky_q_I.beta.detach().clone().cpu()
                fvq_I = self.gaussian_model.features_dc_quantizer_I
                if fvq_I.num_quantizers == 1:
                    codebooks_I = [fvq_I.quantizer._codebook.embed.detach().clone().cpu()]
                else:
                    codebooks_I = [layer._codebook.embed.detach().clone().cpu() for layer in fvq_I.quantizer.layers]
                encoded_Gmodel['features_codebooks_I'] = codebooks_I
        return psnr, ms_ssim_value, bpp, img, decoded_Gmodel, encoded_Gmodel

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
        "--model_path", type=str, default=None, help="Path to a checkpoint"
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
        "--sh_degree", type=int, default=3, help="SH degree (default: %(default)s)"
    )
    parser.add_argument(
        "--num_points", type=int, default=4000, help="2D GS points (default: %(default)s)",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate (default: %(default)s)"
    )
    parser.add_argument("--loss_type", type=str, default="L2", help="Type of Loss")
    parser.add_argument("--seed", type=int, default=1, help="Set random seed for reproducibility")
    parser.add_argument("--save_imgs", action="store_true", help="Save image")
    parser.add_argument("--save_everyimgs", action="store_true", help="Save Every Images")
    parser.add_argument("--removal_rate", type=float, default=0.1, help="Removal rate")
    parser.add_argument("--is_rm", action="store_true", help="Removal control of gaussians setup")
    parser.add_argument(
        "--is_ci", action="store_true",
        help="Initialize Cholesky LSQ+ with scale=1 and beta=0"
    )
    parser.add_argument("--I_interval", type=int, default=5, help="I_frame_interval (default: %(default)s)")
    parser.add_argument("--is_log", action="store_true", help="Save PSNR and loss log")
    parser.add_argument("--use_ema", action="store_true", help="Use EMA candidates for best-model selection and enable EMA early stopping")
    parser.add_argument("--use_ema_decay", type=int,default=0, help="Lr decay until ema model converge, the value is the decay times.")
    parser.add_argument("--ema_decay", type=float, default=0.99, help="EMA decay for model weights")
    parser.add_argument("--ema_warmup", type=int, default=2500, help="Do not maintain EMA before this iteration")
    parser.add_argument("--early_stop_every", type=int, default=100, help="Evaluate candidate loss every N iterations for best-model selection and EMA early stopping")
    parser.add_argument("--early_stop_warmup", type=int, default=2500, help="Do not evaluate EMA plateau before this iteration")
    parser.add_argument("--early_stop_patience", type=int, default=10, help="Stop EMA training after N evaluations without improvement; ignored without --use_ema")
    parser.add_argument("--early_stop_min_delta", type=float, default=0.01, help="Minimum loss improvement to be considered better")
    args = parser.parse_args(argv)
    # Kept for log_run_config(); underscore keeps it out of the argument dump.
    args._parser = parser
    return args

def main(argv):
    args = parse_args(argv)
    args.save_imgs=True
    loss_type=args.loss_type
    savdir=args.savdir
    savdir_m=args.savdir_m
    width = args.width
    height = args.height
    is_rm=args.is_rm
    removal_rate=args.removal_rate
    gmodel_save_path = Path(f"./checkpoints_quant/{savdir_m}/{args.data_name}/{args.model_name}_{args.iterations}_{args.num_points}")
    gmodel_save_path.mkdir(parents=True, exist_ok=True)
    img_list = []
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

    logwriter = LogWriter(Path(f"./checkpoints_quant/{savdir}/{args.data_name}/{args.model_name}_{args.iterations}_{args.num_points}"))
    log_run_config(logwriter, args, argv=argv, parser=getattr(args, "_parser", None),
                   stage="stage 2: encoding (train_GSVC_encoder.py)")
    psnrs, ms_ssims, training_times, eval_times, eval_fpses, bpps = [], [], [], [], [], []
    gaussian_number, iterations_run = [], []
    image_h, image_w = 0, 0
    image_length=args.image_length
    start=0
    video_frames = process_yuv_video_frame_start_end(args.dataset, width, height,start,image_length)
    Gmodel=None
    P_Gmodel=None
    Overfit_gmodels_state_dict = torch.load(args.model_path,map_location=torch.device("cuda:0"))
    encoded_gmodels_state_dict = {}
    output_path_K_frames = Path(f"./checkpoints/{savdir}/{args.data_name}/K_frames.txt")
    if output_path_K_frames.exists():
        with open(output_path_K_frames, "r") as f:
            K_frames = [int(line.strip()) for line in f.readlines()]
    else:
        K_frames = [1]
    I_interval = args.I_interval
    periodic_k = list(range(1, image_length + 1, I_interval))
    K_frames = sorted(set([k for k in K_frames if 1 <= k <= image_length] + periodic_k))
    print(f"K_frames: {K_frames}")
    for i in range(start, start+image_length):
        frame_num=i+1
        modelid=f"frame_{i + 1}"
        Model = Overfit_gmodels_state_dict[modelid]
        if frame_num in K_frames:
            print(f"modelid:frame_{i + 1};")
            trainer = SimpleTrainer2d(image=video_frames[i],frame_num=frame_num,savdir=savdir,loss_type=loss_type, num_points=args.num_points,
                iterations=args.iterations, model_name=args.model_name, args=args, trained_model=Model,isremoval=is_rm,removal_rate=removal_rate)
        else:
            p_modelid = f"frame_{i}"
            print(f"modelid:{modelid}; p_modelid:{p_modelid}")
            if frame_num-1 in K_frames:
                is_first_p_frame = True
            else:
                is_first_p_frame = False
            trainer = SimpleTrainer2d(image=video_frames[i],frame_num=frame_num,savdir=savdir,loss_type=loss_type, num_points=args.num_points,
                iterations=args.iterations, model_name=args.model_name, args=args, p_trained_model =P_Gmodel, trained_model=Model,isremoval=is_rm,removal_rate=removal_rate,is_first_p_frame=is_first_p_frame)
      
        psnr, ms_ssim, training_time, eval_time, eval_fps, bpp, Gmodel, img, encoded_Gmodel = trainer.train()     
        P_Gmodel=Gmodel
        img_list.append(img)
        psnrs.append(psnr)
        ms_ssims.append(ms_ssim)
        training_times.append(training_time) 
        eval_times.append(eval_time)
        eval_fpses.append(eval_fps)
        bpps.append(bpp)
        gaussian_number.append(trainer.num_gaussian_points)
        iterations_run.append(trainer.iters_ran)
        image_h += trainer.H
        image_w += trainer.W
        encoded_Gmodel['is_k_frame'] = bool(frame_num in K_frames)
        encoded_gmodels_state_dict[f"frame_{frame_num}"] = encoded_Gmodel
        torch.cuda.empty_cache()
        if i==0 or (i+1)%1==0:
            logwriter.write("Frame_{}: {}x{}, PSNR:{:.4f}, MS-SSIM:{:.4f}, bpp:{:.4f}, Training:{:.4f}s, Eval:{:.8f}s, FPS:{:.4f}, Gaussians:{}, Iterations:{}/{}".format(
            frame_num, trainer.H, trainer.W, psnr, ms_ssim, bpp, training_time, eval_time, eval_fps, trainer.num_gaussian_points, trainer.iters_ran, args.iterations))
    torch.save(encoded_gmodels_state_dict, gmodel_save_path / "encoded_gmodels_state_dict.pth")
    avg_psnr = torch.tensor(psnrs).mean().item()
    avg_ms_ssim = torch.tensor(ms_ssims).mean().item()
    avg_training_time = torch.tensor(training_times).mean().item()
    avg_eval_time = torch.tensor(eval_times).mean().item()
    avg_eval_fps = torch.tensor(eval_fpses).mean().item()
    avg_bpp = torch.tensor(bpps).mean().item()
    avg_h = image_h//image_length
    avg_w = image_w//image_length
    avg_gaussians = sum(gaussian_number) / len(gaussian_number)
    avg_iterations = sum(iterations_run) / len(iterations_run)
    logwriter.write("Average: {}x{}, PSNR:{:.4f}, MS-SSIM:{:.4f}, Bpp:{:.4f}, Training:{:.4f}s, Eval:{:.8f}s, FPS:{:.4f}, Gaussian_number:{:.4f}, Iterations:{:.1f}/{} (total {})".format(
        avg_h, avg_w, avg_psnr, avg_ms_ssim, avg_bpp, avg_training_time, avg_eval_time, avg_eval_fps, avg_gaussians, avg_iterations, args.iterations, sum(iterations_run)))    
    generate_video_wq(savdir,img_list, args.data_name, args.model_name,args.fps,args.iterations,args.num_points,origin=True)
if __name__ == "__main__":
    
    main(sys.argv[1:])


