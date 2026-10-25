import os
import sys
import torch.nn.functional as F
from pytorch_msssim import ms_ssim, ssim
import torch
import torch.nn as nn
import numpy as np
import cv2
from tqdm import tqdm
from pathlib import Path
import math
import time
import copy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

class LogWriter:
    def __init__(self, file_path, train=True):
        os.makedirs(file_path, exist_ok=True)
        self.file_path = os.path.join(file_path, "train.txt" if train else "test.txt")

    def write(self, text):
        print(text)
        with open(self.file_path, 'a') as file:
            file.write(text + '\n')

    def write_quiet(self, text):
        """Append to the log file without echoing to stdout.

        Used for the run-configuration header, which is long and already
        visible in the log file itself.
        """
        with open(self.file_path, 'a') as file:
            file.write(text + '\n')


# Environment variables that change what the pipeline actually computes, with
# the fallback each call site uses when the variable is unset.  Anything read
# via os.environ.get() in this repo belongs here so that a train.txt records
# the effective value even when the variable was never exported.
from gsvc_runtime import RUNTIME_DEFAULTS
TRACKED_ENV_VARS = RUNTIME_DEFAULTS

# Environment variables that do not change the math but do change which
# hardware/run produced the numbers; recorded when set.
ENV_PREFIXES_LOGGED = ("CUDA_", "TORCH_", "PYTORCH_", "NVIDIA_", "SLURM_",
                       "OMP_", "MKL_", "PYTHONHASHSEED")


def apply_torch_determinism(logwriter=None):
    """Make PyTorch's own operators reproducible, per GSVC_DETERMINISTIC_TORCH.

    `GSVC_DET_ATOMICS` fixes the rasterizer backward, but that is a custom
    CUDA extension and covers only gsplat.  Two identical stage-2 runs still
    differed by 0.10 dB with it on (RESEARCH_LOG, 2026-09-03), so something in
    PyTorch or cuBLAS is also varying between runs.  This is the switch for that
    half.

    Values:
      "0"     leave PyTorch alone.
      "1"     (default) `use_deterministic_algorithms(True)`: pick deterministic kernels
              where they exist, raise a RuntimeError naming the operator where
              they do not.
      "warn"  the same, but warn instead of raising, so one diagnostic run lists
              every offending operator rather than stopping at the first.

    cuBLAS needs the companion environment variable.  A matrix multiplication can
    split the sum over the shared dimension across thread blocks and add the
    partial results together; which split it picks depends on the workspace
    memory free at the time, so the addition order varies between runs.
    CUBLAS_WORKSPACE_CONFIG pins the workspace and therefore the split.  It has
    to be set before the first cuBLAS handle is created.  The training entry
    points install the default before importing torch; this check also rejects
    callers that import these helpers without configuring the workspace first.

    `cudnn.deterministic` and `cudnn.benchmark` are NOT set here; each trainer
    already sets them in its own `--seed` block.
    """
    mode = os.environ.get("GSVC_DETERMINISTIC_TORCH", "0")
    if mode not in ("0", "1", "warn"):
        raise ValueError(
            f"GSVC_DETERMINISTIC_TORCH must be '0', '1' or 'warn', got {mode!r}"
        )
    if mode == "0":
        return False

    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if workspace not in (":4096:8", ":16:8"):
        print("[GSVC runtime] deterministic torch disabled: unsupported "
              f"CUBLAS_WORKSPACE_CONFIG={workspace!r}", flush=True)
        return False

    torch.use_deterministic_algorithms(True, warn_only=(mode == "warn"))
    message = (f"torch determinism: use_deterministic_algorithms(True, "
               f"warn_only={mode == 'warn'}), CUBLAS_WORKSPACE_CONFIG={workspace}")
    if logwriter is not None:
        logwriter.write(message)
    else:
        print(message, flush=True)
    return True


# ---- deterministic Gaussian initialization (Sobol) -------------------------
#
# Stage 1 used to initialize every Gaussian -- position, Cholesky shape, and on
# the non-coverage path colour -- with `torch.rand`.  That had two costs.  The
# values were arbitrary, and because the three initialization sites shared one
# global generator, the values a Gaussian received depended on how many draws
# had happened before it.  Two concrete cases: a P-frame constructs a full
# random model and immediately overwrites it from the previous frame, so the
# discarded draws still shifted every later frame; and the K-frame detection
# pass runs only when `K_frames.txt` is absent, so the same command run twice in
# the same output directory did not initialize the second run the way it did
# the first.
#
# A Sobol sequence replaces the generator.  It is a fixed list of points in the
# unit cube built from a published table of direction numbers, and with
# `scramble=False` torch's SobolEngine draws no random numbers at all -- the
# `_scramble` branch, the only one that needs a generator, is never taken.
# Three properties matter here:
#
#   * It extends.  Elements [N, N+M) continue elements [0, N), which is what
#     `grow_points_to` and `add_Gaussians` need when they append Gaussians to a
#     set that is already training.  A grid does not have this property.
#   * It is indexed by point id, so a Gaussian's initialization depends only on
#     its own identity and not on what ran before it.
#   * It is uniform on the unit cube, which is the distribution `torch.rand`
#     sampled, so every range mapping below is unchanged.  Only the clumping of
#     independent draws goes away.
#
# Element 0 of an unscrambled sequence is all zeros, which would map to a
# position of -1 and then to atanh(-1) = -inf, so the numbering is shifted by
# one.  The shift also keeps `SobolEngine.fast_forward` off its `n - 1` branch,
# which would otherwise be passed -1 for a start id of 0.

SOBOL_DIMS = 8           # 0-1 position, 2-4 cholesky, 5-7 features_dc
SOBOL_ATANH_EPS = 1e-6   # keeps positions off +/-1, where atanh diverges


def sobol_block(start_id, count, device=None, dtype=None):
    """Sobol elements ``[start_id, start_id + count)`` as ``(count, SOBOL_DIMS)`` in [0, 1)."""
    from torch.quasirandom import SobolEngine
    count = int(count)
    if dtype is None:
        dtype = torch.float32
    if count <= 0:
        return torch.zeros((0, SOBOL_DIMS), device=device, dtype=dtype)
    engine = SobolEngine(dimension=SOBOL_DIMS, scramble=False)
    engine.fast_forward(int(start_id) + 1)
    u = engine.draw(count, dtype=dtype)
    return u if device is None else u.to(device)


def sobol_raw_xyz(u):
    """Tanh-space positions (the raw ``_xyz`` parameter) from Sobol columns 0-1."""
    xy = (2.0 * u[:, 0:2] - 1.0).clamp(-1.0 + SOBOL_ATANH_EPS, 1.0 - SOBOL_ATANH_EPS)
    return torch.atanh(xy)


def sobol_cholesky(u, iscover):
    """Cholesky (xx, xy, yy) init from Sobol columns 2-4.

    The ranges are the ones the random version drew from: [0, 1) for all three
    without coverage initialization, and the off-diagonal in [0, 1) with the two
    diagonal entries in [0.5, 1) with it.
    """
    if not iscover:
        return u[:, 2:5].clone()
    chol = torch.empty((u.shape[0], 3), device=u.device, dtype=u.dtype)
    chol[:, 1] = u[:, 2]
    chol[:, 0] = 0.5 + 0.5 * u[:, 3]
    chol[:, 2] = 0.5 + 0.5 * u[:, 4]
    return chol


def _git_state():
    """Return '<short-sha> (dirty)' for the repo holding this file, or 'unknown'."""
    import subprocess
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        sha = subprocess.check_output(
            ["git", "-C", repo_dir, "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
        dirty = subprocess.check_output(
            ["git", "-C", repo_dir, "status", "--porcelain"],
            stderr=subprocess.DEVNULL).decode().strip()
        return sha + (" (dirty)" if dirty else " (clean)")
    except Exception:
        return "unknown"


def log_run_config(logwriter, args, argv=None, parser=None, stage=None, extra=None):
    """Dump the complete configuration of this run at the top of the log.

    Writes, in order: run identity (time, host, git commit, command line),
    software/hardware versions, every parsed argument with whether it came from
    the command line or from its default, and the environment variables that
    affect the run.  The goal is that a train.txt is enough on its own to
    reproduce the run it describes.
    """
    import getpass
    import platform
    import socket
    from datetime import datetime

    argv = list(sys.argv[1:]) if argv is None else list(argv)
    w = logwriter.write

    w("")
    w("=" * 78)
    w("RUN CONFIGURATION" + (f" - {stage}" if stage else ""))
    w("=" * 78)
    w(f"# timestamp     : {datetime.now().isoformat(timespec='seconds')}")
    w(f"# host          : {socket.gethostname()}")
    try:
        w(f"# user          : {getpass.getuser()}")
    except Exception:
        pass
    w(f"# cwd           : {os.getcwd()}")
    w(f"# git commit    : {_git_state()}")
    w(f"# script        : {os.path.abspath(sys.argv[0])}")
    w(f"# command       : {' '.join([sys.executable, sys.argv[0]] + argv)}")

    w("")
    w("--- versions ---")
    w(f"python          : {platform.python_version()} ({platform.platform()})")
    try:
        w(f"torch           : {torch.__version__} (cuda {torch.version.cuda})")
        if torch.cuda.is_available():
            w(f"gpu             : {torch.cuda.get_device_name(0)} "
              f"x{torch.cuda.device_count()}")
            w(f"cudnn           : {torch.backends.cudnn.version()}")
        else:
            w("gpu             : <no cuda device visible>")
    except Exception as e:
        w(f"torch           : <unavailable: {e}>")
    try:
        import gsplat
        w(f"gsplat          : {getattr(gsplat, '__version__', 'unknown')} "
          f"({os.path.dirname(gsplat.__file__)})")
    except Exception as e:
        w(f"gsplat          : <unavailable: {e}>")
    w(f"numpy           : {np.__version__}")

    # Which options were actually typed on the command line, so a reader can
    # tell a deliberate setting from an inherited default.
    cli_dests = set()
    if parser is not None:
        try:
            opt_to_dest = {}
            for action in parser._actions:
                for opt in action.option_strings:
                    opt_to_dest[opt] = action.dest
            for token in argv:
                if not token.startswith("-"):
                    continue
                name = token.split("=", 1)[0]
                if name in opt_to_dest:
                    cli_dests.add(opt_to_dest[name])
        except Exception:
            cli_dests = set()

    w("")
    w("--- arguments (value  [source]) ---")
    arg_items = {k: v for k, v in sorted(vars(args).items()) if not k.startswith("_")}
    width = max([len(k) for k in arg_items] or [1])
    for k, v in arg_items.items():
        if parser is None:
            source = ""
        elif k in cli_dests:
            source = "  [cli]"
        elif parser.get_default(k) == v:
            source = "  [default]"
        else:
            source = "  [set in code]"
        w(f"{k.ljust(width)} : {v!r}{source}")

    w("")
    w("--- environment variables ---")
    for name, fallback in sorted(TRACKED_ENV_VARS.items()):
        raw = os.environ.get(name)
        if raw is None:
            w(f"{name} : {fallback!r}  [unset, default]")
        else:
            w(f"{name} : {raw!r}  [env]")
    for name in sorted(os.environ):
        if name in TRACKED_ENV_VARS:
            continue
        if name.startswith(ENV_PREFIXES_LOGGED):
            w(f"{name} : {os.environ[name]!r}  [env]")

    if extra:
        w("")
        w("--- run context ---")
        for k, v in extra.items():
            w(f"{k} : {v}")

    w("=" * 78)
    w("")
    print(f"[config] run configuration written to {logwriter.file_path}")


def loss_fn(pred, target, loss_type='L2', lambda_value=0.7):
    target = target.detach()
    pred = pred.float()
    target  = target.float()
    if loss_type == 'L2':
        loss = F.mse_loss(pred, target)
    elif loss_type == 'L1':
        loss = F.l1_loss(pred, target)
    elif loss_type == 'SSIM':
        loss = 1 - ssim(pred, target, data_range=1, size_average=True)
    elif loss_type == 'Fusion1':
        loss = lambda_value * F.mse_loss(pred, target) + (1-lambda_value) * (1 - ssim(pred, target, data_range=1, size_average=True))
    elif loss_type == 'Fusion2':
        loss = lambda_value * F.l1_loss(pred, target) + (1-lambda_value) * (1 - ssim(pred, target, data_range=1, size_average=True))
    elif loss_type == 'Fusion3':
        loss = lambda_value * F.mse_loss(pred, target) + (1-lambda_value) * F.l1_loss(pred, target)
    elif loss_type == 'Fusion4':
        loss = lambda_value * F.l1_loss(pred, target) + (1-lambda_value) * (1 - ms_ssim(pred, target, data_range=1, size_average=True))
    elif loss_type == 'Fusion_hinerv':
        loss = lambda_value * F.l1_loss(pred, target) + (1-lambda_value)  * (1 - ms_ssim(pred, target, data_range=1, size_average=True, win_size=5))
    return loss


def strip_lowerdiag(L):
    if L.shape[1] == 3:
        uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")
        uncertainty[:, 0] = L[:, 0, 0]
        uncertainty[:, 1] = L[:, 0, 1]
        uncertainty[:, 2] = L[:, 0, 2]
        uncertainty[:, 3] = L[:, 1, 1]
        uncertainty[:, 4] = L[:, 1, 2]
        uncertainty[:, 5] = L[:, 2, 2]

    elif L.shape[1] == 2:
        uncertainty = torch.zeros((L.shape[0], 3), dtype=torch.float, device="cuda")
        uncertainty[:, 0] = L[:, 0, 0]
        uncertainty[:, 1] = L[:, 0, 1]
        uncertainty[:, 2] = L[:, 1, 1]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def build_rotation_2d(r):
    '''
    Build rotation matrix in 2D.
    '''
    R = torch.zeros((r.size(0), 2, 2), device='cuda')
    R[:, 0, 0] = torch.cos(r)[:, 0]
    R[:, 0, 1] = -torch.sin(r)[:, 0]
    R[:, 1, 0] = torch.sin(r)[:, 0]
    R[:, 1, 1] = torch.cos(r)[:, 0]
    return R

def build_scaling_rotation_2d(s, r, device):
    L = torch.zeros((s.shape[0], 2, 2), dtype=torch.float, device='cuda')
    R = build_rotation_2d(r, device)
    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L = R @ L
    return L
    
def build_covariance_from_scaling_rotation_2d(scaling, scaling_modifier, rotation, device):
    '''
    Build covariance metrix from rotation and scale matricies.
    '''
    L = build_scaling_rotation_2d(scaling_modifier * scaling, rotation, device)
    actual_covariance = L @ L.transpose(1, 2)
    return actual_covariance

def build_triangular(r):
    R = torch.zeros((r.size(0), 2, 2), device=r.device)
    R[:, 0, 0] = r[:, 0]
    R[:, 1, 0] = r[:, 1]
    R[:, 1, 1] = r[:, 2]
    return R



def process_yuv_video(file_path, width, height):
    
    frame_size = width * height * 3 // 2
    
    file_size = os.path.getsize(file_path)
    total_frames = file_size // frame_size
    
    video_frames = []
    
    with open(file_path, 'rb') as f:
        
        for _ in tqdm(range(total_frames), desc="Processing frames"):
            
            yuv_frame = f.read(frame_size)
            if not yuv_frame:
                break  
            
            yuv = np.frombuffer(yuv_frame, dtype=np.uint8).reshape((height * 3 // 2, width))
            
            rgb_frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB_I420)
            
            video_frames.append(rgb_frame)
    return video_frames

def process_yuv_video_frame_start_end(file_path, width, height, start_frame, end_frame):
    frame_size = width * height * 3 // 2
    file_size = os.path.getsize(file_path)
    total_frames = file_size // frame_size

    # 边界保护
    start_frame = max(0, start_frame)
    end_frame = min(end_frame, total_frames)

    if start_frame >= end_frame:
        return []

    video_frames = []

    with open(file_path, 'rb') as f:
        # 跳到起始帧
        f.seek(start_frame * frame_size)

        for _ in tqdm(range(start_frame, end_frame), desc="Processing frames"):
            yuv_frame = f.read(frame_size)
            if len(yuv_frame) < frame_size:
                break

            yuv = np.frombuffer(yuv_frame, dtype=np.uint8).reshape((height * 3 // 2, width))
            rgb_frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB_I420)
            video_frames.append(rgb_frame)

    return video_frames



def generate_video(videodir,image_list, data_name, model_name,fps,iterations,num_points,origin):
    video_path = Path(f"./checkpoints/{videodir}/{data_name}/{model_name}_{iterations}_{num_points}/video")
    video_path.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
    # Define the output video file name
    if origin:
        filename = "video.mp4"
    else:
        filename = "combined_video.mp4"
    # Get the size of the first image dynamically
    first_image = image_list[0]
    width, height = first_image.size  # Extract the size of the first image
    # Create the video writer with the actual image dimensions
    output_size = (width, height)
    video = cv2.VideoWriter(str(video_path / filename), cv2.VideoWriter_fourcc(*'mp4v'), fps, output_size)
    # Add images to the video writer
    for img in tqdm(image_list, desc="Processing images", unit="image"):  # Iterate directly over the image_list      
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        video.write(img_cv)
    # Finalize and close the video writer
    video.release()
    if origin:
        print("video.mp4: MP4 video created successfully.")
    else:
        print("combined_video.mp4: MP4 video created successfully.")


def generate_video_wq(videodir,image_list, data_name, model_name,fps,iterations,num_points,origin):
    video_path = Path(f"./checkpoints_quant/{videodir}/{data_name}/{model_name}_{iterations}_{num_points}/video")
    video_path.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
    # Define the output video file name
    if origin:
        filename = "video.mp4"
    else:
        filename = "combined_video.mp4"
    # Get the size of the first image dynamically
    first_image = image_list[0]
    width, height = first_image.size  # Extract the size of the first image
    # Create the video writer with the actual image dimensions
    output_size = (width, height)
    video = cv2.VideoWriter(str(video_path / filename), cv2.VideoWriter_fourcc(*'mp4v'), fps, output_size)
    # Add images to the video writer
    for img in tqdm(image_list, desc="Processing images", unit="image"):  # Iterate directly over the image_list      
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        video.write(img_cv)
    # Finalize and close the video writer
    video.release()
    if origin:
        print("video.mp4: MP4 video created successfully.")
    else:
        print("combined_video.mp4: MP4 video created successfully.")


def generate_video_decoder(videodir,image_list, data_name, model_name,fps,num_points):
    video_path = Path(f"./checkpoints_quant/{videodir}/{data_name}/{model_name}_{num_points}/video")
    video_path.mkdir(parents=True, exist_ok=True) 
    filename = "video.mp4"
    first_image = image_list[0]
    width, height = first_image.size 
    output_size = (width, height)
    video = cv2.VideoWriter(str(video_path / filename), cv2.VideoWriter_fourcc(*'mp4v'), fps, output_size)
    for img in tqdm(image_list, desc="Processing images", unit="image"): 
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        video.write(img_cv)
    video.release()
    print("video.mp4: MP4 video created successfully.")





class EarlyStopping:
    def __init__(self, patience=100, min_delta=0):
        self.patience = patience  # Number of tolerated iterations with no improvement
        self.min_delta = min_delta  # Minimum improvement threshold
        self.best_loss = None  # Stores the best loss value
        self.counter = 0  # Tracks the number of iterations without improvement

    def __call__(self, current_loss):
        if self.best_loss is None:
            self.best_loss = current_loss
            return False  # Do not stop training

        # If the improvement over the previous best loss is less than min_delta, consider it no improvement
        if self.best_loss - current_loss > self.min_delta:
            self.best_loss = current_loss
            self.counter = 0  # Reset counter
        else:
            self.counter += 1

        # If the counter exceeds patience, stop training
        if self.counter >= self.patience:
            return True  # Stop training

        return False  # Continue training

class EarlyStoppingGPU:
    """
    GPU-friendly early stopping:
    - best_loss / min_delta / counter are stored as torch tensors (on the same device as loss)
    - __call__ expects `current_loss` to be a scalar tensor (0-dim) on GPU (or CPU也行，但建议GPU)
    - returns a Python bool (stop or not) without calling .item()
    """
    def __init__(self, patience: int = 100, min_delta: float = 0.0):
        self.patience = int(patience)
        self.min_delta_value = float(min_delta)

        self.initialized = False
        self.best_loss = None      # torch scalar tensor
        self.min_delta = None      # torch scalar tensor
        self.counter = None        # torch scalar tensor (int64)

    @torch.no_grad()
    def __call__(self, current_loss: torch.Tensor) -> bool:
        # current_loss should be a scalar tensor (shape: [] or [1])
        if not torch.is_tensor(current_loss):
            raise TypeError("current_loss must be a torch.Tensor (scalar).")
        if current_loss.numel() != 1:
            raise ValueError(f"current_loss must be a scalar tensor, got numel={current_loss.numel()}")

        # make it 0-dim scalar
        current_loss = current_loss.reshape(())

        device = current_loss.device
        dtype = current_loss.dtype

        if not self.initialized:
            self.best_loss = current_loss.clone()
            self.min_delta = torch.tensor(self.min_delta_value, device=device, dtype=dtype)
            self.counter = torch.zeros((), device=device, dtype=torch.int64)
            self.initialized = True
            return False

        # improvement = best_loss - current_loss
        improvement = self.best_loss - current_loss

        # If improvement > min_delta => improved
        improved = improvement > self.min_delta

        # Update best_loss and counter on GPU
        if improved:
            self.best_loss = current_loss.clone()
            self.counter.zero_()
        else:
            self.counter.add_(1)

        # Stop if counter >= patience
        # This returns a 0-dim bool tensor; .item() would sync, so use bool(tensor) instead.
        # bool(tensor) is allowed for 0-dim tensors and doesn't force copying a float from GPU.
        stop = self.counter >= self.patience
        return bool(stop)


    

def detect_outliers_mean_diff(values, window_size=10, threshold=3):
    outliers = []
    for i in range(len(values)):
        # Define the window range
        start_idx = max(0, i - window_size)
        end_idx = min(len(values), i + window_size)
        
        # Calculate local mean and standard deviation
        local_mean = np.mean(values[start_idx:end_idx])
        local_std = np.std(values[start_idx:end_idx])
        # Check if the value exceeds the threshold
        if (values[i] - local_mean) > threshold * local_std:
            outliers.append(i)
        elif(values[i] > local_mean*threshold):
            outliers.append(i)
    return outliers






class ResolutionScheduler():
	"""
	DashGaussian training scheduler of resolution.
	"""
	def __init__(self, original_image: torch.Tensor, iterations: int, gaussian_number: int, resolution_mode: str = "freq") -> None:
		if not isinstance(original_image, torch.Tensor):
			raise TypeError("ResolutionScheduler expects original_image to be a torch.Tensor, got {}".format(type(original_image)))
		if original_image.dim() == 3:
			original_image = original_image.unsqueeze(0)
		elif original_image.dim() != 4:
			raise ValueError(
				"ResolutionScheduler expects original_image with shape CHW or BCHW, got {}".format(tuple(original_image.shape))
			)
		self.resolution_mode = resolution_mode
		self.max_steps = iterations  # Total number of training iterations (max steps).
		self.start_significance_factor = 4 # Used to set the initial low-frequency energy threshold:
		self.max_reso_scale = 8 # Upper bound of the coarsest resolution scale (largest downsampling factor);
		self.reso_sample_num = 32 # Must be no less than 2; Number of resolution stages/samples (including the first and last);
		self.reso_scales = None # List of resolution scales per stage (e.g., [max_reso_scale, ..., 1.0]).
		self.reso_level_significance = None # Target "significance" (low-frequency window energy) per stage.
		self.reso_level_begin = None # Starting iteration index for each stage (monotonically increasing).
		self.increase_reso_until = 2500 # Only increase resolution during the first half of training;
		self.next_i = 2 # Stage pointer used in get_res_scale():
		self.gaussina_init= gaussian_number//2
		self.gaussian_number = gaussian_number
		# Generate schedulers.
		self.original_image = original_image
		self.init_reso_scheduler(original_image)
    
    


	def get_res_scale(self, iteration):
		if self.resolution_mode == "const":
			# Constant resolution: always train at full resolution.
			return 1
		elif self.resolution_mode == "freq" or self.resolution_mode == "freq_contiguous":
			# Frequency-based progressive resolution schedule.
			if iteration >= self.increase_reso_until:
				# After a certain point (e.g., first half of training), stop increasing resolution
				# and always use full resolution.
				return 1
			if iteration < self.reso_level_begin[1]:
				# Very early stage: before the first boundary, use the coarsest preset scale.
				return self.reso_scales[0]
			# Maintain a moving pointer 'next_i' to locate which stage interval the current iteration belongs to. 
			while iteration >= self.reso_level_begin[self.next_i]:
				# If the index is out of the range of 'reso_level_begin', there must be something wrong with the scheduler.
				self.next_i += 1
			# Current stage index i satisfies: reso_level_begin[i] <= iteration < reso_level_begin[i+1]
			i = self.next_i - 1
			# Get the current interval [i_now, i_nxt) on the iteration axis.
			# i_now, i_nxt = self.reso_level_begin[i: i + 2]
			i_now = self.reso_level_begin[i]
			i_nxt = self.reso_level_begin[i+1]
			# Get the scales of the previous stage (s_lst) and the current stage (s_now).
			# Note: s_lst = reso_scales[i-1], s_now = reso_scales[i]
			# s_lst, s_now = self.reso_scales[i - 1: i + 1]
			s_lst = self.reso_scales[i - 1]
			s_now = self.reso_scales[i]
			# Compute an interpolated scale inside this interval.
			scale = (1 / ((iteration - i_now) / (i_nxt - i_now) * (1/s_now**2 - 1/s_lst**2) + 1/s_lst**2))**0.5
			if self.resolution_mode == "freq_contiguous":
				return scale
			else:
				return int(scale)
		else:
			raise NotImplementedError("Resolution mode '{}' is not implemented.".format(self.resolution_mode))
	
	def lr_decay_from_iter(self)-> int:
		if self.resolution_mode == "const":
			return 1
		for i, s in zip(self.reso_level_begin, self.reso_scales):
			if s < 2:
				return i
	

	def init_reso_scheduler(self, original_image):
		if self.resolution_mode != "freq":
			print("[ INFO ] Skipped resolution scheduler initialization, the resolution mode is {}".format(self.resolution_mode))
			return

		def compute_win_significance(significance_map: torch.Tensor, scale: float):
			"""
			Compute the "significance" inside a centered window on a 2D map.

			Given a significance/energy map whose last two dimensions are (H, W),
			this function crops a rectangular window centered at the frequency/image center,
			where the window size is proportional to 1/scale:
				win_h = int(H / scale), win_w = int(W / scale)

			It then sums all values inside this window and returns the result as a Python scalar.
			"""
			h, w = significance_map.shape[-2:]
			c = ((h + 1) // 2, (w + 1) // 2)
			win_size = (int(h / scale), int(w / scale))
			win_significance = significance_map[..., c[0]-win_size[0]//2: c[0]+win_size[0]//2, c[1]-win_size[1]//2: c[1]+win_size[1]//2].sum().item()
			return win_significance
		
		def scale_solver(significance_map: torch.Tensor, target_significance: float):
			L, R, T = 0., 1., 64
			for _ in range(T):
				mid = (L + R) / 2
				win_significance = compute_win_significance(significance_map, 1 / mid)
				if win_significance < target_significance:
					L = mid
				else:
					R = mid
			return 1 / mid
		
		print("[ INFO ] Initializing resolution scheduler...")

		self.max_reso_scale = 8
		self.next_i = 2
		centered_fft_mag = None
		

		img_fft_centered = torch.fft.fftshift(torch.fft.fft2(original_image), dim=(-2, -1))
		centered_fft_mag = (img_fft_centered.real.square() + img_fft_centered.imag.square()).sqrt()
		e_total = centered_fft_mag.sum().item()
		e_min = e_total / self.start_significance_factor
		self.max_reso_scale = min(self.max_reso_scale, scale_solver(centered_fft_mag, e_min))

		modulation_func = math.log

		self.reso_scales = []
		self.reso_level_significance = []
		self.reso_level_begin = []
		E_total = centered_fft_mag.sum().item()
		E_min = compute_win_significance(centered_fft_mag, self.max_reso_scale)
		self.reso_level_significance.append(E_min)
		self.reso_scales.append(self.max_reso_scale)
		# self.reso_level_begin.append(0)
		for i in range(1, self.reso_sample_num - 1):
			self.reso_level_significance.append((E_total - E_min) * (i - 0) / (self.reso_sample_num-1 - 0) + E_min)
			self.reso_scales.append(scale_solver(centered_fft_mag, self.reso_level_significance[-1]))
			self.reso_level_significance[-2] = modulation_func(self.reso_level_significance[-2] / E_min)
			self.reso_level_begin.append(int(self.increase_reso_until * self.reso_level_significance[-2] / modulation_func(E_total / E_min)))
		self.reso_level_significance.append(modulation_func(E_total / E_min))
		self.reso_scales.append(1.)
		self.reso_level_significance[-2] = modulation_func(self.reso_level_significance[-2] / E_min)
		self.reso_level_begin.append(int(self.increase_reso_until * self.reso_level_significance[-2] / modulation_func(E_total / E_min)))
		self.reso_level_begin.append(self.increase_reso_until)

	def get_img_scale(self, iteration: int) -> torch.Tensor:
		B, C, H, W = self.original_image.shape
		self.next_i=2
		r = int(self.get_res_scale(iteration))
		if r ==1:
			return self.original_image
		h2 = max(1, H // r)
		w2 = max(1, W // r)
		F = torch.fft.fftshift(torch.fft.fft2(self.original_image, dim=(-2, -1)), dim=(-2, -1))
		cy, cx = H // 2, W // 2
		y0 = cy - h2 // 2
		x0 = cx - w2 // 2
		y1 = y0 + h2
		x1 = x0 + w2
		F_crop = F[..., y0:y1, x0:x1]
		F_crop = F_crop * (1.0 / (r * r))
		# Inverse FFT
		F_crop = torch.fft.ifftshift(F_crop, dim=(-2, -1))
		img_ds = torch.fft.ifft2(F_crop, dim=(-2, -1)).real
		img_ds = img_ds.clamp(0.0, 1.0)
		return img_ds

	def save_scaled_image(self, iteration: int, out_path: str) -> None:
		self.next_i=2
		y = self.get_img_scale(iteration).detach().cpu()
		if y.dim() != 4:
			raise ValueError("Expected scaled image to be BCHW, got shape {}".format(tuple(y.shape)))
		if y.shape[0] != 1:
			raise ValueError("save_scaled_image expects batch size 1, got {}".format(y.shape[0]))
		y = y[0]
		if y.shape[0] == 1:
			y = y.repeat(3, 1, 1)  # grayscale -> RGB
		y_u8 = (y.clamp(0, 1) * 255.0).round().to(torch.uint8)  # CHW
		hwc = y_u8.permute(1, 2, 0).numpy()
		from PIL import Image
		Image.fromarray(hwc, mode="RGB").save(out_path)
	
	def get_gaussian_budget(self, iteration: int) -> int:
		self.next_i=2
		r = self.get_res_scale(iteration)
		r = int(r)
		S = float(self.max_steps)
		expo = 2.0 - float(iteration) / max(S, 1.0)

		# Safety: r >= 1
		r = max(1.0, r)

		P_i = self.gaussina_init + (self.gaussian_number - self.gaussina_init) / (r ** expo)

		# Clamp to [P_init, P_fin] for numerical safety
		P_i = max(self.gaussina_init, min(self.gaussian_number, P_i))

		return int(round(P_i))
    
    
	
	


class IterStageWallAverager:
    """
    统计每次 iter 的 forward/backward/step 的“CPU+GPU 串行完成总时间”(ms)：
    每段：t0=perf_counter(); 做这段; torch.cuda.synchronize(); t1=perf_counter()
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.n = 0
        self.sum_forward_ms = 0.0
        self.sum_backward_ms = 0.0
        self.sum_step_ms = 0.0
        self._t0 = None

    def start(self):
        self._t0 = time.perf_counter()

    def stop_ms(self) -> float:
        # 等 GPU 完成，保证“CPU+GPU 都做完”
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        return (t1 - self._t0) * 1000.0

    def add(self, f_ms: float, b_ms: float, s_ms: float):
        self.sum_forward_ms += float(f_ms)
        self.sum_backward_ms += float(b_ms)
        self.sum_step_ms += float(s_ms)
        self.n += 1

    def mean(self):
        if self.n == 0:
            return dict(n=0, forward_ms=0.0, backward_ms=0.0, step_ms=0.0, total_ms=0.0)
        f = self.sum_forward_ms / self.n
        b = self.sum_backward_ms / self.n
        s = self.sum_step_ms / self.n
        return dict(n=self.n, forward_ms=f, backward_ms=b, step_ms=s, total_ms=f + b + s)

    def summary(self) -> str:
        m = self.mean()
        return (f"[AVG over {m['n']} iters] "
                f"forward_total={m['forward_ms']:.3f} ms, "
                f"backward_total={m['backward_ms']:.3f} ms, "
                f"step_total={m['step_ms']:.3f} ms, "
                f"total={m['total_ms']:.3f} ms")

class EarlyStopperEMA:
    def __init__(self, patience=200, min_rel_improve=1e-4, ema_beta=0.98,
                 check_every=50, warmup=2000):
        """
        min_rel_improve: 相对改进阈值（越小越严格），例如 1e-4 表示提升 <0.01% 视为没进步
        """
        self.patience = patience
        self.min_rel_improve = min_rel_improve
        self.ema_beta = ema_beta
        self.check_every = check_every
        self.warmup = warmup

        self.ema = None
        self.best = None
        self.bad = 0

    @torch.no_grad()
    def step(self, loss: torch.Tensor, it: int) -> bool:
        # warmup：不检查
        if it < self.warmup:
            self._update_ema(loss)
            return False

        # 只在 check_every 才做一次 host decision（会触发一次同步）
        self._update_ema(loss)
        if it % self.check_every != 0:
            return False

        cur = float(self.ema.item())  # 这里会同步，但频率很低
        if self.best is None:
            self.best = cur
            self.bad = 0
            return False

        # 相对改进： (best-cur)/best
        rel_improve = (self.best - cur) / max(self.best, 1e-12)

        if rel_improve > self.min_rel_improve:
            self.best = cur
            self.bad = 0
        else:
            self.bad += 1

        return self.bad >= self.patience

    def _update_ema(self, loss: torch.Tensor):
        x = loss.detach()
        if self.ema is None:
            self.ema = x
        else:
            self.ema = self.ema * self.ema_beta + x * (1 - self.ema_beta)


# class MetricEarlyStopper:
#     def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = "max"):
#         self.patience = int(patience)
#         self.min_delta = float(min_delta)
#         if mode not in ("max", "min"):
#             raise ValueError("mode must be 'max' or 'min'")
#         self.mode = mode

#         self.best = None
#         self.bad = 0

#     @torch.no_grad()
#     def step(self, metric_value: float) -> bool:
#         v = float(metric_value)
#         if self.best is None:
#             self.best = v
#             self.bad = 0
#             return False

#         if self.mode == "max":
#             improved = v > (self.best + self.min_delta)
#         else:
#             improved = v < (self.best - self.min_delta)

#         if improved:
#             self.best = v
#             self.bad = 0
#         else:
#             self.bad += 1

#         return self.bad >= self.patience

class MetricEarlyStopper:
    def __init__(self, patience: int = 10, min_delta: float = 0.01, mode: str = "max"):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        if mode not in ("max", "min"):
            raise ValueError("mode must be 'max' or 'min'")
        self.mode = mode

        self.best = None
        self.bad = 0

    @torch.no_grad()
    def step(self, metric_value: float) -> bool:
        v = float(metric_value)
        if self.best is None:
            self.best = v
            self.bad = 0
            return False
        rel_improve = (v-self.best) / max(self.best, 1e-12)
        if self.mode == "max":
            improved = rel_improve > self.min_delta
        else:
            improved = rel_improve < -self.min_delta

        if improved:
            self.best = v
            self.bad = 0
        else:
            self.bad += 1

        return self.bad >= self.patience


# class ModelEMA:
#     def __init__(self, model: torch.nn.Module, decay: float = 0.999):
#         self.decay = float(decay)
#         self.ema = copy.deepcopy(model).eval()
#         for p in self.ema.parameters():
#             p.requires_grad_(False)

#     @torch.no_grad()
#     def reset(self, model: torch.nn.Module):
#         self.ema = copy.deepcopy(model).eval()
#         for p in self.ema.parameters():
#             p.requires_grad_(False)

#     def to(self, *args, **kwargs):
#         self.ema.to(*args, **kwargs)
#         return self

#     @torch.no_grad()
#     def update(self, model: torch.nn.Module):
#         msd = model.state_dict()
#         esd = self.ema.state_dict()

#         need_reset = False
#         for k, v in msd.items():
#             if k not in esd:
#                 need_reset = True
#                 break
#             if esd[k].shape != v.shape:
#                 need_reset = True
#                 break

#         if need_reset:
#             self.reset(model)
#             return

#         for k, v_ema in esd.items():
#             v = msd[k]
#             if not torch.is_floating_point(v):
#                 v_ema.copy_(v)
#             else:
#                 v_ema.mul_(self.decay).add_(v, alpha=1.0 - self.decay)

# class ModelEMA:
#     def __init__(
#         self,
#         model: torch.nn.Module,
#         decay: float = 0.999,
#         ema_keys: Tuple[str, ...] = ("_cholesky", "_xyz", "_features_dc", "rgb_W"),
#     ):
#         self.decay = float(decay)
#         self.ema_keys = tuple(ema_keys)
#         self.ema = copy.deepcopy(model).eval()
#         for p in self.ema.parameters():
#             p.requires_grad_(False)

#     def _should_ema(self, key: str) -> bool:
#         for ek in self.ema_keys:
#             if key == ek or key.endswith("." + ek):
#                 return True
#         return False

#     def _need_rebuild(self, model: torch.nn.Module) -> bool:
#         msd = model.state_dict()
#         esd = self.ema.state_dict()

#         # key 数不同
#         if msd.keys() != esd.keys():
#             return True

#         return False

#     def _set_tensor_by_key(self, key: str, tensor: torch.Tensor, is_param: bool):
#         parts = key.split(".")
#         obj = self.ema
#         for p in parts[:-1]:
#             obj = getattr(obj, p)
#         name = parts[-1]
#         if is_param:
#             obj._parameters[name] = nn.Parameter(tensor)
#         else:
#             obj._buffers[name] = tensor

#     @torch.no_grad()
#     def reset(self, model: torch.nn.Module):
#         msd = model.state_dict()
#         ema_params = dict(self.ema.named_parameters())
#         ema_bufs = dict(self.ema.named_buffers())
#         model_params = dict(model.named_parameters())
#         model_bufs = dict(model.named_buffers())

#         for k, v in msd.items():
#             if k in ema_params:
#                 if ema_params[k].shape != v.shape:
#                     src = model_params.get(k, None)
#                     if src is not None:
#                         self._set_tensor_by_key(k, src.detach().clone(), is_param=True)
#                     else:
#                         self._set_tensor_by_key(k, v.detach().clone(), is_param=True)
#             elif k in ema_bufs:
#                 if ema_bufs[k].shape != v.shape:
#                     src = model_bufs.get(k, None)
#                     if src is not None:
#                         self._set_tensor_by_key(k, src.detach().clone(), is_param=False)
#                     else:
#                         self._set_tensor_by_key(k, v.detach().clone(), is_param=False)

#         self.ema.load_state_dict(msd, strict=False)
#         self.ema.eval()
#         for p in self.ema.parameters():
#             p.requires_grad_(False)

#     def to(self, *args, **kwargs):
#         self.ema.to(*args, **kwargs)
#         return self

#     @torch.no_grad()
#     def update(self, model: torch.nn.Module):
#         if self._need_rebuild(model):
            
#             self.reset(model)
#             return

#         msd = model.state_dict()
#         esd = self.ema.state_dict()
#         model_params = dict(model.named_parameters())
#         ema_params = dict(self.ema.named_parameters())

#         # 1) hard copy all non-EMA keys (including resizing when shape changed)
#         for k, v in msd.items():
#             if self._should_ema(k) and torch.is_floating_point(v):
#                 continue
#             if (k in esd) and (esd[k].shape == v.shape):
#                 esd[k].copy_(v)
#             else:
#                 self._set_tensor_by_key(k, v.detach().clone(), is_param=(k in model_params))

#         # refresh after potential tensor replacement
#         esd = self.ema.state_dict()

#         # 2) point-id aligned EMA update for selected keys
#         if ("point_ids" in msd) and ("point_ids" in esd):
#             sgd_pids = msd["point_ids"].detach()
#             ema_pids = esd["point_ids"].detach()

#             if (sgd_pids.ndim == 1) and (ema_pids.ndim == 1):
#                 ema_sorted, ema_order = torch.sort(ema_pids)
#                 pos = torch.searchsorted(ema_sorted, sgd_pids)
#                 valid = (pos < ema_sorted.numel()) & (ema_sorted[pos] == sgd_pids)
#                 sgd_idx = torch.nonzero(valid, as_tuple=False).view(-1)
#                 ema_idx = ema_order[pos[valid]]

#                 for k, v in msd.items():
#                     if (not self._should_ema(k)) or (not torch.is_floating_point(v)):
#                         continue
#                     if k not in esd:
#                         continue

#                     v_ema_old = esd[k]
#                     if (v.ndim == 0) or (v_ema_old.ndim == 0):
#                         self._set_tensor_by_key(
#                             k,
#                             v_ema_old.detach().mul(self.decay).add(v, alpha=1.0 - self.decay),
#                             is_param=(k in ema_params),
#                         )
#                         continue

#                     # expect point dimension at dim0
#                     if (v.shape[0] != sgd_pids.shape[0]) or (v_ema_old.shape[0] != ema_pids.shape[0]):
#                         self._set_tensor_by_key(k, v.detach().clone(), is_param=(k in ema_params))
#                         continue

#                     new_v = v.detach().clone()  # new points: directly use SGD values
#                     if sgd_idx.numel() > 0:
#                         old_sel = v_ema_old.index_select(0, ema_idx)
#                         cur_sel = v.index_select(0, sgd_idx)
#                         upd = old_sel.mul(self.decay).add(cur_sel, alpha=1.0 - self.decay)
#                         new_v.index_copy_(0, sgd_idx, upd)

#                     self._set_tensor_by_key(k, new_v, is_param=(k in ema_params))

#         for p in self.ema.parameters():
#             p.requires_grad_(False)

class ModelEMA:
    def __init__(
        self,
        model: torch.nn.Module,
        decay: float = 0.999,
        pointwise_ema_keys=("_cholesky", "_xyz", "_features_dc", "rgb_W"),
        global_ema_suffixes=("scale", "beta"),
        global_ema_contains=("_codebook.embed",),
    ):
        self.decay = float(decay)
        self.pointwise_ema_keys = tuple(pointwise_ema_keys)
        self.global_ema_suffixes = tuple(global_ema_suffixes)
        self.global_ema_contains = tuple(global_ema_contains)
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def _is_pointwise_ema(self, key: str) -> bool:
        for ek in self.pointwise_ema_keys:
            if key == ek or key.endswith("." + ek):
                return True
        return False

    def _is_global_ema(self, key: str) -> bool:
        for s in self.global_ema_suffixes:
            if key == s or key.endswith("." + s):
                return True
        for c in self.global_ema_contains:
            if c in key:
                return True
        return False

    def _should_ema(self, key: str) -> bool:
        return self._is_pointwise_ema(key) or self._is_global_ema(key)

    def _need_rebuild(self, model: torch.nn.Module) -> bool:
        msd = model.state_dict()
        esd = self.ema.state_dict()

        # key 数不同
        if msd.keys() != esd.keys():
            return True

        return False

    def _set_tensor_by_key(self, key: str, tensor: torch.Tensor, is_param: bool):
        parts = key.split(".")
        obj = self.ema
        for p in parts[:-1]:
            obj = getattr(obj, p)
        name = parts[-1]
        if is_param:
            obj._parameters[name] = nn.Parameter(tensor)
        else:
            obj._buffers[name] = tensor

    @torch.no_grad()
    def reset(self, model: torch.nn.Module):
        msd = model.state_dict()
        ema_params = dict(self.ema.named_parameters())
        ema_bufs = dict(self.ema.named_buffers())
        model_params = dict(model.named_parameters())
        model_bufs = dict(model.named_buffers())

        for k, v in msd.items():
            if k in ema_params:
                if ema_params[k].shape != v.shape:
                    src = model_params.get(k, None)
                    if src is not None:
                        self._set_tensor_by_key(k, src.detach().clone(), is_param=True)
                    else:
                        self._set_tensor_by_key(k, v.detach().clone(), is_param=True)
            elif k in ema_bufs:
                if ema_bufs[k].shape != v.shape:
                    src = model_bufs.get(k, None)
                    if src is not None:
                        self._set_tensor_by_key(k, src.detach().clone(), is_param=False)
                    else:
                        self._set_tensor_by_key(k, v.detach().clone(), is_param=False)

        self.ema.load_state_dict(msd, strict=False)
        self.ema.eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def to(self, *args, **kwargs):
        self.ema.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        if self._need_rebuild(model):
            self.reset(model)
            return

        msd = model.state_dict()
        esd = self.ema.state_dict()
        model_params = dict(model.named_parameters())
        ema_params = dict(self.ema.named_parameters())

        # 1. hard copy all non-EMA keys
        for k, v in msd.items():
            if self._should_ema(k) and torch.is_floating_point(v):
                continue

            if (k in esd) and (esd[k].shape == v.shape):
                esd[k].copy_(v)
            else:
                self._set_tensor_by_key(
                    k,
                    v.detach().clone(),
                    is_param=(k in model_params),
                )

        esd = self.ema.state_dict()

        # 2. global EMA for quantizer params: scale / beta / codebook
        for k, v in msd.items():
            if (not self._is_global_ema(k)) or (not torch.is_floating_point(v)):
                continue
            if k not in esd:
                continue

            v_ema_old = esd[k]

            if v_ema_old.shape == v.shape:
                v_ema_old.mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
            else:
                self._set_tensor_by_key(
                    k,
                    v.detach().clone(),
                    is_param=(k in model_params),
                )

        esd = self.ema.state_dict()

        # 3. point-id aligned EMA for Gaussian parameters
        if ("point_ids" in msd) and ("point_ids" in esd):
            sgd_pids = msd["point_ids"].detach()
            ema_pids = esd["point_ids"].detach()

            if (sgd_pids.ndim == 1) and (ema_pids.ndim == 1):
                ema_sorted, ema_order = torch.sort(ema_pids)
                pos = torch.searchsorted(ema_sorted, sgd_pids)
                valid = (pos < ema_sorted.numel()) & (ema_sorted[pos] == sgd_pids)

                sgd_idx = torch.nonzero(valid, as_tuple=False).view(-1)
                ema_idx = ema_order[pos[valid]]

                for k, v in msd.items():
                    if (not self._is_pointwise_ema(k)) or (not torch.is_floating_point(v)):
                        continue
                    if k not in esd:
                        continue

                    v_ema_old = esd[k]

                    if (v.ndim == 0) or (v_ema_old.ndim == 0):
                        self._set_tensor_by_key(
                            k,
                            v_ema_old.detach().mul(self.decay).add(
                                v.detach(), alpha=1.0 - self.decay
                            ),
                            is_param=(k in ema_params),
                        )
                        continue

                    if (v.shape[0] != sgd_pids.shape[0]) or (v_ema_old.shape[0] != ema_pids.shape[0]):
                        self._set_tensor_by_key(
                            k,
                            v.detach().clone(),
                            is_param=(k in ema_params),
                        )
                        continue

                    new_v = v.detach().clone()
                    if sgd_idx.numel() > 0:
                        old_sel = v_ema_old.index_select(0, ema_idx)
                        cur_sel = v.index_select(0, sgd_idx)
                        upd = old_sel.mul(self.decay).add(
                            cur_sel, alpha=1.0 - self.decay
                        )
                        new_v.index_copy_(0, sgd_idx, upd)

                    self._set_tensor_by_key(k, new_v, is_param=(k in ema_params))

        for p in self.ema.parameters():
            p.requires_grad_(False)




def _tensor_to_numpy_2d(tensor):
    if tensor is None:
        return None
    if torch.is_tensor(tensor):
        tensor = tensor.detach().float().cpu().numpy()
    else:
        tensor = np.asarray(tensor, dtype=np.float32)
    if tensor.ndim != 2:
        tensor = tensor.reshape(tensor.shape[0], -1)
    finite_rows = np.isfinite(tensor).all(axis=1)
    return tensor[finite_rows]


def _plot_density_curves(values, labels, colors, title, xlabel, save_path, bins=256, value_range=None, note=None):
    values = _tensor_to_numpy_2d(values)
    if values is None or values.size == 0:
        return

    num_channels = min(values.shape[1], len(labels), len(colors))
    if num_channels == 0:
        return

    if value_range is None:
        lo = float(np.min(values[:, :num_channels]))
        hi = float(np.max(values[:, :num_channels]))
        if not np.isfinite(lo) or not np.isfinite(hi):
            return
        if abs(hi - lo) < 1e-12:
            lo -= 0.5
            hi += 0.5
    else:
        lo, hi = value_range

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=160)
    for channel_idx in range(num_channels):
        channel = values[:, channel_idx]
        channel = channel[np.isfinite(channel)]
        if value_range is not None:
            channel = channel[(channel >= lo) & (channel <= hi)]
        if channel.size == 0:
            continue
        counts, edges = np.histogram(channel, bins=bins, range=(lo, hi), density=False)
        hist = counts.astype(np.float64) / float(channel.size) * 100.0
        if not np.isfinite(hist).any():
            continue
        centers = 0.5 * (edges[:-1] + edges[1:])
        widths = np.diff(edges)
        nonzero = np.isfinite(hist) & (hist > 0)
        if not np.any(nonzero):
            continue
        ax.bar(
            centers[nonzero],
            hist[nonzero],
            width=widths[nonzero],
            label=labels[channel_idx],
            color=colors[channel_idx],
            alpha=0.5,
            edgecolor=colors[channel_idx],
            linewidth=0.35,
            align="center",
        )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Gaussian points (%)")
    ax.set_xlim(lo, hi)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    if note:
        ax.text(
            0.02,
            0.98,
            note,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.85},
        )
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)


def _zero_delta_note(values, name, labels, near_zero_atol=1e-3):
    values = _tensor_to_numpy_2d(values)
    if values is None or values.size == 0:
        return None
    num_channels = min(values.shape[1], len(labels))
    values = values[:, :num_channels]
    zero_mask = values == 0.0
    near_zero_mask = np.abs(values) < near_zero_atol
    zero_count = int(zero_mask.all(axis=1).sum())
    near_zero_count = int((near_zero_mask.all(axis=1) & ~zero_mask.all(axis=1)).sum())
    total_count = int(values.shape[0])
    lines = [
        f"{name} exact zero: {zero_count}/{total_count}",
        f"{name} near zero: {near_zero_count}/{total_count} (0<abs<1e-3)",
    ]
    for channel_idx, label in enumerate(labels[:num_channels]):
        channel_zero_count = int(zero_mask[:, channel_idx].sum())
        channel_near_zero_count = int((near_zero_mask[:, channel_idx] & ~zero_mask[:, channel_idx]).sum())
        lines.append(f"{label} exact zero: {channel_zero_count}/{total_count}")
        lines.append(f"{label} near zero: {channel_near_zero_count}/{total_count}")
    return "\n".join(lines)


def save_parameter_distribution_plots(decoded_model, frame_num, save_root, is_p_frame=False):
    dist_dir = save_root / "parameter_distribution"
    _plot_density_curves(
        decoded_model.get("_xyz"),
        labels=("x", "y"),
        colors=("tab:blue", "tab:orange"),
        title=f"Frame {frame_num} position distribution",
        xlabel="position value",
        save_path=dist_dir / "position" / f"{frame_num}_position_distribution.png",
        note=_zero_delta_note(decoded_model.get("_xyz"), "position", ("x", "y")) if is_p_frame else None,
    )
    _plot_density_curves(
        decoded_model.get("_features_dc"),
        labels=("r", "g", "b"),
        colors=("tab:red", "tab:green", "tab:blue"),
        title=f"Frame {frame_num} color distribution",
        xlabel="color value",
        save_path=dist_dir / "color" / f"{frame_num}_color_distribution.png",
        value_range=(-1.0, 1.0),
        note=_zero_delta_note(decoded_model.get("_features_dc"), "color", ("r", "g", "b")) if is_p_frame else None,
    )
    _plot_density_curves(
        decoded_model.get("_cholesky"),
        labels=("c0", "c1", "c2"),
        colors=("tab:purple", "tab:brown", "tab:pink"),
        title=f"Frame {frame_num} cholesky distribution",
        xlabel="cholesky value",
        save_path=dist_dir / "cholesky" / f"{frame_num}_cholesky_distribution.png",
        note=_zero_delta_note(decoded_model.get("_cholesky"), "cholesky", ("c0", "c1", "c2")) if is_p_frame else None,
    )


