"""Runtime configuration shared by GSVC command-line entry points.

The public interface uses ``GSVC_*`` names.  ``FASTGSVC_*`` is accepted only
as a backwards-compatible input for old experiment scripts.
"""
import os

_OPTIONS = {
    # Enable CUDA Graph training where supported; 0 selects eager execution.
    "CUDAGRAPH_TRAIN": "1",
    # Capture supported overfitting EMA evaluations; does not enable EMA itself.
    "CUDAGRAPH_EMA_EVAL": "1",
    # Eager quantization steps before graph capture, included in the total budget.
    "CG_EAGER_WARMUP": "5000",
    # Accelerate eligible RGB residual VQ; otherwise use the library implementation.
    "FAST_VQ": "1",
    # Initialize color codebooks with deterministic LBG unless already warm-started.
    "LBG_INIT": "1",
    # Warm-start P-frame color codebooks from the previous frame; changes training.
    "WARM_START_VQ": "0",
    # Train P-frame VQ codebooks with gradients instead of default EMA updates.
    "PFRAME_LEARNABLE_VQ": "0",
    # Accumulate scaled gradients as int64 to remove atomic-add ordering variation.
    # Rounding loses gradient precision and may reduce quality, so default to off.
    # For bitwise reproducibility, enable this and DETERMINISTIC_TORCH=1, with
    # fixed seeds, settings and hardware/software; avoid integer overflow.
    "DET_ATOMICS": "0",
    # Fixed-point scale is 2**value (1..62); larger values trade range for precision.
    "DET_ATOMICS_SCALE_LOG2": "40",
    # PyTorch determinism: 0=off, 1=strict, warn=warnings only.
    # Strict mode also needs a valid cuBLAS workspace and DET_ATOMICS=1 for gsplat.
    "DETERMINISTIC_TORCH": "0",
}

RUNTIME_DEFAULTS = {f"GSVC_{suffix}": value for suffix, value in _OPTIONS.items()}
RUNTIME_DEFAULTS.update({"CUBLAS_WORKSPACE_CONFIG": "<unset>", "GSPLAT_ISECT_WARMUP": "8"})


def configure_runtime_environment():
    """Resolve public settings before importing torch and print their values."""
    effective = {}
    for suffix, default in _OPTIONS.items():
        public, legacy = f"GSVC_{suffix}", f"FASTGSVC_{suffix}"
        value = os.environ.get(public, os.environ.get(legacy, default))
        os.environ[public] = value
        # The bundled CUDA extension retains its original ABI/environment
        # spelling.  Bridge just these two settings until that ABI changes.
        if suffix in ("DET_ATOMICS", "DET_ATOMICS_SCALE_LOG2"):
            os.environ[legacy] = value
        effective[public] = value

    mode = effective["GSVC_DETERMINISTIC_TORCH"]
    if mode in ("1", "warn"):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    effective["CUBLAS_WORKSPACE_CONFIG"] = os.environ.get(
        "CUBLAS_WORKSPACE_CONFIG", "<unset>"
    )
    print("[GSVC runtime] " + ", ".join(f"{k}={v}" for k, v in effective.items()), flush=True)
    return effective
