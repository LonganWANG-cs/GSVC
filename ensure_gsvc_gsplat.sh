#!/usr/bin/env bash

# Build/check the bundled GSVC extension in the active Python environment.
# Usage: bash ensure_gsvc_gsplat.sh, or source it and call ensure_gsvc_gsplat.
ensure_gsvc_gsplat() (
  set -eo pipefail
  local script_dir repo_root source_hash runtime_id expected_id stamp_file
  local installed_so installed_pkg lock_file build_root gsplat_src cuda_home cuda_arch_list
  local environment_prefix temp_parent

  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  repo_root="${script_dir}"

  environment_prefix="$(python -c 'import sys; print(sys.prefix)')"
  if [[ ! -d "${repo_root}/gsplat/gsplat/cuda/csrc" ]]; then
    echo "[gsplat-check] ERROR: gsplat source not found under ${repo_root}/gsplat." >&2
    return 1
  fi

  source_hash="$({
    find "${repo_root}/gsplat" -type f \
      \( -name '*.py' -o -name '*.cu' -o -name '*.cuh' -o -name '*.cpp' -o -name '*.h' \) \
      -print0 | sort -z | xargs -0 sha256sum
  } | sha256sum | awk '{print $1}')"
  runtime_id="$(python - <<'PY'
import platform
import torch
if not torch.cuda.is_available():
    raise SystemExit("[gsplat-check] ERROR: no CUDA GPU is visible. Run on a GPU machine or allocated GPU node.")
print(f"python={platform.python_version()}|torch={torch.__version__}|torch_cuda={torch.version.cuda}")
PY
)"
  cuda_home="${CUDA_HOME:-${CUDA_PATH:-$(python -c 'from torch.utils.cpp_extension import CUDA_HOME; print(CUDA_HOME or "")')}}"
  cuda_arch_list="${TORCH_CUDA_ARCH_LIST:-$(python -c 'import torch; print(";".join(sorted({"%d.%d" % torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())})))')}"
  expected_id="source=${source_hash}|${runtime_id}|cuda_home=${cuda_home}|cuda_arch_list=${cuda_arch_list}"
  stamp_file="${environment_prefix}/.gsvc-gsplat-build-id"
  lock_file="${environment_prefix}/.gsvc-gsplat-install.lock"

  _fastgsvc_gsplat_matches() {
    [[ -f "${stamp_file}" ]] || return 1
    [[ "$(<"${stamp_file}")" == "${expected_id}" ]] || return 1
    installed_so="$(python - <<'PY' 2>/dev/null
import gsplat.csrc
print(gsplat.csrc.__file__)
PY
)" || return 1
    [[ "${installed_so}" == "${environment_prefix}"/* ]] || return 1
    grep -a -q 'FASTGSVC_DET_ATOMICS' "${installed_so}" || return 1
  }

  if _fastgsvc_gsplat_matches; then
    echo "[gsplat-check] OK: ${installed_so}"
    echo "[gsplat-check] ${expected_id}"
    return 0
  fi

  command -v flock >/dev/null 2>&1 || {
    echo "[gsplat-check] ERROR: flock is required for safe shared-environment installation." >&2
    return 1
  }
  exec 9>"${lock_file}"
  echo "[gsplat-check] Installed gsplat does not match this checkout; waiting for install lock."
  flock 9
  if _fastgsvc_gsplat_matches; then
    echo "[gsplat-check] OK after lock: ${installed_so}"
    return 0
  fi

  if [[ -z "${cuda_home}" || ! -x "${cuda_home}/bin/nvcc" ]]; then
    echo "[gsplat-check] ERROR: CUDA toolkit not found. Set CUDA_HOME to a toolkit containing bin/nvcc." >&2
    return 1
  fi

  temp_parent="$(cd "${TMPDIR:-/tmp}" && pwd -P)"
  build_root="$(mktemp -d "${temp_parent}/gsvc-gsplat-XXXXXX")"
  trap 'if [[ -n "${build_root:-}" && "$build_root" == "$temp_parent"/gsvc-gsplat-* ]]; then rm -rf -- "$build_root"; fi' EXIT
  gsplat_src="${build_root}/source"
  mkdir -p "${gsplat_src}"
  (
    cd "${repo_root}"
    tar --exclude='build' --exclude='*.egg-info' --exclude='*.so' --exclude='__pycache__' \
      -cf - gsplat
  ) | tar -xf - -C "${gsplat_src}"

  export CUDA_HOME="${cuda_home}"
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
  export TORCH_CUDA_ARCH_LIST="${cuda_arch_list}"
  export MAX_JOBS="${MAX_JOBS:-${SLURM_CPUS_PER_TASK:-4}}"
  export TORCH_EXTENSIONS_DIR="${build_root}/torch-extensions"

  # Install extension dependencies without replacing the active PyTorch build.
  python -m pip install jaxtyping "rich>=12"
  echo "[gsplat-check] Building GSVC gsplat for ${runtime_id} (GPU arch ${cuda_arch_list})."
  python -m pip install --no-deps --no-cache-dir --no-build-isolation \
    --force-reinstall "${gsplat_src}/gsplat"

  installed_pkg="$(python - <<'PY'
import gsplat
print(gsplat.__file__)
PY
)"
  installed_so="$(python - <<'PY'
import gsplat.csrc
print(gsplat.csrc.__file__)
PY
)"
  if [[ "${installed_pkg}" != "${environment_prefix}"/* || "${installed_so}" != "${environment_prefix}"/* ]]; then
    echo "[gsplat-check] ERROR: rebuilt gsplat was not loaded from ${environment_prefix}." >&2
    return 1
  fi
  if ! grep -a -q 'FASTGSVC_DET_ATOMICS' "${installed_so}"; then
    echo "[gsplat-check] ERROR: rebuilt csrc.so lacks FASTGSVC_DET_ATOMICS." >&2
    return 1
  fi

  printf '%s\n' "${expected_id}" >"${stamp_file}"
  echo "[gsplat-check] Installed and verified: ${installed_so}"
)

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  ensure_gsvc_gsplat
fi
