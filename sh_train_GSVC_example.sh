#!/bin/bash
#SBATCH --job-name=GSVC
#SBATCH --output=output/GSVC_o.txt
#SBATCH --error=output/GSVC_e.txt
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:h100-47:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --mail-type=ALL

# Before submitting from the repository root:
# mkdir -p output && sbatch sh_train_GSVC_example.sh
# Slurm opens its log files before this script starts.
set -eo pipefail
source ~/.bashrc
conda activate "${CONDA_ENV:-pGS_GSVC_extension}"
# Enable nounset after loading shell/conda initialization.
set -u

ROOT="/home/GSVC_extension"
cd "$ROOT"
PYTHON_SITE_PACKAGES="$(python -c 'import site; print(site.getsitepackages()[0])')"
export PYTHONPATH="${ROOT}:${PYTHON_SITE_PACKAGES}${PYTHONPATH:+:${PYTHONPATH}}"
export GSPLAT_ISECT_WARMUP=20
export GSPLAT_ISECT_SAFETY=6

source "$ROOT/ensure_gsvc_gsplat.sh"
ensure_gsvc_gsplat


DATASET_PATH="${DATASET_PATH:-/home/data/UVG/Beauty/Beauty_1920x1080_120fps_420_8bit_YUV.yuv}"
DATA_NAME="${DATA_NAME:-Beauty}"


# DATASET_PATH="/home/data/UVG/Beauty/Beauty_1920x1080_120fps_420_8bit_YUV.yuv" #Change to your own dataset path
# DATA_NAME="Beauty"  #Change to your own dataset name
# DATASET_PATH="/home/data/UVG/HoneyBee/HoneyBee_1920x1080_120fps_420_8bit_YUV.yuv"
# DATA_NAME="HoneyBee"
# DATASET_PATH="/home/data/UVG/Jockey/Jockey_1920x1080_120fps_420_8bit_YUV.yuv"
# DATA_NAME="Jockey"
# DATASET_PATH="/home/data/UVG/Bosphorus/Bosphorus_1920x1080_120fps_420_8bit_YUV.yuv"
# DATA_NAME="Bosphorus"
# DATASET_PATH="/home/data/UVG/ReadySetGo/ReadySteadyGo_1920x1080_120fps_420_8bit_YUV.yuv"
# DATA_NAME="ReadySetGo"
# DATASET_PATH="/home/data/UVG/YachtRide/YachtRide_1920x1080_120fps_420_8bit_YUV.yuv"
# DATA_NAME="YachtRide"
# DATASET_PATH="/home/data/UVG/ShakeNDry/ShakeNDry_1920x1080_120fps_420_8bit_YUV.yuv" 
# DATA_NAME="ShakeNDry" 

ITERATIONS=50000
IMAGE_LENGTH=50
FPS=24
if [[ -n "${GSVC_NUM_POINTS_LIST:-}" ]]; then
  read -r -a NUM_POINTS_LIST <<< "${GSVC_NUM_POINTS_LIST}"
elif [[ -n "${FASTGSVC_NUM_POINTS_LIST:-}" ]]; then
  # Compatibility for old experiment launchers.
  read -r -a NUM_POINTS_LIST <<< "${FASTGSVC_NUM_POINTS_LIST}"
else
  NUM_POINTS_LIST=(10000 20000 30000 50000 40000)
fi
EXTRA_ENCODER_ARGS=()


OVERFIT_SAVDIR="GSVC_overfitting_UVG"
OVERFIT_MODEL_DIR="GSVC_overfitting_UVG_m"
ENCODE_SAVDIR="GSVC_encode_UVG"
ENCODE_MODEL_DIR="GSVC_encode_UVG_m"


for num_points in "${NUM_POINTS_LIST[@]}"; do
  echo "[OVERFIT-woCG] data=${DATA_NAME} iters=${ITERATIONS} points=${num_points}"
  python train_GSVC_overfitting.py \
    --dataset "${DATASET_PATH}" \
    --data_name "${DATA_NAME}" \
    --num_points "${num_points}" \
    --savdir "${OVERFIT_SAVDIR}" \
    --savdir_m "${OVERFIT_MODEL_DIR}" \
    --iterations "${ITERATIONS}" \
    --image_length "${IMAGE_LENGTH}" \
    --fps "${FPS}" \
    --look_ahead 1 \
    --is_rm \
    --is_ad

  model_path="${ROOT}/checkpoints/${OVERFIT_MODEL_DIR}/${DATA_NAME}/GaussianVideo_${ITERATIONS}_${num_points}/gmodels_state_dict.pth"

  echo "[ENCODE-woCG] data=${DATA_NAME} iters=${ITERATIONS} points=${num_points}"
  python train_GSVC_encoder.py \
    --model_path "${model_path}" \
    --dataset "${DATASET_PATH}" \
    --data_name "${DATA_NAME}" \
    --num_points "${num_points}" \
    --savdir "${ENCODE_SAVDIR}" \
    --savdir_m "${ENCODE_MODEL_DIR}" \
    --iterations "${ITERATIONS}" \
    --image_length "${IMAGE_LENGTH}" \
    --fps "${FPS}" \
    --is_rm 

  encoded_path="${ROOT}/checkpoints_quant/${ENCODE_MODEL_DIR}/${DATA_NAME}/GaussianVideo_${ITERATIONS}_${num_points}/encoded_gmodels_state_dict.pth"
  echo "[DECODE] data=${DATA_NAME} iters=${ITERATIONS} points=${num_points}"
  python test_GSVC_decoder.py \
    --model_path "${encoded_path}" \
    --dataset "${DATASET_PATH}" \
    --data_name "${DATA_NAME}" \
    --savdir "${ENCODE_SAVDIR}" \
    --image_length "${IMAGE_LENGTH}" \
    --fps "${FPS}"
done

