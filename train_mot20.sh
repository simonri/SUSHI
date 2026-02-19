#!/bin/bash
# MOT20 Training Script for SUSHI
# Usage: bash train_mot20.sh [public|private] [DATA_PATH]
#
# Arguments:
#   MODE      : Detection mode - "public" (aplift) or "private" (byte065) [default: private]
#   DATA_PATH : Path to dataset root containing the MOT20 folder [default: /workspace/data]
#
# Prerequisites:
#   - SUSHI conda environment activated: conda activate SUSHI
#   - MOT20 dataset placed at DATA_PATH/MOT20/
#   - Re-ID model weights at fastreid-models/model_weights/
#     (download from: https://drive.google.com/file/d/1MovixfOLwnnXet05JLIGPy-Ag67fdW-2)
#
# Expected data structure:
#   DATA_PATH/
#   └── MOT20/
#       ├── seqmaps/
#       │   └── mot20-train-all.txt  (and other seqmap files)
#       ├── train/
#       │   ├── MOT20-01/
#       │   │   ├── det/det.txt
#       │   │   ├── gt/gt.txt
#       │   │   ├── img1/
#       │   │   └── seqinfo.ini
#       │   └── ...
#       └── test/
#           └── ...

set -e

# ── Arguments ────────────────────────────────────────────────────────────────
MODE="${1:-private}"        # "public" or "private"
DATA_PATH="${2:-/workspace/data}"

# ── Validate mode ─────────────────────────────────────────────────────────────
if [[ "$MODE" == "public" ]]; then
    DET_FILE="aplift"
    RUN="mot20_public_train"
elif [[ "$MODE" == "private" ]]; then
    DET_FILE="byte065"
    RUN="mot20_private_train"
else
    echo "ERROR: MODE must be 'public' or 'private', got: $MODE"
    exit 1
fi

# ── Config ────────────────────────────────────────────────────────────────────
REID_ARCH='fastreid_msmt_BOT_R50_ibn'

echo "========================================"
echo "SUSHI MOT20 Training"
echo "  Mode      : $MODE (det_file=$DET_FILE)"
echo "  Run ID    : $RUN"
echo "  Data path : $DATA_PATH"
echo "  ReID arch : $REID_ARCH"
echo "========================================"

# ── Move to project root ──────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Run training ──────────────────────────────────────────────────────────────
PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}" python scripts/main.py \
    --experiment_mode train \
    --cuda \
    --train_splits mot20-train-all \
    --val_splits mot20-val-split1 \
    --run_id "${RUN}" \
    --interpolate_motion \
    --linear_center_only \
    --det_file "${DET_FILE}" \
    --data_path "${DATA_PATH}" \
    --reid_embeddings_dir "reid_${REID_ARCH}" \
    --node_embeddings_dir "node_${REID_ARCH}" \
    --zero_nodes \
    --reid_arch "${REID_ARCH}" \
    --edge_level_embed \
    --save_cp \
    --pruning_method geometry motion_01 motion_01 motion_01 motion_01 motion_01 motion_01 motion_01 motion_01

echo ""
echo "Training complete. Checkpoints saved under experiments/${RUN}/"
