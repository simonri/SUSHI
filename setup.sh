#!/bin/bash
# =============================================================================
# SUSHI – Full Environment Setup Script
# =============================================================================
# Installs everything from scratch:
#   1. Miniconda3 (if not present)
#   2. SUSHI conda environment (Python 3.8, PyTorch 1.8.2 + CUDA 10.2)
#   3. fast-reid repository + its dependencies
#   4. MOT20 dataset (~4.7 GB) from motchallenge.net
#   5. ByteTrack (byte065) & APLift detection files for each MOT20 sequence
#   6. Re-ID model weights  (msmt_bot_R50-ibn.pth, ~308 MB)
#   7. Pretrained SUSHI models (mot20private.pth, mot20public.pth, …)
#   8. MOT20 seqmap files required by TrackEval
#
# Usage:
#   bash setup.sh [DATA_PATH]
#
#   DATA_PATH  Root directory for datasets. MOT20 will be placed at
#              DATA_PATH/MOT20/   [default: /workspace/data]
#
# After this script finishes, run training with:
#   bash train_mot20.sh [public|private] [DATA_PATH]
# =============================================================================
set -e

# ── Configuration ─────────────────────────────────────────────────────────────
DATA_PATH="${1:-/workspace/data}"
CONDA_INSTALL_DIR="/opt/miniconda3"
CONDA="${CONDA_INSTALL_DIR}/bin/conda"
PIP="${CONDA_INSTALL_DIR}/envs/SUSHI/bin/pip"
PYTHON="${CONDA_INSTALL_DIR}/envs/SUSHI/bin/python"
GDOWN="${CONDA_INSTALL_DIR}/envs/SUSHI/bin/gdown"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Google Drive direct-download base URL (bypasses rate limits)
GD_BASE="https://drive.usercontent.google.com/download?export=download&authuser=0&id="

log() { echo "[$(date '+%H:%M:%S')] $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

# ── Step 1 – Miniconda ────────────────────────────────────────────────────────
log "Step 1/8 – Miniconda"
if [ -x "$CONDA" ]; then
    log "  conda already installed at $CONDA_INSTALL_DIR, skipping."
else
    log "  Downloading Miniconda3..."
    wget -q --show-progress \
        "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" \
        -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$CONDA_INSTALL_DIR"
    rm /tmp/miniconda.sh
    log "  Miniconda installed."
fi

# Accept ToS for default channels (required by newer conda versions)
"$CONDA" tos accept --override-channels \
    --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
"$CONDA" tos accept --override-channels \
    --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

# Initialise conda in the current shell and in ~/.bashrc
if ! grep -q "miniconda3" "${HOME}/.bashrc" 2>/dev/null; then
    log "  Adding conda init to ~/.bashrc..."
    "$CONDA" init bash
fi
# shellcheck disable=SC1091
source "${CONDA_INSTALL_DIR}/etc/profile.d/conda.sh"

# ── Step 2 – SUSHI conda environment ─────────────────────────────────────────
log "Step 2/8 – SUSHI conda environment"
if "$CONDA" env list | grep -q "^SUSHI "; then
    log "  Environment 'SUSHI' already exists, skipping creation."
else
    log "  Creating environment from environment.yml (this takes ~5-10 min)..."
    "$CONDA" env create -f "${SCRIPT_DIR}/environment.yml"
    log "  Environment created."
fi

# ── Step 3 – fast-reid ────────────────────────────────────────────────────────
log "Step 3/8 – fast-reid"
FASTREID_DIR="${SCRIPT_DIR}/fast-reid"
if [ -d "$FASTREID_DIR/.git" ]; then
    log "  fast-reid already cloned, skipping."
else
    log "  Cloning fast-reid..."
    git clone https://github.com/JDAI-CV/fast-reid.git "$FASTREID_DIR"
    log "  Checking out pinned commit..."
    git -C "$FASTREID_DIR" checkout afe432b8c0ecd309db7921b7292b2c69813d0991
fi
log "  Installing fast-reid dependencies..."
"$PIP" install -q -r "${FASTREID_DIR}/docs/requirements.txt"
# libstdcxx-ng upgrade needed so pre-built lapsolver wheel (GLIBCXX_3.4.29) loads
"$CONDA" install -n SUSHI -c conda-forge libstdcxx-ng -y -q

# ── Step 4 – MOT20 dataset ────────────────────────────────────────────────────
log "Step 4/8 – MOT20 dataset"
MOT20_DIR="${DATA_PATH}/MOT20"
MOT20_TRAIN="${MOT20_DIR}/train"
MOT20_TEST="${MOT20_DIR}/test"
mkdir -p "${MOT20_DIR}/seqmaps" "$MOT20_TRAIN" "$MOT20_TEST"

if [ -d "${MOT20_TRAIN}/MOT20-01" ] && [ -d "${MOT20_TRAIN}/MOT20-05" ]; then
    log "  MOT20 already extracted, skipping download."
else
    MOT20_ZIP="/tmp/MOT20.zip"
    if [ ! -f "$MOT20_ZIP" ]; then
        log "  Downloading MOT20.zip (~4.7 GB)..."
        wget -q --show-progress -c \
            "https://motchallenge.net/data/MOT20.zip" \
            -O "$MOT20_ZIP"
    fi
    log "  Extracting MOT20.zip..."
    "$PYTHON" -c "
import zipfile
print('Extracting...')
with zipfile.ZipFile('${MOT20_ZIP}', 'r') as z:
    z.extractall('${DATA_PATH}/')
print('Done')
"
    log "  Extraction complete."
fi

# ── Step 5 – MOT20 detection files ───────────────────────────────────────────
log "Step 5/8 – MOT20 detection files (byte065 / aplift)"

# File IDs discovered from the Google Drive folder listing
# (https://drive.google.com/drive/folders/1bxw1Hz77LCCW3cWizhg_q03vk4aXUriz)
declare -A BYTE_IDS=(
    ["MOT20-01"]="1DQY2ChghNJS1ASByxmaMAW_axXen1kzf"
    ["MOT20-02"]="1N-A98wr9CTzShbcoyeoGU-wmywBYqs3S"
    ["MOT20-03"]="1wZ5ZF-vmJPH3x6UGToSsqziaPK55Au9a"
    ["MOT20-04"]="1V9CTgB9qj3wDwwo9pz76p5COoR55dWAw"
    ["MOT20-05"]="1MggUl4lb4_Yh0IrteTvB6Die9l9AKWrK"
    ["MOT20-06"]="1Sa74micmrNWxVj1KmRt6xEOnSWejaqHQ"
    ["MOT20-07"]="1PSo3TIjB01oTHmlozM-6-zF86xGKf0IN"
    ["MOT20-08"]="1kUaVZSEcw64ekXDczncu-fXBU5HO_dO-"
)
declare -A APLIFT_IDS=(
    ["MOT20-01"]="1TCne7k_8ER39QhOn-rSxbcQ1jwl_zXY5"
    ["MOT20-02"]="1p3ptWpET7iFY4EtmIZB884Cekd30oZfA"
    ["MOT20-03"]="1anjXEhIk2DvlkqZxCIq9YMHL-bthVmWS"
    ["MOT20-04"]="1IQO_sBTBehJupVImyrnV1zrcNpDWEMqr"
    ["MOT20-05"]="1kSg3TBHDuAt2tK5ZRKZM8LFix5eADxfz"
    ["MOT20-06"]="1A_pYFZIXMMtjHZ0lKDcbxFS8gODkWscj"
    ["MOT20-07"]="17R1af-URPhIg9v3wGlpGb1hYX09sKU4M"
    ["MOT20-08"]="1wscUJtmwv9n_T02pMI_nF8dB9FP0pXZm"
)

download_det_file() {
    local dest_file="$1"
    local file_id="$2"
    if [ -s "$dest_file" ]; then
        log "    $dest_file already exists, skipping."
        return
    fi
    curl -sL "${GD_BASE}${file_id}" -o "$dest_file"
    local lines
    lines=$(wc -l < "$dest_file")
    log "    Downloaded: $dest_file  ($lines lines)"
}

# Train sequences
for seq in MOT20-01 MOT20-02 MOT20-03 MOT20-05; do
    det_dir="${MOT20_TRAIN}/${seq}/det"
    mkdir -p "$det_dir"
    download_det_file "${det_dir}/byte065.txt" "${BYTE_IDS[$seq]}"
    download_det_file "${det_dir}/aplift.txt"  "${APLIFT_IDS[$seq]}"
done

# Test sequences
for seq in MOT20-04 MOT20-06 MOT20-07 MOT20-08; do
    det_dir="${MOT20_TEST}/${seq}/det"
    mkdir -p "$det_dir"
    download_det_file "${det_dir}/byte065.txt" "${BYTE_IDS[$seq]}"
    download_det_file "${det_dir}/aplift.txt"  "${APLIFT_IDS[$seq]}"
done

# ── Step 6 – ReID model weights ───────────────────────────────────────────────
log "Step 6/8 – ReID model weights"
REID_DIR="${SCRIPT_DIR}/fastreid-models/model_weights"
mkdir -p "$REID_DIR"
REID_FILE="${REID_DIR}/msmt_bot_R50-ibn.pth"
if [ -s "$REID_FILE" ]; then
    log "  ReID weights already present, skipping."
else
    log "  Downloading msmt_bot_R50-ibn.pth (~308 MB)..."
    "$GDOWN" "1MovixfOLwnnXet05JLIGPy-Ag67fdW-2" -O "$REID_FILE"
fi

# ── Step 7 – Pretrained SUSHI models ─────────────────────────────────────────
log "Step 7/8 – Pretrained SUSHI models"
MODELS_DIR="${SCRIPT_DIR}/fastreid-models/pretrained_models"
mkdir -p "$MODELS_DIR"

# File IDs from https://drive.google.com/drive/folders/1cU7LeTAeKxS-nvxqrUdUstNpR-Wpp5NV
declare -A MODEL_IDS=(
    ["mot20private.pth"]="1LAx7eWPh6fLWBpJFJwGxOxVnPvR25Zb5"
    ["mot20public.pth"]="112m_lUCkhDi2Wz9_-NRC2a4AnNZLjN5r"
    ["mot17private.pth"]="1GC8gJSotlBEb2BfkI274uNgUoFHhK-5A"
    ["mot17public.pth"]="1a5V4YlWm3KXABb3KPiBD36XRObB1Uqpc"
    ["model_configs.txt"]="1nHcZ0Ix-JExPGO8CSMOIJoVftMr8bWqZ"
)

for fname in "${!MODEL_IDS[@]}"; do
    dest="${MODELS_DIR}/${fname}"
    if [ -s "$dest" ]; then
        log "  $fname already present, skipping."
    else
        log "  Downloading $fname..."
        "$GDOWN" "${MODEL_IDS[$fname]}" -O "$dest"
    fi
done

# ── Step 8 – Seqmap files ─────────────────────────────────────────────────────
log "Step 8/8 – MOT20 seqmap files"
SEQMAP_DIR="${MOT20_DIR}/seqmaps"
mkdir -p "$SEQMAP_DIR"

write_seqmap() {
    local file="$1"; shift
    if [ -f "$file" ]; then return; fi
    printf "name\n" > "$file"
    for seq in "$@"; do printf "%s\n" "$seq" >> "$file"; done
    log "  Created $file"
}

write_seqmap "${SEQMAP_DIR}/mot20-train-all.txt"  MOT20-01 MOT20-02 MOT20-03 MOT20-05
write_seqmap "${SEQMAP_DIR}/mot20-test-all.txt"   MOT20-04 MOT20-06 MOT20-07 MOT20-08
write_seqmap "${SEQMAP_DIR}/mot20-val-split1.txt" MOT20-05
write_seqmap "${SEQMAP_DIR}/mot20-val-split2.txt" MOT20-03
write_seqmap "${SEQMAP_DIR}/mot20-val-split3.txt" MOT20-02
write_seqmap "${SEQMAP_DIR}/mot20-val-split4.txt" MOT20-01

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                    Setup Complete!                           ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Conda env : SUSHI  (activate with: conda activate SUSHI)   ║"
echo "║  Dataset   : ${DATA_PATH}/MOT20/"
echo "║  ReID model: fastreid-models/model_weights/msmt_bot_R50-ibn.pth"
echo "║  Pretrained: fastreid-models/pretrained_models/             ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  To train (private dets / ByteTrack):                       ║"
echo "║    conda activate SUSHI                                      ║"
echo "║    bash train_mot20.sh private ${DATA_PATH}"
echo "║                                                              ║"
echo "║  To train (public dets / APLift):                           ║"
echo "║    bash train_mot20.sh public ${DATA_PATH}"
echo "╚══════════════════════════════════════════════════════════════╝"
