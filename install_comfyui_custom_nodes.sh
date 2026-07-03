#!/usr/bin/env bash
# Install ComfyUI custom nodes required by Coomfy Photo Lab + Video Lab.
#
# Run from this pack after cloning into custom_nodes:
#   cd /workspace/ComfyUI/custom_nodes/ComfyUI-Coomfy
#   bash install_comfyui_custom_nodes.sh
#
# ComfyUI root is auto-detected (../.. from this folder). Override:
#   bash install_comfyui_custom_nodes.sh /path/to/ComfyUI
#   COMFYUI_PATH=/workspace/ComfyUI bash install_comfyui_custom_nodes.sh

set -euo pipefail

PACK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_OPTIONAL="${INSTALL_OPTIONAL:-0}"

resolve_comfyui_dir() {
  if [[ -n "${1:-}" ]]; then
    echo "$1"
    return
  fi
  if [[ -n "${COMFYUI_PATH:-}" ]]; then
    echo "${COMFYUI_PATH}"
    return
  fi
  local guessed
  guessed="$(cd "${PACK_DIR}/../.." && pwd)"
  if [[ -f "${guessed}/main.py" ]] || [[ -d "${guessed}/models" ]]; then
    echo "${guessed}"
    return
  fi
  echo ""
}

COMFYUI_DIR="$(resolve_comfyui_dir "${1:-}")"
if [[ -z "${COMFYUI_DIR}" ]]; then
  echo "Usage: $0 [/path/to/ComfyUI]" >&2
  echo "   or: COMFYUI_PATH=/path/to/ComfyUI $0" >&2
  echo "When run from custom_nodes/ComfyUI-Coomfy, ComfyUI root is inferred as ../.." >&2
  exit 1
fi

if [[ ! -d "${COMFYUI_DIR}" ]]; then
  echo "ComfyUI directory not found: ${COMFYUI_DIR}" >&2
  exit 1
fi

NODES_DIR="${COMFYUI_DIR}/custom_nodes"
mkdir -p "${NODES_DIR}"

clone_or_update() {
  local name="$1"
  local url="$2"
  local dest="${NODES_DIR}/${name}"
  if [[ -d "${dest}/.git" ]]; then
    echo "==> updating ${name}"
    git -C "${dest}" pull --ff-only
  elif [[ -d "${dest}" ]]; then
    echo "==> skip ${name} (exists but is not a git repo): ${dest}" >&2
  else
    echo "==> cloning ${name}"
    git clone --depth 1 "${url}" "${dest}"
  fi
}

echo "ComfyUI-Coomfy pack: ${PACK_DIR}"
echo "ComfyUI: ${COMFYUI_DIR}"
echo "custom_nodes: ${NODES_DIR}"

# --- required for Photo Lab + Video Lab ---
# This pack is already here (you are running from it).
echo "==> ComfyUI-Coomfy already installed at ${PACK_DIR}"
clone_or_update "rgthree-comfy" "https://github.com/rgthree/rgthree-comfy.git"
clone_or_update "ComfyUI-KJNodes" "https://github.com/kijai/ComfyUI-KJNodes.git"
clone_or_update "ComfyUI-LTXVideo" "https://github.com/kijai/ComfyUI-LTXVideo.git"
clone_or_update "10S-Comfy-nodes" "https://github.com/TenStrip/10S-Comfy-nodes.git"
clone_or_update "ComfyUI-Custom-Scripts" "https://github.com/pythongosssss/ComfyUI-Custom-Scripts.git"
clone_or_update "ComfyUI-Easy-Use" "https://github.com/yolain/ComfyUI-Easy-Use.git"
clone_or_update "ComfyUI-Impact-Pack" "https://github.com/ltdrdata/ComfyUI-Impact-Pack.git"
clone_or_update "ComfyUI-Anima-LLLite" "https://github.com/kohya-ss/ComfyUI-Anima-LLLite.git"
clone_or_update "ComfyUI-MultiLoRALoader" "https://github.com/phazei/ComfyUI-MultiLoRALoader.git"

if [[ "${INSTALL_OPTIONAL}" == "1" ]]; then
  echo "--- optional nodes (timed phases, RIFE, RTX VSR) ---"
  clone_or_update "ComfyUI-PromptRelay" "https://github.com/kijai/ComfyUI-PromptRelay.git"
  clone_or_update "ComfyUI-VFI" "https://github.com/GACLove/ComfyUI-VFI.git"
  clone_or_update "Nvidia_RTX_Nodes_ComfyUI" "https://github.com/Comfy-Org/Nvidia_RTX_Nodes_ComfyUI.git"
fi

echo "--- pip requirements (when present) ---"
PYTHON_BIN="${COMFYUI_PYTHON:-python3}"
for req in "${NODES_DIR}"/*/requirements.txt; do
  [[ -f "${req}" ]] || continue
  echo "==> ${PYTHON_BIN} -m pip install -r ${req}"
  "${PYTHON_BIN}" -m pip install -r "${req}"
done

cat <<EOF

Done. Restart ComfyUI, then in Coomfy Settings set:
  COMFYUI_PHOTO_BASE_URL = http://<gpu-host>:8188
  COMFYUI_VIDEO_BASE_URL = http://<gpu-host>:8188

Video export needs ffmpeg on the ComfyUI host PATH.
Re-run this script anytime to update packs (git pull --ff-only).

Optional nodes:
  INSTALL_OPTIONAL=1 bash install_comfyui_custom_nodes.sh ${COMFYUI_DIR}
EOF
