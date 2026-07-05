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
#
# Pip installs must use the same Python as ComfyUI (not system pip):
#   COMFYUI_PYTHON=/workspace/ComfyUI/venv/bin/python bash install_comfyui_custom_nodes.sh
# Windows portable (from this folder):
#   COMFYUI_PYTHON=../../python_embeded/python.exe bash install_comfyui_custom_nodes.sh

set -euo pipefail

PACK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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

PYTHON_BIN="${COMFYUI_PYTHON:-python3}"

# kornia>=0.8 dropped re-export of pad from kornia.geometry.transform.pyramid;
# LTXVideo still imports it there — use torch.nn.functional.pad instead.
patch_ltxvideo_kornia_compat() {
  local pyramid="${NODES_DIR}/ComfyUI-LTXVideo/pyramid_blending.py"
  [[ -f "${pyramid}" ]] || return 0
  echo "==> ComfyUI-LTXVideo kornia compat (${pyramid})"
  "${PYTHON_BIN}" - "${pyramid}" <<'PY'
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
if "from torch.nn.functional import pad" in text:
    print("  already patched")
    raise SystemExit(0)
if "pad," not in text or "kornia.geometry.transform.pyramid" not in text:
    print("  warning: unexpected pyramid_blending.py; manual fix may be needed", file=sys.stderr)
    raise SystemExit(0)
text = re.sub(r"^[ \t]*pad,\n", "", text, count=1, flags=re.MULTILINE)
text = text.replace(
    "from torch import Tensor\n",
    "from torch import Tensor\nfrom torch.nn.functional import pad\n",
    1,
)
path.write_text(text, encoding="utf-8")
print("  patched ok")
PY
}

echo "ComfyUI-Coomfy pack: ${PACK_DIR}"
echo "ComfyUI: ${COMFYUI_DIR}"
echo "custom_nodes: ${NODES_DIR}"

# --- Photo Lab + Video Lab (full stack) ---
# This pack is already here (you are running from it).
echo "==> ComfyUI-Coomfy already installed at ${PACK_DIR}"
clone_or_update "rgthree-comfy" "https://github.com/rgthree/rgthree-comfy.git"
clone_or_update "ComfyUI-KJNodes" "https://github.com/kijai/ComfyUI-KJNodes.git"
clone_or_update "comfyui_controlnet_aux" "https://github.com/Fannovel16/comfyui_controlnet_aux.git"
clone_or_update "ComfyUI-LTXVideo" "https://github.com/kijai/ComfyUI-LTXVideo.git"
clone_or_update "ComfyUI-LTXV-TimeGated-LoRA" "https://github.com/Jinx138/ComfyUI-LTXV-TimeGated-LoRA.git"
clone_or_update "10S-Comfy-nodes" "https://github.com/TenStrip/10S-Comfy-nodes.git"
clone_or_update "ComfyUI-Custom-Scripts" "https://github.com/pythongosssss/ComfyUI-Custom-Scripts.git"
clone_or_update "ComfyUI-Easy-Use" "https://github.com/yolain/ComfyUI-Easy-Use.git"
clone_or_update "ComfyLiterals" "https://github.com/M1kep/ComfyLiterals.git"
clone_or_update "RES4LYF" "https://github.com/ClownsharkBatwing/RES4LYF.git"
clone_or_update "ComfyUI-Impact-Pack" "https://github.com/ltdrdata/ComfyUI-Impact-Pack.git"
clone_or_update "ComfyUI-Impact-Subpack" "https://github.com/ltdrdata/ComfyUI-Impact-Subpack.git"
clone_or_update "ComfyUI-Anima-LLLite" "https://github.com/kohya-ss/ComfyUI-Anima-LLLite.git"
clone_or_update "ComfyUI-MultiLoRALoader" "https://github.com/phazei/ComfyUI-MultiLoRALoader.git"
# Folder must be ComfyUI-Crystools (not comfyui-crystools) for web extension paths.
clone_or_update "ComfyUI-Crystools" "https://github.com/crystian/ComfyUI-Crystools.git"
clone_or_update "ComfyUI-PromptRelay" "https://github.com/kijai/ComfyUI-PromptRelay.git"
clone_or_update "ComfyUI-VFI" "https://github.com/GACLove/ComfyUI-VFI.git"
clone_or_update "Nvidia_RTX_Nodes_ComfyUI" "https://github.com/Comfy-Org/Nvidia_RTX_Nodes_ComfyUI.git"

patch_ltxvideo_kornia_compat

echo "--- pip requirements (when present) ---"
for req in "${NODES_DIR}"/*/requirements.txt; do
  [[ -f "${req}" ]] || continue
  echo "==> ${PYTHON_BIN} -m pip install -r ${req}"
  "${PYTHON_BIN}" -m pip install -r "${req}"
done

cat <<EOF

Done. Restart ComfyUI, then in Coomfy Settings set:
  COMFYUI_PHOTO_BASE_URL = http://<gpu-host>:8188
  COMFYUI_VIDEO_BASE_URL = http://<gpu-host>:8188

If pip installs failed, re-run with the Python ComfyUI actually uses:
  COMFYUI_PYTHON=/path/to/ComfyUI/venv/bin/python bash install_comfyui_custom_nodes.sh
  (Windows portable: COMFYUI_PYTHON=../../python_embeded/python.exe)

Re-run this script anytime to update packs (git pull --ff-only).
EOF
