#!/usr/bin/env bash
# setup_private_cudnn8.sh
# Downloads/extracts cuDNN 8.9.7 runtime libraries into a private directory
# so TensorRT 8.6 plugins can link against them without interfering with PyTorch / system cuDNN 9.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TARGET_DIR="${PROJECT_ROOT}/private_cudnn8"

echo "==> Setting up private cuDNN 8 runtime in: ${TARGET_DIR}"
mkdir -p "${TARGET_DIR}"

# If already populated
if [[ -f "${TARGET_DIR}/libcudnn.so.8" && -f "${TARGET_DIR}/libcudnn_ops_infer.so.8" && -f "${TARGET_DIR}/libcudnn_cnn_infer.so.8" ]]; then
    echo "cuDNN 8 libraries are already present in ${TARGET_DIR}."
    ls -lh "${TARGET_DIR}"
    exit 0
fi

# Attempt to copy from local cache or system if available
FOUND_SYS=0
for sys_dir in /usr/lib /usr/local/cuda/lib64 "$HOME/.local/lib"; do
    if [[ -f "${sys_dir}/libcudnn.so.8" ]]; then
        echo "Found system cuDNN 8 in ${sys_dir}, copying relevant libraries..."
        cp -a "${sys_dir}"/libcudnn*.so.8* "${TARGET_DIR}/" || true
        FOUND_SYS=1
        break
    fi
done

if [[ $FOUND_SYS -eq 1 ]]; then
    echo "Successfully extracted from local system."
    ls -lh "${TARGET_DIR}"
    exit 0
fi

# Otherwise, instruct how to extract from pip or nvidia archive
echo "----------------------------------------------------------------------------------"
echo "Option 1: Extract from Nvidia cuDNN pip wheel (no root required):"
echo "  pip install nvidia-cudnn-cu12==8.9.7.29 --target /tmp/cudnn8_wheel"
echo "  cp -a /tmp/cudnn8_wheel/nvidia/cudnn/lib/libcudnn*.so.8* ${TARGET_DIR}/"
echo "  rm -rf /tmp/cudnn8_wheel"
echo ""
echo "Option 2: Point TRT_PRIVATE_CUDNN_DIR environment variable to your existing folder:"
echo "  export TRT_PRIVATE_CUDNN_DIR=/path/to/your/cudnn8/libs"
echo "----------------------------------------------------------------------------------"
