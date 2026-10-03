"""
trt_bridge.loader
~~~~~~~~~~~~~~~~~

Solves the runtime dynamic linker conflict between PyTorch 2.x (which bundles cuDNN 9)
and TensorRT 8.6 plugin libraries (libnvinfer_plugin.so.8, which expects cuDNN 8 symbols).

By preloading private cuDNN 8 libraries with ctypes.RTLD_GLOBAL before importing
tensorrt or tensorrt_bindings, plugins such as InstanceNormalization_TRT can resolve
their required cuDNN 8 symbols without modifying system libraries (/usr/lib) or
interfering with PyTorch's native CUDA/cuDNN 9 context.
"""

import ctypes
import os
import sys
from typing import List, Optional, Tuple

_LIB_HANDLES: List[ctypes.CDLL] = []
_TENSORRT_MODULE = None


def get_default_private_lib_dir() -> str:
    """Returns the default directory for private cuDNN 8 libraries."""
    env_dir = os.environ.get("TRT_PRIVATE_CUDNN_DIR")
    if env_dir and os.path.isdir(env_dir):
        return env_dir
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, "private_cudnn8")


def preload_private_cudnn(private_lib_dir: Optional[str] = None) -> List[str]:
    """
    Preloads cuDNN 8 shared objects using ctypes.RTLD_GLOBAL so that
    libnvinfer_plugin.so.8 finds its dependencies cleanly.

    Args:
        private_lib_dir: Directory containing libcudnn.so.8, libcudnn_ops_infer.so.8,
                         and libcudnn_cnn_infer.so.8.

    Returns:
        List of successfully preloaded library paths.
    """
    global _LIB_HANDLES
    if private_lib_dir is None:
        private_lib_dir = get_default_private_lib_dir()

    required_libs = [
        "libcudnn_ops_infer.so.8",
        "libcudnn_cnn_infer.so.8",
        "libcudnn.so.8",
    ]

    loaded = []
    if os.path.isdir(private_lib_dir):
        for lib_name in required_libs:
            lib_path = os.path.join(private_lib_dir, lib_name)
            if os.path.exists(lib_path):
                try:
                    handle = ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
                    _LIB_HANDLES.append(handle)
                    loaded.append(lib_path)
                except OSError as err:
                    sys.stderr.write(f"[trt_bridge] Warning: Failed loading {lib_path}: {err}\n")
    return loaded


def load_tensorrt(private_lib_dir: Optional[str] = None):
    """
    Safely preloads cuDNN 8 and imports the TensorRT Python bindings module.

    Returns:
        The imported tensorrt module.

    Raises:
        RuntimeError: If TensorRT bindings cannot be imported.
    """
    global _TENSORRT_MODULE
    if _TENSORRT_MODULE is not None:
        return _TENSORRT_MODULE

    preload_private_cudnn(private_lib_dir)

    try:
        import tensorrt as trt
    except ImportError:
        try:
            import tensorrt_bindings as trt
        except ImportError as exc:
            raise RuntimeError(
                "[trt_bridge] Could not import tensorrt or tensorrt_bindings. "
                "Ensure tensorrt 8.x is installed in your Python environment."
            ) from exc

    _TENSORRT_MODULE = trt
    return trt
