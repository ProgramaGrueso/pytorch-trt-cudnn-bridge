"""
pytorch-trt-cudnn-bridge
~~~~~~~~~~~~~~~~~~~~~~~~

Bridge loader that enables TensorRT 8.6+ plugins (requiring cuDNN 8) to coexist seamlessly
with PyTorch 2.x (which bundles cuDNN 9) on Linux environments.
"""

from .loader import load_tensorrt, preload_private_cudnn, get_default_private_lib_dir

__version__ = "0.1.0"
__all__ = ["load_tensorrt", "preload_private_cudnn", "get_default_private_lib_dir"]
