#!/usr/bin/env python3
"""
Example demonstrating how to safely use PyTorch (cuDNN 9) and TensorRT 8.6 plugins
in the same process without dynamic linker crashes.
"""

import sys

def main():
    print("=== PyTorch + TensorRT cuDNN Bridge Demo ===")

    # Step 1: Import PyTorch first (typically initializes cuDNN 9)
    try:
        import torch
        print(f"[✓] PyTorch version: {torch.__version__}")
        print(f"[✓] PyTorch cuDNN version: {torch.backends.cudnn.version() if torch.cuda.is_available() else 'CPU mode'}")
    except ImportError:
        print("[!] PyTorch is not installed in this environment.")

    # Step 2: Use trt_bridge to preload cuDNN 8 and import TensorRT
    try:
        from trt_bridge import load_tensorrt, preload_private_cudnn
        
        preloaded = preload_private_cudnn()
        print(f"[✓] Preloaded private cuDNN libraries: {len(preloaded)} files")
        for lib in preloaded:
            print(f"    -> {lib}")

        trt = load_tensorrt()
        print(f"[✓] Successfully loaded TensorRT: version {trt.__version__}")
        print("[✓] TensorRT plugins can now execute without symbol lookup failures.")
    except Exception as e:
        print(f"[!] Info/Notice: {e}")

if __name__ == "__main__":
    main()
