"""
MindSpore GPU Configuration
This module configures MindSpore to use GPU acceleration by default.
Import this module at the start of your application.
"""
import mindspore as ms
import os

def configure_gpu():
    """
    Configure MindSpore to use GPU device.
    Falls back to CPU if GPU is not available.
    """
    try:
        if hasattr(ms, 'set_device'):
            ms.set_device("GPU", 0)
        else:
            ms.set_context(device_target="GPU", device_id=0)
        
        print(f"✓ MindSpore {ms.__version__} configured for GPU")
        print(f"✓ Device Target: {ms.get_context('device_target')}")
    except Exception as e:
        print(f"⚠ GPU not available, falling back to CPU: {e}")
        ms.set_context(device_target="CPU")

configure_gpu()
