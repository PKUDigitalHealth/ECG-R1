"""
ECG-R1 vLLM plugin.

Registers ECGR1ForConditionalGeneration via vLLM's plugin system.
"""

import os

_registered_processes = set()


def register():
    """Register ECGR1ForConditionalGeneration to vLLM ModelRegistry."""
    pid = os.getpid()
    print(f"[ECG-R1 Plugin] register() called in process {pid}", flush=True)
    
    if pid in _registered_processes:
        print(f"[ECG-R1 Plugin] Already registered in process {pid}", flush=True)
        return
    
    try:
        from vllm.model_executor.models import ModelRegistry
        from .ecg_r1_model import ECGR1ForConditionalGeneration
        
        ModelRegistry.register_model(
            "ECGR1ForConditionalGeneration",
            ECGR1ForConditionalGeneration
        )
        
        _registered_processes.add(pid)
        print(f"[ECG-R1 Plugin] ECGR1ForConditionalGeneration registered in process {pid}", flush=True)
        
    except Exception as e:
        print(f"[ECG-R1 Plugin] Failed to register: {e}", flush=True)
        import traceback
        traceback.print_exc()


def get_model_class():
    """Get model class (for direct import)"""
    from .ecg_r1_model import ECGR1ForConditionalGeneration
    return ECGR1ForConditionalGeneration
