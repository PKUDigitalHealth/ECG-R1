from setuptools import setup, find_packages
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

setup(
    name="ecg_r1",
    version="0.1.0",
    description="ECG-R1: ECG modality support for Qwen3VL in vLLM",
    author="ECG-R1 Team",
    packages=["ecg_r1", "ecg_r1.vllm_plugin"],
    package_dir={
        "ecg_r1": ".",
        "ecg_r1.vllm_plugin": "./vllm_plugin",
    },
    python_requires=">=3.10",
    install_requires=[
        "torch",
    ],
    entry_points={
        "vllm.general_plugins": [
            "ecg_r1 = ecg_r1.vllm_plugin:register",
        ],
    },
)
