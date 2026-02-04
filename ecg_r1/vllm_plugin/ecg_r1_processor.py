import os
import sys
from typing import Any, Mapping, Optional, Sequence

import torch
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from vllm.multimodal import MultiModalDataDict
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import (
    MultiModalDataItems, 
    MultiModalDataParser,
    ModalityDataItems,
    # Note: Do not import EmbeddingItems! ECGDataItems inherits from ModalityDataItems
)
from vllm.multimodal.processing import PromptUpdate, PromptReplacement
from transformers import BatchFeature

from vllm.model_executor.models.qwen3_vl import (
    Qwen3VLProcessingInfo,
    Qwen3VLDummyInputsBuilder,
    Qwen3VLMultiModalProcessor,
)


class ECGDataItems(ModalityDataItems):    
    def __init__(self, data: torch.Tensor):
        self.data = data
        self._modality = "ecg"
    
    @property
    def modality(self) -> str:
        return self._modality
    
    def get_count(self) -> int:
        """Return number of ECG data items"""
        return self.data.shape[0] if self.data.ndim >= 1 else 1
    
    def get(self, index: int) -> torch.Tensor:
        """Get ECG data at specified index"""
        return self.data[index]
    
    def get_processor_data(self) -> Mapping[str, Any]:
        """Return data for HF Processor - ECG doesn't need this"""
        return {}
    
    def get_passthrough_data(self) -> Mapping[str, Any]:
        """Return data to be passed through to the model"""
        return {"ecg_embeds": self.data}


class ECGR1DataParser(MultiModalDataParser):
    def _parse_ecg_data(self, data) -> Optional[ModalityDataItems]:
        """Parse ECG data"""
        if data is None:
            return None
        
        if isinstance(data, (list, tuple)) and len(data) == 0:
            return None
        
        if isinstance(data, torch.Tensor):
            if data.ndim == 2:
                data = data.unsqueeze(0)
            return ECGDataItems(data)
        
        if isinstance(data, list) and len(data) > 0:
            if isinstance(data[0], torch.Tensor):
                stacked = torch.stack([t if t.dim() == 2 else t.squeeze(0) for t in data])
                return ECGDataItems(stacked)
        
        try:
            tensor_data = torch.tensor(data)
            if tensor_data.ndim == 2:
                tensor_data = tensor_data.unsqueeze(0)
            return ECGDataItems(tensor_data)
        except Exception:
            return None
    
    def _get_subparsers(self) -> Mapping[str, Any]:
        """Extend subparsers to add ECG support"""
        subparsers = dict(super()._get_subparsers())
        subparsers["ecg"] = self._parse_ecg_data
        return subparsers


ECG_PLACEHOLDER = "<|ecg_pad|>"
ECG_START_TOKEN = "<|ecg_start|>"
ECG_END_TOKEN = "<|ecg_end|>"

def get_env_args(key: str, dtype: type, default: Any) -> Any:
    """Get configuration from environment variables"""
    val = os.environ.get(key)
    if val is None:
        return default
    if dtype == bool:
        return val.lower() in ('true', '1', 'yes')
    return dtype(val)


class ECGR1ProcessingInfo(Qwen3VLProcessingInfo):
    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        """Return count limits for each modality"""
        limits = super().get_supported_mm_limits()
        limits["ecg"] = 1
        return limits
    
    def get_ecg_num_tokens(self) -> int:
        """Get number of tokens for ECG modality"""
        ecg_seq_length = get_env_args('ECG_SEQ_LENGTH', int, 5000)
        ecg_patch_size = get_env_args('ECG_PATCH_SIZE', int, 50)
        return ecg_seq_length // ecg_patch_size + 1


class ECGR1DummyInputsBuilder(Qwen3VLDummyInputsBuilder):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        """Generate dummy text containing ECG placeholder"""
        text = super().get_dummy_text(mm_counts)
        
        num_ecgs = mm_counts.get("ecg", 0)
        if num_ecgs > 0:
            ecg_token = f"{ECG_START_TOKEN}{ECG_PLACEHOLDER}{ECG_END_TOKEN}"
            text = text + ecg_token * num_ecgs
        
        return text
    
    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> MultiModalDataDict:
        """Generate dummy multi-modal data"""
        data = super().get_dummy_mm_data(seq_len, mm_counts)
        
        num_ecgs = mm_counts.get("ecg", 0)
        if num_ecgs > 0:
            ecg_seq_length = get_env_args('ECG_SEQ_LENGTH', int, 5000)
            data["ecg"] = [torch.zeros(12, ecg_seq_length) for _ in range(num_ecgs)]
        
        return data


class ECGR1MultiModalProcessor(Qwen3VLMultiModalProcessor):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_mm_kwargs: Mapping[str, object] = {}
    
    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> bool:
        return True
    
    def _get_data_parser(self) -> MultiModalDataParser:
        """Return data parser supporting ECG"""
        return ECGR1DataParser(video_needs_metadata=True)
    
    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        mm_data = dict(mm_data)
        
        ecg_seq_length = get_env_args('ECG_SEQ_LENGTH', int, 5000)
        ecg_patch_size = get_env_args('ECG_PATCH_SIZE', int, 50)
        tokens_per_ecg = ecg_seq_length // ecg_patch_size + 1  # +1 for cls token
        
        ecg_pattern = f"{ECG_START_TOKEN}{ECG_PLACEHOLDER}{ECG_END_TOKEN}"
        has_ecg_placeholder = ecg_pattern in prompt
        
        if has_ecg_placeholder:
            ecg_expanded = f"{ECG_START_TOKEN}" + ECG_PLACEHOLDER * tokens_per_ecg + f"{ECG_END_TOKEN}"
            
            while ecg_pattern in prompt:
                prompt = prompt.replace(ecg_pattern, ecg_expanded, 1)
        
        processed = super()._call_hf_processor(
            prompt=prompt,
            mm_data=mm_data,
            mm_kwargs=mm_kwargs,
            tok_kwargs=tok_kwargs,
        )
        
        return processed
    
    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        """Get multi-modal field config, add ECG field"""
        config = super()._get_mm_fields_config(hf_inputs, hf_processor_mm_kwargs)
        config = dict(config)  # Convert to mutable dict
        
        if "ecg_embeds" in hf_inputs:
            config["ecg_embeds"] = MultiModalFieldConfig.batched("ecg")
        
        return config
    
    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        """Get prompt update rules, add ECG placeholder handling.

        Pattern matches the expanded ECG placeholder created in _call_hf_processor.
        """
        updates = list(super()._get_prompt_updates(
            mm_items, hf_processor_mm_kwargs, out_mm_kwargs
        ))
        
        ecg_seq_length = get_env_args('ECG_SEQ_LENGTH', int, 5000)
        ecg_patch_size = get_env_args('ECG_PATCH_SIZE', int, 50)
        tokens_per_ecg = ecg_seq_length // ecg_patch_size + 1  # +1 for cls token
        
        tokenizer = self.info.get_tokenizer()
        ecg_token_id = tokenizer.convert_tokens_to_ids(ECG_PLACEHOLDER)
        
        def get_ecg_replacement(item_idx: int):
            """Replace single ECG placeholder with multiple tokens"""
            return [ecg_token_id] * tokens_per_ecg
        
        ecg_expanded_pattern = f"{ECG_START_TOKEN}" + ECG_PLACEHOLDER * tokens_per_ecg + f"{ECG_END_TOKEN}"
        
        updates.append(
            PromptReplacement(
                modality="ecg",
                target=ecg_expanded_pattern,
                replacement=get_ecg_replacement,
            )
        )
        
        return updates
