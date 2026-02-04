"""
ECG-R1 vLLM implementation

Inherits from Qwen3VLForConditionalGeneration, adding ECG modality support.
"""

import os
import re
from typing import Any, Iterable, Mapping, Optional, Union

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.qwen3_vl import (
    Qwen3VLForConditionalGeneration,
    Qwen2_5_VLImageInputs,
    Qwen2_5_VLVideoInputs,
)
from vllm.model_executor.models.utils import WeightsMapper, maybe_prefix
from vllm.model_executor.models.interfaces import MultiModalEmbeddings
from vllm.model_executor.models.utils import merge_multimodal_embeddings
from vllm.multimodal import MULTIMODAL_REGISTRY

from .ecg_r1_processor import (
    ECGR1ProcessingInfo,
    ECGR1DummyInputsBuilder,
    ECGR1MultiModalProcessor,
)

logger = init_logger(__name__)


def build_ecg_tower(ecg_tower_path: str, model_config_name: str = 'coca_ViT-B-32', device: str = 'cpu'):
    """Build ECG Tower (structure must match training)"""
    from ecg_coca.training import get_ecg_encoder
    ecg_tower, ecg_processor, ecg_config = get_ecg_encoder(
        model_name=model_config_name,
        checkpoint_path=ecg_tower_path,
        device=device
    )
    return ecg_tower, ecg_config


def build_ecg_projector(ecg_hidden_size: int, llm_hidden_size: int, projector_type: str = 'mlp2x_gelu'):
    """Build ECG Projector (structure must match training)"""
    if projector_type == 'linear':
        return nn.Linear(ecg_hidden_size, llm_hidden_size)
    
    match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if match:
        mlp_depth = int(match.group(1))
        modules = [nn.Linear(ecg_hidden_size, llm_hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(llm_hidden_size, llm_hidden_size))
        return nn.Sequential(*modules)
    
    raise ValueError(f'Unknown projector type: {projector_type}')


class ECGR1EmbeddingInputs:
    """ECG embedding input type"""
    def __init__(self, ecg_embeds: torch.Tensor):
        self.type = "ecg_embeds"
        self.ecg_embeds = ecg_embeds


@MULTIMODAL_REGISTRY.register_processor(
    ECGR1MultiModalProcessor,
    info=ECGR1ProcessingInfo,
    dummy_inputs=ECGR1DummyInputsBuilder,
)
class ECGR1ForConditionalGeneration(Qwen3VLForConditionalGeneration):
    """
    ECG-R1 vLLM implementation
    
    Inherits from Qwen3VLForConditionalGeneration, adding:
    - ecg_tower: ECG encoder
    - ecg_projector: ECG -> LLM dimension mapping
    - Multi-modal processing for ECG modality
    """
    
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.visual.": "visual.",
            "lm_head.": "language_model.lm_head.",
            "model.language_model.": "language_model.model.",
            "model.ecg_tower.": "ecg_tower.",
            "model.ecg_projector.": "ecg_projector.",
        }
    )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> Optional[str]:
        """Get placeholder string for modality"""
        if modality.startswith("image"):
            return "<|vision_start|><|image_pad|><|vision_end|>"
        if modality.startswith("video"):
            return "<|vision_start|><|video_pad|><|vision_end|>"
        if modality.startswith("ecg"):
            return "<|ecg_start|><|ecg_pad|><|ecg_end|>"
        
        raise ValueError(f"Unsupported modality: {modality}")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model"):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        
        config = vllm_config.model_config.hf_config
        
        self._init_ecg_components(config)
        
        self.ecg_token_id = getattr(config, 'ecg_token_id', None)
        
        logger.info(f"[ECG-R1] Initialized with ecg_token_id={self.ecg_token_id}")

    def _init_ecg_components(self, config):
        """Initialize ECG tower and projector"""
        ecg_tower_path = getattr(config, 'ecg_tower_path', None) or os.environ.get('ECG_TOWER_PATH')
        ecg_projector_type = getattr(config, 'ecg_projector_type', None) or os.environ.get('ECG_PROJECTOR_TYPE', 'mlp2x_gelu')
        ecg_model_config = getattr(config, 'ecg_model_config', None) or os.environ.get('ECG_MODEL_CONFIG', 'coca_ViT-B-32')
        
        llm_hidden_size = getattr(config, 'hidden_size', None)
        if llm_hidden_size is None and hasattr(config, 'text_config'):
            llm_hidden_size = getattr(config.text_config, 'hidden_size', None)
        
        print(f"[ECG-R1] _init_ecg_components called in process {os.getpid()}", flush=True)
        print(f"[ECG-R1] ecg_tower_path={ecg_tower_path}", flush=True)
        
        if ecg_tower_path and llm_hidden_size:
            try:
                ecg_tower, ecg_cfg = build_ecg_tower(ecg_tower_path, ecg_model_config, device='cpu')
                ecg_hidden_size = ecg_cfg.get('ecg_cfg', {}).get('width', 768)
                
                ecg_projector = build_ecg_projector(ecg_hidden_size, llm_hidden_size, ecg_projector_type)
                
                self.ecg_tower = ecg_tower
                self.ecg_projector = ecg_projector
                
                self.ecg_hidden_size = ecg_hidden_size
                self.llm_hidden_size = llm_hidden_size
                
                print(f"[ECG-R1] ECG components attached: ecg_hidden={ecg_hidden_size}, llm_hidden={llm_hidden_size}", flush=True)
                
            except Exception as e:
                logger.error(f"[ECG-R1] Failed to initialize ECG components: {e}")
                self.ecg_tower = None
                self.ecg_projector = None
        else:
            logger.warning(f"[ECG-R1] ECG components not initialized: ecg_tower_path={ecg_tower_path}, llm_hidden_size={llm_hidden_size}")
            self.ecg_tower = None
            self.ecg_projector = None

    def _parse_and_validate_ecg_input(self, **kwargs) -> Optional[ECGR1EmbeddingInputs]:
        """
        Parse ECG input
        
        Input format: ecg_embeds is raw ECG signal, shape (batch, 12, 5000)
        """
        ecg_embeds = kwargs.pop("ecg_embeds", None)
        
        if ecg_embeds is None:
            return None
        
        if isinstance(ecg_embeds, torch.Tensor):
            while ecg_embeds.ndim > 3:
                ecg_embeds = ecg_embeds.squeeze(0)
            if ecg_embeds.ndim == 2:
                ecg_embeds = ecg_embeds.unsqueeze(0)
        elif isinstance(ecg_embeds, list):
            ecg_embeds = torch.stack([e.squeeze(0) if e.ndim > 2 else e for e in ecg_embeds])
        
        print(f"[ECG-R1] ECG input shape: {ecg_embeds.shape}", flush=True)
        return ECGR1EmbeddingInputs(ecg_embeds=ecg_embeds)

    def _parse_and_validate_multimodal_inputs(self, **kwargs) -> dict[str, Any]:
        """Parse all multi-modal inputs (extend parent method, add ECG)"""
        mm_input_by_modality = {}
        
        input_keys = list(kwargs.keys())
        
        for input_key in input_keys:
            if input_key in ("pixel_values", "image_embeds") and "image" not in mm_input_by_modality:
                mm_input_by_modality["image"] = self._parse_and_validate_image_input(**kwargs)
            if input_key in ("pixel_values_videos", "video_embeds") and "video" not in mm_input_by_modality:
                mm_input_by_modality["video"] = self._parse_and_validate_video_input(**kwargs)
            if input_key == "ecg_embeds" and "ecg" not in mm_input_by_modality:
                mm_input_by_modality["ecg"] = self._parse_and_validate_ecg_input(**kwargs)
        
        mm_input_by_modality = {k: v for k, v in mm_input_by_modality.items() if v is not None}
        
        return mm_input_by_modality

    def _process_ecg_input(self, ecg_input: ECGR1EmbeddingInputs) -> tuple[torch.Tensor, ...]:
        """
        Process ECG input, return embeddings
        
        Output format matches vLLM's multimodal contract: a tuple of 2D tensors.
        """
        ecg_features = ecg_input.ecg_embeds  # (batch, 12, 5000)
        batch_size = ecg_features.shape[0]
        
        if self.ecg_tower is None or self.ecg_projector is None:
            raise RuntimeError("ECG tower/projector not initialized")
        
        target_device = ecg_features.device
        target_dtype = ecg_features.dtype
        
        tower_device = next(self.ecg_tower.parameters()).device
        if tower_device != target_device:
            print(f"[ECG-R1] Moving ECG components: {tower_device} -> {target_device}", flush=True)
            self.ecg_tower = self.ecg_tower.to(device=target_device, dtype=target_dtype)
            self.ecg_projector = self.ecg_projector.to(device=target_device, dtype=target_dtype)
        
        print(f"[ECG-R1] Processing: input={ecg_features.shape}, device={target_device}", flush=True)
        
        ecg_embeds = self.ecg_tower(ecg_features, output_last_transformer_layer=True)
        ecg_embeds = self.ecg_projector(ecg_embeds)
        
        print(f"[ECG-R1] After tower+projector: {ecg_embeds.shape}", flush=True)
        
        num_tokens_per_ecg = ecg_embeds.shape[1]  # 101
        ecg_embeds_flat = ecg_embeds.reshape(-1, ecg_embeds.shape[-1])  # (batch*101, 4096)
        sizes = [num_tokens_per_ecg] * batch_size
        result = ecg_embeds_flat.split(sizes)  # tuple of (101, 4096)
        
        print(f"[ECG-R1] Output: {len(result)} items, shape={result[0].shape}", flush=True)
        return result

    def get_multimodal_embeddings(self, **kwargs) -> Optional[MultiModalEmbeddings]:
        """Get all multi-modal embeddings (extend parent method, add ECG)"""
        print(f"[ECG-R1 get_multimodal_embeddings] kwargs keys: {list(kwargs.keys())}", flush=True)
        
        mm_input_by_modality = self._parse_and_validate_multimodal_inputs(**kwargs)
        
        if not mm_input_by_modality:
            return None
        
        multimodal_embeddings: tuple[torch.Tensor, ...] = ()
        
        for modality in mm_input_by_modality:
            multimodal_input = mm_input_by_modality[modality]
            
            if modality == "image":
                vision_embeddings = self._process_image_input(multimodal_input)
                multimodal_embeddings += vision_embeddings
                
            if modality == "video":
                video_embeddings = self._process_video_input(multimodal_input)
                multimodal_embeddings += video_embeddings
                
            if modality == "ecg":
                ecg_embeddings = self._process_ecg_input(multimodal_input)
                multimodal_embeddings += ecg_embeddings
                print(f"[ECG-R1 get_multimodal_embeddings] ECG processed, shape={ecg_embeddings[0].shape}", flush=True)
        
        return multimodal_embeddings

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
    ) -> torch.Tensor:
        """
        Merge multi-modal embeddings into text embeddings
        """
        deepstack_input_embeds = None
        inputs_embeds = self.language_model.get_input_embeddings(input_ids)
        
        if multimodal_embeddings is None:
            return inputs_embeds
        
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        ecg_token_id = self.ecg_token_id
        
        n_image_tokens = (input_ids == image_token_id).sum().item()
        n_video_tokens = (input_ids == video_token_id).sum().item()
        n_ecg_tokens = (input_ids == ecg_token_id).sum().item() if ecg_token_id else 0
        
        print(f"[ECG-R1 merge] token_ids: image={image_token_id}, video={video_token_id}, ecg={ecg_token_id}", flush=True)
        print(f"[ECG-R1 merge] input_ids shape: {input_ids.shape}", flush=True)
        print(f"[ECG-R1 merge] tokens in input_ids: image={n_image_tokens}, video={n_video_tokens}, ecg={n_ecg_tokens}", flush=True)
        
        for i, emb in enumerate(multimodal_embeddings):
            print(f"[ECG-R1 merge] embedding[{i}] shape: {emb.shape}", flush=True)
        
        image_video_embeddings = []
        ecg_embeddings = []
        
        expected_ecg_hidden = self.llm_hidden_size  # 4096
        
        for emb in multimodal_embeddings:
            emb_hidden = emb.shape[-1]
            if emb_hidden == expected_ecg_hidden:
                ecg_embeddings.append(emb)
            else:
                image_video_embeddings.append(emb)
        
        print(f"[ECG-R1 merge] split: image_video={len(image_video_embeddings)}, ecg={len(ecg_embeddings)}", flush=True)
        
        if image_video_embeddings:
            image_video_tuple = tuple(image_video_embeddings)
            if self.use_deepstack:
                deepstack_input_embeds, image_video_tuple = self._compute_deepstack_embeds(
                    input_ids, inputs_embeds, image_video_tuple
                )
            inputs_embeds = merge_multimodal_embeddings(
                input_ids, inputs_embeds, image_video_tuple,
                [self.config.image_token_id, self.config.video_token_id]
            )
        
        if ecg_embeddings and self.ecg_token_id is not None:
            ecg_tuple = tuple(ecg_embeddings)
            inputs_embeds = merge_multimodal_embeddings(
                input_ids, inputs_embeds, ecg_tuple,
                self.ecg_token_id
            )
            print(f"[ECG-R1 merge] ECG merged", flush=True)
        
        if self.use_deepstack:
            if deepstack_input_embeds is None:
                deepstack_input_embeds = torch.zeros_like(
                    inputs_embeds).unsqueeze(0).repeat(
                        self.deepstack_num_level, 1, 1).contiguous()
            self._set_deepstack_input_embeds(deepstack_input_embeds)
        
        return inputs_embeds
