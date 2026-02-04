import asyncio
import os
import re
import textwrap
from collections import Counter
from copy import deepcopy
from typing import Dict, List, Union

import json
import torch
import torch
import torch.nn.functional as F
from modelscope import AutoTokenizer, AutoModel
import re
import os
from typing import List, Optional

from swift.llm import PtEngine, RequestConfig, RolloutInferRequest, Template, to_device
from swift.llm.infer.protocol import ChatCompletionResponse, ChatCompletionResponseChoice
from swift.plugin import ORM, orms, rm_plugins
from swift.plugin.context_manager import ContextManager, context_managers
from swift.plugin.env import Env, envs
from swift.plugin.multi_turn import MultiTurnScheduler, multi_turns
from swift.plugin.rm_plugin import DefaultRMPlugin
from swift.utils import get_logger, get_env_args


logger = get_logger()


class KeyDiagnosticEvidenceORM(ORM):
    STEP_MAPPING = {
        1: "step_1_technical_rate_rhythm",
        2: "step_2_conduction_axis_intervals",
        3: "step_3_chamber_hypertrophy_voltage",
        4: "step_4_ischemia_infarction_mimics",
        5: "step_5_electrolytes_qt",
        6: "step_6_final_medical_reasoning"
    }

    def __init__(self):
        self.evidence_weight = get_env_args('evidence_weight', float, 1.0)

    def _parse_model_steps(self, think_text: str) -> Dict[int, str]:
        """
        Parse Steps from model output.
        """
        steps_content = {}
        pattern = re.compile(
            r'\*\*Step\s+(\d+).*?(\*\*|:|\n)(.*?)(?=\*\*Step|\Z)', 
            re.DOTALL | re.IGNORECASE
        )
        
        for match in pattern.findall(think_text):
            step_num = int(match[0])
            content = match[2].strip()
            if content:
                steps_content[step_num] = content
        return steps_content

    def _parse_gt_custom(self, gt_text: str) -> Dict[str, List[str]]:
        parsed_data = {}
        pattern = re.compile(r'(step_\d+_[a-z_]+):\s*(.*?)(?=\s*step_\d+|$)', re.DOTALL | re.IGNORECASE)
        matches = pattern.findall(gt_text)
        
        for key, content in matches:
            evidences = [e.strip() for e in content.split(';') if e.strip()]
            parsed_data[key.lower()] = evidences
            
        return parsed_data

    def _check_evidence_hit(self, evidence: str, step_text: str) -> bool:
        if not evidence or not step_text:
            return False
            
        clean_evidence = re.escape(evidence)
        
        match = re.search(clean_evidence, step_text, re.IGNORECASE)
        return match is not None

    def __call__(self, completions: List[str], solution: List[str], **kwargs) -> List[float]:
        rewards = []

        think_pattern = r'<think>(.*?)</think>'
        gt_pattern = r'<\|key_diagnostic_evidence\|>(.*?)<\|[/\\]key_diagnostic_evidence\|>'

        for content, sol in zip(completions, solution):
            reward = 0.0
            try:
                think_match = re.search(think_pattern, content, flags=re.DOTALL | re.IGNORECASE)
                if not think_match:
                    rewards.append(0.0)
                    continue
                
                sol_match = re.search(gt_pattern, sol, flags=re.DOTALL | re.IGNORECASE)
                gt_raw_text = sol_match.group(1).strip() if sol_match else sol.strip()

                model_steps = self._parse_model_steps(think_match.group(1).strip())
                gt_data = self._parse_gt_custom(gt_raw_text)
                
                if not model_steps or not gt_data:
                    rewards.append(0.0)
                    continue

                total_step_score = 0.0
                valid_steps_count = 0

                for step_id, gt_key in self.STEP_MAPPING.items():
                    gt_evidences = gt_data.get(gt_key, [])
                    if not gt_evidences:
                        continue

                    valid_steps_count += 1
                    step_text = model_steps.get(step_id, "")
                    
                    if not step_text:
                        continue 

                    hits = 0
                    for anchor in gt_evidences:
                        if self._check_evidence_hit(anchor, step_text):
                            hits += 1
                    
                    step_score = hits / len(gt_evidences)
                    total_step_score += step_score

                if valid_steps_count > 0:
                    reward = (total_step_score / valid_steps_count) * self.evidence_weight
            
            except Exception:
                reward = 0.0
            
            rewards.append(reward)

        return rewards

orms["key_diagnostic_evidence_reward"] = KeyDiagnosticEvidenceORM

class FormatRewardORM(ORM):
    def __call__(self, completions, solution, **kwargs) -> List[float]:
        rewards = []

        pattern = r'<think>(.*?)</think>'

        for content in completions:
            reward = 0.0
            try:
                think_match = re.search(pattern, content, flags=re.DOTALL)
                
                if think_match and len(think_match.group(1).strip()) > 0:
                    reward += 1.0
                
                rewards.append(reward)
                            
            except Exception:
                rewards.append(reward)
        
        return rewards

orms['format_reward'] = FormatRewardORM


class DiagnosisAccuracyORM(ORM):
    def __call__(self, completions, solution, **kwargs) -> List[float]:
        rewards = []

        content_pattern = r'<answer>(.*?)</answer>'
        solution_pattern = r'<\|answer\|>(.*?)<\|[/\\]answer\|>'

        for content, sol in zip(completions, solution):
            reward = 0.0
            try:
                content_match = re.search(content_pattern, content, flags=re.DOTALL)
                sol_match = re.search(solution_pattern, sol, flags=re.DOTALL)

                if not sol_match or not content_match:
                    rewards.append(reward)
                    continue

                gt_text = sol_match.group(1).strip().lower()
                pd_text = content_match.group(1).strip().lower()

                if ';' in gt_text or ';' in pd_text:
                    gt_labels = {label.strip().lower() for label in gt_text.split(';') if label.strip()}
                    pd_labels = {label.strip().lower() for label in pd_text.split(';') if label.strip()}
                else:  # No semicolon case
                    gt_labels = {gt_text} if gt_text else set()
                    pd_labels = {pd_text} if pd_text else set()

                jaccard_reward = self._jaccard(gt_labels, pd_labels)
                reward += jaccard_reward
                rewards.append(reward)
                            
            except Exception:
                rewards.append(reward)
        
        return rewards
    
    @staticmethod
    def _jaccard(set1, set2):
        if not set1 and not set2:  # Both sets are empty
            return 1.0
        if not set1 or not set2:  # One of the sets is empty
            return 0.0

        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0

orms['diagnosis_accuracy_reward'] = DiagnosisAccuracyORM
