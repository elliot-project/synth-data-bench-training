import torch
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Union
from transformers import AutoProcessor
from megatron.energon import TaskEncoder, stateless, InterleavedSample
import numpy as np
from PIL import Image
import io

@dataclass
class EncodedSample:
    __key__: str
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    length: int

class VQABaseTaskEncoder(TaskEncoder):
    """Base class for VQA task encoders providing visualization metadata."""
    def __init__(self, model_id: str, max_length: int = 2048):
        self.model_id = model_id
        self.max_length = max_length
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.tokenizer = self.processor.tokenizer
        
        # Visualization metadata
        self.CAT_MAP = {
            "Padding": 0,
            "User Text": 1,
            "Assistant Text": 2,
            "Image": 3,
            "Special": 4,
            "Boundary": 5
        }
        self.COLORS = ["#ecf0f1", "#3498db", "#9b59b6", "#2ecc71", "#e74c3c", "#000000"]
        
        # Token IDs for categorization
        self.special_ids = set(self.tokenizer.all_special_ids)
        vision_tokens = ["<|image_pad|>", "<|vision_pad|>", "<|vision_start|>", "<|vision_end|>"]
        self.vision_ids = {self.tokenizer.convert_tokens_to_ids(vt) for vt in vision_tokens if self.tokenizer.convert_tokens_to_ids(vt) is not None}
        self.im_start_id = self.tokenizer.convert_tokens_to_ids("<|im_start|>")

    def categorize(self, input_ids, attention_mask, cu_seqlens=None):
        """Maps token IDs to category integers for visualization."""
        batch_size, seq_len = input_ids.shape
        cat_matrix = np.zeros((batch_size, seq_len), dtype=int)
        
        for i in range(batch_size):
            current_role = self.CAT_MAP["User Text"]
            boundaries = set(cu_seqlens[i].tolist()) if cu_seqlens is not None else set()

            for j in range(seq_len):
                token_id = input_ids[i, j].item()
                is_valid = attention_mask[i, j].item()
                
                if not is_valid:
                    cat_matrix[i, j] = self.CAT_MAP["Padding"]
                    continue

                if j in boundaries and j > 0:
                    cat_matrix[i, j] = self.CAT_MAP["Boundary"]
                    current_role = self.CAT_MAP["User Text"]
                    continue

                if token_id == self.im_start_id and j + 1 < seq_len:
                    next_str = self.tokenizer.decode([input_ids[i, j+1].item()]).strip().lower()
                    if "user" in next_str:
                        current_role = self.CAT_MAP["User Text"]
                    elif "assistant" in next_str:
                        current_role = self.CAT_MAP["Assistant Text"]

                if token_id in self.vision_ids:
                    cat_matrix[i, j] = self.CAT_MAP["Image"]
                elif token_id in self.special_ids:
                    cat_matrix[i, j] = self.CAT_MAP["Special"]
                else:
                    cat_matrix[i, j] = current_role
                    
        return cat_matrix

class MultimodalTaskEncoder(VQABaseTaskEncoder):
    """InterleavedSample encoder with DATA PACKING"""
    @property
    def packing_buffer_size(self):
        return 100

    def _prepare_messages(self, sample: InterleavedSample):
        messages = []
        images = []
        current_role = None
        current_content = []
        
        for part in getattr(sample, 'sequence', []):
            if isinstance(part, dict):
                role = part['role']
                text = part['text']
                
                if role != current_role:
                    if current_role is not None:
                        messages.append({"role": current_role, "content": current_content})
                        current_content = []
                    current_role = role
                
                current_content.append({"type": "text", "text": text})
            else:
                if isinstance(part, bytes):
                    part = Image.open(io.BytesIO(part))
                images.append(part)
                current_content.append({"type": "image", "image": part})
        
        if current_role is not None:
            messages.append({"role": current_role, "content": current_content})
            
        return messages, images

    @stateless(restore_seeds=True)
    def encode_sample(self, sample: InterleavedSample) -> EncodedSample:
        messages, images = self._prepare_messages(sample)
        
        if not messages:
            return EncodedSample(__key__=sample.__key__, input_ids=torch.zeros(1, dtype=torch.long), attention_mask=torch.zeros(1, dtype=torch.long), length=1)
            
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        inputs = self.processor(text=[text], images=images, padding=False, return_tensors="pt")
        
        return EncodedSample(
            __key__=sample.__key__,
            input_ids=inputs["input_ids"][0],
            attention_mask=inputs["attention_mask"][0],
            length=len(inputs["input_ids"][0])
        )

    def select_samples_to_pack(self, samples: List[EncodedSample]) -> List[List[EncodedSample]]:
        samples.sort(key=lambda x: x.length, reverse=True)
        groups = []
        while samples:
            current_group = [samples.pop(0)]
            current_len = current_group[0].length
            i = 0
            while i < len(samples):
                if current_len + samples[i].length <= self.max_length:
                    sample = samples.pop(i)
                    current_group.append(sample)
                    current_len += sample.length
                else:
                    i += 1
            groups.append(current_group)
        return groups

    @stateless
    def pack_selected_samples(self, samples: List[EncodedSample]) -> Dict[str, Any]:
        packed_input_ids = torch.cat([s.input_ids for s in samples])
        packed_attention_mask = torch.cat([s.attention_mask for s in samples])
        cu_seqlens = torch.tensor([0] + list(np.cumsum([s.length for s in samples])), dtype=torch.int32)
        
        pad_len = self.max_length - packed_input_ids.size(0)
        if pad_len > 0:
            packed_input_ids = torch.cat([packed_input_ids, torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=torch.long)])
            packed_attention_mask = torch.cat([packed_attention_mask, torch.zeros((pad_len,), dtype=torch.long)])
        
        return {
            "input_ids": packed_input_ids,
            "attention_mask": packed_attention_mask,
            "cu_seqlens": cu_seqlens,
        }
