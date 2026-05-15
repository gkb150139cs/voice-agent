"""Torch / transformers backend (import only when running local inference)."""

from __future__ import annotations

import logging
from typing import Any, Tuple

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file as load_safetensors_file
from transformers import Wav2Vec2Config, Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model

logger = logging.getLogger(__name__)


class ModelHead(nn.Module):
    def __init__(self, config: Any, num_labels: int) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, features: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class AgeGenderModel(nn.Module):
    """Plain ``nn.Module`` wrapper: avoids HF ``PreTrainedModel.init_weights`` / ``tie_weights`` on custom heads."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.age = ModelHead(config, 1)
        self.gender = ModelHead(config, 3)

    def forward(self, input_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits_age = self.age(hidden_states)
        logits_gender = torch.softmax(self.gender(hidden_states), dim=1)
        return hidden_states, logits_age, logits_gender


def load_model(model_id: str, device: str) -> tuple[Wav2Vec2Processor, AgeGenderModel]:
    """
    Build AgeGenderModel from config + hub weights.

    Avoids ``AgeGenderModel.from_pretrained`` — recent ``transformers`` runs a
    ``_finalize_model_loading`` path that expects ``PreTrainedModel`` internals
    this custom head does not expose the same way.
    """
    processor = Wav2Vec2Processor.from_pretrained(model_id)
    config = Wav2Vec2Config.from_pretrained(model_id)
    model = AgeGenderModel(config)

    try:
        weights_path = hf_hub_download(repo_id=model_id, filename="model.safetensors")
        state = load_safetensors_file(weights_path)
    except Exception as e:
        logger.warning("safetensors load failed (%s), trying pytorch_model.bin", e)
        weights_path = hf_hub_download(repo_id=model_id, filename="pytorch_model.bin")
        try:
            state = torch.load(weights_path, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(weights_path, map_location="cpu")

    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]

    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys:
        logger.warning("missing keys (first 8): %s", incompatible.missing_keys[:8])
    if incompatible.unexpected_keys:
        logger.warning("unexpected keys (first 8): %s", incompatible.unexpected_keys[:8])

    model.to(device)
    model.eval()
    return processor, model


def run_forward(
    processor: Wav2Vec2Processor,
    model: AgeGenderModel,
    y: np.ndarray,
    sr: int,
    device: str,
) -> tuple[float, float, float, float]:
    with torch.no_grad():
        inputs = processor(y, sampling_rate=sr, return_tensors="pt", padding=True)
        input_values = inputs.input_values.to(device)
        _, logits_age, logits_gender = model(input_values)
        la = float(logits_age.squeeze().cpu().numpy())
        lg = logits_gender.squeeze().cpu().numpy()
        p_female, p_male, p_child = float(lg[0]), float(lg[1]), float(lg[2])
    return la, p_female, p_male, p_child
