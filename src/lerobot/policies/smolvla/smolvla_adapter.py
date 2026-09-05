#!/usr/bin/env python3
"""
SmolVLA Model Adapter for lerobot-mavi Evaluation Compatibility

This adapter allows lerobot-t modified models to be loaded and used in
lerobot-mavi evaluation environment by wrapping the new architecture
with an interface compatible with the default model format.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
import json
import os


class SmolVLAArchitectureAdapter(nn.Module):
    """
    Adapter wrapper for lerobot-t modified SmolVLA models

    Handles:
    1. Expert layer dimension changes (960 -> 720 when needed)
    2. Stage completion head outputs
    3. Architecture compatibility with default evaluation code
    """

    def __init__(self, original_model, target_expert_dim: int = 720):
        """
        Args:
            original_model: The trained lerobot-t SmolVLA model
            target_expert_dim: Target expert hidden dimension for compatibility (default 720)
        """
        super().__init__()
        self.original_model = original_model
        self.target_expert_dim = target_expert_dim

        # Store original dimensions for reference
        self.original_expert_dim = getattr(
            original_model.vlm_with_expert, 'expert_hidden_size', 960
        )

        # Check if dimension adaptation is needed
        self.needs_dim_adaptation = self.original_expert_dim != target_expert_dim

        if self.needs_dim_adaptation:
            print(f"[Adapter] Enabling dimension adaptation: {self.original_expert_dim} -> {target_expert_dim}")

            # Create projection layer for dimension reduction
            self.expert_dim_projection = nn.Linear(
                self.original_expert_dim,
                target_expert_dim,
                bias=True
            )
            # Initialize with identity-like projection
            with torch.no_grad():
                self.expert_dim_projection.weight.copy_(
                    torch.eye(target_expert_dim, self.original_expert_dim)
                )
                self.expert_dim_projection.bias.zero_()
        else:
            self.expert_dim_projection = None

        # Store config for compatibility checks
        self.config = getattr(original_model, 'config', None)

    def forward(self, *args, **kwargs):
        """Forward pass through adapter"""
        outputs = self.original_model(*args, **kwargs)

        # Handle output if dimension adaptation is needed
        if self.needs_dim_adaptation and outputs is not None:
            if isinstance(outputs, torch.Tensor):
                outputs = self._adapt_output_dim(outputs)
            elif isinstance(outputs, tuple):
                outputs = tuple(
                    self._adapt_output_dim(o) if isinstance(o, torch.Tensor) else o
                    for o in outputs
                )
            elif isinstance(outputs, dict):
                for key in outputs:
                    if isinstance(outputs[key], torch.Tensor):
                        outputs[key] = self._adapt_output_dim(outputs[key])

        return outputs

    def _adapt_output_dim(self, tensor: torch.Tensor) -> torch.Tensor:
        """Adapt tensor dimension from original to target expert dim"""
        if tensor.shape[-1] == self.original_expert_dim:
            # Project last dimension
            return self.expert_dim_projection(tensor)
        return tensor

    def select_action(self, observation):
        """Select action - forward to original model"""
        return self.original_model.select_action(observation)

    @property
    def device(self):
        """Get device from original model"""
        return next(self.original_model.parameters()).device

    def to(self, *args, **kwargs):
        """Move to device"""
        self.original_model = self.original_model.to(*args, **kwargs)
        if self.expert_dim_projection is not None:
            self.expert_dim_projection = self.expert_dim_projection.to(*args, **kwargs)
        return self

    def eval(self):
        """Set to evaluation mode"""
        self.original_model.eval()
        if self.expert_dim_projection is not None:
            self.expert_dim_projection.eval()
        return self

    def train(self, mode: bool = True):
        """Set to training mode"""
        self.original_model.train(mode)
        if self.expert_dim_projection is not None and mode:
            self.expert_dim_projection.train(mode)
        return self


class SmolVLAModelLoader:
    """
    Enhanced model loader that detects model architecture version
    and applies appropriate adapter if needed
    """

    @staticmethod
    def detect_model_version(checkpoint_path: str) -> str:
        """
        Detect if model is from lerobot-t (modified) or lerobot-0.4.0 (default)

        Returns:
            'modified' - lerobot-t with new architecture
            'default' - lerobot-0.4.0 default architecture
        """
        config_path = os.path.join(checkpoint_path, 'config.json')

        if not os.path.exists(config_path):
            return 'unknown'

        with open(config_path, 'r') as f:
            config = json.load(f)

        # Check for new architecture markers
        if config.get('expert_width_multiplier') == 1.0:
            return 'modified'

        return 'default'

    @staticmethod
    def load_model_with_adapter(checkpoint_path: str, device: str = 'cpu'):
        """
        Load model and apply adapter if needed for modified architecture

        Args:
            checkpoint_path: Path to model checkpoint
            device: Device to load model on ('cpu' or 'cuda')

        Returns:
            model: Original model or wrapped with adapter
            is_adapted: Whether adapter was applied
        """
        # The policy implementation lives in modeling_smolvla in LeRobot
        # 0.4.4 (the old 0.4.0 module was named smolvla.py).
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        print(f"\n[Loader] Loading model from: {checkpoint_path}")

        # Detect model version
        version = SmolVLAModelLoader.detect_model_version(checkpoint_path)
        print(f"[Loader] Detected model version: {version}")

        # Load base model
        model = SmolVLAPolicy.from_pretrained(checkpoint_path)
        model = model.to(device)

        # Apply adapter if modified architecture detected
        if version == 'modified':
            print(f"[Loader] Applying adapter for modified architecture")
            model = SmolVLAArchitectureAdapter(
                model,
                target_expert_dim=720
            )
            model = model.to(device)
            model.eval()
            print(f"[Loader] Adapter applied successfully")
            return model, True
        else:
            model.eval()
            print(f"[Loader] Using default model (no adapter needed)")
            return model, False


def wrap_model_for_evaluation(model, checkpoint_path: str):
    """
    Utility function to wrap model if needed for evaluation

    Usage in eval scripts:
        from smolvla_adapter import wrap_model_for_evaluation
        model = wrap_model_for_evaluation(model, checkpoint_path)
    """
    version = SmolVLAModelLoader.detect_model_version(checkpoint_path)

    if version == 'modified':
        print(f"[Evaluation] Wrapping model with adapter for modified architecture")
        return SmolVLAArchitectureAdapter(model, target_expert_dim=720)

    return model
