"""Backward-compatible SmolVLA import path.

Older LeRobot examples imported the implementation from ``smolvla.py``.
The implementation now lives in :mod:`modeling_smolvla`; re-exporting it here
keeps those examples and checkpoints usable without duplicating model code.
"""

from .modeling_smolvla import (
    SmolVLAPolicy,
    VLAFlowMatching,
    create_sinusoidal_pos_embedding,
    make_att_2d_masks,
    normalize,
    pad_vector,
    resize_with_pad,
    unnormalize,
)

__all__ = [
    "SmolVLAPolicy",
    "VLAFlowMatching",
    "create_sinusoidal_pos_embedding",
    "make_att_2d_masks",
    "normalize",
    "pad_vector",
    "resize_with_pad",
    "unnormalize",
]
