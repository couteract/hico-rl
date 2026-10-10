"""Explicit adapter to the unchanged public Pistar06 processor pipeline."""
from __future__ import annotations

from lerobot.values.pistar06.processor_pistar06 import make_pistar06_pre_post_processors


def make_memory_value_processors(config, dataset_stats=None):
    return make_pistar06_pre_post_processors(config, dataset_stats)
