"""
B01 Test: Frozen backbone eval mode (migrated from Window A)

Verifies that when freeze_vision_encoder=True and freeze_language_model=True,
calling .train() on the model keeps frozen backbones in eval mode.

Original: A_libero_baseline/tests/workstream_a/test_frozen_value_mode.py
Migration: Removed path injection, tests Window B implementation directly
"""
import unittest
from types import SimpleNamespace
import torch
from lerobot.values.pistar06.modeling_pistar06 import Pistar06Model


class FrozenModeTest(unittest.TestCase):
    def test_train_keeps_frozen_pretrained_backbones_eval(self):
        """Frozen backbones stay in eval mode when model.train() called"""
        m = Pistar06Model.__new__(Pistar06Model)
        torch.nn.Module.__init__(m)
        m.cfg = SimpleNamespace(
            freeze_vision_encoder=True,
            freeze_language_model=True
        )
        m.vision_encoder = torch.nn.Sequential(torch.nn.Dropout(0.9))
        m.language_model = torch.nn.Sequential(torch.nn.Dropout(0.9))
        m.head = torch.nn.Dropout(0.5)

        m.train()

        # Head should be in training mode
        self.assertTrue(m.head.training)

        # Frozen backbones should stay in eval mode
        self.assertFalse(m.vision_encoder.training)
        self.assertFalse(m.language_model.training)

        # Dropout=0.9 in eval mode should pass input unchanged
        x = torch.ones(20)
        self.assertTrue(torch.equal(m.vision_encoder(x), x))
        self.assertTrue(torch.equal(m.language_model(x), x))


if __name__ == '__main__':
    unittest.main()
