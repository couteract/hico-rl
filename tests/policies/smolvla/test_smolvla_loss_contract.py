import pytest
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import compute_smolvla_action_loss


def config_for_action_dim(action_dim: int, **kwargs) -> SmolVLAConfig:
    return SmolVLAConfig(
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))},
        max_action_dim=32,
        **kwargs,
    )


def test_so101_checkpoint_contract_resolves_six_actions_and_gripper_index_five():
    # Mirrors the observed SO101 checkpoint: output action feature is six-wide,
    # while the model output remains padded to 32 dimensions.
    cfg = config_for_action_dim(6, gripper_loss_weight=3.0)
    assert cfg.resolve_action_loss_contract() == (6, 5)
    assert cfg.max_action_dim == 32


def test_non_so101_action_width_is_resolved_from_its_own_metadata():
    cfg = config_for_action_dim(14, gripper_action_dim=12, gripper_loss_weight=1.0)
    assert cfg.resolve_action_loss_contract() == (14, 12)


def test_index_five_affects_gripper_and_index_31_and_padding_do_not_affect_loss_or_gradient():
    errors = torch.zeros((1, 2, 32), dtype=torch.float32, requires_grad=True)
    pad = torch.tensor([[False, True]], dtype=torch.bool)
    with torch.no_grad():
        errors[0, 0, 5] = 8.0
        errors[0, 0, 31] = 1000.0
        errors[0, 1, 10:31] = 500.0
    loss, pieces = compute_smolvla_action_loss(
        errors, pad, actual_action_dim=6, gripper_action_dim=5, gripper_loss_weight=3.0
    )
    assert loss.item() == pytest.approx(6.0)  # (0 + 3*8) / 4
    assert pieces["arm_loss"].item() == pytest.approx(0.0)
    assert pieces["gripper_loss"].item() == pytest.approx(8.0)
    loss.backward()
    assert errors.grad[0, 0, 5].abs().item() > 0
    assert errors.grad[..., 6:32].count_nonzero().item() == 0


def test_padding_timestep_is_excluded_from_numerator_and_denominator():
    errors = torch.zeros((1, 2, 32), dtype=torch.float64, requires_grad=True)
    with torch.no_grad():
        errors[0, 0, :5] = 2.0
        errors[0, 0, 5] = 3.0
        errors[0, 1, :6] = 100.0
    pad = torch.tensor([[False, True]], dtype=torch.bool)
    loss, pieces = compute_smolvla_action_loss(
        errors, pad, actual_action_dim=6, gripper_action_dim=5, gripper_loss_weight=1.0
    )
    assert pieces["valid_timestep_count"].item() == 1
    assert pieces["arm_loss"].item() == pytest.approx(2.0)
    assert pieces["gripper_loss"].item() == pytest.approx(3.0)
    assert loss.item() == pytest.approx(2.5)
    loss.backward()
    assert errors.grad[0, 1].count_nonzero().item() == 0
    assert errors.grad[..., 6:].count_nonzero().item() == 0


@pytest.mark.parametrize("weight", [1.0, 3.0])
def test_weighted_group_formula_matches_manual_calculation(weight):
    errors = torch.tensor([[[1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 999.0]]])
    pad = torch.tensor([[False]])
    loss, pieces = compute_smolvla_action_loss(
        errors, pad, actual_action_dim=6, gripper_action_dim=5, gripper_loss_weight=weight
    )
    arm = (1 + 4 + 9 + 16 + 25) / 5
    grip = 36
    expected = (arm + weight * grip) / (1 + weight)
    assert pieces["arm_loss"].item() == pytest.approx(arm)
    assert pieces["gripper_loss"].item() == pytest.approx(grip)
    assert loss.item() == pytest.approx(expected)


def test_missing_or_malformed_action_pad_and_all_padding_fail_closed():
    errors = torch.zeros((2, 3, 32))
    with pytest.raises(KeyError, match="action_is_pad"):
        compute_smolvla_action_loss(errors, None, actual_action_dim=6,
                                    gripper_action_dim=5, gripper_loss_weight=3.0)
    with pytest.raises(ValueError, match=r"bool \[B,L\]"):
        compute_smolvla_action_loss(errors, torch.zeros((2, 3)), actual_action_dim=6,
                                    gripper_action_dim=5, gripper_loss_weight=3.0)
    with pytest.raises(ValueError, match="all action_is_pad"):
        compute_smolvla_action_loss(errors, torch.ones((2, 3), dtype=torch.bool), actual_action_dim=6,
                                    gripper_action_dim=5, gripper_loss_weight=3.0)


def test_action_feature_metadata_mismatch_and_invalid_split_fail():
    cfg = config_for_action_dim(6, actual_action_dim=7)
    with pytest.raises(ValueError, match="disagrees"):
        cfg.resolve_action_loss_contract()
    cfg = config_for_action_dim(6, gripper_action_dim=6)
    with pytest.raises(ValueError, match="non-empty arm and gripper"):
        cfg.resolve_action_loss_contract()


def test_cpu_forward_backward_is_finite_without_optimizer_step():
    errors = torch.rand((2, 4, 32), requires_grad=True)
    pad = torch.tensor([[False, False, False, True], [False, True, True, True]], dtype=torch.bool)
    loss, _ = compute_smolvla_action_loss(
        errors, pad, actual_action_dim=6, gripper_action_dim=5, gripper_loss_weight=3.0
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert errors.grad is not None and torch.isfinite(errors.grad).all()
    # Deliberately no optimizer is constructed or stepped in this suite.
