"""Bounded CPU engineering learner for the B02Q2 Q-only SAC contract."""
from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from .sac_q_interface import ACTION_DIM, ARM_DIM, BETA, TARGET_ENTROPY, TorchQOnlyCritic

GAMMA = 0.99
POLYAK_TAU = 0.005
SYNTHETIC_EXACT_CONTRACT = "synthetic_exact_contract"
RECORDED_FOLD_LOCAL_EXACT_CONTRACT = "recorded_fold_local_exact_contract"
RECORDED_FOLD_LOCAL_BOUNDED_TRAIN_CONTRACT = "recorded_fold_local_bounded_train"
LATENT_ENTROPY_DOMAIN = "latent_log_pi_u"


def compose_submitted_action_torch(
    raw_vla7: torch.Tensor,
    u_arm6: torch.Tensor,
    *,
    beta: float = BETA,
) -> torch.Tensor:
    """Compose submitted7 while preserving the actor-to-Q autograd path."""
    if beta <= 0.0 or beta > 1.0:
        raise ValueError("differentiable composition requires 0 < beta <= 1")
    if raw_vla7.ndim != 2 or raw_vla7.shape[-1] != ACTION_DIM:
        raise ValueError(f"raw_vla7 must have shape [batch,{ACTION_DIM}]")
    if u_arm6.ndim != 2 or u_arm6.shape != (raw_vla7.shape[0], ARM_DIM):
        raise ValueError(f"u_arm6 must have shape [batch,{ARM_DIM}]")
    if not torch.isfinite(raw_vla7).all() or not torch.isfinite(u_arm6).all():
        raise ValueError("composition inputs must be finite")
    base = torch.clamp(raw_vla7[:, :ARM_DIM].detach(), -1.0, 1.0)
    submitted_arm = (1.0 - beta) * base + beta * u_arm6
    return torch.cat([submitted_arm, raw_vla7[:, ARM_DIM:ACTION_DIM].detach()], dim=-1)


def beta_zero_vla_baseline(raw_vla7: torch.Tensor) -> torch.Tensor:
    """Independent beta=0 path: no actor call and no Jacobian computation."""
    if raw_vla7.ndim != 2 or raw_vla7.shape[-1] != ACTION_DIM:
        raise ValueError(f"raw_vla7 must have shape [batch,{ACTION_DIM}]")
    if not torch.isfinite(raw_vla7).all():
        raise ValueError("raw_vla7 must be finite")
    return raw_vla7.clone()


@dataclass(frozen=True)
class QOnlyBatch:
    observation: torch.Tensor
    memory: torch.Tensor
    raw_vla7: torch.Tensor
    recorded_submitted7: torch.Tensor
    reward: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    next_observation: torch.Tensor | None
    next_memory: torch.Tensor | None
    next_raw_vla7: torch.Tensor | None
    provenance: str
    current_memory_role: str = "current_memory"
    next_memory_role: str = "next_memory"

    @property
    def batch_size(self) -> int:
        return int(self.observation.shape[0])

    @property
    def bootstrap_mask(self) -> torch.Tensor:
        return ~(self.terminated.bool() | self.truncated.bool())

    def validate(
        self,
        *,
        observation_dim: int,
        memory_dim: int,
        require_synthetic: bool,
        expected_provenance: str | None = None,
    ) -> None:
        if require_synthetic and self.provenance != SYNTHETIC_EXACT_CONTRACT:
            raise ValueError("optimizer steps require synthetic_exact_contract provenance")
        if expected_provenance is not None and self.provenance != expected_provenance:
            raise ValueError(f"batch provenance must be {expected_provenance}")
        if self.current_memory_role != "current_memory":
            raise ValueError("current memory provenance role mismatch")
        if self.next_memory_role != "next_memory":
            raise ValueError("next memory provenance role mismatch")
        batch = self.batch_size
        expected = {
            "observation": (batch, observation_dim),
            "memory": (batch, memory_dim),
            "raw_vla7": (batch, ACTION_DIM),
            "recorded_submitted7": (batch, ACTION_DIM),
            "reward": (batch,),
            "terminated": (batch,),
            "truncated": (batch,),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if tuple(value.shape) != shape:
                raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} must be finite")
        if torch.any(self.terminated.bool() & self.truncated.bool()):
            raise ValueError("terminated and truncated are mutually exclusive")
        live = self.bootstrap_mask
        if torch.any(live):
            if self.next_observation is None or self.next_memory is None or self.next_raw_vla7 is None:
                raise ValueError("nonterminal transition requires next observation, memory, and raw VLA base")
            next_expected = {
                "next_observation": (batch, observation_dim),
                "next_memory": (batch, memory_dim),
                "next_raw_vla7": (batch, ACTION_DIM),
            }
            for name, shape in next_expected.items():
                value = getattr(self, name)
                if tuple(value.shape) != shape:
                    raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")
                if not torch.isfinite(value[live]).all():
                    raise ValueError(f"live rows of {name} must be finite")


def set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def validate_optimizer_partition(
    learner: "QOnlySACLearner",
    *,
    frozen_modules: tuple[nn.Module, ...] = (),
) -> dict[str, int]:
    """Fail closed if any optimizer owns another parameter domain."""
    actor_ids = {id(parameter) for parameter in learner.actor.parameters()}
    critic_ids = {id(parameter) for parameter in learner.critic.parameters()}
    temperature_ids = {id(learner.log_alpha)}
    target_ids = {id(parameter) for parameter in learner.target_critic.parameters()}
    frozen_ids = {id(parameter) for module in frozen_modules for parameter in module.parameters()}

    def optimizer_ids(optimizer: torch.optim.Optimizer) -> set[int]:
        return {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}

    actual = {
        "actor": optimizer_ids(learner.actor_optimizer),
        "critic": optimizer_ids(learner.critic_optimizer),
        "temperature": optimizer_ids(learner.temperature_optimizer),
    }
    expected = {"actor": actor_ids, "critic": critic_ids, "temperature": temperature_ids}
    for domain in expected:
        if actual[domain] != expected[domain]:
            raise ValueError(f"{domain} optimizer parameter domain mismatch")
    forbidden = target_ids | frozen_ids
    if any(parameters & forbidden for parameters in actual.values()):
        raise ValueError("target, VLA, or memory parameters must not be optimizer-owned")
    return {
        "actor": len(actor_ids),
        "critic": len(critic_ids),
        "temperature": len(temperature_ids),
        "target_excluded": len(target_ids),
        "frozen_excluded": len(frozen_ids),
    }


class QOnlySACLearner:
    """B-only split-dimension learner with hard one-step execution guards."""

    def __init__(
        self,
        actor: nn.Module,
        critic: TorchQOnlyCritic,
        target_critic: TorchQOnlyCritic,
        *,
        observation_dim: int,
        memory_dim: int,
        log_alpha: nn.Parameter,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        temperature_lr: float = 3e-4,
        gamma: float = GAMMA,
        beta: float = BETA,
        target_entropy: float = TARGET_ENTROPY,
        tau: float = POLYAK_TAU,
        entropy_domain: str = LATENT_ENTROPY_DOMAIN,
        input_contract: str = SYNTHETIC_EXACT_CONTRACT,
    ) -> None:
        if entropy_domain != LATENT_ENTROPY_DOMAIN:
            raise ValueError("all losses must use latent_log_pi_u; log_pi_a is telemetry only")
        if beta != BETA or gamma != GAMMA or target_entropy != TARGET_ENTROPY or tau != POLYAK_TAU:
            raise ValueError("B02Q2 frozen hyperparameter mismatch")
        if critic.state_dict().keys() != target_critic.state_dict().keys():
            raise ValueError("current and target critic structures differ")
        if any(not torch.equal(v, target_critic.state_dict()[k]) for k, v in critic.state_dict().items()):
            raise ValueError("current and target critic initial states must be identical")
        self.actor = actor
        self.critic = critic
        self.target_critic = target_critic
        self.observation_dim = observation_dim
        self.memory_dim = memory_dim
        self.log_alpha = log_alpha
        self.gamma = gamma
        self.beta = beta
        self.target_entropy = target_entropy
        self.tau = tau
        if input_contract not in (
            SYNTHETIC_EXACT_CONTRACT,
            RECORDED_FOLD_LOCAL_EXACT_CONTRACT,
            RECORDED_FOLD_LOCAL_BOUNDED_TRAIN_CONTRACT,
        ):
            raise ValueError("unsupported Q-only input contract")
        self.input_contract = input_contract
        self.max_updates = 1
        set_requires_grad(self.target_critic, False)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.temperature_optimizer = torch.optim.Adam([self.log_alpha], lr=temperature_lr)
        self.step_counts = {"critic": 0, "actor": 0, "temperature": 0, "polyak": 0}

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def _validate(self, batch: QOnlyBatch, *, require_synthetic: bool = True) -> None:
        batch.validate(
            observation_dim=self.observation_dim,
            memory_dim=self.memory_dim,
            require_synthetic=require_synthetic and self.input_contract == SYNTHETIC_EXACT_CONTRACT,
            expected_provenance=self.input_contract,
        )

    def _guard_once(self, domain: str) -> None:
        limit = self.max_updates
        if self.step_counts[domain] >= limit:
            if limit == 1:
                raise RuntimeError(f"B02Q2 permits exactly one {domain} update")
            raise RuntimeError(f"bounded train permits exactly {limit} {domain} updates")

    def configure_bounded_train(self, *, max_updates: int) -> None:
        """Explicitly opt into the single bounded multi-step engineering contract."""
        if self.input_contract != RECORDED_FOLD_LOCAL_BOUNDED_TRAIN_CONTRACT:
            raise ValueError("bounded training requires recorded_fold_local_bounded_train contract")
        if self.step_counts != {"critic": 0, "actor": 0, "temperature": 0, "polyak": 0}:
            raise RuntimeError("bounded train must be configured before any update")
        if max_updates != 109:
            raise ValueError("B02Q6 frozen budget is exactly 109 updates")
        self.max_updates = max_updates

    def configure_formal_train(self, *, max_updates: int) -> None:
        """Opt into the separately registered B02Q9 formal update budget."""
        if self.input_contract != RECORDED_FOLD_LOCAL_BOUNDED_TRAIN_CONTRACT:
            raise ValueError("formal training requires recorded_fold_local_bounded_train contract")
        if self.step_counts != {"critic": 0, "actor": 0, "temperature": 0, "polyak": 0}:
            raise RuntimeError("formal train must be configured before any update")
        if max_updates != 2000:
            raise ValueError("B02Q9 frozen budget is exactly 2000 updates")
        self.max_updates = max_updates

    def clear_all_gradients(self) -> None:
        self.actor_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)
        self.temperature_optimizer.zero_grad(set_to_none=True)

    def bellman_target(self, batch: QOnlyBatch) -> torch.Tensor:
        self._validate(batch)
        target = batch.reward.detach().clone()
        live = batch.bootstrap_mask
        if not torch.any(live):
            return target
        assert batch.next_observation is not None
        assert batch.next_memory is not None
        assert batch.next_raw_vla7 is not None
        indices = torch.nonzero(live, as_tuple=False).squeeze(-1)
        with torch.no_grad():
            u_next, log_pi_u_next, _ = self.actor(batch.next_observation.index_select(0, indices))
            next_action7 = compose_submitted_action_torch(
                batch.next_raw_vla7.index_select(0, indices), u_next, beta=self.beta
            )
            next_q = self.target_critic(
                batch.next_observation.index_select(0, indices),
                batch.next_memory.index_select(0, indices),
                next_action7,
            ).min(dim=0).values
            live_target = batch.reward.index_select(0, indices) + self.gamma * (
                next_q - self.alpha.detach() * log_pi_u_next
            )
        target.index_copy_(0, indices, live_target)
        return target.detach()

    def critic_loss(self, batch: QOnlyBatch) -> tuple[torch.Tensor, torch.Tensor]:
        target = self.bellman_target(batch)
        predicted = self.critic(batch.observation, batch.memory, batch.recorded_submitted7)
        return F.mse_loss(predicted, target.unsqueeze(0).expand_as(predicted)), target

    def actor_loss(self, batch: QOnlyBatch) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        self._validate(batch)
        u_arm6, log_pi_u, _ = self.actor(batch.observation)
        submitted7 = compose_submitted_action_torch(batch.raw_vla7, u_arm6, beta=self.beta)
        min_q = self.critic(batch.observation, batch.memory, submitted7).min(dim=0).values
        entropy_component = (self.alpha.detach() * log_pi_u).mean()
        q_component = -min_q.mean()
        return entropy_component + q_component, {
            "entropy_component": entropy_component,
            "q_component": q_component,
            "log_pi_u": log_pi_u,
            "u_arm6": u_arm6,
            "submitted7": submitted7,
        }

    def temperature_loss(self, batch: QOnlyBatch) -> torch.Tensor:
        self._validate(batch)
        with torch.no_grad():
            _, log_pi_u, _ = self.actor(batch.observation)
        return (-self.alpha * (log_pi_u.detach() + self.target_entropy)).mean()

    def critic_step(self, batch: QOnlyBatch) -> tuple[float, torch.Tensor]:
        self._guard_once("critic")
        self.clear_all_gradients()
        loss, target = self.critic_loss(batch)
        loss.backward()
        self.critic_optimizer.step()
        self.step_counts["critic"] += 1
        return float(loss.detach()), target

    def actor_step(self, batch: QOnlyBatch) -> tuple[float, dict[str, torch.Tensor]]:
        self._guard_once("actor")
        self.clear_all_gradients()
        set_requires_grad(self.critic, False)
        try:
            loss, detail = self.actor_loss(batch)
            loss.backward()
            self.actor_optimizer.step()
        finally:
            set_requires_grad(self.critic, True)
        self.step_counts["actor"] += 1
        return float(loss.detach()), {k: v.detach() for k, v in detail.items()}

    def temperature_step(self, batch: QOnlyBatch) -> float:
        self._guard_once("temperature")
        self.clear_all_gradients()
        loss = self.temperature_loss(batch)
        loss.backward()
        self.temperature_optimizer.step()
        self.step_counts["temperature"] += 1
        return float(loss.detach())

    def polyak_update(self) -> None:
        self._guard_once("polyak")
        with torch.no_grad():
            for target_parameter, parameter in zip(
                self.target_critic.parameters(), self.critic.parameters(), strict=True
            ):
                target_parameter.copy_((1.0 - self.tau) * target_parameter + self.tau * parameter)
        self.step_counts["polyak"] += 1
