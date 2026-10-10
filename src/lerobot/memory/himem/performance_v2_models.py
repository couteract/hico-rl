"""B-local Performance V2 interfaces; no optimizer or checkpoint creation."""
from dataclasses import asdict, dataclass

import torch
from torch import nn

from lerobot.policies.sac.modeling_sac import MLP, Policy
from .performance_v2_contract import action_torch, actor_terms, soft_target


@dataclass(frozen=True)
class V2Config:
    current_dim: int = 1981
    memory_dim: int = 256
    submitted_dim: int = 7
    latent_dim: int = 6
    critic_dim: int = 2244
    hidden_dims: tuple = (128, 128)

    def validate(self):
        if (self.current_dim, self.memory_dim, self.submitted_dim, self.latent_dim, self.critic_dim) != (1981, 256, 7, 6, 2244):
            raise ValueError('V2 dimension contract; old 1928D/A weights forbidden')
        if not self.hidden_dims or any(width <= 0 for width in self.hidden_dims):
            raise ValueError('hidden dimensions')


def typed_batch(value, width):
    if value.dtype != torch.float32 or value.ndim != 2 or value.shape[1] != width or not torch.isfinite(value).all():
        raise ValueError('float32 batch shape/finite')
    return value


class TensorEncoder(nn.Module):
    def forward(self, observations, cache=None, detach=False):
        value = typed_batch(observations, 1981)
        return value.detach() if detach else value


class V2Models(nn.Module):
    def __init__(self, config=V2Config()):
        super().__init__()
        config.validate()
        self.config = config
        self.actor = Policy(TensorEncoder(), MLP(1981, list(config.hidden_dims)), 6,
                            std_min=0.01, std_max=2.0)
        self.critics = nn.ModuleList([
            nn.Sequential(MLP(2244, list(config.hidden_dims)), nn.Linear(config.hidden_dims[-1], 1))
            for _ in range(2)
        ])

    def critic_input(self, current, memory, submitted, transform, memory_off=False):
        typed_batch(current, 1981)
        typed_batch(memory, 256)
        typed_batch(submitted, 7)
        transformed = typed_batch(transform(memory), 256)
        if memory_off:
            transformed = torch.zeros_like(transformed, dtype=torch.float32)
        if not (len(current) == len(transformed) == len(submitted)):
            raise ValueError('batch cardinality')
        return torch.cat([current, transformed, submitted], dim=-1)

    def critic(self, current, memory, submitted, transform, memory_off=False):
        inputs = self.critic_input(current, memory, submitted, transform, memory_off)
        return torch.cat([critic(inputs) for critic in self.critics], dim=-1)

    def project(self, raw_base, latent):
        typed_batch(raw_base, 7)
        typed_batch(latent, 6)
        if raw_base.device.type != 'cpu' or latent.device.type != 'cpu' or len(raw_base) != len(latent):
            raise ValueError('Q27 projection requires matching CPU batches')
        return torch.stack([action_torch(base, action)[0] for base, action in zip(raw_base, latent)])

    def schema(self):
        config = asdict(self.config)
        config['hidden_dims'] = list(config['hidden_dims'])
        return dict(schema_version=2, route='B_LOCAL_PERFORMANCE_V2', config=config,
                    normalizer_hash=None, split_hash=None, rng_fields=['python', 'numpy', 'torch_cpu', 'torch_cuda'],
                    sampler_fields=['generator_state', 'member_order', 'position'],
                    optimizer_fields=['actor', 'twin_critic', 'alpha', 'step'],
                    entropy_domain='current_actor_latent_log_pi_u_only', checkpoint_created=False)

    def validate_checkpoint_schema(self, schema):
        if schema.get('route') != 'B_LOCAL_PERFORMANCE_V2' or schema.get('config') != self.schema()['config']:
            raise ValueError('old 1928D/A or incompatible checkpoint schema')
        if set(schema) != set(self.schema()) or schema.get('entropy_domain') != 'current_actor_latent_log_pi_u_only':
            raise ValueError('checkpoint schema fields/entropy')
        if schema['normalizer_hash'] is not None or schema['split_hash'] is not None or schema['checkpoint_created']:
            raise ValueError('readiness forbids fitted normalizer/split/training checkpoint')

    target = staticmethod(soft_target)
    objective = staticmethod(actor_terms)
