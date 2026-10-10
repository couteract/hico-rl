"""B02Q27 CPU-only numeric contracts; no learner or model construction."""
from dataclasses import dataclass

import numpy as np
import torch

FIELDS = dict(visual=1920, state=8, task_onehot=10, base=7, queue_actions=28,
              queue_mask=4, queue_cursor=1, queue_remaining=1, time=2)
CONSTANTS = dict(gamma=.99, target_entropy=-3., epsilon=.05, anchor_lambda=10.)
META = {'episode', 'decision', 'queue_offset', 'memory_version', 'fold',
        'checkpoint_hash', 'provider_hash', 'frame', 'observed_frame', 'queue_length'}


def array(value, shape):
    result = np.asarray(value)
    if result.dtype != np.float32 or result.shape != shape or not np.isfinite(result).all():
        raise ValueError('shape/dtype/finite')
    return result


def condition(fields, meta):
    if list(fields) != list(FIELDS) or set(meta) != META:
        raise ValueError('missing/ordered/causal fields')
    for name, width in FIELDS.items():
        array(fields[name], (width,))
    frame = meta['frame']
    if not isinstance(frame, int) or not 0 <= frame <= 520 or meta['observed_frame'] != frame:
        raise ValueError('frame causality')
    for name in ('decision', 'queue_offset', 'memory_version', 'fold'):
        if not isinstance(meta[name], int) or meta[name] < 0:
            raise ValueError('version/identity')
    for name in ('checkpoint_hash', 'provider_hash'):
        if len(meta[name]) != 64 or any(char not in '0123456789abcdef' for char in meta[name]):
            raise ValueError('hash binding')
    if not meta['episode']:
        raise ValueError('episode')
    task = fields['task_onehot']
    if not np.isin(task, [0, 1]).all() or task.sum() != 1:
        raise ValueError('task onehot')
    cursor, remaining = fields['queue_cursor'][0], fields['queue_remaining'][0]
    if cursor != int(cursor) or remaining != int(remaining) or not 0 <= cursor <= 5 or not 0 <= remaining <= 4:
        raise ValueError('queue range')
    if cursor != meta['queue_offset'] or cursor + remaining != meta['queue_length'] or not 1 <= meta['queue_length'] <= 5:
        raise ValueError('queue cursor')
    mask = (np.arange(4) < remaining).astype(np.float32)
    if not np.array_equal(mask, fields['queue_mask']) or np.any(fields['queue_actions'].reshape(4, 7)[int(remaining):] != 0):
        raise ValueError('queue padding/mask')
    if np.any(np.abs(fields['base'][:6]) > 1):
        raise ValueError('bounded base')
    if not np.array_equal(fields['time'], np.array([frame / 520, (520-frame)/520], np.float32)):
        raise ValueError('time')
    return np.concatenate(list(fields.values()))


def action_numpy(raw_base, latent, epsilon=.05):
    raw_base, latent = array(raw_base, (7,)), array(latent, (6,))
    if epsilon != .05 or np.any(np.abs(latent) > 1):
        raise ValueError('latent/epsilon')
    bounded = np.clip(raw_base[:6], -1, 1)
    pre = bounded + np.float32(epsilon) * latent
    arm = np.clip(pre, -1, 1)
    arm = np.where(latent == 0, bounded, arm)
    return np.concatenate([arm, raw_base[6:]]), pre != arm


def action_torch(raw_base, latent, epsilon=.05):
    if raw_base.device.type != 'cpu' or latent.device.type != 'cpu':
        raise ValueError('CPU only')
    action_numpy(raw_base.detach().numpy(), latent.detach().numpy(), epsilon)
    bounded = raw_base[:6].clamp(-1, 1)
    pre = bounded + epsilon * latent
    arm = torch.where(latent == 0, bounded, pre.clamp(-1, 1))
    return torch.cat([arm, raw_base[6:]]), pre != arm


def action_jacobian(raw_base, latent):
    action_numpy(raw_base, latent)
    pre = np.clip(raw_base[:6], -1, 1) + .05 * latent
    if np.any(np.abs(pre) == 1):
        raise ValueError('nondifferentiable boundary')
    return np.diag(.05 * (np.abs(pre) < 1))


def validate_action(raw_base, latent, submitted, density_domain='latent_u'):
    expected, saturation = action_numpy(raw_base, latent)
    array(submitted, (7,))
    if expected.tobytes() != submitted.tobytes() or density_domain != 'latent_u':
        raise ValueError('action bytes/gripper/displacement/density')
    return saturation


def soft_target(reward, target_q1, target_q2, alpha, log_pi_u_next, *, terminated,
                truncated, frame, target_type='soft_q', bootstrap=None):
    if target_type != 'soft_q' or not 1 <= frame <= 520 or (truncated and frame != 520) or (terminated and truncated):
        raise ValueError('target type/boundary')
    mask = int(not (terminated or truncated or frame == 520))
    if bootstrap is not None and bootstrap != mask:
        raise ValueError('bootstrap')
    values = np.asarray([reward, target_q1, target_q2, alpha, log_pi_u_next], np.float64)
    if not np.isfinite(values).all() or alpha < 0:
        raise ValueError('finite/alpha')
    return reward + .99 * mask * (min(target_q1, target_q2) - alpha * log_pi_u_next)


def actor_terms(raw_base, latent, q1, q2, alpha, log_pi_u):
    submitted, _ = action_numpy(raw_base, latent)
    if not np.isfinite([q1, q2, alpha, log_pi_u]).all() or alpha < 0:
        raise ValueError('finite/alpha')
    anchor = float(10 * np.mean((submitted[:6] - np.clip(raw_base[:6], -1, 1)) ** 2))
    return dict(anchor=float(anchor), objective=float(alpha * log_pi_u - min(q1, q2) + anchor))


def validate_splits(rows):
    assignments = {}
    for row in rows:
        if row['split'] not in {'train', 'development', 'eval', 'confirmation'}:
            raise ValueError('split')
        for name in ('physical_group', 'task', 'route', 'seed'):
            key = name, row[name]
            if key in assignments and assignments[key] != row['split']:
                raise ValueError('split leakage')
            assignments[key] = row['split']


@dataclass
class Normalizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values, rows):
        validate_splits(rows)
        identities = [(row['physical_group'], row['decision']) for row in rows]
        if len(set(identities)) != len(rows) or any(row['split'] != 'train' or row['kind'] != 'current' for row in rows):
            raise ValueError('normalizer train-current unique only')
        values = np.asarray(values, np.float64)
        if values.ndim != 2 or len(values) != len(rows) or not len(rows) or not np.isfinite(values).all():
            raise ValueError('normalizer data')
        return cls(values.mean(0), np.maximum(values.std(0, ddof=0), 1e-6))

    def transform(self, values, memory_off=False):
        result = ((np.asarray(values, np.float64) - self.mean) / self.scale).astype(np.float32)
        if not np.isfinite(result).all():
            raise ValueError('normalizer finite')
        return np.zeros_like(result) if memory_off else result


def validate_replay(row):
    required = {'raw_base', 'u', 'submitted7', 'current', 'next', 'reward', 'terminated',
                'truncated', 'source', 'age', 'weight', 'split', 'target_type', 'source_hash'}
    if set(row) != required or row['split'] != 'train' or row['source'] not in {'prior', 'online'} or row['target_type'] != 'soft_q':
        raise ValueError('replay missing/forbidden/source/type')
    current = row['current']
    condition(current['fields'], current['meta'])
    validate_action(row['raw_base'], row['u'], row['submitted7'])
    end = row['terminated'] or row['truncated']
    if row['terminated'] and row['truncated'] or row['truncated'] and current['meta']['frame'] + 1 != 520:
        raise ValueError('replay boundary')
    if row['next'] is None:
        if not end:
            raise ValueError('next placeholder')
    else:
        next_row = row['next']
        condition(next_row['fields'], next_row['meta'])
        for name in ('episode', 'fold', 'checkpoint_hash', 'provider_hash'):
            if current['meta'][name] != next_row['meta'][name]:
                raise ValueError('next binding')
        if next_row['meta']['frame'] != current['meta']['frame'] + 1 or next_row['meta']['memory_version'] < current['meta']['memory_version']:
            raise ValueError('next causality/version')
    if not np.isfinite([row['reward'], row['age'], row['weight']]).all() or row['age'] < 0 or row['weight'] <= 0 or len(row['source_hash']) != 64:
        raise ValueError('replay provenance')


def sampling_contract(prior_fraction=.5, online_fraction=.5, task_balanced=True, delete_failure=False):
    if (prior_fraction, online_fraction, task_balanced, delete_failure) != (.5, .5, True, False):
        raise ValueError('sampling proposal')
    return dict(prior=.5, online=.5, task_balanced=True, keep_failures=True)


def normalize_condition(fields, meta, normalizer):
    native = condition(fields, meta)
    if normalizer.mean.shape != (1928,) or normalizer.scale.shape != (1928,):
        raise ValueError('visual/state normalizer only')
    return np.concatenate([normalizer.transform(native[:1928]), native[1928:]])


def validate_target_record(record):
    names = {'behavior_g': 'behavior_g', 'hard_return': 'pi_replan_g', 'soft_q': 'soft_q'}
    if record.get('target_type') not in names or set(record) != {'target_type', names[record['target_type']]}:
        raise ValueError('exclusive target type')
    if not np.isfinite(record[names[record['target_type']]]):
        raise ValueError('target finite')
