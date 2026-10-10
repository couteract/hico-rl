"""Pure lifecycle validation. Evidence is input, never an environment API."""
import hashlib

import numpy as np

from .performance_v2_contract import array

PHASES = ['PLANNED', 'RESTORED', 'DECISION0_VALIDATED', 'STEP1_INTENT_DURABLE',
          'STEP1_RESULT_DURABLE', 'LIVE_0_1_ENCODER_GATE', 'STEP2_INTENT_DURABLE',
          'STEP2_RESULT_DURABLE', 'QUEUE_RUNNING', 'TERMINAL_OR_TIMEOUT', 'ACCEPTED']


def action_hash(submitted):
    return hashlib.sha256(array(submitted, (7,)).tobytes()).hexdigest()


def physical_key(role, meta, identity, manifest, assignment, consumed, numeric_unique):
    entries = (role, meta, identity, manifest, assignment)
    key = role['physical_group_key']
    if not role.get('before_construction') or not key or any(entry['physical_group_key'] != key for entry in entries):
        raise ValueError('physical key posthoc/binding')
    if key in consumed or not numeric_unique:
        raise ValueError('physical key collision/numeric duplicate')
    return key


class Chain:
    def __init__(self):
        self.phase = 'PLANNED'
        self.intent = None
        self.results = []
        self.failed = False
        self.ended = False
        self.queue = False

    def fail(self):
        self.failed = True
        self.phase = 'TECHNICAL_INCOMPLETE'

    def advance(self, phase, **evidence):
        try:
            self._advance(phase, **evidence)
        except (ValueError, KeyError, TypeError, IndexError):
            if self.results or evidence.get('environment_advanced'):
                self.fail()
            raise

    def _advance(self, phase, **evidence):
        if self.failed:
            raise ValueError('technical-incomplete cannot replay/accept')
        expected = PHASES[PHASES.index(self.phase) + 1] if self.phase != 'ACCEPTED' else None
        early_end = self.phase in {'STEP1_RESULT_DURABLE', 'STEP2_RESULT_DURABLE'} and self.ended
        if phase != expected and not (early_end and phase == 'TERMINAL_OR_TIMEOUT'):
            raise ValueError('phase order/intent required')
        if self.ended and phase not in {'TERMINAL_OR_TIMEOUT', 'ACCEPTED'}:
            raise ValueError('step after end')
        if phase == 'DECISION0_VALIDATED':
            if set(evidence) != {'previous'} or evidence['previous'] is not None:
                raise ValueError('decision0 previous null')
        if 'INTENT' in phase:
            self.intent = action_hash(evidence['submitted7'])
            if evidence['action_hash'] != self.intent or not evidence['durable']:
                raise ValueError('intent bytes/durability')
        if 'RESULT' in phase:
            if action_hash(evidence['submitted7']) != self.intent or evidence['action_hash'] != self.intent or not evidence['durable'] or not evidence['environment_advanced']:
                raise ValueError('result bytes/provenance')
            if not np.isfinite(evidence['reward']) or evidence['frame'] != len(self.results) + 1:
                raise ValueError('result reward/frame')
            if evidence['terminated'] and evidence['truncated'] or not 1 <= evidence['frame'] <= 520:
                raise ValueError('result boundary')
            self.results.append(evidence.copy())
            self.ended = evidence['terminated'] or evidence['truncated']
            self.intent = None
        if phase == 'LIVE_0_1_ENCODER_GATE':
            previous = self.results[0]
            if evidence['interval'] != [0, 1] or evidence['count'] != 1 or not evidence['encoder_pass'] or evidence['result_hash'] != previous['action_hash']:
                raise ValueError('live gate provenance/interval')
            for name in ('mean7', 'last7'):
                if action_hash(evidence[name]) != previous['action_hash']:
                    raise ValueError('live action mismatch')
        if phase == 'QUEUE_RUNNING':
            trajectory = evidence['trajectory']
            if not trajectory:
                raise ValueError('queue evidence')
            for result in trajectory:
                if self.ended or result['frame'] != len(self.results) + 1 or not result['environment_advanced'] or not result['durable']:
                    raise ValueError('queue provenance/order')
                if result['intent_hash'] != action_hash(result['submitted7']) or result['action_hash'] != result['intent_hash'] or not result['intent_before_result']:
                    raise ValueError('queue intent')
                queue_position = (result['frame'] - 2) % 5 + 1
                decision = 1 + (result['frame'] - 2) // 5
                if result['queue_cursor'] != queue_position or result['queue_remaining'] != 5 - queue_position or result['decision_index'] != decision or not np.isfinite(result['reward']):
                    raise ValueError('queue cursor/reward')
                if result['terminated'] and result['truncated'] or not 1 <= result['frame'] <= 520:
                    raise ValueError('queue boundary')
                self.results.append(result.copy())
                self.ended = result['terminated'] or result['truncated']
            self.queue = True
        if phase == 'TERMINAL_OR_TIMEOUT':
            if not self.ended or not evidence['real_end'] or evidence['hard_return'] != sum(result['reward'] for result in self.results):
                raise ValueError('end/return evidence')
        if phase == 'ACCEPTED' and not (len(self.results) in {1, 2} and self.ended) and not (len(self.results) >= 3 and self.queue and self.ended):
            raise ValueError('incomplete chain')
        self.phase = phase
