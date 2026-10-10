"""Causal recorded-decision bookkeeping; finalized fields are retrospective only."""
from __future__ import annotations


class DecisionRecorder:
    def __init__(self):
        self.decisions = []
        self.current = None

    def begin(self, frame, planned_count):
        if self.current is not None:
            self.finish(frame, 'queue_depleted' if frame-self.current['chunk_start_frame'] == self.current['planned_action_count'] else 'early_replan')
        self.current = dict(decision_index=len(self.decisions), chunk_start_frame=frame,
                            planned_action_count=planned_count, boundary_source='recorded',
                            environment_reset=frame == 0)

    def finish(self, end, reason):
        if self.current is None:
            raise RuntimeError('No recorded prediction for executed action')
        count = end-self.current['chunk_start_frame']
        if count <= 0 or count > self.current['planned_action_count']:
            raise ValueError('Invalid actual executed count')
        self.current.update(chunk_end_frame=end, executed_action_count=count, end_reason=reason)
        self.decisions.append(self.current)
        self.current = None
