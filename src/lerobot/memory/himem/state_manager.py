"""
State manager for B01R rolling window memory.

Implements per-environment ring buffers of recent W tokens.
Each query recomputes from zero state over the most recent W tokens.
"""

from collections import deque
from typing import Dict, Optional, List
import warnings

from .config import CausalToken, HiMemConfig


class RollingWindowStateManager:
    """
    Manages per-environment rolling window of causal tokens.

    B01R2 semantics:
    - Each env_id maintains a ring buffer of up to W recent tokens
    - Each env_id binds to an active episode_id
    - Memory query m_k = Encoder_zero(tokens[max(0, k-W+1):k+1])[-1]
    - No persistent hidden state across windows
    - Episode reset clears that environment's buffer and allows new episode
    """

    def __init__(self, config: HiMemConfig):
        self.config = config
        self.window_size = config.window_size

        # Per-environment token buffers
        # Key: env_id, Value: deque of CausalToken (maxlen=W)
        self.env_buffers: Dict[str, deque] = {}

        # Track active episode per environment
        # Key: env_id, Value: current episode_id
        self.env_to_episode: Dict[str, str] = {}

        # Track last decision index per environment for validation
        self.last_decision_index: Dict[str, int] = {}

    def add_token(self, token: CausalToken) -> None:
        """
        Add a token to the appropriate environment buffer.

        Args:
            token: CausalToken with model_input and audit_metadata

        Raises:
            ValueError: If validation fails
        """
        # Validate token first using the config's schema
        token.validate(self.config.token_schema)

        env_id = token.audit_metadata["env_id"]
        episode_id = token.audit_metadata["episode_id"]
        decision_index = token.audit_metadata["decision_index"]
        decision_start_frame = token.audit_metadata["decision_start_frame"]

        # Check if this env has an active episode
        if env_id in self.env_to_episode:
            active_episode = self.env_to_episode[env_id]
            if episode_id != active_episode:
                raise ValueError(
                    f"Env {env_id} is bound to episode {active_episode}, "
                    f"cannot accept token from episode {episode_id}. "
                    f"Call reset_env() before starting a new episode."
                )
        else:
            # First token for this env - must be decision 0
            if decision_index != 0:
                raise ValueError(
                    f"First decision for env {env_id} must have index 0, got {decision_index}"
                )
            self.env_to_episode[env_id] = episode_id
            self.env_buffers[env_id] = deque(maxlen=self.window_size)
            self.last_decision_index[env_id] = -1

        # Validate chronological order
        last_idx = self.last_decision_index[env_id]
        if decision_index != last_idx + 1:
            raise ValueError(
                f"Non-chronological decision: env {env_id}, episode {episode_id}, "
                f"expected index {last_idx + 1}, got {decision_index}"
            )

        # Validate interval continuity with previous token
        if decision_index > 0 and len(self.env_buffers[env_id]) > 0:
            prev_token = self.env_buffers[env_id][-1]
            prev_decision_start = prev_token.audit_metadata["decision_start_frame"]
            prev_executed_end = prev_token.audit_metadata["prev_executed_end"]

            curr_prev_start = token.audit_metadata["prev_executed_start"]
            curr_prev_end = token.audit_metadata["prev_executed_end"]

            # Current token's previous interval must match what was actually recorded
            if curr_prev_start != prev_decision_start:
                raise ValueError(
                    f"Interval discontinuity: decision {decision_index} claims prev_start={curr_prev_start}, "
                    f"but previous decision {decision_index-1} started at {prev_decision_start}"
                )

            if curr_prev_end != decision_start_frame:
                raise ValueError(
                    f"Interval discontinuity: decision {decision_index} at frame {decision_start_frame} "
                    f"claims prev_end={curr_prev_end}, but they must match"
                )

        # Add token to buffer (automatically evicts oldest if at maxlen)
        self.env_buffers[env_id].append(token)

        # Update last index
        self.last_decision_index[env_id] = decision_index

    def get_recent_tokens(self, env_id: str) -> List[CausalToken]:
        """
        Get the most recent W tokens for an environment.

        Args:
            env_id: Environment identifier

        Returns:
            List of recent tokens (up to W tokens)

        Raises:
            ValueError: If env not found or buffer contains invalid tokens
        """
        if env_id not in self.env_buffers:
            raise ValueError(f"Env {env_id} not found in state manager")

        tokens = list(self.env_buffers[env_id])

        # Revalidate buffer tokens to catch mutations
        bound_episode = self.env_to_episode[env_id]
        for token in tokens:
            # Check token hasn't been mutated to another episode
            if token.audit_metadata["episode_id"] != bound_episode:
                raise ValueError(
                    f"Buffer corruption: env {env_id} bound to episode {bound_episode}, "
                    f"but buffer contains token from episode {token.audit_metadata['episode_id']}"
                )

            # Revalidate token structure
            token.validate(self.config.token_schema)

        return tokens

    def reset_env(self, env_id: str) -> None:
        """
        Reset (clear) the buffer for a specific environment.

        This should be called when an episode terminates and a new episode
        begins for the same environment.

        Args:
            env_id: Environment identifier to reset
        """
        if env_id in self.env_buffers:
            del self.env_buffers[env_id]
        if env_id in self.env_to_episode:
            del self.env_to_episode[env_id]
        if env_id in self.last_decision_index:
            del self.last_decision_index[env_id]

    def reset_all(self) -> None:
        """Clear all environment buffers."""
        self.env_buffers.clear()
        self.env_to_episode.clear()
        self.last_decision_index.clear()

    def validate_sequence(
        self,
        tokens: List[CausalToken],
        allow_cross_episode: bool = False
    ) -> None:
        """
        Validate a sequence of tokens for protocol compliance.

        Args:
            tokens: List of tokens to validate
            allow_cross_episode: If False, all tokens must be from same episode

        Raises:
            ValueError: If validation fails
        """
        if not tokens:
            raise ValueError("Cannot validate empty token sequence")

        env_ids = [t.audit_metadata["env_id"] for t in tokens]
        episode_ids = [t.audit_metadata["episode_id"] for t in tokens]
        decision_indices = [t.audit_metadata["decision_index"] for t in tokens]

        # Check single episode if required
        if not allow_cross_episode:
            unique_episodes = set(episode_ids)
            if len(unique_episodes) > 1:
                raise ValueError(
                    f"Cross-episode tokens not allowed: {unique_episodes}"
                )
            unique_envs = set(env_ids)
            if len(unique_envs) > 1:
                raise ValueError(
                    f"Cross-env tokens not allowed: {unique_envs}"
                )

        # Check chronological order within same env+episode
        for i in range(len(decision_indices) - 1):
            if env_ids[i] == env_ids[i + 1] and episode_ids[i] == episode_ids[i + 1]:
                # Same env and episode: must be consecutive
                if decision_indices[i + 1] != decision_indices[i] + 1:
                    raise ValueError(
                        f"Non-consecutive decisions: {decision_indices[i]} -> {decision_indices[i+1]}"
                    )

        # Check for duplicates
        seen = set()
        for env_id, ep_id, dec_idx in zip(env_ids, episode_ids, decision_indices):
            key = (env_id, ep_id, dec_idx)
            if key in seen:
                raise ValueError(
                    f"Duplicate token: env {env_id}, episode {ep_id}, decision {dec_idx}"
                )
            seen.add(key)

        # Validate executed action intervals
        for token in tokens:
            metadata = token.audit_metadata
            prev_start = metadata["prev_executed_start"]
            prev_end = metadata["prev_executed_end"]
            prev_count = metadata["prev_executed_count"]

            if prev_start is not None and prev_end is not None:
                # Check interval validity
                if prev_start >= prev_end:
                    raise ValueError(
                        f"Invalid execution interval: [{prev_start}, {prev_end})"
                    )

                if prev_count is not None:
                    expected_count = prev_end - prev_start
                    if prev_count != expected_count:
                        raise ValueError(
                            f"Execution count mismatch: interval [{prev_start}, {prev_end}) "
                            f"implies count {expected_count}, got {prev_count}"
                        )

    def get_window_for_decision(
        self,
        env_id: str,
        decision_index: int
    ) -> List[CausalToken]:
        """
        Get the window of tokens for computing memory at decision k.

        Returns tokens [max(0, k-W+1):k+1] for environment.

        Args:
            env_id: Environment identifier
            decision_index: Target decision index

        Returns:
            List of tokens in the window (up to W tokens)

        Raises:
            ValueError: If requested decision not in buffer
        """
        if env_id not in self.env_buffers:
            raise ValueError(f"Env {env_id} not found")

        tokens = list(self.env_buffers[env_id])

        # Find target decision
        target_idx = None
        for i, token in enumerate(tokens):
            if token.audit_metadata["decision_index"] == decision_index:
                target_idx = i
                break

        if target_idx is None:
            raise ValueError(
                f"Decision {decision_index} not found in buffer for env {env_id}"
            )

        # Return window: tokens[max(0, target_idx - W + 1):target_idx + 1]
        start_idx = max(0, target_idx - self.window_size + 1)
        return tokens[start_idx:target_idx + 1]

    def get_stats(self) -> Dict:
        """Get statistics about current state."""
        return {
            "num_envs": len(self.env_buffers),
            "env_ids": list(self.env_buffers.keys()),
            "env_to_episode": dict(self.env_to_episode),
            "buffer_sizes": {
                env_id: len(buf) for env_id, buf in self.env_buffers.items()
            },
            "window_size": self.window_size,
        }
