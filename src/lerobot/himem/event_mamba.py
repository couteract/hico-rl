"""
Event-Mamba Core Module

Processes decision events with limited window state management.
Causal protocol: events provided chronologically, no future leakage.

B01 scope: Independent module with CPU/GPU verification.
NOT connected to actor/critic in this stage.
"""
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class EventMambaConfig:
    """Event-Mamba configuration"""
    hidden_dim: int = 256
    state_dim: int = 64  # SSM state dimension
    window_size: int = 8  # Maximum events to retain
    backend: str = "gru"  # "gru" | "mamba2"
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5


class EventEncoder(nn.Module):
    """Encode decision event into fixed-size representation"""

    def __init__(self, action_dim: int, obs_keys: List[str], hidden_dim: int):
        super().__init__()
        self.action_dim = action_dim
        self.obs_keys = obs_keys
        self.hidden_dim = hidden_dim

        # Simple MLP encoder for B01
        # Future: Replace with transformer or attention
        self.action_encoder = nn.Linear(action_dim, hidden_dim // 2)
        self.obs_encoder = nn.Linear(128, hidden_dim // 2)  # Placeholder obs dim
        self.fusion = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, event: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encode single event into vector

        Args:
            event: Dict with 'actions' [chunk_size, action_dim] and 'observations'

        Returns:
            encoded: [hidden_dim] event representation
        """
        # Mean-pool actions over chunk
        actions = event['actions']  # [chunk_size, action_dim]
        action_feat = self.action_encoder(actions.mean(dim=0))  # [hidden_dim//2]

        # Placeholder: assume observations already embedded
        # Real implementation would encode image/proprio observations
        obs_feat = torch.randn(self.hidden_dim // 2, device=actions.device)

        # Concatenate and fuse
        combined = torch.cat([action_feat, obs_feat], dim=-1)  # [hidden_dim]
        encoded = self.fusion(combined)
        return self.norm(encoded)


class GRUMemory(nn.Module):
    """CPU reference implementation using GRU"""

    def __init__(self, hidden_dim: int, state_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.gru = nn.GRU(hidden_dim, state_dim, batch_first=True)
        self.proj = nn.Linear(state_dim, hidden_dim)

    def forward(
        self,
        event_seq: torch.Tensor,
        prev_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process event sequence causally

        Args:
            event_seq: [batch, seq_len, hidden_dim]
            prev_state: [1, batch, state_dim] or None

        Returns:
            output: [batch, seq_len, hidden_dim]
            final_state: [1, batch, state_dim]
        """
        gru_out, h_n = self.gru(event_seq, prev_state)  # [B, L, state_dim], [1, B, state_dim]
        output = self.proj(gru_out)  # [B, L, hidden_dim]
        return output, h_n


class SimplifiedSSM(nn.Module):
    """
    Simplified State Space Model (SSM) for B01 verification

    This is a pedagogical implementation that captures SSM's core idea:
    - Selective state updates based on input
    - Causal temporal processing
    - Linear complexity in sequence length

    B01 scope: Pure PyTorch implementation for CPU/GPU verification.
    Does NOT use optimized CUDA kernels (causal-conv1d, selective-scan).
    Production version in B02+ can swap to mamba-ssm with CUDA kernels.

    Architecture inspired by Mamba2 but simplified:
    - Causal 1D conv for local context
    - Input-dependent state transitions (selectivity)
    - Linear output projection
    """

    def __init__(self, dim: int, state_dim: int, conv_kernel: int = 4):
        super().__init__()
        self.dim = dim
        self.state_dim = state_dim
        self.conv_kernel = conv_kernel

        # Causal conv1d for local context (manual implementation)
        self.conv_weight = nn.Parameter(torch.randn(dim, conv_kernel) * 0.02)
        self.conv_bias = nn.Parameter(torch.zeros(dim))

        # SSM parameters (input-dependent via linear projections)
        self.x_proj = nn.Linear(dim, state_dim)  # Input to state
        self.dt_proj = nn.Linear(dim, state_dim)  # Delta (step size)
        self.A = nn.Parameter(torch.randn(state_dim))  # State transition
        self.D = nn.Parameter(torch.ones(dim))  # Skip connection

        # Output projection
        self.out_proj = nn.Linear(state_dim, dim)

    def causal_conv1d(self, x: torch.Tensor) -> torch.Tensor:
        """
        Manual causal conv1d: [B, L, D] -> [B, L, D]
        Each position only sees current and past (causal)
        """
        B, L, D = x.shape
        K = self.conv_kernel

        # Pad left to maintain causality
        x_padded = torch.cat([
            torch.zeros(B, K - 1, D, device=x.device, dtype=x.dtype),
            x
        ], dim=1)  # [B, L+K-1, D]

        # Apply conv manually per position
        out = []
        for i in range(L):
            # Extract window [i:i+K] from padded sequence
            window = x_padded[:, i:i+K, :]  # [B, K, D]
            # Weighted sum: [B, K, D] * [D, K] -> [B, D]
            conv_out = (window * self.conv_weight.T.unsqueeze(0)).sum(dim=1)
            out.append(conv_out)

        result = torch.stack(out, dim=1) + self.conv_bias  # [B, L, D]
        return result

    def forward(
        self,
        x: torch.Tensor,
        prev_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, L, dim]
            prev_state: [B, state_dim] or None

        Returns:
            y: [B, L, dim]
            final_state: [B, state_dim]
        """
        B, L, D = x.shape

        # Causal convolution for local context
        x_conv = self.causal_conv1d(x)  # [B, L, D]
        x_conv = torch.nn.functional.silu(x_conv)

        # Initialize state
        if prev_state is None:
            h = torch.zeros(B, self.state_dim, device=x.device, dtype=x.dtype)
        else:
            h = prev_state  # [B, state_dim]

        # Process sequence causally (recurrent loop)
        outputs = []
        for t in range(L):
            x_t = x_conv[:, t, :]  # [B, D]

            # Input-dependent parameters (selectivity)
            x_state = self.x_proj(x_t)  # [B, state_dim]
            dt = torch.sigmoid(self.dt_proj(x_t))  # [B, state_dim]

            # SSM state update: h_t = A * h_{t-1} + dt * x_t
            # (Simplified: real Mamba uses discretization and more complex update)
            A_step = torch.exp(self.A * dt)  # [B, state_dim]
            h = A_step * h + dt * x_state  # [B, state_dim]

            # Output from state
            y_t = self.out_proj(h)  # [B, D]

            # Skip connection (Mamba's D parameter)
            y_t = y_t + self.D * x_t  # [B, D]

            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)  # [B, L, D]
        return y, h  # [B, L, D], [B, state_dim]


class Mamba2Memory(nn.Module):
    """
    Mamba2-style memory backend using SimplifiedSSM

    B01 implementation: Pure PyTorch for verification.
    No dependency on causal-conv1d or mamba-ssm CUDA kernels.

    Future optimization (B02+):
    - Replace SimplifiedSSM with mamba_ssm.Mamba2 + CUDA kernels
    - Add inference_params for KV caching
    - Profile and optimize batch processing
    """

    def __init__(self, hidden_dim: int, state_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim

        self.proj_in = nn.Linear(hidden_dim, state_dim)
        self.ssm = SimplifiedSSM(dim=state_dim, state_dim=state_dim // 2, conv_kernel=4)
        self.proj_out = nn.Linear(state_dim, hidden_dim)
        self.norm = nn.LayerNorm(state_dim)

    def forward(
        self,
        event_seq: torch.Tensor,
        prev_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            event_seq: [batch, seq_len, hidden_dim]
            prev_state: [1, batch, state_dim] or None (GRU-compatible format)

        Returns:
            output: [batch, seq_len, hidden_dim]
            final_state: [1, batch, state_dim]
        """
        B, L, D = event_seq.shape

        # Project to SSM dimension
        x = self.proj_in(event_seq)  # [B, L, state_dim]
        x = self.norm(x)

        # Convert prev_state from [1, B, state_dim] to [B, state_dim//2]
        if prev_state is not None:
            ssm_state = prev_state.squeeze(0)[:, :self.state_dim // 2]
        else:
            ssm_state = None

        # SSM forward
        ssm_out, final_ssm_state = self.ssm(x, ssm_state)  # [B, L, state_dim], [B, state_dim//2]

        # Project back
        output = self.proj_out(ssm_out)  # [B, L, hidden_dim]

        # Pad final_state to match expected shape [1, B, state_dim]
        # (Padding because SSM state is state_dim//2, but interface expects state_dim)
        final_state_padded = torch.cat([
            final_ssm_state,
            torch.zeros(B, self.state_dim - self.state_dim // 2,
                       device=final_ssm_state.device, dtype=final_ssm_state.dtype)
        ], dim=-1)  # [B, state_dim]
        final_state = final_state_padded.unsqueeze(0)  # [1, B, state_dim]

        return output, final_state


class EventMamba(nn.Module):
    """
    Event-Mamba: Limited window state management

    Input: Chronological sequence of decision events
    Output: Memory-enriched representations for recent window

    Causal guarantee: Model never attends to future events.
    Window limit: Retains at most W=8 recent events.
    """

    def __init__(self, config: EventMambaConfig, action_dim: int, obs_keys: List[str]):
        super().__init__()
        self.config = config
        self.action_dim = action_dim
        self.obs_keys = obs_keys

        # Event encoder
        self.encoder = EventEncoder(action_dim, obs_keys, config.hidden_dim)

        # Memory backend
        if config.backend == "gru":
            self.memory = GRUMemory(config.hidden_dim, config.state_dim)
        elif config.backend == "mamba2":
            self.memory = Mamba2Memory(config.hidden_dim, config.state_dim)
        else:
            raise ValueError(f"Unknown backend: {config.backend}")

        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        events: List[Dict[str, torch.Tensor]],
        prev_memory_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process event sequence with limited window

        Args:
            events: List of event dicts, chronologically ordered
            prev_memory_state: Previous memory state or None

        Returns:
            memory_output: [batch=1, seq_len, hidden_dim] representations
            memory_state: [1, batch=1, state_dim] final state for next call
        """
        # Apply window limit
        if len(events) > self.config.window_size:
            events = events[-self.config.window_size:]

        # Encode events
        encoded = []
        for event in events:
            enc = self.encoder(event)  # [hidden_dim]
            encoded.append(enc)

        if len(encoded) == 0:
            # Empty sequence: return zeros
            dummy = torch.zeros(1, 1, self.config.hidden_dim)
            dummy_state = torch.zeros(1, 1, self.config.state_dim)
            return dummy, dummy_state

        # Stack into sequence
        event_seq = torch.stack(encoded, dim=0).unsqueeze(0)  # [1, seq_len, hidden_dim]
        event_seq = self.dropout(event_seq)

        # Process through memory
        memory_output, memory_state = self.memory(event_seq, prev_memory_state)

        return memory_output, memory_state

    def reset_memory(self):
        """Reset memory state (call at episode start)"""
        return None


def create_event_mamba(
    action_dim: int,
    obs_keys: List[str],
    backend: str = "gru",
    hidden_dim: int = 256,
    window_size: int = 8
) -> EventMamba:
    """
    Factory function for Event-Mamba

    Args:
        action_dim: Action space dimension
        obs_keys: List of observation keys
        backend: "gru" for CPU reference, "mamba2" for GPU production
        hidden_dim: Hidden dimension
        window_size: Maximum events to retain

    Returns:
        EventMamba instance
    """
    config = EventMambaConfig(
        hidden_dim=hidden_dim,
        state_dim=hidden_dim // 4,
        window_size=window_size,
        backend=backend,
        dropout=0.1
    )
    return EventMamba(config, action_dim, obs_keys)
