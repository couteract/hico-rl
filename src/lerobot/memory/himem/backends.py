"""
Memory backend implementations for B01R.

Backends:
1. gru_reference: PyTorch GRU for CPU protocol reference
2. mamba2: Official mamba_ssm.Mamba2 (requires use_mem_eff_path=False)
3. simplified_ssm_experimental: Pure PyTorch prototype (not production)
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Literal
import warnings


class GRUMemoryBackend(nn.Module):
    """
    GRU-based memory backend for CPU reference.

    This is the protocol reference implementation.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        self.backend_type = "gru_reference"

    def forward(
        self,
        x: torch.Tensor,
        prev_state: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through GRU.

        Args:
            x: Input tokens, shape [batch, seq_len, input_dim]
            prev_state: NOT USED in B01R (rolling window from zero)

        Returns:
            output: [batch, seq_len, hidden_dim]
            final_state: [num_layers, batch, hidden_dim] (not used in rolling window)
        """
        if prev_state is not None:
            warnings.warn(
                "GRUMemoryBackend received prev_state but B01R implements "
                "rolling window from zero. Ignoring prev_state."
            )

        # Always start from zero state
        batch_size = x.size(0)
        h0 = torch.zeros(
            self.num_layers, batch_size, self.hidden_dim,
            device=x.device, dtype=x.dtype
        )

        output, final_state = self.gru(x, h0)
        return output, final_state

    def get_backend_manifest(self) -> Dict[str, Any]:
        """Return backend provenance information."""
        return {
            "backend_type": "gru_reference",
            "implementation": "torch.nn.GRU",
            "torch_version": torch.__version__,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "total_parameters": sum(p.numel() for p in self.parameters()),
        }


class Mamba2Backend(nn.Module):
    """
    Official Mamba2 backend using mamba_ssm library.

    Requires mamba_ssm>=2.2.0 installed.
    Uses use_mem_eff_path=False because causal-conv1d is not installed.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        num_layers: int = 2,
        use_mem_eff_path: bool = False,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()

        # Lazy import to avoid CUDA/Triton initialization in CPU-only processes
        try:
            from mamba_ssm import Mamba2
            import mamba_ssm
            self.mamba_ssm_version = mamba_ssm.__version__
            self.mamba_ssm_path = mamba_ssm.__file__
        except ImportError as e:
            raise ImportError(
                f"mamba_ssm not installed. Please install with: pip install mamba-ssm\n"
                f"Error: {e}"
            )

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.headdim = headdim
        self.num_layers = num_layers
        self.use_mem_eff_path = use_mem_eff_path

        # Create Mamba2 layers with residual connections
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = Mamba2(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                headdim=headdim,
                use_mem_eff_path=use_mem_eff_path,
            )
            self.layers.append(layer)

        # Layer norms for residual connections
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model, eps=layer_norm_eps)
            for _ in range(num_layers)
        ])

        # Dropout
        self.dropout = nn.Dropout(dropout)

        self.backend_type = "mamba2"

        # Verify this is actually official Mamba2
        from mamba_ssm import Mamba2 as OfficialMamba2
        for layer in self.layers:
            assert isinstance(layer, OfficialMamba2), (
                f"Backend mamba2 must use official mamba_ssm.Mamba2, "
                f"got {type(layer)}"
            )

    def forward(
        self,
        x: torch.Tensor,
        prev_state: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through Mamba2 layers.

        Args:
            x: Input tokens, shape [batch, seq_len, d_model]
            prev_state: NOT USED in B01R (rolling window from zero)

        Returns:
            output: [batch, seq_len, d_model]
            final_state: Dummy state (not used in rolling window)
        """
        if prev_state is not None:
            warnings.warn(
                "Mamba2Backend received prev_state but B01R implements "
                "rolling window from zero. Ignoring prev_state."
            )

        # Process through Mamba2 layers with residuals
        output = x
        for layer, norm in zip(self.layers, self.layer_norms):
            residual = output
            output = norm(output)
            output = layer(output) + residual
            output = self.dropout(output)

        # Return dummy state (not used in rolling window)
        batch_size = x.size(0)
        dummy_state = torch.zeros(
            1, batch_size, self.d_model,
            device=x.device, dtype=x.dtype
        )

        return output, dummy_state

    def get_backend_manifest(self) -> Dict[str, Any]:
        """Return backend provenance information."""
        return {
            "backend_type": "mamba2",
            "implementation": "mamba_ssm.Mamba2",
            "mamba_ssm_version": self.mamba_ssm_version,
            "mamba_ssm_path": self.mamba_ssm_path,
            "torch_version": torch.__version__,
            "d_model": self.d_model,
            "d_state": self.d_state,
            "d_conv": self.d_conv,
            "expand": self.expand,
            "headdim": self.headdim,
            "num_layers": self.num_layers,
            "use_mem_eff_path": self.use_mem_eff_path,
            "causal_conv1d_available": self.use_mem_eff_path,
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
        }


class SimplifiedSSMExperimental(nn.Module):
    """
    Simplified state space model (pure PyTorch prototype).

    This is an EXPERIMENTAL backend for prototyping.
    NOT recommended for production use.
    Use gru_reference for CPU or mamba2 for GPU production.
    """

    def __init__(
        self,
        input_dim: int,
        state_dim: int,
        num_layers: int = 1,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.num_layers = num_layers

        warnings.warn(
            "SimplifiedSSMExperimental is a pure PyTorch prototype. "
            "Use gru_reference for CPU or mamba2 for production GPU workloads."
        )

        # Simplified SSM parameters (not optimized)
        self.input_proj = nn.Linear(input_dim, state_dim)
        self.output_proj = nn.Linear(state_dim, input_dim)
        self.layer_norm = nn.LayerNorm(input_dim, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)

        # State transition (simplified)
        self.state_transition = nn.Parameter(torch.randn(state_dim, state_dim) * 0.01)

        self.backend_type = "simplified_ssm_experimental"

    def forward(
        self,
        x: torch.Tensor,
        prev_state: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass (simplified SSM logic).

        Args:
            x: Input tokens, shape [batch, seq_len, input_dim]
            prev_state: NOT USED in B01R

        Returns:
            output: [batch, seq_len, input_dim]
            final_state: [1, batch, state_dim]
        """
        if prev_state is not None:
            warnings.warn(
                "SimplifiedSSMExperimental received prev_state but B01R implements "
                "rolling window from zero. Ignoring prev_state."
            )

        batch, seq_len, _ = x.shape

        # Project to state space
        h = self.input_proj(x)  # [batch, seq_len, state_dim]

        # Simple recurrence
        outputs = []
        state = torch.zeros(batch, self.state_dim, device=x.device, dtype=x.dtype)

        for t in range(seq_len):
            state = torch.tanh(h[:, t, :] + state @ self.state_transition.t())
            outputs.append(state)

        states = torch.stack(outputs, dim=1)  # [batch, seq_len, state_dim]

        # Project back
        output = self.output_proj(states)  # [batch, seq_len, input_dim]
        output = self.layer_norm(output + x)  # Residual
        output = self.dropout(output)

        final_state = states[:, -1:, :].transpose(0, 1)  # [1, batch, state_dim]

        return output, final_state

    def get_backend_manifest(self) -> Dict[str, Any]:
        """Return backend provenance information."""
        return {
            "backend_type": "simplified_ssm_experimental",
            "implementation": "custom_pytorch_prototype",
            "warning": "NOT RECOMMENDED FOR PRODUCTION",
            "torch_version": torch.__version__,
            "input_dim": self.input_dim,
            "state_dim": self.state_dim,
            "num_layers": self.num_layers,
            "total_parameters": sum(p.numel() for p in self.parameters()),
        }


def create_backend(
    backend_type: Literal["gru_reference", "mamba2", "simplified_ssm_experimental"],
    input_dim: int,
    output_dim: int,
    **kwargs
) -> nn.Module:
    """
    Factory function to create memory backend.

    Args:
        backend_type: Backend selection
        input_dim: Token dimension
        output_dim: Memory output dimension
        **kwargs: Backend-specific parameters

    Returns:
        Memory backend module
    """
    if backend_type == "gru_reference":
        return GRUMemoryBackend(
            input_dim=input_dim,
            hidden_dim=output_dim,
            num_layers=kwargs.get("num_layers", 1),
            dropout=kwargs.get("dropout", 0.1),
        )

    elif backend_type == "mamba2":
        return Mamba2Backend(
            d_model=output_dim,  # Assumes input_dim == output_dim via projection
            d_state=kwargs.get("d_state", 64),
            d_conv=kwargs.get("d_conv", 4),
            expand=kwargs.get("expand", 2),
            headdim=kwargs.get("headdim", 64),
            num_layers=kwargs.get("num_layers", 2),
            use_mem_eff_path=kwargs.get("use_mem_eff_path", False),
            dropout=kwargs.get("dropout", 0.1),
            layer_norm_eps=kwargs.get("layer_norm_eps", 1e-5),
        )

    elif backend_type == "simplified_ssm_experimental":
        warnings.warn(
            "Using simplified_ssm_experimental backend. "
            "This is a prototype and not recommended for production."
        )
        return SimplifiedSSMExperimental(
            input_dim=input_dim,
            state_dim=kwargs.get("state_dim", output_dim // 4),
            num_layers=kwargs.get("num_layers", 1),
            dropout=kwargs.get("dropout", 0.1),
            layer_norm_eps=kwargs.get("layer_norm_eps", 1e-5),
        )

    else:
        raise ValueError(
            f"Unknown backend: {backend_type}. "
            f"Must be one of: gru_reference, mamba2, simplified_ssm_experimental"
        )
