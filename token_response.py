from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class MuscleQueryCrossAttention(nn.Module):
    """Learnable muscle queries attend to stimulation tokens."""

    def __init__(
        self,
        num_muscles: int,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        self.muscle_queries = nn.Parameter(torch.randn(num_muscles, hidden_dim) * 0.02)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, tokens: Tensor) -> tuple[Tensor, Tensor]:
        batch_size = int(tokens.shape[0])
        queries = self.muscle_queries.unsqueeze(0).expand(batch_size, -1, -1)
        attended, weights = self.attention(
            query=queries,
            key=tokens,
            value=tokens,
            need_weights=True,
            average_attn_weights=False,
        )
        context = self.norm1(queries + attended)
        context = self.norm2(context + self.ffn(context))
        # Average over heads for easier downstream visualization.
        return context, weights.mean(dim=1)


class DoseResponseHead(nn.Module):
    """Current-conditioned MLP response head.

    The final sigmoid bounds predictions to [0, 1]; the recruitment curve
    shape is learned from the muscle context and current amplitude instead of
    being fixed to a parametric sigmoid.
    """

    def __init__(
        self,
        num_muscles: int,
        hidden_dim: int,
        output_activation: str = "sigmoid",
        amplitude_scale: float = 10.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if output_activation not in {"sigmoid", "saturation_sigmoid", "softplus", "relu", "identity"}:
            raise ValueError(
                "output_activation must be 'sigmoid', 'saturation_sigmoid', 'softplus', 'relu', or 'identity'."
            )
        self.num_muscles = num_muscles
        self.output_activation = output_activation
        self.amplitude_scale = amplitude_scale
        self.current_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.film = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.response_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.raw_saturation = nn.Parameter(torch.zeros(num_muscles))
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, muscle_context: Tensor, amplitude: Tensor) -> tuple[Tensor, Tensor]:
        if amplitude.ndim == 1:
            amplitude = amplitude[:, None]
        normalized_amplitude = amplitude / self.amplitude_scale
        current_features = torch.cat([normalized_amplitude, normalized_amplitude.square()], dim=-1)
        current_embedding = self.current_encoder(current_features)
        gamma, beta = self.film(current_embedding).chunk(2, dim=-1)
        gamma = gamma[:, None, :]
        beta = beta[:, None, :]
        normalized_context = self.norm(muscle_context)
        modulated_context = (1.0 + gamma) * normalized_context + beta
        logits = self.response_mlp(modulated_context).squeeze(-1)

        if self.output_activation in {"sigmoid", "saturation_sigmoid"}:
            response = torch.sigmoid(logits)
            if self.output_activation == "saturation_sigmoid":
                response = F.softplus(self.raw_saturation)[None, :] * response
            return response, logits
        if self.output_activation == "softplus":
            return F.softplus(logits), logits
        if self.output_activation == "relu":
            return F.relu(logits), logits
        return logits, logits


class TokenResponseHead(nn.Module):
    """Shared downstream head for stimulation-token models."""

    def __init__(
        self,
        num_muscles: int,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.0,
        output_activation: str = "sigmoid",
        amplitude_scale: float = 10.0,
    ) -> None:
        super().__init__()
        self.muscle_attention = MuscleQueryCrossAttention(
            num_muscles=num_muscles,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.dose_response = DoseResponseHead(
            num_muscles=num_muscles,
            hidden_dim=hidden_dim,
            output_activation=output_activation,
            amplitude_scale=amplitude_scale,
            dropout=dropout,
        )

    def forward(self, tokens: Tensor, amplitude: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        muscle_context, attention_weights = self.muscle_attention(tokens)
        prediction, drive = self.dose_response(muscle_context, amplitude)
        return prediction, drive, muscle_context, attention_weights
