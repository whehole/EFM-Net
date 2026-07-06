from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .field_map_github import HdSafeBOStyleFieldMapBuilder, RasterContactLayout
from .token_response_github import TokenResponseHead


@dataclass
class EFMNetOutput:
    prediction: Tensor
    drive: Tensor
    tokens: Tensor
    unit_field_map: Tensor
    field_map_input: Tensor
    muscle_context: Tensor
    attention_weights: Tensor


class FieldMapTokenEncoder(nn.Module):
    """Encode unit field maps into spatial tokens for EFM-Net."""

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        base_channels: int = 32,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(base_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
        )

        self.token_norm = nn.LayerNorm(hidden_dim)

    def forward(self, field_map_input: Tensor) -> Tensor:
        features = self.encoder(field_map_input)
        tokens = features.flatten(2).transpose(1, 2)
        return self.token_norm(tokens)


class EFMNet(nn.Module):
    """Electric Field Map Attention Network.

    EFM-Net implements the project's scheme A:

    signed electrode configuration
      -> HdSafeBO-style unit field map without current scaling
      -> spatial field-map tokens
      -> muscle-query cross-attention
      -> amplitude dose-response head.
    """

    def __init__(
        self,
        num_muscles: int,
        layout: RasterContactLayout,
        hidden_dim: int = 64,
        base_channels: int = 32,
        num_heads: int = 4,
        dropout: float = 0.0,
        output_activation: str = "sigmoid",
        amplitude_scale: float = 10.0,
        field_output_mode: str = "potential",
        include_coordinates: bool = True,
        include_polarity_masks: bool = False,
    ) -> None:
        super().__init__()
        self.include_coordinates = include_coordinates
        self.include_polarity_masks = include_polarity_masks
        self.map_builder = HdSafeBOStyleFieldMapBuilder(
            layout=layout,
            output_mode=field_output_mode,
        )

        height, width = layout.map_shape
        rows = torch.linspace(-1.0, 1.0, height).view(1, 1, height, 1).expand(1, 1, height, width)
        cols = torch.linspace(-1.0, 1.0, width).view(1, 1, 1, width).expand(1, 1, height, width)
        self.register_buffer("coordinate_channels", torch.cat([rows, cols], dim=1))

        in_channels = 1
        if include_polarity_masks:
            in_channels += 2
        if include_coordinates:
            in_channels += 2
        self.token_encoder = FieldMapTokenEncoder(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            base_channels=base_channels,
            dropout=dropout,
        )
        self.response_head = TokenResponseHead(
            num_muscles=num_muscles,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            output_activation=output_activation,
            amplitude_scale=amplitude_scale,
        )

    def forward(self, electrode_config: Tensor, amplitude: Tensor) -> EFMNetOutput:
        unit_field_map = self.map_builder(electrode_config)
        field_map_input = unit_field_map
        if self.include_polarity_masks:
            polarity_masks = self.map_builder.polarity_masks(electrode_config).to(dtype=unit_field_map.dtype)
            field_map_input = torch.cat([field_map_input, polarity_masks], dim=1)
        if self.include_coordinates:
            coords = self.coordinate_channels.to(dtype=unit_field_map.dtype).expand(unit_field_map.shape[0], -1, -1, -1)
            field_map_input = torch.cat([field_map_input, coords], dim=1)

        tokens = self.token_encoder(field_map_input)
        prediction, drive, muscle_context, attention_weights = self.response_head(tokens, amplitude)
        return EFMNetOutput(
            prediction=prediction,
            drive=drive,
            tokens=tokens,
            unit_field_map=unit_field_map,
            field_map_input=field_map_input,
            muscle_context=muscle_context,
            attention_weights=attention_weights,
        )
