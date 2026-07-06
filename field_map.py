from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class RasterContactLayout:
    """Raster contact layout for HdSafeBO-style 2D stimulation maps.

    ``contact_anchors`` are zero-indexed ``[row, col]`` top-left contact
    coordinates. HdSafeBO's original layout uses integer pixel anchors, while
    provisional or measured paddle layouts may use floating-point coordinates.
    A contact is represented as a small rectangle with ``contact_height`` and
    ``contact_width`` sampled points.
    """

    contact_anchors: Tensor
    map_shape: tuple[int, int]
    contact_height: int = 3
    contact_width: int = 1
    name: str = "custom"


def make_hdsafebo_16_layout(device: torch.device | str | None = None) -> RasterContactLayout:
    """Return the first-16-contact HdSafeBO SCS layout used in the public code."""

    anchors = torch.tensor(
        [
            [5, 2],
            [11, 2],
            [17, 2],
            [23, 2],
            [29, 2],
            [35, 2],
            [41, 2],
            [47, 2],
            [2, 5],
            [8, 5],
            [14, 5],
            [20, 5],
            [26, 5],
            [32, 5],
            [38, 5],
            [44, 5],
        ],
        dtype=torch.long,
        device=device,
    )
    return RasterContactLayout(
        contact_anchors=anchors,
        map_shape=(52, 14),
        contact_height=3,
        contact_width=1,
        name="hdsafebo_first16",
    )


def make_hdsafebo_32_layout(device: torch.device | str | None = None) -> RasterContactLayout:
    """Return the full 32-contact raster layout from ``task/scs_task.py``."""

    anchors = torch.tensor(
        [
            [5, 2],
            [11, 2],
            [17, 2],
            [23, 2],
            [29, 2],
            [35, 2],
            [41, 2],
            [47, 2],
            [2, 5],
            [8, 5],
            [14, 5],
            [20, 5],
            [26, 5],
            [32, 5],
            [38, 5],
            [44, 5],
            [2, 8],
            [8, 8],
            [14, 8],
            [20, 8],
            [26, 8],
            [32, 8],
            [38, 8],
            [44, 8],
            [5, 11],
            [11, 11],
            [17, 11],
            [23, 11],
            [29, 11],
            [35, 11],
            [41, 11],
            [47, 11],
        ],
        dtype=torch.long,
        device=device,
    )
    return RasterContactLayout(
        contact_anchors=anchors,
        map_shape=(52, 14),
        contact_height=3,
        contact_width=1,
        name="hdsafebo_full32",
    )


def make_staggered_paddle16_layout(device: torch.device | str | None = None) -> RasterContactLayout:
    """Approximate 16-contact layout for the second figure's staggered paddle.

    This follows the HdSafeBO raster convention with integer contact anchors.
    The provisional map uses a 58x11 grid, which roughly preserves the paddle's
    physical aspect ratio from the figure (73 mm / 13.8 mm ~= 5.29). Columns
    ``2``, ``5``, and ``8`` represent left, center, and right tracks, while the
    bottom row uses ``2``, ``4``, ``6``, and ``8`` so its endpoints align with
    the lateral tracks. Replace this template with measured contact coordinates
    when the exact paddle geometry is available.

    Contact order follows the Sim4Life E1-E16 numbering in the current dataset:
    1-5   left lateral column from upper to bottom;
    6     bottom inner-left;
    7-10  center column from upper to lower;
    11-15 right lateral column from upper to bottom;
    16    bottom inner-right.

    Sim4Life dataset layout:

             E7

      E1           E11
             E8

      E2           E12
             E9

      E3           E13
            E10

      E4           E14

      E5  E6  E16  E15
    """

    anchors = torch.tensor(
        [
            [8, 2],
            [20, 2],
            [32, 2],
            [44, 2],
            [50, 2],
            [50, 4],
            [2, 5],
            [14, 5],
            [26, 5],
            [38, 5],
            [8, 8],
            [20, 8],
            [32, 8],
            [44, 8],
            [50, 8],
            [50, 6],
        ],
        dtype=torch.long,
        device=device,
    )
    return RasterContactLayout(
        contact_anchors=anchors,
        map_shape=(58, 11),
        contact_height=3,
        contact_width=1,
        name="staggered_paddle16_provisional",
    )


class HdSafeBOStyleFieldMapBuilder(nn.Module):
    """Map signed electrode configurations to unit HdSafeBO-style field maps.

    The implementation follows ``HdSafeBO-master/task/scs_task.py`` but omits
    current multiplication. In other words, it returns a unit-current
    potential-like stimulation map that can later be combined with amplitude in
    a dose-response head.
    """

    def __init__(
        self,
        layout: RasterContactLayout,
        negative_only_scale: float = 0.15,
        bipolar_scale: float = 0.2,
        eps: float = 1e-12,
        output_mode: str = "potential",
    ) -> None:
        super().__init__()
        if output_mode not in {"potential", "hdsafebo_normalized", "positional"}:
            raise ValueError("output_mode must be 'potential', 'hdsafebo_normalized', or 'positional'.")
        if layout.contact_anchors.ndim != 2 or layout.contact_anchors.shape[1] != 2:
            raise ValueError("layout.contact_anchors must have shape [E, 2].")

        self.map_shape = layout.map_shape
        self.contact_height = layout.contact_height
        self.contact_width = layout.contact_width
        self.negative_only_scale = negative_only_scale
        self.bipolar_scale = bipolar_scale
        self.eps = eps
        self.output_mode = output_mode

        contact_pixels = self._expand_contact_pixels(layout.contact_anchors.float())
        grid_points = self._make_grid_points(
            layout.map_shape,
            device=contact_pixels.device,
            dtype=contact_pixels.dtype,
        )
        dist_sq = (grid_points[:, None, None, :] - contact_pixels[None, :, :, :]).square().sum(dim=-1)

        self.register_buffer("contact_anchors", layout.contact_anchors.float())
        self.register_buffer("contact_pixels", contact_pixels.float())
        self.register_buffer("grid_points", grid_points.float())
        self.register_buffer("dist_sq", dist_sq.float())

    @property
    def num_contacts(self) -> int:
        return int(self.contact_anchors.shape[0])

    def _expand_contact_pixels(self, anchors: Tensor) -> Tensor:
        row_offsets = torch.arange(self.contact_height, device=anchors.device, dtype=anchors.dtype)
        col_offsets = torch.arange(self.contact_width, device=anchors.device, dtype=anchors.dtype)
        dr, dc = torch.meshgrid(row_offsets, col_offsets, indexing="ij")
        offsets = torch.stack([dr.reshape(-1), dc.reshape(-1)], dim=-1)
        pixels = anchors[:, None, :] + offsets[None, :, :]

        height, width = self.map_shape
        if int(pixels[..., 0].max()) >= height or int(pixels[..., 1].max()) >= width:
            raise ValueError("Contact pixels exceed map_shape.")
        return pixels

    @staticmethod
    def _make_grid_points(
        map_shape: tuple[int, int],
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> Tensor:
        height, width = map_shape
        rows = torch.arange(height, device=device, dtype=dtype)
        cols = torch.arange(width, device=device, dtype=dtype)
        grid_rows, grid_cols = torch.meshgrid(rows, cols, indexing="ij")
        return torch.stack([grid_rows.reshape(-1), grid_cols.reshape(-1)], dim=-1)

    def forward(self, electrode_config: Tensor) -> Tensor:
        if electrode_config.ndim != 2:
            raise ValueError("electrode_config must have shape [B, E].")
        if electrode_config.shape[1] != self.num_contacts:
            raise ValueError("electrode_config does not match the layout contact count.")

        signed = electrode_config.to(dtype=self.dist_sq.dtype)
        positive = (signed > 0)[:, None, :, None]
        negative = (signed < 0)[:, None, :, None]
        has_positive = positive.any(dim=(2, 3))
        has_negative = negative.any(dim=(2, 3))

        zero_distance = self.dist_sq[None] <= self.eps
        positive_hit = (zero_distance & positive).any(dim=(2, 3))
        negative_hit = (zero_distance & negative).any(dim=(2, 3))

        inv_distance = torch.rsqrt(self.dist_sq.clamp_min(self.eps))[None]
        inv_distance = inv_distance.masked_fill(zero_distance, 0.0)

        positive_sum = (inv_distance * positive.to(inv_distance.dtype)).sum(dim=(2, 3))
        negative_sum = (inv_distance * negative.to(inv_distance.dtype)).sum(dim=(2, 3))
        pixels_per_contact = int(self.contact_pixels.shape[1])
        negative_count = (negative.to(inv_distance.dtype).sum(dim=(2, 3)) * pixels_per_contact).clamp_min(1.0)
        negative_mean = negative_sum / negative_count

        potential = electrode_config.new_zeros(electrode_config.shape[0], self.dist_sq.shape[0])

        negative_only = has_negative & ~has_positive
        bipolar = has_negative & has_positive

        potential = torch.where(
            negative_only,
            -self.negative_only_scale * negative_sum,
            potential,
        )
        bipolar_value = -self.bipolar_scale * negative_mean / (positive_sum + negative_mean).clamp_min(self.eps)
        potential = torch.where(bipolar, bipolar_value, potential)

        potential = torch.where(positive_hit & bipolar, torch.zeros_like(potential), potential)
        potential = torch.where(negative_hit & has_negative, -torch.ones_like(potential), potential)

        height, width = self.map_shape
        potential_map = potential.reshape(electrode_config.shape[0], 1, height, width)

        if self.output_mode == "potential":
            return potential_map
        if self.output_mode == "hdsafebo_normalized":
            return (potential_map + 5.0) / 5.0
        return self._positional_map(electrode_config)

    def polarity_masks(self, electrode_config: Tensor) -> Tensor:
        """Return cathode/anode contact masks shaped ``[B, 2, H, W]``."""

        if electrode_config.ndim != 2:
            raise ValueError("electrode_config must have shape [B, E].")
        if electrode_config.shape[1] != self.num_contacts:
            raise ValueError("electrode_config does not match the layout contact count.")

        batch_size = int(electrode_config.shape[0])
        height, width = self.map_shape
        masks = electrode_config.new_zeros(batch_size, 2, height, width)
        signed = electrode_config
        for contact_idx, pixels in enumerate(self.contact_pixels.round().long().tolist()):
            cathode_value = (signed[:, contact_idx] < 0).to(masks.dtype).view(batch_size, 1)
            anode_value = (signed[:, contact_idx] > 0).to(masks.dtype).view(batch_size, 1)
            rows = [pixel[0] for pixel in pixels]
            cols = [pixel[1] for pixel in pixels]
            masks[:, 0, rows, cols] = cathode_value
            masks[:, 1, rows, cols] = anode_value
        return masks

    def _positional_map(self, electrode_config: Tensor) -> Tensor:
        batch_size = int(electrode_config.shape[0])
        height, width = self.map_shape
        out = electrode_config.new_zeros(batch_size, 1, height, width)
        signed = electrode_config
        for contact_idx, pixels in enumerate(self.contact_pixels.round().long().tolist()):
            value = signed[:, contact_idx].view(batch_size, 1, 1)
            rows = [pixel[0] for pixel in pixels]
            cols = [pixel[1] for pixel in pixels]
            out[:, 0, rows, cols] = value
        return out
