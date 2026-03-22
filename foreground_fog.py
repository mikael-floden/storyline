"""
Foreground fog overlays for subtle scene depth.

These layers render after the actor so they sit in front of the character,
while still being part of the overscanned scene that the camera presents.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass
class FogLayerSpec:
    """Configuration for a single scrolling fog layer."""

    filename: str
    speed_px_per_sec: float
    max_alpha: int


class _ScrollingFogLayer:
    """Single horizontally scrolling fog layer."""

    def __init__(self, surface, speed_px_per_sec: float):
        self.surface = surface
        self.speed_px_per_sec = speed_px_per_sec
        self.mirror_surface = None
        self.x_offset = 0.0

    def update(self, dt: float) -> None:
        """Advance the fog layer offset."""
        self.x_offset -= self.speed_px_per_sec * dt

    def draw(self, surf) -> None:
        """Tile the fog layer across the target surface."""
        import pygame

        if self.mirror_surface is None:
            self.mirror_surface = pygame.transform.flip(self.surface, True, False)

        tile_width = self.surface.get_width()
        start_index = int((-self.x_offset) // tile_width)
        x = self.x_offset + (start_index * tile_width)
        tile_index = start_index

        while x < surf.get_width():
            tile_surface = self.surface if tile_index % 2 == 0 else self.mirror_surface
            surf.blit(tile_surface, (int(round(x)), 0))
            x += tile_width
            tile_index += 1


class ForegroundFog:
    """Multiple scrolling fog overlays rendered in front of the actor."""

    def __init__(self, render_width: int, render_height: int, asset_dir: str):
        import pygame

        self.render_width = render_width
        self.render_height = render_height
        self.layers: list[_ScrollingFogLayer] = []

        specs = [
            FogLayerSpec("mist_0.png", speed_px_per_sec=10.0, max_alpha=82),
            FogLayerSpec("mist_1.png", speed_px_per_sec=18.0, max_alpha=58),
        ]

        for spec in specs:
            path = os.path.join(asset_dir, spec.filename)
            if not os.path.isfile(path):
                print(f"Warning: Foreground fog layer not found: {path}")
                continue

            surface = self._load_surface(path, spec.max_alpha)

            if surface.get_height() != render_height:
                scale_factor = render_height / surface.get_height()
                surface = pygame.transform.smoothscale(
                    surface,
                    (
                        max(1, int(round(surface.get_width() * scale_factor))),
                        render_height,
                    ),
                )

            self.layers.append(_ScrollingFogLayer(surface, spec.speed_px_per_sec))

    def update(self, dt: float) -> None:
        """Advance all fog layers."""
        for layer in self.layers:
            layer.update(dt)

    def draw(self, surf) -> None:
        """Draw all fog layers from back to front."""
        for layer in self.layers:
            layer.draw(surf)

    def _load_surface(self, path: str, max_alpha: int):
        """Load a fog image and synthesize a stronger alpha mask when needed."""
        import pygame

        image = Image.open(path).convert("RGBA")
        arr = np.array(image, dtype=np.uint8)
        alpha = arr[..., 3]

        # Some fog exports carry almost no usable alpha. In that case, rebuild
        # it from luminance so the layer is visible in-game.
        if int(alpha.max()) < 16:
            luma = arr[..., :3].mean(axis=2)
            luma_min = float(luma.min())
            luma_max = float(luma.max())

            if luma_max > luma_min:
                normalized = (luma - luma_min) / (luma_max - luma_min)
            else:
                normalized = np.zeros_like(luma, dtype=np.float32)

            rebuilt_alpha = np.power(normalized, 1.8) * max_alpha
            arr[..., 3] = np.clip(rebuilt_alpha, 0, 255).astype(np.uint8)

        surface = pygame.image.fromstring(
            arr.tobytes(),
            image.size,
            "RGBA",
        ).convert_alpha()
        return surface
