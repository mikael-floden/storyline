"""
Scene camera helpers for subtle cinematic motion.

Provides a lightweight camera that can add a small amount of look-ahead,
micro drift, and gentle zoom without changing the underlying scene logic.
"""

from __future__ import annotations

import math


class SceneCamera:
    """Camera with subtle motion, micro zoom, and overscan presentation."""

    def __init__(
        self,
        screen_width: int,
        screen_height: int,
        *,
        actor_screen_x_ratio: float = 1 / 3,
        floor_offset: int = 90,
        overscan_ratio: float = 0.08,
        base_zoom: float = 1.0,
        zoom_amplitude: float = 0.012,
        sway_x: float = 6.0,
        sway_y: float = 4.0,
        max_look_ahead_px: float = 22.0,
        look_ahead_response: float = 2.4,
        enable_motion: bool = True,
        enable_look_ahead: bool = True,
    ):
        """Initialize a scene camera.

        Args:
            screen_width: Final on-screen width.
            screen_height: Final on-screen height.
            actor_screen_x_ratio: Default horizontal actor framing.
            floor_offset: Distance from the bottom edge to the visible ground line.
            overscan_ratio: Extra scene area rendered beyond the visible frame.
            base_zoom: Baseline zoom multiplier.
            zoom_amplitude: Maximum zoom variation around the baseline.
            sway_x: Horizontal drift in pixels.
            sway_y: Vertical drift in pixels.
            max_look_ahead_px: Maximum forward framing shift in pixels.
            look_ahead_response: Interpolation speed for look-ahead changes.
            enable_motion: Enable drift and zoom modulation.
            enable_look_ahead: Enable forward framing bias.
        """
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.actor_screen_x_ratio = actor_screen_x_ratio
        self.floor_offset = floor_offset
        self.overscan_ratio = overscan_ratio
        self.base_zoom = base_zoom
        self.zoom_amplitude = zoom_amplitude
        self.sway_x = sway_x
        self.sway_y = sway_y
        self.max_look_ahead_px = max_look_ahead_px
        self.look_ahead_response = look_ahead_response
        self.enable_motion = enable_motion
        self.enable_look_ahead = enable_look_ahead

        self.render_width = int(round(screen_width * (1.0 + overscan_ratio * 2.0)))
        self.render_height = int(round(screen_height * (1.0 + overscan_ratio * 2.0)))
        self.render_margin_x = (self.render_width - screen_width) // 2
        self.render_margin_y = (self.render_height - screen_height) // 2

        self.time = 0.0
        self.travel_x = 0.0
        self.look_ahead_px = 0.0
        self.motion_x = 0.0
        self.motion_y = 0.0
        self.zoom = base_zoom
        self._actor_ground_y_override = None

    @property
    def actor_render_x(self) -> float:
        """Return the actor anchor x on the overscanned render surface."""
        base_x = self.render_margin_x + (self.screen_width * self.actor_screen_x_ratio)
        return base_x - self.look_ahead_px

    @property
    def actor_ground_y(self) -> float:
        """Return the actor ground line on the overscanned render surface."""
        if self._actor_ground_y_override is not None:
            return self._actor_ground_y_override
        return self.render_margin_y + self.screen_height - self.floor_offset

    @property
    def background_camera_x(self) -> float:
        """Return the world camera x used for parallax layers."""
        return self.travel_x + self.look_ahead_px

    def update(self, dt: float, travel_speed: float) -> None:
        """Advance camera timing and subtle motion state."""
        self.time += dt
        self.travel_x += travel_speed * dt

        target_look_ahead = 0.0
        if self.enable_look_ahead:
            target_look_ahead = min(self.max_look_ahead_px, travel_speed * 0.16)

        blend = min(1.0, dt * self.look_ahead_response)
        self.look_ahead_px += (target_look_ahead - self.look_ahead_px) * blend

        if not self.enable_motion:
            self.motion_x = 0.0
            self.motion_y = 0.0
            self.zoom = self.base_zoom
            return

        # Two low-frequency waves keep the motion subtle and less mechanical.
        self.motion_x = (
            math.sin(self.time * 0.47) * self.sway_x
            + math.sin(self.time * 1.11 + 1.2) * (self.sway_x * 0.32)
        )
        self.motion_y = (
            math.sin(self.time * 0.63 + 0.8) * self.sway_y
            + math.sin(self.time * 1.39 + 2.1) * (self.sway_y * 0.22)
        )
        self.zoom = (
            self.base_zoom
            + math.sin(self.time * 0.36 + 0.3) * self.zoom_amplitude
            + math.sin(self.time * 0.91 + 2.0) * (self.zoom_amplitude * 0.45)
        )

    def set_actor_ground_y(self, ground_y: float) -> None:
        """Set an absolute actor ground line on the render surface."""
        self._actor_ground_y_override = ground_y

    def draw(self, scene_surface, target_surface) -> None:
        """Project the overscanned scene surface into the visible screen."""
        import pygame

        view_w = min(self.render_width, max(1, int(round(self.screen_width / self.zoom))))
        view_h = min(self.render_height, max(1, int(round(self.screen_height / self.zoom))))

        base_center_x = self.render_margin_x + (self.screen_width / 2)
        base_center_y = self.render_margin_y + (self.screen_height / 2)
        view_left = base_center_x - (view_w / 2) + self.motion_x
        view_top = base_center_y - (view_h / 2) + self.motion_y

        view_left = max(0, min(view_left, self.render_width - view_w))
        view_top = max(0, min(view_top, self.render_height - view_h))

        view_rect = pygame.Rect(
            int(round(view_left)),
            int(round(view_top)),
            view_w,
            view_h,
        )
        frame = scene_surface.subsurface(view_rect)

        if frame.get_width() != self.screen_width or frame.get_height() != self.screen_height:
            frame = pygame.transform.smoothscale(frame, (self.screen_width, self.screen_height))

        target_surface.blit(frame, (0, 0))
