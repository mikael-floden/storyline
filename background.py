"""
Parallax background system for 2D side-scrolling games.

Provides multi-layer background rendering with parallax scrolling effect.
"""

import math
import os
from typing import Dict, List, Tuple

import yaml

from background_interface import IBackgroundRenderer


class ParallaxLayer:
    """Represents a single layer in the parallax background."""

    def __init__(
        self,
        surface,
        scroll_speed: float,
        y_offset: float = 0,
        *,
        base_x: float = 0.0,
        base_y: float | None = None,
        repeat_x: bool = True,
        repeat_step: float | None = None,
        draw_phase: str = "background",
        mirror_x: bool = False,
        mirror_surface=None,
    ):
        """Initialize a parallax layer.

        Args:
            surface: Pygame surface containing the layer image
            scroll_speed: Parallax scroll multiplier (0-1, where 0=static, 1=full speed)
            y_offset: Vertical offset from bottom of screen
        """
        self.surface = surface
        self.scroll_speed = scroll_speed
        self.y_offset = y_offset
        self.base_x = base_x
        self.base_y = y_offset if base_y is None else base_y
        self.repeat_x = repeat_x
        self.repeat_step = surface.get_width() if repeat_step is None else repeat_step
        self.draw_phase = draw_phase
        self.mirror_x = mirror_x
        self.mirror_surface = mirror_surface
        self.x_offset = 0.0


class ParallaxBackground(IBackgroundRenderer):
    """Parallax background implementation with multiple scrolling layers."""

    def __init__(self, screen_width: int, screen_height: int):
        """Initialize the parallax background system.

        Args:
            screen_width: Width of the game screen
            screen_height: Height of the game screen
        """
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.layers: List[ParallaxLayer] = []
        self.camera_x = 0.0
        self.camera_y = 0.0
        self.metadata = None

    def load_background(self, background_path: str, layer_config: List[Tuple[str, float]]) -> None:
        """Load a multi-layer background from a directory.

        Args:
            background_path: Path to background directory (e.g., "background/01_AshenValley")
            layer_config: List of (filename, scroll_speed) tuples, ordered back to front
                         Example: [("L1_Sky.png", 0.0), ("L2_FarMountains.png", 0.2), ...]
        """
        import pygame

        self.layers.clear()

        for filename, scroll_speed in layer_config:
            layer_path = os.path.join(background_path, filename)

            if not os.path.isfile(layer_path):
                print(f"Warning: Background layer not found: {layer_path}")
                continue

            # Load the layer image
            surface = pygame.image.load(layer_path).convert_alpha()

            # Scale to screen height if needed
            layer_height = surface.get_height()
            if layer_height != self.screen_height:
                scale_factor = self.screen_height / layer_height
                new_width = int(surface.get_width() * scale_factor)
                surface = pygame.transform.smoothscale(surface, (new_width, self.screen_height))

            layer = ParallaxLayer(
                surface,
                scroll_speed,
                base_y=self.screen_height - surface.get_height(),
            )
            self.layers.append(layer)

    def load_background_from_metadata(self, metadata_path: str) -> None:
        """Load a multi-layer background using a metadata recipe."""
        import pygame

        with open(metadata_path, "r", encoding="utf-8") as fh:
            self.metadata = yaml.safe_load(fh) or {}

        scene = self.metadata.get("scene", {})
        canvas = scene.get("canvas", {})
        canvas_height = canvas.get("height") or self.screen_height
        base_scale = self.screen_height / canvas_height

        background_dir = os.path.dirname(metadata_path)
        layer_defs = {
            layer["id"]: layer for layer in self.metadata.get("layers", []) if "id" in layer
        }
        recipe = self.metadata.get("default_scene_recipe", {})
        instances_by_source: Dict[str, List[dict]] = {}

        for instance in recipe.get("instances", []):
            instances_by_source.setdefault(instance["source_layer"], []).append(instance)

        for instance in recipe.get("optional_instances", []):
            if instance.get("enabled"):
                instances_by_source.setdefault(instance["source_layer"], []).append(instance)

        self.layers.clear()

        source_cache: Dict[str, object] = {}

        for layer_id in recipe.get("stack", []):
            layer_meta = layer_defs.get(layer_id)
            if not layer_meta:
                print(f"Warning: Background layer id not found in metadata: {layer_id}")
                continue

            placement = layer_meta.get("placement", {})
            mode = placement.get("mode", "tile")
            draw_phase = layer_meta.get("draw_phase", "background")

            if mode in {"tile", "panorama_plate"}:
                image_path = os.path.join(background_dir, layer_meta["file"])
                if not os.path.isfile(image_path):
                    print(f"Warning: Background layer not found: {image_path}")
                    continue

                surface = pygame.image.load(image_path).convert_alpha()
                if placement.get("crop_to_content"):
                    surface = self._crop_surface_to_bbox(
                        surface,
                        layer_meta.get("content_bbox"),
                    )
                surface = self._scale_surface(surface, base_scale)
                surface = self._apply_opacity(surface, placement.get("opacity", 1.0))

                y_offset = placement.get("y_offset", 0)
                scaled_y_offset = self._scale_value(y_offset, base_scale)
                align = placement.get("align", "bottom")
                if align == "top":
                    base_y = scaled_y_offset
                else:
                    base_y = self.screen_height - surface.get_height() + scaled_y_offset

                repeat_x = placement.get("repeat_x", "seamless")
                mirror_x = repeat_x == "mirror"

                self.layers.append(
                    ParallaxLayer(
                        surface,
                        float(layer_meta.get("parallax", 0.0)),
                        base_y=base_y,
                        repeat_x=repeat_x != "none",
                        repeat_step=self._scale_value(
                            placement.get("repeat_step", surface.get_width() / base_scale),
                            base_scale,
                        )
                        if repeat_x != "none"
                        else None,
                        draw_phase=draw_phase,
                        mirror_x=mirror_x,
                        mirror_surface=pygame.transform.flip(surface, True, False)
                        if mirror_x
                        else None,
                    )
                )
                continue

            if mode != "extract_objects":
                print(f"Warning: Unsupported background placement mode: {mode}")
                continue

            image_path = os.path.join(background_dir, layer_meta["file"])
            if not os.path.isfile(image_path):
                print(f"Warning: Background layer not found: {image_path}")
                continue

            source_surface = source_cache.get(image_path)
            if source_surface is None:
                source_surface = pygame.image.load(image_path).convert_alpha()
                source_cache[image_path] = source_surface

            for instance in instances_by_source.get(layer_id, []):
                object_meta = self._find_object(layer_meta, instance["object"])
                if not object_meta:
                    print(
                        f"Warning: Background object not found: "
                        f"{layer_id}.{instance['object']}"
                    )
                    continue

                object_surface = self._extract_object_surface(
                    source_surface,
                    object_meta,
                    base_scale,
                    placement.get("opacity", 1.0),
                    instance.get("scale", 1.0),
                )
                pivot = object_meta.get("pivot", {"x": 0.5, "y": 1.0})
                anchor = instance.get("screen_anchor", {"x": 0.5, "y": 1.0})
                x_offset = self._scale_value(placement.get("x_offset", 0), base_scale)
                y_offset = self._scale_value(placement.get("y_offset", 0), base_scale)
                repeat_every = instance.get("repeat_every")
                repeat_x = repeat_every is not None or instance.get("repeat_x", False)
                instance_draw_phase = instance.get("draw_phase", draw_phase)

                base_x = anchor.get("x", 0.5) * self.screen_width
                base_x -= pivot.get("x", 0.5) * object_surface.get_width()
                base_x += x_offset

                base_y = anchor.get("y", 1.0) * self.screen_height
                base_y -= pivot.get("y", 1.0) * object_surface.get_height()
                base_y += y_offset

                self.layers.append(
                    ParallaxLayer(
                        object_surface,
                        float(instance.get("parallax", layer_meta.get("parallax", 0.0))),
                        base_x=base_x,
                        base_y=base_y,
                        repeat_x=repeat_x,
                        repeat_step=self._scale_value(repeat_every, base_scale)
                        if repeat_every is not None
                        else None,
                        draw_phase=instance_draw_phase,
                    )
                )

    def update(self, camera_x: float, dt: float) -> None:
        """Update background state based on camera position.

        Args:
            camera_x: Camera x position in world coordinates
            dt: Delta time in seconds
        """
        self.camera_x = camera_x

        # Update each layer's offset based on camera position and scroll speed
        for layer in self.layers:
            layer.x_offset = -camera_x * layer.scroll_speed

    def set_camera_position(self, x: float, y: float = 0) -> None:
        """Set the camera position for parallax scrolling.

        Args:
            x: Camera x position in world coordinates
            y: Camera y position in world coordinates (optional)
        """
        self.camera_x = x
        self.camera_y = y

        # Update layer offsets
        for layer in self.layers:
            layer.x_offset = -x * layer.scroll_speed

    def draw(self, surf) -> None:
        """Render the background layers to a surface.

        Args:
            surf: Pygame surface to draw on
        """
        self._draw_layers(surf, draw_phase="background")

    def draw_foreground(self, surf) -> None:
        """Render foreground layers that should appear in front of actors."""
        self._draw_layers(surf, draw_phase="foreground")

    def _draw_layers(self, surf, draw_phase: str) -> None:
        """Draw a subset of layers for a given phase."""
        for layer in self.layers:
            if layer.draw_phase != draw_phase:
                continue

            if not layer.repeat_x:
                surf.blit(
                    layer.surface,
                    (int(layer.base_x + layer.x_offset), int(layer.base_y)),
                )
                continue

            layer_width = layer.surface.get_width()
            repeat_step = max(1, int(round(layer.repeat_step)))
            world_x = layer.base_x + layer.x_offset
            tile_index = math.floor(-world_x / repeat_step)
            x = world_x + (tile_index * repeat_step)

            while x < self.screen_width:
                tile_surface = layer.surface
                if layer.mirror_x and layer.mirror_surface and tile_index % 2:
                    tile_surface = layer.mirror_surface

                surf.blit(tile_surface, (int(x), int(layer.base_y)))
                x += repeat_step
                tile_index += 1

    def _apply_opacity(self, surface, opacity: float):
        """Apply whole-surface opacity without mutating the cached source surface."""
        if opacity >= 1.0:
            return surface

        surface = surface.copy()
        surface.set_alpha(max(0, min(255, int(opacity * 255))))
        return surface

    def _extract_object_surface(
        self,
        source_surface,
        object_meta: dict,
        base_scale: float,
        opacity: float,
        instance_scale: float,
    ):
        """Crop and scale an object from a sprite sheet."""
        import pygame

        bbox = object_meta["bbox"]
        rect = pygame.Rect(bbox["x"], bbox["y"], bbox["w"], bbox["h"])
        surface = source_surface.subsurface(rect).copy()

        total_scale = base_scale * instance_scale
        surface = self._scale_surface(surface, total_scale)
        return self._apply_opacity(surface, opacity)

    def _find_object(self, layer_meta: dict, object_id: str) -> dict | None:
        """Return an object definition by id."""
        for obj in layer_meta.get("objects", []):
            if obj.get("id") == object_id:
                return obj
        return None

    def _scale_surface(self, surface, scale_factor: float):
        """Scale a surface while preserving aspect ratio."""
        import pygame

        if scale_factor == 1.0:
            return surface

        new_width = max(1, int(round(surface.get_width() * scale_factor)))
        new_height = max(1, int(round(surface.get_height() * scale_factor)))
        return pygame.transform.smoothscale(surface, (new_width, new_height))

    def _crop_surface_to_bbox(self, surface, bbox: dict | None):
        """Crop a surface to a metadata bounding box."""
        import pygame

        if not bbox:
            return surface

        rect = pygame.Rect(bbox["x"], bbox["y"], bbox["w"], bbox["h"])
        return surface.subsurface(rect).copy()

    def _scale_value(self, value: float, scale_factor: float) -> int:
        """Scale a metadata value from source-canvas pixels to screen pixels."""
        return int(round(value * scale_factor))


def create_ashen_valley_background(screen_width: int, screen_height: int) -> ParallaxBackground:
    """Create a pre-configured Ashen Valley parallax background.

    Args:
        screen_width: Width of the game screen
        screen_height: Height of the game screen

    Returns:
        Configured ParallaxBackground instance
    """
    bg = ParallaxBackground(screen_width, screen_height)
    background_dir = os.path.join(os.path.dirname(__file__), "backgrounds", "ashen_valley")

    for metadata_name in ("metadata.yaml", "metedata.yaml"):
        metadata_path = os.path.join(background_dir, metadata_name)
        if os.path.isfile(metadata_path):
            bg.load_background_from_metadata(metadata_path)
            return bg

    raise FileNotFoundError(
        f"No background metadata file found in {background_dir!r}"
    )
