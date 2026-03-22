#!/usr/bin/env python3
"""
JSON-driven cinematic scene player for stories/golem_story_scene.json.

Controls: SPACE pause/resume | R restart | ESC quit
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pygame

from actor import Actor
from background import ParallaxBackground, create_ashen_valley_background
from foreground_fog import ForegroundFog

BASE_DIR = os.path.dirname(__file__)
STORY_PATH = os.path.join(BASE_DIR, "stories", "golem_story_scene.json")

W, H = 960, 580
FPS = 60

BG_CLEAR = (6, 9, 15, 255)
CAPTION_BG = (12, 16, 28)
CAPTION_BORDER = (82, 112, 150)
TEXT_COLOR = (238, 243, 250)
MUTED_TEXT = (176, 186, 202)
TITLE_COLOR = (210, 227, 255)
CHARACTER_COLORS = {
    "stone": (188, 193, 202),
    "fire": (255, 161, 88),
    "crystal": (126, 214, 255),
    "earth": (196, 178, 134),
    "narrator": (196, 204, 218),
}
WALK_BOB = {
    "stone": 2.4,
    "fire": 3.1,
    "crystal": 1.8,
    "earth": 2.2,
}


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def smoothstep01(value: float) -> float:
    value = clamp(value)
    return value * value * (3.0 - (2.0 * value))


def normalized_time(now: float, start: float, end: float) -> float:
    if end <= start:
        return 1.0
    return clamp((now - start) / (end - start))


def event_active(now: float, event: dict[str, Any]) -> bool:
    return float(event.get("start", 0.0)) <= now < float(event.get("end", 0.0))


def event_envelope(now: float, start: float, end: float, ramp: float = 0.24) -> float:
    if now < start or now >= end:
        return 0.0
    fade_in = clamp((now - start) / max(0.001, ramp))
    fade_out = clamp((end - now) / max(0.001, ramp))
    return min(fade_in, fade_out)


def average(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def wrap_text(font: pygame.font.Font, text: str, max_width: int) -> list[str]:
    words = text.split()
    if not words:
        return [""]

    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if font.size(candidate)[0] <= max_width:
            current = candidate
            continue
        lines.append(current)
        current = word
    lines.append(current)
    return lines


@dataclass(frozen=True)
class ActorRuntime:
    actor_root: str
    part_scale: int
    discover_models: Callable[[str], list[dict[str, Any]]]
    load_model: Callable[[dict[str, Any]], dict[str, Any]]


@dataclass
class MotionSegment:
    actor_id: str
    start: float
    end: float
    start_x: float
    end_x: float
    event_type: str


@dataclass
class LoadedStoryActor:
    actor_id: str
    name: str
    z_order: int
    ground_offset: float
    model: dict[str, Any]
    actor: Actor
    sprite_set: dict[str, Any]
    world_x: float = 0.0


class PresentationCamera:
    """Overscanned scene camera with follow, drift, shake, and zoom."""

    def __init__(self, screen_width: int, screen_height: int, overscan_ratio: float = 0.08):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.overscan_ratio = overscan_ratio

        self.render_width = int(round(screen_width * (1.0 + overscan_ratio * 2.0)))
        self.render_height = int(round(screen_height * (1.0 + overscan_ratio * 2.0)))
        self.render_margin_x = (self.render_width - screen_width) // 2
        self.render_margin_y = (self.render_height - screen_height) // 2

        self.time = 0.0
        self.world_x = 0.0
        self.anchor_ratio = 0.46
        self.zoom = 1.0
        self.motion_x = 0.0
        self.motion_y = 0.0
        self.actor_ground_y = self.render_margin_y + self.screen_height - 90

    def set_actor_ground_y(self, ground_y: float) -> None:
        self.actor_ground_y = ground_y

    def world_to_render_x(self, world_x: float) -> float:
        anchor_x = self.render_margin_x + (self.screen_width * self.anchor_ratio)
        return anchor_x + (world_x - self.world_x)

    def update(
        self,
        dt: float,
        *,
        target_world_x: float,
        target_zoom: float,
        target_anchor_ratio: float,
        sway_x: float,
        sway_y: float,
        follow_speed: float,
        shake_strength: float = 0.0,
    ) -> None:
        self.time += dt

        follow_blend = min(1.0, dt * max(0.1, follow_speed))
        zoom_blend = min(1.0, dt * 1.8)
        anchor_blend = min(1.0, dt * 2.2)

        self.world_x += (target_world_x - self.world_x) * follow_blend
        self.zoom += (target_zoom - self.zoom) * zoom_blend
        self.anchor_ratio += (target_anchor_ratio - self.anchor_ratio) * anchor_blend

        base_x = (
            math.sin(self.time * 0.43) * sway_x
            + math.sin(self.time * 0.96 + 1.3) * (sway_x * 0.35)
        )
        base_y = (
            math.sin(self.time * 0.57 + 0.7) * sway_y
            + math.sin(self.time * 1.18 + 2.1) * (sway_y * 0.28)
        )

        shake_x = 0.0
        shake_y = 0.0
        if shake_strength > 0.0:
            shake_x = math.sin(self.time * 27.0) * shake_strength
            shake_y = math.sin((self.time * 31.0) + 0.9) * (shake_strength * 0.58)

        self.motion_x = base_x + shake_x
        self.motion_y = base_y + shake_y

    def present(self, scene_surface: pygame.Surface, target_surface: pygame.Surface) -> None:
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
        if frame.get_size() != target_surface.get_size():
            frame = pygame.transform.smoothscale(frame, target_surface.get_size())

        target_surface.blit(frame, (0, 0))


class AshField:
    """Simple screen-space ash particle field."""

    def __init__(
        self,
        width: int,
        height: int,
        *,
        count: int,
        seed: int,
        speed_x: tuple[float, float],
        speed_y: tuple[float, float],
        size_range: tuple[int, int],
    ):
        self.width = width
        self.height = height
        self._rng = random.Random(seed)
        self.particles: list[dict[str, float]] = []

        for _ in range(count):
            self.particles.append(
                {
                    "x": self._rng.uniform(0, width),
                    "y": self._rng.uniform(0, height),
                    "vx": self._rng.uniform(*speed_x),
                    "vy": self._rng.uniform(*speed_y),
                    "size": self._rng.randint(*size_range),
                    "alpha": self._rng.uniform(0.45, 1.0),
                }
            )

    def update(self, dt: float) -> None:
        for particle in self.particles:
            particle["x"] += particle["vx"] * dt
            particle["y"] += particle["vy"] * dt

            if particle["x"] > self.width + 40:
                particle["x"] = -40
                particle["y"] = self._rng.uniform(0, self.height)
            if particle["y"] > self.height + 20:
                particle["y"] = -20
                particle["x"] = self._rng.uniform(0, self.width)

    def draw(
        self,
        surf: pygame.Surface,
        intensity: float,
        *,
        color: tuple[int, int, int],
        alpha_scale: int,
    ) -> None:
        intensity = clamp(intensity)
        if intensity <= 0.0:
            return

        overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        visible_count = max(1, int(len(self.particles) * max(0.2, intensity)))
        for particle in self.particles[:visible_count]:
            alpha = int(alpha_scale * intensity * particle["alpha"])
            size = int(particle["size"])
            pygame.draw.circle(
                overlay,
                (color[0], color[1], color[2], alpha),
                (int(particle["x"]), int(particle["y"])),
                size,
            )
        surf.blit(overlay, (0, 0))


class StoryPlayer:
    def __init__(self, screen: pygame.Surface, actor_runtime: ActorRuntime):
        self.screen = screen
        self.actor_runtime = actor_runtime

        with open(STORY_PATH, "r", encoding="utf-8") as fh:
            self.story = json.load(fh)

        self.characters = {
            character["id"]: character for character in self.story.get("characters", [])
        }
        self.scenes = sorted(self.story.get("scenes", []), key=lambda scene: float(scene["start"]))
        self.duration = max(float(scene["end"]) for scene in self.scenes)

        self.camera = PresentationCamera(W, H)
        self.scene_surface = pygame.Surface(
            (self.camera.render_width, self.camera.render_height)
        ).convert_alpha()

        self.background, self.background_asset_dir = self._build_background()
        self._apply_background_ground_line()
        self.foreground_fog = ForegroundFog(
            self.camera.render_width,
            self.camera.render_height,
            self.background_asset_dir,
        )

        (
            self.motion_tracks,
            self.initial_positions,
            self.ground_offsets,
            self.z_orders,
            self.scene_refs,
            self.events_by_type,
            self.all_events,
        ) = self._build_tracks()

        self.actors = self._load_actors()

        self.back_ash = AshField(
            self.camera.render_width,
            self.camera.render_height,
            count=64,
            seed=7,
            speed_x=(10.0, 22.0),
            speed_y=(2.0, 8.0),
            size_range=(1, 2),
        )
        self.front_ash = AshField(
            self.camera.render_width,
            self.camera.render_height,
            count=92,
            seed=13,
            speed_x=(18.0, 34.0),
            speed_y=(5.0, 14.0),
            size_range=(1, 3),
        )

        self.title_font = self._sys_font("georgia", 32, bold=True)
        self.scene_font = self._sys_font("georgia", 22, bold=True)
        self.speaker_font = self._sys_font("verdana", 20, bold=True)
        self.caption_font = self._sys_font("verdana", 24)
        self.small_font = self._sys_font("verdana", 16)

        self.current_time = 0.0
        self.paused = False
        self.current_scene_id = ""
        self.scene_entered_at = 0.0
        self.active_scene = self.scenes[0]
        self.active_text: dict[str, Any] | None = None
        self.frame_positions = dict(self.initial_positions)
        self.frame_effects: dict[str, float] = {}

        self.restart()

    def _sys_font(self, name: str, size: int, bold: bool = False) -> pygame.font.Font:
        try:
            return pygame.font.SysFont(name, size, bold=bold)
        except Exception:
            fallback_size = max(12, int(round(size * 0.9)))
            return pygame.font.Font(None, fallback_size)

    def _build_background(self) -> tuple[ParallaxBackground, str]:
        story_dir = os.path.dirname(STORY_PATH)
        layer_config: list[tuple[str, float]] = []
        for layer in self.story.get("layers", []):
            file_name = layer.get("file")
            if not file_name:
                continue
            speed = float(layer.get("parallax_speed", layer.get("parallax", 0.0)))
            layer_config.append((file_name, speed))

        if layer_config:
            all_present = all(
                os.path.isfile(os.path.join(story_dir, file_name))
                for file_name, _ in layer_config
            )
            if all_present:
                bg = ParallaxBackground(self.camera.render_width, self.camera.render_height)
                bg.load_background(story_dir, layer_config)
                return bg, story_dir

        asset_dir = os.path.join(BASE_DIR, "backgrounds", "ashen_valley")
        return (
            create_ashen_valley_background(self.camera.render_width, self.camera.render_height),
            asset_dir,
        )

    def _apply_background_ground_line(self) -> None:
        metadata = getattr(self.background, "metadata", None)
        if not metadata:
            return

        scene = metadata.get("scene", {})
        canvas = scene.get("canvas", {})
        canvas_height = canvas.get("height")
        player_ground_y = scene.get("player_ground_y")
        if canvas_height and player_ground_y is not None:
            scaled_ground_y = self.camera.render_height * (
                float(player_ground_y) / float(canvas_height)
            )
            self.camera.set_actor_ground_y(scaled_ground_y)

    def _build_tracks(
        self,
    ) -> tuple[
        dict[str, list[MotionSegment]],
        dict[str, float],
        dict[str, float],
        dict[str, int],
        dict[str, dict[str, float]],
        dict[str, list[dict[str, Any]]],
        list[dict[str, Any]],
    ]:
        motion_tracks = {character_id: [] for character_id in self.characters}
        initial_positions = {
            character_id: -180.0 - (index * 80.0)
            for index, character_id in enumerate(self.characters)
        }
        ground_offsets = {character_id: 0.0 for character_id in self.characters}
        z_orders = {
            character_id: 10 + index for index, character_id in enumerate(self.characters)
        }
        scene_refs: dict[str, dict[str, float]] = {}
        events_by_type: dict[str, list[dict[str, Any]]] = {}
        all_events: list[dict[str, Any]] = []

        positions = dict(initial_positions)

        for scene in self.scenes:
            for setup in scene.get("character_setup", []):
                character_id = setup.get("character")
                if character_id not in self.characters:
                    continue

                start_position = setup.get("start_position", {})
                positions[character_id] = float(
                    start_position.get("x", positions.get(character_id, 0.0))
                )
                initial_positions[character_id] = positions[character_id]
                ground_offsets[character_id] = float(start_position.get("y", 0.0))
                z_orders[character_id] = int(setup.get("z_order", z_orders[character_id]))

            lead_x = positions.get("stone", average(list(positions.values())))
            center_x = average(list(positions.values()))
            scene_refs[scene["id"]] = {
                "lead_x": lead_x,
                "center_x": center_x,
            }

            scene_events = sorted(
                scene.get("events", []),
                key=lambda event: (
                    float(event.get("start", 0.0)),
                    float(event.get("end", 0.0)),
                    event.get("type", ""),
                ),
            )

            for event in scene_events:
                event_copy = dict(event)
                event_copy["scene_id"] = scene["id"]
                event_copy["scene_title"] = scene["title"]
                all_events.append(event_copy)
                events_by_type.setdefault(event_copy["type"], []).append(event_copy)

                event_type = event_copy["type"]
                if event_type == "move":
                    distance = float(event_copy.get("distance_px", 0.0))
                    direction = -1.0 if event_copy.get("direction") == "left" else 1.0
                    speed_profile = event_copy.get("speed_profile", {})
                    for character_id in event_copy.get("characters", []):
                        if character_id not in positions:
                            continue
                        multiplier = float(speed_profile.get(character_id, 1.0))
                        start_x = positions[character_id]
                        end_x = start_x + (distance * multiplier * direction)
                        motion_tracks[character_id].append(
                            MotionSegment(
                                actor_id=character_id,
                                start=float(event_copy["start"]),
                                end=float(event_copy["end"]),
                                start_x=start_x,
                                end_x=end_x,
                                event_type=event_type,
                            )
                        )
                        positions[character_id] = end_x
                elif event_type == "micro_step":
                    character_id = event_copy.get("character")
                    if character_id not in positions:
                        continue
                    direction = -1.0 if event_copy.get("direction") == "left" else 1.0
                    distance = float(event_copy.get("distance_px", 0.0))
                    start_x = positions[character_id]
                    end_x = start_x + (distance * direction)
                    motion_tracks[character_id].append(
                        MotionSegment(
                            actor_id=character_id,
                            start=float(event_copy["start"]),
                            end=float(event_copy["end"]),
                            start_x=start_x,
                            end_x=end_x,
                            event_type=event_type,
                        )
                    )
                    positions[character_id] = end_x

        return (
            motion_tracks,
            initial_positions,
            ground_offsets,
            z_orders,
            scene_refs,
            events_by_type,
            all_events,
        )

    def _load_actors(self) -> dict[str, LoadedStoryActor]:
        model_specs = self.actor_runtime.discover_models(self.actor_runtime.actor_root)
        model_lookup = {model_spec["name"]: model_spec for model_spec in model_specs}

        actors: dict[str, LoadedStoryActor] = {}
        for character_id, character in self.characters.items():
            model_spec = model_lookup.get(character_id)
            if model_spec is None:
                raise RuntimeError(f"Actor model not found for {character_id!r}")

            model = self.actor_runtime.load_model(model_spec)
            actor = Actor(
                0.0,
                self.camera.actor_ground_y,
                mode=model["name"],
                pivots=model["pivots"],
                metrics=model["metrics"],
                draw_order=model["draw_order"],
                animations=model["animations"],
                skeleton=model["skeleton"],
                part_scale=self.actor_runtime.part_scale,
            )
            if "idle" in actor.animations:
                actor.current_anim = "idle"
                actor.anim_time = 0.0
                actor.prev_anim = None
                actor.blend_time = actor.blend_duration

            actors[character_id] = LoadedStoryActor(
                actor_id=character_id,
                name=character.get("name", character_id.title()),
                z_order=self.z_orders[character_id],
                ground_offset=self.ground_offsets[character_id],
                model=model,
                actor=actor,
                sprite_set=model["sprites"],
                world_x=self.initial_positions[character_id],
            )

        return actors

    def restart(self) -> None:
        self.current_time = 0.0
        self.current_scene_id = ""
        self.scene_entered_at = 0.0
        first_scene = self.scenes[0]
        scene_ref = self.scene_refs.get(first_scene["id"], {"lead_x": 0.0, "center_x": 0.0})
        self.camera.world_x = scene_ref["lead_x"]
        self.camera.anchor_ratio = 0.42
        self.camera.zoom = 0.96
        self.camera.motion_x = 0.0
        self.camera.motion_y = 0.0
        self.camera.time = 0.0

        for loaded_actor in self.actors.values():
            loaded_actor.actor.stop_speaking()
            if "idle" in loaded_actor.actor.animations:
                loaded_actor.actor.current_anim = "idle"
            loaded_actor.actor.anim_time = 0.0
            loaded_actor.actor.prev_anim = None
            loaded_actor.actor.blend_time = loaded_actor.actor.blend_duration
            loaded_actor.world_x = self.initial_positions.get(loaded_actor.actor_id, 0.0)

        self.update(0.0)

    def scene_for_time(self, now: float) -> dict[str, Any]:
        for scene in self.scenes:
            if float(scene["start"]) <= now < float(scene["end"]):
                return scene
        return self.scenes[-1]

    def position_at(self, actor_id: str, now: float) -> float:
        position = self.initial_positions.get(actor_id, 0.0)
        for segment in self.motion_tracks.get(actor_id, []):
            if now < segment.start:
                break
            if now >= segment.end:
                position = segment.end_x
                continue
            progress = smoothstep01(normalized_time(now, segment.start, segment.end))
            return lerp(segment.start_x, segment.end_x, progress)
        return position

    def active_motion(self, actor_id: str, now: float) -> MotionSegment | None:
        for segment in self.motion_tracks.get(actor_id, []):
            if segment.start <= now < segment.end:
                return segment
        return None

    def active_text_event(self, now: float) -> dict[str, Any] | None:
        active = [
            event
            for event in self.all_events
            if event["type"] in {"caption", "dialogue"} and event_active(now, event)
        ]
        if not active:
            return None
        return max(active, key=lambda event: float(event["start"]))

    def active_state_event(self, actor_id: str, now: float) -> dict[str, Any] | None:
        active = []
        for event in self.events_by_type.get("state_change", []):
            if actor_id in event.get("characters", []) and event_active(now, event):
                active.append(event)
        if not active:
            return None
        return max(active, key=lambda event: float(event["start"]))

    def pose_offsets(
        self,
        actor_id: str,
        positions: dict[str, float],
        now: float,
    ) -> tuple[float, float]:
        x_offset = 0.0
        y_offset = 0.0

        for event in self.events_by_type.get("look_at", []):
            if event.get("source") != actor_id or not event_active(now, event):
                continue
            target_id = event.get("target")
            source_x = positions.get(actor_id, 0.0)
            target_x = positions.get(target_id, source_x)
            strength = event_envelope(now, float(event["start"]), float(event["end"]))
            direction = 1.0 if target_x >= source_x else -1.0
            x_offset += direction * 5.0 * strength

        for event in self.events_by_type.get("look_down", []):
            if event.get("character") != actor_id or not event_active(now, event):
                continue
            strength = event_envelope(now, float(event["start"]), float(event["end"]))
            y_offset += 4.0 * strength

        for event in self.events_by_type.get("gesture", []):
            if event.get("character") != actor_id or not event_active(now, event):
                continue
            strength = event_envelope(now, float(event["start"]), float(event["end"]))
            y_offset += 5.0 * strength
            x_offset -= 2.0 * strength

        for event in self.events_by_type.get("reaction", []):
            if event.get("character") != actor_id or not event_active(now, event):
                continue
            strength = event_envelope(now, float(event["start"]), float(event["end"]))
            x_offset += math.sin(now * 18.0) * 2.5 * strength

        return x_offset, y_offset

    def choose_animation(
        self,
        actor_id: str,
        motion: MotionSegment | None,
        active_text: dict[str, Any] | None,
        now: float,
    ) -> str:
        actor = self.actors[actor_id].actor

        if (
            active_text
            and active_text["type"] == "dialogue"
            and active_text.get("speaker") == actor_id
            and "talk" in actor.animations
        ):
            return "talk"

        if motion and "walk" in actor.animations:
            return "walk"

        state_event = self.active_state_event(actor_id, now)
        if state_event:
            state_name = state_event.get("state", "")
            if state_name == "slow_to_stop" and "walk" in actor.animations:
                progress = normalized_time(now, float(state_event["start"]), float(state_event["end"]))
                return "walk" if progress < 0.45 else "idle"
            if state_name in actor.animations:
                return state_name

        if "idle" in actor.animations:
            return "idle"
        return actor.current_anim or ""

    def walk_bob(self, actor_id: str, motion: MotionSegment | None, actor: Actor) -> float:
        if motion is None or actor.current_anim != "walk":
            return 0.0
        amplitude = WALK_BOB.get(actor_id, 2.0)
        return abs(math.sin(actor.anim_time * 5.8)) * amplitude

    def shake_strength(self, now: float) -> float:
        strength = 0.0
        for event in self.events_by_type.get("sfx", []):
            if not event_active(now, event):
                continue
            if event.get("name") != "low_rumble":
                continue
            progress = normalized_time(now, float(event["start"]), float(event["end"]))
            volume = float(event.get("volume", 0.4))
            strength = max(strength, 9.0 * volume * (1.0 - (progress * 0.35)))
        return strength

    def collect_effects(self, scene: dict[str, Any], now: float) -> dict[str, float]:
        effects = {
            "mist": 0.0,
            "ash": 0.0,
            "foreground_ash": 0.0,
            "ruin_glow": 0.0,
            "fade": 0.0,
            "shake": self.shake_strength(now),
        }

        for fx in scene.get("background_fx", []):
            start = float(fx.get("start", scene["start"]))
            end = float(fx.get("end", scene["end"]))
            if start <= now < end:
                effects[fx["type"]] = max(
                    effects.get(fx["type"], 0.0),
                    float(fx.get("intensity", 0.0)),
                )

        for event in scene.get("events", []):
            if not event_active(now, event):
                continue

            progress = normalized_time(now, float(event["start"]), float(event["end"]))
            if event["type"] == "reveal":
                effects["ruin_glow"] = max(effects["ruin_glow"], 0.22 + (0.28 * progress))
            elif event["type"] == "light_pulse":
                envelope = math.sin(progress * math.pi)
                effects["ruin_glow"] = max(
                    effects["ruin_glow"],
                    float(event.get("intensity", 0.6)) * envelope,
                )
            elif event["type"] == "transition" and event.get("name") == "fade_to_black":
                effects["fade"] = max(effects["fade"], smoothstep01(progress))

        return effects

    def camera_profile(
        self,
        scene: dict[str, Any],
        positions: dict[str, float],
        active_text: dict[str, Any] | None,
    ) -> dict[str, float]:
        shot = scene.get("camera", {}).get("shot", "medium_side_view")
        movement = scene.get("camera", {}).get("movement", "locked")
        scene_start = float(scene["start"])
        scene_end = float(scene["end"])
        progress = normalized_time(self.current_time, scene_start, scene_end)
        scene_ref = self.scene_refs.get(scene["id"], {"lead_x": 0.0, "center_x": 0.0})

        lead_x = positions.get("stone", average(list(positions.values())))
        center_x = average(list(positions.values()))

        target_zoom = 1.0
        target_anchor = 0.46
        sway_x = 4.0
        sway_y = 2.0
        follow_speed = 2.0

        if shot in {"wide_side_scroll", "side_scroll_follow", "wide_side_scroll_to_fade"}:
            target_zoom = 0.94
            target_anchor = 0.36
            sway_x = 6.0
            sway_y = 4.0
            follow_speed = 2.5
        elif shot == "medium_side_view":
            target_zoom = 1.03
            target_anchor = 0.48
            sway_x = 2.2
            sway_y = 1.4
            follow_speed = 2.2
        elif shot == "wide_reveal_then_medium":
            reveal_progress = smoothstep01(min(1.0, progress / 0.28))
            target_zoom = lerp(0.91, 0.99, reveal_progress)
            target_anchor = lerp(0.34, 0.46, reveal_progress)
            sway_x = 3.2
            sway_y = 1.8
            follow_speed = 1.8
        elif shot == "medium_push_in":
            target_zoom = lerp(1.03, 1.12, smoothstep01(progress))
            target_anchor = 0.49
            sway_x = 1.5
            sway_y = 1.0
            follow_speed = 1.6

        if movement == "slow_track_right":
            target_world_x = lead_x + 32.0
        elif movement == "locked":
            target_world_x = lerp(scene_ref["center_x"], scene_ref["lead_x"], 0.25)
        elif movement == "subtle_drift_right":
            target_world_x = scene_ref["center_x"] + (50.0 * smoothstep01(progress))
        elif movement == "follow_group":
            target_world_x = lead_x + 18.0
        elif movement == "reveal_arch_then_lock":
            reveal_amount = 210.0 * smoothstep01(min(1.0, progress / 0.22))
            target_world_x = scene_ref["lead_x"] + 60.0 + reveal_amount
        elif movement == "slow_push_in":
            target_world_x = scene_ref["center_x"] + 28.0
        elif movement == "follow_into_mist":
            target_world_x = lead_x + 56.0
        else:
            target_world_x = center_x

        if active_text and active_text.get("speaker") in positions:
            speaker_x = positions[active_text["speaker"]]
            target_world_x = lerp(target_world_x, speaker_x, 0.18)

        return {
            "target_world_x": target_world_x,
            "target_zoom": target_zoom,
            "target_anchor": target_anchor,
            "sway_x": sway_x,
            "sway_y": sway_y,
            "follow_speed": follow_speed,
        }

    def update(self, dt: float) -> None:
        sim_dt = 0.0 if self.paused else dt
        if not self.paused:
            self.current_time += dt
            if self.current_time > self.duration + 1.5:
                self.restart()
                return

        scene = self.scene_for_time(self.current_time)
        if scene["id"] != self.current_scene_id:
            self.current_scene_id = scene["id"]
            self.scene_entered_at = self.current_time

        positions = {
            actor_id: self.position_at(actor_id, self.current_time)
            for actor_id in self.actors
        }
        active_text = self.active_text_event(self.current_time)
        effects = self.collect_effects(scene, self.current_time)
        camera_profile = self.camera_profile(scene, positions, active_text)

        self.camera.update(
            sim_dt,
            target_world_x=camera_profile["target_world_x"],
            target_zoom=camera_profile["target_zoom"],
            target_anchor_ratio=camera_profile["target_anchor"],
            sway_x=camera_profile["sway_x"],
            sway_y=camera_profile["sway_y"],
            follow_speed=camera_profile["follow_speed"],
            shake_strength=effects["shake"],
        )

        self.background.update(self.camera.world_x, sim_dt)
        self.foreground_fog.update(sim_dt)
        self.back_ash.update(sim_dt)
        self.front_ash.update(sim_dt)

        for actor_id, loaded_actor in self.actors.items():
            motion = self.active_motion(actor_id, self.current_time)
            x_offset, y_offset = self.pose_offsets(actor_id, positions, self.current_time)
            target_animation = self.choose_animation(
                actor_id,
                motion,
                active_text,
                self.current_time,
            )
            if (
                target_animation
                and target_animation in loaded_actor.actor.animations
                and loaded_actor.actor.current_anim != target_animation
            ):
                loaded_actor.actor.set_animation(target_animation)

            loaded_actor.actor.update(sim_dt)

            loaded_actor.world_x = positions[actor_id] + x_offset
            loaded_actor.actor.x = self.camera.world_to_render_x(loaded_actor.world_x)
            loaded_actor.actor.ground_y = (
                self.camera.actor_ground_y
                + loaded_actor.ground_offset
                + y_offset
                + self.walk_bob(actor_id, motion, loaded_actor.actor)
            )

        self.active_scene = scene
        self.active_text = active_text
        self.frame_positions = positions
        self.frame_effects = effects

    def draw_mist_overlay(self, surf: pygame.Surface, intensity: float) -> None:
        intensity = clamp(intensity)
        if intensity <= 0.0:
            return

        overlay = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
        width, height = surf.get_size()

        pygame.draw.rect(
            overlay,
            (160, 188, 214, int(16 * intensity)),
            pygame.Rect(0, int(height * 0.42), width, int(height * 0.58)),
        )

        band_width = int(width * 0.62)
        band_height = int(height * 0.24)
        for index in range(4):
            phase = (self.current_time * (0.26 + (index * 0.04))) + (index * 1.9)
            center_x = int((width * 0.16) + (index * width * 0.23) + (math.sin(phase) * 34))
            center_y = int((height * 0.36) + (index * 32) + (math.cos(phase * 0.7) * 14))
            rect = pygame.Rect(
                center_x - (band_width // 2),
                center_y,
                band_width,
                band_height,
            )
            alpha = int((42 + (index * 12)) * intensity)
            pygame.draw.ellipse(overlay, (176, 198, 222, alpha), rect)

        surf.blit(overlay, (0, 0))

    def draw_ruin_glow(self, surf: pygame.Surface, intensity: float) -> None:
        intensity = clamp(intensity)
        if intensity <= 0.0:
            return

        overlay = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
        center_x = int(self.camera.world_to_render_x(self.camera.world_x + 240.0))
        center_y = int(surf.get_height() * 0.34)
        radius = int(90 + (70 * intensity))

        for factor, alpha in ((1.0, 70), (0.66, 110), (0.38, 150)):
            pygame.draw.circle(
                overlay,
                (90, 170, 255, int(alpha * intensity)),
                (center_x, center_y),
                max(8, int(radius * factor)),
            )

        surf.blit(overlay, (0, 0))

    def draw_ground_pulse(self, surf: pygame.Surface, actor_id: str, intensity: float) -> None:
        intensity = clamp(intensity)
        if intensity <= 0.0 or actor_id not in self.actors:
            return

        actor = self.actors[actor_id].actor
        overlay = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
        center_x = int(actor.x)
        center_y = int(actor.ground_y)

        for scale, alpha in ((1.0, 92), (0.7, 140)):
            width = int((110 + (36 * intensity)) * scale)
            height = int((22 + (8 * intensity)) * scale)
            rect = pygame.Rect(center_x - (width // 2), center_y - (height // 2), width, height)
            pygame.draw.ellipse(
                overlay,
                (94, 180, 255, int(alpha * intensity)),
                rect,
                width=2,
            )

        surf.blit(overlay, (0, 0))

    def draw_foreground_fog(self, surf: pygame.Surface, intensity: float) -> None:
        intensity = clamp(intensity)
        if intensity <= 0.0:
            return
        fog_surface = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
        self.foreground_fog.draw(fog_surface)
        fog_surface.set_alpha(int(255 * intensity))
        surf.blit(fog_surface, (0, 0))

    def draw_caption(self, surf: pygame.Surface, text_event: dict[str, Any]) -> None:
        text = text_event.get("text", "").strip()
        if not text:
            return

        alpha = int(
            255
            * event_envelope(
                self.current_time,
                float(text_event["start"]),
                float(text_event["end"]),
                ramp=0.22,
            )
        )
        if alpha <= 0:
            return

        speaker_id = text_event.get("speaker", "narrator")
        speaker_label = "Narrator"
        if speaker_id in self.characters:
            speaker_label = self.characters[speaker_id].get("name", speaker_id.title())

        accent = CHARACTER_COLORS.get(speaker_id, CHARACTER_COLORS["narrator"])
        lines = wrap_text(self.caption_font, text, W - 160)
        line_height = self.caption_font.get_linesize()

        box_width = W - 100
        speaker_h = self.speaker_font.get_linesize()
        padding_x = 22
        padding_y = 16
        box_height = padding_y * 2 + speaker_h + 6 + (line_height * len(lines))
        box_rect = pygame.Rect(
            (W - box_width) // 2,
            H - box_height - 22,
            box_width,
            box_height,
        )

        box = pygame.Surface(box_rect.size, pygame.SRCALPHA)
        box.fill((CAPTION_BG[0], CAPTION_BG[1], CAPTION_BG[2], int(220 * alpha / 255)))
        pygame.draw.rect(
            box,
            (*CAPTION_BORDER, int(205 * alpha / 255)),
            box.get_rect(),
            width=2,
            border_radius=14,
        )
        pygame.draw.rect(
            box,
            (*accent, int(255 * alpha / 255)),
            pygame.Rect(14, 14, 8, box_rect.height - 28),
            border_radius=4,
        )

        speaker_surface = self.speaker_font.render(speaker_label.upper(), True, accent)
        box.blit(speaker_surface, (padding_x + 12, padding_y))

        text_y = padding_y + speaker_h + 8
        for line in lines:
            line_surface = self.caption_font.render(line, True, TEXT_COLOR)
            box.blit(line_surface, (padding_x + 12, text_y))
            text_y += line_height

        surf.blit(box, box_rect.topleft)

    def draw_scene_title(self, surf: pygame.Surface) -> None:
        age = self.current_time - self.scene_entered_at
        if age < 0.0 or age > 2.4:
            return

        fade = event_envelope(age, 0.0, 2.4, ramp=0.45)
        if fade <= 0.0:
            return

        title_surface = self.scene_font.render(self.active_scene["title"], True, TITLE_COLOR)
        subtitle_surface = self.small_font.render(
            self.active_scene["id"].replace("_", " "),
            True,
            MUTED_TEXT,
        )

        panel = pygame.Surface(
            (max(title_surface.get_width(), subtitle_surface.get_width()) + 22, 56),
            pygame.SRCALPHA,
        )
        panel.fill((8, 12, 20, int(138 * fade)))
        panel.blit(subtitle_surface, (11, 7))
        panel.blit(title_surface, (11, 24))
        surf.blit(panel, (28, 24))

    def draw_intro_title(self, surf: pygame.Surface) -> None:
        if self.current_time > 3.5:
            return

        fade = event_envelope(self.current_time, 0.0, 3.5, ramp=0.7)
        if fade <= 0.0:
            return

        project = self.story.get("project", {})
        title = project.get("title", "Untitled Story")
        subtitle = project.get("format", "")

        title_surface = self.title_font.render(title, True, TITLE_COLOR)
        subtitle_surface = self.small_font.render(subtitle, True, MUTED_TEXT)

        center_x = W // 2
        title_y = 44
        shadow = pygame.Surface((W, 90), pygame.SRCALPHA)
        shadow.fill((4, 7, 12, int(42 * fade)))
        surf.blit(shadow, (0, 24))
        surf.blit(title_surface, (center_x - title_surface.get_width() // 2, title_y))
        surf.blit(subtitle_surface, (center_x - subtitle_surface.get_width() // 2, title_y + 36))

    def draw_controls(self, surf: pygame.Surface) -> None:
        controls = "SPACE pause/resume   R restart   ESC quit"
        controls_surface = self.small_font.render(controls, True, MUTED_TEXT)
        surf.blit(
            controls_surface,
            (W - controls_surface.get_width() - 18, 18),
        )

        if self.paused:
            paused_surface = self.scene_font.render("PAUSED", True, TITLE_COLOR)
            surf.blit(
                paused_surface,
                (W - paused_surface.get_width() - 18, 42),
            )

    def draw_fade(self, surf: pygame.Surface, fade: float) -> None:
        fade = clamp(fade)
        if fade <= 0.0:
            return
        overlay = pygame.Surface(surf.get_size())
        overlay.fill((0, 0, 0))
        overlay.set_alpha(int(255 * fade))
        surf.blit(overlay, (0, 0))

    def draw(self) -> None:
        self.scene_surface.fill(BG_CLEAR)
        self.background.draw(self.scene_surface)

        if self.frame_effects.get("mist", 0.0) > 0.0:
            self.draw_mist_overlay(self.scene_surface, self.frame_effects["mist"])
        if self.frame_effects.get("ash", 0.0) > 0.0:
            self.back_ash.draw(
                self.scene_surface,
                self.frame_effects["ash"],
                color=(215, 222, 232),
                alpha_scale=92,
            )

        for loaded_actor in sorted(self.actors.values(), key=lambda item: item.z_order):
            loaded_actor.actor.draw(self.scene_surface, loaded_actor.sprite_set)

        for event in self.events_by_type.get("gesture", []):
            if event.get("name") != "hand_to_ground" or not event_active(self.current_time, event):
                continue
            self.draw_ground_pulse(
                self.scene_surface,
                event["character"],
                event_envelope(
                    self.current_time,
                    float(event["start"]),
                    float(event["end"]),
                    ramp=0.22,
                ),
            )

        self.background.draw_foreground(self.scene_surface)

        if self.frame_effects.get("ruin_glow", 0.0) > 0.0:
            self.draw_ruin_glow(self.scene_surface, self.frame_effects["ruin_glow"])

        fog_intensity = clamp((self.frame_effects.get("mist", 0.0) * 1.15) + 0.08)
        self.draw_foreground_fog(self.scene_surface, fog_intensity)

        front_ash_intensity = max(
            self.frame_effects.get("ash", 0.0) * 0.6,
            self.frame_effects.get("foreground_ash", 0.0),
        )
        if front_ash_intensity > 0.0:
            self.front_ash.draw(
                self.scene_surface,
                front_ash_intensity,
                color=(228, 234, 244),
                alpha_scale=122,
            )

        self.camera.present(self.scene_surface, self.screen)

        if self.active_text is not None:
            self.draw_caption(self.screen, self.active_text)
        self.draw_scene_title(self.screen)
        self.draw_intro_title(self.screen)
        self.draw_controls(self.screen)
        self.draw_fade(self.screen, self.frame_effects.get("fade", 0.0))


def load_actor_runtime() -> ActorRuntime:
    from main import ACTOR_ROOT, PART_SCALE, discover_models, load_model

    return ActorRuntime(
        actor_root=ACTOR_ROOT,
        part_scale=PART_SCALE,
        discover_models=discover_models,
        load_model=load_model,
    )


def main() -> None:
    pygame.init()
    actor_runtime = load_actor_runtime()

    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("The Road Under Glass")

    player = StoryPlayer(screen, actor_runtime)
    clock = pygame.time.Clock()

    while True:
        dt = clock.tick(FPS) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
                if event.key == pygame.K_SPACE:
                    player.paused = not player.paused
                if event.key == pygame.K_r:
                    player.restart()

        player.update(dt)
        player.draw()
        pygame.display.flip()


if __name__ == "__main__":
    main()
