#!/usr/bin/env python3
"""
Story demo - Stone golem walking through Ashen Valley with parallax background.

Controls: ESC quit

Requirements: pip install pygame pillow scipy pyyaml
"""

import os
import sys

import pygame

from actor import Actor
from background import create_ashen_valley_background
from camera import SceneCamera
from foreground_fog import ForegroundFog
from main import (
    ACTOR_ROOT,
    PART_SCALE,
    discover_models,
    load_model,
)

# Screen settings
W, H = 960, 580

pygame.init()
screen = pygame.display.set_mode((W, H))
pygame.display.set_caption("Stone Golem - Ashen Valley")


def main():
    # Load stone golem model
    model_specs = discover_models(ACTOR_ROOT)
    stone_spec = next((m for m in model_specs if m["name"] == "stone"), None)

    if not stone_spec:
        print("\nERROR: Stone golem model not found.\n")
        pygame.quit()
        sys.exit(1)

    stone_model = load_model(stone_spec)

    camera = SceneCamera(W, H)
    scene_surface = pygame.Surface((camera.render_width, camera.render_height)).convert_alpha()

    # Create background on the overscanned scene surface
    background = create_ashen_valley_background(camera.render_width, camera.render_height)
    foreground_fog = ForegroundFog(
        camera.render_width,
        camera.render_height,
        os.path.join(os.path.dirname(__file__), "background", "ashen_valley"),
    )

    # Create actor - start at left side of screen
    golem = Actor(
        camera.actor_render_x,
        camera.actor_ground_y,
        mode=stone_model["name"],
        pivots=stone_model["pivots"],
        metrics=stone_model["metrics"],
        draw_order=stone_model["draw_order"],
        animations=stone_model["animations"],
        skeleton=stone_model["skeleton"],
        part_scale=PART_SCALE,
    )

    # Set to walk animation
    if "walk" in golem.animations:
        golem.set_animation("walk")

    # Camera settings
    walk_speed = 100.0  # pixels per second

    clock = pygame.time.Clock()

    while True:
        dt = clock.tick(60) / 1000.0  # Delta time in seconds

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()

        camera.update(dt, walk_speed)

        # Update background with camera position
        background.update(camera.background_camera_x, dt)
        foreground_fog.update(dt)

        # Update actor animation
        golem.x = camera.actor_render_x
        golem.ground_y = camera.actor_ground_y
        golem.update(dt)

        # Draw
        scene_surface.fill((0, 0, 0, 0))
        background.draw(scene_surface)
        golem.draw(scene_surface, stone_model["sprites"])
        foreground_fog.draw(scene_surface)
        camera.draw(scene_surface, screen)

        pygame.display.flip()


if __name__ == "__main__":
    main()
