#!/usr/bin/env python3
"""
Animated part-based actor viewer.

Controls: ← → change model | ↑ ↓ change animation | ESC quit

Requirements: pip install pygame pillow scipy pyyaml
"""

import io
import math
import os
import sys

import pygame
import yaml

from actor import Actor

ACTOR_ROOT = "actor"
PART_SCALE = 1
DISPLAY_MAX_W = 360
DISPLAY_MAX_H = 440


def compute_rig_metrics(crops, pivots, draw_order=None):
    """Derive rest-pose bounds and a safe display scale from driver data."""
    part_names = draw_order or list(crops.keys())
    min_x = min_y = 10**9
    max_x = max_y = -10**9
    max_part_dim = 0

    for name in part_names:
        if name not in crops or name not in pivots:
            continue
        x1, y1, x2, y2 = crops[name]
        w, h = x2 - x1 + 1, y2 - y1 + 1
        wx, wy, lx, ly = pivots[name]
        px, py = wx - lx, wy - ly
        min_x = min(min_x, px)
        min_y = min(min_y, py)
        max_x = max(max_x, px + w - 1)
        max_y = max(max_y, py + h - 1)
        max_part_dim = max(max_part_dim, w, h)

    if min_x == 10**9:
        raise ValueError("draw_order does not contain any valid parts")

    canvas_w = max_x - min_x + 1
    canvas_h = max_y - min_y + 1
    pad = max(40, max_part_dim // 3)
    full_w = canvas_w + pad * 2
    full_h = canvas_h + pad * 2
    render_scale = min(1.0, DISPLAY_MAX_W / full_w, DISPLAY_MAX_H / full_h)

    return {
        "min_x": min_x,
        "min_y": min_y,
        "canvas_w": canvas_w,
        "canvas_h": canvas_h,
        "canvas_cx": min_x + canvas_w / 2.0,
        "foot_y": max_y + 1.0,
        "pad": pad,
        "offset_x": pad - min_x,
        "offset_y": pad - min_y,
        "full_w": full_w,
        "full_h": full_h,
        "render_scale": render_scale,
    }


def load_driver(sheet_path):
    """Load crops, pivots, draw_order, and skeleton from driver.yaml next to the sheet."""
    driver_path = os.path.join(os.path.dirname(sheet_path), "driver.yaml")
    with open(driver_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    crops = {name: tuple(coords) for name, coords in data["crops"].items()}
    pivots = {name: tuple(coords) for name, coords in data["pivots"].items()}
    draw_order = data["draw_order"]
    skeleton = data.get("skeleton")
    scale = data.get("scale", 1.0)  # Optional scale override

    # Don't compute metrics here - let load_model do it after filtering to skeleton parts
    return crops, pivots, draw_order, skeleton, scale, driver_path


def load_animations(sheet_path):
    """Load animation data from animation.yaml next to the sheet."""
    anim_path = os.path.join(os.path.dirname(sheet_path), "animation.yaml")
    if not os.path.isfile(anim_path):
        return {}, None, 0

    mtime = os.path.getmtime(anim_path)
    with open(anim_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return data.get("animations", {}), anim_path, mtime


def find_sheet_path(model_dir):
    for filename in ("sheet.png", "sheet.jpg", "sheet.jpeg"):
        path = os.path.join(model_dir, filename)
        if os.path.isfile(path):
            return path
    return None


def discover_models(actor_root):
    """Find actor directories that have both driver.yaml and a sheet image."""
    models = []
    if not os.path.isdir(actor_root):
        return models

    for entry in sorted(os.scandir(actor_root), key=lambda item: item.name):
        if not entry.is_dir():
            continue

        driver_path = os.path.join(entry.path, "driver.yaml")
        if not os.path.isfile(driver_path):
            continue

        sheet_path = find_sheet_path(entry.path)
        if not sheet_path:
            continue

        models.append({
            "name": entry.name,
            "sheet_path": sheet_path,
        })

    return models


def filter_connected_parts(skeleton, all_parts):
    """Filter parts to only include those connected to the skeleton root.

    Returns the subset of all_parts that are connected to the root via the skeleton hierarchy.
    Preserves the original order from all_parts.
    """
    if not skeleton or "root" not in skeleton:
        return all_parts

    root = skeleton["root"]
    hierarchy = skeleton.get("hierarchy", {})

    # Find the true root by walking up the parent chain
    # In case the declared root actually has parents
    true_root = root
    visited_during_root_search = set()
    while true_root in hierarchy and hierarchy[true_root].get("parent"):
        parent = hierarchy[true_root].get("parent")
        if parent in visited_during_root_search or parent == true_root:
            # Circular reference (including self-reference), stop here
            break
        visited_during_root_search.add(true_root)
        true_root = parent

    # Find all parts connected to the root
    connected = set()

    def traverse(part):
        if part in connected:
            return
        connected.add(part)

        # Find all children of this part
        for child, data in hierarchy.items():
            if data.get("parent") == part:
                traverse(child)

    # Start from true root - find all connected parts
    traverse(true_root)

    # Filter all_parts to only include connected parts, preserving original order
    return [part for part in all_parts if part in connected]


def load_model(model_spec):
    crops, pivots, draw_order, skeleton, scale, driver_path = load_driver(model_spec["sheet_path"])

    # If skeleton exists, filter draw_order to only include connected parts
    # but preserve the original draw_order sequence
    if skeleton:
        draw_order = filter_connected_parts(skeleton, draw_order)

    visible_crops = {name: crops[name] for name in draw_order if name in crops and name in pivots}
    visible_pivots = {name: pivots[name] for name in draw_order if name in crops and name in pivots}

    # Compute metrics AFTER filtering to only visible (skeleton-connected) parts
    metrics = compute_rig_metrics(visible_crops, visible_pivots, draw_order)

    # Override render_scale with scale from driver (always apply it)
    if scale is not None:
        metrics["render_scale"] = scale

    sprites = load_sprites(model_spec["sheet_path"], visible_crops)
    animations, anim_path, anim_mtime = load_animations(model_spec["sheet_path"])

    return {
        "name": model_spec["name"],
        "sheet_path": model_spec["sheet_path"],
        "driver_path": driver_path,
        "driver_mtime": os.path.getmtime(driver_path),
        "anim_path": anim_path,
        "anim_mtime": anim_mtime,
        "pivots": visible_pivots,
        "metrics": metrics,
        "draw_order": draw_order,
        "skeleton": skeleton,
        "sprites": sprites,
        "animations": animations,
    }


def try_reload_model(model):
    """Reload a model if its driver or animations have changed; keep the old version on failure."""
    driver_path = model["driver_path"]
    anim_path = model.get("anim_path")

    driver_changed = False
    anim_changed = False

    # Check driver.yaml
    try:
        current_driver_mtime = os.path.getmtime(driver_path)
        if current_driver_mtime > model.get("driver_mtime", 0):
            driver_changed = True
    except OSError:
        pass

    # Check animation.yaml
    if anim_path:
        try:
            current_anim_mtime = os.path.getmtime(anim_path)
            if current_anim_mtime > model.get("anim_mtime", 0):
                anim_changed = True
        except OSError:
            pass

    if not driver_changed and not anim_changed:
        return model, False

    try:
        reloaded = load_model(model)
    except Exception as exc:
        print(f"Reload failed for {model['name']}: {exc}")
        if driver_changed:
            model["driver_mtime"] = current_driver_mtime
        if anim_changed:
            model["anim_mtime"] = current_anim_mtime
        return model, False

    if driver_changed and anim_changed:
        print(f"Reloaded driver and animations for {reloaded['name']}")
    elif driver_changed:
        print(f"Reloaded driver for {reloaded['name']}")
    elif anim_changed:
        print(f"Reloaded animations for {reloaded['name']}")

    return reloaded, True


def remove_black_bg(pil_img):
    import numpy as np
    from PIL import Image as PILImage
    from scipy import ndimage

    a = np.array(pil_img.convert("RGBA"))

    dark = (a[:, :, 0] < 20) & (a[:, :, 1] < 20) & (a[:, :, 2] < 20)
    lab, _ = ndimage.label(dark)

    border = set(lab[0, :].tolist() + lab[-1, :].tolist() +
                 lab[:, 0].tolist() + lab[:, -1].tolist()) - {0}
    bg_mask = np.isin(lab, list(border))

    distance = ndimage.distance_transform_edt(~bg_mask)
    alpha_mult = np.clip(distance / 2.0, 0, 1)
    a[:, :, 3] = (a[:, :, 3] * alpha_mult).astype(np.uint8)

    return PILImage.fromarray(a, "RGBA")


def load_sprites(path, crops):
    from PIL import Image as PILImage

    try:
        raw = PILImage.open(path)
    except FileNotFoundError:
        print(f"\nERROR: '{path}' not found.")
        pygame.quit()
        sys.exit(1)

    masked = remove_black_bg(raw)
    buf = io.BytesIO()
    masked.save(buf, "PNG")
    buf.seek(0)
    sheet = pygame.image.load(buf).convert_alpha()

    out = {}
    for name, (x1, y1, x2, y2) in crops.items():
        w, h = x2 - x1 + 1, y2 - y1 + 1
        surface = pygame.Surface((w, h), pygame.SRCALPHA)
        surface.blit(sheet, (0, 0), (x1, y1, w, h))
        out[name] = pygame.transform.scale(surface, (w * PART_SCALE, h * PART_SCALE))
    return out


pygame.init()
W, H = 960, 580
screen = pygame.display.set_mode((W, H))
pygame.display.set_caption("Static Actor Viewer")
FLOOR_Y = H - 90


def build_bg():
    bg = pygame.Surface((W, H))
    bg.fill((0, 0, 0))  # Pure black background
    return bg


bg_surf = build_bg()


try:
    hf = pygame.font.SysFont("monospace", 17)
except Exception:
    hf = pygame.font.Font(None, 19)


def draw_hud(surf, golem):
    mode = golem.mode
    has_glow = "fire" in mode
    accent = (255, 140, 40) if has_glow else (110, 190, 120)
    title_color = (80, 40, 20) if has_glow else (48, 42, 36)

    current_anim = golem.current_anim or "none"
    items = [
        ("← →", "Change Model"),
        ("↑ ↓", "Change Animation"),
        ("ESC", "Quit"),
        ("", f"[ {mode.upper()} - {current_anim.upper()} ]")
    ]

    x = y = 14
    for k, v in items:
        if k:
            ks = hf.render(k, True, (90, 170, 100))
            surf.blit(ks, (x, y))
            surf.blit(hf.render(" " + v, True, (70, 65, 58)), (x + ks.get_width(), y))
        else:
            surf.blit(hf.render(v, True, accent), (x, y))
        y += 22

    title_str = mode.replace("_", " ").upper()
    title = hf.render(f"{title_str}  -  Animated Part-Based Actor", True, title_color)
    surf.blit(title, (W // 2 - title.get_width() // 2, 12))


def main():
    model_specs = discover_models(ACTOR_ROOT)
    if not model_specs:
        print("\nERROR: no actor models with both driver.yaml and a sheet image were found.\n")
        pygame.quit()
        sys.exit(1)

    models = [load_model(spec) for spec in model_specs]
    current_index = next((i for i, model in enumerate(models) if model["name"] == "stone"), 0)
    if current_index:
        models = models[current_index:] + models[:current_index]
        current_index = 0

    current_model = models[current_index]
    golem = Actor(
        W // 2,
        FLOOR_Y,
        mode=current_model["name"],
        pivots=current_model["pivots"],
        metrics=current_model["metrics"],
        draw_order=current_model["draw_order"],
        animations=current_model["animations"],
        skeleton=current_model["skeleton"],
        part_scale=PART_SCALE,
    )

    clock = pygame.time.Clock()

    while True:
        dt = clock.tick(60) / 1000.0  # Delta time in seconds

        current_model, reloaded = try_reload_model(current_model)
        if reloaded:
            models[current_index] = current_model
            golem.apply_model(current_model)

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
                if ev.key == pygame.K_LEFT:
                    # Previous character
                    current_index = (current_index - 1) % len(models)
                    current_model = models[current_index]
                    golem.apply_model(current_model)
                if ev.key == pygame.K_RIGHT:
                    # Next character
                    current_index = (current_index + 1) % len(models)
                    current_model = models[current_index]
                    golem.apply_model(current_model)
                if ev.key == pygame.K_UP:
                    # Previous animation
                    if golem.animations:
                        anim_list = list(golem.animations.keys())
                        if golem.current_anim in anim_list:
                            current_idx = anim_list.index(golem.current_anim)
                            prev_idx = (current_idx - 1) % len(anim_list)
                            golem.set_animation(anim_list[prev_idx])
                        else:
                            golem.set_animation(anim_list[0])
                if ev.key == pygame.K_DOWN:
                    # Next animation
                    if golem.animations:
                        anim_list = list(golem.animations.keys())
                        if golem.current_anim in anim_list:
                            current_idx = anim_list.index(golem.current_anim)
                            next_idx = (current_idx + 1) % len(anim_list)
                            golem.set_animation(anim_list[next_idx])
                        else:
                            golem.set_animation(anim_list[0])

        golem.update(dt)

        screen.blit(bg_surf, (0, 0))
        golem.draw(screen, current_model["sprites"])
        draw_hud(screen, golem)
        pygame.display.flip()


if __name__ == "__main__":
    main()
