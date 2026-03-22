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
        return {}

    with open(anim_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return data.get("animations", {})


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

    # Override render_scale if scale is specified in driver
    if scale != 1.0:
        metrics["render_scale"] = scale

    sprites = load_sprites(model_spec["sheet_path"], visible_crops)
    animations = load_animations(model_spec["sheet_path"])

    return {
        "name": model_spec["name"],
        "sheet_path": model_spec["sheet_path"],
        "driver_path": driver_path,
        "driver_mtime": os.path.getmtime(driver_path),
        "pivots": visible_pivots,
        "metrics": metrics,
        "draw_order": draw_order,
        "skeleton": skeleton,
        "sprites": sprites,
        "animations": animations,
    }


def try_reload_model(model):
    """Reload a model if its driver has changed; keep the old version on failure."""
    driver_path = model["driver_path"]
    try:
        current_mtime = os.path.getmtime(driver_path)
    except OSError:
        return model, False

    if current_mtime <= model.get("driver_mtime", 0):
        return model, False

    try:
        reloaded = load_model(model)
    except Exception as exc:
        print(f"Driver reload failed for {model['name']}: {exc}")
        model["driver_mtime"] = current_mtime
        return model, False

    print(f"Reloaded driver for {reloaded['name']}")
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


def blit_part(surf, spr, angle_deg, wx, wy, lpx, lpy):
    """
    Draw `spr` rotated so its local pivot lands at the target world pivot.
    """
    scale = PART_SCALE
    rad = math.radians(angle_deg)
    c, s = math.cos(rad), math.sin(rad)
    dx = lpx * scale - spr.get_width() / 2
    dy = lpy * scale - spr.get_height() / 2
    rdx = dx * c + dy * s
    rdy = -dx * s + dy * c
    rot = pygame.transform.rotate(spr, -angle_deg)
    rect = rot.get_rect(center=(round(wx * scale - rdx), round(wy * scale - rdy)))
    surf.blit(rot, rect)


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


def interpolate_keyframes(keyframes, current_time, duration, loop):
    """Interpolate part rotations between keyframes."""
    if not keyframes:
        return {}

    # Handle looping
    if loop and current_time >= duration:
        current_time = current_time % duration
    elif not loop and current_time >= duration:
        current_time = duration

    # Find surrounding keyframes
    prev_kf = keyframes[0]
    next_kf = keyframes[-1]

    for i, kf in enumerate(keyframes):
        if kf["time"] <= current_time:
            prev_kf = kf
        if kf["time"] >= current_time:
            next_kf = kf
            break

    # If we're exactly on a keyframe, use it
    if prev_kf["time"] == current_time:
        return prev_kf.get("parts", {})

    # Interpolate between keyframes
    if prev_kf["time"] == next_kf["time"]:
        return prev_kf.get("parts", {})

    t = (current_time - prev_kf["time"]) / (next_kf["time"] - prev_kf["time"])
    result = {}

    # Get all parts mentioned in either keyframe
    all_parts = set(prev_kf.get("parts", {}).keys()) | set(next_kf.get("parts", {}).keys())

    for part in all_parts:
        prev_rot = prev_kf.get("parts", {}).get(part, {}).get("rotation", 0)
        next_rot = next_kf.get("parts", {}).get(part, {}).get("rotation", 0)
        result[part] = {"rotation": prev_rot + (next_rot - prev_rot) * t}

    return result


class Golem:
    def __init__(self, x, ground_y, mode="stone", pivots=None, metrics=None, draw_order=None, animations=None, skeleton=None):
        self.x = float(x)
        self.ground_y = float(ground_y)
        self.mode = mode
        self.pivots = pivots
        self.metrics = metrics
        self.draw_order = draw_order or []
        self.animations = animations or {}
        self.skeleton = skeleton
        self.current_anim = None
        self.anim_time = 0.0

        # Set default animation to idle if available
        if self.animations:
            anim_names = list(self.animations.keys())
            self.current_anim = anim_names[0] if anim_names else None

    def apply_model(self, model):
        self.mode = model["name"]
        self.pivots = model["pivots"]
        self.metrics = model["metrics"]
        self.draw_order = model["draw_order"]
        self.animations = model.get("animations", {})
        self.skeleton = model.get("skeleton")

        # Reset to first animation (idle)
        if self.animations:
            anim_names = list(self.animations.keys())
            self.current_anim = anim_names[0] if anim_names else None
            self.anim_time = 0.0

    def update(self, dt):
        """Update animation state."""
        if self.current_anim and self.current_anim in self.animations:
            self.anim_time += dt

    def set_animation(self, anim_name):
        """Switch to a different animation."""
        if anim_name in self.animations and anim_name != self.current_anim:
            self.current_anim = anim_name
            self.anim_time = 0.0

    def compute_world_transforms(self, part_rotations):
        """Compute world-space position and rotation for each part considering hierarchy."""
        skeleton = self.skeleton

        if not skeleton:
            # No skeleton - use simple pivot positions
            transforms = {}
            for name in self.draw_order:
                wx, wy, lx, ly = self.pivots[name]
                rotation = part_rotations.get(name, {}).get("rotation", 0.0)
                transforms[name] = {
                    "world_x": wx,
                    "world_y": wy,
                    "rotation": rotation
                }
            return transforms

        hierarchy = skeleton.get("hierarchy", {})
        root = skeleton.get("root")

        # Find true root
        true_root = root
        visited = set()
        while true_root in hierarchy and hierarchy[true_root].get("parent"):
            parent = hierarchy[true_root].get("parent")
            if parent in visited or parent == true_root:
                break
            visited.add(true_root)
            true_root = parent

        transforms = {}

        def compute_transform(part_name):
            """Recursively compute transform for a part and its children."""
            if part_name in transforms:
                return

            # Get part's data
            if part_name not in self.pivots:
                return

            wx, wy, lx, ly = self.pivots[part_name]
            local_rotation = part_rotations.get(part_name, {}).get("rotation", 0.0)

            # Check if this part has a parent in the skeleton
            part_info = hierarchy.get(part_name, {})
            parent_name = part_info.get("parent")

            if parent_name and parent_name in self.pivots:
                # Ensure parent is computed first
                if parent_name not in transforms:
                    compute_transform(parent_name)

                # Parent's transform
                parent_transform = transforms[parent_name]
                parent_world_x = parent_transform["world_x"]
                parent_world_y = parent_transform["world_y"]
                parent_rotation = parent_transform["rotation"]

                # Get joint position in rest pose (world coordinates)
                joint = part_info.get("joint", [wx, wy])
                joint_x, joint_y = joint[0], joint[1]

                # In rest pose, this part's pivot is at (wx, wy) and joint is at (joint_x, joint_y)
                # Offset from this part's pivot to the joint in rest pose
                pivot_to_joint_x = joint_x - wx
                pivot_to_joint_y = joint_y - wy

                # When we rotate this part by local_rotation around its pivot,
                # the joint moves. But the joint should stay attached to the parent.

                # Step 1: Where is the joint in parent's coordinate space?
                parent_wx_rest, parent_wy_rest, _, _ = self.pivots[parent_name]
                joint_in_parent_x = joint_x - parent_wx_rest
                joint_in_parent_y = joint_y - parent_wy_rest

                # Step 2: Rotate joint position by parent's rotation
                rad_parent = math.radians(parent_rotation)
                cos_p = math.cos(rad_parent)
                sin_p = math.sin(rad_parent)

                rotated_joint_x = joint_in_parent_x * cos_p - joint_in_parent_y * sin_p
                rotated_joint_y = joint_in_parent_x * sin_p + joint_in_parent_y * cos_p

                # Step 3: Joint's world position
                joint_world_x = parent_world_x + rotated_joint_x
                joint_world_y = parent_world_y + rotated_joint_y

                # Step 4: Rotate pivot-to-joint offset by this part's rotation
                total_rotation = parent_rotation + local_rotation
                rad_total = math.radians(total_rotation)
                cos_t = math.cos(rad_total)
                sin_t = math.sin(rad_total)

                rotated_ptj_x = pivot_to_joint_x * cos_t - pivot_to_joint_y * sin_t
                rotated_ptj_y = pivot_to_joint_x * sin_t + pivot_to_joint_y * cos_t

                # Step 5: This part's pivot is joint position - rotated pivot-to-joint offset
                world_x = joint_world_x - rotated_ptj_x
                world_y = joint_world_y - rotated_ptj_y
                world_rotation = total_rotation
            else:
                # Root part - use pivot position directly
                world_x = wx
                world_y = wy
                world_rotation = local_rotation

            transforms[part_name] = {
                "world_x": world_x,
                "world_y": world_y,
                "rotation": world_rotation
            }

            # Recursively compute children
            for child_name, child_info in hierarchy.items():
                if child_info.get("parent") == part_name:
                    compute_transform(child_name)

        # Start from true root
        compute_transform(true_root)

        return transforms

    def draw(self, surf, sprite_set):
        scale = PART_SCALE
        pivots = self.pivots
        metrics = self.metrics
        off_x = metrics["offset_x"]
        off_y = metrics["offset_y"]
        width = max(1, int(math.ceil(metrics["full_w"] * scale)))
        height = max(1, int(math.ceil(metrics["full_h"] * scale)))
        actor_surf = pygame.Surface((width, height), pygame.SRCALPHA)

        # Get current animation rotations
        part_rotations = {}
        if self.current_anim and self.current_anim in self.animations:
            anim = self.animations[self.current_anim]
            part_rotations = interpolate_keyframes(
                anim.get("keyframes", []),
                self.anim_time,
                anim.get("duration", 1.0),
                anim.get("loop", True)
            )

        # Compute hierarchical transforms
        transforms = self.compute_world_transforms(part_rotations)

        for name in self.draw_order:
            if name not in transforms:
                continue

            wx, wy, lx, ly = pivots[name]
            transform = transforms[name]
            world_x = transform["world_x"]
            world_y = transform["world_y"]
            rotation = transform["rotation"]

            blit_part(actor_surf, sprite_set[name], rotation, world_x + off_x, world_y + off_y, lx, ly)

        render_scale = metrics["render_scale"]
        if render_scale != 1.0:
            scaled_size = (
                max(1, int(round(actor_surf.get_width() * render_scale))),
                max(1, int(round(actor_surf.get_height() * render_scale))),
            )
            actor_surf = pygame.transform.smoothscale(actor_surf, scaled_size)

        anchor_x = (metrics["canvas_cx"] + off_x) * scale * render_scale
        anchor_y = (metrics["foot_y"] + off_y) * scale * render_scale
        blit_x = int(round(self.x - anchor_x))
        blit_y = int(round(self.ground_y - anchor_y))
        surf.blit(actor_surf, (blit_x, blit_y))

        shadow_w = max(80, int(160 * render_scale))
        shadow_h = max(12, int(22 * render_scale))
        shadow = pygame.Surface((shadow_w, shadow_h), pygame.SRCALPHA)
        pygame.draw.ellipse(shadow, (0, 0, 0, 55), (0, 0, shadow_w, shadow_h))
        surf.blit(shadow, (int(self.x) - shadow_w // 2, int(self.ground_y) - shadow_h // 3))


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
    golem = Golem(
        W // 2,
        FLOOR_Y,
        mode=current_model["name"],
        pivots=current_model["pivots"],
        metrics=current_model["metrics"],
        draw_order=current_model["draw_order"],
        animations=current_model["animations"],
        skeleton=current_model["skeleton"],
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
