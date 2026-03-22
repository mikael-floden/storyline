"""
Actor module for 2D skeletal animation system.

Provides a clean interface for loading, animating, and rendering skeletal actors.
"""

import math
from abc import ABC, abstractmethod


class IActorRenderer(ABC):
    """Interface for actor rendering - defines public API."""

    @abstractmethod
    def update(self, dt: float) -> None:
        """Update animation state.

        Args:
            dt: Delta time in seconds
        """
        pass

    @abstractmethod
    def set_animation(self, anim_name: str) -> None:
        """Switch to a different animation with blending.

        Args:
            anim_name: Name of animation to switch to
        """
        pass

    @abstractmethod
    def draw(self, surf, sprite_set: dict) -> None:
        """Render the actor to a surface.

        Args:
            surf: Pygame surface to draw on
            sprite_set: Dictionary of part name -> sprite surface
        """
        pass

    @abstractmethod
    def apply_model(self, model: dict) -> None:
        """Apply a loaded model to this actor.

        Args:
            model: Model data dictionary containing pivots, metrics, animations, etc.
        """
        pass


class Actor(IActorRenderer):
    """Actor implementation with skeletal animation and blending."""

    def __init__(self, x: float, ground_y: float, mode: str = "stone",
                 pivots=None, metrics=None, draw_order=None,
                 animations=None, skeleton=None, part_scale: int = 1):
        """Initialize an actor.

        Args:
            x: World x position
            ground_y: Ground y position (foot placement)
            mode: Actor type name
            pivots: Part pivot data
            metrics: Rendering metrics
            draw_order: List of part names in draw order
            animations: Animation data dictionary
            skeleton: Skeleton hierarchy data
            part_scale: Scale factor for parts
        """
        self.x = float(x)
        self.ground_y = float(ground_y)
        self.mode = mode
        self.pivots = pivots
        self.metrics = metrics
        self.draw_order = draw_order or []
        self.animations = animations or {}
        self.skeleton = skeleton
        self.part_scale = part_scale

        # Animation state
        self.current_anim = None
        self.anim_time = 0.0

        # Animation blending
        self.prev_anim = None
        self.prev_anim_time = 0.0
        self.blend_time = 0.0
        self.blend_duration = 0.5  # 500ms blend time

        # Set default animation to idle if available
        if self.animations:
            anim_names = list(self.animations.keys())
            self.current_anim = anim_names[0] if anim_names else None

    def apply_model(self, model: dict) -> None:
        """Apply a loaded model to this actor."""
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
            self.prev_anim = None
            self.blend_time = 0.0

    def update(self, dt: float) -> None:
        """Update animation state."""
        if self.current_anim and self.current_anim in self.animations:
            self.anim_time += dt

        # Update blend timer
        if self.blend_time < self.blend_duration:
            self.blend_time += dt

    def set_animation(self, anim_name: str) -> None:
        """Switch to a different animation with blending."""
        if anim_name in self.animations and anim_name != self.current_anim:
            # Store previous animation state for blending
            self.prev_anim = self.current_anim
            self.prev_anim_time = self.anim_time

            # Switch to new animation
            self.current_anim = anim_name
            self.anim_time = 0.0
            self.blend_time = 0.0

    def compute_world_transforms(self, part_rotations: dict) -> dict:
        """Compute world-space position and rotation for each part considering hierarchy.

        Args:
            part_rotations: Dictionary of part name -> rotation data

        Returns:
            Dictionary of part name -> transform data (world_x, world_y, rotation)
        """
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

    def draw(self, surf, sprite_set: dict) -> None:
        """Render the actor to a surface."""
        import pygame

        scale = self.part_scale
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
            current_rotations = self._interpolate_keyframes(
                anim.get("keyframes", []),
                self.anim_time,
                anim.get("duration", 1.0),
                anim.get("loop", True)
            )

            # Blend with previous animation if transitioning
            if self.blend_time < self.blend_duration and self.prev_anim and self.prev_anim in self.animations:
                prev_anim = self.animations[self.prev_anim]
                prev_rotations = self._interpolate_keyframes(
                    prev_anim.get("keyframes", []),
                    self.prev_anim_time,
                    prev_anim.get("duration", 1.0),
                    prev_anim.get("loop", True)
                )

                # Blend factor (0 = prev, 1 = current)
                blend_factor = self.blend_time / self.blend_duration

                # Blend all parts
                all_parts = set(current_rotations.keys()) | set(prev_rotations.keys())
                part_rotations = {}
                for part in all_parts:
                    prev_rot = prev_rotations.get(part, {}).get("rotation", 0)
                    curr_rot = current_rotations.get(part, {}).get("rotation", 0)
                    blended_rot = prev_rot + (curr_rot - prev_rot) * blend_factor
                    part_rotations[part] = {"rotation": blended_rot}
            else:
                part_rotations = current_rotations

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

            self._blit_part(actor_surf, sprite_set[name], rotation, world_x + off_x, world_y + off_y, lx, ly)

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

    def _interpolate_keyframes(self, keyframes, current_time, duration, loop):
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

    def _blit_part(self, surf, spr, angle_deg, wx, wy, lpx, lpy):
        """Draw sprite rotated so its local pivot lands at the target world pivot."""
        import pygame

        scale = self.part_scale
        rad = math.radians(angle_deg)
        c, s = math.cos(rad), math.sin(rad)
        dx = lpx * scale - spr.get_width() / 2
        dy = lpy * scale - spr.get_height() / 2
        rdx = dx * c + dy * s
        rdy = -dx * s + dy * c
        rot = pygame.transform.rotate(spr, -angle_deg)
        rect = rot.get_rect(center=(round(wx * scale - rdx), round(wy * scale - rdy)))
        surf.blit(rot, rect)
