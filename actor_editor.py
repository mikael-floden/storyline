#!/usr/bin/env python3
"""
Actor Editor - Complete workflow for creating skeletal animations.

Workflow:
1. Select actor
2. Calculate crops and auto-placement (if not already done)
3. Adjust part positions and draw order
4. Define skeleton hierarchy
5. Save configuration

Controls:
- Arrow Keys: Switch between actors
- Mouse: Click/drag to interact
- S: Save current state
- ESC: Quit

Requirements: pip install pygame pillow scipy pyyaml numpy
"""

import io
import os
import sys
import math
from enum import Enum

import pygame
import yaml
import numpy as np


ACTOR_ROOT = "actor"


class EditorMode(Enum):
    ACTOR_SELECT = "actor_select"
    CROP_SETUP = "crop_setup"
    PLACEMENT = "placement"
    SKELETON = "skeleton"


def remove_black_bg(pil_img):
    """Process sprite sheet, respecting existing transparency.

    Only uses the alpha channel - doesn't treat black pixels as transparent.
    Black pixels with alpha > 0 are kept as black, not made transparent.
    """
    from PIL import Image as PILImage

    # Simply convert to RGBA and return - completely respect the alpha channel
    # Do NOT modify the alpha based on color values
    return pil_img.convert("RGBA")


def detect_ref_image(sheet_path):
    """Detect the reference image (leftmost dense region in the sheet).

    Strategy: Find dense clusters of pixels on the left side that form a tall region.
    """
    from PIL import Image as PILImage
    from scipy import ndimage

    img = PILImage.open(sheet_path)
    img_clean = remove_black_bg(img)

    # Convert to numpy array
    a = np.array(img_clean.convert("RGBA"))
    alpha = a[:, :, 3]

    # Use a more aggressive detection - look at pixel density in regions
    sheet_height, sheet_width = a.shape[:2]

    # Use column-based detection for fragmented sprites
    # This works better for fire/particle effects
    ref_blob = None

    if True:  # Always use this strategy for consistency
        # Look at vertical slices from left to right to find the reference figure
        # The ref should be a narrow column on the left (the character in rest pose)
        # Limit width to avoid including separate body parts laid out to the right
        max_ref_width = 450  # Maximum width for reference image

        # Find the leftmost pixels
        ref_x1 = None
        ref_y1 = sheet_height
        ref_y2 = 0

        # Search left to right until we find a tall column
        for x in range(min(sheet_width, 500)):
            col = alpha[:, x]
            if np.any(col > 30):
                if ref_x1 is None:
                    ref_x1 = x

                # Update y bounds
                y_indices = np.where(col > 30)[0]
                ref_y1 = min(ref_y1, int(y_indices[0]))
                ref_y2 = max(ref_y2, int(y_indices[-1]))

        if ref_x1 is not None:
            h = ref_y2 - ref_y1 + 1
            # Accept if reasonably tall (at least 20% of sheet height)
            if h > sheet_height * 0.2:
                # Set x2 to a reasonable width from x1, not the full extent
                # Check columns to the right but stop at first significant gap
                ref_x2 = ref_x1
                gap_size = 0
                max_gap = 15  # Stop if we see a gap wider than this

                for x in range(ref_x1 + 1, min(sheet_width, ref_x1 + max_ref_width)):
                    col = alpha[:, x]
                    # Check if this column has pixels in the ref y range
                    ref_region = col[ref_y1:ref_y2+1]
                    if np.any(ref_region > 30):
                        ref_x2 = x
                        gap_size = 0  # Reset gap counter
                    else:
                        gap_size += 1
                        if gap_size >= max_gap:
                            # Found a significant gap, stop here
                            break

                ref_blob = (ref_x1, ref_y1, ref_x2, ref_y2)

    return ref_blob


def detect_crops(sheet_path, min_particle_size=200):
    """Detect individual parts in sprite sheet using blob detection, excluding ref image.

    Args:
        sheet_path: Path to sprite sheet
        min_particle_size: Minimum area for a blob to be considered a separate part.
                          Smaller blobs are merged into nearest part.
    """
    from PIL import Image as PILImage
    from scipy import ndimage

    img = PILImage.open(sheet_path)
    img_clean = remove_black_bg(img)

    # Convert to numpy array
    a = np.array(img_clean.convert("RGBA"))
    alpha = a[:, :, 3]

    # Detect reference image
    ref_bbox = detect_ref_image(sheet_path)

    # Find connected components in alpha channel
    binary = alpha > 30  # Threshold for transparency
    labeled, num_features = ndimage.label(binary)

    blobs = {}
    particles = {}

    for i in range(1, num_features + 1):
        mask = labeled == i

        # Get bounding box
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)

        if not rows.any() or not cols.any():
            continue

        y_coords = np.where(rows)[0]
        x_coords = np.where(cols)[0]

        x1, x2 = int(x_coords[0]), int(x_coords[-1])
        y1, y2 = int(y_coords[0]), int(y_coords[-1])

        # Skip if this is the reference image
        if ref_bbox:
            rx1, ry1, rx2, ry2 = ref_bbox
            # Check if bboxes overlap significantly
            overlap_x = min(x2, rx2) - max(x1, rx1)
            overlap_y = min(y2, ry2) - max(y1, ry1)
            if overlap_x > 0 and overlap_y > 0:
                w, h = x2 - x1 + 1, y2 - y1 + 1
                overlap_area = overlap_x * overlap_y
                blob_area = w * h
                if overlap_area / blob_area > 0.5:  # More than 50% overlap
                    continue

        # Skip very small blobs (noise)
        w, h = x2 - x1 + 1, y2 - y1 + 1
        if w < 5 or h < 5:
            continue

        # Calculate center of mass for sorting
        cy, cx = ndimage.center_of_mass(mask)
        area = w * h

        blob_data = {
            "coords": [x1, y1, x2, y2],
            "center": (cx, cy),
            "area": area
        }

        # Separate into parts and particles
        if area < min_particle_size:
            particles[i] = blob_data
        else:
            blobs[i] = blob_data

    # Merge particles into nearest parts
    for particle_id, particle_data in particles.items():
        px, py = particle_data["center"]
        min_dist = float('inf')
        nearest_blob_id = None

        # Find nearest blob
        for blob_id, blob_data in blobs.items():
            bx, by = blob_data["center"]
            dist = math.sqrt((px - bx)**2 + (py - by)**2)
            if dist < min_dist:
                min_dist = dist
                nearest_blob_id = blob_id

        # Merge particle into nearest blob
        if nearest_blob_id:
            blob = blobs[nearest_blob_id]
            px1, py1, px2, py2 = particle_data["coords"]
            bx1, by1, bx2, by2 = blob["coords"]

            # Expand blob bbox to include particle
            new_coords = [
                min(bx1, px1),
                min(by1, py1),
                max(bx2, px2),
                max(by2, py2)
            ]
            blob["coords"] = new_coords

            # Update area
            w = new_coords[2] - new_coords[0] + 1
            h = new_coords[3] - new_coords[1] + 1
            blob["area"] = w * h

    # Sort by vertical position then horizontal (top-left to bottom-right)
    sorted_blobs = sorted(blobs.items(), key=lambda x: (x[1]["center"][0] // 100, x[1]["center"][1]))

    # Rename with better ordering
    final_crops = {}
    for idx, (_, data) in enumerate(sorted_blobs):
        name = f"part_{idx + 1:02d}"
        final_crops[name] = data["coords"]

    return final_crops, ref_bbox


def auto_place_parts(crops, ref_image_path=None):
    """
    Auto-place parts in rest pose.
    This is a simple grid-based placement. For better results, use reference image matching.
    """
    pivots = {}

    # Simple strategy: place parts in a grid pattern with center pivots
    for name, (x1, y1, x2, y2) in crops.items():
        w, h = x2 - x1 + 1, y2 - y1 + 1

        # World pivot at center of bounding box for now
        # Local pivot at center of part
        lx, ly = w // 2, h // 2

        # World position - we'll arrange in a simple rest pose
        # User will adjust these manually
        wx, wy = lx, ly

        pivots[name] = [wx, wy, lx, ly]

    return pivots


def find_sheet_path(model_dir):
    """Find sprite sheet in model directory."""
    for filename in ("sheet.png", "sheet.jpg", "sheet.jpeg"):
        path = os.path.join(model_dir, filename)
        if os.path.isfile(path):
            return path
    return None


def discover_models(actor_root):
    """Find all actor directories with sheet images."""
    models = []
    if not os.path.isdir(actor_root):
        return models

    for entry in sorted(os.scandir(actor_root), key=lambda item: item.name):
        if not entry.is_dir():
            continue

        sheet_path = find_sheet_path(entry.path)
        if not sheet_path:
            continue

        models.append({
            "name": entry.name,
            "sheet_path": sheet_path,
            "dir": entry.path,
        })

    return models


def load_driver(sheet_path):
    """Load driver.yaml configuration."""
    driver_path = os.path.join(os.path.dirname(sheet_path), "driver.yaml")

    if not os.path.exists(driver_path):
        return None, driver_path

    with open(driver_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return data, driver_path


def save_driver(driver_path, data):
    """Save driver.yaml configuration."""
    with open(driver_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    print(f"Saved to {driver_path}")


def load_sheet_surface(sheet_path, max_width=None, max_height=None):
    """Load and process sprite sheet with optional scaling."""
    from PIL import Image as PILImage

    try:
        raw = PILImage.open(sheet_path)
    except FileNotFoundError:
        print(f"\nERROR: '{sheet_path}' not found.")
        sys.exit(1)

    # Scale down if needed
    if max_width or max_height:
        w, h = raw.size
        scale = 1.0
        if max_width and w > max_width:
            scale = min(scale, max_width / w)
        if max_height and h > max_height:
            scale = min(scale, max_height / h)

        if scale < 1.0:
            new_w, new_h = int(w * scale), int(h * scale)
            raw = raw.resize((new_w, new_h), PILImage.Resampling.LANCZOS)

    masked = remove_black_bg(raw)
    buf = io.BytesIO()
    masked.save(buf, "PNG")
    buf.seek(0)
    return pygame.image.load(buf).convert_alpha()


def extract_ref_image(sheet_surface, ref_bbox, scale=1.0):
    """Extract and desaturate reference image from sheet."""
    if not ref_bbox:
        return None

    x1, y1, x2, y2 = ref_bbox
    x1, y1, x2, y2 = int(x1 * scale), int(y1 * scale), int(x2 * scale), int(y2 * scale)
    w, h = x2 - x1 + 1, y2 - y1 + 1

    # Extract reference image
    ref_surf = pygame.Surface((w, h), pygame.SRCALPHA)
    ref_surf.blit(sheet_surface, (0, 0), (x1, y1, w, h))

    # Desaturate (convert to grayscale but keep as RGBA)
    arr = pygame.surfarray.pixels3d(ref_surf)
    alpha_arr = pygame.surfarray.pixels_alpha(ref_surf)

    # Calculate grayscale
    gray = np.dot(arr[..., :3].transpose(1, 0, 2), [0.299, 0.587, 0.114])

    # Create new surface with reduced saturation
    result = pygame.Surface((w, h), pygame.SRCALPHA)
    result_arr = pygame.surfarray.pixels3d(result)
    result_alpha = pygame.surfarray.pixels_alpha(result)

    # Blend original with grayscale (50% saturation)
    for i in range(3):
        result_arr[:, :, i] = (arr[:, :, i].transpose(1, 0) * 0.3 + gray * 0.7).astype(np.uint8).transpose(1, 0)

    result_alpha[:, :] = (alpha_arr.transpose(1, 0) * 0.6).astype(np.uint8).transpose(1, 0)  # Reduce opacity

    del arr, alpha_arr, result_arr, result_alpha

    return result


def load_sprites(sheet_surface, crops, scale=1.0):
    """Extract individual sprites from sheet."""
    out = {}
    for name, coords in crops.items():
        x1, y1, x2, y2 = coords
        x1, y1, x2, y2 = int(x1 * scale), int(y1 * scale), int(x2 * scale), int(y2 * scale)
        w, h = x2 - x1 + 1, y2 - y1 + 1
        surface = pygame.Surface((w, h), pygame.SRCALPHA)
        surface.blit(sheet_surface, (0, 0), (x1, y1, w, h))
        out[name] = surface
    return out


class Button:
    def __init__(self, x, y, width, height, text, color=(80, 80, 80)):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.hover = False

    def draw(self, screen, font):
        color = tuple(min(c + 20, 255) for c in self.color) if self.hover else self.color
        pygame.draw.rect(screen, color, self.rect)
        pygame.draw.rect(screen, (200, 200, 200), self.rect, 2)

        text_surf = font.render(self.text, True, (255, 255, 255))
        text_rect = text_surf.get_rect(center=self.rect.center)
        screen.blit(text_surf, text_rect)

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.hover = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                return True
        return False


class ActorEditor:
    def __init__(self, screen_width, screen_height):
        self.screen_width = screen_width
        self.screen_height = screen_height

        # Discover actors
        self.models = discover_models(ACTOR_ROOT)
        if not self.models:
            print(f"No actors found in '{ACTOR_ROOT}'")
            sys.exit(1)

        self.current_actor_idx = 0
        self.mode = EditorMode.ACTOR_SELECT

        # Font
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 16)  # Smaller font for lists

        # UI elements
        self.buttons = {}

        # Editor state
        self.sheet_surface = None
        self.sheet_scale = 1.0
        self.sprites = {}
        self.data = None
        self.driver_path = None
        self.ref_image = None
        self.ref_bbox = None

        # Placement mode state
        self.dragging_part = None
        self.drag_offset = (0, 0)
        self.part_positions = {}
        self.selected_part_for_rename = None
        self.renaming_active = False
        self.rename_text = ""

        # Skeleton mode state
        self.skeleton_state = {
            "selected_part": None,
            "selected_parent": None,
            "mode": "select_part"  # select_part, select_parent, place_pivot
        }

        # Draw order list
        self.draw_order_dragging = None
        self.draw_order_scroll = 0  # Scroll offset for draw order list

        # Viewport
        self.assembly_offset = [100, 100]
        self.sheet_offset = [100, 100]

        # Load first actor
        self.load_actor(self.current_actor_idx)

    def load_actor(self, idx):
        """Load an actor and determine what mode to start in."""
        self.current_actor_idx = idx
        model = self.models[idx]

        self.data, self.driver_path = load_driver(model["sheet_path"])

        # Clear cached images when switching actors
        self.ref_image = None
        self.ref_bbox = None

        # Determine sheet scale to fit screen
        from PIL import Image as PILImage
        img = PILImage.open(model["sheet_path"])
        w, h = img.size
        max_sheet_width = self.screen_width // 2 - 150
        max_sheet_height = self.screen_height - 200

        self.sheet_scale = min(1.0, max_sheet_width / w, max_sheet_height / h)

        # Load sheet surface
        if self.sheet_scale < 1.0:
            self.sheet_surface = load_sheet_surface(model["sheet_path"],
                                                    int(w * self.sheet_scale),
                                                    int(h * self.sheet_scale))
        else:
            self.sheet_surface = load_sheet_surface(model["sheet_path"])

        # Determine mode
        if self.data is None or "crops" not in self.data:
            self.mode = EditorMode.CROP_SETUP
            self.setup_crop_mode()
        elif "skeleton" not in self.data:
            self.mode = EditorMode.PLACEMENT
            self.setup_placement_mode()
        else:
            self.mode = EditorMode.SKELETON
            self.setup_skeleton_mode()

    def setup_crop_mode(self):
        """Setup crop detection mode."""
        self.buttons["calc_crops"] = Button(
            self.screen_width // 2 - 150,
            self.screen_height // 2,
            300, 50,
            "Calculate Crops and Placement"
        )

    def setup_placement_mode(self):
        """Setup placement adjustment mode."""
        # Clear old buttons
        self.buttons.clear()

        # Load sprites
        self.sprites = load_sprites(self.sheet_surface, self.data["crops"], self.sheet_scale)

        # Load reference image if available
        if "ref_bbox" in self.data and not self.ref_image:
            self.ref_bbox = tuple(self.data["ref_bbox"])
            self.ref_image = extract_ref_image(self.sheet_surface, self.ref_bbox, self.sheet_scale)

        # Calculate initial positions if not set
        self.part_positions = {}
        for name in self.data["draw_order"]:
            if name in self.data["pivots"]:
                wx, wy, lx, ly = self.data["pivots"][name]
                px = (wx - lx) * self.sheet_scale
                py = (wy - ly) * self.sheet_scale
                self.part_positions[name] = [px, py]

        # Renaming state
        self.selected_part_for_rename = None

        # Draw order dragging state
        self.draw_order_drag_idx = None
        self.draw_order_hover_idx = None

        self.buttons["apply_placement"] = Button(20, self.screen_height - 60, 200, 40, "Apply & Continue")
        self.buttons["recalc_crops"] = Button(240, self.screen_height - 60, 200, 40, "Recalculate Crops")
        self.buttons["rename_part"] = Button(460, self.screen_height - 60, 150, 40, "Rename Part")

    def setup_skeleton_mode(self):
        """Setup skeleton editing mode."""
        # Clear old buttons
        self.buttons.clear()

        self.sprites = load_sprites(self.sheet_surface, self.data["crops"], self.sheet_scale)

        # Load reference image if available
        if "ref_bbox" in self.data and not self.ref_image:
            self.ref_bbox = tuple(self.data["ref_bbox"])
            self.ref_image = extract_ref_image(self.sheet_surface, self.ref_bbox, self.sheet_scale)

        # Calculate rest positions
        self.part_positions = {}
        for name in self.data["draw_order"]:
            if name in self.data["pivots"]:
                wx, wy, lx, ly = self.data["pivots"][name]
                x1, y1, x2, y2 = self.data["crops"][name]
                px = (wx - lx) * self.sheet_scale
                py = (wy - ly) * self.sheet_scale
                self.part_positions[name] = [px, py]

        # Initialize skeleton if not present
        if "skeleton" not in self.data:
            self.data["skeleton"] = {
                "root": self.data["draw_order"][0] if self.data["draw_order"] else "part_01",
                "hierarchy": {}
            }

        self.skeleton_state = {
            "selected_part": None,
            "selected_parent": None,
            "mode": "select_part"
        }

        self.buttons["save_skeleton"] = Button(20, self.screen_height - 60, 150, 40, "Save")
        self.buttons["set_root"] = Button(190, self.screen_height - 60, 150, 40, "Set Root")
        self.buttons["back_to_placement"] = Button(360, self.screen_height - 60, 200, 40, "Back to Placement")

    def calculate_crops(self):
        """Calculate crops and initial placement."""
        model = self.models[self.current_actor_idx]

        print("Detecting parts...")
        print(f"Sheet path: {model['sheet_path']}")

        # Debug: detect ref first
        ref_bbox_detected = detect_ref_image(model["sheet_path"])
        print(f"Reference detection result: {ref_bbox_detected}")

        crops, ref_bbox = detect_crops(model["sheet_path"])
        self.ref_bbox = ref_bbox

        print(f"Found {len(crops)} parts")
        print(f"Final ref_bbox: {ref_bbox}")

        # Extract and process reference image
        if ref_bbox:
            self.ref_image = extract_ref_image(self.sheet_surface, ref_bbox, self.sheet_scale)
            x1, y1, x2, y2 = ref_bbox
            print(f"Reference image: x=[{x1}, {x2}] ({x2-x1+1}px), y=[{y1}, {y2}] ({y2-y1+1}px)")

        # Auto-place
        pivots = auto_place_parts(crops)

        # Create draw order (sorted by name for now)
        draw_order = sorted(crops.keys())

        # Save to data
        self.data = {
            "crops": crops,
            "pivots": pivots,
            "draw_order": draw_order
        }

        if ref_bbox:
            self.data["ref_bbox"] = list(ref_bbox)

        save_driver(self.driver_path, self.data)

        # Force clear the cached ref image so it reloads with new bbox
        self.ref_image = None
        self.ref_bbox = None

        # Switch to placement mode
        self.mode = EditorMode.PLACEMENT
        self.setup_placement_mode()

    def find_part_at_pos(self, x, y):
        """Find part at screen position (in assembly view)."""
        # Check in reverse draw order (front to back)
        for name in reversed(self.data["draw_order"]):
            if name not in self.part_positions or name not in self.sprites:
                continue

            px, py = self.part_positions[name]
            sx, sy = px + self.assembly_offset[0], py + self.assembly_offset[1]
            sprite = self.sprites[name]
            w, h = sprite.get_width(), sprite.get_height()

            if sx <= x <= sx + w and sy <= y <= sy + h:
                # Check pixel alpha
                local_x, local_y = x - sx, y - sy
                if 0 <= local_x < w and 0 <= local_y < h:
                    try:
                        alpha = sprite.get_at((int(local_x), int(local_y)))[3]
                        if alpha > 30:
                            return name
                    except IndexError:
                        pass
        return None

    def get_draw_order_rect(self, idx, sheet_x, list_y):
        """Get the rect for a draw order list item."""
        y = list_y + 25 + idx * 20
        return pygame.Rect(sheet_x + 10, y, 200, 18)

    def handle_placement_click(self, x, y, button):
        """Handle clicks in placement mode."""
        if button == 1:  # Left click
            # Check if clicking on draw order list
            if hasattr(self, 'last_draw_order_rects'):
                for idx, rect in enumerate(self.last_draw_order_rects):
                    if rect.collidepoint(x, y):
                        self.draw_order_drag_idx = idx
                        return

            part = self.find_part_at_pos(x, y)
            if part:
                self.dragging_part = part
                self.selected_part_for_rename = part
                px, py = self.part_positions[part]
                self.drag_offset = (x - self.assembly_offset[0] - px,
                                   y - self.assembly_offset[1] - py)

    def handle_placement_drag(self, x, y):
        """Handle dragging in placement mode."""
        # Handle draw order dragging
        if self.draw_order_drag_idx is not None:
            if hasattr(self, 'last_draw_order_rects'):
                self.draw_order_hover_idx = None
                for idx, rect in enumerate(self.last_draw_order_rects):
                    if rect.collidepoint(x, y):
                        self.draw_order_hover_idx = idx
                        break
            return

        if self.dragging_part:
            new_x = x - self.assembly_offset[0] - self.drag_offset[0]
            new_y = y - self.assembly_offset[1] - self.drag_offset[1]
            self.part_positions[self.dragging_part] = [new_x, new_y]

            # Update pivots
            name = self.dragging_part
            x1, y1, x2, y2 = self.data["crops"][name]
            _, _, lx, ly = self.data["pivots"][name]
            wx = new_x / self.sheet_scale + lx
            wy = new_y / self.sheet_scale + ly
            self.data["pivots"][name] = [wx, wy, lx, ly]

    def handle_placement_release(self):
        """Handle mouse release in placement mode."""
        # Handle draw order reordering
        if self.draw_order_drag_idx is not None and self.draw_order_hover_idx is not None:
            if self.draw_order_drag_idx != self.draw_order_hover_idx:
                # Reorder
                item = self.data["draw_order"].pop(self.draw_order_drag_idx)
                self.data["draw_order"].insert(self.draw_order_hover_idx, item)
                print(f"Reordered: moved {item} from position {self.draw_order_drag_idx} to {self.draw_order_hover_idx}")

        self.draw_order_drag_idx = None
        self.draw_order_hover_idx = None
        self.dragging_part = None

    def rename_part(self):
        """Rename the selected part."""
        if not self.selected_part_for_rename:
            print("No part selected. Click a part first.")
            return

        print(f"\nRenaming {self.selected_part_for_rename}")
        print("Enter new name (or press Enter to cancel): ", end="", flush=True)

        # We need to use pygame event handling for text input
        self.renaming_active = True
        self.rename_text = ""

    def finish_rename(self):
        """Complete the rename operation."""
        if not self.rename_text or not self.selected_part_for_rename:
            self.renaming_active = False
            return

        old_name = self.selected_part_for_rename
        new_name = self.rename_text.strip()

        if not new_name or new_name == old_name:
            self.renaming_active = False
            return

        # Check if name already exists
        if new_name in self.data["crops"]:
            print(f"Name '{new_name}' already exists!")
            self.renaming_active = False
            return

        # Update crops
        self.data["crops"][new_name] = self.data["crops"].pop(old_name)

        # Update pivots
        if old_name in self.data["pivots"]:
            self.data["pivots"][new_name] = self.data["pivots"].pop(old_name)

        # Update draw order
        if old_name in self.data["draw_order"]:
            idx = self.data["draw_order"].index(old_name)
            self.data["draw_order"][idx] = new_name

        # Update part positions
        if old_name in self.part_positions:
            self.part_positions[new_name] = self.part_positions.pop(old_name)

        # Update sprites
        if old_name in self.sprites:
            self.sprites[new_name] = self.sprites.pop(old_name)

        # Update skeleton if present
        if "skeleton" in self.data:
            if self.data["skeleton"].get("root") == old_name:
                self.data["skeleton"]["root"] = new_name

            if "hierarchy" in self.data["skeleton"]:
                # Update as child
                if old_name in self.data["skeleton"]["hierarchy"]:
                    self.data["skeleton"]["hierarchy"][new_name] = self.data["skeleton"]["hierarchy"].pop(old_name)

                # Update as parent
                for part_data in self.data["skeleton"]["hierarchy"].values():
                    if part_data.get("parent") == old_name:
                        part_data["parent"] = new_name

        self.selected_part_for_rename = new_name
        print(f"Renamed {old_name} -> {new_name}")
        self.renaming_active = False
        self.rename_text = ""

    def apply_placement(self):
        """Save placement and move to skeleton mode."""
        save_driver(self.driver_path, self.data)
        self.mode = EditorMode.SKELETON
        self.setup_skeleton_mode()

    def find_part_at_sheet_pos(self, x, y):
        """Find part at position in sheet view (right side)."""
        if not hasattr(self, 'last_crop_rects'):
            return None

        for name, rect in self.last_crop_rects.items():
            if rect.collidepoint(x, y):
                return name
        return None

    def handle_skeleton_click(self, x, y):
        """Handle clicks in skeleton mode."""
        state = self.skeleton_state

        if state["mode"] == "select_part":
            # Select a part - check both assembly view and sheet view
            part = self.find_part_at_pos(x, y)
            if not part:
                part = self.find_part_at_sheet_pos(x, y)

            if part:
                state["selected_part"] = part
                state["mode"] = "select_parent"
                print(f"Selected: {part}. Click parent or press Set Root.")

        elif state["mode"] == "select_parent":
            # Select parent - check both assembly view and sheet view
            parent = self.find_part_at_pos(x, y)
            if not parent:
                parent = self.find_part_at_sheet_pos(x, y)

            if parent:
                state["selected_parent"] = parent
                state["mode"] = "place_pivot"
                print(f"Parent: {parent}. Click to place joint.")

        elif state["mode"] == "place_pivot":
            # Place pivot
            wx = (x - self.assembly_offset[0]) / self.sheet_scale
            wy = (y - self.assembly_offset[1]) / self.sheet_scale

            if "hierarchy" not in self.data["skeleton"]:
                self.data["skeleton"]["hierarchy"] = {}

            self.data["skeleton"]["hierarchy"][state["selected_part"]] = {
                "parent": state["selected_parent"],
                "joint": [float(wx), float(wy)]
            }

            print(f"Joint created: {state['selected_part']} -> {state['selected_parent']}")

            # Reset
            state["selected_part"] = None
            state["selected_parent"] = None
            state["mode"] = "select_part"

    def set_root(self):
        """Set selected part as root."""
        if self.skeleton_state["selected_part"]:
            self.data["skeleton"]["root"] = self.skeleton_state["selected_part"]
            print(f"Root set to: {self.skeleton_state['selected_part']}")
            self.skeleton_state["selected_part"] = None
            self.skeleton_state["mode"] = "select_part"

    def save_current(self):
        """Save current state."""
        if self.data:
            save_driver(self.driver_path, self.data)

    def next_actor(self):
        """Switch to next actor."""
        self.current_actor_idx = (self.current_actor_idx + 1) % len(self.models)
        self.load_actor(self.current_actor_idx)

    def prev_actor(self):
        """Switch to previous actor."""
        self.current_actor_idx = (self.current_actor_idx - 1) % len(self.models)
        self.load_actor(self.current_actor_idx)

    def draw(self, screen):
        """Draw current state."""
        screen.fill((25, 25, 30))

        # Draw title
        model = self.models[self.current_actor_idx]
        title = self.font.render(f"Actor: {model['name']} ({self.current_actor_idx + 1}/{len(self.models)}) - Mode: {self.mode.value}",
                                True, (255, 255, 255))
        screen.blit(title, (20, 20))

        # Draw actor navigation
        nav_text = self.small_font.render("← → to switch actors", True, (150, 150, 150))
        screen.blit(nav_text, (20, 50))

        if self.mode == EditorMode.CROP_SETUP:
            self.draw_crop_setup(screen)
        elif self.mode == EditorMode.PLACEMENT:
            self.draw_placement(screen)
        elif self.mode == EditorMode.SKELETON:
            self.draw_skeleton(screen)

    def draw_crop_setup(self, screen):
        """Draw crop setup screen."""
        # Show sheet
        if self.sheet_surface:
            x = self.screen_width // 2 - self.sheet_surface.get_width() // 2
            y = 150
            screen.blit(self.sheet_surface, (x, y))

        # Draw button
        self.buttons["calc_crops"].draw(screen, self.font)

        # Instructions
        inst = self.small_font.render("No crops detected. Click button to auto-detect parts.", True, (200, 200, 200))
        screen.blit(inst, (self.screen_width // 2 - 200, self.screen_height // 2 - 50))

    def draw_coordinate_axes(self, screen, origin_x, origin_y, size=400):
        """Draw 2D coordinate axes."""
        # X-axis (horizontal)
        pygame.draw.line(screen, (100, 100, 120),
                        (origin_x - size, origin_y),
                        (origin_x + size, origin_y), 2)
        # Y-axis (vertical)
        pygame.draw.line(screen, (100, 100, 120),
                        (origin_x, origin_y - size),
                        (origin_x, origin_y + size), 2)

        # Tick marks every 50 pixels
        for i in range(-size, size + 1, 50):
            # X-axis ticks
            pygame.draw.line(screen, (80, 80, 100),
                           (origin_x + i, origin_y - 3),
                           (origin_x + i, origin_y + 3), 1)
            # Y-axis ticks
            pygame.draw.line(screen, (80, 80, 100),
                           (origin_x - 3, origin_y + i),
                           (origin_x + 3, origin_y + i), 1)

        # Origin marker
        pygame.draw.circle(screen, (150, 150, 170), (origin_x, origin_y), 4)

    def draw_ref_image_centered(self, screen, origin_x, origin_y):
        """Draw reference image centered on Y-axis and standing on X-axis."""
        if not self.ref_image:
            return

        ref_w = self.ref_image.get_width()
        ref_h = self.ref_image.get_height()

        # Center on Y-axis (horizontally), stand on X-axis (feet at origin_y)
        ref_x = origin_x - ref_w // 2
        ref_y = origin_y - ref_h  # Bottom at X-axis

        screen.blit(self.ref_image, (ref_x, ref_y))

    def draw_placement(self, screen):
        """Draw placement adjustment screen."""
        # Split screen: assembly on left, sheet on right
        split_x = self.screen_width // 2

        # Assembly view
        pygame.draw.rect(screen, (35, 35, 40), (0, 80, split_x, self.screen_height - 80))
        label = self.small_font.render("Assembly (Drag parts to adjust)", True, (200, 200, 200))
        screen.blit(label, (20, 85))

        # Draw coordinate axes
        origin_x = self.assembly_offset[0] + 200
        origin_y = self.assembly_offset[1] + 350
        self.draw_coordinate_axes(screen, origin_x, origin_y)

        # Draw reference image
        self.draw_ref_image_centered(screen, origin_x, origin_y)

        # Draw assembled parts
        for name in self.data["draw_order"]:
            if name not in self.part_positions or name not in self.sprites:
                continue

            px, py = self.part_positions[name]
            sx, sy = px + self.assembly_offset[0], py + self.assembly_offset[1]

            # Highlight if dragging
            if name == self.dragging_part:
                sprite = self.sprites[name]
                highlight = sprite.copy()
                highlight.fill((100, 255, 100, 80), special_flags=pygame.BLEND_RGBA_ADD)
                screen.blit(highlight, (sx, sy))

            screen.blit(self.sprites[name], (sx, sy))

        # Sheet view
        sheet_x = split_x + 20
        sheet_y = 110
        label2 = self.small_font.render("Sprite Sheet", True, (200, 200, 200))
        screen.blit(label2, (sheet_x, 85))

        if self.sheet_surface:
            screen.blit(self.sheet_surface, (sheet_x, sheet_y))

        # Draw order list (right side, below sheet)
        list_y = sheet_y + self.sheet_surface.get_height() + 20
        list_label = self.small_font.render("Draw Order (drag to reorder, scroll with mouse wheel):", True, (200, 200, 200))
        screen.blit(list_label, (sheet_x, list_y))

        # Define scrollable area
        list_start_y = list_y + 25
        list_max_height = self.screen_height - list_start_y - 150  # Leave room for buttons and rename input
        list_width = 250

        # Create clipping rect for scrollable area
        clip_rect = pygame.Rect(sheet_x, list_start_y, list_width, list_max_height)
        screen.set_clip(clip_rect)

        # Draw background for list area
        pygame.draw.rect(screen, (25, 25, 30), clip_rect)

        # Store rects for hit detection
        self.last_draw_order_rects = []

        for i, name in enumerate(self.data["draw_order"]):
            # Skip the dragged item at its original position
            if i == self.draw_order_drag_idx:
                self.last_draw_order_rects.append(pygame.Rect(0, 0, 0, 0))  # Dummy rect
                continue

            y = list_start_y + i * 16 - self.draw_order_scroll  # Smaller line height

            # Skip if outside visible area (optimization)
            if y < list_start_y - 16 or y > list_start_y + list_max_height + 16:
                self.last_draw_order_rects.append(pygame.Rect(0, 0, 0, 0))
                continue

            # Draw insertion indicator
            if self.draw_order_drag_idx is not None and i == self.draw_order_hover_idx:
                pygame.draw.line(screen, (100, 255, 100), (sheet_x + 5, y - 2), (sheet_x + 205, y - 2), 2)

            # Highlight selected part
            is_selected = name == self.selected_part_for_rename
            is_dragging = i == self.draw_order_drag_idx

            if is_selected and not is_dragging:
                color = (255, 255, 100)
            else:
                color = (180, 180, 180)

            text = self.small_font.render(f"{i+1}. {name}", True, color)
            rect = pygame.Rect(sheet_x + 10, y, 200, 14)  # Smaller rect height
            self.last_draw_order_rects.append(rect)

            # Highlight on hover
            if rect.collidepoint(pygame.mouse.get_pos()):
                pygame.draw.rect(screen, (60, 60, 70), rect.inflate(4, 2))

            screen.blit(text, (sheet_x + 10, y))

        # Draw scrollbar if needed
        total_height = len(self.data["draw_order"]) * 16  # Match line height
        if total_height > list_max_height:
            scrollbar_x = sheet_x + list_width - 10
            scrollbar_height = max(20, int(list_max_height * list_max_height / total_height))
            scrollbar_y = list_start_y + int(self.draw_order_scroll * list_max_height / total_height)
            pygame.draw.rect(screen, (100, 100, 120), (scrollbar_x, scrollbar_y, 8, scrollbar_height))

        # Clear clipping
        screen.set_clip(None)

        # Draw the dragged item at mouse position
        if self.draw_order_drag_idx is not None:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            dragged_name = self.data["draw_order"][self.draw_order_drag_idx]
            text = self.small_font.render(f"{self.draw_order_drag_idx+1}. {dragged_name}", True, (150, 255, 150))
            shadow_rect = pygame.Rect(mouse_x - 5, mouse_y - 10, 210, 20)
            pygame.draw.rect(screen, (40, 40, 50), shadow_rect)
            pygame.draw.rect(screen, (100, 255, 100), shadow_rect, 1)
            screen.blit(text, (mouse_x, mouse_y - 10))

        # Show rename input if active
        if self.renaming_active:
            input_y = list_y + 25 + len(self.data["draw_order"]) * 20 + 20
            prompt = self.small_font.render(f"Rename '{self.selected_part_for_rename}' to:", True, (150, 255, 150))
            screen.blit(prompt, (sheet_x, input_y))

            input_text = self.small_font.render(self.rename_text + "_", True, (255, 255, 255))
            screen.blit(input_text, (sheet_x, input_y + 20))

            hint = self.small_font.render("(Enter to confirm, Esc to cancel)", True, (150, 150, 150))
            screen.blit(hint, (sheet_x, input_y + 40))

        # Draw buttons
        for btn in self.buttons.values():
            btn.draw(screen, self.font)

    def draw_skeleton(self, screen):
        """Draw skeleton editing screen."""
        # Split screen
        split_x = self.screen_width // 2

        # Assembly view
        pygame.draw.rect(screen, (35, 35, 40), (0, 80, split_x, self.screen_height - 80))
        label = self.small_font.render("Assembly (Click to define skeleton)", True, (200, 200, 200))
        screen.blit(label, (20, 85))

        # Draw coordinate axes
        origin_x = self.assembly_offset[0] + 200
        origin_y = self.assembly_offset[1] + 350
        self.draw_coordinate_axes(screen, origin_x, origin_y)

        # Draw reference image
        self.draw_ref_image_centered(screen, origin_x, origin_y)

        # Draw parts
        state = self.skeleton_state
        for name in self.data["draw_order"]:
            if name not in self.part_positions or name not in self.sprites:
                continue

            px, py = self.part_positions[name]
            sx, sy = px + self.assembly_offset[0], py + self.assembly_offset[1]

            # Highlight
            sprite = self.sprites[name]
            if name == state["selected_part"]:
                highlight = sprite.copy()
                highlight.fill((255, 255, 0, 100), special_flags=pygame.BLEND_RGBA_ADD)
                screen.blit(highlight, (sx, sy))
            elif name == state["selected_parent"]:
                highlight = sprite.copy()
                highlight.fill((0, 255, 0, 100), special_flags=pygame.BLEND_RGBA_ADD)
                screen.blit(highlight, (sx, sy))

            screen.blit(sprite, (sx, sy))

        # Draw skeleton connections
        for part, data in self.data["skeleton"].get("hierarchy", {}).items():
            parent = data.get("parent")
            joint = data.get("joint")
            if parent and joint and part in self.part_positions:
                jx = joint[0] * self.sheet_scale + self.assembly_offset[0]
                jy = joint[1] * self.sheet_scale + self.assembly_offset[1]

                # Draw line to parent
                if parent in self.data["pivots"]:
                    pwx, pwy, _, _ = self.data["pivots"][parent]
                    px = pwx * self.sheet_scale + self.assembly_offset[0]
                    py = pwy * self.sheet_scale + self.assembly_offset[1]
                    pygame.draw.line(screen, (0, 255, 255), (jx, jy), (px, py), 2)

                # Draw joint
                pygame.draw.circle(screen, (255, 0, 0), (int(jx), int(jy)), 5)
                pygame.draw.circle(screen, (255, 255, 255), (int(jx), int(jy)), 3)

        # Sheet view
        sheet_x = split_x + 20
        sheet_y = 110
        label2 = self.small_font.render("Parts (Click to select)", True, (200, 200, 200))
        screen.blit(label2, (sheet_x, 85))

        if self.sheet_surface:
            screen.blit(self.sheet_surface, (sheet_x, sheet_y))

            # Draw crop boxes and store rects for hit detection
            self.last_crop_rects = {}
            for name, (x1, y1, x2, y2) in self.data["crops"].items():
                bx = int(x1 * self.sheet_scale) + sheet_x
                by = int(y1 * self.sheet_scale) + sheet_y
                bw = int((x2 - x1 + 1) * self.sheet_scale)
                bh = int((y2 - y1 + 1) * self.sheet_scale)

                # Store rect for click detection
                self.last_crop_rects[name] = pygame.Rect(bx, by, bw, bh)

                # Color based on selection state
                if name == state["selected_part"]:
                    color = (255, 255, 0)
                elif name == state["selected_parent"]:
                    color = (0, 255, 0)
                else:
                    color = (100, 100, 100)

                pygame.draw.rect(screen, color, (bx, by, bw, bh), 1)

        # Status
        status_y = self.screen_height - 100
        if state["mode"] == "select_part":
            status_text = "Click a part in assembly to select it"
        elif state["mode"] == "select_parent":
            status_text = f"Selected: {state['selected_part']} | Click parent (or Set Root)"
        elif state["mode"] == "place_pivot":
            status_text = f"{state['selected_part']} -> {state['selected_parent']} | Click to place joint"
        else:
            status_text = ""

        status = self.small_font.render(status_text, True, (150, 255, 150))
        screen.blit(status, (20, status_y))

        root_text = self.small_font.render(f"Root: {self.data['skeleton'].get('root', 'none')}", True, (255, 200, 100))
        screen.blit(root_text, (20, status_y + 25))

        # Buttons
        for btn in self.buttons.values():
            btn.draw(screen, self.font)

    def handle_event(self, event):
        """Handle pygame events."""
        # Mouse wheel scrolling for draw order list
        if event.type == pygame.MOUSEWHEEL and self.mode == EditorMode.PLACEMENT:
            if self.data and "draw_order" in self.data:
                # Scroll the draw order list
                self.draw_order_scroll -= event.y * 16  # Scroll by 16 pixels per wheel tick

                # Clamp scroll to valid range
                total_height = len(self.data["draw_order"]) * 16  # Match line height
                list_max_height = self.screen_height - (self.sheet_surface.get_height() + 150) - 150
                max_scroll = max(0, total_height - list_max_height)
                self.draw_order_scroll = max(0, min(self.draw_order_scroll, max_scroll))

        # Button events
        for key, btn in self.buttons.items():
            if btn.handle_event(event):
                if key == "calc_crops":
                    self.calculate_crops()
                elif key == "recalc_crops":
                    self.calculate_crops()
                elif key == "apply_placement":
                    self.apply_placement()
                elif key == "save_skeleton":
                    self.save_current()
                elif key == "set_root":
                    self.set_root()
                elif key == "rename_part":
                    self.rename_part()
                elif key == "back_to_placement":
                    self.mode = EditorMode.PLACEMENT
                    self.setup_placement_mode()
                return

        # Keyboard
        if event.type == pygame.KEYDOWN:
            # Handle rename text input
            if self.mode == EditorMode.PLACEMENT and self.renaming_active:
                if event.key == pygame.K_RETURN:
                    self.finish_rename()
                elif event.key == pygame.K_ESCAPE:
                    self.renaming_active = False
                    self.rename_text = ""
                elif event.key == pygame.K_BACKSPACE:
                    self.rename_text = self.rename_text[:-1]
                elif event.unicode.isprintable() and len(self.rename_text) < 30:
                    self.rename_text += event.unicode
                return

            # Normal keyboard handling
            if event.key == pygame.K_RIGHT:
                self.next_actor()
            elif event.key == pygame.K_LEFT:
                self.prev_actor()
            elif event.key == pygame.K_s:
                self.save_current()

        # Mouse
        if self.mode == EditorMode.PLACEMENT:
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                self.handle_placement_click(*event.pos, event.button)
            elif event.type == pygame.MOUSEMOTION:
                self.handle_placement_drag(*event.pos)
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                self.handle_placement_release()

        elif self.mode == EditorMode.SKELETON:
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                self.handle_skeleton_click(*event.pos)


def main():
    pygame.init()

    # Create large window
    flags = pygame.SCALED | pygame.RESIZABLE
    screen = pygame.display.set_mode((1600, 1100) , flags)
    pygame.display.set_caption("Actor Editor")
    clock = pygame.time.Clock()

    editor = ActorEditor(1600, 1100)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

            editor.handle_event(event)

        editor.draw(screen)
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
