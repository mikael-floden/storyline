import argparse
import io
import math
from pathlib import Path

import numpy as np
import pygame
import yaml
from PIL import Image, ImageChops
from scipy import ndimage


ROOT = Path(__file__).resolve().parents[1]
STONE_DIR = ROOT / "actor" / "stone"
SHEET_PATH = STONE_DIR / "sheet.png"
REF_PATH = STONE_DIR / "ref_grid.png"


def remove_black_bg(pil_img: Image.Image) -> Image.Image:
    arr = np.array(pil_img.convert("RGBA"))
    dark = (arr[:, :, 0] < 20) & (arr[:, :, 1] < 20) & (arr[:, :, 2] < 20)
    labels, _ = ndimage.label(dark)
    border = set(labels[0, :].tolist() + labels[-1, :].tolist() + labels[:, 0].tolist() + labels[:, -1].tolist()) - {0}
    bg_mask = np.isin(labels, list(border))
    distance = ndimage.distance_transform_edt(~bg_mask)
    alpha_mult = np.clip(distance / 2.0, 0, 1)
    arr[:, :, 3] = (arr[:, :, 3] * alpha_mult).astype(np.uint8)
    return Image.fromarray(arr, "RGBA")


def load_driver(driver_path: Path):
    with driver_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    crops = {name: tuple(coords) for name, coords in data["crops"].items()}
    pivots = {name: tuple(coords) for name, coords in data["pivots"].items()}
    draw_order = list(data["draw_order"])
    return crops, pivots, draw_order


def load_sprites(crops):
    masked = remove_black_bg(Image.open(SHEET_PATH))
    out = {}
    for name, (x1, y1, x2, y2) in crops.items():
        out[name] = masked.crop((x1, y1, x2 + 1, y2 + 1))
    return out


def compute_bounds(crops, pivots):
    min_x = min_y = 10**9
    max_x = max_y = -10**9
    for name, (x1, y1, x2, y2) in crops.items():
        w, h = x2 - x1 + 1, y2 - y1 + 1
        wx, wy, lx, ly = pivots[name]
        px, py = wx - lx, wy - ly
        min_x = min(min_x, px)
        min_y = min(min_y, py)
        max_x = max(max_x, px + w - 1)
        max_y = max(max_y, py + h - 1)
    return min_x, min_y, max_x, max_y


def blit_part(canvas, sprite, angle_deg, wx, wy, lpx, lpy):
    rad = math.radians(angle_deg)
    c, s = math.cos(rad), math.sin(rad)
    dx = lpx - sprite.get_width() / 2
    dy = lpy - sprite.get_height() / 2
    rdx = dx * c + dy * s
    rdy = -dx * s + dy * c
    rot = pygame.transform.rotate(sprite, -angle_deg)
    rect = rot.get_rect(center=(round(wx - rdx), round(wy - rdy)))
    canvas.blit(rot, rect)


def pil_to_surface(img: Image.Image):
    buf = io.BytesIO()
    img.save(buf, "PNG")
    buf.seek(0)
    return pygame.image.load(buf).convert_alpha()


def crop_visible(img: Image.Image) -> Image.Image:
    bbox = img.getbbox()
    return img.crop(bbox) if bbox else img


def ref_crop() -> Image.Image:
    ref = Image.open(REF_PATH).convert("RGBA")
    arr = np.array(ref)
    grid_mask = (
        ((arr[:, :, 1] > 120) & (arr[:, :, 0] < 80) & (arr[:, :, 2] < 80)) |
        ((arr[:, :, 0] > 180) & (arr[:, :, 1] > 180) & (arr[:, :, 2] < 120))
    )
    non_black = (arr[:, :, 0] > 5) | (arr[:, :, 1] > 5) | (arr[:, :, 2] > 5)
    alpha = np.where(non_black & ~grid_mask, 255, 0).astype(np.uint8)
    rgba = arr.copy()
    rgba[:, :, 3] = alpha
    return crop_visible(Image.fromarray(rgba, "RGBA"))


def render(driver_path: Path):
    crops, pivots, draw_order = load_driver(driver_path)
    sprites = load_sprites(crops)
    pygame.init()
    pygame.display.set_mode((1, 1))

    min_x, min_y, max_x, max_y = compute_bounds(crops, pivots)
    pad = 40
    width = max_x - min_x + 1 + pad * 2
    height = max_y - min_y + 1 + pad * 2
    off_x = pad - min_x
    off_y = pad - min_y

    canvas = pygame.Surface((width, height), pygame.SRCALPHA)
    for name in draw_order:
        wx, wy, lx, ly = pivots[name]
        blit_part(canvas, pil_to_surface(sprites[name]), 0.0, wx + off_x, wy + off_y, lx, ly)

    data = pygame.image.tobytes(canvas, "RGBA")
    out = Image.frombytes("RGBA", canvas.get_size(), data)
    pygame.quit()
    return crop_visible(out)


def compare(render_img: Image.Image, ref_img: Image.Image):
    target_h = max(render_img.height, ref_img.height)
    if render_img.height != target_h:
        scale = target_h / render_img.height
        render_img = render_img.resize((round(render_img.width * scale), target_h), Image.Resampling.LANCZOS)
    if ref_img.height != target_h:
        scale = target_h / ref_img.height
        ref_img = ref_img.resize((round(ref_img.width * scale), target_h), Image.Resampling.LANCZOS)

    diff = ImageChops.difference(ref_img, render_img)
    score = float(np.array(diff).mean())
    side = Image.new("RGBA", (ref_img.width + render_img.width, target_h), (0, 0, 0, 255))
    side.paste(ref_img, (0, 0), ref_img)
    side.paste(render_img, (ref_img.width, 0), render_img)
    return side, diff, score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--driver", default=str(STONE_DIR / "driver.yaml"))
    parser.add_argument("--prefix", default="_iter")
    args = parser.parse_args()

    driver_path = Path(args.driver)
    render_img = render(driver_path)
    ref_img = ref_crop()
    side, diff, score = compare(render_img, ref_img)
    render_img.save(STONE_DIR / f"{args.prefix}_render.png")
    side.save(STONE_DIR / f"{args.prefix}_compare.png")
    diff.save(STONE_DIR / f"{args.prefix}_diff.png")
    print(f"score={score:.4f}")


if __name__ == "__main__":
    main()
