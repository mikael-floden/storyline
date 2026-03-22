#!/usr/bin/env python3
"""
Stone Golem  –  Pygame part-based sprite animation
=======================================================
Controls:  A/D or ←/→  move  |  SHIFT  run  |  SPACE  jump  |  ESC  quit

Requirements:  pip install pygame pillow scipy pyyaml
Put the sprite sheet (renamed golem_sheet.png) next to this script.
"""
import sys, math, io, os
import pygame
import yaml

ACTOR_ROOT = "actor"
PART_SCALE = 1      # each original pixel becomes PART_SCALE×PART_SCALE screen pixels
FPS        = 60
DISPLAY_MAX_W = 360
DISPLAY_MAX_H = 440

# ── Load driver configuration ─────────────────────────────────────────────────
def compute_rig_metrics(crops, pivots):
    """Derive rest-pose bounds and a safe display scale from driver data."""
    min_x = min_y = 10**9
    max_x = max_y = -10**9
    max_part_dim = 0

    for name, (x1, y1, x2, y2) in crops.items():
        w, h = x2 - x1 + 1, y2 - y1 + 1
        wx, wy, lx, ly = pivots[name]
        px, py = wx - lx, wy - ly
        min_x = min(min_x, px)
        min_y = min(min_y, py)
        max_x = max(max_x, px + w - 1)
        max_y = max(max_y, py + h - 1)
        max_part_dim = max(max_part_dim, w, h)

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
    """Load crops, pivots, and draw_order from driver.yaml next to sheet.png"""
    driver_path = os.path.join(os.path.dirname(sheet_path), "driver.yaml")
    with open(driver_path, 'r') as f:
        data = yaml.safe_load(f)

    # Convert lists to tuples for crops and pivots
    crops = {name: tuple(coords) for name, coords in data['crops'].items()}
    pivots = {name: tuple(coords) for name, coords in data['pivots'].items()}
    draw_order = data['draw_order']
    metrics = compute_rig_metrics(crops, pivots)

    return crops, pivots, draw_order, metrics


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


def load_model(model_spec):
    crops, pivots, draw_order, metrics = load_driver(model_spec["sheet_path"])
    sprites = load_sprites(model_spec["sheet_path"], crops)
    return {
        "name": model_spec["name"],
        "sheet_path": model_spec["sheet_path"],
        "pivots": pivots,
        "metrics": metrics,
        "draw_order": draw_order,
        "sprites": sprites,
    }

# ── 132×202 canvas metrics ───────────────────────────────────────────────────
# ── Background masking with antialiasing ─────────────────────────────────────
def remove_black_bg(pil_img):
    import numpy as np
    from scipy import ndimage
    from PIL import Image as PILImage, ImageFilter

    a = np.array(pil_img.convert("RGBA"))

    # Detect dark pixels with a slightly higher threshold for softer edge detection
    dark = (a[:,:,0] < 20) & (a[:,:,1] < 20) & (a[:,:,2] < 20)

    # Label connected regions
    lab, _ = ndimage.label(dark)
    h, w = dark.shape

    # Find border-connected regions
    border = set(lab[0,:].tolist() + lab[-1,:].tolist() +
                 lab[:,0].tolist() + lab[:,-1].tolist()) - {0}

    # Create a mask for background pixels
    bg_mask = np.isin(lab, list(border))

    # Create distance transform for antialiasing
    # Distance from foreground to background
    distance = ndimage.distance_transform_edt(~bg_mask)

    # Create smooth alpha transition (2-pixel feather for antialiasing)
    feather_distance = 2.0
    alpha_mult = np.clip(distance / feather_distance, 0, 1)

    # Apply smooth alpha
    a[:, :, 3] = (a[:, :, 3] * alpha_mult).astype(np.uint8)

    return PILImage.fromarray(a, "RGBA")

# ── Load + scale sprites ──────────────────────────────────────────────────────
def load_sprites(path, crops):
    from PIL import Image as PILImage
    try:
        raw = PILImage.open(path)
    except FileNotFoundError:
        print(f"\nERROR: '{path}' not found.")
        print("Put the golem sprite sheet (named golem_sheet.png) next to this script.\n")
        pygame.quit(); sys.exit(1)
    masked = remove_black_bg(raw)
    buf = io.BytesIO(); masked.save(buf, "PNG"); buf.seek(0)
    sheet = pygame.image.load(buf).convert_alpha()
    out = {}
    for name, (x1,y1,x2,y2) in crops.items():
        w, h = x2-x1+1, y2-y1+1
        s = pygame.Surface((w,h), pygame.SRCALPHA)
        s.blit(sheet, (0,0), (x1,y1,w,h))
        out[name] = pygame.transform.scale(s, (w*PART_SCALE, h*PART_SCALE))
    return out

# ── Rotation blit ─────────────────────────────────────────────────────────────
def blit_part(surf, spr, angle_deg, wx, wy, lpx, lpy):
    """
    Draw spr on surf rotated CW by angle_deg so that local point
    (lpx, lpy) in original-pixel coords lands at world point (wx, wy)
    in original-pixel coords.  PART_SCALE is applied internally.
    """
    S   = PART_SCALE
    rad = math.radians(angle_deg)
    c, s = math.cos(rad), math.sin(rad)
    # offset from sprite centre to local pivot (in screen pixels)
    dx = lpx*S - spr.get_width()/2
    dy = lpy*S - spr.get_height()/2
    # rotate that offset
    rdx = dx*c + dy*s
    rdy = -dx*s + dy*c
    rot  = pygame.transform.rotate(spr, -angle_deg)
    rect = rot.get_rect(center=(round(wx*S - rdx),
                                round(wy*S - rdy)))
    surf.blit(rot, rect)

# ── Eye-glow helper ───────────────────────────────────────────────────────────
# ── pygame init ───────────────────────────────────────────────────────────────
pygame.init()
W, H    = 960, 580
screen  = pygame.display.set_mode((W, H))
pygame.display.set_caption("Sprite Rig Viewer")
clock   = pygame.time.Clock()
FLOOR_Y = H - 90


# ── Background ────────────────────────────────────────────────────────────────
def build_bg():
    bg = pygame.Surface((W, H))
    bg.fill((12, 10, 8))
    BW, BH = 64, 32
    for gy in range(0, FLOOR_Y, BH):
        for col in range(0, W, BW):
            row = gy // BH
            bx  = col - (BW//2 if row%2 else 0)
            tone = (22,18,14) if (row+col//BW)%2==0 else (17,14,11)
            pygame.draw.rect(bg, tone,    (bx, gy, BW-1, BH-1))
            pygame.draw.rect(bg, (8,6,4), (bx, gy, BW-1, BH-1), 1)
    for gx in range(0, W, 80):
        pygame.draw.rect(bg, (36,30,24), (gx, FLOOR_Y,    79, H-FLOOR_Y))
        pygame.draw.rect(bg, (8, 6, 4),  (gx, FLOOR_Y,    79, H-FLOOR_Y), 1)
    pygame.draw.line(bg, (55,45,35), (0, FLOOR_Y-1), (W, FLOOR_Y-1), 2)
    return bg

bg_surf = build_bg()

# ── Golem ─────────────────────────────────────────────────────────────────────
class Golem:
    WALK_SPD = 2.5
    RUN_SPD  = 5.5

    def __init__(self, x, ground_y, mode="stone", pivots=None, metrics=None, draw_order=None):
        self.x        = float(x)
        self.ground_y = float(ground_y)
        self.t        = 0.0
        self.state    = "idle"
        self.facing   = 1           # +1=right -1=left
        self.air_vel  = 0.0
        self.air_off  = 0.0
        self.on_gnd   = True
        self.squash   = 0.0
        self.mode     = mode        # "stone" or "fire"
        self.pivots   = pivots      # pivot data for current mode
        self.metrics  = metrics     # derived canvas metrics for current mode
        self.draw_order = draw_order or []

    def apply_model(self, model):
        self.mode = model["name"]
        self.pivots = model["pivots"]
        self.metrics = model["metrics"]
        self.draw_order = model["draw_order"]

    def jump(self):
        if self.on_gnd:
            self.air_vel = -20.0
            self.on_gnd  = False

    def move(self, dx):
        self.x = max(160.0, min(float(W-160), self.x + dx))
        if dx: self.facing = 1 if dx > 0 else -1

    def update(self, moving, running):
        self.t += 1.0
        if not self.on_gnd:
            self.air_vel += 1.0
            self.air_off += self.air_vel
            if self.air_off >= 0:
                self.air_off = 0.0
                self.air_vel = 0.0
                self.on_gnd  = True
                self.squash  = 1.0
        if self.squash > 0:
            self.squash = max(0.0, self.squash - 0.07)
        if   not self.on_gnd:          self.state = "jump"
        elif moving and running:        self.state = "run"
        elif moving:                    self.state = "walk"
        else:                           self.state = "idle"

    # ── Pose: returns per-part angle offsets ─────────────────────────────────
    def pose(self):
        t  = self.t
        sq = self.squash

        if self.state == "idle":
            f = 0.030
            bob    =  math.sin(t*f*1.8)*4
            jaw    =  max(0, math.sin(t*f*1.5))*6
            head   =  math.sin(t*f*0.9)*3
            la_up  =  math.sin(t*f)*10 + 8
            la_lo  =  math.sin(t*f*1.1)*5
            hand   =  math.sin(t*f*0.9)*4
            ra_up  = -math.sin(t*f)*10 - 8
            ra_lo  = -math.sin(t*f*1.1)*5
            ll_th  =  math.sin(t*f*0.8)*4
            ll_sh  =  math.sin(t*f*0.8)*2
            rl_up  = -math.sin(t*f*0.8)*4
            rl_lo  =  0.0
            lean   =  0.0

        elif self.state == "walk":
            p   = t * 0.08
            bob = -abs(math.sin(p))*7
            jaw =  abs(math.sin(p*0.5))*9
            head =  math.sin(p*0.5)*5
            la_up  =  math.sin(p)*35 + 10
            la_lo  =  math.sin(p)*18
            hand   =  math.sin(p)*12
            ra_up  = -math.sin(p)*35 - 10
            ra_lo  = -math.sin(p)*18
            ll_th  = -math.sin(p)*32
            ll_sh  = -math.sin(p)*16
            rl_up  =  math.sin(p)*32
            rl_lo  =  math.sin(p)*14
            lean   =  8.0 * self.facing

        elif self.state == "run":
            p   = t * 0.15
            bob = -abs(math.sin(p))*12
            jaw =  abs(math.sin(p*0.5))*18
            head =  math.sin(p*0.5)*8
            la_up  =  math.sin(p)*55 + 15
            la_lo  =  math.sin(p)*28
            hand   =  math.sin(p)*20
            ra_up  = -math.sin(p)*55 - 15
            ra_lo  = -math.sin(p)*28
            ll_th  = -math.sin(p)*50
            ll_sh  = -math.sin(p)*25
            rl_up  =  math.sin(p)*50
            rl_lo  =  math.sin(p)*22
            lean   =  18.0 * self.facing

        else:  # jump
            av = self.air_vel
            bob    = 0.0
            jaw    = 22.0
            head   = -5.0 if av < 0 else 5.0
            lift   = max(0, -av) * 1.5    # arms rise on ascent
            la_up  = -55.0 - lift
            la_lo  = -22.0
            hand   = -15.0
            ra_up  =  55.0 + lift
            ra_lo  =  22.0
            ll_th  = -28.0
            ll_sh  = -14.0
            rl_up  =  28.0
            rl_lo  =  14.0
            lean   =  0.0

        bob -= sq * 16.0   # squash on landing compresses bob

        return dict(bob=bob, jaw=jaw, head=head,
                    la_up=la_up, la_lo=la_lo, hand=hand,
                    ra_up=ra_up, ra_lo=ra_lo,
                    ll_th=ll_th, ll_sh=ll_sh,
                    rl_up=rl_up, rl_lo=rl_lo,
                    lean=lean)

    # ── Draw ──────────────────────────────────────────────────────────────────
    def draw(self, surf, sprite_set):
        S   = PART_SCALE
        p   = self.pose()
        pivots = self.pivots
        sq  = self.squash
        ay  = int(self.air_off)

        # Squash/stretch scale applied to each sprite
        ssx = 1.0 + sq*0.22
        ssy = 1.0 - sq*0.18

        def sc(name):
            spr = sprite_set[name]
            if sq < 0.01:
                return spr
            return pygame.transform.scale(
                spr, (max(1, int(spr.get_width()*ssx)),
                      max(1, int(spr.get_height()*ssy))))

        # ── Offscreen canvas (golem-local space) ──────────────────────────
        # Canvas is CANVAS_W×CANVAS_H original pixels, scaled up.
        # We add padding to accommodate parts that swing outside the rest-pose box.
        metrics = self.metrics
        PAD   = metrics["pad"]
        off_x = metrics["offset_x"]
        off_y = metrics["offset_y"]
        GW    = max(1, int(math.ceil(metrics["full_w"] * S)))
        GH    = max(1, int(math.ceil(metrics["full_h"] * S)))
        gsurf = pygame.Surface((GW, GH), pygame.SRCALPHA)

        # Helper: blit part at its rest-pose world pivot + optional angle.
        # PAD shifts the whole coordinate system right/down.
        def gp(name, angle=0.0):
            wx, wy, lx, ly = pivots[name]
            blit_part(gsurf, sc(name), angle,
                      wx + off_x, wy + off_y, lx, ly)

        lean = p["lean"]
        part_angles = {
            "shin_R": p["rl_lo"] * 0.4,
            "leg_R_lo": p["rl_lo"],
            "leg_R_up": p["rl_up"],
            "arm_R_lo": p["ra_lo"] * 0.5,
            "shldr_R": p["ra_up"] * 0.3 + lean * 0.5,
            "arm_R_up": p["ra_up"],
            "thigh_L": p["ll_th"],
            "shin_L": p["ll_sh"],
            "torso_lo": lean * 0.3,
            "waist": lean * 0.3,
            "torso": lean * 0.5,
            "leg_L": p["ll_th"] * 0.6,
            "arm_L_up": p["la_up"],
            "arm_L_lo": p["la_lo"],
            "hand_L": p["hand"],
            "jaw": p["head"] + lean * 0.4 + p["jaw"] * 0.25,
            "head": p["head"] + lean * 0.4,
        }

        # ── Determine which arm/leg is "front" based on facing ────────────
        # The sprite sheet was drawn facing right.
        # When facing right: left-arm (screen-left) = front arm with claw.
        # When facing left (flipped): right becomes front.
        # We just draw and flip the whole canvas at the end.

        for name in self.draw_order:
            gp(name, part_angles.get(name, 0.0))

        # ── Optional Eyes/Glow ────────────────────────────────────────────
        # If in fire mode, we can add extra eye-glow intensity
        # ── Flip for left-facing ───────────────────────────────────────────
        if self.facing == -1:
            gsurf = pygame.transform.flip(gsurf, True, False)

        render_scale = metrics["render_scale"]
        if render_scale != 1.0:
            scaled_size = (
                max(1, int(round(gsurf.get_width() * render_scale))),
                max(1, int(round(gsurf.get_height() * render_scale))),
            )
            gsurf = pygame.transform.smoothscale(gsurf, scaled_size)

        # ── Blit composite to screen ──────────────────────────────────────
        # Anchor: golem feet = CANVAS_FOOT_Y+PAD in gsurf → FLOOR_Y+ay on screen
        # Horizontal: CANVAS_CX+PAD in gsurf → self.x on screen
        anchor_x = (metrics["canvas_cx"] + off_x) * S * render_scale
        anchor_y = (metrics["foot_y"] + off_y) * S * render_scale
        bob_y = (ay + int(p["bob"])) * render_scale
        blit_x = int(round(self.x - anchor_x))
        blit_y = int(round(self.ground_y + bob_y - anchor_y))
        surf.blit(gsurf, (blit_x, blit_y))

        # Ground shadow
        shw_w = max(80, int(160 * render_scale))
        shw_h = max(12, int(22 * render_scale))
        shw = pygame.Surface((shw_w, shw_h), pygame.SRCALPHA)
        pygame.draw.ellipse(shw, (0,0,0,55), (0,0,shw_w,shw_h))
        surf.blit(shw, (int(self.x) - shw_w//2, int(self.ground_y) - shw_h//3))

# ── HUD ───────────────────────────────────────────────────────────────────────
try:    hf = pygame.font.SysFont("monospace", 17)
except: hf = pygame.font.Font(None, 19)

def draw_hud(surf, state, mode, has_glow):
    accent = (255,140,40) if has_glow else (110,190,120)
    title_color = (80,40,20) if has_glow else (48,42,36)
    items = [("A/D","Move"), ("SHIFT","Run"), ("SPACE","Jump"), ("C","Next Model"), ("ESC","Quit"),
             ("", f"[ {mode.upper()} {state.upper()} ]")]
    x = y = 14
    for k, v in items:
        if k:
            ks = hf.render(k, True, (90,170,100))
            surf.blit(ks, (x, y))
            surf.blit(hf.render(" "+v, True, (70,65,58)), (x+ks.get_width(), y))
        else:
            surf.blit(hf.render(v, True, accent), (x, y))
        y += 22
    title_str = mode.replace("_", " ").upper()
    title = hf.render(f"{title_str}  –  Verified Part-Based Sprite Rig", True, title_color)
    surf.blit(title, (W//2 - title.get_width()//2, 12))

def main():
    # ── Load driver configurations ────────────────────────────────────────────────
    model_specs = discover_models(ACTOR_ROOT)
    if not model_specs:
        print("\nERROR: no actor models with both driver.yaml and a sheet image were found.\n")
        pygame.quit(); sys.exit(1)

    models = [load_model(spec) for spec in model_specs]
    current_index = next((i for i, model in enumerate(models) if model["name"] == "stone"), 0)
    if current_index:
        models = models[current_index:] + models[:current_index]
        current_index = 0

    # ── Load both sprite sets ─────────────────────────────────────────────────────
    current_model = models[current_index]

    # ── Main loop ─────────────────────────────────────────────────────────────────
    golem = Golem(W//2, FLOOR_Y, mode=current_model["name"], pivots=current_model["pivots"],
                  metrics=current_model["metrics"],
                  draw_order=current_model["draw_order"])

    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE: pygame.quit(); sys.exit()
                if ev.key == pygame.K_SPACE:  golem.jump()
                if ev.key == pygame.K_c:
                    current_index = (current_index + 1) % len(models)
                    current_model = models[current_index]
                    golem.apply_model(current_model)

        keys = pygame.key.get_pressed()
        run  = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]
        dx   = 0
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:  dx -= Golem.RUN_SPD if run else Golem.WALK_SPD
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]: dx += Golem.RUN_SPD if run else Golem.WALK_SPD

        golem.move(dx)
        golem.update(dx != 0, run)

        screen.blit(bg_surf, (0,0))
        golem.draw(screen, current_model["sprites"])
        draw_hud(screen, golem.state, golem.mode, "fire" in golem.mode)
        pygame.display.flip()
        clock.tick(FPS)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
