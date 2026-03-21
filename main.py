#!/usr/bin/env python3
"""
Stone Golem  –  Pygame part-based sprite animation
=======================================================
Controls:  A/D or ←/→  move  |  SHIFT  run  |  SPACE  jump  |  ESC  quit

Requirements:  pip install pygame pillow scipy
Put the sprite sheet (renamed golem_sheet.png) next to this script.
"""
import sys, math, io
import pygame

SHEET_PATH = "actor/golem/golem_sheet.png"
PART_SCALE = 1      # each original pixel becomes PART_SCALE×PART_SCALE screen pixels
FPS        = 60

# ── Sprite-sheet crop boxes  (x1, y1, x2, y2)  ──────────────────────────────
# Identified by flood-fill blob detection + pixel-accurate template matching
# against the 600×256 original sheet.
CROPS = {
    "head":     (276,144,348,211),   # 73×68  skull
    "jaw":      (276,213,346,250),   # 71×38  lower jaw / teeth
    "arm_L_up": (256, 16,304, 77),   # 49×62  left upper arm  (has orange ring)
    "arm_L_lo": (205, 80,260,137),   # 56×58  left lower arm / ball-fist
    "hand_L":   (195,  6,253, 64),   # 59×59  left open claw
    "arm_R_up": (202,144,253,197),   # 52×54  right upper arm (has orange ring)
    "arm_R_lo": (316, 85,352,132),   # 37×48  right forearm stub
    "shldr_R":  (445,166,488,198),   # 44×33  right shoulder plate (orange ring)
    "torso":    (505, 88,564,139),   # 60×52  main torso
    "torso_lo": (506, 21,567, 69),   # 62×49  lower torso with energy orb
    "waist":    (508,159,558,184),   # 51×26  waist connector
    "thigh_L":  (317, 18,348, 65),   # 32×48  left thigh / hip cylinder
    "shin_L":   (437, 24,482, 63),   # 46×40  left shin
    "leg_L":    (374,164,429,197),   # 56×34  left foot/ankle (orange ring)
    "leg_R_up": (378,110,420,154),   # 43×45  right upper leg
    "leg_R_lo": (208,209,244,250),   # 37×42  right lower leg (orange ring)
    "shin_R":   (458,111,484,149),   # 27×39  right shin
}

# ── Pivot table  (world_x, world_y, local_x, local_y)  ─────────────────────
# Derived from pixel-accurate template matching:
#   world_pivot = template_top_left + local_pivot
# Verified: at angle=0, blit places the sprite exactly at template position.
#
# Natural pivot choices:
#   • orange connector ring  → actual mechanical joint
#   • top-centre             → segment hangs downward
#   • bottom-centre          → segment pivots upward (e.g. head neck)
#   • centre                 → free-floating part
PIVOTS = {
    # world   local
    "head":     ( 87, 68, 36, 67),   # neck at head bottom-centre
    "jaw":      ( 92, 50, 35,  0),   # jaw top-centre
    "arm_L_up": ( 24, 78, 24, 21),   # left shoulder orange ring
    "arm_L_lo": ( 40, 95, 28,  0),   # left elbow top-centre
    "hand_L":   ( 36, 50, 30, 30),   # left wrist centre
    "arm_R_up": ( 53,117, 18,  6),   # right shoulder orange ring
    "arm_R_lo": ( 97, 98, 18,  0),   # right elbow top-centre
    "shldr_R":  ( 76, 98, 10,  4),   # right shoulder-plate orange ring
    "torso":    ( 70, 94, 30, 26),   # torso centre
    "torso_lo": ( 70, 84, 31, 25),   # lower torso centre
    "waist":    ( 63, 79, 25, 13),   # waist centre
    "thigh_L":  ( 76,154, 16,  0),   # left hip top-centre
    "shin_L":   ( 60,156, 23,  0),   # left knee top-centre
    "leg_L":    ( 42,173, 10,  5),   # left ankle orange ring
    "leg_R_up": ( 84,122, 21,  0),   # right hip top-centre
    "leg_R_lo": (109,125, 14,  4),   # right knee orange ring
    "shin_R":   ( 47, 65, 13,  0),   # right shin top-centre
}

# Draw order: back parts → torso → front parts
# Determined by visual depth analysis of the assembled sprite.
DRAW_ORDER = [
    # ── back ──────────────────────────────────────────────
    "leg_R_up", "leg_R_lo", "shin_R",   # right leg (behind)
    "arm_R_lo",                          # right forearm (behind torso)
    "shldr_R",  "arm_R_up",              # right shoulder + arm
    # ── torso cluster ─────────────────────────────────────
    "thigh_L",  "shin_L",                # left leg thigh+shin (between torso + foot)
    "torso_lo", "waist",    "torso",     # body core
    # ── front ─────────────────────────────────────────────
    "leg_L",                              # left foot (front)
    "arm_L_up", "arm_L_lo", "hand_L",    # left arm + claw (frontmost)
    "jaw",      "head",                  # head (topmost)
]

# ── 132×202 canvas metrics ───────────────────────────────────────────────────
CANVAS_W  = 132   # width  of assembled golem in original pixels
CANVAS_H  = 202   # height
CANVAS_CX = 66    # horizontal centre of canvas
CANVAS_FOOT_Y = 202   # feet sit at bottom of canvas

# ── Background masking (flood-fill from border) ──────────────────────────────
def remove_black_bg(pil_img):
    import numpy as np
    from scipy import ndimage
    from PIL import Image as PILImage
    a = np.array(pil_img.convert("RGBA"))
    dark = (a[:,:,0]<15) & (a[:,:,1]<15) & (a[:,:,2]<15)
    lab, _ = ndimage.label(dark)
    h, w = dark.shape
    border = set(lab[0,:].tolist()+lab[-1,:].tolist()+
                 lab[:,0].tolist()+lab[:,-1].tolist()) - {0}
    a[np.isin(lab, list(border)), 3] = 0
    return PILImage.fromarray(a, "RGBA")

# ── Load + scale sprites ──────────────────────────────────────────────────────
def load_sprites(path):
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
    for name, (x1,y1,x2,y2) in CROPS.items():
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
_gc: dict = {}
def draw_glow(surf, cx, cy, r, col, intensity):
    key = (r, col, round(intensity, 1))
    if key not in _gc:
        g = pygame.Surface((r*4, r*4), pygame.SRCALPHA)
        for i in range(r, 0, -1):
            a = int(200*intensity*(i/r)**0.6)
            pygame.draw.circle(g, (*col, a), (r*2, r*2), i)
        _gc[key] = g
    surf.blit(_gc[key], (int(cx)-r*2, int(cy)-r*2),
              special_flags=pygame.BLEND_ADD)

# ── pygame init ───────────────────────────────────────────────────────────────
pygame.init()
W, H    = 960, 580
screen  = pygame.display.set_mode((W, H))
pygame.display.set_caption("Stone Golem")
clock   = pygame.time.Clock()
FLOOR_Y = H - 90

sprites = load_sprites(SHEET_PATH)

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

    def __init__(self, x, ground_y):
        self.x        = float(x)
        self.ground_y = float(ground_y)
        self.t        = 0.0
        self.state    = "idle"
        self.facing   = 1           # +1=right -1=left
        self.air_vel  = 0.0
        self.air_off  = 0.0
        self.on_gnd   = True
        self.squash   = 0.0

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
            eye    = (math.sin(t*f*4)+1)/2*0.5 + 0.5

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
            eye    =  1.0

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
            eye    =  1.0

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
            eye    =  1.0

        bob -= sq * 16.0   # squash on landing compresses bob

        return dict(bob=bob, jaw=jaw, head=head,
                    la_up=la_up, la_lo=la_lo, hand=hand,
                    ra_up=ra_up, ra_lo=ra_lo,
                    ll_th=ll_th, ll_sh=ll_sh,
                    rl_up=rl_up, rl_lo=rl_lo,
                    lean=lean,   eye=eye)

    # ── Draw ──────────────────────────────────────────────────────────────────
    def draw(self, surf):
        S   = PART_SCALE
        p   = self.pose()
        sq  = self.squash
        ay  = int(self.air_off)

        # Squash/stretch scale applied to each sprite
        ssx = 1.0 + sq*0.22
        ssy = 1.0 - sq*0.18

        def sc(name):
            spr = sprites[name]
            if sq < 0.01:
                return spr
            return pygame.transform.scale(
                spr, (max(1, int(spr.get_width()*ssx)),
                      max(1, int(spr.get_height()*ssy))))

        # ── Offscreen canvas (golem-local space) ──────────────────────────
        # Canvas is CANVAS_W×CANVAS_H original pixels, scaled up.
        # We add padding to accommodate parts that swing outside the rest-pose box.
        PAD   = 40   # padding in original-pixel units
        GW    = (CANVAS_W  + PAD*2) * S
        GH    = (CANVAS_H  + PAD   ) * S
        gsurf = pygame.Surface((GW, GH), pygame.SRCALPHA)

        # Helper: blit part at its rest-pose world pivot + optional angle.
        # PAD shifts the whole coordinate system right/down.
        def gp(name, angle=0.0):
            wx, wy, lx, ly = PIVOTS[name]
            blit_part(gsurf, sc(name), angle,
                      wx + PAD, wy + PAD, lx, ly)

        lean = p["lean"]

        # ── Determine which arm/leg is "front" based on facing ────────────
        # The sprite sheet was drawn facing right.
        # When facing right: left-arm (screen-left) = front arm with claw.
        # When facing left (flipped): right becomes front.
        # We just draw and flip the whole canvas at the end.

        # Back right leg
        gp("shin_R",   p["rl_lo"]*0.4)
        gp("leg_R_lo", p["rl_lo"])
        gp("leg_R_up", p["rl_up"])

        # Back right arm
        gp("arm_R_lo", p["ra_lo"]*0.5)
        gp("shldr_R",  p["ra_up"]*0.3 + lean*0.5)
        gp("arm_R_up", p["ra_up"])

        # Torso cluster
        gp("thigh_L",  p["ll_th"])
        gp("shin_L",   p["ll_sh"])
        gp("torso_lo", lean*0.3)
        gp("waist",    lean*0.3)
        gp("torso",    lean*0.5)

        # Front left leg
        gp("leg_L",    p["ll_th"]*0.6)

        # Front left arm
        gp("arm_L_up", p["la_up"])
        gp("arm_L_lo", p["la_lo"])
        gp("hand_L",   p["hand"])

        # Head
        gp("jaw",  p["head"] + lean*0.4 + p["jaw"]*0.25)
        gp("head", p["head"] + lean*0.4)

        # Eyes are baked into the head sprite - no extra drawing needed

        # ── Flip for left-facing ───────────────────────────────────────────
        if self.facing == -1:
            gsurf = pygame.transform.flip(gsurf, True, False)

        # ── Blit composite to screen ──────────────────────────────────────
        # Anchor: golem feet = CANVAS_FOOT_Y+PAD in gsurf → FLOOR_Y+ay on screen
        # Horizontal: CANVAS_CX+PAD in gsurf → self.x on screen
        blit_x = int(self.x) - (CANVAS_CX + PAD)*S
        blit_y = int(self.ground_y) + ay + int(p["bob"]) - (CANVAS_FOOT_Y + PAD)*S
        surf.blit(gsurf, (blit_x, blit_y))

        # Ground shadow
        shw = pygame.Surface((160, 22), pygame.SRCALPHA)
        pygame.draw.ellipse(shw, (0,0,0,55), (0,0,160,22))
        surf.blit(shw, (int(self.x)-80, int(self.ground_y)-8))

# ── HUD ───────────────────────────────────────────────────────────────────────
try:    hf = pygame.font.SysFont("monospace", 17)
except: hf = pygame.font.Font(None, 19)

def draw_hud(surf, state):
    items = [("A/D","Move"), ("SHIFT","Run"), ("SPACE","Jump"), ("ESC","Quit"),
             ("", f"[ {state.upper()} ]")]
    x = y = 14
    for k, v in items:
        if k:
            ks = hf.render(k, True, (90,170,100))
            surf.blit(ks, (x, y))
            surf.blit(hf.render(" "+v, True, (70,65,58)), (x+ks.get_width(), y))
        else:
            surf.blit(hf.render(v, True, (110,190,120)), (x, y))
        y += 22
    title = hf.render("STONE GOLEM  –  Verified Part-Based Sprite Rig", True, (48,42,36))
    surf.blit(title, (W//2 - title.get_width()//2, 12))

def main():
    # ── Main loop ─────────────────────────────────────────────────────────────────
    golem = Golem(W//2, FLOOR_Y)

    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE: pygame.quit(); sys.exit()
                if ev.key == pygame.K_SPACE:  golem.jump()

        keys = pygame.key.get_pressed()
        run  = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]
        dx   = 0
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:  dx -= Golem.RUN_SPD if run else Golem.WALK_SPD
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]: dx += Golem.RUN_SPD if run else Golem.WALK_SPD

        golem.move(dx)
        golem.update(dx != 0, run)

        screen.blit(bg_surf, (0,0))
        golem.draw(screen)
        draw_hud(screen, golem.state)
        pygame.display.flip()
        clock.tick(FPS)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
