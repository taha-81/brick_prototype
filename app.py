"""Photo -> brick mosaic prototype: upload, size, palette match, preview, counts.
Day 5: minimum-colour optimisation (flat-graphic and photo profiles), strict 54-colour cap, skin tone adjustment."""
import base64, csv, io, math, os
import re
from typing import Any
import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image, ImageDraw, ImageOps

cv2: Any = None
_FACE: Any = None
FACE_WHY = ""
try:
    import cv2
    if not hasattr(cv2, "CascadeClassifier"):  # OpenCV 5 removed the face detector
        FACE_WHY = f"OpenCV {cv2.__version__} has no face detector. Run: py -m pip install \"opencv-python-headless<5\"."
    else:
        _FACE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        if _FACE.empty():
            _FACE, FACE_WHY = None, "The face detector file could not be loaded."
except ImportError:
    FACE_WHY = "OpenCV is not installed. Run: py -m pip install \"opencv-python-headless<5\"."
except Exception as e:  # anything else: face help is off, everything else still works
    _FACE, FACE_WHY = None, f"Face detection could not start ({type(e).__name__})."

APP_VERSION = "day5-skin-54-2"
app = Flask(__name__)
BASE = os.path.dirname(os.path.abspath(__file__))  # data files are found next to app.py, wherever the server starts
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024  # Vercel refuses bigger requests anyway (about 4.5 MB)

@app.errorhandler(413)
def too_big(_):
    return jsonify(error="The photo is too large. Try a smaller photo."), 413
DITHER = {"off": 0.0, "soft": 0.5, "normal": 0.8, "strong": 1.0}  # strength on the background
FACE_KEEP = 0.25  # on the face, dithering is cut to this share, so eyes and lips stay crisp
FS = ((0, 1, 7 / 16), (1, -1, 3 / 16), (1, 0, 5 / 16), (1, 1, 1 / 16))  # Floyd-Steinberg
STUD_CM, PLATE_STUDS = 0.8, 16  # client module: 16x16 studs (12.8 cm)
MAX_PLATE_UNITS = 12
DEFAULT_SIZE = "3x3"
MAX_COLOURS = 54  # strict client limit: NO mosaic ever uses more than this many of the 94 colours (a maximum, not a target)
# Colour-count modes. "stop": the greedy palette search stops when the next colour would cut the total error by less than this share.
# "merge": after conversion, a brick colour is dropped when losing it raises the mean blended ΔE by less than this.
COLOUR_MODES = {"min": dict(stop=0.010, merge=0.05), "balanced": dict(stop=0.004, merge=0.02), "max": dict(stop=0.0, merge=0.0)}
FLAT_MERGE = 6.0       # flat graphics: colour clusters closer than this (CIEDE2000) are the same colour (JPEG noise, anti-aliasing)
FLAT_DISTINCT = 8.0    # flat graphics: clusters further apart than this should end up as different bricks when the palette allows
SKIN_RANGE = dict(cr=(133, 173), cb=(77, 127))  # classic YCrCb skin window, used together with the face's own skin colour
SKIN_PALETTE_DELTA = 12.0  # keep the two skin bricks close in CIEDE2000

def load_sizes(path=os.path.join(BASE, "sizes.csv")):
    """Load the client's standard frame layouts, measured in 16x16-stud plate units."""
    sizes, groups = {}, {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            plates_w, plates_h = int(r["plates_w"]), int(r["plates_h"])
            assert 1 <= plates_w <= MAX_PLATE_UNITS and 1 <= plates_h <= MAX_PLATE_UNITS
            w, h = plates_w * PLATE_STUDS, plates_h * PLATE_STUDS
            key = f"{plates_w}x{plates_h}"
            sizes[key] = (w, h)
            groups.setdefault(r["orientation"], []).append(
                dict(key=key, label=f"{key} plate units, {w}x{h} studs, {w * STUD_CM:.1f}x{h * STUD_CM:.1f} cm",
                     plates_w=plates_w, plates_h=plates_h, plate_cells=[None] * (plates_w * plates_h),
                     cm_w=f"{w * STUD_CM:.1f}", cm_h=f"{h * STUD_CM:.1f}",
                     frame_w=f"{w * STUD_CM + 0.2:.1f}", frame_h=f"{h * STUD_CM + 0.2:.1f}"))
    return sizes, list(groups.items())

SIZES, SIZE_GROUPS = load_sizes()

def resolve_size(key):
    """Resolve a listed layout or a custom layout, both expressed in plate units."""
    if key in SIZES:
        return SIZES[key]
    match = re.fullmatch(r"custom:(\d{1,2})x(\d{1,2})", key or "")
    if not match:
        return None
    plates_w, plates_h = map(int, match.groups())
    if not (1 <= plates_w <= MAX_PLATE_UNITS and 1 <= plates_h <= MAX_PLATE_UNITS):
        return None
    return plates_w * PLATE_STUDS, plates_h * PLATE_STUDS

def load_palette(path=os.path.join(BASE, "palette_94.csv")):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    rgb = np.array([[int(r["hex"][i:i + 2], 16) for i in (1, 3, 5)] for r in rows], float)
    return rows, rgb

def srgb_to_lab(rgb):
    c = rgb / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    m = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = (c @ m.T) / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)

def lab_to_srgb(lab):
    """Inverse of srgb_to_lab, clipped to 0..255."""
    lab = np.asarray(lab, float)
    fy = (lab[..., 0] + 16) / 116
    fx, fz = fy + lab[..., 1] / 500, fy - lab[..., 2] / 200
    inv = lambda t: np.where(t ** 3 > 0.008856, t ** 3, (t - 16 / 116) / 7.787)
    xyz = np.stack([inv(fx), inv(fy), inv(fz)], -1) * np.array([0.95047, 1.0, 1.08883])
    m = np.array([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]])
    c = np.clip(xyz @ m.T, 0, 1)
    c = np.where(c > 0.0031308, 1.055 * c ** (1 / 2.4) - 0.055, 12.92 * c)
    return np.clip(c * 255, 0, 255)

def ciede2000(p, q) -> np.ndarray:
    """Strict colour difference (CIEDE2000). p: (N,1,3) Lab, q: (1,M,3) Lab -> (N,M)."""
    L1, a1, b1 = p[..., 0], p[..., 1], p[..., 2]
    L2, a2, b2 = q[..., 0], q[..., 1], q[..., 2]
    Cb = (np.hypot(a1, b1) + np.hypot(a2, b2)) / 2
    G = 0.5 * (1 - np.sqrt(Cb**7 / (Cb**7 + 25.0**7)))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1, C2 = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1, h2 = np.degrees(np.arctan2(b1, a1p)) % 360, np.degrees(np.arctan2(b2, a2p)) % 360
    dh = h2 - h1
    dh = np.where(dh > 180, dh - 360, np.where(dh < -180, dh + 360, dh))
    zero = C1 * C2 == 0
    dh = np.where(zero, 0, dh)
    dH = 2 * np.sqrt(C1 * C2) * np.sin(np.radians(dh) / 2)
    Lb, Cbp = (L1 + L2) / 2, (C1 + C2) / 2
    hs = h1 + h2
    hb = np.where(zero, hs, np.where(np.abs(h1 - h2) <= 180, hs / 2, np.where(hs < 360, (hs + 360) / 2, (hs - 360) / 2)))
    T = (1 - 0.17 * np.cos(np.radians(hb - 30)) + 0.24 * np.cos(np.radians(2 * hb))
         + 0.32 * np.cos(np.radians(3 * hb + 6)) - 0.20 * np.cos(np.radians(4 * hb - 63)))
    Rt = -np.sin(np.radians(2 * 30 * np.exp(-(((hb - 275) / 25) ** 2)))) * 2 * np.sqrt(Cbp**7 / (Cbp**7 + 25.0**7))
    Sl = 1 + 0.015 * (Lb - 50) ** 2 / np.sqrt(20 + (Lb - 50) ** 2)
    Sc, Sh = 1 + 0.045 * Cbp, 1 + 0.015 * Cbp * T
    x, y, z = (L2 - L1) / Sl, (C2 - C1) / Sc, dH / Sh
    return np.sqrt(np.maximum(x * x + y * y + z * z + Rt * y * z, 0))

PALETTE, PAL_RGB = load_palette()
PAL_LAB = srgb_to_lab(PAL_RGB)
COOL = (PAL_LAB[:, 1] < -3) | (PAL_LAB[:, 2] < -3)  # green or blue bricks: never allowed on skin

def center_crop(img, w, h):
    target = w / h
    iw, ih = img.size
    if iw / ih > target:
        nw = int(ih * target); x = (iw - nw) // 2; return img.crop((x, 0, x + nw, ih))
    nh = int(iw / target); y = (ih - nh) // 2
    return img.crop((0, y, iw, y + nh))

def detect_face(img):
    """Largest frontal face as (x, y, w, h) in pixels, or None."""
    if _FACE is None:
        return None
    g = np.asarray(img.convert("L"))
    sc = min(1.0, 800 / max(g.shape))
    if sc < 1:
        g = cv2.resize(g, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
    ms = max(30, g.shape[1] // 20)
    r = _FACE.detectMultiScale(g, 1.1, 5, minSize=(ms, ms))
    if len(r) == 0:
        return None
    x, y, fw, fh = max(r, key=lambda b: b[2] * b[3])
    return x / sc, y / sc, fw / sc, fh / sc

def detect_faces(img):
    """Every frontal face as a list of (x, y, w, h) in pixels, largest first; [] when none or no detector."""
    if _FACE is None:
        return []
    g = np.asarray(img.convert("L"))
    sc = min(1.0, 800 / max(g.shape))
    if sc < 1:
        g = cv2.resize(g, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
    ms = max(30, g.shape[1] // 20)
    r = _FACE.detectMultiScale(g, 1.1, 5, minSize=(ms, ms))
    return [tuple(v / sc for v in b) for b in sorted(r, key=lambda b: -b[2] * b[3])]

def face_mask(w, h, crop_size, face):
    """1 inside the face (plus a little hair and chin), fading to 0 outside, one value per brick cell."""
    if face is None:
        return np.zeros((h, w))
    sx, sy = w / crop_size[0], h / crop_size[1]
    cx, cy = (face[0] + face[2] / 2) * sx, (face[1] + face[3] / 2) * sy
    rx, ry = face[2] * sx * 0.7, face[3] * sy * 0.8
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.sqrt(((xx + .5 - cx) / rx) ** 2 + ((yy + .5 - cy) / ry) ** 2)
    return np.clip((1.3 - d) / 0.5, 0, 1)

def strength_map(m, base):
    """Dithering strength per cell: full on the background, much lower on and around the face."""
    return base * (1 - (1 - FACE_KEEP) * m)

def pick_palette(lab, m, k=MAX_COLOURS, stop=0.0):
    """Choose at most k palette colours that serve this photo best. Returns a True/False mask over the 94 colours.
    Greedy pick (add the colour that cuts the total error most), then swaps to polish. The face counts 4x, so skin tones win.
    Squared-ish error (d ** 1.5) so a few badly matched cells weigh more than many slightly off ones.
    stop: the search ends early once the next colour would cut the remaining error by less than this share (0 = always take k)."""
    n = len(PALETTE)
    k = min(k, MAX_COLOURS)  # the 54-colour cap is absolute
    if k >= n:
        return np.ones(n, bool)
    pts = lab.reshape(-1, 3)
    w = (1 + 3 * m).reshape(-1)
    step = 1.0 if len(pts) <= 20000 else 2.0  # merge near-identical cells: same result, much less work
    _, inv = np.unique(np.round(pts / step).astype(int), axis=0, return_inverse=True)
    inv = inv.ravel()
    wu = np.bincount(inv, weights=w)
    pu = np.stack([np.bincount(inv, weights=w * pts[:, j]) for j in range(3)], 1) / wu[:, None]
    U = len(pu)
    D = np.vstack([ciede2000(pu[i:i + 2048, None, :], PAL_LAB[None, :, :]) for i in range(0, U, 2048)])
    F = D ** 1.5  # (cells, 94) cost of showing each cell with each colour
    sel, cur = [], np.full(U, np.inf)
    for _ in range(k):  # greedy
        cost = (wu[:, None] * np.minimum(cur[:, None], F)).sum(0)
        cost[sel] = np.inf
        c = int(cost.argmin())
        if sel and stop > 0:
            now = (wu * cur).sum()
            if now - cost[c] < stop * now:  # the next colour would barely help: stop here, fewer colours win
                break
        sel.append(c); cur = np.minimum(cur, F[:, c])
    k = len(sel)
    rows = np.arange(U)
    for _ in range(2):  # polish: swap a chosen colour for an unused one whenever that lowers the total error
        moved = False
        for j in range(k):
            S = np.array(sel)
            order = np.argsort(F[:, S], axis=1)[:, :2]
            best, second = F[rows, S[order[:, 0]]], F[rows, S[order[:, 1]]]
            without = np.where(order[:, 0] == j, second, best)  # error if colour j were removed
            others = np.setdiff1d(np.arange(n), S)
            cost = (wu[:, None] * np.minimum(without[:, None], F[:, others])).sum(0)
            b = int(cost.argmin())
            if cost[b] < (wu * best).sum() - 1e-9:
                sel[j] = int(others[b]); moved = True
        if not moved:
            break
    mask = np.zeros(n, bool); mask[sel] = True
    return mask

def dither_match(lab, strength, skin=None, allowed=None, skin_palette=None):
    """Pick a brick per cell with error diffusion. Every pick is still one of the 94 palette colours."""
    h, w, _ = lab.shape
    work, idx = lab.copy(), np.zeros((h, w), int)
    for y in range(h):
        for x in range(w):
            t = work[y, x].copy(); t[0] = min(100.0, max(0.0, t[0]))
            d2 = ((PAL_LAB - t) ** 2).sum(1)
            if allowed is not None:
                d2 = np.where(allowed, d2, 1e9)  # colour limit: only the chosen colours may be used
            if skin is not None and skin[y, x]:
                d2 = np.where(COOL, 1e9, d2)  # skin protection: no green or blue bricks on skin
                if skin_palette is not None:
                    d2 = np.where(skin_palette, d2, 1e9)
            cand = np.argpartition(d2, 8)[:8]  # 8 closest, then judge strictly
            i = int(cand[ciede2000(t[None, None, :], PAL_LAB[cand][None, :, :])[0].argmin()])
            idx[y, x] = i
            err = t - PAL_LAB[i]; n = np.linalg.norm(err)
            if n > 20: err *= 20 / n  # stop streaks when the photo colour is outside the palette
            err *= strength[y, x]
            for dy, dx, wt in FS:
                yy, xx = y + dy, x + dx
                if 0 <= yy < h and 0 <= xx < w:
                    work[yy, xx] += err * wt
    return idx

def blur3(a) -> np.ndarray:
    h, w, _ = a.shape
    p = np.pad(a, ((1, 1), (1, 1), (0, 0)), mode="edge")
    acc = np.zeros(a.shape, float)
    for dy in range(3):
        for dx in range(3):
            acc += p[dy:dy + h, dx:dx + w]
    return acc / 9

def blurL(L):
    return blur3(np.stack([L] * 3, -1))[..., 0]

LUMA = np.array([0.2126, 0.7152, 0.0722])

def grade_small(rgb, g):
    """Grade the photo at brick resolution (this is the size that matters): levels, brightness, contrast, colour, sharpness.
    Auto mode only changes LIGHTNESS (the same gain on red, green and blue), so skin never gets redder or more colourful."""
    if g["mode"] == "off":
        return rgb
    x = rgb.copy()
    if g["mode"] == "auto":
        lum = x @ LUMA
        lo, hi = np.percentile(lum, 1), np.percentile(lum, 99)
        if hi - lo > 20:
            target = lum + 0.7 * ((lum - lo) * (255 / (hi - lo)) - lum)  # stretch the tones, 70% of the way
            x = x * np.clip(target / np.maximum(lum, 1), 0.6, 1.8)[..., None]  # same gain on R, G and B: hue and colour strength stay
        g = dict(g, br=0, ct=0, sat=0, sh=40)
    x = x + g["br"] * 1.28
    x = (x - 128) * (1 + g["ct"] / 100) + 128
    gray = (x @ LUMA)[..., None]
    x = gray + (x - gray) * (1 + g["sat"] / 100)
    lum = x @ LUMA  # sharpen lightness only, so edges get crisper without coloured fringes
    x = x + (g["sh"] / 100 * 1.5) * (lum - blurL(lum))[..., None]
    return np.clip(x, 0, 255)

def face_metrics(orig_lab, brick_lab, m):
    """How well the face survives: correlation of lightness (structure) and of fine detail (eyes, brows, lips)."""
    sel = m > 0.5
    if sel.sum() < 20:
        return None, None
    Lo, Lb = orig_lab[..., 0], brick_lab[..., 0]
    c = lambda a, b: round(float(np.nan_to_num(np.corrcoef(a[sel], b[sel])[0, 1])), 3)
    return c(Lo, Lb), c(Lo - blurL(Lo), Lb - blurL(Lb))


def kmeans_lab(pts, K, iters=25):
    """K-means in Lab (OpenCV when present, plain NumPy otherwise). pts: (N,3) float -> labels (N,), centres (K,3)."""
    pts = np.asarray(pts, np.float32)
    K = max(1, min(K, len(pts)))
    if cv2 is not None and hasattr(cv2, "kmeans"):
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, iters, 0.5)
        _, lbl, cen = cv2.kmeans(pts, K, None, crit, 2, cv2.KMEANS_PP_CENTERS)
        return lbl.ravel(), cen.astype(float)
    rng = np.random.default_rng(0)  # k-means++ seeding, then Lloyd iterations
    cen = [pts[rng.integers(len(pts))]]
    for _ in range(1, K):
        d = np.min(((pts[:, None, :] - np.array(cen)[None, :, :]) ** 2).sum(2), 1)
        cen.append(pts[rng.choice(len(pts), p=d / d.sum())] if d.sum() > 0 else pts[rng.integers(len(pts))])
    cen = np.array(cen, float)
    for _ in range(iters):
        lbl = ((pts[:, None, :] - cen[None, :, :]) ** 2).sum(2).argmin(1)
        new = np.array([pts[lbl == i].mean(0) if (lbl == i).any() else cen[i] for i in range(K)])
        if np.abs(new - cen).max() < 0.05:
            cen = new; break
        cen = new
    lbl = ((pts[:, None, :] - cen[None, :, :]) ** 2).sum(2).argmin(1)
    return lbl, cen

def nearest_centre(lab, centres):
    """Index of the closest centre (plain Lab distance) for every row of lab, in chunks."""
    out = np.empty(len(lab), int)
    for i in range(0, len(lab), 65536):
        out[i:i + 65536] = ((lab[i:i + 65536, None, :] - centres[None, :, :]) ** 2).sum(2).argmin(1)
    return out

def merge_centres(cen, within):
    """Union-find: centres closer than `within` (CIEDE2000) collapse into one. Returns the group of each centre."""
    K = len(cen)
    D = ciede2000(cen[:, None, :], cen[None, :, :])
    parent = list(range(K))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for i in range(K):
        for j in range(i + 1, K):
            if D[i, j] < within:
                parent[find(i)] = find(j)
    roots = [find(i) for i in range(K)]
    _, inv = np.unique(roots, return_inverse=True)
    return inv

def analyse_image(crop):
    """Is this a flat graphic (flag, logo, illustration) or a photograph? Looks at a small copy of the framed photo.
    flat_frac: share of pixels that sit right on one of a few colour centres. soft_edge: share of changing pixels that change
    gently (gradients, shading) rather than sharply (boundaries). Flat graphics: high flat_frac, low soft_edge, few big colours."""
    small = crop.copy(); small.thumbnail((256, 256), Image.Resampling.BOX)
    rgb = np.asarray(small, float)
    lab = srgb_to_lab(rgb.reshape(-1, 3))
    lbl, cen = kmeans_lab(lab, 16)
    grp = merge_centres(cen, FLAT_MERGE)
    cen2 = np.array([lab[grp[lbl] == g].mean(0) for g in range(grp.max() + 1)])
    near = nearest_centre(lab, cen2)
    dmin = np.sqrt(((lab - cen2[near]) ** 2).sum(1))
    area = np.bincount(near, minlength=len(cen2)) / len(lab)
    L = lab[:, 0].reshape(rgb.shape[:2])
    g = np.maximum(np.abs(np.diff(L, axis=1))[:-1, :], np.abs(np.diff(L, axis=0))[:, :-1])
    moving = g > 1.0
    soft = float(((g > 1.0) & (g < 8)).sum() / max(1, moving.sum()))
    flat_frac = float((dmin <= 4.0).mean())
    major = int((area >= 0.01).sum())
    flat = flat_frac >= 0.93 or (flat_frac >= 0.85 and soft <= 0.55 and major <= 12)  # thin lines on a flat flag soften the edge count
    return dict(profile="flat" if flat else "photo", flat_frac=round(flat_frac, 3), soft_edge=round(soft, 3),
                major_colours=major, clusters=int(len(cen2)))

def flat_cells(crop, w, h):
    """Flat-graphic profile. Group the pixels of the framed image into a few colour regions, then give every brick cell the
    colour of the region that covers MOST of it (majority vote). Anti-aliased edges, JPEG ringing and tiny tonal wobbles never
    become brick colours, and boundaries between regions stay sharp. Returns cell labels (h,w), region Lab centres (K,3), region areas."""
    s = max(1, min(8, crop.width // w, crop.height // h))  # a few pixels per cell are enough for a majority vote
    work = crop.resize((w * s, h * s), Image.Resampling.BOX if s > 1 or crop.width >= w else Image.Resampling.LANCZOS)
    lab = srgb_to_lab(np.asarray(work, float).reshape(-1, 3))
    rng = np.random.default_rng(0)
    sample = lab if len(lab) <= 60000 else lab[rng.choice(len(lab), 60000, replace=False)]
    _, cen = kmeans_lab(sample, 16)
    grp = merge_centres(cen, FLAT_MERGE)
    cen = np.array([cen[grp == g].mean(0) for g in range(grp.max() + 1)])
    near = nearest_centre(lab, cen)
    cen = np.array([lab[near == i].mean(0) if (near == i).any() else cen[i] for i in range(len(cen))])
    near = nearest_centre(lab, cen).reshape(h * s, w * s)
    area = np.bincount(near.ravel(), minlength=len(cen)) / near.size
    interior = np.ones(near.shape, bool)  # a pixel is "interior" when its 4 neighbours share its region
    interior[1:, :] &= near[1:, :] == near[:-1, :]; interior[:-1, :] &= near[:-1, :] == near[1:, :]
    interior[:, 1:] &= near[:, 1:] == near[:, :-1]; interior[:, :-1] &= near[:, :-1] == near[:, 1:]
    coherence = np.array([interior[near == i].mean() if (near == i).any() else 0.0 for i in range(len(cen))])
    keep = (area >= 0.02) | ((coherence >= 0.25) & (area >= 0.0005))  # real regions; edge-only "colours" are artifacts
    if not keep.any():
        keep[area.argmax()] = True
    kept = np.flatnonzero(keep)
    if len(kept) < len(cen):  # artifact pixels join the closest real region
        near = kept[nearest_centre(lab, cen[kept])].reshape(h * s, w * s)
    _, near = np.unique(near, return_inverse=True); near = near.reshape(h * s, w * s)
    cen = np.array([lab[near.ravel() == i].mean(0) for i in range(near.max() + 1)])
    K = len(cen)
    onehot = (near.reshape(h, s, w, s)[..., None] == np.arange(K)).sum((1, 3))  # (h, w, K) pixel votes per cell
    labels = onehot.argmax(2)
    area = np.bincount(labels.ravel(), minlength=K) / labels.size
    return labels, cen, area

def flat_bricks(cen, area, skin_ok=None):
    """One brick per flat-graphic region: the closest palette colour (CIEDE2000). Two clearly different regions are kept apart
    when a second brick is nearly as close; two near-identical regions are allowed to share a brick (fewer colours)."""
    D = ciede2000(cen[:, None, :], PAL_LAB[None, :, :])
    between = ciede2000(cen[:, None, :], cen[None, :, :])
    brick, owner = np.zeros(len(cen), int), {}
    for i in np.argsort(-area):
        order = np.argsort(D[i])
        best = int(order[0])
        if best in owner and between[i, owner[best]] > FLAT_DISTINCT:
            for alt in order[1:]:
                if int(alt) not in owner and D[i, alt] <= D[i, best] + 6.0:
                    best = int(alt); break
        brick[i] = best; owner.setdefault(best, i)
    return brick

def dilate1(mask):
    """3x3 dilation of a boolean (h,w) mask."""
    p = np.pad(mask, 1)
    out = np.zeros_like(mask)
    for dy in range(3):
        for dx in range(3):
            out |= p[dy:dy + mask.shape[0], dx:dx + mask.shape[1]]
    return out

def merge_pass(idx, lab, m, threshold, floor=1):
    """Final palette reduction. Repeatedly drop the used brick colour whose loss hurts least: its cells move to the closest
    remaining used colour, and the cost is the rise in mean blended ΔE (what the eye sees), face cells counting 4x.
    Stops when the cheapest drop costs more than `threshold`. Returns the new matrix and how many colours were dropped."""
    if threshold <= 0:
        return idx, 0
    h, w = idx.shape
    wgt = 1 + 3 * m
    target = blur3(lab)
    dropped = 0
    while True:
        used = np.unique(idx)
        if len(used) <= floor:
            break
        cur = blur3(PAL_LAB[idx])
        best_c, best_cost, best_new = None, np.inf, None
        for c in used:
            cells = idx == c
            others = used[used != c]
            d = ciede2000(lab[cells][:, None, :], PAL_LAB[others][None, :, :])
            new = idx.copy(); new[cells] = others[d.argmin(1)]
            aff = dilate1(cells)
            old_err = ciede2000(target[aff], cur[aff])
            new_err = ciede2000(target[aff], blur3(PAL_LAB[new])[aff])
            cost = float(((new_err - old_err) * wgt[aff]).sum() / wgt.sum())
            if cost < best_cost:
                best_c, best_cost, best_new = c, cost, new
        if best_cost > threshold:
            break
        idx = best_new; dropped += 1
    return idx, dropped

def ycrcb_skin(rgb):
    """Classic YCrCb skin window, per cell."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cr, cb = (r - y) * 0.713 + 128, (b - y) * 0.564 + 128
    return (cr >= SKIN_RANGE["cr"][0]) & (cr <= SKIN_RANGE["cr"][1]) & (cb >= SKIN_RANGE["cb"][0]) & (cb <= SKIN_RANGE["cb"][1])

def components(mask):
    """Connected components (4-neighbour) of a boolean (h,w) mask -> label map (0 = background), count."""
    if cv2 is not None and hasattr(cv2, "connectedComponents"):
        n, lab = cv2.connectedComponents(mask.astype(np.uint8), connectivity=4)
        return lab, n - 1
    h, w = mask.shape
    lab, n = np.zeros((h, w), int), 0
    for y0 in range(h):
        for x0 in range(w):
            if mask[y0, x0] and lab[y0, x0] == 0:
                n += 1; stack = [(y0, x0)]; lab[y0, x0] = n
                while stack:
                    y, x = stack.pop()
                    for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                        if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and lab[yy, xx] == 0:
                            lab[yy, xx] = n; stack.append((yy, xx))
    return lab, n

def skin_mask(w, h, crop_size, faces, rgb, lab):
    """Soft 0..1 mask of skin cells. Seeds: warm cells inside each detected face. Each face gives a reference skin colour;
    cells anywhere that look like skin (YCrCb window) AND are close to a reference colour are candidates. A candidate blob that
    touches a face is kept where it lies around / below that face (ears, neck, chest), so a skin-coloured hat or wall above the
    face stays out; any blob whose colour matches a reference very closely is kept too (hands, other people's skin)."""
    if not faces:
        return np.zeros((h, w))
    warm = (lab[..., 1] > 3) & (lab[..., 2] > 6) & (lab[..., 0] > 20)
    skin_like = ycrcb_skin(rgb) & warm
    seeds, refs, zone = np.zeros((h, w), bool), [], np.zeros((h, w), bool)
    sx, sy = w / crop_size[0], h / crop_size[1]
    yy, xx = np.mgrid[0:h, 0:w]
    for f in faces:
        fm = face_mask(w, h, crop_size, f) > 0.6
        fx = (xx + .5 - f[0] * sx) / max(f[2] * sx, 1)
        fy = (yy + .5 - f[1] * sy) / max(f[3] * sy, 1)
        reference_zone = (fx >= 0.2) & (fx <= 0.8) & (fy >= 0.25) & (fy <= 0.8)
        cells = fm & reference_zone & skin_like
        if cells.sum() < 4:
            continue
        seeds |= cells
        refs.append(np.median(lab[cells], axis=0))
        cx, cy = (f[0] + f[2] / 2) * sx, (f[1] + f[3] / 2) * sy
        rx, ry = max(f[2] * sx * 0.7, 1), max(f[3] * sy * 0.8, 1)
        dx, dy = np.abs(xx + .5 - cx) / rx, (yy + .5 - cy) / ry
        zone |= (dy >= -1.0) & (dy <= 3.0) & (dx <= np.where(dy < 0.8, 1.6, 2.4))  # face and ears, then neck and chest; not a hat or wall above
    if not refs:
        return np.zeros((h, w))
    refs = np.array(refs)
    dref = ciede2000(lab.reshape(-1, 1, 3), refs[None, :, :]).min(1).reshape(h, w)
    cand = skin_like & (dref <= 12)
    lab_cc, n = components(cand | seeds)
    keep = np.zeros((h, w), bool)
    for i in range(1, n + 1):
        blob = lab_cc == i
        if (blob & seeds).any():
            keep |= blob & zone
        elif blob.sum() >= 0.003 * w * h and np.median(dref[blob]) <= 7:
            keep |= blob
    soft = blur3(np.stack([keep.astype(float)] * 3, -1))[..., 0]
    return np.clip(np.maximum(keep, soft), 0, 1)

def adjust_skin(lab, mask, adj):
    """Skin tone adjustment on skin cells only (Lab). br: lightness, warm: toward yellow-orange (+) or cooler (-), sat: colour strength.
    Sliders run -50..50; 0 everywhere leaves the photo untouched."""
    if adj is None or not adj.get("on") or not mask.any():
        return lab
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    L2 = L + adj["br"] * 0.2
    a2, b2 = a + adj["warm"] * 0.06, b + adj["warm"] * 0.24
    scale = 1 + adj["sat"] / 100
    a2, b2 = a2 * scale, b2 * scale
    out = np.stack([np.clip(L2, 0, 100), a2, b2], -1)
    mk = mask[..., None]
    return lab * (1 - mk) + out * mk

def pick_skin_palette(lab, skin, face):
    """Choose the closest pair of warm palette colours whose CIEDE2000 difference is at most 12."""
    if skin is None or not skin.any():
        return None
    candidates = np.flatnonzero((PAL_LAB[:, 1] > 3) & (PAL_LAB[:, 2] > 6))
    if len(candidates) < 2:
        return None
    pts = lab[skin]
    weights = 1 + 3 * face[skin]
    errors = ciede2000(pts[:, None, :], PAL_LAB[candidates][None, :, :]) ** 1.5
    between = ciede2000(PAL_LAB[candidates, None, :], PAL_LAB[None, candidates, :])
    best_cost, best_pair = np.inf, None
    for i, colour in enumerate(candidates[:-1]):
        partners = np.flatnonzero((between[i] <= SKIN_PALETTE_DELTA) & (np.arange(len(candidates)) > i))
        if not len(partners):
            continue
        costs = (weights[:, None] * np.minimum(errors[:, i, None], errors[:, partners])).sum(0)
        j = int(partners[costs.argmin()])
        if costs.min() < best_cost:
            best_cost, best_pair = float(costs.min()), (int(colour), int(candidates[j]))
    if best_pair is None:
        return None
    selected = np.zeros(len(PALETTE), bool)
    selected[list(best_pair)] = True
    return selected

def constrain_skin_palette(idx, lab, skin, palette):
    if palette is None:
        return idx
    cells = np.flatnonzero(skin)
    candidates = np.flatnonzero(palette)
    distances = ciede2000(lab.reshape(-1, 3)[cells, None, :], PAL_LAB[candidates][None, :, :])
    idx.ravel()[cells] = candidates[distances.argmin(1)]
    return idx

def convert(img, w, h, box=None, dither="normal", grade=None, boost=1.0, skin=True, limit=None, profile="auto", colours="min", skin_adj=None):
    """Return the brick matrix (single source of truth), target colours, errors, face measures and a report dict.
    profile: "auto" (detect), "flat" (flags, logos, illustrations) or "photo". colours: "min", "balanced" or "max" (all capped at 54).
    limit (old API): None = "max", anything else = colours mode as given."""
    grade = grade or dict(mode="off", br=0, ct=0, sat=0, sh=0)
    mode = COLOUR_MODES.get(colours, COLOUR_MODES["min"])
    img = ImageOps.exif_transpose(img).convert("RGB")  # match how the browser shows phone photos
    img.thumbnail((2500, 2500), Image.Resampling.LANCZOS)  # far more detail than 240 bricks can use; crop fractions stay valid
    if box:  # (x, y, width, height) as fractions of the photo, set by the person in the frame editor
        iw, ih = img.size
        crop = img.crop((round(box[0] * iw), round(box[1] * ih), round((box[0] + box[2]) * iw), round((box[1] + box[3]) * ih)))
    else:
        crop = center_crop(img, w, h)
    upscaled = crop.width < w or crop.height < h
    resampling = Image.Resampling.LANCZOS if upscaled else Image.Resampling.BOX
    rgb0 = np.asarray(crop.resize((w, h), resampling), float)  # interpolate small photos; average larger photos per cell
    orig_lab = srgb_to_lab(rgb0.reshape(-1, 3)).reshape(h, w, 3)
    analysis = analyse_image(crop)
    faces = detect_faces(crop)
    if profile == "auto":  # a real face on a plain background is still a photograph
        profile = "photo" if (faces and analysis["flat_frac"] < 0.985) else analysis["profile"]
    face = faces[0] if faces else None
    m = face_mask(w, h, crop.size, face)
    report = dict(profile=profile, detected=analysis, faces=len(faces), dither=dither, grading=grade["mode"], skin_cells=0.0,
                  skin_palette=[], dropped=0, allowed=None)
    if profile == "flat":
        # ---- flat graphics: regions -> majority vote per cell -> one brick per region. No dithering, no auto grading, no sharpening.
        labels, cen, area = flat_cells(crop, w, h)
        cen_rgb = lab_to_srgb(cen)
        if grade["mode"] == "custom":  # brightness / contrast / colour still work on the region colours (sharpening is pointless here)
            cen_rgb = grade_small(cen_rgb.reshape(1, -1, 3), dict(grade, sh=0)).reshape(-1, 3)
            cen = srgb_to_lab(cen_rgb)
        report["grading"] = "custom" if grade["mode"] == "custom" else "off"
        report["dither"] = "off"
        report["regions"] = int(len(cen))
        brick_of = flat_bricks(cen, area)
        allowed = np.zeros(len(PALETTE), bool); allowed[brick_of] = True
        if allowed.sum() > MAX_COLOURS:  # cannot happen with 16 regions, kept for the strict cap
            allowed = pick_palette(cen, np.zeros(len(cen)), MAX_COLOURS)
            brick_of = np.array([int(np.flatnonzero(allowed)[ciede2000(c[None, None, :], PAL_LAB[allowed][None, :, :])[0].argmin()]) for c in cen])
        idx = brick_of[labels]
        rgb = cen_rgb[labels]
        lab = cen[labels]
        skin_cells = None
        report["allowed"] = int(allowed.sum())
    else:
        # ---- photographs: grade, adjust skin, boost the face, choose the smallest useful palette, dither within it, then merge.
        rgb = grade_small(rgb0, grade)  # the target the bricks aim for
        lab = srgb_to_lab(rgb.reshape(-1, 3)).reshape(h, w, 3)
        smask = skin_mask(w, h, crop.size, faces, rgb, lab) if (skin or (skin_adj and skin_adj.get("on"))) else np.zeros((h, w))
        if skin_adj and skin_adj.get("on"):
            lab = adjust_skin(lab, smask, skin_adj)  # skin cells only; clothing and background are untouched
            rgb = np.where((smask > 0)[..., None], lab_to_srgb(lab.reshape(-1, 3)).reshape(h, w, 3), rgb)
        report["skin_cells"] = round(float((smask > 0.5).mean() * 100), 1)
        if boost > 0 and face is not None:  # extra local contrast on the face only
            L = lab[..., 0]; lab[..., 0] = np.clip(L + boost * m * (L - blurL(L)), 0, 100)
        skin_cells = (smask > 0.5) if skin else None
        skin_palette = pick_skin_palette(lab, skin_cells, m)
        reserved = int(skin_palette.sum()) if skin_palette is not None else 0
        allowed = pick_palette(lab, m, MAX_COLOURS - reserved, mode["stop"])  # reserve room for the coordinated skin colours
        if skin_palette is not None:
            allowed |= skin_palette
            report["skin_palette"] = [PALETTE[i]["code"] for i in np.flatnonzero(skin_palette)]
        report["allowed"] = int(allowed.sum())
        idx = dither_match(lab, strength_map(m, DITHER.get(dither, 0.8)), skin_cells, allowed, skin_palette)
        idx, report["dropped"] = merge_pass(idx, lab, m, mode["merge"])
        idx = constrain_skin_palette(idx, lab, skin_cells, skin_palette) if skin_palette is not None else idx
    assert len(np.unique(idx)) <= MAX_COLOURS  # the strict cap, checked on the final matrix
    brick = PAL_LAB[idx]
    raw = ciede2000(lab.reshape(-1, 3), brick.reshape(-1, 3))  # strict, cell by cell
    blend = ciede2000(blur3(lab).reshape(-1, 3), blur3(brick).reshape(-1, 3))  # how it looks from a distance
    vs_orig = ciede2000(blur3(orig_lab).reshape(-1, 3), blur3(brick).reshape(-1, 3))  # against YOUR untouched photo
    structure, detail = face_metrics(orig_lab, brick, m)
    return idx, rgb.reshape(-1, 3), raw, blend, crop, face is not None, vs_orig, structure, detail, report

def stud_shade(cell):
    """Light/shadow pattern for one brick: a round stud on a flat tile, tiles touching with only a hairline between."""
    yy, xx = (np.mgrid[0:cell, 0:cell] + 0.5) / cell - 0.5
    r = np.hypot(xx, yy)
    inside = np.clip((0.33 - r) * cell + 0.5, 0, 1)                  # soft-edged round stud
    rim = np.clip((r - 0.24) / 0.09, 0, 1)                            # stud gets darker toward its edge
    stud = 0.07 - 0.20 * rim - 0.30 * (xx + yy)                       # lit from the top left
    around = -0.14 * np.clip(1 - (r - 0.33) * cell / 2.5, 0, 1)       # small shadow around the stud
    edge = np.zeros((cell, cell)); edge[0, :] = edge[:, 0] = -0.08    # hairline between neighbouring bricks
    return inside * stud + (1 - inside) * around + edge

def render(matrix, cell=16):
    h, w = matrix.shape
    base = np.repeat(np.repeat(PAL_RGB[matrix], cell, 0), cell, 1)    # every brick fills its whole square, no gaps
    shade = np.tile(stud_shade(cell), (h, w))
    return Image.fromarray(np.clip(base * (1 + shade[..., None]), 0, 255).astype(np.uint8))

@app.route("/")
def index():
    return render_template("index.html", groups=SIZE_GROUPS, default_size=DEFAULT_SIZE,
                           max_plate_units=MAX_PLATE_UNITS, version=APP_VERSION,
                           boot=dict(sizes=SIZES, version=APP_VERSION))

@app.route("/convert", methods=["POST"])
def do_convert():
    f, key = request.files.get("photo"), request.form.get("size", DEFAULT_SIZE)
    dimensions = resolve_size(key)
    if not f or dimensions is None:
        return jsonify(error="Upload a JPG/PNG photo and choose a size."), 400
    try:
        img = Image.open(f.stream)
    except Exception:
        return jsonify(error="That file is not a valid image."), 400
    w, h = dimensions
    try:
        box = [float(request.form[k]) for k in ("cx", "cy", "cw", "ch")]
        ok = box[2] > 0.01 and box[3] > 0.01 and min(box[:2]) >= -0.001 and box[0] + box[2] <= 1.001 and box[1] + box[3] <= 1.001
        box = [min(max(v, 0.0), 1.0) for v in box] if ok else None
    except (KeyError, ValueError):
        box = None
    num = lambda k, lo, hi: min(hi, max(lo, float(request.form.get(k, 0) or 0)))
    mode = request.form.get("gmode", "off")
    grade = dict(mode=mode if mode in ("off", "auto", "custom") else "off",
                 br=num("gbr", -50, 50), ct=num("gct", -50, 50), sat=num("gsat", -50, 50), sh=num("gsh", 0, 100))
    colours = request.form.get("colours") or ("min" if request.form.get("limit", "1") == "1" else "max")  # old "limit" form field still works
    if colours not in COLOUR_MODES:
        colours = "min"
    profile = request.form.get("profile", "auto")
    if profile not in ("auto", "flat", "photo"):
        profile = "auto"
    skin_adj = dict(on=request.form.get("sk_on", "0") == "1", br=num("sk_br", -50, 50), warm=num("sk_warm", -50, 50), sat=num("sk_sat", -50, 50))
    m, photo_rgb, raw, de, crop, face, vs_orig, structure, detail, report = convert(
        img, w, h, box, request.form.get("dither", "normal"), grade,
        1.0 if request.form.get("boost", "1") == "1" else 0.0, request.form.get("skin", "1") == "1",
        None, profile, colours, skin_adj)
    counts = np.bincount(m.ravel(), minlength=len(PALETTE))
    assert counts.sum() == w * h  # quantities must equal total cells
    assert int((counts > 0).sum()) <= MAX_COLOURS  # strict 54-colour cap on the final mosaic (counts, CSV and stats all come from this matrix)
    rows = [dict(code=PALETTE[i]["code"], name=PALETTE[i]["name"], hex=PALETTE[i]["hex"], count=int(c))
            for i, c in enumerate(counts) if c]
    rows.sort(key=lambda r: -int(r["count"]))
    buf = io.BytesIO(); render(m, cell=max(5, min(16, 1600 // max(w, h)))).save(buf, "JPEG", quality=92, subsampling=0)  # smaller bricks on huge mosaics keeps the picture a sensible size
    flat = m.ravel()
    cells = [[int(round(r)), int(round(g)), int(round(b)), int(i), round(float(e0), 2), round(float(e), 2)] for (r, g, b), i, e0, e in zip(photo_rgb, flat, raw, de)]
    stats = dict(mean=round(float(de.mean()), 2), median=round(float(np.median(de)), 2),
                 p90=round(float(np.percentile(de, 90)), 2), max=round(float(de.max()), 2),
                 over5=round(float((de > 5).mean() * 100), 1), raw_mean=round(float(raw.mean()), 2),
                 vs_orig=round(float(vs_orig.mean()), 2), face_structure=structure, face_detail=detail)
    return jsonify(preview="data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(),
                   cells=cells, stats=stats, w=w, h=h, face=face, upscaled=crop.width < w or crop.height < h, version=APP_VERSION,
                   used=len(rows), limit=MAX_COLOURS, colours_mode=colours, report=report,
                   palette=[[p["code"], p["name"], p["hex"]] for p in PALETTE],
                   studs=f"{w} x {h}", cm=f"{w * STUD_CM:.1f} x {h * STUD_CM:.1f}",
                   frame_cm=f"{w * STUD_CM + 0.2:.1f} x {h * STUD_CM + 0.2:.1f}",
                   plates=(w // PLATE_STUDS) * (h // PLATE_STUDS), total=w * h, colours=rows)

@app.route("/detect", methods=["POST"])
def detect():
    if _FACE is None:  # OpenCV is not installed: say so instead of pretending no face exists
        return jsonify(face=None, unavailable=FACE_WHY)
    f = request.files.get("photo")
    if f is None:
        return jsonify(face=None)
    try:
        img = ImageOps.exif_transpose(Image.open(f.stream)).convert("RGB")
    except Exception:
        return jsonify(face=None)
    fb = detect_face(img)
    if fb is None:
        return jsonify(face=None)
    iw, ih = img.size
    return jsonify(face=dict(cx=(fb[0] + fb[2] / 2) / iw, cy=(fb[1] + fb[3] / 2) / ih, w=fb[2] / iw, h=fb[3] / ih))

if __name__ == "__main__":
    app.run(debug=True)