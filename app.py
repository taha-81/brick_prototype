"""Photo -> brick mosaic prototype (Day 1): upload, size, palette match, preview, counts."""
import base64, csv, io, math, os
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

APP_VERSION = "day4-colour-limit-1"
app = Flask(__name__)
BASE = os.path.dirname(os.path.abspath(__file__))  # data files are found next to app.py, wherever the server starts
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024  # Vercel refuses bigger requests anyway (about 4.5 MB)

@app.errorhandler(413)
def too_big(_):
    return jsonify(error="The photo is too large. Try a smaller photo."), 413
DITHER = {"off": 0.0, "soft": 0.5, "normal": 0.8, "strong": 1.0}  # strength on the background
FACE_KEEP = 0.25  # on the face, dithering is cut to this share, so eyes and lips stay crisp
FS = ((0, 1, 7 / 16), (1, -1, 3 / 16), (1, 0, 5 / 16), (1, 1, 1 / 16))  # Floyd-Steinberg
STUD_CM, PLATE_STUDS = 0.8, 24  # 8 mm brick, 24x24-stud baseplate (19.2 cm)
MAX_COLOURS = 55  # client limit: with the colour-limit option on, a mosaic uses at most this many of the 94 colours

def load_sizes(path=os.path.join(BASE, "sizes.csv")):
    """All sizes from the supplier's Excel sheet (kept in sizes.csv). Every side must be whole baseplates (multiples of 24 studs)."""
    sizes, groups = {}, {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            w, h = int(r["studs_w"]), int(r["studs_h"])
            assert w % PLATE_STUDS == 0 and h % PLATE_STUDS == 0, f"size {w}x{h} is not whole baseplates"
            key = f"{w // PLATE_STUDS}x{h // PLATE_STUDS}"  # baseplate layout, e.g. 2x3 = 2 plates wide, 3 tall
            sizes[key] = (w, h)
            groups.setdefault(r["orientation"], []).append(
                dict(key=key, label=f"{key} plates, {w}x{h} studs, {w * STUD_CM:.1f}x{h * STUD_CM:.1f} cm"))
    return sizes, list(groups.items())

SIZES, SIZE_GROUPS = load_sizes()

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

def pick_palette(lab, m, k=MAX_COLOURS):
    """Choose the k palette colours that serve this photo best. Returns a True/False mask over the 94 colours.
    Greedy pick (add the colour that cuts the total error most), then swaps to polish. The face counts 4x, so skin tones win.
    Squared-ish error (d ** 1.5) so a few badly matched cells weigh more than many slightly off ones."""
    n = len(PALETTE)
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
        sel.append(c); cur = np.minimum(cur, F[:, c])
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

def dither_match(lab, strength, skin=None, allowed=None):
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

def grade_small(rgb, g):
    """Grade the photo at brick resolution (this is the size that matters): levels, brightness, contrast, colour, sharpness."""
    if g["mode"] == "off":
        return rgb
    x = rgb.copy()
    lum = x @ np.array([0.2126, 0.7152, 0.0722])
    if g["mode"] == "auto":
        lo, hi = np.percentile(lum, 1), np.percentile(lum, 99)
        if hi - lo > 20:
            x = x + 0.7 * ((x - lo) * (255 / (hi - lo)) - x)  # stretch the tones, 70% of the way
        g = dict(g, br=0, ct=0, sat=10, sh=40)
    x = x + g["br"] * 1.28
    x = (x - 128) * (1 + g["ct"] / 100) + 128
    gray = (x @ np.array([0.2126, 0.7152, 0.0722]))[..., None]
    x = gray + (x - gray) * (1 + g["sat"] / 100)
    x = x + (g["sh"] / 100 * 1.5) * (x - blur3(x))
    return np.clip(x, 0, 255)

def face_metrics(orig_lab, brick_lab, m):
    """How well the face survives: correlation of lightness (structure) and of fine detail (eyes, brows, lips)."""
    sel = m > 0.5
    if sel.sum() < 20:
        return None, None
    Lo, Lb = orig_lab[..., 0], brick_lab[..., 0]
    c = lambda a, b: round(float(np.nan_to_num(np.corrcoef(a[sel], b[sel])[0, 1])), 3)
    return c(Lo, Lb), c(Lo - blurL(Lo), Lb - blurL(Lb))

def convert(img, w, h, box=None, dither="normal", grade=None, boost=1.0, skin=True, limit=None):
    """Return the brick matrix (single source of truth), target colours, errors and face measures."""
    grade = grade or dict(mode="off", br=0, ct=0, sat=0, sh=0)
    img = ImageOps.exif_transpose(img).convert("RGB")  # match how the browser shows phone photos
    img.thumbnail((2500, 2500), Image.Resampling.LANCZOS)  # far more detail than 240 bricks can use; crop fractions stay valid
    if box:  # (x, y, width, height) as fractions of the photo, set by the person in the frame editor
        iw, ih = img.size
        crop = img.crop((round(box[0] * iw), round(box[1] * ih), round((box[0] + box[2]) * iw), round((box[1] + box[3]) * ih)))
    else:
        crop = center_crop(img, w, h)
    rgb0 = np.asarray(crop.resize((w, h), Image.Resampling.BOX), float)  # average photo colour per cell, untouched
    orig_lab = srgb_to_lab(rgb0.reshape(-1, 3)).reshape(h, w, 3)
    face = detect_face(crop)
    m = face_mask(w, h, crop.size, face)
    rgb = grade_small(rgb0, grade)  # the target the bricks aim for
    lab = srgb_to_lab(rgb.reshape(-1, 3)).reshape(h, w, 3)
    if boost > 0 and face is not None:  # extra local contrast on the face only
        L = lab[..., 0]; lab[..., 0] = np.clip(L + boost * m * (L - blurL(L)), 0, 100)
    skin_cells = (m > 0.5) & (lab[..., 1] > 3) & (lab[..., 2] > 6) & (lab[..., 0] > 25) if skin else None
    allowed = pick_palette(lab, m, limit) if limit else None  # None = all 94 colours, exactly as before
    idx = dither_match(lab, strength_map(m, DITHER.get(dither, 0.8)), skin_cells, allowed)
    brick = PAL_LAB[idx]
    raw = ciede2000(lab.reshape(-1, 3), brick.reshape(-1, 3))  # strict, cell by cell
    blend = ciede2000(blur3(lab).reshape(-1, 3), blur3(brick).reshape(-1, 3))  # how it looks from a distance
    vs_orig = ciede2000(blur3(orig_lab).reshape(-1, 3), blur3(brick).reshape(-1, 3))  # against YOUR untouched photo
    structure, detail = face_metrics(orig_lab, brick, m)
    return idx, rgb.reshape(-1, 3), raw, blend, crop, face is not None, vs_orig, structure, detail

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
    return render_template("index.html", groups=SIZE_GROUPS, version=APP_VERSION, boot=dict(sizes=SIZES, version=APP_VERSION))

@app.route("/convert", methods=["POST"])
def do_convert():
    f, key = request.files.get("photo"), request.form.get("size", "2x2")
    if not f or key not in SIZES:
        return jsonify(error="Upload a JPG/PNG photo and choose a size."), 400
    try:
        img = Image.open(f.stream)
    except Exception:
        return jsonify(error="That file is not a valid image."), 400
    w, h = SIZES[key]
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
    m, photo_rgb, raw, de, crop, face, vs_orig, structure, detail = convert(
        img, w, h, box, request.form.get("dither", "normal"), grade,
        1.0 if request.form.get("boost", "1") == "1" else 0.0, request.form.get("skin", "1") == "1",
        MAX_COLOURS if request.form.get("limit", "0") == "1" else None)
    counts = np.bincount(m.ravel(), minlength=len(PALETTE))
    assert counts.sum() == w * h  # quantities must equal total cells
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
                   cells=cells, stats=stats, w=w, h=h, face=face, version=APP_VERSION,
                   used=len(rows), limit=MAX_COLOURS if request.form.get("limit", "0") == "1" else None,
                   palette=[[p["code"], p["name"], p["hex"]] for p in PALETTE],
                   studs=f"{w} x {h}", cm=f"{w * STUD_CM:.1f} x {h * STUD_CM:.1f}",
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