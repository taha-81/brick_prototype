# brick_prototype

Photo -> brick mosaic prototype (Flask + NumPy + Pillow + OpenCV).

## Day 5: minimum colours, strict 54-colour cap, skin tone adjustment

**Colour pipeline (app.py)**

1. `analyse_image()` looks at a 256 px copy of the framed picture and decides between two profiles:
   - **flat** (flags, logos, cartoons): most pixels sit on a few colour centres and edges are sharp
     (`flat_frac >= 0.93`, or `>= 0.85` with few soft edges). A detected face always means "photo".
   - **photo**: everything else. The user can override this with the *Image type* control (`profile=auto|flat|photo`).
2. **Flat profile** (`flat_cells`, `flat_bricks`): k-means in CIELAB on the framed picture, centres closer than
   ΔE 6 are merged, colour groups that only exist on edges (anti-aliasing, JPEG ringing) are dropped, every brick
   cell takes the colour of the group that covers most of it (majority vote, so boundaries stay sharp), and each
   group gets its closest brick (CIEDE2000). No dithering, no auto grading, no sharpening. A flag with two colours
   gives a mosaic with two colours.
3. **Photo profile**: grading -> skin tone adjustment (skin cells only) -> face boost -> `pick_palette()` greedy
   search that stops when the next colour would cut the remaining error by less than `stop` (so it picks the
   smallest useful palette, never more than 54) -> Floyd-Steinberg dithering restricted to that palette ->
   `merge_pass()` drops any brick colour whose removal raises the mean blended ΔE by less than `merge`.
4. `COLOUR_MODES`: `min` (default, fewest colours), `balanced`, `max` (most detail, still capped at 54).
   `MAX_COLOURS = 54` is enforced in `pick_palette`, asserted on the final matrix and on the counts that feed the
   table and the CSV.

**Skin tone adjustment** (`skin_mask`, `adjust_skin`): every detected face seeds a skin colour; cells that look like
skin (YCrCb window + close to a face's skin colour) are kept when they touch a face (face, ears, neck, chest zone) or
match a face's skin colour very closely (hands, other people). Brightness, warmth and colour strength are applied in
Lab to cells matching the skin-color range and face reference only; 0/0/0 changes nothing. The brightness slider is
limited to a moderate lightness change (+/-10 Lab L at its endpoints) to avoid washing out skin. Skin protection now
uses a coordinated pair of warm palette colours with at most ΔE 12 between them, preventing unrelated colors in hair
and shadows from being forced into skin shades. Form fields: `sk_on`, `sk_br`, `sk_warm`, `sk_sat` (-50..50).

Form fields added: `colours` (min|balanced|max), `profile` (auto|flat|photo). The old `limit` field still works
(`limit=0` -> max, otherwise min). The response carries `report` (profile, detection numbers, allowed/dropped
colours, skin coverage) and `limit` (54).

## Run

    pip install -r requirements.txt
    python app.py
