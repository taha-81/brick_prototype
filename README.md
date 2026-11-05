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
3. **Photo profile**: grading -> skin tone adjustment (skin cells only) -> face-detail emphasis from local luminance
   edges -> `pick_palette()` greedy search that stops when the next colour would cut the remaining error by less than
   `stop` (so it picks the smallest useful palette, never more than 54) -> Floyd-Steinberg dithering restricted to
   that palette and reduced further on facial detail edges ->
   `merge_pass()` drops any brick colour whose removal raises the mean blended ΔE by less than `merge`.
4. `COLOUR_MODES`: `min` (default, fewest colours), `balanced`, `max` (most detail, still capped at 54).
   `MAX_COLOURS = 54` is enforced in `pick_palette`, asserted on the final matrix and on the counts that feed the
   table and the CSV.

**Skin tone adjustment** (`skin_mask`, `adjust_skin`): every detected face seeds a skin colour; cells that look like
skin (YCrCb window + close to a face's skin colour) are kept when they touch a face (face, ears, neck, chest zone) or
match a face's skin colour very closely (hands, other people). Brightness, warmth and colour strength are applied in
Lab to cells matching the skin-color range and face reference only; 0/0/0 changes nothing. The brightness slider is
limited to a moderate lightness change (+/-10 Lab L at its endpoints) to avoid washing out skin. Skin protection
uses up to four coordinated warm palette shades, each within ΔE 12 of another selected shade and no more than ΔE 24
across the range. Facial feature edges receive extra palette weight and less dithering. The default photo dithering
is Soft; larger portrait sizes including Portrait 6x8 (96x128 bricks) are available, and detected faces are framed
more prominently by default. Form fields: `sk_on`, `sk_br`, `sk_warm`, `sk_sat` (-50..50).

Form fields added: `colours` (min|balanced|max), `profile` (auto|flat|photo). The old `limit` field still works
(`limit=0` -> max, otherwise min). The response carries `report` (profile, detection numbers, allowed/dropped
colours, skin coverage) and `limit` (54).

**Photo Enhancer** is an optional, local OpenCV preprocessing step. It applies bilateral edge-preserving smoothing,
moderate Lab colour clustering and edge-aware blending to create a visibly more illustrated image while retaining
important boundaries. The `/enhance` endpoint returns a preview image; the browser shows it beside the original
and uses the enhanced file for mosaic conversion only when the checkbox is enabled. No external API or cloud image
service is used. It is unchecked by default.

## Run

    pip install -r requirements.txt
    python app.py
