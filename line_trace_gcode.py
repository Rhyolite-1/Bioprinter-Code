#!/usr/bin/env python3
"""
Continuous line-tracing G-code generator (SVG-plotter style) with hybrid
skeleton+outline path generation and circular-dish scaling.

- Black/dark pixels are printed (explicit rule, DARK_THRESHOLD).
- Hybrid pathing:
    * Skeleton (medial axis) used for thin/centerlines
    * Outline contours included for regions whose medial radius exceeds a threshold
- Scaling: fits design so the farthest foreground pixel from image center
  maps to the printable circle of diameter MAX_AREA_MM.
- Preview: shows source image, dish circle, toolpaths and a simulated line width
  based on extrusion_per_mm with a simple proportional mapping.

Usage:
  Put one or more images (.png/.jpg/.jpeg) next to this script and run it.
"""

import os
import glob
import math
import sys
import argparse
from functools import lru_cache

from PIL import Image
import numpy as np
from skimage import measure, morphology
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# ---- parameters ----
DISH_CENTER_X = 100.0
DISH_CENTER_Y = 100.0
START_Z = 5.000
PRINT_HEIGHT_Z = 4.500
TRAVEL_HEIGHT_Z = 4.800

DEFAULT_EXTRUSION_PER_MM = 0.0030  # mm^3/mm -- lowest value that worked in past physical testing
RETRACT_E = 0.02

MAX_AREA_MM = 79.0  # mm -- max span of the longer image dimension. 85mm dish, radius 42.5mm,
                     # minus ~2mm spread and ~1mm edge buffer = 39.5mm max design radius = 79mm diameter
DARK_THRESHOLD = 128
MIN_CONTOUR_LEN_PX = 15

DISH_DIAMETER_MM = 85.0  # physical dish -- kept only as a reference measurement now
DISH_RADIUS_MM = DISH_DIAMETER_MM / 2.0
PRINT_AREA_RADIUS_MM = MAX_AREA_MM / 2.0  # 39.5mm -- the actual hard boundary paths are clipped to;
                                           # this is the buffered print area, not the raw dish edge, so
                                           # nothing prints in that edge margin even if a path reaches it.
DEFAULT_AGAR_VOLUME_ML = 20.0
# clearance kept between the print-height Z and the travel-height Z, preserved from the original
# fixed PRINT_HEIGHT_Z=4.500 / TRAVEL_HEIGHT_Z=4.800 pair so the retract/travel gap doesn't change
# just because print height is now computed instead of hardcoded.
TRAVEL_CLEARANCE_MM = TRAVEL_HEIGHT_Z - PRINT_HEIGHT_Z
FINAL_LIFT_MM = 50.0  # raise the needle this far above the travel height once printing is done
PRINT_FEED_MM_PER_MIN = 600.0  # feed rate used for the actual line-tracing moves (matches prior F600 literal)
EXTRUDE_TRAIL_SEC = 1.0  # stop extruding this many seconds before movement finishes at the end of the whole print


def agar_surface_height_mm(volume_ml=DEFAULT_AGAR_VOLUME_ML, dish_diameter_mm=DISH_DIAMETER_MM):
    """
    Height of the agar surface (mm) for a given poured volume in a cylindrical
    dish, assuming a flat fill: height = volume / (pi * radius^2).
    1 mL == 1000 mm^3, so a volume in mL converts directly.
    """
    volume_mm3 = volume_ml * 1000.0
    radius_mm = dish_diameter_mm / 2.0
    area_mm2 = math.pi * radius_mm ** 2
    return volume_mm3 / area_mm2


PRINT_HEIGHT_CLEARANCE_MM = 0.1  # keep the needle just above the computed agar surface, not touching it
DEFAULT_PRINT_HEIGHT_MM = agar_surface_height_mm() + PRINT_HEIGHT_CLEARANCE_MM

# Hybrid path parameters
OUTLINE_MIN_RADIUS_PX = 3  # include outline for regions whose medial radius (px) >= this
MIN_SKELETON_LEN_PX = 4    # ignore very short skeleton fragments
MAX_TRACE_SIDE_PX = 300    # reduce large images before expensive contour extraction

# Preview mapping (Option A: simple proportional mapping)
PREVIEW_WIDTH_SCALE = 10.0  # mm of visual line width per 1.0 unit of extrusion_per_mm
# e.g. line_width_mm = extrusion_per_mm * PREVIEW_WIDTH_SCALE

def find_images(folder):
    exts = ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG")
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(folder, ext)))
    return sorted(set(files))


def prepare_image_for_print(path, out_path=None, trace_mode="both", max_side_px=MAX_TRACE_SIDE_PX):
    """
    Load the source image once, threshold it into a foreground mask according to
    trace_mode, and downsample that mask (if needed) -- all in a single pass.

    Previously this conversion happened in up to three separate places (here,
    again in load_dark_mask after a disk round-trip, and again inside
    hybrid_paths_from_mask), and the downsample step there used dimensions that
    didn't match what got used later for mm scaling. This is now the one place
    dark-pixel thresholding and resolution reduction happen; every downstream
    caller uses the same mask and the same (h, w).

    Returns (preview_image, mask, h, w) where mask/h/w are already at the final
    working resolution used for path extraction AND mm scaling.
    """
    # Load and flatten transparency onto white BEFORE grayscale conversion.
    # img.convert("L") alone ignores alpha entirely: for a palette/RGBA image
    # with a transparent background, it reads whatever RGB value happens to
    # sit underneath the transparent pixels, which is not necessarily white.
    # On the real yin-yang PNG this pulled in leftover watermark color data
    # and inverted parts of the read. Compositing onto a white background
    # first makes "transparent" mean "background" like it visually should.
    # Verified this is a no-op (byte-identical output) on plain RGB, grayscale,
    # and fully-opaque RGBA images -- it only changes anything for pixels that
    # are actually partially/fully transparent.
    img = Image.open(path).convert("RGBA")
    white_bg = Image.fromarray(np.full((*img.size[::-1], 4), 255, dtype=np.uint8), mode="RGBA")
    img = Image.alpha_composite(white_bg, img).convert("L")
    arr = np.array(img)
    mask = arr < DARK_THRESHOLD  # explicit: only dark pixels count as foreground

    # Downsample the raw foreground mask FIRST, before any thinning. Doing it
    # after skeletonize/edge extraction (the previous order) fed a mask that
    # was already only 1px wide into a resize step that could erase it
    # entirely (see downsample_mask_for_tracing docstring). Downsampling the
    # thicker raw mask first is both correct and cheaper, since skeletonize
    # then runs on a smaller array.
    if max_side_px is not None and (mask.shape[0] > max_side_px or mask.shape[1] > max_side_px):
        mask, _ = downsample_mask_for_tracing(mask, max_side_px=max_side_px)

    if trace_mode == "skeletonize":
        mask = morphology.skeletonize(mask)
    # else ("edge" or "both"/hybrid): keep the raw foreground mask as-is.
    # "edge" mode used to erode+dilate this into a synthetic outline "ring"
    # band before contour-tracing it -- but find_contours on a ring traces
    # BOTH its inner and outer edge, since both are boundaries of the ring
    # shape. That produced two nearly-identical overlapping contours (~2x
    # the path length of the true boundary) instead of one clean outline.
    # Tracing the raw mask directly with find_contours (same as hybrid mode
    # already does per-region) gives a single accurate boundary contour.

    h, w = mask.shape
    base = np.where(mask, 0, 255).astype(np.uint8)
    bw = Image.fromarray(base, mode="L")
    if out_path is not None:
        bw.save(out_path)

    return bw, mask, h, w


def contour_to_mm(contour_px, h, w, arg1, arg2=None, arg3=None, arg4=None):
    if arg4 is not None:
        area_w_mm = arg1
        area_h_mm = arg2
        origin_x = arg3
        origin_y = arg4
        pts_mm = []
        for row, col in contour_px:
            x = origin_x + (col / (w - 1)) * area_w_mm
            y = origin_y + (1 - row / (h - 1)) * area_h_mm
            pts_mm.append((float(x), float(y)))
        return pts_mm

    pixel_to_mm = arg1
    origin_x = arg2
    origin_y = arg3
    pts_mm = []
    for row, col in contour_px:
        x = origin_x + col * pixel_to_mm
        y = origin_y + (h - 1 - row) * pixel_to_mm
        pts_mm.append((float(x), float(y)))
    return pts_mm


def pixels_to_mm_point(row, col, h, pixel_to_mm, origin_x, origin_y):
    x = origin_x + col * pixel_to_mm
    y = origin_y + (h - 1 - row) * pixel_to_mm
    return (float(x), float(y))


def downsample_mask_for_tracing(mask, max_side_px=MAX_TRACE_SIDE_PX):
    """
    Reduce large masks to a coarser resolution before contour extraction.

    Uses max-pooling (a block is foreground if ANY pixel in it is foreground)
    instead of bilinear averaging + threshold. Averaging kills thin,
    single-pixel-wide content: a skeleton line or edge outline is mostly
    surrounded by background within any downsample block, so its average
    value never crosses the threshold and the whole line disappears. This
    was silently zeroing out all output in skeletonize/edge trace modes on
    large images.
    """
    if mask.ndim != 2:
        mask = np.asarray(mask)

    h, w = mask.shape
    if max(h, w) <= max_side_px:
        return mask.astype(bool, copy=False), (1.0, 1.0)

    scale = max_side_px / max(h, w)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))

    row_bounds = np.linspace(0, h, new_h + 1).astype(int)
    col_bounds = np.linspace(0, w, new_w + 1).astype(int)
    out = np.zeros((new_h, new_w), dtype=bool)
    for i in range(new_h):
        r0, r1 = row_bounds[i], max(row_bounds[i + 1], row_bounds[i] + 1)
        row_slice = mask[r0:r1]
        for j in range(new_w):
            c0, c1 = col_bounds[j], max(col_bounds[j + 1], col_bounds[j] + 1)
            out[i, j] = row_slice[:, c0:c1].any()

    return out, (h / new_h, w / new_w)


@lru_cache(maxsize=128)
def _cached_skeleton(mask_bytes, shape):
    # mask_bytes must be paired with shape: tobytes() alone throws away the
    # array's dimensions, so np.array(mask_bytes, dtype=bool) previously came
    # back 0-D and crashed skeletonize. frombuffer + reshape restores it correctly.
    # frombuffer gives a read-only view; skimage's Cython skeletonize needs a
    # writable buffer, so copy() it.
    mask = np.frombuffer(mask_bytes, dtype=bool).reshape(shape).copy()
    return morphology.skeletonize(mask)


def _skeleton_neighbors(p):
    r, c = p
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield (r + dr, c + dc)


def _walk_skeleton_from(start, coords_set, visited):
    """Greedy neighbor-following walk starting at `start`, preferring the
    neighbor that keeps the path smoothest (max dot product with the
    previous step direction). Marks visited pixels in place."""
    path = [start]
    visited.add(start)
    cur = start
    while True:
        next_candidates = [n for n in _skeleton_neighbors(cur) if n in coords_set and n not in visited]
        if not next_candidates:
            break
        if len(next_candidates) == 1:
            nxt = next_candidates[0]
        else:
            prev = path[-2] if len(path) >= 2 else None
            if prev is None:
                nxt = next_candidates[0]
            else:
                best = None
                best_dot = -9999
                vprev = (cur[0] - prev[0], cur[1] - prev[1])
                for cand in next_candidates:
                    v = (cand[0] - cur[0], cand[1] - cur[1])
                    dot = vprev[0] * v[0] + vprev[1] * v[1]
                    if dot > best_dot:
                        best_dot = dot
                        best = cand
                nxt = best
        path.append(nxt)
        visited.add(nxt)
        cur = nxt
    return path


def extract_skeleton_paths(mask):
    """
    Returns list of skeleton paths as arrays of (row, col) coordinates (floats).
    Finds connected components of the skeleton, then orders pixels in each
    component into one or more paths by traversing neighbors.

    Branch-aware: at a junction (a pixel with 3+ skeleton neighbors, e.g. a Y
    or X crossing, or the internal branch point of a filled shape's medial
    axis), a single greedy walk from one endpoint only follows one branch and
    silently drops the others -- the walk has no way to backtrack once it
    commits to a neighbor. To keep every pixel, this starts a walk from every
    endpoint (degree-1 pixel) in the component, so each branch gets its own
    walk, and then sweeps up anything still unvisited afterward (covers loops
    with no endpoints, and any fragment an endpoint-walk didn't reach). A
    component with N branch points now emits multiple path segments instead
    of one incomplete one -- downstream code (path ordering, gcode
    generation) sees more, shorter paths per component rather than fewer,
    longer ones.
    """
    if mask.sum() == 0:
        return [], None

    # Use a cached skeleton for repeated calls and avoid the slower medial-axis pass.
    if mask.size < 500000:
        skeleton = _cached_skeleton(mask.tobytes(), mask.shape)
    else:
        skeleton = morphology.skeletonize(mask)

    # Label connected skeleton components
    labeled = measure.label(skeleton, connectivity=2)
    paths = []
    for lbl in range(1, labeled.max() + 1):
        coords = np.column_stack(np.nonzero(labeled == lbl))  # rows, cols
        if coords.shape[0] < MIN_SKELETON_LEN_PX:
            continue

        coords_set = {tuple(map(int, c)) for c in coords}
        deg = {p: sum(1 for n in _skeleton_neighbors(p) if n in coords_set) for p in coords_set}

        visited = set()
        # Start a walk from every endpoint so each branch off a junction gets
        # traced, instead of only whichever one a single walk happens upon.
        endpoints = sorted(p for p, d in deg.items() if d == 1)
        for start in endpoints:
            if start in visited:
                continue
            path = _walk_skeleton_from(start, coords_set, visited)
            if len(path) >= MIN_SKELETON_LEN_PX:
                paths.append(np.array(path, dtype=float))

        # Mop up anything left over: loops (no degree-1 pixels at all) or
        # fragments an endpoint-walk didn't reach. Keep walking from
        # whatever unvisited pixel remains until the component is covered.
        remaining = coords_set - visited
        while remaining:
            start = min(remaining)
            path = _walk_skeleton_from(start, coords_set, visited)
            if len(path) >= MIN_SKELETON_LEN_PX:
                paths.append(np.array(path, dtype=float))
            remaining = coords_set - visited

    return paths, None


def extract_outline_paths(mask):
    """
    Returns list of outline contours (arrays of (row, col)) found via measure.find_contours.
    Each contour is returned as an Nx2 array of (row, col) floats.
    """
    contours = measure.find_contours(mask.astype(float), 0.5)
    contours = [c for c in contours if c.shape[0] >= MIN_CONTOUR_LEN_PX]
    return contours


def hybrid_paths_from_mask(mask, h, w, trace_mode="both"):
    """
    Build path set (list of pixel-coordinate paths) using the requested trace mode.
    """
    # NOTE: this function no longer downsamples internally. It used to, but the
    # caller (process_image) was scaling the resulting pixel paths to mm using
    # the *original* image's h/w, not the downsampled ones -- so paths came out
    # squeezed into a corner of the plate instead of filling it. Downsampling
    # now happens exactly once, in process_image, before h/w are ever used for
    # scaling, so every consumer of `mask` (path extraction and mm scaling)
    # agrees on the same dimensions.
    paths = []
    if mask.sum() == 0:
        return paths

    if trace_mode == "skeletonize":
        skel_paths, _ = extract_skeleton_paths(mask)
        for p in skel_paths:
            if p.shape[0] >= MIN_SKELETON_LEN_PX:
                paths.append(p)
        return paths

    if trace_mode == "edge":
        contours = extract_outline_paths(mask)
        for c in contours:
            if c.shape[0] >= MIN_CONTOUR_LEN_PX:
                paths.append(np.array(c, dtype=float))
        return paths

    # Hybrid: use both outline and skeleton information for every foreground region.
    labeled_regions = measure.label(mask, connectivity=2)
    n_regions = labeled_regions.max()

    for region_label in range(1, n_regions + 1):
        region_mask = (labeled_regions == region_label)
        if region_mask.sum() == 0:
            continue

        region_contours = measure.find_contours(region_mask.astype(float), 0.5)
        for c in region_contours:
            if c.shape[0] >= MIN_CONTOUR_LEN_PX:
                paths.append(np.array(c, dtype=float))

        skel_paths, _ = extract_skeleton_paths(region_mask)
        for p in skel_paths:
            if p.shape[0] >= MIN_SKELETON_LEN_PX:
                paths.append(p)

    if not paths:
        skel_paths, _ = extract_skeleton_paths(mask)
        for p in skel_paths:
            if p.shape[0] >= MIN_SKELETON_LEN_PX:
                paths.append(p)

    return paths


def pixel_center_and_scale(mask, h, w, return_legacy=True,
                            scale_pct=100.0, offset_x_mm=0.0, offset_y_mm=0.0):
    """
    Compute the image bounding box size and placement for the agar plate.
    Returns either (pixel_to_mm, origin_x, origin_y) for compatibility or
    (area_w_mm, area_h_mm, origin_x, origin_y) for the main workflow.

    scale_pct scales the auto-fit footprint (100 = fills MAX_AREA_MM as
    before). offset_x_mm/offset_y_mm shift the origin away from the
    auto-centered position -- positive X is right, positive Y is up, matching
    the existing mm coordinate convention used elsewhere in this file (see
    contour_to_mm: y is flipped from pixel row order, not offset here).
    """
    if w >= h:
        area_w_mm = MAX_AREA_MM
        area_h_mm = MAX_AREA_MM * h / w
    else:
        area_h_mm = MAX_AREA_MM
        area_w_mm = MAX_AREA_MM * w / h

    scale_factor = scale_pct / 100.0
    area_w_mm *= scale_factor
    area_h_mm *= scale_factor

    origin_x = DISH_CENTER_X - area_w_mm / 2.0 + offset_x_mm
    origin_y = DISH_CENTER_Y - area_h_mm / 2.0 + offset_y_mm

    if return_legacy:
        pixel_to_mm = area_w_mm / (w - 1)
        return pixel_to_mm, origin_x, origin_y

    return area_w_mm, area_h_mm, origin_x, origin_y


def _farthest_corner_mm(area_w_mm, area_h_mm, origin_x, origin_y):
    corners = [
        (origin_x, origin_y),
        (origin_x + area_w_mm, origin_y),
        (origin_x, origin_y + area_h_mm),
        (origin_x + area_w_mm, origin_y + area_h_mm),
    ]
    return max(math.hypot(x - DISH_CENTER_X, y - DISH_CENTER_Y) for x, y in corners)


def design_bounds_check(mask, h, w, area_w_mm, area_h_mm, origin_x, origin_y):
    """
    Check whether the current placement's bounding-box corners extend
    farther from the dish center than the DEFAULT auto-fit placement
    (scale_pct=100, no offset) would for the same image.

    This is deliberately NOT a check against the literal dish radius
    (MAX_AREA_MM / 2): the existing auto-fit only caps the longer image
    dimension to MAX_AREA_MM, so a square or non-circular image's bounding
    box already extends past the inscribed dish circle at its corners by
    design, even with no scale/offset adjustment applied. Warning on that
    baseline case would be a false positive. What scale/offset controls can
    actually make worse is pushing the design farther out than that
    already-accepted baseline, so that's what's compared here.

    Returns (fits, farthest_corner_mm, baseline_farthest_mm).
    """
    baseline_w, baseline_h, baseline_ox, baseline_oy = pixel_center_and_scale(
        mask, h, w, return_legacy=False, scale_pct=100.0, offset_x_mm=0.0, offset_y_mm=0.0,
    )
    baseline_farthest = _farthest_corner_mm(baseline_w, baseline_h, baseline_ox, baseline_oy)
    farthest = _farthest_corner_mm(area_w_mm, area_h_mm, origin_x, origin_y)
    return farthest <= baseline_farthest + 1e-6, farthest, baseline_farthest


def _circle_intersections(p0, p1, cx, cy, r):
    """t-values in [0,1] where segment p0->p1 crosses circle (cx,cy,r), sorted ascending."""
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    fx, fy = p0[0] - cx, p0[1] - cy
    a = dx * dx + dy * dy
    if a == 0:
        return []
    b = 2 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - r * r
    disc = b * b - 4 * a * c
    if disc < 0:
        return []
    sq = math.sqrt(disc)
    t1, t2 = (-b - sq) / (2 * a), (-b + sq) / (2 * a)
    return sorted(t for t in (t1, t2) if 0.0 <= t <= 1.0)


def clip_polyline_to_circle(points, cx, cy, r):
    """
    Clip a polyline (list of (x,y)) to the disk of radius r around (cx,cy).
    Since the disk is convex, any run of points inside it is itself inside,
    so this only needs to insert an interpolated boundary point wherever the
    line crosses in or out, and split into a new sub-path at each exit.
    Returns a list of polylines (0, 1, or more) that lie entirely within the
    disk -- one input polyline can become several if it dips outside and
    back in.
    """
    def inside(p):
        return (p[0] - cx) ** 2 + (p[1] - cy) ** 2 <= r * r + 1e-9

    if not points:
        return []

    result = []
    current = [points[0]] if inside(points[0]) else []
    prev = points[0]
    prev_in = inside(prev)

    for nxt in points[1:]:
        nxt_in = inside(nxt)
        if prev_in and nxt_in:
            current.append(nxt)
        elif prev_in and not nxt_in:
            ts = _circle_intersections(prev, nxt, cx, cy, r)
            if ts:
                t = ts[0]
                current.append((prev[0] + t * (nxt[0] - prev[0]), prev[1] + t * (nxt[1] - prev[1])))
            if len(current) >= 2:
                result.append(current)
            current = []
        elif not prev_in and nxt_in:
            ts = _circle_intersections(prev, nxt, cx, cy, r)
            if ts:
                t = ts[-1]
                current = [(prev[0] + t * (nxt[0] - prev[0]), prev[1] + t * (nxt[1] - prev[1])), nxt]
            else:
                current = [nxt]
        else:
            ts = _circle_intersections(prev, nxt, cx, cy, r)
            if len(ts) == 2:
                t0, t1 = ts
                p_in = (prev[0] + t0 * (nxt[0] - prev[0]), prev[1] + t0 * (nxt[1] - prev[1]))
                p_out = (prev[0] + t1 * (nxt[0] - prev[0]), prev[1] + t1 * (nxt[1] - prev[1]))
                result.append([p_in, p_out])
            current = []
        prev, prev_in = nxt, nxt_in

    if len(current) >= 2:
        result.append(current)
    return result


def clip_paths_to_dish(paths_mm, cx, cy, r, min_len_mm=0.5):
    """
    Clip every path to the physical dish circle so nothing prints past the
    dish edge, dropping any resulting fragment shorter than min_len_mm (tiny
    slivers left over from a near-tangent crossing aren't worth a travel
    move). Returns (clipped_paths, was_clipped, dropped_len_mm) where
    dropped_len_mm is the total path length that fell outside the dish.
    """
    def path_len(pts):
        return sum(math.hypot(pts[i+1][0]-pts[i][0], pts[i+1][1]-pts[i][1]) for i in range(len(pts)-1))

    original_len = sum(path_len(p) for p in paths_mm if len(p) >= 2)

    clipped = []
    for p in paths_mm:
        for piece in clip_polyline_to_circle(p, cx, cy, r):
            if path_len(piece) >= min_len_mm:
                clipped.append(piece)

    clipped_len = sum(path_len(p) for p in clipped)
    dropped_len_mm = max(0.0, original_len - clipped_len)
    was_clipped = dropped_len_mm > 1e-6
    return clipped, was_clipped, dropped_len_mm


def optimize_path_order(all_paths_mm, start_xy=None):
    """
    Simple nearest-neighbor greedy path ordering.
    Each path is a list of (x,y) tuples. We choose the next path by distance
    from the current position to the nearest endpoint among remaining paths.
    Returns a new ordered list of paths (unchanged path directions).
    """
    if not all_paths_mm:
        return []

    remaining = [list(p) for p in all_paths_mm]
    ordered = []
    if start_xy is None:
        cur = remaining[0][0]
        ordered.append(remaining.pop(0))
        cur = ordered[-1][-1]
    else:
        cur = start_xy

    while remaining:
        best_idx = None
        best_dist = float("inf")
        best_choice_reverse = False
        for i, p in enumerate(remaining):
            d0 = math.hypot(p[0][0] - cur[0], p[0][1] - cur[1])  # distance to start
            d1 = math.hypot(p[-1][0] - cur[0], p[-1][1] - cur[1])  # distance to end
            if d0 < best_dist:
                best_dist = d0
                best_idx = i
                best_choice_reverse = False
            if d1 < best_dist:
                best_dist = d1
                best_idx = i
                best_choice_reverse = True
        chosen = remaining.pop(best_idx)
        if best_choice_reverse:
            chosen = list(reversed(chosen))
        ordered.append(chosen)
        cur = ordered[-1][-1]
    return ordered


def generate_gcode(all_contours_mm, print_height_mm=DEFAULT_PRINT_HEIGHT_MM,
                    extrusion_per_mm=DEFAULT_EXTRUSION_PER_MM):
    travel_height_mm = print_height_mm + TRAVEL_CLEARANCE_MM
    # keep the pre-move Z comfortably above the travel height regardless of what
    # print_height_mm is set to, so homing/approach never collides with the agar
    start_z = max(START_Z, travel_height_mm + 0.5)

    lines = [
        "; Continuous line-trace pattern (black regions only)",
        "; Agar-printing workflow: start at dish center and keep the needle near the surface",
        "G21 ; units mm",
        "G90 ; absolute positioning",
        f"G0 Z{start_z:.3f} F6000",
        f"G0 X{DISH_CENTER_X:.3f} Y{DISH_CENTER_Y:.3f} F3000",
        "M83",
        "G92 E0",
        f"G1 Z{print_height_mm:.3f} F600",
    ]

    feed_mm_per_sec = PRINT_FEED_MM_PER_MIN / 60.0
    trail_distance_mm = feed_mm_per_sec * EXTRUDE_TRAIL_SEC

    last_valid_idx = None
    for idx, contour in enumerate(all_contours_mm):
        if len(contour) >= 2:
            last_valid_idx = idx

    for idx, contour in enumerate(all_contours_mm):
        if len(contour) < 2:
            continue
        x0, y0 = contour[0]
        lines.append(f"G0 X{x0:.3f} Y{y0:.3f} F3000")
        lines.append(f"G1 Z{print_height_mm:.3f} F600")
        lines.append("SAVE_GCODE_STATE NAME=line")
        lines.append("M83")

        seg_lens = [((contour[i+1][0]-contour[i][0])**2 + (contour[i+1][1]-contour[i][1])**2) ** 0.5
                    for i in range(len(contour) - 1)]

        is_last_contour = (idx == last_valid_idx)
        if is_last_contour:
            # Only the final printed path gets cut off early -- this is the actual end
            # of the whole print, so extrusion should stop EXTRUDE_TRAIL_SEC before the
            # needle finishes moving, not at the end of every individual path.
            remaining_after = [0.0] * len(seg_lens)
            running = 0.0
            for i in range(len(seg_lens) - 1, -1, -1):
                remaining_after[i] = running
                running += seg_lens[i]
            remaining_before = [remaining_after[i] + seg_lens[i] for i in range(len(seg_lens))]
        else:
            remaining_before = None

        for i, (x, y) in enumerate(contour[1:]):
            seg_len = seg_lens[i]
            if is_last_contour and remaining_before[i] <= trail_distance_mm:
                e_amount = 0.0  # within the last EXTRUDE_TRAIL_SEC of travel before the print ends -- coast
            else:
                e_amount = -extrusion_per_mm * seg_len
            lines.append(f"G1 X{x:.3f} Y{y:.3f} E{e_amount:.4f} F600")
        lines.append(f"G1 E{RETRACT_E:.4f} F300")
        lines.append("RESTORE_GCODE_STATE NAME=line")
        lines.append(f"G1 Z{travel_height_mm:.3f} F600")
        lines.append("G92 E0")

    lines.append("; End of continuous line-trace pattern")
    lines.append(f"G0 Z{travel_height_mm + FINAL_LIFT_MM:.3f} F3000 ; raise clear of the dish now that printing is done")
    return "\n".join(lines)


def process_image(path, trace_mode="both", scale_pct=100.0, offset_x_mm=0.0, offset_y_mm=0.0,
                   print_height_mm=None, extrusion_per_mm=None):
    if print_height_mm is None:
        print_height_mm = DEFAULT_PRINT_HEIGHT_MM
    if extrusion_per_mm is None:
        extrusion_per_mm = DEFAULT_EXTRUSION_PER_MM
    name = os.path.splitext(os.path.basename(path))[0]
    out_dir = os.path.join(os.path.dirname(os.path.abspath(path)), "output")
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n--- {name} ---")

    prepared_path = os.path.join(out_dir, f"{name}_prepared.png")
    _, mask, h, w = prepare_image_for_print(path, prepared_path, trace_mode=trace_mode)
    print(f"prepared image: {prepared_path} ({w}x{h}px working resolution)")

    fg_fraction = mask.sum() / mask.size
    print(f"foreground (dark) fraction: {fg_fraction:.2%}")
    if fg_fraction > 0.9:
        print("WARNING: >90% counted as foreground -- background may also be dark. Check output before printing.")

    # Build hybrid paths in pixel coordinates (row, col) at a reduced resolution.
    pixel_paths = hybrid_paths_from_mask(mask, h, w, trace_mode=trace_mode)
    print(f"{len(pixel_paths)} raw path(s) from hybrid generator")

    area_w_mm, area_h_mm, origin_x, origin_y = pixel_center_and_scale(
        mask, h, w, return_legacy=False,
        scale_pct=scale_pct, offset_x_mm=offset_x_mm, offset_y_mm=offset_y_mm,
    )
    print(f"area_w_mm: {area_w_mm:.3f}, area_h_mm: {area_h_mm:.3f}, origin_x: {origin_x:.3f}, origin_y: {origin_y:.3f}")

    fits_dish, farthest_corner_mm, baseline_farthest_mm = design_bounds_check(
        mask, h, w, area_w_mm, area_h_mm, origin_x, origin_y)
    if not fits_dish:
        print(f"WARNING: this placement's bounding box reaches {farthest_corner_mm:.1f}mm from dish "
              f"center, farther than the {baseline_farthest_mm:.1f}mm the default auto-fit placement "
              f"would reach for this image.")

    # Convert pixel paths to mm coordinates
    all_contours_mm = []
    for p in pixel_paths:
        if p.shape[0] < MIN_CONTOUR_LEN_PX:
            continue
        coords_mm = [contour_to_mm([(r, c)], h, w, area_w_mm, area_h_mm, origin_x, origin_y)[0] for (r, c) in p]
        all_contours_mm.append(coords_mm)

    # Clip to the print area so nothing is asked to print past it -- scale/offset
    # controls (or a generous auto-fit on a non-square image) can push parts of a
    # design past PRINT_AREA_RADIUS_MM even when the bounding-box check above
    # doesn't flag it, since that check only looks at the box corners.
    all_contours_mm, was_clipped, clipped_len_mm = clip_paths_to_dish(
        all_contours_mm, DISH_CENTER_X, DISH_CENTER_Y, PRINT_AREA_RADIUS_MM)
    if was_clipped:
        print(f"WARNING: {clipped_len_mm:.1f}mm of path length fell outside the {PRINT_AREA_RADIUS_MM:.1f}mm "
              f"print area and was clipped or dropped.")

    # Path order optimization (start at dish center to reduce travel)
    start_xy = (DISH_CENTER_X, DISH_CENTER_Y)
    all_contours_mm_ordered = optimize_path_order(all_contours_mm, start_xy=start_xy)
    print(f"{len(all_contours_mm_ordered)} ordered path(s) after optimization")

    # Compute total length & extrusion estimate
    total_len_mm = 0.0
    for c in all_contours_mm_ordered:
        for i in range(len(c) - 1):
            total_len_mm += math.hypot(c[i+1][0] - c[i][0], c[i+1][1] - c[i][1])
    print(f"total traced line length: {total_len_mm:.1f}mm")
    print(f"estimated total extrusion: {total_len_mm * extrusion_per_mm:.2f}mm")

    # Generate gcode
    gcode = generate_gcode(all_contours_mm_ordered, print_height_mm=print_height_mm,
                            extrusion_per_mm=extrusion_per_mm)
    out_gcode_path = os.path.join(out_dir, f"{name}_line_trace.gcode")
    with open(out_gcode_path, "w") as f:
        f.write(gcode)
    print(f"written: {out_gcode_path}")

    # Create preview
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    # Left: source image
    axes[0].imshow(Image.open(path))
    axes[0].set_title(f"Source: {name}")
    axes[0].axis("off")

    # Right: preview in mm coordinates
    ax = axes[1]
    ax.set_aspect("equal")

    # Draw the print area -- the hard boundary paths are now clipped to
    circle = plt.Circle((DISH_CENTER_X, DISH_CENTER_Y), PRINT_AREA_RADIUS_MM, facecolor="none", edgecolor="tab:red", linewidth=1.5, linestyle="-")
    ax.add_patch(circle)

    # Draw the physical dish edge, for reference only (not the clip boundary)
    physical_circle = plt.Circle((DISH_CENTER_X, DISH_CENTER_Y), DISH_RADIUS_MM, facecolor="none", edgecolor="tab:gray", linewidth=1.2, linestyle=":")
    ax.add_patch(physical_circle)

    # Draw origin cross at dish center
    ax.plot([DISH_CENTER_X], [DISH_CENTER_Y], marker="+", color="gray")

    # Prepare line segments (in mm) for plotting
    segs = []
    widths_pts = []
    # convert mm->points: 1 point = 1/72 inch = 25.4/72 mm = 0.352777... mm
    mm_per_point = 25.4 / 72.0
    for contour in all_contours_mm_ordered:
        if len(contour) < 2:
            continue
        seg_list = []
        for (x, y) in contour:
            seg_list.append((x, y))
        segs.append(seg_list)
        # visual line width (mm) from extrusion param (Option A)
        visual_mm = extrusion_per_mm * PREVIEW_WIDTH_SCALE
        width_pt = max(0.1, visual_mm / mm_per_point)
        widths_pts.append(width_pt)

    if segs:
        lc = LineCollection(segs, linewidths=widths_pts, colors="black", linestyles="solid", alpha=0.9)
        ax.add_collection(lc)

    # Optionally also draw skeleton as thin red lines (helpful for debugging)
    # We'll draw a faint red overlay showing centerlines
    sk_color = (1.0, 0.0, 0.0, 0.35)
    sk_segs = []
    for contour in all_contours_mm_ordered:
        if len(contour) < 2:
            continue
        sk_segs.append(contour)
    if sk_segs:
        lc2 = LineCollection(sk_segs, linewidths=[0.5], colors=[sk_color], linestyles="solid")
        ax.add_collection(lc2)

    ax.set_xlim(DISH_CENTER_X - DISH_RADIUS_MM - 5, DISH_CENTER_X + DISH_RADIUS_MM + 5)
    ax.set_ylim(DISH_CENTER_Y - DISH_RADIUS_MM - 5, DISH_CENTER_Y + DISH_RADIUS_MM + 5)
    ax.set_title(f"Preview: {len(all_contours_mm_ordered)} paths; visual scale (mm)")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    preview_path = os.path.join(out_dir, f"{name}_line_trace_preview.png")
    plt.savefig(preview_path, dpi=130)
    plt.close(fig)
    print(f"preview written: {preview_path}")

    return {
        "prepared_path": prepared_path,
        "preview_path": preview_path,
        "gcode_path": out_gcode_path,
        "gcode_text": gcode,
        "n_paths": len(all_contours_mm_ordered),
        "fg_fraction": fg_fraction,
        "total_len_mm": total_len_mm,
        "est_extrusion_mm": total_len_mm * extrusion_per_mm,
        "area_w_mm": area_w_mm,
        "area_h_mm": area_h_mm,
        "working_resolution": (w, h),
        "scale_pct": scale_pct,
        "offset_x_mm": offset_x_mm,
        "offset_y_mm": offset_y_mm,
        "fits_dish": fits_dish,
        "farthest_corner_mm": farthest_corner_mm,
        "baseline_farthest_mm": baseline_farthest_mm,
        "print_height_mm": print_height_mm,
        "extrusion_per_mm": extrusion_per_mm,
        "was_clipped": was_clipped,
        "clipped_len_mm": clipped_len_mm,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate line-trace G-code from an image")
    parser.add_argument("images", nargs="*", help="Image files to process")
    parser.add_argument("--trace-mode", choices=["skeletonize", "edge", "both"], default="both")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    cwd = os.getcwd()

    if args.images:
        images = [os.path.abspath(path) for path in args.images if os.path.exists(os.path.abspath(path))]
        print(f"processing {len(images)} requested image(s)")
    else:
        search_dirs = [script_dir]
        if os.path.abspath(cwd) != os.path.abspath(script_dir):
            search_dirs.append(cwd)

        images = []
        for d in search_dirs:
            images.extend(find_images(d))
        images = sorted(set(images))
        print(f"searched: {search_dirs}")

    if not images:
        print("=" * 60)
        print("NO IMAGES FOUND. Make sure a .png/.jpg/.jpeg is in the")
        print("SAME folder as this script, OR run this script from")
        print("inside the folder that contains the image (e.g. via")
        print("'cd <that folder>' then 'python3 line_trace_gcode.py').")
        print("=" * 60)
    else:
        print(f"found {len(images)} image(s)")
        for img_path in images:
            process_image(img_path, trace_mode=args.trace_mode)