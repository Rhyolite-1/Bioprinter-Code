#!/usr/bin/env python3
"""
Local web UI for the agar bioprinter line-trace pipeline.

Wraps the tested functions in line_trace_gcode.py (unchanged logic -- this
file only adds upload handling, a trace-mode selector, and HTML rendering).
Run with:  python3 app.py
Then open: http://127.0.0.1:5000
"""

import os
import uuid
import base64
import tempfile
import shutil

from flask import Flask, request, render_template, send_file, abort

import line_trace_gcode as ltg

app = Flask(__name__)

ALLOWED_EXTS = {".png", ".jpg", ".jpeg"}
TRACE_MODES = {
    "both": "Hybrid (outline + skeleton) -- best general-purpose default",
    "edge": "Edge only -- traces outlines/borders of shapes",
    "skeletonize": "Skeleton only -- traces centerlines, good for thin line art",
}

# session_id -> temp directory holding that session's uploaded image + output/
SESSIONS = {}

SCALE_PCT_MIN, SCALE_PCT_MAX = 10.0, 150.0
OFFSET_MM_LIMIT = 40.0  # generous clamp; design_bounds_check catches the real dish-radius violation
PRINT_HEIGHT_MM_MIN, PRINT_HEIGHT_MM_MAX = 0.5, 15.0  # sane physical bounds for a needle Z height
EXTRUSION_PER_MM_MIN, EXTRUSION_PER_MM_MAX = 0.0005, 0.05  # mm^3/mm -- wide enough for recalibration


def _session_dir(session_id):
    d = SESSIONS.get(session_id)
    if d is None or not os.path.isdir(d):
        abort(404, "That result has expired -- please upload the image again.")
    return d


def _find_input_path(session_dir):
    for fn in os.listdir(session_dir):
        if os.path.splitext(fn)[0] == "input":
            return os.path.join(session_dir, fn)
    abort(404, "Original uploaded image not found for this session.")


def _clamped_float(form, name, default, lo, hi):
    raw = form.get(name, "")
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, val))


def _parse_placement_fields(form):
    """
    Parse scale_pct/offset_x_mm/offset_y_mm/print_height_mm/extrusion_per_mm
    from a form, clamping to sane ranges. Falls back to defaults on missing
    or unparseable values rather than erroring -- these are refinement
    controls, not required input.
    """
    scale_pct = _clamped_float(form, "scale_pct", 100.0, SCALE_PCT_MIN, SCALE_PCT_MAX)
    offset_x_mm = _clamped_float(form, "offset_x_mm", 0.0, -OFFSET_MM_LIMIT, OFFSET_MM_LIMIT)
    offset_y_mm = _clamped_float(form, "offset_y_mm", 0.0, -OFFSET_MM_LIMIT, OFFSET_MM_LIMIT)
    print_height_mm = _clamped_float(form, "print_height_mm", ltg.DEFAULT_PRINT_HEIGHT_MM,
                                      PRINT_HEIGHT_MM_MIN, PRINT_HEIGHT_MM_MAX)
    extrusion_per_mm = _clamped_float(form, "extrusion_per_mm", ltg.DEFAULT_EXTRUSION_PER_MM,
                                       EXTRUSION_PER_MM_MIN, EXTRUSION_PER_MM_MAX)
    return scale_pct, offset_x_mm, offset_y_mm, print_height_mm, extrusion_per_mm


def _index_defaults():
    return dict(
        default_print_height_mm=f"{ltg.DEFAULT_PRINT_HEIGHT_MM:.2f}",
        default_extrusion_per_mm=f"{ltg.DEFAULT_EXTRUSION_PER_MM:.4f}",
    )


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", trace_modes=TRACE_MODES, **_index_defaults())


@app.route("/generate", methods=["POST"])
def generate():
    f = request.files.get("image")
    trace_mode = request.form.get("trace_mode", "both")

    if trace_mode not in TRACE_MODES:
        abort(400, "Unknown trace mode.")
    if f is None or f.filename == "":
        return render_template("index.html", trace_modes=TRACE_MODES,
                                error="Please choose an image file.")

    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in ALLOWED_EXTS:
        return render_template("index.html", trace_modes=TRACE_MODES,
                                error=f"Unsupported file type '{ext}'. Use PNG or JPEG.")

    scale_pct, offset_x_mm, offset_y_mm, print_height_mm, extrusion_per_mm = _parse_placement_fields(request.form)

    session_id = uuid.uuid4().hex
    session_dir = tempfile.mkdtemp(prefix=f"agar_{session_id}_")
    SESSIONS[session_id] = session_dir

    input_path = os.path.join(session_dir, f"input{ext}")
    f.save(input_path)

    try:
        result = ltg.process_image(input_path, trace_mode=trace_mode,
                                    scale_pct=scale_pct, offset_x_mm=offset_x_mm, offset_y_mm=offset_y_mm,
                                    print_height_mm=print_height_mm, extrusion_per_mm=extrusion_per_mm)
    except Exception as exc:
        shutil.rmtree(session_dir, ignore_errors=True)
        SESSIONS.pop(session_id, None)
        return render_template("index.html", trace_modes=TRACE_MODES, **_index_defaults(),
                                error=f"Processing failed: {exc}")

    return render_template("result.html", session_id=session_id, trace_mode=trace_mode,
                            trace_modes=TRACE_MODES, **_result_template_args(result))


@app.route("/regenerate/<session_id>", methods=["POST"])
def regenerate(session_id):
    session_dir = _session_dir(session_id)
    input_path = _find_input_path(session_dir)

    trace_mode = request.form.get("trace_mode", "both")
    if trace_mode not in TRACE_MODES:
        abort(400, "Unknown trace mode.")
    scale_pct, offset_x_mm, offset_y_mm, print_height_mm, extrusion_per_mm = _parse_placement_fields(request.form)

    try:
        result = ltg.process_image(input_path, trace_mode=trace_mode,
                                    scale_pct=scale_pct, offset_x_mm=offset_x_mm, offset_y_mm=offset_y_mm,
                                    print_height_mm=print_height_mm, extrusion_per_mm=extrusion_per_mm)
    except Exception as exc:
        return render_template("index.html", trace_modes=TRACE_MODES, **_index_defaults(),
                                error=f"Processing failed: {exc}")

    return render_template("result.html", session_id=session_id, trace_mode=trace_mode,
                            trace_modes=TRACE_MODES, **_result_template_args(result))


def _result_template_args(result):
    """Shared formatting from a process_image() result dict into result.html's template vars."""
    warnings = []
    if result["was_clipped"]:
        warnings.append(f"{result['clipped_len_mm']:.1f}mm of the traced path fell outside the "
                         f"{ltg.PRINT_AREA_RADIUS_MM:.1f}mm print area and was clipped -- part of the design won't "
                         f"print. Reduce scale or offset to fit it all.")
    if not result["fits_dish"]:
        warnings.append(f"This placement reaches {result['farthest_corner_mm']:.1f}mm from dish center -- "
                         f"farther out than the {result['baseline_farthest_mm']:.1f}mm the default auto-fit "
                         f"placement uses for this image.")
    if result["fg_fraction"] > 0.9:
        warnings.append("Over 90% of the image counted as foreground -- the background may also be dark. Check the preview before printing.")
    if result["n_paths"] == 0:
        warnings.append("No printable paths were found. Try a different trace mode, or check that your design is dark-on-light.")

    with open(result["preview_path"], "rb") as fh:
        preview_b64 = base64.b64encode(fh.read()).decode("ascii")

    return dict(
        preview_b64=preview_b64,
        n_paths=result["n_paths"],
        fg_fraction=f"{result['fg_fraction']*100:.2f}",
        total_len_mm=f"{result['total_len_mm']:.1f}",
        est_extrusion_mm=f"{result['est_extrusion_mm']:.2f}",
        area_w_mm=f"{result['area_w_mm']:.1f}",
        area_h_mm=f"{result['area_h_mm']:.1f}",
        working_resolution=f"{result['working_resolution'][0]}x{result['working_resolution'][1]}",
        scale_pct=result["scale_pct"],
        offset_x_mm=result["offset_x_mm"],
        offset_y_mm=result["offset_y_mm"],
        print_height_mm=f"{result['print_height_mm']:.2f}",
        extrusion_per_mm=f"{result['extrusion_per_mm']:.4f}",
        warnings=warnings,
    )


@app.route("/download/<session_id>/gcode")
def download_gcode(session_id):
    d = _session_dir(session_id)
    matches = [fn for fn in os.listdir(os.path.join(d, "output")) if fn.endswith(".gcode")]
    if not matches:
        abort(404)
    return send_file(os.path.join(d, "output", matches[0]), as_attachment=True,
                      download_name="line_trace.gcode")


if __name__ == "__main__":
    app.run(debug=True, port=5000)