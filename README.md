# Agar Line-Trace Generator — Web UI

A local Flask front end for `line_trace_gcode.py`. Upload an image, pick a
trace mode, get back a preview and a downloadable `.gcode` file sized to your
plate. No processing logic was reimplemented for the web version — it calls
the same tested `process_image()` function the CLI script uses.

## Setup

```
pip install -r requirements.txt --break-system-packages
```
(drop `--break-system-packages` if you're using a virtualenv, which is fine too)

## Run

```
python3 app.py
```

Then open http://127.0.0.1:5000 in a browser. Stop it with Ctrl+C.

## Trace modes

- **both** (hybrid) — outline + skeleton, good general default
- **edge** — traces outlines/borders only
- **skeletonize** — traces centerlines only, best for thin line art

## Notes

- Plate geometry (`DISH_CENTER_X/Y`, `MAX_AREA_MM`), print heights, and
  extrusion rate all live in `line_trace_gcode.py` as module-level constants
  — same place as in the CLI version. Edit there if your dish or Allstruder
  calibration changes.
- Each upload gets its own temp working directory (`/tmp/agar_<id>_...`) so
  concurrent uploads don't collide. Nothing is cleaned up automatically —
  if you're running this for a while, periodically clear old `agar_*` temp
  dirs, or restart the app.
- This is a local single-user tool (`app.run(debug=True)`) — not set up for
  exposing to a network or multiple simultaneous users.
