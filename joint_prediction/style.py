"""
Shared style configuration loader.

All visual constants (colors, stroke widths, opacities, outlines) are read
from style_config.json so they can be changed in one place.
"""

import json
import os

_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'style_config.json')


def _hex_to_bgr(h):
    h = h.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


def _hex_to_float(h):
    h = h.lstrip('#')
    return (int(h[0:2], 16) / 255.0, int(h[2:4], 16) / 255.0, int(h[4:6], 16) / 255.0)


def load_config(path=None):
    with open(path or _CONFIG_PATH) as f:
        return json.load(f)


# Load once at import time
_cfg = load_config()

# ── Derived color formats ───────────────────────────────────────────────────
JOINT_COLOR_HEX = _cfg['joint_color']
JOINT_COLOR_BGR = _hex_to_bgr(_cfg['joint_color'])
JOINT_COLOR_FLOAT = _hex_to_float(_cfg['joint_color'])
MESH_COLOR = _hex_to_float(_cfg['mesh_color'])

# ── SVG settings ────────────────────────────────────────────────────────────
SVG_BONE_STROKE_WIDTH = _cfg['svg']['bone_stroke_width']
SVG_BONE_CAP_STYLE = _cfg['svg']['bone_cap_style']
SVG_JOINT_MARKER_SIZE = _cfg['svg']['joint_marker_size']
SVG_JOINT_OUTLINE = _cfg['svg']['joint_outline']
SVG_JOINT_OUTLINE_COLOR = _cfg['svg']['joint_outline_color']
SVG_JOINT_OUTLINE_WIDTH = _cfg['svg']['joint_outline_width']
SVG_OPACITY_MIN = _cfg['svg']['opacity_min']
SVG_OPACITY_MAX = _cfg['svg']['opacity_max']

# ── Image overlay settings ──────────────────────────────────────────────────
IMG_BONE_THICKNESS = _cfg['image_overlay']['bone_thickness']
IMG_JOINT_RADIUS = _cfg['image_overlay']['joint_radius']
IMG_JOINT_OUTLINE = _cfg['image_overlay']['joint_outline']
IMG_JOINT_OUTLINE_COLOR = tuple(_cfg['image_overlay']['joint_outline_color'])
IMG_JOINT_OUTLINE_THICKNESS = _cfg['image_overlay']['joint_outline_thickness']

# ── 3D viewer settings ─────────────────────────────────────────────────────
V3D_BONE_LINE_WIDTH = _cfg['viewer_3d']['bone_line_width']
V3D_JOINT_POINT_SIZE = _cfg['viewer_3d']['joint_point_size']
V3D_JOINT_OUTLINE = _cfg['viewer_3d']['joint_outline']
V3D_JOINT_OUTLINE_COLOR = _cfg['viewer_3d']['joint_outline_color']
V3D_JOINT_OUTLINE_WIDTH = _cfg['viewer_3d']['joint_outline_width']
