# Joint Prediction Tools

Predict 21 hand joints and MANO meshes from images, then draw overlays and SVGs separately.

## Prerequisites

Get the main WiLoR demo running first (see the root README). These scripts use the same Python environment and pretrained models.

## Workflow

Prediction and drawing are separate steps. Run prediction once, then re-run the drawing scripts as many times as you want to tweak the style.

```
1. predict.py  →  saves .json (joints) + .npz (mesh) data
2. draw_joints.py  →  reads .json, outputs overlay JPG + skeleton SVG
3. draw_mesh.py    →  reads .npz, outputs mesh overlay JPG
```

## Scripts

All scripts run from the **repo root**.

### `predict.py` — Run prediction (slow, GPU)

Place images in `to-predict/`, then:

```bash
python joint_prediction/predict.py
```

Saves raw data to `joint_prediction/to-predict/output/`:

| File | Contents |
|------|----------|
| `*_joints.json` | 21 joints in 3D (model-space) and 2D (image-space), plus bone connectivity |
| `*_mesh.npz` | MANO vertices, camera params, face topology for mesh rendering |

Custom folders:

```bash
python joint_prediction/predict.py --img_folder path/to/images --out_folder path/to/output
```

### `draw_joints.py` — Draw joint overlays + SVGs (fast, no GPU)

```bash
python joint_prediction/draw_joints.py
```

Reads the `*_joints.json` files and original images, outputs:

| File | Contents |
|------|----------|
| `*_skeleton.svg` | Vector skeleton per hand |
| `*_joints_overlay.jpg` | Original image with joints and bones drawn on top |

Custom folders:

```bash
python joint_prediction/draw_joints.py --data_dir path/to/output --img_dir path/to/images
```

### `draw_mesh.py` — Draw mesh overlays (fast, no GPU)

```bash
python joint_prediction/draw_mesh.py
```

Reads the `*_mesh.npz` files and original images, outputs:

| File | Contents |
|------|----------|
| `*_mesh_overlay.jpg` | Original image with MANO mesh rendered on top |

Custom folders:

```bash
python joint_prediction/draw_mesh.py --data_dir path/to/output --img_dir path/to/images
```

### `view_joints.py` — Interactive 3D viewer

Open a predicted JSON to rotate the hand skeleton in 3D:

```bash
python joint_prediction/view_joints.py joint_prediction/to-predict/output/test1_hand0_right_joints.json
```

- **Drag** to rotate
- **Press S** to save an SVG from the current angle (`*_view.svg` next to the JSON)
- Press S multiple times for additional saves (`_view_1.svg`, `_view_2.svg`, ...)

## Configuration — `style_config.json`

All visual styling is in one file. Edit it and re-run the draw scripts — no prediction needed.

### Top-level colors

| Key | What it controls |
|-----|-----------------|
| `joint_color` | Color of joints and bones in all outputs (hex) |
| `mesh_color` | Color of the MANO mesh overlay (hex) |

### `svg` — SVG skeleton output (used by `draw_joints.py` and `view_joints.py`)

| Key | Default | Effect |
|-----|---------|--------|
| `bone_stroke_width` | `5` | Thickness of bone lines |
| `bone_cap_style` | `"projecting"` | Line cap: `"projecting"`, `"round"`, or `"butt"` |
| `joint_marker_size` | `10` | Diameter of joint circles |
| `joint_outline` | `true` | Toggle border ring on joints |
| `joint_outline_color` | `"#000000"` | Border color (hex) |
| `joint_outline_width` | `0.4` | Border thickness |
| `opacity_min` | `0.3` | Opacity of the furthest joint (depth-based) |
| `opacity_max` | `1.0` | Opacity of the nearest joint |

### `image_overlay` — JPG joint overlay (used by `draw_joints.py`)

| Key | Default | Effect |
|-----|---------|--------|
| `bone_thickness` | `4` | Bone line thickness in pixels |
| `joint_radius` | `5` | Joint circle radius in pixels |
| `joint_outline` | `true` | Toggle border ring on joints |
| `joint_outline_color` | `[0,0,0]` | Border color as BGR array |
| `joint_outline_thickness` | `1` | Border thickness in pixels |

### `viewer_3d` — Interactive 3D viewer (used by `view_joints.py`)

| Key | Default | Effect |
|-----|---------|--------|
| `bone_line_width` | `4` | Bone line thickness |
| `joint_point_size` | `60` | Joint scatter point size |
| `joint_outline` | `true` | Toggle border on joints |
| `joint_outline_color` | `"#000000"` | Border color (hex) |
| `joint_outline_width` | `0.4` | Border thickness |

## File structure

```
joint_prediction/
├── README.md
├── style_config.json    ← all visual settings
├── style.py             ← loads config, provides constants to scripts
├── predict.py           ← run prediction (saves .json + .npz)
├── draw_joints.py       ← draw joint overlays + SVGs from .json
├── draw_mesh.py         ← draw mesh overlays from .npz
├── view_joints.py       ← interactive 3D joint viewer
└── to-predict/          ← drop images here
    └── output/          ← prediction data + generated images
```
