# Joint Prediction Tools

Predict 21 hand joints from images, visualize them as overlays and SVGs, and interactively rotate the 3D skeleton.

## Prerequisites

Get the main WiLoR demo running first (see the root README). These scripts use the same Python environment and pretrained models.

## Scripts

### `predict.py` — Run prediction on images

Place images in `to-predict/`, then run from the **repo root**:

```bash
python joint_prediction/predict.py
```

Outputs go to `joint_prediction/to-predict/output/`:

| File | Description |
|------|-------------|
| `*_joints.json` | 21 joints as x,y,z in model-space (one per hand) |
| `*_skeleton.svg` | Vector skeleton of the hand (one per hand) |
| `*_mesh_overlay.jpg` | Original image with MANO mesh rendered on top |
| `*_joints_overlay.jpg` | Original image with joint circles and bone lines |

Custom folders:

```bash
python joint_prediction/predict.py --img_folder path/to/images --out_folder path/to/output
```

### `view_joints.py` — Interactive 3D joint viewer

Open a predicted JSON file to rotate the hand skeleton in 3D:

```bash
python joint_prediction/view_joints.py joint_prediction/to-predict/output/test1_hand0_right_joints.json
```

- **Drag** to rotate
- **Press S** to save an SVG from the current angle (saves as `*_view.svg` next to the JSON)
- Press S multiple times for additional saves (`_view_1.svg`, `_view_2.svg`, ...)

Optional second argument for a specific output path:

```bash
python joint_prediction/view_joints.py joints.json my_output.svg
```

## Configuration — `style_config.json`

All visual styling is in one file. Edit it and re-run — no code changes needed.

```jsonc
{
  "joint_color": "#c58bbd",   // Main color for joints and bones (hex)
  "mesh_color": "#4046A8",    // 3D mesh overlay color (hex)
  ...
}
```

### Top-level colors

| Key | What it controls |
|-----|-----------------|
| `joint_color` | Color of joints and bones in all outputs (SVG, image overlay, 3D viewer) |
| `mesh_color` | Color of the MANO mesh in the mesh overlay image |

### `svg` — SVG skeleton output

| Key | Default | What it controls |
|-----|---------|-----------------|
| `bone_stroke_width` | `6` | Thickness of bone lines |
| `bone_cap_style` | `"projecting"` | Line cap style (`"projecting"`, `"round"`, `"butt"`) |
| `joint_marker_size` | `8` | Diameter of joint circles |
| `joint_outline` | `true` | Whether joints have a border ring |
| `joint_outline_color` | `"#000000"` | Border color (hex) |
| `joint_outline_width` | `0.6` | Border thickness |
| `opacity_min` | `0.3` | Opacity for the furthest joint (depth-based) |
| `opacity_max` | `1.0` | Opacity for the nearest joint |

### `image_overlay` — JPG joint overlay

| Key | Default | What it controls |
|-----|---------|-----------------|
| `bone_thickness` | `4` | Bone line thickness in pixels |
| `joint_radius` | `5` | Joint circle radius in pixels |
| `joint_outline` | `true` | Whether joints have a border ring |
| `joint_outline_color` | `[0,0,0]` | Border color as BGR array |
| `joint_outline_thickness` | `1` | Border thickness in pixels |

### `viewer_3d` — Interactive 3D viewer

| Key | Default | What it controls |
|-----|---------|-----------------|
| `bone_line_width` | `5` | Bone line thickness |
| `joint_point_size` | `60` | Joint scatter point size |
| `joint_outline` | `true` | Whether joints have a border |
| `joint_outline_color` | `"#000000"` | Border color (hex) |
| `joint_outline_width` | `0.6` | Border thickness |

## File structure

```
joint_prediction/
├── README.md
├── style_config.json    ← all visual settings
├── style.py             ← loads config, provides constants to scripts
├── predict.py           ← run prediction
├── view_joints.py       ← interactive 3D viewer
└── to-predict/          ← drop images here
    └── output/          ← generated outputs
```
