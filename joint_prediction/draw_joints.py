"""
Draw joint overlays and SVG skeletons from predicted JSON files.

Re-run this after changing style_config.json — no prediction needed.

Usage:
    python joint_prediction/draw_joints.py
    python joint_prediction/draw_joints.py --data_dir path/to/output --img_dir path/to/images
"""

import os
import sys
import argparse
import json
import glob
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

from style import (
    JOINT_COLOR_HEX, JOINT_COLOR_BGR,
    SVG_BONE_STROKE_WIDTH, SVG_BONE_CAP_STYLE, SVG_JOINT_MARKER_SIZE,
    SVG_JOINT_OUTLINE, SVG_JOINT_OUTLINE_COLOR, SVG_JOINT_OUTLINE_WIDTH,
    SVG_OPACITY_MIN, SVG_OPACITY_MAX,
    IMG_BONE_THICKNESS, IMG_JOINT_RADIUS,
    IMG_JOINT_OUTLINE, IMG_JOINT_OUTLINE_COLOR, IMG_JOINT_OUTLINE_THICKNESS,
)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_joints_json(json_path):
    with open(json_path) as f:
        data = json.load(f)
    joints_3d = np.array([[j['x'], j['y'], j['z']] for j in data['joints_3d']])
    joints_2d = np.array([[j['x'], j['y']] for j in data['joints_2d']])
    bones = [tuple(b) for b in data['bones']]
    return joints_3d, joints_2d, bones


def draw_joints_on_image(img, joints_2d_list, bones):
    overlay = img.copy()
    for joints_2d in joints_2d_list:
        pts = joints_2d.astype(int)
        for a, b in bones:
            cv2.line(overlay, tuple(pts[a]), tuple(pts[b]), JOINT_COLOR_BGR, IMG_BONE_THICKNESS, cv2.LINE_AA)
        for pt in pts:
            cv2.circle(overlay, tuple(pt), IMG_JOINT_RADIUS, JOINT_COLOR_BGR, -1, cv2.LINE_AA)
            if IMG_JOINT_OUTLINE:
                cv2.circle(overlay, tuple(pt), IMG_JOINT_RADIUS, IMG_JOINT_OUTLINE_COLOR, IMG_JOINT_OUTLINE_THICKNESS, cv2.LINE_AA)
    return overlay


def generate_svg(joints_2d, joints_3d, bones, filepath):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')
    ax.axis('off')
    fig.patch.set_alpha(0)
    ax.set_facecolor('none')

    x = joints_2d[:, 0]
    y = joints_2d[:, 1]

    for a, b in bones:
        ax.plot([x[a], x[b]], [y[a], y[b]],
                color=JOINT_COLOR_HEX, linewidth=SVG_BONE_STROKE_WIDTH,
                solid_capstyle=SVG_BONE_CAP_STYLE)

    z_vals = joints_3d[:, 2]
    z_min, z_max = z_vals.min(), z_vals.max()
    z_range = z_max - z_min if z_max > z_min else 1.0
    opacity_range = SVG_OPACITY_MAX - SVG_OPACITY_MIN
    opacities = SVG_OPACITY_MIN + opacity_range * ((z_vals - z_min) / z_range)

    edge_color = SVG_JOINT_OUTLINE_COLOR if SVG_JOINT_OUTLINE else 'none'
    edge_width = SVG_JOINT_OUTLINE_WIDTH if SVG_JOINT_OUTLINE else 0
    for i in range(len(joints_3d)):
        ax.plot(x[i], y[i], 'o', markersize=SVG_JOINT_MARKER_SIZE,
                markerfacecolor=JOINT_COLOR_HEX,
                markeredgecolor=edge_color,
                markeredgewidth=edge_width,
                alpha=float(opacities[i]))

    ax.invert_yaxis()
    margin = 20
    ax.set_xlim(x.min() - margin, x.max() + margin)
    ax.set_ylim(y.max() + margin, y.min() - margin)

    fig.tight_layout(pad=0.5)
    fig.savefig(filepath, format='svg', transparent=True, bbox_inches='tight')
    plt.close(fig)


def find_source_image(img_name, img_dir, extensions=('jpg', 'png', 'jpeg')):
    for ext in extensions:
        path = os.path.join(img_dir, f'{img_name}.{ext}')
        if os.path.exists(path):
            return path
    return None


def main():
    default_data = os.path.join(_SCRIPT_DIR, 'to-predict', 'output')
    default_img = os.path.join(_SCRIPT_DIR, 'to-predict')

    parser = argparse.ArgumentParser(description='Draw joint overlays and SVGs from predicted data')
    parser.add_argument('--data_dir', type=str, default=default_data,
                        help='Folder containing *_joints.json files')
    parser.add_argument('--img_dir', type=str, default=default_img,
                        help='Folder containing original images')
    args = parser.parse_args()

    json_files = sorted(glob.glob(os.path.join(args.data_dir, '*_joints.json')))
    if not json_files:
        print(f'No *_joints.json files found in {args.data_dir}')
        return

    # Group JSONs by source image name (e.g. "test1_hand0_right_joints.json" → "test1")
    grouped = defaultdict(list)
    for jf in json_files:
        base = os.path.basename(jf)
        # Strip _handN_side_joints.json to get original image name
        img_name = base.rsplit('_hand', 1)[0]
        grouped[img_name].append(jf)

    for img_name, hand_jsons in grouped.items():
        print(f'Drawing: {img_name} ({len(hand_jsons)} hand(s))')

        # Per-hand SVGs
        all_joints_2d = []
        bones = None
        for jf in hand_jsons:
            joints_3d, joints_2d, bones = load_joints_json(jf)
            all_joints_2d.append(joints_2d)

            svg_path = jf.replace('_joints.json', '_skeleton.svg')
            generate_svg(joints_2d, joints_3d, bones, svg_path)
            print(f'  Saved SVG:    {svg_path}')

        # Joint overlay on original image
        src_img_path = find_source_image(img_name, args.img_dir)
        if src_img_path is None:
            print(f'  Source image not found for "{img_name}" in {args.img_dir}, skipping overlay.')
            continue

        img_cv2 = cv2.imread(src_img_path)
        overlay = draw_joints_on_image(img_cv2, all_joints_2d, bones)
        overlay_path = os.path.join(args.data_dir, f'{img_name}_joints_overlay.jpg')
        cv2.imwrite(overlay_path, overlay)
        print(f'  Saved overlay: {overlay_path}')

    print('Done.')


if __name__ == '__main__':
    main()
