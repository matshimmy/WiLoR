"""
Interactive 3D hand joint viewer.

Usage:
    python joint_prediction/view_joints.py path/to/joints.json

Controls:
    - Click and drag to rotate the 3D view
    - Press 'S' to save an SVG from the current viewing angle
    - The SVG is saved next to the JSON file with '_view.svg' suffix
"""

import sys
import os
import json
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d
from style import (
    JOINT_COLOR_HEX,
    SVG_BONE_STROKE_WIDTH, SVG_BONE_CAP_STYLE, SVG_JOINT_MARKER_SIZE,
    SVG_JOINT_OUTLINE, SVG_JOINT_OUTLINE_COLOR, SVG_JOINT_OUTLINE_WIDTH,
    SVG_OPACITY_MIN, SVG_OPACITY_MAX,
    V3D_BONE_LINE_WIDTH, V3D_JOINT_POINT_SIZE,
    V3D_JOINT_OUTLINE, V3D_JOINT_OUTLINE_COLOR, V3D_JOINT_OUTLINE_WIDTH,
)


def load_joints(json_path):
    """Load joints and bones from a prediction JSON file."""
    with open(json_path) as f:
        data = json.load(f)
    joints = np.array([[j['x'], j['y'], j['z']] for j in data['joints_3d']])
    bones = [tuple(b) for b in data['bones']]
    names = [j['name'] for j in data['joints_3d']]
    return joints, bones, names


def project_3d_to_2d(ax, points_3d):
    """Project 3D points to 2D using the current matplotlib 3D axes view."""
    pts_2d = []
    for p in points_3d:
        x2, y2, _ = proj3d.proj_transform(p[0], p[1], p[2], ax.get_proj())
        pts_2d.append([x2, y2])
    return np.array(pts_2d)


def compute_depth_from_view(ax, points_3d):
    """Compute per-point depth relative to the current camera view direction."""
    elev = np.radians(ax.elev)
    azim = np.radians(ax.azim)
    view_dir = np.array([
        np.cos(elev) * np.cos(azim),
        np.cos(elev) * np.sin(azim),
        np.sin(elev),
    ])
    depths = points_3d @ view_dir
    return depths


def save_svg_from_view(ax, joints_3d, bones, output_path):
    """Save an SVG of the hand skeleton from the current 3D viewing angle."""
    pts_2d = project_3d_to_2d(ax, joints_3d)

    depths = compute_depth_from_view(ax, joints_3d)
    d_min, d_max = depths.min(), depths.max()
    d_range = d_max - d_min if d_max > d_min else 1.0
    opacity_range = SVG_OPACITY_MAX - SVG_OPACITY_MIN
    opacities = SVG_OPACITY_MIN + opacity_range * ((depths - d_min) / d_range)

    fig, ax2d = plt.subplots(figsize=(8, 8))
    ax2d.set_aspect('equal')
    ax2d.axis('off')
    fig.patch.set_alpha(0)
    ax2d.set_facecolor('none')

    x = pts_2d[:, 0]
    y = pts_2d[:, 1]

    for a, b in bones:
        ax2d.plot([x[a], x[b]], [y[a], y[b]],
                  color=JOINT_COLOR_HEX, linewidth=SVG_BONE_STROKE_WIDTH,
                  solid_capstyle=SVG_BONE_CAP_STYLE)

    edge_color = SVG_JOINT_OUTLINE_COLOR if SVG_JOINT_OUTLINE else 'none'
    edge_width = SVG_JOINT_OUTLINE_WIDTH if SVG_JOINT_OUTLINE else 0
    for i in range(len(joints_3d)):
        ax2d.plot(x[i], y[i], 'o', markersize=SVG_JOINT_MARKER_SIZE,
                  markerfacecolor=JOINT_COLOR_HEX,
                  markeredgecolor=edge_color,
                  markeredgewidth=edge_width,
                  alpha=float(opacities[i]))

    ax2d.invert_yaxis()

    margin_x = (x.max() - x.min()) * 0.15 + 0.01
    margin_y = (y.max() - y.min()) * 0.15 + 0.01
    ax2d.set_xlim(x.min() - margin_x, x.max() + margin_x)
    ax2d.set_ylim(y.max() + margin_y, y.min() - margin_y)

    fig.tight_layout(pad=0.5)
    fig.savefig(output_path, format='svg', transparent=True, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved SVG: {output_path}')


def main():
    if len(sys.argv) < 2:
        print('Usage: python joint_prediction/view_joints.py <joints.json> [output.svg]')
        sys.exit(1)

    json_path = sys.argv[1]
    if len(sys.argv) >= 3:
        svg_output = sys.argv[2]
    else:
        base, _ = os.path.splitext(json_path)
        svg_output = base + '_view.svg'

    joints, bones, names = load_joints(json_path)

    fig = plt.figure(figsize=(9, 9))
    # Remove matplotlib's default 's' binding (save dialog) so our handler works cleanly
    plt.rcParams['keymap.save'] = []
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    for a, b in bones:
        ax.plot([joints[a, 0], joints[b, 0]],
                [joints[a, 1], joints[b, 1]],
                [joints[a, 2], joints[b, 2]],
                color=JOINT_COLOR_HEX, linewidth=V3D_BONE_LINE_WIDTH)

    edge_color = V3D_JOINT_OUTLINE_COLOR if V3D_JOINT_OUTLINE else 'none'
    edge_width = V3D_JOINT_OUTLINE_WIDTH if V3D_JOINT_OUTLINE else 0
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2],
               c=JOINT_COLOR_HEX, s=V3D_JOINT_POINT_SIZE,
               edgecolors=edge_color, linewidths=edge_width,
               depthshade=True, zorder=5)

    max_range = np.ptp(joints, axis=0).max() / 2
    mid = joints.mean(axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("Rotate freely, press 'S' to save SVG from this angle", fontsize=11)

    save_counter = [0]

    def on_key(event):
        if event.key in ('s', 'S'):
            if save_counter[0] == 0:
                path = svg_output
            else:
                base, ext = os.path.splitext(svg_output)
                path = f'{base}_{save_counter[0]}{ext}'
            save_svg_from_view(ax, joints, bones, path)
            save_counter[0] += 1

    fig.canvas.mpl_connect('key_press_event', on_key)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
