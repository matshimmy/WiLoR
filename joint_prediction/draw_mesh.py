"""
Draw MANO mesh overlays from predicted .npz files.

Re-run this after changing mesh_color in style_config.json — no prediction needed.

Usage:
    python joint_prediction/draw_mesh.py
    python joint_prediction/draw_mesh.py --data_dir path/to/output --img_dir path/to/images
"""

import sys
import os

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _REPO_ROOT)

import argparse
import glob
import cv2
import numpy as np
from collections import defaultdict

from wilor.utils.renderer import Renderer
from style import MESH_COLOR

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


class _MinimalCfg:
    """Minimal config to satisfy Renderer.__init__."""
    def __init__(self, focal_length, image_size):
        self.EXTRA = type('', (), {'FOCAL_LENGTH': focal_length})()
        self.MODEL = type('', (), {'IMAGE_SIZE': image_size})()


def load_mesh_npz(path):
    d = np.load(path, allow_pickle=False)
    return {
        'vertices': d['vertices'],
        'cam_t': d['cam_t'],
        'is_right': d['is_right'],
        'scaled_focal_length': float(d['scaled_focal_length']),
        'img_size': d['img_size'],
        'faces': d['faces'],
    }


def find_source_image(img_name, img_dir, extensions=('jpg', 'png', 'jpeg')):
    for ext in extensions:
        path = os.path.join(img_dir, f'{img_name}.{ext}')
        if os.path.exists(path):
            return path
    return None


def main():
    default_data = os.path.join(_SCRIPT_DIR, 'to-predict', 'output')
    default_img = os.path.join(_SCRIPT_DIR, 'to-predict')

    parser = argparse.ArgumentParser(description='Draw MANO mesh overlays from predicted data')
    parser.add_argument('--data_dir', type=str, default=default_data,
                        help='Folder containing *_mesh.npz files')
    parser.add_argument('--img_dir', type=str, default=default_img,
                        help='Folder containing original images')
    args = parser.parse_args()

    npz_files = sorted(glob.glob(os.path.join(args.data_dir, '*_mesh.npz')))
    if not npz_files:
        print(f'No *_mesh.npz files found in {args.data_dir}')
        return

    # Group by source image name
    grouped = defaultdict(list)
    for nf in npz_files:
        base = os.path.basename(nf)
        img_name = base.rsplit('_hand', 1)[0]
        grouped[img_name].append(nf)

    renderer = None

    for img_name, hand_npzs in grouped.items():
        print(f'Drawing mesh: {img_name} ({len(hand_npzs)} hand(s))')

        src_img_path = find_source_image(img_name, args.img_dir)
        if src_img_path is None:
            print(f'  Source image not found for "{img_name}" in {args.img_dir}, skipping.')
            continue

        img_cv2 = cv2.imread(src_img_path)

        all_verts = []
        all_cam_t = []
        all_right = []
        scaled_focal_length = None

        for nf in hand_npzs:
            data = load_mesh_npz(nf)
            all_verts.append(data['vertices'])
            all_cam_t.append(data['cam_t'])
            all_right.append(data['is_right'])
            scaled_focal_length = data['scaled_focal_length']
            img_size = data['img_size']
            faces = data['faces']

        # Create renderer once (faces are always the same)
        if renderer is None:
            cfg = _MinimalCfg(focal_length=scaled_focal_length, image_size=256)
            renderer = Renderer(cfg, faces=faces)

        cam_view = renderer.render_rgba_multiple(
            all_verts, cam_t=all_cam_t, render_res=img_size,
            is_right=all_right,
            mesh_base_color=MESH_COLOR,
            scene_bg_color=(1, 1, 1),
            focal_length=scaled_focal_length,
        )

        input_img = img_cv2.astype(np.float32)[:, :, ::-1] / 255.0
        input_img = np.concatenate([input_img, np.ones_like(input_img[:, :, :1])], axis=2)
        mesh_overlay = input_img[:, :, :3] * (1 - cam_view[:, :, 3:]) + cam_view[:, :, :3] * cam_view[:, :, 3:]

        mesh_path = os.path.join(args.data_dir, f'{img_name}_mesh_overlay.jpg')
        cv2.imwrite(mesh_path, 255 * mesh_overlay[:, :, ::-1])
        print(f'  Saved: {mesh_path}')

    print('Done.')


if __name__ == '__main__':
    main()
