import sys
import os

# Add repo root to path so the wilor package can be imported
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _REPO_ROOT)

from pathlib import Path
import torch
import argparse
import cv2
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from wilor.models import WiLoR, load_wilor
from wilor.utils import recursive_to
from wilor.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from wilor.utils.renderer import Renderer, cam_crop_to_full
from ultralytics import YOLO
from style import (
    JOINT_COLOR_HEX, JOINT_COLOR_BGR, MESH_COLOR,
    SVG_BONE_STROKE_WIDTH, SVG_BONE_CAP_STYLE, SVG_JOINT_MARKER_SIZE,
    SVG_JOINT_OUTLINE, SVG_JOINT_OUTLINE_COLOR, SVG_JOINT_OUTLINE_WIDTH,
    SVG_OPACITY_MIN, SVG_OPACITY_MAX,
    IMG_BONE_THICKNESS, IMG_JOINT_RADIUS,
    IMG_JOINT_OUTLINE, IMG_JOINT_OUTLINE_COLOR, IMG_JOINT_OUTLINE_THICKNESS,
)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── 21-joint hand skeleton connectivity ─────────────────────────────────────
BONES = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Knuckle cross-links
    (5, 9), (9, 13), (13, 17),
]

JOINT_NAMES = [
    'Wrist',
    'Thumb_CMC', 'Thumb_MCP', 'Thumb_IP', 'Thumb_Tip',
    'Index_MCP', 'Index_PIP', 'Index_DIP', 'Index_Tip',
    'Middle_MCP', 'Middle_PIP', 'Middle_DIP', 'Middle_Tip',
    'Ring_MCP', 'Ring_PIP', 'Ring_DIP', 'Ring_Tip',
    'Pinky_MCP', 'Pinky_PIP', 'Pinky_DIP', 'Pinky_Tip',
]


def project_full_img(points, cam_trans, focal_length, img_res):
    camera_center = [img_res[0] / 2., img_res[1] / 2.]
    K = torch.eye(3)
    K[0, 0] = focal_length
    K[1, 1] = focal_length
    K[0, 2] = camera_center[0]
    K[1, 2] = camera_center[1]
    points = points + cam_trans
    points = points / points[..., -1:]
    V_2d = (K @ points.T).T
    return V_2d[..., :-1]


def save_joints_json(joints_3d, filepath):
    """Save 21 model-space 3D joints to JSON."""
    data = {
        'num_joints': 21,
        'joint_names': JOINT_NAMES,
        'joints': []
    }
    for i, name in enumerate(JOINT_NAMES):
        data['joints'].append({
            'id': i,
            'name': name,
            'x': float(joints_3d[i, 0]),
            'y': float(joints_3d[i, 1]),
            'z': float(joints_3d[i, 2]),
        })
    data['bones'] = [list(b) for b in BONES]
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def draw_joints_on_image(img, joints_2d_list):
    """Draw 2D joint circles and bone lines on an image for all detected hands."""
    overlay = img.copy()
    for joints_2d in joints_2d_list:
        pts = joints_2d.astype(int)
        for a, b in BONES:
            cv2.line(overlay, tuple(pts[a]), tuple(pts[b]), JOINT_COLOR_BGR, IMG_BONE_THICKNESS, cv2.LINE_AA)
        for pt in pts:
            cv2.circle(overlay, tuple(pt), IMG_JOINT_RADIUS, JOINT_COLOR_BGR, -1, cv2.LINE_AA)
            if IMG_JOINT_OUTLINE:
                cv2.circle(overlay, tuple(pt), IMG_JOINT_RADIUS, IMG_JOINT_OUTLINE_COLOR, IMG_JOINT_OUTLINE_THICKNESS, cv2.LINE_AA)
    return overlay


def generate_svg(joints_2d, joints_3d, filepath):
    """Generate an SVG skeleton matching the example style using matplotlib."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')
    ax.axis('off')
    fig.patch.set_alpha(0)
    ax.set_facecolor('none')

    x = joints_2d[:, 0]
    y = joints_2d[:, 1]

    # Draw bone lines
    for a, b in BONES:
        ax.plot([x[a], x[b]], [y[a], y[b]],
                color=JOINT_COLOR_HEX, linewidth=SVG_BONE_STROKE_WIDTH,
                solid_capstyle=SVG_BONE_CAP_STYLE)

    # Depth-based opacity
    z_vals = joints_3d[:, 2]
    z_min, z_max = z_vals.min(), z_vals.max()
    z_range = z_max - z_min if z_max > z_min else 1.0
    opacity_range = SVG_OPACITY_MAX - SVG_OPACITY_MIN
    opacities = SVG_OPACITY_MIN + opacity_range * ((z_vals - z_min) / z_range)

    # Draw joint circles with depth-based opacity
    edge_color = SVG_JOINT_OUTLINE_COLOR if SVG_JOINT_OUTLINE else 'none'
    edge_width = SVG_JOINT_OUTLINE_WIDTH if SVG_JOINT_OUTLINE else 0
    for i in range(21):
        ax.plot(x[i], y[i], 'o', markersize=SVG_JOINT_MARKER_SIZE,
                markerfacecolor=JOINT_COLOR_HEX,
                markeredgecolor=edge_color,
                markeredgewidth=edge_width,
                alpha=float(opacities[i]))

    # Invert y-axis to match image coordinates
    ax.invert_yaxis()

    # Tight layout with small margins
    margin = 20
    ax.set_xlim(x.min() - margin, x.max() + margin)
    ax.set_ylim(y.max() + margin, y.min() - margin)

    fig.tight_layout(pad=0.5)
    fig.savefig(filepath, format='svg', transparent=True, bbox_inches='tight')
    plt.close(fig)


def main():
    default_img = os.path.join(_SCRIPT_DIR, 'to-predict')
    default_out = os.path.join(_SCRIPT_DIR, 'to-predict', 'output')
    default_model = os.path.join(_REPO_ROOT, 'pretrained_models', 'wilor_final.ckpt')
    default_cfg = os.path.join(_REPO_ROOT, 'pretrained_models', 'model_config.yaml')
    default_detector = os.path.join(_REPO_ROOT, 'pretrained_models', 'detector.pt')

    parser = argparse.ArgumentParser(description='WiLoR single-image joint prediction')
    parser.add_argument('--img_folder', type=str, default=default_img,
                        help='Folder with input images')
    parser.add_argument('--out_folder', type=str, default=default_out,
                        help='Output folder for results')
    parser.add_argument('--rescale_factor', type=float, default=2.0,
                        help='Factor for padding the bbox')
    parser.add_argument('--file_type', nargs='+', default=['*.jpg', '*.png', '*.jpeg'],
                        help='Image file extensions')

    args = parser.parse_args()

    # Load model and detector
    model, model_cfg = load_wilor(
        checkpoint_path=default_model,
        cfg_path=default_cfg,
    )
    detector = YOLO(default_detector)
    renderer = Renderer(model_cfg, faces=model.mano.faces)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    detector = detector.to(device)
    model.eval()

    os.makedirs(args.out_folder, exist_ok=True)

    # Collect images, skip the output subfolder
    out_folder_abs = os.path.abspath(args.out_folder)
    img_paths = []
    for ext in args.file_type:
        for p in Path(args.img_folder).glob(ext):
            if not os.path.abspath(str(p)).startswith(out_folder_abs):
                img_paths.append(p)

    if not img_paths:
        print(f'No images found in {args.img_folder}')
        return

    for img_path in img_paths:
        print(f'Processing: {img_path}')
        img_cv2 = cv2.imread(str(img_path))
        detections = detector(img_cv2, conf=0.3, verbose=False)[0]

        bboxes = []
        is_right = []
        for det in detections:
            bbox = det.boxes.data.cpu().detach().squeeze().numpy()
            is_right.append(det.boxes.cls.cpu().detach().squeeze().item())
            bboxes.append(bbox[:4].tolist())

        if len(bboxes) == 0:
            print(f'  No hands detected, skipping.')
            continue

        boxes = np.stack(bboxes)
        right = np.stack(is_right)
        dataset = ViTDetDataset(model_cfg, img_cv2, boxes, right, rescale_factor=args.rescale_factor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

        all_verts = []
        all_cam_t = []
        all_right = []
        all_joints_3d = []
        all_joints_2d = []

        for batch in dataloader:
            batch = recursive_to(batch, device)

            with torch.no_grad():
                out = model(batch)

            multiplier = (2 * batch['right'] - 1)
            pred_cam = out['pred_cam']
            pred_cam[:, 1] = multiplier * pred_cam[:, 1]
            box_center = batch['box_center'].float()
            box_size = batch['box_size'].float()
            img_size = batch['img_size'].float()
            scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()

            batch_size = batch['img'].shape[0]
            for n in range(batch_size):
                img_fn, _ = os.path.splitext(os.path.basename(img_path))

                verts = out['pred_vertices'][n].detach().cpu().numpy()
                joints = out['pred_keypoints_3d'][n].detach().cpu().numpy()

                hand_is_right = batch['right'][n].cpu().numpy()
                verts[:, 0] = (2 * hand_is_right - 1) * verts[:, 0]
                joints[:, 0] = (2 * hand_is_right - 1) * joints[:, 0]
                cam_t = pred_cam_t_full[n]

                # 2D projections for overlay and SVG
                joints_2d = project_full_img(joints, cam_t, scaled_focal_length, img_size[n]).numpy()

                all_verts.append(verts)
                all_cam_t.append(cam_t)
                all_right.append(hand_is_right)
                all_joints_3d.append(joints)
                all_joints_2d.append(joints_2d)

                # ── Per-hand outputs ──
                hand_idx = len(all_verts) - 1
                side = 'right' if hand_is_right > 0.5 else 'left'

                # 1. Save 3D joint coordinates (model-space)
                json_path = os.path.join(args.out_folder, f'{img_fn}_hand{hand_idx}_{side}_joints.json')
                save_joints_json(joints, json_path)
                print(f'  Saved joints: {json_path}')

                # 2. Save SVG skeleton
                svg_path = os.path.join(args.out_folder, f'{img_fn}_hand{hand_idx}_{side}_skeleton.svg')
                generate_svg(joints_2d, joints, svg_path)
                print(f'  Saved SVG:    {svg_path}')

        if len(all_verts) > 0:
            # 3. Mesh overlay image
            misc_args = dict(
                mesh_base_color=MESH_COLOR,
                scene_bg_color=(1, 1, 1),
                focal_length=scaled_focal_length,
            )
            cam_view = renderer.render_rgba_multiple(
                all_verts, cam_t=all_cam_t, render_res=img_size[n],
                is_right=all_right, **misc_args
            )
            input_img = img_cv2.astype(np.float32)[:, :, ::-1] / 255.0
            input_img = np.concatenate([input_img, np.ones_like(input_img[:, :, :1])], axis=2)
            mesh_overlay = input_img[:, :, :3] * (1 - cam_view[:, :, 3:]) + cam_view[:, :, :3] * cam_view[:, :, 3:]
            mesh_path = os.path.join(args.out_folder, f'{img_fn}_mesh_overlay.jpg')
            cv2.imwrite(mesh_path, 255 * mesh_overlay[:, :, ::-1])
            print(f'  Saved mesh overlay:   {mesh_path}')

            # 4. Joint overlay image
            joints_overlay = draw_joints_on_image(img_cv2, all_joints_2d)
            joints_img_path = os.path.join(args.out_folder, f'{img_fn}_joints_overlay.jpg')
            cv2.imwrite(joints_img_path, joints_overlay)
            print(f'  Saved joints overlay: {joints_img_path}')

    print('Done.')


if __name__ == '__main__':
    main()
