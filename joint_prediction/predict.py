"""
Run WiLoR prediction on images. Saves raw data only (no drawing).

Outputs per hand:
    *_joints.json  — 21 joints (3D model-space + 2D image-space) + bones
    *_mesh.npz     — MANO vertices, camera params, faces for mesh rendering

Usage:
    python joint_prediction/predict.py
"""

import sys
import os

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _REPO_ROOT)

from pathlib import Path
import torch
import argparse
import cv2
import numpy as np
import json

from wilor.models import load_wilor
from wilor.utils import recursive_to
from wilor.datasets.vitdet_dataset import ViTDetDataset
from wilor.utils.renderer import cam_crop_to_full
from ultralytics import YOLO

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
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


def save_joints_json(joints_3d, joints_2d, filepath):
    """Save 21 joints (3D model-space + 2D image-space) to JSON."""
    data = {
        'num_joints': 21,
        'joint_names': JOINT_NAMES,
        'joints_3d': [],
        'joints_2d': [],
        'bones': [list(b) for b in BONES],
    }
    for i, name in enumerate(JOINT_NAMES):
        data['joints_3d'].append({
            'id': i, 'name': name,
            'x': float(joints_3d[i, 0]),
            'y': float(joints_3d[i, 1]),
            'z': float(joints_3d[i, 2]),
        })
        data['joints_2d'].append({
            'id': i, 'name': name,
            'x': float(joints_2d[i, 0]),
            'y': float(joints_2d[i, 1]),
        })
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def save_mesh_npz(verts, cam_t, is_right, scaled_focal_length, img_size, faces, filepath):
    """Save MANO mesh data needed for rendering."""
    np.savez(filepath,
             vertices=verts,
             cam_t=cam_t,
             is_right=is_right,
             scaled_focal_length=float(scaled_focal_length),
             img_size=np.array(img_size),
             faces=faces)


def main():
    default_img = os.path.join(_SCRIPT_DIR, 'to-predict')
    default_out = os.path.join(_SCRIPT_DIR, 'to-predict', 'output')
    default_model = os.path.join(_REPO_ROOT, 'pretrained_models', 'wilor_final.ckpt')
    default_cfg = os.path.join(_REPO_ROOT, 'pretrained_models', 'model_config.yaml')
    default_detector = os.path.join(_REPO_ROOT, 'pretrained_models', 'detector.pt')

    parser = argparse.ArgumentParser(description='WiLoR joint + mesh prediction (data only)')
    parser.add_argument('--img_folder', type=str, default=default_img)
    parser.add_argument('--out_folder', type=str, default=default_out)
    parser.add_argument('--rescale_factor', type=float, default=2.0)
    parser.add_argument('--file_type', nargs='+', default=['*.jpg', '*.png', '*.jpeg'])
    args = parser.parse_args()

    model, model_cfg = load_wilor(checkpoint_path=default_model, cfg_path=default_cfg)
    detector = YOLO(default_detector)
    faces = model.mano.faces.copy()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    detector = detector.to(device)
    model.eval()

    os.makedirs(args.out_folder, exist_ok=True)

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

        hand_idx = 0
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

                joints_2d = project_full_img(joints, cam_t, scaled_focal_length, img_size[n]).numpy()

                side = 'right' if hand_is_right > 0.5 else 'left'

                # Save joints JSON (3D + 2D)
                json_path = os.path.join(args.out_folder, f'{img_fn}_hand{hand_idx}_{side}_joints.json')
                save_joints_json(joints, joints_2d, json_path)
                print(f'  Saved joints: {json_path}')

                # Save mesh data
                npz_path = os.path.join(args.out_folder, f'{img_fn}_hand{hand_idx}_{side}_mesh.npz')
                save_mesh_npz(verts, cam_t, hand_is_right, scaled_focal_length, img_size[n].cpu().numpy(), faces, npz_path)
                print(f'  Saved mesh:   {npz_path}')

                hand_idx += 1

    print('Done.')


if __name__ == '__main__':
    main()
