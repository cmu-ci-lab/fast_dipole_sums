# adapted from https://github.com/xxlong0/SparseNeuS/blob/main/evaluation/clean_mesh.py

import numpy as np
import os
import trimesh
import torch

import sys
import argparse

sys.path.append("../")

BMVS_DIR = 'public_data/bmvs'


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, default='exp/bmvs/')
    parser.add_argument('--mesh_name', type=str, default='00040000.ply')
    parser.add_argument('--scan', type=str, default='bear')
    parser.add_argument('--bounding_sphere', type=float, default=1.0)

    args = parser.parse_args()
    base_dir = args.base_dir
    mesh_name = args.mesh_name
    scan = args.scan

    mesh = trimesh.load(os.path.join(base_dir, f'{scan}/meshes/{mesh_name}'))

    cameras = np.load('{}/{}/cameras_sphere.npz'.format(BMVS_DIR, scan))
    P = cameras['scale_mat_0']

    mesh.vertices = (mesh.vertices - P[:3, 3][None]) / P[0, 0]
    points_norm = torch.linalg.norm(torch.tensor(mesh.vertices), ord=2, dim=-1)
    mask = points_norm < args.bounding_sphere
    face_mask = mask[mesh.faces].all(axis=1)
    mesh.update_faces(face_mask)
    mesh.remove_unreferenced_vertices()
    mesh.export(os.path.join(base_dir, f'{scan}/meshes/{mesh_name.replace(".ply", "_scaled.ply")}'))

    mesh_gt = trimesh.load(f'{BMVS_DIR}/{scan}/gt_pts.ply')
    points_norm = torch.linalg.norm(torch.tensor(mesh_gt.vertices), ord=2, dim=-1)
    mask = points_norm < args.bounding_sphere
    mesh_gt.vertices = mesh_gt.vertices[mask]
    mesh_gt.export(f'{BMVS_DIR}/{scan}/gt_pts_pruned.ply')
