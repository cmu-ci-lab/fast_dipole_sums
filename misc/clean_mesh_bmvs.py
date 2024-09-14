# adapted from https://github.com/xxlong0/SparseNeuS/blob/main/evaluation/clean_mesh.py

import numpy as np
import cv2 as cv
import os
from glob import glob
from scipy.io import loadmat
import trimesh
import open3d as o3d
import torch
from tqdm import tqdm

import sys
import argparse
import shutil

sys.path.append("../")

BMVS_DIR = 'public_data/bmvs'


def gen_rays_from_single_image(H, W, image, intrinsic, c2w, depth=None, mask=None):
    """
    generate rays in world space, for image image
    :param H:
    :param W:
    :param intrinsics: [3,3]
    :param c2ws: [4,4]
    :return:
    """
    device = image.device
    ys, xs = torch.meshgrid(torch.linspace(0, H - 1, H),
                            torch.linspace(0, W - 1, W))  # pytorch's meshgrid has indexing='ij'
    p = torch.stack([xs, ys, torch.ones_like(ys)], dim=-1)  # H, W, 3

    # normalized ndc uv coordinates, (-1, 1)
    ndc_u = 2 * xs / (W - 1) - 1
    ndc_v = 2 * ys / (H - 1) - 1
    rays_ndc_uv = torch.stack([ndc_u, ndc_v], dim=-1).view(-1, 2).float().to(device)

    intrinsic_inv = torch.inverse(intrinsic)

    p = p.view(-1, 3).float().to(device)  # N_rays, 3
    p = torch.matmul(intrinsic_inv[None, :3, :3], p[:, :, None]).squeeze()  # N_rays, 3
    rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # N_rays, 3
    rays_v = torch.matmul(c2w[None, :3, :3], rays_v[:, :, None]).squeeze()  # N_rays, 3
    rays_o = c2w[None, :3, 3].expand(rays_v.shape)  # N_rays, 3

    image = image.permute(1, 2, 0)
    color = image.view(-1, 3)
    depth = depth.view(-1, 1) if depth is not None else None
    mask = mask.view(-1, 1) if mask is not None else torch.ones([H * W, 1]).to(device)
    sample = {
        'rays_o': rays_o,
        'rays_v': rays_v,
        'rays_ndc_uv': rays_ndc_uv,
        'rays_color': color,
        # 'rays_depth': depth,
        'rays_mask': mask,
        'rays_norm_XYZ_cam': p  # - XYZ_cam, before multiply depth
    }
    if depth is not None:
        sample['rays_depth'] = depth

    return sample


def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()  # ? why need transpose here
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose  # ! return cam2world matrix here


def clean_mesh_faces_outside_frustum(old_mesh_file, new_mesh_file, imgs_idx, H=576, W=768, mask_dilated_size=11,
                                     isolated_face_num=500, keep_largest=True):
    '''Remove faces of mesh which cannot be orserved by all cameras
    '''
    # if path_mask_npz:
    #     path_save_clean = IOUtils.add_file_name_suffix(path_save_clean, '_mask')

    cameras = np.load('{}/{}/cameras_sphere.npz'.format(BMVS_DIR, scan))
    mask_lis = sorted(glob('{}/{}/mask/*.png'.format(BMVS_DIR, scan)))

    mesh = trimesh.load(old_mesh_file)
    intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)

    n_images = len(mask_lis)

    if imgs_idx is None:
        imgs_idx = [i for i in range(n_images)]

    all_indices = []
    chunk_size = 5120
    for i in imgs_idx:
        mask_image = cv.imread(mask_lis[i])
        # kernel_size = mask_dilated_size  # default 101
        # kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel_size, kernel_size))
        # mask_image = cv.dilate(mask_image, kernel, iterations=1)

        P = cameras['world_mat_{}'.format(i)]

        intrinsic, pose = load_K_Rt_from_P(None, P[:3, :])

        rays = gen_rays_from_single_image(H, W, torch.from_numpy(mask_image).permute(2, 0, 1).float(),
                                          torch.from_numpy(intrinsic)[:3, :3].float(),
                                          torch.from_numpy(pose).float())
        rays_o = rays['rays_o']
        rays_d = rays['rays_v']
        rays_mask = rays['rays_color']

        rays_o = rays_o.split(chunk_size)
        rays_d = rays_d.split(chunk_size)
        rays_mask = rays_mask.split(chunk_size)

        for rays_o_batch, rays_d_batch, rays_mask_batch in zip(rays_o, rays_d, rays_mask):
            rays_mask_batch = rays_mask_batch[:, 0] > 128
            rays_o_batch = rays_o_batch[rays_mask_batch]
            rays_d_batch = rays_d_batch[rays_mask_batch]

            idx_faces_hits = intersector.intersects_first(rays_o_batch.cpu().numpy(), rays_d_batch.cpu().numpy())
            all_indices.append(idx_faces_hits)

    values = np.unique(np.concatenate(all_indices, axis=0))
    mask_faces = np.ones(len(mesh.faces))
    mask_faces[values[1:]] = 0
    print(f'Surfaces/Kept: {len(mesh.faces)}/{len(values)}')

    mesh_o3d = o3d.io.read_triangle_mesh(old_mesh_file)
    print("removing triangles by mask")
    mesh_o3d.remove_triangles_by_mask(mask_faces)

    o3d.io.write_triangle_mesh(new_mesh_file, mesh_o3d)

    # # clean meshes
    new_mesh = trimesh.load(new_mesh_file)
    cc = trimesh.graph.connected_components(new_mesh.face_adjacency, min_len=500)
    mask = np.zeros(len(new_mesh.faces), dtype=bool)
    mask[np.concatenate(cc)] = True
    new_mesh.update_faces(mask)
    new_mesh.remove_unreferenced_vertices()
    new_mesh.export(new_mesh_file)

    # meshes = new_mesh.split(only_watertight=False)

    # if not keep_largest:
    #     meshes = [mesh for mesh in meshes if len(mesh.faces) > isolated_face_num]
    #     # new_mesh = meshes[np.argmax([len(mesh.faces) for mesh in meshes])]
    #     merged_mesh = trimesh.util.concatenate(meshes)
    #     merged_mesh.export(new_mesh_file)
    # else:
    #     new_mesh = meshes[np.argmax([len(mesh.faces) for mesh in meshes])]
    #     new_mesh.export(new_mesh_file)

    # o3d.io.write_triangle_mesh(new_mesh_file.replace(".ply", "_raw.ply"), mesh_o3d)
    print("finishing removing triangles")


def clean_outliers(old_mesh_file, new_mesh_file):
    new_mesh = trimesh.load(old_mesh_file)

    meshes = new_mesh.split(only_watertight=False)
    new_mesh = meshes[np.argmax([len(mesh.faces) for mesh in meshes])]

    new_mesh.export(new_mesh_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, default='exp/bmvs/')
    parser.add_argument('--mesh_name', type=str, default='00040000.ply')
    parser.add_argument('--scan', type=str, default='bear')

    args = parser.parse_args()
    base_dir = args.base_dir
    mesh_name = args.mesh_name
    scan = args.scan

    exp_dir = os.path.join(base_dir, f'{scan}')
    old_mesh_file = os.path.join(exp_dir, f'meshes/{mesh_name}')
    clean_mesh_file = os.path.join(exp_dir, f'meshes/clean_{scan}.ply')
    new_mesh_file = os.path.join(exp_dir, f'meshes/new_{scan}.ply')
    shutil.copy(old_mesh_file, os.path.join(exp_dir, f'meshes/old_{scan}.ply'))

    print("processing scan", scan)
    clean_mesh_faces_outside_frustum(old_mesh_file, new_mesh_file, None, mask_dilated_size=0)
    print("finish processing scan", scan)
