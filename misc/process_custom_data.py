import numpy as np
import pycolmap

import os
import sys
import shutil
sys.path.append('.')

from util.read_write_fused_vis import read_fused, write_fused, MeshPoint


def recover_P(intrinsics, pose):
    # Extract R and t from the pose matrix
    K = intrinsics[:3, :3]
    R = pose[:3, :3].transpose()  # Transpose back to get original R
    t = pose[:3, 3]

    P = np.eye(4)
    P[:3, :3] = K @ R
    P[:3, 3] = -K @ R @ t
    
    return P


def convert(colmap_dir, out_dir):
    data_out = dict()

    # changes these manually so that scene is centered at origin
    # preferably bounded within a unit sphere
    scale = 1.0
    dx = 0.0
    dy = 0.0
    dz = 0.0

    S = np.eye(4)
    S[:3, :3] *= scale
    S[0, 3] = dx
    S[1, 3] = dy
    S[2, 3] = dz

    image_dir = f'{colmap_dir}/dense/images'
    reconstruction = pycolmap.Reconstruction(f'{colmap_dir}/dense/sparse')

    os.makedirs(f'{out_dir}/image', exist_ok=True)

    image_idx_map = {}
    for idx, image in reconstruction.images.items():
        image_idx_map[image.name] = idx

    for i, (im, idx) in enumerate(image_idx_map.items()):
        image = reconstruction.images[idx]
        camera = reconstruction.cameras[image.camera_id]

        # assume pinhole camera for now
        focal_x = camera.params[0]
        focal_y = camera.params[1]
        cx = camera.params[2]
        cy = camera.params[3]

        # simple radial model (colmap default)
        # focal_x = focal_y = camera.params[0]
        # cx = camera.params[1]
        # cy = camera.params[2]

        K = np.array([[focal_x, 0, cx],
                      [0, focal_y, cy],
                      [0, 0, 1]])

        pose = image.cam_from_world.inverse().matrix()
        P = recover_P(K, pose)

        data_out[f'world_mat_{i}'] = P
        data_out[f'scale_mat_{i}'] = S

        suffix = im.split('.')[-1]
        shutil.copy(f'{image_dir}/{im}', f'{out_dir}/image/{i:04}.{suffix}')

    np.savez(f'{out_dir}/cameras_sphere.npz', **data_out)

    mesh_points = read_fused(f'{colmap_dir}/dense/points.ply', f'{colmap_dir}/dense/points.ply.vis')
    points = np.array([m.position for m in mesh_points], dtype=np.float32)
    normals = np.array([m.normal for m in mesh_points], dtype=np.float32)
    colors = np.array([m.color for m in mesh_points], dtype=np.float32) / 255

    points = (points - np.array([dx, dy, dz])) / scale
    mesh_points = [MeshPoint(position=points[i], color=colors[i], normal=normals[i], num_visible_images=0, visible_image_idxs=[])
                   for i in range(len(points))]
    write_fused(mesh_points, f'{out_dir}/points.ply', f'{out_dir}/points.ply.vis')


def main():
    colmap_dir = ''  # path to COLMAP reconstruction, should contain 'dense' folder
    out_dir = ''  # output path to write images, camera parameters, and point cloud
    convert(colmap_dir, out_dir)


if __name__ == '__main__':
    main()
