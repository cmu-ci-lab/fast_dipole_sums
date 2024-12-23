import json
import os
import numpy as np
from scipy.spatial.transform import Rotation as R
import sqlite3
import subprocess
import cv2 as cv


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
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose


def process(data_folder, colmap_folder):
    for folder in os.listdir(data_folder):
        camera_dict_path = os.path.join(data_folder, folder, 'cameras_sphere.npz')
        image_path = os.path.join(data_folder, folder, 'image')
        db_path = os.path.join(colmap_folder, folder, 'database.db')
        init_path = os.path.join(colmap_folder, folder, 'manual')
        sparse_path = os.path.join(colmap_folder, folder, 'sparse')
        dense_path = os.path.join(colmap_folder, folder, 'dense')
        sparse_ply_path = os.path.join(colmap_folder, folder, 'sparse', 'points.ply')
        dense_ply_path = os.path.join(colmap_folder, folder, 'dense', 'points.ply')

        os.makedirs(init_path, exist_ok=True)
        os.makedirs(sparse_path, exist_ok=True)

        camera_dict = np.load(camera_dict_path)

        # sparse reconstruction
        print('running feature_extractor')
        subprocess.run(['colmap', 'feature_extractor',
                        '--database_path', db_path,
                        '--image_path', image_path,
                        '--ImageReader.camera_model', 'PINHOLE',
                        '--SiftExtraction.max_num_features', '16384',
                        '--SiftExtraction.estimate_affine_shape', '1',
                        '--SiftExtraction.domain_size_pooling', '1'])

        print('running exhaustive_matcher')
        subprocess.run(['colmap', 'exhaustive_matcher',
                        '--database_path', db_path,
                        '--SiftMatching.guided_matching', '1'])

        database = sqlite3.connect(db_path)
        results = database.execute('SELECT name, image_id FROM images')
        order_map = {name : image_id for name, image_id in results}

        intrinsics_dict = dict()
        pose_dict = dict()

        for i, (k, v) in enumerate(order_map.items()):
            world_mat = camera_dict[f'world_mat_{i}']
            scale_mat = camera_dict[f'scale_mat_{i}']
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = load_K_Rt_from_P(None, P)
            intrinsics_dict[v] = intrinsics
            pose_dict[v] = pose

        sample_image = cv.imread(os.path.join(os.getcwd(), image_path, list(order_map.keys())[0]))
        height, width = sample_image.shape[:2]

        with open(os.path.join(init_path, 'cameras.txt'), 'w') as f:
            for i, (k, v) in enumerate(order_map.items()):
                intrinsics = intrinsics_dict[v]
                cx = intrinsics[0, 2]
                cy = intrinsics[1, 2]
                fx = intrinsics[0, 0]
                fy = intrinsics[1, 1]
                f.write(f'{i + 1} PINHOLE {width} {height} {fx} {fy} {cx} {cy}\n')

        # create empty file
        with open(os.path.join(init_path, 'points3D.txt'), 'w') as f:
            pass

        with open(os.path.join(init_path, 'images.txt'), 'w') as f:
            # sort images according to database order
            for i, (k, v) in enumerate(order_map.items()):
                r = pose_dict[v][:3, :3]
                t = pose_dict[v][:3, -1]

                t = -r.T @ t
                r = R.from_matrix(r.T).as_quat()

                f.write(f'{i + 1} {r[3]} {r[0]} {r[1]} {r[2]} {t[0]} {t[1]} {t[2]} {i + 1} {k}\n\n')

        subprocess.run(['colmap', 'point_triangulator',
                        '--database_path', db_path,
                        '--image_path', image_path,
                        '--input_path', init_path,
                        '--output_path', sparse_path])

        subprocess.run(['colmap', 'model_converter',
                        '--input_path', sparse_path,
                        '--output_path', sparse_ply_path,
                        '--output_type', 'PLY'])

        # dense reconstruction
        subprocess.run(['colmap', 'image_undistorter',
                        '--image_path', image_path,
                        '--input_path', sparse_path,
                        '--output_path', dense_path])

        subprocess.run(['colmap', 'patch_match_stereo',
                        '--workspace_path', dense_path,
                        '--PatchMatchStereo.gpu_index', '0,1'])

        subprocess.run(['colmap', 'stereo_fusion',
                        '--workspace_path', dense_path,
                        '--output_path', dense_ply_path])


if __name__ == '__main__':
    process('public_data/bmvs', 'point_cloud_data/bmvs')
    process('public_data/dtu', 'point_cloud_data/dtu')
