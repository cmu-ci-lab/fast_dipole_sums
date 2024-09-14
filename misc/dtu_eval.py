'''
Eval helper from https://github.com/jzhangbs/DTUeval-python

DTUeval-python

A python implementation of DTU MVS 2014 evaluation. It only takes 1min for each mesh evaluation. And the gap between the two implementations is negligible.

## Setup and Usage

This script requires the following dependencies.

```
numpy open3d scikit-learn tqdm scipy multiprocessing argparse
```

Download the STL point clouds and Sample Set and prepare the ground truth folder as follows.

```
<dataset_dir>
- Points
    - stl
        - stlxxx_total.ply
- ObsMask
    - ObsMaskxxx_10.mat
    - Planexxx.mat
```

Run the evaluation script (e.g. scan24, mesh mode)
```
python eval.py --data <input> --scan 24 --mode mesh --dataset_dir <dataset_dir> --vis_out_dir <out_dir_for_visualization>
```

## Discussion on randomness
There is randomness in point cloud downsampling in both versions. It iterates through the points and delete the points with distance < 0.2. So the order of points matters. We randomly shuffle the points before downsampling.

## Comparison with the official script
We evaluate a set of meshes from Colmap and compare the results. We run our script 10 times and take the average.

|     | diff/official | official | py_avg   | py_std/official |
|-----|---------------|----------|----------|-----------------|
| 24  | 0.0184%       | 0.986317 | 0.986135 | 0.0108%         |
| 37  | 0.0001%       | 2.354124 | 2.354122 | 0.0091%         |
| 40  | 0.0038%       | 0.730464 | 0.730492 | 0.0104%         |
| 55  | 0.0436%       | 0.530899 | 0.531131 | 0.0104%         |
| 63  | 0.0127%       | 1.555828 | 1.556025 | 0.0118%         |
| 65  | 0.0409%       | 1.007686 | 1.008098 | 0.0080%         |
| 69  | 0.0082%       | 0.888434 | 0.888361 | 0.0125%         |
| 83  | 0.0207%       | 1.136882 | 1.137117 | 0.0096%         |
| 97  | 0.0314%       | 0.907528 | 0.907813 | 0.0089%         |
| 105 | 0.0129%       | 1.463337 | 1.463526 | 0.0118%         |
| 106 | 0.1424%       | 0.785527 | 0.786646 | 0.0151%         |
| 110 | 0.0592%       | 1.076125 | 1.075488 | 0.0132%         |
| 114 | 0.0049%       | 0.436169 | 0.436190 | 0.0074%         |
| 118 | 0.1123%       | 0.679574 | 0.680337 | 0.0099%         |
| 122 | 0.0347%       | 0.726771 | 0.726519 | 0.0178%         |
| avg | 0.0153%       | 1.017711 | 1.017867 |                 |


## Error visualization
`vis_xxx_d2s.ply` and `vis_xxx_s2d.ply` are error visualizations.
- Blue: Out of bounding box or ObsMask
- Green: Errors larger than threshold (20)
- White to Red: Errors counted in the reported statistics
'''

import numpy as np
import open3d as o3d
import sklearn.neighbors as skln
from tqdm import tqdm
from scipy.io import loadmat
import multiprocessing as mp
import os
import argparse


def sample_single_tri(input_):
    n1, n2, v1, v2, tri_vert = input_
    c = np.mgrid[:n1+1, :n2+1]
    c += 0.5
    c[0] /= max(n1, 1e-7)
    c[1] /= max(n2, 1e-7)
    c = np.transpose(c, (1,2,0))
    k = c[c.sum(axis=-1) < 1]  # m2
    q = v1 * k[:,:1] + v2 * k[:,1:] + tri_vert
    return q

def write_vis_pcd(file, points, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(file, pcd)

def eval_data(args):
    mp.freeze_support()

    thresh = args.downsample_density
    if args.mode == 'mesh':
        pbar = tqdm(total=9)
        pbar.set_description('read data mesh')
        data_mesh = o3d.io.read_triangle_mesh(args.data)

        vertices = np.asarray(data_mesh.vertices)
        triangles = np.asarray(data_mesh.triangles)
        tri_vert = vertices[triangles]

        pbar.update(1)
        pbar.set_description('sample pcd from mesh')
        v1 = tri_vert[:,1] - tri_vert[:,0]
        v2 = tri_vert[:,2] - tri_vert[:,0]
        l1 = np.linalg.norm(v1, axis=-1, keepdims=True)
        l2 = np.linalg.norm(v2, axis=-1, keepdims=True)
        area2 = np.linalg.norm(np.cross(v1, v2), axis=-1, keepdims=True)
        non_zero_area = (area2 > 0)[:,0]
        l1, l2, area2, v1, v2, tri_vert = [
            arr[non_zero_area] for arr in [l1, l2, area2, v1, v2, tri_vert]
        ]
        thr = thresh * np.sqrt(l1 * l2 / area2)
        n1 = np.floor(l1 / thr)
        n2 = np.floor(l2 / thr)

        with mp.Pool() as mp_pool:
            new_pts = mp_pool.map(sample_single_tri, ((n1[i,0], n2[i,0], v1[i:i+1], v2[i:i+1], tri_vert[i:i+1,0]) for i in range(len(n1))), chunksize=1024)

        new_pts = np.concatenate(new_pts, axis=0)
        data_pcd = np.concatenate([vertices, new_pts], axis=0)

    elif args.mode == 'pcd':
        pbar = tqdm(total=8)
        pbar.set_description('read data pcd')
        data_pcd_o3d = o3d.io.read_point_cloud(args.data)
        data_pcd = np.asarray(data_pcd_o3d.points)

    pbar.update(1)
    pbar.set_description('random shuffle pcd index')
    shuffle_rng = np.random.default_rng()
    shuffle_rng.shuffle(data_pcd, axis=0)

    pbar.update(1)
    pbar.set_description('downsample pcd')
    nn_engine = skln.NearestNeighbors(n_neighbors=1, radius=thresh, algorithm='kd_tree', n_jobs=-1)
    nn_engine.fit(data_pcd)
    rnn_idxs = nn_engine.radius_neighbors(data_pcd, radius=thresh, return_distance=False)
    mask = np.ones(data_pcd.shape[0], dtype=np.bool_)
    for curr, idxs in enumerate(rnn_idxs):
        if mask[curr]:
            mask[idxs] = 0
            mask[curr] = 1
    data_down = data_pcd[mask]

    pbar.update(1)
    pbar.set_description('masking data pcd')
    obs_mask_file = loadmat(f'{args.dataset_dir}/ObsMask/ObsMask{args.scan}_10.mat')
    ObsMask, BB, Res = [obs_mask_file[attr] for attr in ['ObsMask', 'BB', 'Res']]
    BB = BB.astype(np.float32)

    patch = args.patch_size
    inbound = ((data_down >= BB[:1]-patch) & (data_down < BB[1:]+patch*2)).sum(axis=-1) ==3
    data_in = data_down[inbound]

    data_grid = np.around((data_in - BB[:1]) / Res).astype(np.int32)
    grid_inbound = ((data_grid >= 0) & (data_grid < np.expand_dims(ObsMask.shape, 0))).sum(axis=-1) ==3
    data_grid_in = data_grid[grid_inbound]
    in_obs = ObsMask[data_grid_in[:,0], data_grid_in[:,1], data_grid_in[:,2]].astype(np.bool_)
    data_in_obs = data_in[grid_inbound][in_obs]

    pbar.update(1)
    pbar.set_description('read STL pcd')
    stl_pcd = o3d.io.read_point_cloud(f'{args.dataset_dir}/Points/stl/stl{args.scan:03}_total.ply')
    stl = np.asarray(stl_pcd.points)

    pbar.update(1)
    pbar.set_description('compute data2stl')
    nn_engine.fit(stl)
    dist_d2s, idx_d2s = nn_engine.kneighbors(data_in_obs, n_neighbors=1, return_distance=True)
    max_dist = args.max_dist
    mean_d2s = dist_d2s[dist_d2s < max_dist].mean()

    pbar.update(1)
    pbar.set_description('compute stl2data')
    ground_plane = loadmat(f'{args.dataset_dir}/ObsMask/Plane{args.scan}.mat')['P']

    stl_hom = np.concatenate([stl, np.ones_like(stl[:,:1])], -1)
    above = (ground_plane.reshape((1,4)) * stl_hom).sum(-1) > 0
    stl_above = stl[above]

    nn_engine.fit(data_in)
    dist_s2d, idx_s2d = nn_engine.kneighbors(stl_above, n_neighbors=1, return_distance=True)
    mean_s2d = dist_s2d[dist_s2d < max_dist].mean()

    pbar.update(1)
    pbar.set_description('visualize error')
    vis_dist = args.visualize_threshold
    R = np.array([[1,0,0]], dtype=np.float64)
    G = np.array([[0,1,0]], dtype=np.float64)
    B = np.array([[0,0,1]], dtype=np.float64)
    W = np.array([[1,1,1]], dtype=np.float64)
    data_color = np.tile(B, (data_down.shape[0], 1))
    data_alpha = dist_d2s.clip(max=vis_dist) / vis_dist
    data_color[ np.where(inbound)[0][grid_inbound][in_obs] ] = R * data_alpha + W * (1-data_alpha)
    data_color[ np.where(inbound)[0][grid_inbound][in_obs][dist_d2s[:,0] >= max_dist] ] = G
    write_vis_pcd(f'{args.vis_out_dir}/vis_{args.scan:03}_d2s.ply', data_down, data_color)
    stl_color = np.tile(B, (stl.shape[0], 1))
    stl_alpha = dist_s2d.clip(max=vis_dist) / vis_dist
    stl_color[ np.where(above)[0] ] = R * stl_alpha + W * (1-stl_alpha)
    stl_color[ np.where(above)[0][dist_s2d[:,0] >= max_dist] ] = G
    write_vis_pcd(f'{args.vis_out_dir}/vis_{args.scan:03}_s2d.ply', stl, stl_color)

    pbar.update(1)
    pbar.set_description('done')
    pbar.close()
    over_all = (mean_d2s + mean_s2d) / 2

    return over_all



def eval_mesh(mesh,
              scan,
              base_dir = './exp',
              dataset_dir = './dtu_eval',
              downsample_density = 0.2,
              patch_size = 60,
              max_dist = 20,
              visualize_threshold = 10,
              num_iters = 1):
    mesh_path = os.path.join(base_dir, mesh)
    visual_dir = os.path.join(base_dir, 'eval_visuals')

    os.makedirs(visual_dir, exist_ok=True)

    # evaluate a single mesh, then return scores array
    args = argparse.Namespace(
        data=mesh_path,
        scan=int(scan),
        mode='mesh',
        dataset_dir=dataset_dir,
        vis_out_dir=visual_dir,
        downsample_density=downsample_density,
        patch_size=patch_size,
        max_dist=max_dist,
        visualize_threshold=visualize_threshold
    )
    scores = []
    for _ in range(num_iters):
        overall_dist = eval_data(args)
        scores.append(overall_dist)

    return scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, default='exp/dtu/')
    parser.add_argument('--scan', type=int, default=24)
    parser.add_argument('--mesh_name', type=str, default='new_24.ply')

    args = parser.parse_args()
    base_dir = args.base_dir
    scan = args.scan
    mesh_name = args.mesh_name

    exp_dir = os.path.join(base_dir, f'dtu_scan{scan}')
    mesh_path = f'meshes/{mesh_name}'

    print(f'evaluating scan {scan}: ')
    print(eval_mesh(mesh_path, f'{scan}', exp_dir, 'dtu_eval_data'))
