import numpy as np
import torch
from scipy.spatial import Voronoi, voronoi_plot_2d, ConvexHull, convex_hull_plot_2d
import open3d as o3d
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from joblib import Parallel, delayed

from util.read_write_fused_vis import read_fused
import point_cloud_utils as pcu

np.random.seed(1234)

def estimate_areas(points, n_neighbors=8):
    knn = NearestNeighbors(n_neighbors=n_neighbors).fit(points)

    indices = knn.kneighbors(return_distance=False)

    def worker(i):
        pca = PCA(2)
        idx = indices[i]
        idx = np.insert(idx, 0, i)

        nbs = points[idx]
        nbs_new = pca.fit_transform(nbs)

        try:
            vor = Voronoi(nbs_new)
            reg = vor.regions[vor.point_region[0]]
            if -1 in reg:
                return 0.
            hull = ConvexHull(vor.vertices[reg])
            area = hull.volume
        except Exception as e:
            return 0.

        return area

    print(f'Estimating areas for {len(points)} points')
    areas = Parallel(32)(delayed(worker)(i) for i in range(len(points)))
    areas = np.array(areas)

    zero_area_indices = np.where(areas == 0)[0]
    for i in zero_area_indices:
        neighbor_indices = indices[i]
        non_zero_area_indices = np.where(areas[neighbor_indices] != 0)[0]
        if len(non_zero_area_indices) == 0:
            continue
        mean_area = np.mean(areas[neighbor_indices[non_zero_area_indices]])
        areas[i] = mean_area

    perc_98 = np.percentile(areas, 98)
    areas = np.clip(areas, 0., perc_98)

    return areas.reshape(-1, 1)


def add_ground_plane(pcd, bounding_sphere=-1, grid_size=0.01, is_dtu=False):
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
    [a, b, c, d] = plane_model

    inlier_cloud = pcd.select_by_index(inliers)
    outlier_cloud = pcd.select_by_index(inliers, invert=True)

    inlier_perc = len(inlier_cloud.points) / len(pcd.points)
    if inlier_perc < 0.15:
        return None, None, None, None

    x_min, x_max = np.min(inlier_cloud.points, axis=0)[0], np.max(inlier_cloud.points, axis=0)[0]
    y_min, y_max = np.min(inlier_cloud.points, axis=0)[1], np.max(inlier_cloud.points, axis=0)[1]
    z_min, z_max = np.min(inlier_cloud.points, axis=0)[2], np.max(inlier_cloud.points, axis=0)[2]

    if x_max - x_min == min(x_max - x_min, y_max - y_min, z_max - z_min):
        y_grid, z_grid = np.meshgrid(np.arange(y_min, y_max, grid_size), np.arange(z_min, z_max, grid_size))
        x_grid = -(b * y_grid + c * z_grid + d) / a
    elif y_max - y_min == min(x_max - x_min, y_max - y_min, z_max - z_min):
        x_grid, z_grid = np.meshgrid(np.arange(x_min, x_max, grid_size), np.arange(z_min, z_max, grid_size))
        y_grid = -(a * x_grid + c * z_grid + d) / b
    else:
        x_grid, y_grid = np.meshgrid(np.arange(x_min, x_max, grid_size), np.arange(y_min, y_max, grid_size))
        z_grid = -(a * x_grid + b * y_grid + d) / c

    points = np.column_stack((x_grid.flatten(), y_grid.flatten(), z_grid.flatten()))

    center = np.mean(inlier_cloud.points, axis=0, keepdims=True)
    normal = np.array([a, b, c]) / np.linalg.norm([a, b, c])

    perc = (np.dot(inlier_cloud.normals, normal) > 0).sum() / len(inlier_cloud.points)
    if perc > 0.5:
        normal = -normal

    outlier_points = np.asarray(outlier_cloud.points)
    perc = (np.dot(outlier_points - center, normal) > 0).sum() / len(outlier_points)

    # not a valid plane (purely heuristic)
    if 0.3 < perc and is_dtu:
        return None, None, None, None

    points = points + 0.05 * normal
    normals = np.tile(normal, (len(points), 1))
    colors = np.zeros_like(points)

    p1 = np.array([x_grid[0, 0], y_grid[0, 0], z_grid[0, 0]])
    p2 = np.array([x_grid[0, 1], y_grid[0, 1], z_grid[0, 1]])
    p3 = np.array([x_grid[1, 0], y_grid[1, 0], z_grid[1, 0]])

    area = np.linalg.norm(np.cross(p2 - p1, p3 - p1))
    areas = np.tile(area, (len(points), 1))

    pcd_points = np.array(pcd.points)
    knn = NearestNeighbors(n_neighbors=1).fit(pcd_points)
    _, indices = knn.kneighbors(points)
    vec = pcd_points[indices].reshape(-1, 3) - points
    dot = np.sum(vec * normal, axis=-1)

    norm = np.linalg.norm(points, axis=-1)
    if is_dtu:
        keep_idx = np.where(np.logical_and(norm < bounding_sphere * 0.85, dot < 0))[0].tolist()
    else:
        proj = dot[:, None] * normal.reshape(-1, 3)
        dist = np.linalg.norm(vec - proj, axis=-1)
        keep_idx = np.where(np.logical_and(np.logical_and(dist < 0.05, dot < 0), norm < bounding_sphere))[0].tolist()

    points = points[keep_idx]
    normals = normals[keep_idx]
    areas = areas[keep_idx]
    colors = colors[keep_idx]

    return points, normals, areas, colors


def load_point_cloud(path='point_clouds/spsr_model.ply', num_points=None, from_colmap=True,
                     bounding_sphere=-1, is_dtu=False):
    if from_colmap:
        mesh_points = read_fused(path, path + '.vis')
        points = np.array([m.position for m in mesh_points], dtype=np.float32)
        normals = np.array([m.normal for m in mesh_points], dtype=np.float32)
        colors = np.array([m.color for m in mesh_points], dtype=np.float32) / 255

        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        point_cloud.normals = o3d.utility.Vector3dVector(normals)
        point_cloud.colors = o3d.utility.Vector3dVector(colors)
    else:
        point_cloud = o3d.io.read_point_cloud(path)
        point_cloud.remove_duplicated_points()
        points = np.array(point_cloud.points)
        if not point_cloud.has_normals():
            point_cloud.estimate_normals()
            point_cloud.orient_normals_consistent_tangent_plane(10)

    print(f'total points: {len(np.array(point_cloud.points))}')

    if bounding_sphere > 0:
        points_norm = np.linalg.norm(points, axis=-1)
        inside_sphere = np.where(points_norm < bounding_sphere)[0].tolist()
        point_cloud = point_cloud.select_by_index(inside_sphere)

    # point_cloud, _ = point_cloud.remove_statistical_outlier(nb_neighbors=10, std_ratio=6.0)
    if num_points is not None:
        downsample_idx = pcu.downsample_point_cloud_poisson_disk(np.array(point_cloud.points), -1, target_num_samples=num_points)
        point_cloud = point_cloud.select_by_index(downsample_idx)
    # point_cloud = point_cloud.voxel_down_sample(voxel_size=0.007)

    point_cloud.normalize_normals()
    normals = np.array(point_cloud.normals)
    points = np.array(point_cloud.points)

    print(f'kept points: {len(points)}')

    if from_colmap:
        colors = np.array(point_cloud.colors)
    else:
        colors = None

    if is_dtu:
        areas = estimate_areas(points, n_neighbors=5)
    else:
        areas = estimate_areas(points, n_neighbors=8)

    point_ground, normal_ground, areas_ground, color_ground =\
        add_ground_plane(point_cloud, bounding_sphere, is_dtu=is_dtu)
    if point_ground is not None:
        points = np.concatenate((points, point_ground), axis=0)
        normals = np.concatenate((normals, normal_ground), axis=0)
        areas = np.concatenate((areas, areas_ground), axis=0)
        if colors is not None:
            colors = np.concatenate((colors, color_ground), axis=0)

    mask = areas[:, 0] != 0
    points = points[mask]
    normals = normals[mask]
    areas = areas[mask]
    if colors is not None:
        colors = colors[mask]

    final_cloud = o3d.geometry.PointCloud()
    final_cloud.points = o3d.utility.Vector3dVector(points)
    final_cloud.normals = o3d.utility.Vector3dVector(normals)
    if colors is not None:
        final_cloud.colors = o3d.utility.Vector3dVector(colors)

    points = torch.tensor(points, dtype=torch.float32)
    normals = torch.tensor(normals, dtype=torch.float32)
    areas = torch.tensor(areas, dtype=torch.float32)
    if colors is not None:
        colors = torch.tensor(colors, dtype=torch.float32)

    return points, normals, areas, colors


# Taken from https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html
def _axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.

    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("letter must be either X, Y or Z.")

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))


def euler_angles_to_matrix(euler_angles: torch.Tensor, convention: str) -> torch.Tensor:
    """
    Convert rotations given as Euler angles in radians to rotation matrices.

    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = [
        _axis_angle_rotation(c, e)
        for c, e in zip(convention, torch.unbind(euler_angles, -1))
    ]
    # return functools.reduce(torch.matmul, matrices)
    return torch.matmul(torch.matmul(matrices[0], matrices[1]), matrices[2])

