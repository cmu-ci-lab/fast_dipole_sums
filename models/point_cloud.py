from typing import Tuple, NamedTuple

from sklearn.decomposition import PCA
import torch
from torch.nn import Module
import torch.nn.functional as F

import open3d as o3d
import trimesh

import numpy as np
from sklearn.neighbors import NearestNeighbors

import interp_appearance_cuda as ap
import interp_geometry_cuda as gm

from util.point_cloud_util import load_point_cloud, estimate_areas

class AxisAlignedBoundingBox(NamedTuple):
    x_range: Tuple[float, float]
    y_range: Tuple[float, float]
    z_range: Tuple[float, float]

INV_SQRT_2 = 1.0 / np.sqrt(2.0)


class PointCloud(Module):
    def __init__(
        self,
        ply_path=None,
        num_points=None,
        from_colmap=True,
        num_features=32,
        beta=2.0,
        bounding_sphere=-1,
        trained_model=False,
        optimize_normals=False,
        use_color_nn=False,
        activation='erf',
        is_dtu=False
    ):
        super().__init__()

        self.optimize_normals = optimize_normals
        self.use_color_nn = use_color_nn
        self.bounding_sphere = bounding_sphere
        self.activation = activation
        self.is_dtu = is_dtu

        if trained_model:
            return

        points, normals, areas, colors = load_point_cloud(ply_path, num_points, from_colmap, bounding_sphere, is_dtu)
        self.init_point_cloud(points, normals, areas, colors, num_features, beta, optimize_normals)


    def init_point_cloud(
        self,
        points,
        normals,
        areas,
        colors,
        num_features,
        beta,
        optimize_normals
    ):
        self.beta = beta

        self.points = points
        self.normals = normals
        self.areas = areas
        self.num_features = num_features
        self.optimize_normals = optimize_normals

        self.s = torch.tensor(1.5)

        self.features_point = torch.randn((self.points.shape[0], self.num_features))

        if colors is not None and not self.use_color_nn:
            self.features_point = self.features_point.reshape((self.points.shape[0], 3, -1))
            self.features_point[:, :, 0] = torch.flip(colors, dims=[1])
            self.features_point = self.features_point.reshape((self.points.shape[0], -1))

        self.w_point = torch.ones((self.points.shape[0], 1))

        self.inv_delta_w = torch.tensor(250.)
        self.inv_delta_f = torch.tensor(250.)

        self.build_octree()
        self.initialize_additional_variables()


    def get_point_cloud(self):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points.cpu().detach().numpy())
        pcd.normals = o3d.utility.Vector3dVector(self.normals.cpu().detach().numpy())
        return pcd


    def visualize_on_plane(self, path, color_network):
        plane_mesh = o3d.io.read_triangle_mesh(path)
        vertices = torch.tensor(np.array(plane_mesh.vertices), dtype=torch.float32, device='cuda')

        w = gm.interp_forward(self.points, vertices.cuda(), self.centers, self.child_nodes,
                                self.pi_flat, self.pi_lengths, self.pi_starts, self.radii,
                                self.wan_point, self.wan_node, self.beta, self.inv_delta_w, 1024)
        f = ap.interp_forward(self.points, vertices.cuda(), self.centers, self.child_nodes,
                                self.pi_flat, self.pi_lengths, self.pi_starts, self.radii,
                                self.fa_point, self.fa_node, self.beta, self.inv_delta_f, 1024)
        dw = gm.interp_pos_grad(self.points, self.normalized_normals, self.areas, vertices.cuda(),
                                self.centers, self.child_nodes, self.pi_flat, self.pi_lengths,
                                self.pi_starts, self.radii, self.wan_point, self.wan_node,
                                self.beta, self.inv_delta_w, 1024)

        dw = dw.reshape(-1, 3)
        dw = torch.nan_to_num(dw, 0.0, 0.0, 0.0)
        normals = -dw / (torch.norm(dw, dim=-1, keepdim=True) + 1e-8)

        dirs = torch.tensor([0, 1, 0], dtype=torch.float32, device='cuda').repeat(len(vertices), 1)
        raw_color = color_network(vertices, normals, dirs, f).reshape(len(vertices), 3)

        occu = torch.sigmoid(torch.exp(self.s) * (w - 1/2)).detach().cpu().numpy()
        w = w.detach().cpu().numpy()
        raw_color = raw_color.detach().cpu().numpy()[:, ::-1]
        raw_color *= occu

        import matplotlib.cm as cm

        occu = cm.magma(occu.reshape(-1))[:, :3]
        perc_95 = np.percentile(w, 95)
        w = cm.coolwarm((w.reshape(-1).clip(-perc_95, perc_95) + perc_95) / 2 / perc_95)[:, :3]

        plane_mesh.vertex_colors = o3d.utility.Vector3dVector(raw_color)
        o3d.io.write_triangle_mesh(path.replace('.ply', '_color.ply'), plane_mesh)

        plane_mesh.vertex_colors = o3d.utility.Vector3dVector(w)
        o3d.io.write_triangle_mesh(path.replace('.ply', '_w.ply'), plane_mesh)

        plane_mesh.vertex_colors = o3d.utility.Vector3dVector(occu)
        o3d.io.write_triangle_mesh(path.replace('.ply', '_occu.ply'), plane_mesh)


    def build_octree(self):
        print('building octree')
        point_indices, child_nodes = gm.build_octree(self.points.cpu())

        print('initializing octree')
        centers, radii = gm.initialize_octree(self.points.cpu(), self.areas.cpu(), point_indices, child_nodes)

        self.points = self.points.cuda()
        self.normals = self.normals.cuda()
        self.areas = self.areas.cuda()
        self.child_nodes = child_nodes.cuda()

        self.centers = centers.cuda()
        self.radii = radii.cuda()

        # stores all points in a node
        self.point_indices = point_indices

        pi_flat = []
        pi_lengths = []
        pi_starts = [0]

        for pi in point_indices:
            pi_flat.extend(pi)
            pi_lengths.append(len(pi))
            pi_starts.append(pi_starts[-1] + len(pi))

        pi_starts = pi_starts[:-1]
        self.pi_flat = torch.tensor(pi_flat, dtype=torch.int32).cuda()
        self.pi_lengths = torch.tensor(pi_lengths, dtype=torch.int32).cuda()
        self.pi_starts = torch.tensor(pi_starts, dtype=torch.int32).cuda()

        # stores all ancestors of a point
        self.node_indices = [[] for _ in range(len(self.points))]
        for i, pi in enumerate(point_indices):
            for p in pi:
                self.node_indices[p].append(i)

        ni_flat = []
        ni_lengths = []
        ni_starts = [0]

        for ni in self.node_indices:
            ni_flat.extend(ni)
            ni_lengths.append(len(ni))
            ni_starts.append(ni_starts[-1] + len(ni))

        ni_starts = ni_starts[:-1]
        self.ni_flat = torch.tensor(ni_flat, dtype=torch.int32).cuda()
        self.ni_lengths = torch.tensor(ni_lengths, dtype=torch.int32).cuda()
        self.ni_starts = torch.tensor(ni_starts, dtype=torch.int32).cuda()


    # these are separated out so loading the model is easier
    def initialize_additional_variables(self):
        if self.optimize_normals:
            self.init_normals = self.normals.clone()

        self.features_point = self.features_point.cuda()
        self.features_point.requires_grad = True

        self.w_point = self.w_point.cuda()
        self.w_point.requires_grad = True

        self.s = self.s.cuda()
        self.s.requires_grad = True

        self.inv_delta_f = self.inv_delta_f.cuda()
        self.inv_delta_f.requires_grad = True

        self.inv_delta_w = self.inv_delta_w.cuda()
        self.inv_delta_w.requires_grad = True

        self.normals = self.normals.cuda()

        if self.optimize_normals:
            self.normals.requires_grad = True
            self.optimizeable_parameters = [
                self.w_point, self.features_point, self.normals,
                self.s, self.inv_delta_w, self.inv_delta_f
            ]
        else:
            self.optimizeable_parameters = [
                self.w_point, self.features_point,
                self.s, self.inv_delta_w, self.inv_delta_f
            ]

        self.aabb = self.setup_bounding_box_planes()

        self.update_node_features()

        # these are necessary for backprop
        self.curr_interpolated_features = []
        self.curr_interpolated_ws = []
        self.curr_queries = []


    def update_node_features(self):
        self.wpos_point = F.softplus(self.w_point, 4)
        self.normalized_normals = self.normals / (torch.norm(self.normals, dim=-1, keepdim=True) + 1e-10)

        self.wan_point, self.wan_node = gm.initialize_features_fan(
            self.wpos_point, self.normalized_normals, self.areas, self.ni_flat, self.ni_lengths, self.ni_starts, len(self.centers))
        self.fa_point, self.fa_node = ap.initialize_features_fa(
            self.features_point, self.areas, self.ni_flat, self.ni_lengths, self.ni_starts, len(self.centers))


    def grow_points(self, candidate_points, view_dirs, thresh=0.01):
        points = self.points.cpu().detach().numpy()

        inside_volume = self.test_inside_volume(candidate_points)[:, 0]
        candidate_points = candidate_points[inside_volume]
        view_dirs = view_dirs[inside_volume]

        candidate_points = candidate_points.cpu().detach().numpy()
        candidate_points, unique_indices = np.unique(candidate_points, return_index=True, axis=0)

        view_dirs = view_dirs[unique_indices].cpu().detach().numpy()
        knn = NearestNeighbors(n_neighbors=5).fit(points)
        neigh_dist, neigh_idx = knn.kneighbors(candidate_points, 5, True)

        valid_indices = np.logical_and(neigh_dist[:, 0] > thresh, neigh_dist[:, 0] < 0.2)

        print(f'grew {valid_indices.sum()} points')

        if valid_indices.sum() == 0:
            return

        if valid_indices.sum() > 2000:
            valid_indices = np.where(valid_indices)[0]
            valid_indices = valid_indices[torch.randperm(len(valid_indices))[:2000].tolist()]

        candidate_points = candidate_points[valid_indices]
        view_dirs = view_dirs[valid_indices]
        neigh_idx = neigh_idx[valid_indices]

        candidate_features = torch.zeros((len(candidate_points), self.features_point.shape[1]))
        candidate_w = torch.zeros((len(candidate_points), self.w_point.shape[1]))
        candidate_points = torch.tensor(candidate_points, dtype=torch.float32)

        dw = gm.interp_pos_grad(
                self.points, self.normalized_normals, self.areas, candidate_points, self.centers,
                self.child_nodes, self.pi_flat, self.pi_lengths, self.pi_starts, self.radii,
                self.wan_point, self.wan_node, self.beta, self.inv_delta_w, 1024)
        dw = dw.reshape(-1, 3)
        dw = torch.nan_to_num(dw, 0.0, 0.0, 0.0)

        candidate_normals = -dw / (torch.norm(dw, dim=-1, keepdim=True) + 1e-7)
        candidate_normals = candidate_normals.cpu().detach().numpy()
        candidate_points = candidate_points.cpu().detach().numpy()

        cos = (candidate_normals * view_dirs).sum(-1)
        candidate_normals[cos < 0] = -candidate_normals[cos < 0]

        for i in range(len(candidate_points)):
            candidate_features[i] = self.features_point[neigh_idx[i]].mean(axis=0)
            candidate_w[i] = self.w_point[neigh_idx[i]].mean(axis=0)

            # use pca normal for new points close to point cloud
            pca = PCA(2)
            if neigh_dist[i].max() < 0.05:
                nbs = points[neigh_idx[i]]
                pca.fit(nbs)
                p = pca.transform(candidate_points[i].reshape(1, -1))
                p = pca.inverse_transform(p)

                normal = np.cross(pca.components_[0], pca.components_[1])
                normal /= np.linalg.norm(normal)
                if np.dot(view_dirs[i], normal) < 0:
                    normal *= -1

                candidate_normals[i] = normal
                candidate_points[i] = p

        candidate_points = torch.tensor(candidate_points, dtype=torch.float32)
        candidate_normals = torch.tensor(candidate_normals, dtype=torch.float32)

        points_new = torch.concat((self.points, candidate_points), dim=0)
        areas_new = torch.tensor(estimate_areas(points_new.cpu().detach().numpy()), dtype=torch.float32)

        if self.optimize_normals:
            self.init_normals = torch.concat((self.init_normals, candidate_normals), dim=0)

        normals_new = torch.concat((self.normals, candidate_normals), dim=0)
        features_point_new = torch.concat((self.features_point, candidate_features), dim=0)
        w_point_new = torch.concat((self.w_point, candidate_w), dim=0)

        if True:
            candidate_points = candidate_points.cpu().detach().numpy()
            candidate_normals = candidate_normals.cpu().detach().numpy()
            pcd = o3d.geometry.PointCloud()
            c1 = np.zeros([len(self.points), 3])
            c2 = np.ones([len(candidate_points), 3])
            c2[:, 0] = 0
            pcd.colors = o3d.utility.Vector3dVector(np.concatenate((c1, c2), axis=0))
            pcd.points = o3d.utility.Vector3dVector(np.concatenate((self.points.cpu().detach().numpy(), candidate_points), axis=0))
            pcd.normals = o3d.utility.Vector3dVector(np.concatenate((self.normalized_normals.cpu().detach().numpy(), candidate_normals), axis=0))
        else:
            pcd = None

        self.points = points_new
        self.normals = normals_new
        self.areas = areas_new
        self.features_point = features_point_new
        self.w_point = w_point_new

        self.features_point.requires_grad = True
        self.w_point.requires_grad = True

        del candidate_points, candidate_features, candidate_normals, candidate_w
        del points_new, normals_new, areas_new, features_point_new, w_point_new
        del self.centers, self.child_nodes, self.radii, self.point_indices, self.pi_flat, self.pi_lengths, self.pi_starts
        del self.fa_node, self.fa_point, self.wan_node, self.wan_point

        self.build_octree()

        if self.optimize_normals:
            self.normals.requires_grad = True
            self.optimizeable_parameters = [
                self.w_point, self.features_point, self.normals,
                self.s, self.inv_delta_w, self.inv_delta_f
            ]
        else:
            self.optimizeable_parameters = [
                self.w_point, self.features_point,
                self.s, self.inv_delta_w, self.inv_delta_f
            ]

        return pcd


    def save_info(self):
        parameters = {
            'beta': self.beta,
            's': self.s,
            'inv_delta_f': self.inv_delta_f,
            'inv_delta_w': self.inv_delta_w,
            'points': self.points,
            'areas': self.areas,
            'normals': self.normals,
            'num_features': self.num_features,
            'features_point': self.features_point,
            'w_point': self.w_point
        }
        return parameters


    def load_model(self, parameters):
        self.beta = parameters['beta']
        self.s = parameters['s']
        self.inv_delta_f = parameters['inv_delta_f']
        self.inv_delta_w = parameters['inv_delta_w']
        self.points = parameters['points']
        self.areas = parameters['areas']
        self.normals = parameters['normals']
        self.num_features = parameters['num_features']
        self.features_point = parameters['features_point']
        self.w_point = parameters['w_point']

        self.build_octree()
        self.initialize_additional_variables()


    def setup_bounding_box_planes(self):
        width_x_range = (self.points[:, 0].min() - 0.1, self.points[:, 0].max() + 0.1)
        depth_y_range = (self.points[:, 1].min() - 0.1, self.points[:, 1].max() + 0.1)
        height_z_range = (self.points[:, 2].min() - 0.1, self.points[:, 2].max() + 0.1)

        if self.bounding_sphere <= 0:
            self.bounding_sphere = max(
                -width_x_range[0], width_x_range[1],
                -depth_y_range[0], depth_y_range[1],
                -height_z_range[0], height_z_range[1],
            )

        # return the computed planes in the packed AABB datastructure:
        return AxisAlignedBoundingBox(
            x_range=width_x_range,
            y_range=depth_y_range,
            z_range=height_z_range,
        )


    def test_inside_volume(self, points):
        inside_aabb = torch.logical_and(
            torch.logical_and(
                torch.logical_and(
                    points[..., 0:1] > self.aabb.x_range[0],
                    points[..., 0:1] < self.aabb.x_range[1],
                ),
                torch.logical_and(
                    points[..., 1:2] > self.aabb.y_range[0],
                    points[..., 1:2] < self.aabb.y_range[1],
                ),
            ),
            torch.logical_and(
                points[..., 2:] > self.aabb.z_range[0],
                points[..., 2:] < self.aabb.z_range[1],
            ),
        )

        if self.bounding_sphere > 0:
            inside_sphere = torch.norm(points, dim=-1, keepdim=True) < self.bounding_sphere
            return torch.logical_and(inside_aabb, inside_sphere)

        return inside_aabb

    def forward(self, queries, is_test=False):
        mask = self.test_inside_volume(queries)[:, 0]
        ws = torch.full((len(queries), 1), -1e10, dtype=torch.float32, device='cuda')
        features = torch.zeros((len(queries), self.num_features), dtype=torch.float32, device='cuda')

        if mask.sum() != 0:
            queries_filtered = queries[mask == 1]

            w = gm.interp_forward(self.points, queries_filtered, self.centers, self.child_nodes,
                                  self.pi_flat, self.pi_lengths, self.pi_starts, self.radii,
                                  self.wan_point, self.wan_node, self.beta, self.inv_delta_w, 1024)
            f = ap.interp_forward(self.points, queries_filtered, self.centers, self.child_nodes,
                                  self.pi_flat, self.pi_lengths, self.pi_starts, self.radii,
                                  self.fa_point, self.fa_node, self.beta, self.inv_delta_f, 1024)
            w = torch.nan_to_num(w, 0.0, 0.0, 0.0)
            f = torch.nan_to_num(f, 0.0, 0.0, 0.0)

            ws[mask == 1] = w
            features[mask == 1] = f

        if not is_test:
            ws.requires_grad = True
            features.requires_grad = True

            self.curr_interpolated_features.append(features)
            self.curr_interpolated_ws.append(ws)
            self.curr_queries.append(queries)

        # some random trick to prevent the scale from blowing up
        s = 6 * self.s / (3 + self.s)
        if self.activation == 'erf':
            interpolated_occupancies = 0.5 + 0.5 * torch.erf((ws - 1/2) * torch.exp(s) * INV_SQRT_2)
        elif self.activation == 'logistic':
            interpolated_occupancies = torch.sigmoid(torch.exp(s) * (ws - 1/2))

        return features, interpolated_occupancies


    def set_gradients(self, tensor, gradients):
        if tensor.grad is not None:
            tensor.grad += gradients
        else:
            tensor.grad = gradients


    def backprop_gradients(self):
        if len(self.curr_queries) == 0:
            return

        interpolated_ws_grad = torch.concat([x.grad for x in self.curr_interpolated_ws], dim=0)
        interpolated_features_grad = torch.concat([x.grad for x in self.curr_interpolated_features], dim=0)
        queries = torch.concat(self.curr_queries, dim=0)

        mask = self.test_inside_volume(queries)[:, 0]

        if mask.sum() == 0:
            return

        queries_filtered = queries[mask == 1]
        interpolated_ws_grad = interpolated_ws_grad[mask == 1]
        interpolated_features_grad = interpolated_features_grad[mask == 1]

        dw_point, dn_point, dinv_delta_w =\
            gm.interp_backward(self.points, self.normalized_normals, self.areas, queries_filtered,
                               self.centers, self.child_nodes, self.pi_flat, self.pi_lengths,
                               self.pi_starts, self.ni_flat, self.ni_lengths, self.ni_starts,
                               self.radii, self.wpos_point, self.wan_point, self.wan_node,
                               interpolated_ws_grad, self.beta, self.inv_delta_w, 1024)
        df_point, dinv_delta_f =\
            ap.interp_backward(self.points, self.areas, queries_filtered,self.centers,
                               self.child_nodes, self.pi_flat, self.pi_lengths,self.pi_starts,
                               self.ni_flat, self.ni_lengths, self.ni_starts, self.radii,
                               self.fa_point, self.fa_node, interpolated_features_grad,
                               self.beta, self.inv_delta_f, 1024)
        dw_point = torch.nan_to_num(dw_point, 0.0, 0.0, 0.0)
        dn_point = torch.nan_to_num(dn_point, 0.0, 0.0, 0.0)
        df_point = torch.nan_to_num(df_point, 0.0, 0.0, 0.0)

        with torch.no_grad():
            self.set_gradients(self.inv_delta_w, dinv_delta_w.reshape(()))
            self.set_gradients(self.inv_delta_f, dinv_delta_f.reshape(()))

            dw_point = dw_point * F.sigmoid(self.w_point * 4)
            self.set_gradients(self.w_point, dw_point)
            self.set_gradients(self.features_point, df_point)

            if self.optimize_normals:
                # don't want to manually compute this
                self.normalized_normals.backward(dn_point, retain_graph=True)

        self.curr_interpolated_ws = []
        self.curr_interpolated_features = []
        self.curr_queries = []


    def dip_sum_gradients(self, queries):
        gradients = torch.zeros((len(queries), 3), dtype=torch.float32, device='cuda')
        mask = self.test_inside_volume(queries)[:, 0]

        if mask.sum() == 0:
            return gradients

        queries_filtered = queries[mask == 1]

        dw = gm.interp_pos_grad(self.points, self.normalized_normals, self.areas, queries_filtered,
                                self.centers, self.child_nodes, self.pi_flat, self.pi_lengths,
                                self.pi_starts, self.radii, self.wan_point, self.wan_node,
                                self.beta, self.inv_delta_w, 1024)
        dw = torch.nan_to_num(dw, 0.0, 0.0, 0.0)
        gradients[mask == 1] = dw.reshape(-1, 3)

        return gradients


    def dip_sum_normals(self, queries):
        gradients = self.dip_sum_gradients(queries)
        normals = -gradients / (torch.norm(gradients, dim=-1, keepdim=True) + 1e-8)
        return normals


    def dip_sum(self, queries):
        mask = self.test_inside_volume(queries)[:, 0]
        ws = torch.full((len(queries), 1), -1e10, dtype=torch.float32, device='cuda')

        if mask.sum() == 0:
            return ws

        queries_filtered = queries[mask == 1]

        w = gm.interp_forward(self.points, queries_filtered, self.centers, self.child_nodes,
                              self.pi_flat, self.pi_lengths, self.pi_starts, self.radii,
                              self.wan_point, self.wan_node, self.beta, self.inv_delta_w, 1024)
        w = torch.nan_to_num(w, 0.0, 0.0, 0.0)

        ws[mask == 1] = w

        return ws
    

    # not actually an sdf, called this for compatibility
    def sdf(self, queries):
        return self.dip_sum(queries) - 1/2
    

    def sdf_normals(self, queries):
        return self.dip_sum_normals(queries)


    def occupancy(self, queries):
        sdf = self.sdf(queries)
        s = 6 * self.s / (3 + self.s)
        if self.activation == 'erf':
            occupancy = 0.5 + 0.5 * torch.erf(sdf * torch.exp(s) * INV_SQRT_2)
        elif self.activation == 'logistic':
            occupancy = torch.sigmoid(torch.exp(s) * sdf)
        return occupancy
