import torch
import torch.nn.functional as F


class Renderer:
    def __init__(self,
                 nerf,
                 color_network,
                 point_cloud,
                 point_sampler,
                 n_total_samples,
                 isect_sampling):
        self.nerf = nerf
        self.color_network = color_network
        self.point_cloud = point_cloud
        self.isect_sampler = point_sampler
        self.n_total_samples = n_total_samples
        self.isect_sampling = isect_sampling


    def render_core_outside(self, rays_o, rays_d, z_vals, sample_dist, nerf, background_rgb=None):
        """
        Render background
        """
        batch_size, n_samples = z_vals.shape

        # section length
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.Tensor([sample_dist]).expand(dists[..., :1].shape)], -1)
        mid_z_vals = z_vals + dists * 0.5

        # section midpoints
        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # batch_size, n_samples, 3

        dis_to_center = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).clip(1.0, 1e10)
        pts = torch.cat([pts / dis_to_center, 1.0 / dis_to_center], dim=-1)       # batch_size, n_samples, 4

        dirs = rays_d[:, None, :].expand(batch_size, n_samples, 3)

        pts = pts.reshape(-1, 4)
        dirs = dirs.reshape(-1, 3)

        # query neural fields
        density, sampled_color = nerf(pts, dirs)
        sampled_color = torch.sigmoid(sampled_color)

        density[:, :-1] = (density[:, :-1] + density[:, 1:]) / 2

        # compute alpha
        alpha = 1.0 - torch.exp(-F.softplus(density.reshape(batch_size, n_samples), beta=4) * dists)
        alpha = alpha.reshape(batch_size, n_samples)

        # aggregate along rays
        weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]
        sampled_color = sampled_color.reshape(batch_size, n_samples, 3)
        color = (weights[:, :, None] * sampled_color).sum(dim=1)
        if background_rgb is not None:
            color = color + background_rgb * (1.0 - weights.sum(dim=-1, keepdim=True))

        return {
            'color': color,
            'sampled_color': sampled_color,
            'alpha': alpha,
            'weights': weights,
        }


    def accumulate(self,
                   raw_color,
                   batch_size,
                   n_samples,
                   inside_sphere,
                   inside_volume,
                   weights,
                   weights_sum,
                   background_alpha,
                   background_sampled_color,
                   background_rgb):
        sampled_color = raw_color.reshape(batch_size, n_samples, 3)
        sampled_color = sampled_color * inside_volume[:, :, None]

        # aggregate along rays
        if background_alpha is not None:
            sampled_color = sampled_color * inside_sphere[:, :, None] +\
                            background_sampled_color[:, :n_samples] * (1.0 - inside_sphere)[:, :, None]
            sampled_color = torch.cat([sampled_color, background_sampled_color[:, n_samples:]], dim=1)

        color = (sampled_color * weights[:, :, None]).sum(dim=1)
        if background_rgb is not None:    # Fixed background, usually black
            color = color + background_rgb * (1.0 - weights_sum)

        return color


    def render_core(self,
                    rays_o,
                    rays_d,
                    z_vals,
                    background_alpha=None,
                    background_sampled_color=None,
                    background_rgb=None,
                    is_test=False):
        batch_size, n_samples = z_vals.shape

        pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]  # n_rays, n_samples, 3
        dirs = rays_d[:, None, :].expand(pts.shape)

        pts = pts.reshape(-1, 3)
        dirs = dirs.reshape(-1, 3)

        features, occupancy = self.point_cloud(pts, is_test=is_test)
        normals = self.point_cloud.sdf_normals(pts)

        occupancy = occupancy.reshape(batch_size, n_samples)
        alpha = torch.zeros_like(occupancy)
        alpha[:, 1:] = 1 - (1 - torch.maximum(occupancy[:, :-1], occupancy[:, 1:])) / (1 - torch.minimum(occupancy[:, :-1], occupancy[:, 1:]) + 1e-7)

        pts_norm = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).reshape(batch_size, n_samples)

        # both endpoints inside sphere
        inside_sphere = (pts_norm < self.point_cloud.bounding_sphere).float().detach()
        inside_sphere[:, 1:] = inside_sphere[:, 1:] * inside_sphere[:, :-1]

        inside_volume = self.point_cloud.test_inside_volume(pts).float().detach().reshape(batch_size, n_samples)
        inside_volume[:, 1:] = inside_volume[:, 1:] * inside_volume[:, :-1]
        alpha = alpha * inside_volume

        # aggregate along rays
        if background_alpha is not None:
            alpha = alpha * inside_sphere + background_alpha[:, :n_samples] * (1.0 - inside_sphere)
            alpha = torch.cat([alpha, background_alpha[:, n_samples:]], dim=-1)

        weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]
        weights_sum = weights.sum(dim=-1, keepdim=True)

        raw_color = self.color_network(pts, normals, dirs, features).reshape(batch_size, n_samples, 3)
        color = self.accumulate(raw_color, batch_size, n_samples, inside_sphere,
                                inside_volume, weights, weights_sum, background_alpha,
                                background_sampled_color, background_rgb)

        ret = {
            'weights': weights,
            'occupancy': occupancy,
            'inside_sphere': inside_sphere,
            'inside_volume': inside_volume,
            'gradients': normals.reshape(batch_size, n_samples, 3),
            'color_fine': color
        }

        return ret


    def render(self, rays_o, rays_d, near, far, background_rgb=None, isect_sampling=False, is_test=False):
        # sample points along rays

        if self.isect_sampling and isect_sampling:
            z_vals, surf_z_image, near, far = self.isect_sampler.sample_intersection(rays_o, rays_d, near, far, self.point_cloud.sdf)
        else:
            z_vals, surf_z_image, near, far = self.isect_sampler.sample_uniformly(rays_o, rays_d, near, far)
        z_vals_outside = self.isect_sampler.sample_outside(rays_o, rays_d, far)

        sample_dist = 2.0 / self.n_total_samples

        background_alpha = None
        background_sampled_color = None

        # Background model
        if z_vals_outside is not None:
            z_vals_feed = torch.cat([z_vals, z_vals_outside], dim=-1)
            z_vals_feed, _ = torch.sort(z_vals_feed, dim=-1)
            ret_outside = self.render_core_outside(rays_o, rays_d, z_vals_feed, sample_dist, self.nerf)

            background_sampled_color = ret_outside['sampled_color']
            background_alpha = ret_outside['alpha']

        # render core
        ret_fine = self.render_core(rays_o,
                                    rays_d,
                                    z_vals,
                                    background_rgb=background_rgb,
                                    background_alpha=background_alpha,
                                    background_sampled_color=background_sampled_color,
                                    is_test=is_test)

        ret = {
            'z_vals': z_vals,
            'depth': surf_z_image
        }
        ret_fine.update(ret)

        return ret_fine


    def render_occupancy(self, rays_o, rays_d, near, far, isect_sampling=False, override_n_samples=None):
        # sample points along rays

        z_vals, surf_z_image, near, far = self.isect_sampler.sample_uniformly(rays_o, rays_d, near, far, override_n_samples)

        batch_size, n_samples = z_vals.shape

        pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]  # n_rays, n_samples, 3
        dirs = rays_d[:, None, :].expand(pts.shape)

        pts = pts.reshape(-1, 3)
        dirs = dirs.reshape(-1, 3)

        occupancy = self.point_cloud.occupancy(pts)
        occupancy = occupancy.reshape(batch_size, n_samples)

        alpha = torch.zeros_like(occupancy)
        alpha[:, 1:] = 1 - (1 - torch.maximum(occupancy[:, :-1], occupancy[:, 1:])) / (1 - torch.minimum(occupancy[:, :-1], occupancy[:, 1:]) + 1e-8)

        pts_norm = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).reshape(batch_size, n_samples)
        inside_sphere = (pts_norm < self.point_cloud.bounding_sphere).float().detach()
        inside_sphere[:, 1:] = inside_sphere[:, 1:] * inside_sphere[:, :-1]

        inside_volume = self.point_cloud.test_inside_volume(pts).float().detach().reshape(batch_size, n_samples)
        inside_volume[:, 1:] = inside_volume[:, 1:] * inside_volume[:, :-1]

        alpha = alpha * inside_sphere * inside_volume
        weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]

        ret = {
            'alpha': alpha,
            'weights': weights,
            'occupancy': occupancy,
            'inside_sphere': inside_sphere,
            'inside_volume': inside_volume,
            'z_vals': z_vals
        }

        return ret
