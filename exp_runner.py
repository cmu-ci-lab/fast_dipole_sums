import os
import gc
import logging
import argparse
import numpy as np
import cv2 as cv
import trimesh
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from shutil import copyfile
from tqdm import tqdm
from pyhocon import ConfigFactory
import open3d as o3d

from models.dataset import Dataset
from models.sampler import PointSampler
from models.fields import (
    RenderingNetwork,
    NeRF
)
from models.point_cloud import PointCloud
from models.renderer import Renderer
from util.mesh_util import read_mesh, write_mesh, extract_geometry
from util.point_cloud_util import euler_angles_to_matrix


logging.getLogger('matplotlib.font_manager').disabled = True
torch.set_float32_matmul_precision('high')

torch.manual_seed(0)
np.random.seed(0)

class Runner:
    def __init__(self, conf_path, mode='train', case='CASE_NAME', is_continue=False, max_n_training_images=-1):
        self.device = torch.device('cuda')

        # Configuration
        self.conf_path = conf_path
        f = open(self.conf_path)
        conf_text = f.read()
        conf_text = conf_text.replace('CASE_NAME', case)
        f.close()

        self.conf = ConfigFactory.parse_string(conf_text)
        self.conf['dataset.data_dir'] = self.conf['dataset.data_dir'].replace('CASE_NAME', case)
        self.base_exp_dir = self.conf['general.base_exp_dir']
        os.makedirs(self.base_exp_dir, exist_ok=True)
        self.dataset = Dataset(self.conf['dataset'])
        self.iter_step = 0

        # Training parameters
        self.end_iter = self.conf.get_int('train.end_iter')
        self.save_freq = self.conf.get_int('train.save_freq')
        self.report_freq = self.conf.get_int('train.report_freq')
        self.val_freq = self.conf.get_int('train.val_freq')
        self.val_mesh_freq = self.conf.get_int('train.val_mesh_freq')
        self.batch_size = self.conf.get_int('train.batch_size')
        self.validate_resolution_level = self.conf.get_int('train.validate_resolution_level')
        self.validate_mesh_resolution = self.conf.get_int('train.validate_mesh_resolution')

        self.learning_rate_pc = self.conf.get_float('train.learning_rate_pc')
        self.learning_rate_nerf = self.conf.get_float('train.learning_rate_nerf', default=0.0)
        self.learning_rate_color = self.conf.get_float('train.learning_rate_color', default=0.0)
        self.learning_rate_alpha = self.conf.get_float('train.learning_rate_alpha')
        self.use_white_bkgd = self.conf.get_bool('train.use_white_bkgd')
        self.warm_up_end = self.conf.get_float('train.warm_up_end', default=0.0)
        self.anneal_end = self.conf.get_float('train.anneal_end', default=0.0)
        self.max_n_training_images = max_n_training_images

        self.entropy_weight = self.conf.get_float('train.entropy_weight', default=0)
        self.reg_weight_w = self.conf.get_float('train.reg_weight_w', default=0)
        self.reg_weight_n = self.conf.get_float('train.reg_weight_n', default=0)

        self.isect_sampling = self.conf.get_bool('train.isect_sampling', default=True)
        self.gradient_accumulation = self.conf.get_int('train.gradient_accumulation', default=1)

        self.grow_point = self.conf.get_bool('train.grow_point', default=False)
        self.grow_point_start = self.conf.get_int('train.grow_point_start', default=0)
        self.grow_point_end = self.conf.get_int('train.grow_point_end', default=0)
        self.grow_point_step = self.conf.get_int('train.grow_point_step', default=0)
        self.grow_point_target_n_images = self.conf.get_int('train.grow_point_target_n_images', default=16)
        self.grow_point_thresh = self.conf.get_float('train.grow_point_thresh', default=0.01)

        self.is_continue = is_continue
        self.mode = mode
        self.model_list = []
        self.writer = None

        point_cloud_params = []
        point_cloud_args = self.conf['model.point_cloud']
        if self.is_continue:
            point_cloud_args['trained_model'] = True
        self.point_cloud = PointCloud(**point_cloud_args).to(self.device)

        if not self.is_continue:
            point_cloud_params += self.point_cloud.optimizeable_parameters

        self.params = [{'params': point_cloud_params, 'lr': self.learning_rate_pc}]
        self.lrs = [self.learning_rate_pc]

        # optionally initialize a background NeRF network
        self.nerf_outside = None
        self.renders_background = 'model.point_sampler.n_outside' not in self.conf or self.conf['model.point_sampler.n_outside'] > 0
        if self.renders_background:
            self.nerf_outside = NeRF(**self.conf['model.nerf']).to(self.device)
            # self.nerf_outside = torch.compile(self.nerf_outside)
            self.nerf_params = list(self.nerf_outside.parameters())
            self.params += [{'params': self.nerf_params, 'lr': self.learning_rate_nerf}]
            self.lrs += [self.learning_rate_nerf]

        self.color_network = None
        self.use_nn = 'model.rendering_network' in self.conf
        if self.use_nn:
            self.color_network = RenderingNetwork(**self.conf['model.rendering_network']).to(self.device)
            # self.color_network = torch.compile(self.color_network)
            self.color_params = list(self.color_network.parameters())
            self.params += [{'params': self.color_params, 'lr': self.learning_rate_color}]
            self.lrs += [self.learning_rate_color]

        self.point_sampler = PointSampler(**self.conf['model.point_sampler'])
        self.renderer = Renderer(self.nerf_outside,
                                 self.color_network,
                                 self.point_cloud,
                                 self.point_sampler,
                                 self.point_sampler.n_total_samples,
                                 self.isect_sampling)

        # Load checkpoint
        latest_model_name = None
        if is_continue:
            model_list_raw = os.listdir(os.path.join(self.base_exp_dir, 'checkpoints'))
            model_list = []
            for model_name in model_list_raw:
                if model_name[-3:] == 'pth' and int(model_name[5:-4]) <= self.end_iter:
                    model_list.append(model_name)
            model_list.sort()
            latest_model_name = model_list[-1]

        checkpoint = None
        if latest_model_name is not None:
            logging.info('Find checkpoint: {}'.format(latest_model_name))
            checkpoint = self.load_checkpoint(latest_model_name)

            self.point_cloud.load_model(checkpoint['point_cloud'])
            self.params[0]['params'] = self.point_cloud.optimizeable_parameters

            if self.nerf_outside is not None and 'nerf' in checkpoint:
                self.nerf_outside.load_state_dict(checkpoint['nerf'])

            if self.color_network is not None and 'color' in checkpoint:
                self.color_network.load_state_dict(checkpoint['color'])

            self.iter_step = checkpoint['iter_step']

        self.optimizer = torch.optim.Adam(self.params, lr=self.learning_rate_pc)
        if checkpoint is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        # Backup codes and configs for debug
        if self.mode[:5] == 'train' and not self.is_continue:
            self.file_backup()


    def train(self):
        self.writer = SummaryWriter(log_dir=os.path.join(self.base_exp_dir, 'logs'))
        self.update_learning_rate()
        res_step = self.end_iter - self.iter_step
        image_perm = self.get_image_perm()

        for iter_i in tqdm(range(res_step)):
            image_idx = image_perm[self.iter_step % len(image_perm)]
            data = self.dataset.gen_random_rays_at(image_idx, self.batch_size)

            rays_o, rays_d, true_rgb = data[:, :3], data[:, 3: 6], data[:, 6: 9]
            near, far = self.dataset.near_far_from_sphere(rays_o, rays_d, self.point_cloud.bounding_sphere)

            background_rgb = None
            if self.use_white_bkgd:
                background_rgb = torch.ones([1, 3])

            render_out = self.renderer.render(rays_o, rays_d, near, far,
                                              background_rgb=background_rgb,
                                              isect_sampling=True)

            color_fine = render_out['color_fine']
            weights = render_out['weights']
            inside_sphere = render_out['inside_sphere']
            inside_volume = render_out['inside_volume']

            # Loss
            color_error = color_fine - true_rgb
            color_fine_loss = F.l1_loss(color_error, torch.zeros_like(color_error), reduction='sum') / true_rgb.shape[0]
            psnr = 20.0 * torch.log10(1.0 / (((color_fine - true_rgb)**2).sum() / (true_rgb.shape[0] * 3.0)).sqrt())

            weights = weights[:, :self.point_sampler.n_total_samples] * inside_sphere * inside_volume
            weight_sum = weights.sum(-1)

            weights = torch.clip(weights, 1e-4, 1 - 1e-4)
            weight_sum = torch.clip(weight_sum, 1e-4, 1 - 1e-4)

            entropy_loss = (-(weights * torch.log(weights)).sum(dim=-1) - (1 - weight_sum) * torch.log(1 - weight_sum)).mean()
            entropy_loss = torch.nan_to_num(entropy_loss, 0.0, 0.0, 0.0)

            loss = color_fine_loss +\
                   entropy_loss * self.entropy_weight

            reg_loss_w = (torch.relu(self.point_cloud.wpos_point - 1) ** 2).mean()
            if self.point_cloud.optimize_normals:
                reg_loss_n = ((self.point_cloud.normalized_normals - self.point_cloud.init_normals) ** 2).mean()
            else:
                reg_loss_n = 0.

            reg_loss = (reg_loss_w * self.reg_weight_w + reg_loss_n * self.reg_weight_n)
            loss += reg_loss

            (loss / self.gradient_accumulation).backward(retain_graph=True)

            if (iter_i + 1) % self.gradient_accumulation == 0:
                # IMPORTANT TO CALL THIS MANUALLY
                self.point_cloud.backprop_gradients()

                self.optimizer.step()
                self.optimizer.zero_grad()

                # IMPORTANT TO CALL THIS MANUALLY
                self.point_cloud.update_node_features()

            self.iter_step += 1

            self.writer.add_scalar('Loss/loss', loss, self.iter_step)
            self.writer.add_scalar('Loss/color_loss', color_fine_loss, self.iter_step)
            self.writer.add_scalar('Loss/entropy_loss', entropy_loss, self.iter_step)
            self.writer.add_scalar('Loss/reg_loss_w', reg_loss_w, self.iter_step)
            self.writer.add_scalar('Loss/reg_loss_n', reg_loss_n, self.iter_step)
            self.writer.add_scalar('Statistics/s_val', self.point_cloud.s.item(), self.iter_step)
            self.writer.add_scalar('Statistics/inv_delta_w', self.point_cloud.inv_delta_w.item(), self.iter_step)
            self.writer.add_scalar('Statistics/inv_delta_f', self.point_cloud.inv_delta_f.item(), self.iter_step)
            self.writer.add_scalar('Statistics/psnr', psnr, self.iter_step)

            if self.iter_step == 1:
                pcd = self.point_cloud.get_point_cloud()
                o3d.io.write_point_cloud(os.path.join(self.base_exp_dir, 'logs', 'point_cloud_{:0>8d}.ply'.format(self.iter_step)), pcd)

            if self.iter_step % self.report_freq == 0:
                lrs = [pg['lr'] for pg in self.optimizer.param_groups]
                print('iter:{:8>d} loss = {} lr={}'.format(self.iter_step, loss, lrs))

            if self.iter_step % self.save_freq == 0:
                self.save_checkpoint()

            if self.iter_step % self.val_freq == 0 or self.iter_step == 1:
                self.validate_image()

            if self.iter_step % self.val_mesh_freq == 0 or self.iter_step == 1 or self.iter_step == 1000 or self.iter_step == 3000:
                self.validate_mesh()

            if self.grow_point and self.iter_step % self.grow_point_step == 0 and self.grow_point_start <= self.iter_step < self.grow_point_end:
                self.grow_points()

            self.update_learning_rate()

            if self.iter_step % len(image_perm) == 0:
                image_perm = self.get_image_perm()


    def grow_points(self):
        with torch.no_grad():
            candidate_points = []
            view_dirs = []

            if self.max_n_training_images <= 0:
                n_images = self.dataset.n_images
            else:
                n_images = min(self.dataset.n_images, self.max_n_training_images)

            if n_images > self.grow_point_target_n_images:
                image_indices = np.random.permutation(n_images)[:self.grow_point_target_n_images]
            else:
                image_indices = range(n_images)

            for k in image_indices:
                data = self.dataset.gen_random_rays_at(k, self.batch_size)

                rays_o, rays_d, true_rgb = data[:, :3], data[:, 3: 6], data[:, 6: 9]

                # random rays
                if self.iter_step >= (self.grow_point_end + self.grow_point_start) / 2:
                    random_rotation = euler_angles_to_matrix(torch.rand(3) * 2 * torch.pi / 6, 'XYZ')
                    rays_o = rays_o @ random_rotation.T
                    rays_d = rays_d @ random_rotation.T

                near, far = self.dataset.near_far_from_sphere(rays_o, rays_d, self.point_cloud.bounding_sphere)

                render_out = self.renderer.render_occupancy(rays_o, rays_d, near, far,
                                                            isect_sampling=False, override_n_samples=1024)

                z_vals = render_out['z_vals'][:, :1023]
                sampled_pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]

                # don't grow points in the background obviously...
                weights = render_out['weights'][:, :1023]
                weights *= render_out['inside_sphere'][:, :1023] * render_out['inside_volume'][:, :1023]

                weight_sum = weights.sum(-1)
                candidate_idx = torch.argmax(weights, dim=-1)
                point_grow_indices = weight_sum > 0.5
                rows = torch.arange(sampled_pts.shape[0], device=weights.device)[point_grow_indices]
                cols = candidate_idx[point_grow_indices]

                sampled_pts = (sampled_pts[rows, cols] + sampled_pts[rows, cols - 1]) / 2
                rays_d = -rays_d[rows]

                candidate_points.append(sampled_pts)
                view_dirs.append(rays_d)

            candidate_points = torch.concat(candidate_points, dim=0)
            view_dirs = torch.concat(view_dirs, dim=0)

            pcd = self.point_cloud.grow_points(candidate_points, view_dirs, thresh=self.grow_point_thresh)

            del candidate_points, view_dirs
            gc.collect()

        self.point_cloud.update_node_features()

        # reinitialize optimizers (is there a better way to do this?)
        self.params[0]['params'] = self.point_cloud.optimizeable_parameters
        self.optimizer = torch.optim.Adam(self.params, lr=self.learning_rate_pc)

        if pcd is not None:
            o3d.io.write_point_cloud(os.path.join(self.base_exp_dir, 'logs', 'point_cloud_{:0>8d}.ply'.format(self.iter_step)), pcd)


    def get_image_perm(self):
        if self.max_n_training_images <= 0:
            return torch.randperm(self.dataset.n_images)
        else:
            return torch.randperm(min(self.dataset.n_images, self.max_n_training_images))


    def update_learning_rate(self):
        if self.iter_step < self.warm_up_end:
            learning_factor = self.iter_step / self.warm_up_end
        else:
            alpha = self.learning_rate_alpha
            progress = (self.iter_step - self.warm_up_end) / (self.end_iter - self.warm_up_end)
            learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha

        for i in range(len(self.optimizer.param_groups)):
            self.optimizer.param_groups[i]['lr'] = self.lrs[i] * learning_factor


    def file_backup(self):
        dir_lis = self.conf['general.recording']
        os.makedirs(os.path.join(self.base_exp_dir, 'recording'), exist_ok=True)
        for dir_name in dir_lis:
            cur_dir = os.path.join(self.base_exp_dir, 'recording', dir_name)
            os.makedirs(cur_dir, exist_ok=True)
            files = os.listdir(dir_name)
            for f_name in files:
                if f_name[-3:] == '.py':
                    copyfile(os.path.join(dir_name, f_name), os.path.join(cur_dir, f_name))

        copyfile(self.conf_path, os.path.join(self.base_exp_dir, 'recording', 'config.conf'))


    def load_checkpoint(self, checkpoint_name):
        checkpoint = torch.load(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name), map_location=self.device)
        return checkpoint


    def save_checkpoint(self):
        checkpoint = {
            'optimizer': self.optimizer.state_dict(),
            'point_cloud': self.point_cloud.save_info(),
            'iter_step': self.iter_step,
        }

        if self.nerf_outside is not None:
            checkpoint['nerf'] = self.nerf_outside.state_dict()
        if self.use_nn:
            checkpoint['color'] = self.color_network.state_dict()

        os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints'), exist_ok=True)
        torch.save(checkpoint, os.path.join(self.base_exp_dir, 'checkpoints', 'ckpt_{:0>6d}.pth'.format(self.iter_step)))


    def validate_image(self, idx=-1, resolution_level=-1):
        if idx < 0:
            idx = np.random.randint(self.dataset.n_images)

        print('Validate: iter: {}, camera: {}'.format(self.iter_step, idx))

        if resolution_level < 0:
            resolution_level = self.validate_resolution_level

        rays_o, rays_d = self.dataset.gen_rays_at(idx, resolution_level=resolution_level)
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)

        out_rgb_fine = []
        out_normal_fine = []
        out_depth = []

        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch, self.point_cloud.bounding_sphere)
            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

            render_out = self.renderer.render(rays_o_batch,
                                              rays_d_batch,
                                              near,
                                              far,
                                              background_rgb=background_rgb,
                                              isect_sampling=True,
                                              is_test=True)

            def feasible(key): return (key in render_out) and (render_out[key] is not None)

            if feasible('color_fine'):
                out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())

            if feasible('gradients') and feasible('weights'):
                n_samples = self.point_sampler.n_total_samples
                normals = render_out['gradients'] * render_out['weights'][:, :n_samples, None]
                if feasible('inside_sphere'):
                    normals = normals * render_out['inside_sphere'][..., None]
                normals = normals.sum(dim=1).detach().cpu().numpy()
                out_normal_fine.append(normals)

            if feasible('depth'):
                out_depth.append(render_out['depth'].detach().cpu().numpy())

            del render_out

        if len(out_rgb_fine) > 0:
            img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3, -1]) * 256).clip(0, 255)

        normal_img = None
        if len(out_normal_fine) > 0:
            normal_img = np.concatenate(out_normal_fine, axis=0)
            rot = np.linalg.inv(self.dataset.pose_all[idx, :3, :3].detach().cpu().numpy())
            normal_img = (np.matmul(rot[None, :, :], normal_img[:, :, None])
                          .reshape([H, W, 3, -1]) * 128 + 128).clip(0, 255)

        depth_img = None
        if len(out_depth) > 0:
            depth_img = (np.concatenate(out_depth, axis=0).reshape([H, W, 3]) * 256).clip(0, 255).astype(np.uint8)
            depth_img = cv.applyColorMap(depth_img, cv.COLORMAP_MAGMA)

        os.makedirs(os.path.join(self.base_exp_dir, 'validations_fine'), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir, 'normals'), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir, 'depth'), exist_ok=True)

        if self.mode == 'render':
            os.makedirs(os.path.join(self.base_exp_dir, 'renders'), exist_ok=True)

        if len(out_rgb_fine) > 0:
            for i in range(img_fine.shape[-1]):
                if self.mode == 'render':
                    cv.imwrite(os.path.join(self.base_exp_dir,
                                            'renders',
                                            'render_{}.png'.format(idx)),
                               img_fine[..., i])
                else:
                    cv.imwrite(os.path.join(self.base_exp_dir,
                                            'validations_fine',
                                            '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
                            np.concatenate([img_fine[..., i],
                                            self.dataset.image_at(idx, resolution_level=resolution_level)]))
            if len(out_normal_fine) > 0:
                cv.imwrite(os.path.join(self.base_exp_dir,
                                        'normals',
                                        '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
                           normal_img[..., i])

        if len(out_depth) > 0:
            cv.imwrite(os.path.join(self.base_exp_dir,
                                        'depth',
                                        '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
                           depth_img)


    def validate_mesh(self, scale_mesh=True, resolution=-1, threshold=0.0):
        if resolution < 0:
            resolution = self.validate_mesh_resolution

        bound_min = torch.tensor(self.dataset.object_bbox_min, dtype=torch.float32) * self.point_cloud.bounding_sphere
        bound_max = torch.tensor(self.dataset.object_bbox_max, dtype=torch.float32) * self.point_cloud.bounding_sphere

        query_func = self.point_cloud.sdf
        vertices, triangles = extract_geometry(bound_min, bound_max,
                                               resolution=resolution,
                                               threshold=threshold,
                                               query_func=query_func)

        os.makedirs(os.path.join(self.base_exp_dir, 'meshes'), exist_ok=True)

        if scale_mesh:
            vertices = vertices * self.dataset.scale_mats_np[0][0, 0] + self.dataset.scale_mats_np[0][:3, 3][None]
            name = '{:0>8d}.ply'.format(self.iter_step)
        else:
            name = '{:0>8d}-local.ply'.format(self.iter_step)

        mesh = trimesh.Trimesh(vertices, triangles)
        mesh.export(os.path.join(self.base_exp_dir, 'meshes', name))

        logging.info('End')


    def eval_psnr(self):
        with torch.no_grad():
            psnrs = []

            for idx in range(self.dataset.n_images):
                rays_o, rays_d = self.dataset.gen_rays_at(idx, resolution_level=1)
                H, W, _ = rays_o.shape
                rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
                rays_d = rays_d.reshape(-1, 3).split(self.batch_size)

                out_rgb_fine = []
                for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
                    near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch, self.point_cloud.bounding_sphere)
                    background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

                    render_out = self.renderer.render(rays_o_batch,
                                                    rays_d_batch,
                                                    near,
                                                    far,
                                                    background_rgb=background_rgb,
                                                    isect_sampling=True)
                    out_rgb_fine.append(render_out['color_fine'])

                image_render = torch.concat(out_rgb_fine, dim=0).reshape([H, W, 3]).cpu()
                image_gt = self.dataset.images[idx].cpu()

                cv.imwrite(os.path.join(self.base_exp_dir,'renders','render_1.png'.format(idx)), (image_render.numpy() * 256).clip(0, 255))
                cv.imwrite(os.path.join(self.base_exp_dir,'renders','render_2.png'.format(idx)), (image_gt.numpy() * 256).clip(0, 255))

                psnr = 20.0 * torch.log10(1.0 / (((image_render - image_gt)**2).sum() / (H * W * 3.0)).sqrt())
                psnrs.append(psnr)

                del image_render

                print(f'image {idx} / {self.dataset.n_images} psnr: ', psnr)

            print(f'average psnr: {sum(psnrs) / len(psnrs)}')


if __name__ == '__main__':
    print('Running..')

    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/base.conf')
    parser.add_argument('--case', type=str, default='bmvs/bmvs_bear')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--is_continue', default=False, action="store_true")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--max_n_training_images', type=int, default=-1)

    # novel view synthesis
    parser.add_argument('--image_idx', type=int, default=0)
    parser.add_argument('--image_resolution_level', type=int, default=1)

    # mesh extraction
    parser.add_argument('--mesh_resolution', type=int, default=512)
    parser.add_argument('--use_local_scale', default=False, action="store_true")
    parser.add_argument('--mcube_threshold', type=float, default=0.0)

    parser.add_argument('--plane_ply', type=str, default='visualizations/planes/clock/clock_plane.ply')

    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    runner = Runner(args.conf, args.mode, args.case, args.is_continue, args.max_n_training_images)

    if args.mode == 'train':
        runner.train()
        for i in range(5):
            runner.validate_image(resolution_level=1)

    elif args.mode == 'render':
        runner.validate_image(idx=args.image_idx,
                              resolution_level=args.image_resolution_level)
    elif args.mode == 'validate_mesh':
        runner.validate_mesh(scale_mesh=(not args.use_local_scale),
                             resolution=args.mesh_resolution,
                             threshold=args.mcube_threshold)
    elif args.mode == 'eval_psnr':
        runner.eval_psnr()
    elif args.mode == 'visualize_plane':
        runner.point_cloud.visualize_on_plane(args.plane_ply, runner.color_network)
    else:
        raise Exception(f'Mode not recognized: {args.mode}')