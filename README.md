# 3D Reconstruction with Fast Dipole Sums
## [Project website](https://imaging.cs.cmu.edu/fast_dipole_sums/)

## Datasets

### Data Preparation

The following DTU and Blended MVS datasets can be readily ingested by our training pipeline:

* DTU Dataset ([train & eval data](https://fast-dipole-sums-data.s3.us-east-2.amazonaws.com/public/dtu_data.zip), [point clouds](https://fast-dipole-sums-data.s3.us-east-2.amazonaws.com/public/dtu_pcd.zip))

* Blended MVS Dataset ([train & eval data](https://fast-dipole-sums-data.s3.us-east-2.amazonaws.com/public/bmvs_data.zip), [point clouds](https://fast-dipole-sums-data.s3.us-east-2.amazonaws.com/public/bmvs_pcd.zip))

To work with other datasets, organize your data as shown below and refer to the [Colmap tutorial](https://colmap.github.io/tutorial.html#dense-reconstruction) on reconstructing a dense initial point cloud.

### Data Convention
The data is organized as follows:

```
dtu_eval_data                     # DTU evaluation data
public_data
|-- <dataset_name>
    |-- <case_name>
        |-- cameras_sphere.npz    # camera parameters
        |-- image
            |-- 000.png           # target image for each view
            |-- 001.png
        |-- mask
            |-- 000.png           # masks used only during evaluation
            |-- 001.png
          ...
point_cloud_data
|-- <dataset_name>
    |-- <case_name>
        |-- dense
            |-- points.ply
            |-- points.ply.vis    # point cloud in Colmap output format
```

Here the `cameras_sphere.npz` follows the data format in [IDR](https://github.com/lioryariv/idr/blob/main/DATA_CONVENTION.md), where `world_mat_xx` denotes the world to image projection matrix, and `scale_mat_xx` denotes the normalization matrix.

## Usage

### Setup

Building CUDA extensions requires the [Ninja](https://ninja-build.org/) build system. We also recommend ensuring that your system CUDA version matches or is newer than your PyTorch CUDA version before installing the CUDA extensions.

```shell
pip install -r requirements.txt
cd cuda_extensions
bash build_cuda_extensions.sh
```

<details>
  <summary> Dependencies (click to expand) </summary>

* joblib==1.3.2
* matplotlib==3.8.2
* numpy==2.1.1
* open3d==0.18.0
* opencv_python==4.9.0.80
* pandas==2.2.2
* point_cloud_utils==0.30.4
* pyhocon==0.3.60
* PyMCubes==0.1.4
* pyntcloud==0.3.1
* scikit_learn==1.4.0
* scipy==1.14.1
* torch==2.2.0
* tqdm==4.66.1
* trimesh==4.1.3

</details>

### Running

For training and evaluation on all DTU/BMVS scenes:

- **Training**

```shell
bash train_dtu.sh
bash train_bmvs.sh
```

- **Evaluation**

```shell
bash eval_meshes_dtu.sh
bash eval_meshes_bmvs.sh
```

To evaluate the extracted meshes at different iterations, pass the corresponding mesh filename `{iter_steps}.ply` using the `--mesh_name` argument in the corresponding `.sh` file.

----

For working with a single DTU/BMVS scene (replace `bmvs/bear` with any `{dataset}/{case}`):

- **Training**

```shell
python exp_runner.py \
  --conf ./confs/bmvs.conf \
  --case bmvs/bear \
  --mode train
```

- **Extract mesh from trained model**

```shell
python exp_runner.py \
  --conf ./confs/bmvs.conf \
  --case bmvs/bear \
  --mode validate_mesh \
  --mesh_resolution 1024 \
  --is_continue
```

The extracted mesh can be found at `exp/bmvs/bear/meshes/<iter_steps>.ply`.

- **Render Image**

```shell
python exp_runner.py \
  --conf ./confs/bmvs.conf \
  --case bmvs/bear \
  --mode render \
  --image_idx 0 \
  --is_continue
```

The rendered image can be found at `exp/bmvs/bear/renders/<iter_steps>.png`.

## Acknowledgement

This codebase builds upon a simplified version of [NeuS](https://github.com/Totoro97/NeuS), which makes use of code snippets borrowed from [IDR](https://github.com/lioryariv/idr) and [NeRF-pytorch](https://github.com/yenchenlin/nerf-pytorch).

Our custom CUDA extensions are adapted from the [libigl](https://libigl.github.io/) C++ implementation of the fast winding number.

For DTU evaluations, we use a [Python implementation](https://github.com/jzhangbs/DTUeval-python) of the original DTU evaluation code; for Blended MVS evaluations, we use a modified version of the DTU evaluation code with ground truth point clouds from [Gaussian surfels](https://github.com/turandai/gaussian_surfels). Our mesh cleaning code is borrowed from [SparseNeuS](https://github.com/xxlong0/SparseNeuS/blob/main/evaluation/clean_mesh.py).

Thanks for all of these great projects.
