#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#ifndef UTILS
#define UTILS
#include "kernel_utils.cu"
#endif

using namespace std;


__global__ void initialize_features_fa_kernel(
    const float* fa_point,
    float* fa_node,
    const int* ni_flat,
    const int* ni_lengths,
    const int* ni_starts,
    const int num_points,
    const int num_features)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int feature_index = index % num_features;
    const int point_index = index / num_features;

    if (point_index < num_points) {
        const int ni_start = ni_starts[point_index];
        const int ni_length = ni_lengths[point_index];

        for (int i = ni_start; i < ni_start + ni_length; i += 1) {
            const int node_index = ni_flat[i];
            atomicAdd(&fa_node[node_index * num_features + feature_index], fa_point[point_index * num_features + feature_index]);
        }
    }

}


vector<torch::Tensor> initialize_features_fa_cuda(
    const torch::Tensor& features_point,
    const torch::Tensor& areas,
    const torch::Tensor& ni_flat,
    const torch::Tensor& ni_lengths,
    const torch::Tensor& ni_starts,
    const int num_nodes)
{

    int num_points = features_point.size(0);
    int num_features = features_point.size(1);

    torch::Tensor fa_point = features_point * areas;
    torch::Tensor fa_node = torch::zeros({num_nodes, num_features});

    fa_point = fa_point.to(torch::kCUDA);
    fa_node = fa_node.to(torch::kCUDA);

    const int threads = 1024;
    const int point_blocks = (num_points * num_features + threads - 1) / threads;
    initialize_features_fa_kernel<<<point_blocks, threads>>>(
        fa_point.data_ptr<float>(),
        fa_node.data_ptr<float>(),
        ni_flat.data_ptr<int>(),
        ni_lengths.data_ptr<int>(),
        ni_starts.data_ptr<int>(),
        num_points,
        num_features
    );

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    return {fa_point, fa_node};

}
