#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#ifndef UTILS
#define UTILS
#include "kernel_utils.cu"
#endif

using namespace std;


__global__ void initialize_features_fan_kernel(
    const float* fan_point,
    float* fan_node,
    const int* ni_flat,
    const int* ni_lengths,
    const int* ni_starts,
    const int num_points,
    const int num_features)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < num_points) {
        const int ni_start = ni_starts[index];
        const int ni_length = ni_lengths[index];

        for (int i = ni_start; i < ni_start + ni_length; i += 1) {
            const int node_index = ni_flat[i];
            for (int j = 0; j < num_features * 3; j += 1) {
                atomicAdd(&fan_node[node_index * num_features * 3 + j], fan_point[index * num_features * 3 + j]);
            }
        }
    }

}


vector<torch::Tensor> initialize_features_fan_cuda(
    const torch::Tensor& features_point,
    const torch::Tensor& normals,
    const torch::Tensor& areas,
    const torch::Tensor& ni_flat,
    const torch::Tensor& ni_lengths,
    const torch::Tensor& ni_starts,
    const int num_nodes)
{

    int num_points = features_point.size(0);
    int num_features = features_point.size(1);

    torch::Tensor fan_point = (features_point * areas).reshape({num_points, 1, num_features}) * normals.reshape({num_points, 3, 1});
    torch::Tensor fan_node = torch::zeros({num_nodes, 3, num_features});

    fan_point = fan_point.to(torch::kCUDA);
    fan_node = fan_node.to(torch::kCUDA);

    const int threads = 1024;
    const int point_blocks = (num_points + threads - 1) / threads;
    initialize_features_fan_kernel<<<point_blocks, threads>>>(
        fan_point.data_ptr<float>(),
        fan_node.data_ptr<float>(),
        ni_flat.data_ptr<int>(),
        ni_lengths.data_ptr<int>(),
        ni_starts.data_ptr<int>(),
        num_points,
        num_features
    );

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    return {fan_point, fan_node};

}
