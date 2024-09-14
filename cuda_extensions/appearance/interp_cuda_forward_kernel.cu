#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#ifndef UTILS
#define UTILS
#include "kernel_utils.cu"
#endif

using namespace std;


__device__ void forward_helper(
    const float* __restrict__ points,
    const float* __restrict__ centers,
    const int* __restrict__ children,
    const float* __restrict__ radii,
    const float* __restrict__ fa_point,
    const float* __restrict__ fa_node,
    const int* __restrict__ pi_flat,
    const int* __restrict__ pi_lengths,
    const int* __restrict__ pi_starts,
    const int num_queries,
    const int num_points,
    const int num_nodes,
    const int num_features,
    const float beta,
    const float inv_delta,
    const float* __restrict__ curr_query,
    float* __restrict__ curr_features)
{

    const int max_stack_size = 128;
    int s = 0;
    int t = 1;
    int node_indices_stack[max_stack_size];

    node_indices_stack[0] = 0;

    while (s != t) {

        const int node_index = node_indices_stack[s];
        s = (s + 1) % max_stack_size;

        const int pi_start = pi_starts[node_index];
        const int pi_length = pi_lengths[node_index];

        if (children[node_index * 8] == -1) {
            for (int i = pi_start; i < pi_start + pi_length; i += 1) {
                const int point_index = pi_flat[i];
                update_features(curr_query, points + point_index * 3, curr_features,
                    fa_point + point_index * num_features, inv_delta, num_features, num_queries);
            }
        } else {
            for (int i = 0; i < 8; i += 1) {
                const int child_index = children[node_index * 8 + i];
                const int pi_start_child = pi_starts[child_index];
                const int pi_length_child = pi_lengths[child_index];
                if (pi_length_child > 0) {
                    const float child_dist = compute_dist(centers + child_index * 3, curr_query);
                    if (child_dist > beta * radii[child_index]) {
                        if (children[child_index * 8] == -1) {
                            for (int i = pi_start_child; i < pi_start_child + pi_length_child; i += 1) {
                                const int point_index = pi_flat[i];
                                update_features(curr_query, points + point_index * 3, curr_features,
                                    fa_point + point_index * num_features, inv_delta, num_features, num_queries);
                            }
                        } else {
                            update_features(curr_query, centers + child_index * 3, curr_features,
                                fa_node + child_index * num_features, inv_delta, num_features, num_queries);
                        }
                    } else {
                        node_indices_stack[t] = child_index;
                        t = (t + 1) % max_stack_size;
                    }
                }
            }
        }

    }

}


// template <typename T>
__global__ void interp_cuda_forward_kernel(
    const float* __restrict__ points,
    const float* __restrict__ queries,
    const float* __restrict__ centers,
    const int* __restrict__ children,
    const float* __restrict__ radii,
    const float* __restrict__ fa_point,
    const float* __restrict__ fa_node,
    const int* __restrict__ pi_flat,
    const int* __restrict__ pi_lengths,
    const int* __restrict__ pi_starts,
    const int num_queries,
    const int num_points,
    const int num_nodes,
    const int num_features,
    const float beta,
    const float inv_delta,
    float* __restrict__ features)
{

    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int query_index = index / num_features;
    const int feature_index = index % num_features;

    if (query_index < num_queries) {

        forward_helper(points, centers, children, radii, fa_point, fa_node, 
            pi_flat, pi_lengths, pi_starts, num_queries, num_points, num_nodes, 
            num_features, beta, inv_delta, queries + query_index * 3, 
            features + query_index * num_features);

    }

}


torch::Tensor interp_cuda_forward(
    const torch::Tensor& points,
    const torch::Tensor& queries,
    const torch::Tensor& centers,
    const torch::Tensor& children,
    const torch::Tensor& pi_flat,
    const torch::Tensor& pi_lengths,
    const torch::Tensor& pi_starts,
    const torch::Tensor& radii,
    const torch::Tensor& fa_point,
    const torch::Tensor& fa_node,
    const float& beta,
    const float& inv_delta,
    const int& threads)
{

    const int num_queries = queries.size(0);
    const int num_points = points.size(0);
    const int num_nodes = centers.size(0);
    const int num_features = fa_point.size(1);

    auto features = torch::zeros({num_queries, num_features}, torch::TensorOptions().device(torch::kCUDA, 0));

    const int blocks = (num_queries * num_features + threads - 1) / threads;

    interp_cuda_forward_kernel<<<blocks, threads>>>(
        points.data_ptr<float>(),
        queries.data_ptr<float>(),
        centers.data_ptr<float>(),
        children.data_ptr<int>(),
        radii.data_ptr<float>(),
        fa_point.data_ptr<float>(),
        fa_node.data_ptr<float>(),
        pi_flat.data_ptr<int>(),
        pi_lengths.data_ptr<int>(),
        pi_starts.data_ptr<int>(),
        num_queries,
        num_points,
        num_nodes,
        num_features,
        beta,
        inv_delta,
        features.data_ptr<float>()
    );

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    return features;
}
