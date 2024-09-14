#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#ifndef UTILS
#define UTILS
#include "kernel_utils.cu"
#endif

using namespace std;


__device__ void pos_grad_helper(
    const float* __restrict__ points,
    const float* __restrict__ normals,
    const float* __restrict__ areas,
    const float* __restrict__ centers,
    const int* __restrict__ children,
    const float* __restrict__ radii,
    const float* __restrict__ fan_point,
    const float* __restrict__ fan_node,
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
    float* __restrict__ curr_pos_grad)
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
            #pragma unroll
            for (int i = pi_start; i < pi_start + pi_length; i += 1) {
                const int point_index = pi_flat[i];
                update_pos_grad(curr_query, points + point_index * 3, curr_pos_grad,
                    fan_point + point_index * num_features * 3, inv_delta, num_features, num_queries);
            }
        } else {
            #pragma unroll
            for (int i = 0; i < 8; i += 1) {
                const int child_index = children[node_index * 8 + i];
                const int pi_start_child = pi_starts[child_index];
                const int pi_length_child = pi_lengths[child_index];
                if (pi_length_child > 0) {
                    const float child_dist = compute_dist(centers + child_index * 3, curr_query);
                    if (child_dist > beta * radii[child_index]) {
                        if (children[child_index * 8] == -1) {
                            #pragma unroll
                            for (int i = pi_start_child; i < pi_start_child + pi_length_child; i += 1) {
                                const int point_index = pi_flat[i];
                                update_pos_grad(curr_query, points + point_index * 3, curr_pos_grad,
                                    fan_point + point_index * num_features * 3, inv_delta, num_features, num_queries);
                            }
                        } else {
                            update_pos_grad(curr_query, centers + child_index * 3, curr_pos_grad,
                                fan_node + child_index * num_features * 3, inv_delta, num_features, num_queries);
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
__global__ void interp_cuda_pos_grad_kernel(
    const float* __restrict__ points,
    const float* __restrict__ normals,
    const float* __restrict__ areas,
    const float* __restrict__ queries,
    const float* __restrict__ centers,
    const int* __restrict__ children,
    const float* __restrict__ radii,
    const float* __restrict__ fan_point,
    const float* __restrict__ fan_node,
    const int* __restrict__ pi_flat,
    const int* __restrict__ pi_lengths,
    const int* __restrict__ pi_starts,
    const int num_queries,
    const int num_points,
    const int num_nodes,
    const int num_features,
    const float beta,
    const float inv_delta,
    float* __restrict__ pos_grad)
{

    const int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < num_queries) {
        pos_grad_helper(points, normals, areas, centers, children, radii,
            fan_point, fan_node, pi_flat, pi_lengths, pi_starts, num_queries,
            num_points, num_nodes, num_features, beta, inv_delta,
            queries + index * 3, pos_grad + index * num_features * 3);

    }

}


torch::Tensor interp_cuda_pos_grad(
    const torch::Tensor& points,
    const torch::Tensor& normals,
    const torch::Tensor& areas,
    const torch::Tensor& queries,
    const torch::Tensor& centers,
    const torch::Tensor& children,
    const torch::Tensor& pi_flat,
    const torch::Tensor& pi_lengths,
    const torch::Tensor& pi_starts,
    const torch::Tensor& radii,
    const torch::Tensor& fan_point,
    const torch::Tensor& fan_node,
    const float& beta,
    const float& inv_delta,
    const int& threads)
{

    const int num_queries = queries.size(0);
    const int num_points = points.size(0);
    const int num_nodes = centers.size(0);
    const int num_features = fan_point.size(2);

    auto pos_grad = torch::zeros({num_queries, 3, num_features}, torch::TensorOptions().device(torch::kCUDA, 0));

    const int blocks = (num_queries + threads - 1) / threads;

    interp_cuda_pos_grad_kernel<<<blocks, threads>>>(
        points.data_ptr<float>(),
        normals.data_ptr<float>(),
        areas.data_ptr<float>(),
        queries.data_ptr<float>(),
        centers.data_ptr<float>(),
        children.data_ptr<int>(),
        radii.data_ptr<float>(),
        fan_point.data_ptr<float>(),
        fan_node.data_ptr<float>(),
        pi_flat.data_ptr<int>(),
        pi_lengths.data_ptr<int>(),
        pi_starts.data_ptr<int>(),
        num_queries,
        num_points,
        num_nodes,
        num_features,
        beta,
        inv_delta,
        pos_grad.data_ptr<float>()
    );

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    return pos_grad;
}
