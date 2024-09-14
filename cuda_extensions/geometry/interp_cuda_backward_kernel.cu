#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#ifndef UTILS
#define UTILS
#include "kernel_utils.cu"
#endif

using namespace std;


__device__ void backward_helper(
    const float* __restrict__ points,
    const float* __restrict__ centers,
    const int* __restrict__ children,
    const float* __restrict__ radii,
    const int* __restrict__ pi_flat,
    const int* __restrict__ pi_lengths,
    const int* __restrict__ pi_starts,
    const float* __restrict__ df_query,
    const int num_queries,
    const int num_points,
    const int num_nodes,
    const int num_features,
    const float beta,
    const float inv_delta,
    float* __restrict__ dinv_delta,
    const float* __restrict__ curr_query,
    const float* __restrict__ fan_point,
    const float* __restrict__ fan_node,
    float* __restrict__ dfan_point,
    float* __restrict__ dfan_node)
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
                update_gradients(curr_query, points + point_index * 3, df_query,
                    fan_point + point_index * num_features * 3,
                    dfan_point + point_index * num_features * 3,
                    inv_delta, dinv_delta, num_features, num_queries);
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
                                update_gradients(curr_query, points + point_index * 3, df_query,
                                    fan_point + point_index * num_features * 3,
                                    dfan_point + point_index * num_features * 3,
                                    inv_delta, dinv_delta, num_features, num_queries);
                            }
                        } else {
                            update_gradients(curr_query, centers + child_index * 3, df_query,
                                fan_node + child_index * num_features * 3,
                                dfan_node + child_index * num_features * 3,
                                inv_delta, dinv_delta, num_features, num_queries);
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
__global__ void interp_cuda_backward_kernel(
    const float* __restrict__ points,
    const float* __restrict__ queries,
    const float* __restrict__ centers,
    const int* __restrict__ children,
    const float* __restrict__ radii,
    const int* __restrict__ pi_flat,
    const int* __restrict__ pi_lengths,
    const int* __restrict__ pi_starts,
    const float* __restrict__ df_query,
    const int num_queries,
    const int num_points,
    const int num_nodes,
    const int num_features,
    const float beta,
    const float inv_delta,
    float* __restrict__ dinv_delta,
    const float* __restrict__ fan_point,
    const float* __restrict__ fan_node,
    float* __restrict__ dfan_point,
    float* __restrict__ dfan_node)
{

    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int query_index = index / num_features;
    const int feature_index = index % num_features;

    if (query_index < num_queries) {

        backward_helper(points, centers, children, radii,  pi_flat, pi_lengths, 
            pi_starts, df_query + query_index * num_features, num_queries, 
            num_points, num_nodes, num_features, beta, inv_delta, dinv_delta, 
            queries + query_index * 3, fan_point, fan_node, dfan_point, dfan_node);

    }

}


__global__ void propagate_gradients_down_kernel(
    float* dfan_point,
    const float* dfan_node,
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
                dfan_point[index * num_features * 3 + j] += dfan_node[node_index * num_features * 3 + j];
            }
        }
    }

}


std::vector<torch::Tensor> interp_cuda_backward(
    const torch::Tensor& points,
    const torch::Tensor& normals,
    const torch::Tensor& areas,
    const torch::Tensor& queries,
    const torch::Tensor& centers,
    const torch::Tensor& children,
    const torch::Tensor& pi_flat,
    const torch::Tensor& pi_lengths,
    const torch::Tensor& pi_starts,
    const torch::Tensor& ni_flat,
    const torch::Tensor& ni_lengths,
    const torch::Tensor& ni_starts,
    const torch::Tensor& radii,
    const torch::Tensor& features_point,
    const torch::Tensor& fan_point,
    const torch::Tensor& fan_node,
    const torch::Tensor& df_query,
    const float& beta,
    const float& inv_delta,
    const int& threads)
{

    const int num_queries = queries.size(0);
    const int num_points = points.size(0);
    const int num_nodes = centers.size(0);
    const int num_features = df_query.size(1);

    // stores a 3 x num_features matrix
    // multiply by a 1 x 3 normal gives the df vector
    auto dfan_node = torch::zeros({num_nodes, 3, num_features}, torch::TensorOptions().device(torch::kCUDA, 0));
    auto dfan_point = torch::zeros({num_points, 3, num_features}, torch::TensorOptions().device(torch::kCUDA, 0));

    auto dinv_delta = torch::zeros({1}, torch::TensorOptions().device(torch::kCUDA, 0));;

    const int query_blocks = (num_queries + threads * num_features - 1) / threads;

    interp_cuda_backward_kernel<<<query_blocks, threads>>>(
        points.data_ptr<float>(),
        queries.data_ptr<float>(),
        centers.data_ptr<float>(),
        children.data_ptr<int>(),
        radii.data_ptr<float>(),
        pi_flat.data_ptr<int>(),
        pi_lengths.data_ptr<int>(),
        pi_starts.data_ptr<int>(),
        df_query.data_ptr<float>(),
        num_queries,
        num_points,
        num_nodes,
        num_features,
        beta,
        inv_delta,
        dinv_delta.data_ptr<float>(),
        fan_point.data_ptr<float>(),
        fan_node.data_ptr<float>(),
        dfan_point.data_ptr<float>(),
        dfan_node.data_ptr<float>()
    );

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    const int point_blocks = (num_points + threads - 1) / threads;
    propagate_gradients_down_kernel<<<point_blocks, threads>>>(
        dfan_point.data_ptr<float>(),
        dfan_node.data_ptr<float>(),
        ni_flat.data_ptr<int>(),
        ni_lengths.data_ptr<int>(),
        ni_starts.data_ptr<int>(),
        num_points,
        num_features
    );

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    auto anorm = (areas * normals).reshape({num_points, 3, 1});
    auto df_point = (dfan_point * anorm).sum(1, false);

    auto af = (areas * features_point).reshape({num_points, 1, num_features});
    auto dn_point = (dfan_point * af).sum(2, false);

    return {df_point, dn_point, dinv_delta};
}

