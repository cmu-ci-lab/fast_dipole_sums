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
    const float* __restrict__ fa_point,
    const float* __restrict__ fa_node,
    float* __restrict__ dfa_point,
    float* __restrict__ dfa_node)
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
                    fa_point + point_index * num_features,
                    dfa_point + point_index * num_features,
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
                                    fa_point + point_index * num_features,
                                    dfa_point + point_index * num_features,
                                    inv_delta, dinv_delta, num_features, num_queries);
                            }
                        } else {
                            update_gradients(curr_query, centers + child_index * 3, df_query,
                                fa_node + child_index * num_features,
                                dfa_node + child_index * num_features,
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
    const float* __restrict__ fa_point,
    const float* __restrict__ fa_node,
    float* __restrict__ dfa_point,
    float* __restrict__ dfa_node)
{

    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int query_index = index / num_features;
    const int feature_index = index % num_features;

    if (query_index < num_queries) {

        backward_helper(points, centers, children, radii, pi_flat, pi_lengths, pi_starts, 
            df_query + query_index * num_features, num_queries, num_points, num_nodes, 
            num_features, beta, inv_delta, dinv_delta, queries + query_index * 3, fa_point, 
            fa_node, dfa_point, dfa_node);

    }

}


__global__ void propagate_gradients_down_kernel(
    float* dfa_point,
    const float* dfa_node,
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
            for (int j = 0; j < num_features; j += 1) {
                dfa_point[index * num_features + j] += dfa_node[node_index * num_features + j];
            }
        }
    }

}


std::vector<torch::Tensor> interp_cuda_backward(
    const torch::Tensor& points,
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
    const torch::Tensor& fa_point,
    const torch::Tensor& fa_node,
    const torch::Tensor& df_query,
    const float& beta,
    const float& inv_delta,
    const int& threads)
{

    const int num_queries = queries.size(0);
    const int num_points = points.size(0);
    const int num_nodes = centers.size(0);
    const int num_features = df_query.size(1);

    auto dfa_node = torch::zeros({num_nodes, num_features}, torch::TensorOptions().device(torch::kCUDA, 0));
    auto dfa_point = torch::zeros({num_points, num_features}, torch::TensorOptions().device(torch::kCUDA, 0));

    auto dinv_delta = torch::zeros({1}, torch::TensorOptions().device(torch::kCUDA, 0));;

    const int query_blocks = (num_queries * num_features + threads - 1) / threads;

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
        fa_point.data_ptr<float>(),
        fa_node.data_ptr<float>(),
        dfa_point.data_ptr<float>(),
        dfa_node.data_ptr<float>()
    );

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    const int point_blocks = (num_points + threads - 1) / threads;
    propagate_gradients_down_kernel<<<point_blocks, threads>>>(
        dfa_point.data_ptr<float>(),
        dfa_node.data_ptr<float>(),
        ni_flat.data_ptr<int>(),
        ni_lengths.data_ptr<int>(),
        ni_starts.data_ptr<int>(),
        num_points,
        num_features
    );

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    auto df_point = dfa_point * areas;

    return {df_point, dinv_delta};
}

