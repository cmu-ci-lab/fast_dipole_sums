#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#define ONE_OVER_PI4 0.079577471f
#define TWO_OVER_SQRT_PI 1.1283791671f
#define PI_NEG_3OVER2_POWER 0.179587122125f
#define ONE_OVER_SQRT_PI 0.564189584f


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


__device__ __forceinline__ float compute_dist(const float* a, const float* b)
{
    const float dx = a[0] - b[0];
    const float dy = a[1] - b[1];
    const float dz = a[2] - b[2];
    return sqrtf(dx * dx + dy * dy + dz * dz);
}


__device__ __forceinline__ void update_gradients(
    const float* __restrict__ query,
    const float* __restrict__ point,
    const float* __restrict__ df_query,
    const float* __restrict__ fan_point,
    float* __restrict__ dfan_point,
    const float inv_delta,
    float* __restrict__ dinv_delta,
    const int num_features,
    const int num_queries)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int feature_index = index % num_features;

    const float dx = point[0] - query[0];
    const float dy = point[1] - query[1];
    const float dz = point[2] - query[2];

    const float dist2 = fmaxf(dx * dx + dy * dy + dz * dz, 1e-8f);
    const float dist = sqrtf(dist2);

    const float weight = ONE_OVER_PI4 / dist2;
    const float dot = fan_point[feature_index] * dx + fan_point[num_features + feature_index] * dy + fan_point[num_features * 2 + feature_index] * dz;
    float s = 1.0f;

    // if regularizer term is active
    if (dist < 2. / inv_delta) {
        const float t = dist * inv_delta;
        s = erff(t) - TWO_OVER_SQRT_PI * t * expf(-(t * t));

        // gradient of inv_delta term
        const float inv_delta_2 = inv_delta * inv_delta;
        const float dinv_coeff = inv_delta_2 * expf(-dist2 * inv_delta_2) * PI_NEG_3OVER2_POWER;
        atomicAdd(dinv_delta, dinv_coeff * dot * df_query[feature_index]);
    }

    // grad = (dx, dy, dz) / (4 * PI * dist ** 3)
    // dfan_point : 3 x num_features = grad.T @ df_query
    const float tmp = weight * s / dist * df_query[feature_index];

    atomicAdd(&dfan_point[feature_index], dx * tmp);
    atomicAdd(&dfan_point[feature_index + num_features], dy * tmp);
    atomicAdd(&dfan_point[feature_index + 2 * num_features], dz * tmp);

}


__device__ __forceinline__ void update_features(
    const float* __restrict__ query,
    const float* __restrict__ point,
    float* __restrict__ f_query,
    const float* __restrict__ fan_point,
    const float inv_delta,
    const int num_features,
    const int num_queries)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int feature_index = index % num_features;

    const float dx = point[0] - query[0];
    const float dy = point[1] - query[1];
    const float dz = point[2] - query[2];

    const float dist2 = fmaxf(dx * dx + dy * dy + dz * dz, 1e-8f);
    const float dist = sqrtf(dist2);

    const float weight = ONE_OVER_PI4 / dist2;
    const float dot = fan_point[feature_index] * dx + fan_point[num_features + feature_index] * dy + fan_point[num_features * 2 + feature_index] * dz;
    const float poisson = dot * weight / dist;
    float s = 1.0;

    if (dist < 2. / inv_delta) {
        const float t = dist * inv_delta;
        s = erff(t) - TWO_OVER_SQRT_PI * t * expf(-(t * t));
    }

    f_query[feature_index] += poisson * s;

}


__device__ __forceinline__ void update_pos_grad(
    const float* __restrict__ query,
    const float* __restrict__ point,
    float* __restrict__ pos_grad_query,
    const float* __restrict__ fan_point,
    const float inv_delta,
    const int num_features,
    const int num_queries)
{
    const float dx = point[0] - query[0];
    const float dy = point[1] - query[1];
    const float dz = point[2] - query[2];

    const float dist2 = fmaxf(dx * dx + dy * dy + dz * dz, 1e-8);
    const float inv_dist = rsqrtf(dist2);
    const float inv_dist2 = 1 / dist2;

    const float dist = 1 / inv_dist;

    // float weight = ONE_OVER_PI4 * inv_dist2 * inv_dist;

    // for (int i = 0; i < num_features; i += 1) {
    //     const float dot = fan_point[i] * dx + fan_point[num_features + i] * dy + fan_point[num_features * 2 + i] * dz;

    //     float grad_x = (-fan_point[i] + 3 * dx * dot * inv_dist2) * weight;
    //     float grad_y = (-fan_point[num_features + i] + 3 * dy * dot * inv_dist2) * weight;
    //     float grad_z = (-fan_point[num_features * 2 + i] + 3 * dz * dot * inv_dist2) * weight;

    //     if (isnan(grad_x) || isinf(grad_x)) grad_x = 0;
    //     if (isnan(grad_y) || isinf(grad_y)) grad_y = 0;
    //     if (isnan(grad_z) || isinf(grad_z)) grad_z = 0;

    //     pos_grad_query[i] += grad_x;
    //     pos_grad_query[num_features + i] += grad_y;
    //     pos_grad_query[num_features * 2 + i] += grad_z;
    // }

    const float weight = ONE_OVER_PI4 / dist2;
    const float inv_delta2 = inv_delta * inv_delta;
    const float inv_delta3 = inv_delta2 * inv_delta;

    float dpoisson_common = ONE_OVER_PI4 * inv_dist2 * inv_dist;

    const float t = dist * inv_delta;
    float reg = erff(t) - TWO_OVER_SQRT_PI * t * expf(-(t * t));

    // -4 dx e^(-R^2 t^2) R t^3 / sqrt(PI)
    float dreg_common = -4 * expf(-dist2 * inv_delta2) * dist * inv_delta3 * ONE_OVER_SQRT_PI;
    float dreg_x = dreg_common * dx;
    float dreg_y = dreg_common * dy;
    float dreg_z = dreg_common * dz;

    for (int i = 0; i < num_features; i += 1) {
        const float dot = fan_point[i] * dx + fan_point[num_features + i] * dy + fan_point[num_features * 2 + i] * dz;
        const float poisson = dot * weight / dist;

        // 3 C dx / (4 PI R^5) - fx / (4 PI R^3)
        float dpoisson_x = dpoisson_common * (-fan_point[i] + 3 * dx * dot * inv_dist2);
        float dpoisson_y = dpoisson_common * (-fan_point[num_features + i] + 3 * dy * dot * inv_dist2);
        float dpoisson_z = dpoisson_common * (-fan_point[num_features * 2 + i] + 3 * dz * dot * inv_dist2);

        float grad_x, grad_y, grad_z;

        if (dist < 2. / inv_delta) {
            grad_x = dpoisson_x * reg + poisson * dreg_x;
            grad_y = dpoisson_y * reg + poisson * dreg_y;
            grad_z = dpoisson_z * reg + poisson * dreg_z;
        } else {
            float reg = 1.0;
            grad_x = dpoisson_x * reg;
            grad_y = dpoisson_y * reg;
            grad_z = dpoisson_z * reg;
        }

        if (isnan(grad_x) || isinf(grad_x)) grad_x = 0;
        if (isnan(grad_y) || isinf(grad_y)) grad_y = 0;
        if (isnan(grad_z) || isinf(grad_z)) grad_z = 0;

        pos_grad_query[i] += grad_x;
        pos_grad_query[num_features + i] += grad_y;
        pos_grad_query[num_features * 2 + i] += grad_z;
    }

}
