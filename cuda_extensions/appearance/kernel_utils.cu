#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#define ONE_OVER_PI4 0.079577471f
#define TWO_OVER_SQRT_PI 1.1283791671f
#define PI_NEG_3OVER2_POWER 0.179587122125f


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
    const float* __restrict__ fa_point,
    float* __restrict__ dfa_point,
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
    const float inv_delta_2 = inv_delta * inv_delta;

    float weight = ONE_OVER_PI4 / dist2;
    if (dist2 < 4. / inv_delta_2) {
        const float t2 = dist2 * inv_delta_2;
        const float s = erff(t2);
        weight *= s;

        const float dinv_coeff = expf(-t2 * t2) * inv_delta * PI_NEG_3OVER2_POWER;
        atomicAdd(dinv_delta, dinv_coeff * fa_point[feature_index] * df_query[feature_index]);
    }

    atomicAdd(&dfa_point[feature_index], df_query[feature_index] * weight);

}


__device__ __forceinline__ void update_features(
    const float* __restrict__ query,
    const float* __restrict__ point,
    float* __restrict__ f_query,
    const float* __restrict__ fa_point,
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
    const float inv_delta_2 = inv_delta * inv_delta;

    float weight = ONE_OVER_PI4 / dist2;
    if (dist2 < 4. / inv_delta_2) {
        const float t2 = dist2 * inv_delta_2;
        const float s = erff(t2);
        weight *= s;
    }

    atomicAdd(&f_query[feature_index], fa_point[feature_index] * weight);
}