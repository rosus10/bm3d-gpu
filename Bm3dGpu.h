#pragma once

#include <opencv2/core.hpp>
#include <cuda_runtime.h>
#include "params.hpp"

// Expose kernels from original CUDA code
extern "C" {
void run_block_matching(const uchar* image, ushort* stacks, uint* num_patches_in_stack,
                        const uint2 image_dim, const uint2 stacks_dim, const Params params,
                        const uint2 start_point, const dim3 num_threads, const dim3 num_blocks,
                        const uint shared_memory_size);

void run_get_block(const uint2 start_point, const uchar* image, const ushort* stacks,
                   const uint* num_patches_in_stack, float* patch_stack, const uint2 image_dim,
                   const uint2 stacks_dim, const Params params, const dim3 num_threads,
                   const dim3 num_blocks);

void run_DCT2D8x8(float* d_transformed_stacks, const float* d_gathered_stacks, const uint size,
                  const dim3 num_threads, const dim3 num_blocks);

void run_hard_treshold_block(const uint2 start_point, float* patch_stack, float* w_P,
                             const uint* num_patches_in_stack, const uint2 stacks_dim,
                             const Params params, const uint sigma,
                             const dim3 num_threads, const dim3 num_blocks,
                             const uint shared_memory_size);

void run_IDCT2D8x8(float* d_gathered_stacks, const float* d_transformed_stacks, const uint size,
                   const dim3 num_threads, const dim3 num_blocks);

void run_aggregate_block(const uint2 start_point, const float* patch_stack, const float* w_P,
                         const ushort* stacks, const float* kaiser_window,
                         float* numerator, float* denominator,
                         const uint* num_patches_in_stack, const uint2 image_dim,
                         const uint2 stacks_dim, const Params params,
                         const dim3 num_threads, const dim3 num_blocks);

void run_aggregate_final(const float* numerator, const float* denominator,
                         const uint2 image_dim, uchar* denoised_image,
                         const dim3 num_threads, const dim3 num_blocks);

void run_wiener_filtering(const uint2 start_point, float* patch_stack,
                          const float* patch_stack_basic, float* w_P,
                          const uint* num_patches_in_stack, uint2 stacks_dim,
                          const Params params, const uint sigma,
                          const dim3 num_threads, const dim3 num_blocks,
                          const uint shared_memory_size);
}

class BM3D; // forward declaration

class Bm3dGpu {
public:
    explicit Bm3dGpu(float sigma = 15.0f, bool two_step = false);
    ~Bm3dGpu();

    cv::Mat operator()(const cv::Mat& src, float sigma = 15.0f);

private:
    BM3D* impl_;
    bool two_step_;
    float default_sigma_;
};
