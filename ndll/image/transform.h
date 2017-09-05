#ifndef NDLL_TRANSFORM_H_
#define NDLL_TRANSFORM_H_

#include "ndll/common.h"
#include "ndll/error_handling.h"

namespace ndll {

/**
 * @brief Performs resize, crop, & random mirror on the input image on the CPU. Input
 * data is assumed to be stored in HWC layout in memory.
 *
 * Note: We leave the calculate of the resize dimesions & the decision of whether 
 * to mirror the image or not external to the function. With the GPU version of 
 * this function, these params will need to have been calculated before-hand 
 * and, in the case of a batched call, copied to the device. Separating these 
 * parameters from this function will make the API consistent across the CPU
 * & GPU versions.
 */
NDLLError_t ResizeCropMirrorHost(const uint8 *image, int H, int W, int C,
    int rsz_h, int rsz_w, int crop_x, int crop_y, int crop_h, int crop_w,
    bool mirror, uint8 *out_img);

/**
 * @brief Performs mean subtraction & stddev division per channel, cast 
 * to output type, and NHWC->NCHW permutation.
 *
 * 'mean' and 'std' are assumed to point to device memory of size `c`.
 * Input data is assumed to be stored in NHWC layout in memory. Output
 * data will be stored in NCHW.
 */
template <typename OUT>
NDLLError_t BatchedNormalizePermute(const uint8 *image_batch,
    int N, int H, int W, int C,  float *mean, float *std,
    OUT *out_batch, cudaStream_t stream);

} // namespace ndll

#endif // NDLL_TRANSFORM_H_
