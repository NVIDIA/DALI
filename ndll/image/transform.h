#ifndef NDLL_TRANSFORM_H_
#define NDLL_TRANSFORM_H_

#include "ndll/common.h"
#include "ndll/error_handling.h"

namespace ndll {

/**
 * @brief Performs resize, crop, & random mirror on the input image on the CPU. Input
 * data is assumed to be stored in HWC layout in memory.
 *
 * This method takes in an optional 'workspace' buffer. Because we perform the resize
 * and mirror in separate steps, and intermediate buffer is needed to store the
 * resutls of the resize before mirroring into the output buffer. Rather than
 * always allocate memory inside the function, we provide the option to pass in
 * this temporary workspace pointer to avoid extra memory allocation. The size
 * of the memory pointed to by 'workspace' should be rsz_h*rsz_w*C bytes
 *
 * Note: We leave the calculate of the resize dimesions & the decision of whether 
 * to mirror the image or not external to the function. With the GPU version of 
 * this function, these params will need to have been calculated before-hand 
 * and, in the case of a batched call, copied to the device. Separating these 
 * parameters from this function will make the API consistent across the CPU
 * & GPU versions.
 */
NDLLError_t ResizeCropMirrorHost(const uint8 *img, int H, int W, int C,
    int rsz_h, int rsz_w, int crop_x, int crop_y, int crop_h, int crop_w,
    bool mirror, uint8 *out_img, NDLLInterpType type = NDLL_INTERP_LINEAR,
    uint8 *workspace = nullptr);

/**
 * @brief Performs resize, crop, & random mirror on the input image on the CPU. Input
 * data is assumed to be stored in HWC layout in memory.
 *
 * 'Fast' ResizeCropMirrorHost does not perform the full image resize. Instead, it
 * takes advantage of the fact that we are going to crop, and backprojects the crop
 * into the input image. We then resize the backprojected crop region to the crop
 * dimensions (crop_w/crop_h), avoiding a significant amount of work on data that
 * would have been cropped away immediately.
 *
 * This method takes in an optional 'workspace' buffer. Because we perform the resize
 * and mirror in separate steps, and intermediate buffer is needed to store the
 * resutls of the resize before mirroring into the output buffer. Rather than
 * always allocate memory inside the function, we provide the option to pass in
 * this temporary workspace pointer to avoid extra memory allocation. The size
 * of the memory pointed to by 'workspace' should be crop_h*crop_w*C bytes
 */
NDLLError_t FastResizeCropMirrorHost(const uint8 *img, int H, int W, int C,
    int rsz_h, int rsz_w, int crop_y, int crop_x, int crop_h, int crop_w,
    bool mirror, uint8 *out_img, NDLLInterpType type = NDLL_INTERP_LINEAR,
    uint8 *workspace = nullptr);
  
/**
 * @brief Performs mean subtraction & stddev division per channel, cast 
 * to output type, and NHWC->NCHW permutation.
 *
 * 'mean' and 'inv_std' are assumed to point to device memory of size `c`.
 * Input data is assumed to be stored in NHWC layout in memory. Output
 * data will be stored in NCHW.
 */
template <typename OUT>
NDLLError_t BatchedNormalizePermute(const uint8 *in_batch,
    int N, int H, int W, int C,  float *mean, float *inv_std,
    OUT *out_batch, cudaStream_t stream);

/**
 * @brief Takes in a jagged batch of images and crops, (optional) mirrors,
 * performs mean subtraction & stddev division per channel, cast to output
 * type, and NHWC->NCHW permutation
 *
 * The crop is performed by offsetting the ptrs in 'in_batch' to the beginning
 * of the crop region, and then passing in the stride of each image so that
 * the kernel can correctly process the crop region.
 *
 * @param in_batch device pointer to pointer to the beginning of the crop 
 * region for each image
 * @param in_strides device pointer to `N` ints whose value is the stride 
 * of each input image
 * @param mirror device pointer to `N` bools whose values indicate whether 
 * the image should be mirrored or not
 * @param N number of elements in the batch
 * @param H output height for all images in the batch
 * @param W output width for all images in the batch
 * @param C number of channels of images in the batch
 * @param mean device pointer of length `C` to the mean to subtract for 
 * each image channel
 * @param std device pointer of length `C` to the inverse std dev. to multiply by 
 * for each image channel
 * @param out_batch pointer of size `N*C*H*W` to store the dense, cropped, 
 * NCHW output batch
 * @param stream cuda stream to operate in
 */
template <typename OUT>
NDLLError_t BatchedCropMirrorNormalizePermute(const uint8 * const *in_batch,
    const int *in_strides, int N, int H, int W, int C, const bool *mirror,
    const float *mean, const float *inv_std, OUT *out_batch, cudaStream_t stream);

/**
 * @brief Validates the parameters for 'BatchedCropMirrorNormalizePermute' 
 * on host
 *
 * All parameters are host-side versions of the arguments to 
 * 'BatchedCropMirrorNormalizePermute'. This method exists so that 
 * the user can efficiently manage memory copies to the GPU, but stil
 * have a method for validating input arguments before calling the 
 * batched function.
 *
 * Checks that...
 * - in_batch device pointers are not nullptr
 * - in_strides values are >= W*C
 * - N > 0, H > 0, W > 0, C == 1 || C == 3
 */
template <typename OUT>
NDLLError_t ValidateBatchedCropMirrorNormalizePermute(const uint8 * const *in_batch,
    const int *in_strides, int N, int H, int W, int C, const bool *mirror,
    const float *mean, const float *inv_std, OUT *out_batch);

/**
 * @brief Resizes an input batch of images. 
 *
 * Note: This API is subject to change. It currently launches a kernel
 * for every image in the batch, but if we move to a fully batched kernel
 * we will likely need more meta-data setup beforehand
 *
 * This method currently uses an npp kernel, to set the stream for this
 * kernel, call 'nppSetStream()' prior to calling.
 */
NDLLError_t BatchedResize(const uint8 **in_batch, int N, int C, const NDLLSize *in_sizes,
    uint8 **out_batch, const NDLLSize *out_sizes, NDLLInterpType type = NDLL_INTERP_LINEAR);

} // namespace ndll

#endif // NDLL_TRANSFORM_H_
