#ifndef NDLL_TRANSFORM_H_
#define NDLL_TRANSFORM_H_

#include "ndll/common.h"
#include "ndll/error_handling.h"

namespace ndll {

/**
 * @brief Performs resize, crop, & random mirror on the input image on the CPU. Input
 * data is assumed to be stored in HWC layour in memory.
 *
 * Note: We leave the calculate of the resize dimesions & the decision of whether 
 * to mirror the image or not external to the function. With the GPU version of 
 * this function, these params will need to have been calculated before-hand 
 * and, in the case of a batched call, copied to the device. Separating these 
 * parameters from this function will make the API consistent across the CPU
 * & GPU versions.
 */
NDLLError_t ResizeCropMirror(const uint8 *image, int h, int w, int c,
    int rsz_h, int rsz_w, int crop_x, int crop_y, int crop_h,
    int crop_w, bool mirror, uint8 *out_img);

} // namespace ndll

#endif // NDLL_TRANSFORM_H_
