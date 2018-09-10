#ifndef DALI_TIFF_H
#define DALI_TIFF_H

#include <opencv2/opencv.hpp>

#include "dali/common.h"
#include "dali/error_handling.h"
#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/data/tensor.h"

namespace dali {

/**
 * @brief Returns 'true' if input compressed image is a tiff
 */
extern bool CheckIsTiff(const unsigned char *tiff);


/**
 * @brief Get dimensions of tiff encoded image
 */
extern DALIError_t GetTiffImageDims(const unsigned char *tiff, int size, int *h, int *w);

/**
 * @brief Decodes 'tiff' into the buffer pointed to by 'image'
 */
extern DALIError_t
DecodeTiffHost(const unsigned char *tiff, int size, DALIImageType image_type, Tensor<CPUBackend> *output);

} // namespace dali

#endif //DALI_TIFF_H
