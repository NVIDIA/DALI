#ifndef Y_TO_RGB_H_
#define Y_TO_RGB_H_

#include <nppdefs.h>

// Duplicates the single plane 3x in HWC format (interleaved)
void grayToRgb(const Npp8u *img, int step, Npp8u *dst, int dstStep,
    NppiSize dims, cudaStream_t stream);

#endif // Y_TO_RGB_H_
