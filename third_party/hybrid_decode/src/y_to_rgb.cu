#include "y_to_rgb.h"

namespace {

__global__ void grayToRgbKernel(const Npp8u *img, int step, Npp8u *dst,
    int dstStep, NppiSize dims) {
  // pixel is duplicated in the two subsequent memory locations in dst,
  // try writing in uchar3 vectors
  uchar3 *vec_dst = (uchar3*)dst;
  for (int h = threadIdx.y; h < dims.height; h += blockDim.y) {
    for (int w = threadIdx.x; w < dims.width; w += blockDim.x) {
      unsigned char gray = img[h*step + w];
      uchar3 pixel = {gray, gray, gray};
      vec_dst[h*dstStep + w] = pixel;
    }
  }
}

} // namespace

void grayToRgb(const Npp8u *img, int step, Npp8u *dst, int dstStep,
    NppiSize dims, cudaStream_t stream) {
  // TODO: The launch params on this have a large effect on perf.
  // We should try launching more blocks and maybe adapt block size
  // to image dimensions. We could also make a batched version of this
  // quite easily
  grayToRgbKernel<<<1, dim3(32, 32), 0, stream>>>(img, step, dst, dstStep, dims);
}
