#include "ndll/util/type_conversion.h"

#include <cmath>

namespace ndll {

namespace {
template <typename IN, typename OUT>
__global__ void ConvertKernel(const IN *data, int n, OUT *out) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) {
    out[tid] = (OUT)data[tid];
  }
}
} // namespace ndll

template <typename IN, typename OUT>
void Convert(const IN *data, int n, OUT *out) {
  int block_size = 512;
  int blocks = ceil(float(n) / block_size);
  ConvertKernel<<<blocks, block_size, 0, 0>>>(data, n, out);
}

// Note: These are used in the test suite for output verification, we
// don't care if we do extra copy from T to T.
template void Convert<uint8, double>(const uint8*, int, double*);
template void Convert<float16, double>(const float16*, int, double*);
template void Convert<int, double>(const int*, int, double*);
template void Convert<float, double>(const float*, int, double*);
template void Convert<double, double>(const double*, int, double*);

} // namespace ndll
