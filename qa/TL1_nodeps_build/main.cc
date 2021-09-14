#include <iostream>
#include "dali/kernels/imgproc/resample/resampling_impl_cpu.h"
#include "dali/kernels/imgproc/resample/resampling_filters.cuh"

int main(int argc, char **argv) {
  // Just using some dali kernel functions
  {
    int32_t idx[1] = { 0 };
    float coeffs[32] = {};
    dali::kernels::ResamplingFilter flt = {};
    flt.coeffs = coeffs;
    flt.num_coeffs = 1;
    dali::kernels::InitializeResamplingFilter(idx, coeffs, 1, 0, 1, flt);
  }

  return 0;
}
