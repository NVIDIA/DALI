// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#ifndef NDLL_PIPELINE_OPERATORS_DISPLACEMENT_FILTER_H_
#define NDLL_PIPELINE_OPERATORS_DISPLACEMENT_FILTER_H_

#include "ndll/common.h"
#include "ndll/pipeline/operator.h"

/**
 * @brief Provides a framework for doing displacement filter operations
 * such as flip, jitter, water, swirl, etc.
 */

namespace ndll {

namespace {
template <typename T, class Displacement>
__global__
void DisplacementKernel(const T *in, T* out, const int N, const int H, const int W, const int C, Displacement displace) {
  // block per image
  for (int n = blockIdx.x; n < N; n += gridDim.x) {
    // thread per pixel
    const T *image_in = in + H * W * C;
    T *image_out = out + H * W * C;
    for (int out_idx = threadIdx.x; out_idx < H * W * C; out_idx += blockDim.x) {
      const int c = out_idx % C;
      const int w = (out_idx / C) % W;
      const int h = (out_idx / W / C);

      // calculate input idx from function
      auto in_idx = displace(h, w, c, H, W, C);

      image_out[out_idx] = image_in[in_idx];
    }
  }
}

}

template <class Displacement, typename Backend>
class DisplacementFilter : public Operator<Backend> {
 public:
  explicit DisplacementFilter(const OpSpec &spec)
    : Operator<Backend>(spec) {

  }

  void RunPerSampleCPU(SampleWorkspace* ws, const int idx) override {
    DataDependentSetup(ws, idx);

    if (IsType<float>(ws->Input<CPUBackend>(idx).type())) {
      PerSampleCPULoop<float>(ws, idx);
    } else if (IsType<uint8_t>(ws->Input<CPUBackend>(idx).type())) {
      PerSampleCPULoop<uint8_t>(ws, idx);
    } else {
      NDLL_FAIL("Unexpected input type");
    }
  }

  void RunBatchedGPU(DeviceWorkspace* ws, const int idx) override {
    DataDependentSetup(ws, idx);

    if (IsType<float>(ws->Input<GPUBackend>(idx).type())) {
      BatchedGPUKernel<float>(ws, idx);
    } else if (IsType<uint8_t>(ws->Input<GPUBackend>(idx).type())) {
      BatchedGPUKernel<uint8_t>(ws, idx);
    } else {
      NDLL_FAIL("Unexpected input type");
    }
  }

  virtual void DataDependentSetup(SampleWorkspace *ws, const int idx) {}
  virtual void DataDependentSetup(DeviceWorkspace *ws, const int idx) {}

 private:
  template <typename T>
  bool PerSampleCPULoop(SampleWorkspace *ws, const int idx) {
    auto& input = ws->Input<CPUBackend>(idx);
    auto *output = ws->Output<CPUBackend>(idx);

    Displacement displace;

    const auto H = input.shape()[0];
    const auto W = input.shape()[1];
    const auto C = input.shape()[2];

    auto *in = input.data<T>();
    auto *out = output->template mutable_data<T>();

    for (Index h = 0; h < H; ++h) {
      for (Index w = 0; w < W; ++w) {
        for (Index c = 0; c < C; ++c) {
          // output idx is set by location
          Index out_idx = h * W * C + w * C + c;
          // input idx is calculated by function
          Index in_idx = displace(h, w, c, H, W, C);

          // copy
          out[out_idx] = in[in_idx];
        }
      }
    }
    return true;
  }

  template <typename T>
  bool BatchedGPUKernel(DeviceWorkspace *ws, const int idx) {
    Displacement displace;

    auto &input = ws->Input<GPUBackend>(idx);
    auto *output = ws->Output<GPUBackend>(idx);

    const auto N = input.ntensor();
    auto shape = input.tensor_shape(0);

    output->ResizeLike(input);

    DisplacementKernel<T, Displacement><<<N, 256, 0, ws->stream()>>>(
        input.template data<T>(), output->template mutable_data<T>(),
        input.ntensor(), shape[0], shape[1], shape[2],
        displace);
    return true;
  }

  USE_OPERATOR_MEMBERS();
};

}  // namespace ndll

#endif  // NDLL_PIPELINE_OPERATORS_DISPLACEMENT_FILTER_H_
