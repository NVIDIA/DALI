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

template <typename T, class Displacement, class Color>
__global__
void DisplacementKernel(const T *in, T* out,
                        const int N, const int H, const int W, const int C,
                        Displacement displace, Color color) {
  // block per image
  for (int n = blockIdx.x; n < N; n += gridDim.x) {
    // thread per pixel
    const T *image_in = in + n * H * W * C;
    T *image_out = out + n * H * W * C;
    for (int out_idx = threadIdx.x; out_idx < H * W * C; out_idx += blockDim.x) {
      const int c = out_idx % C;
      const int w = (out_idx / C) % W;
      const int h = (out_idx / W / C);

      // calculate input idx from function
      auto in_idx = displace(h, w, c, H, W, C);

      image_out[out_idx] = color(image_in[in_idx], h, w, c, H, W, C);
    }
  }
}

class ColorIdentity {
 public:
  explicit ColorIdentity(const OpSpec& spec) {}

  template <typename T>
  __host__ __device__
  T operator()(const T in, const Index h, const Index w, const Index c,
               const Index H, const Index W, const Index C) {
    // identity
    return in;
  }

  void Cleanup() {}
};

class DisplacementIdentity {
 public:
  explicit DisplacementIdentity(const OpSpec& spec) {}

  __host__ __device__
  Index operator()(const Index h, const Index w, const Index c,
                   const Index H, const Index W, const Index C) {
    // identity
    return (h * W + w) * C + c;
  }

  void Cleanup() {}
};

template <typename Backend,
          class Displacement = DisplacementIdentity,
          class Augment = ColorIdentity,
          bool per_channel_transform = false>
class DisplacementFilter : public Operator<Backend> {
 public:
  explicit DisplacementFilter(const OpSpec &spec)
    : Operator<Backend>(spec),
      displace_(spec),
      augment_(spec) {}

  ~DisplacementFilter() {
    displace_.Cleanup();
    augment_.Cleanup();
  }

  void RunPerSampleCPU(SampleWorkspace* ws, const int idx) override {
    DataDependentSetup(ws, idx);

    auto &input = ws->Input<CPUBackend>(idx);
    if (IsType<float>(input.type())) {
      PerSampleCPULoop<float>(ws, idx);
    } else if (IsType<uint8_t>(input.type())) {
      PerSampleCPULoop<uint8_t>(ws, idx);
    } else {
      NDLL_FAIL("Unexpected input type " + input.type().name());
    }
  }

  void RunBatchedGPU(DeviceWorkspace* ws, const int idx) override {
    DataDependentSetup(ws, idx);

    auto &input = ws->Input<GPUBackend>(idx);
    if (IsType<float>(input.type())) {
      BatchedGPUKernel<float>(ws, idx);
    } else if (IsType<uint8_t>(input.type())) {
      BatchedGPUKernel<uint8_t>(ws, idx);
    } else {
      NDLL_FAIL("Unexpected input type " + input.type().name());
    }
  }

  /**
   * @brief Do basic input checking and output setup
   * assuming output_shape = input_shape
   */
  virtual void DataDependentSetup(SampleWorkspace *ws, const int idx) {
    auto &input = ws->Input<CPUBackend>(idx);
    auto *output = ws->Output<CPUBackend>(idx);
    output->ResizeLike(input);
  }

  virtual void DataDependentSetup(DeviceWorkspace *ws, const int idx) {
    // check input is valid, resize output
    auto &input = ws->Input<GPUBackend>(idx);
    auto *output = ws->Output<GPUBackend>(idx);
    output->ResizeLike(input);
  }

 private:
  template <typename T>
  bool PerSampleCPULoop(SampleWorkspace *ws, const int idx) {
    auto& input = ws->Input<CPUBackend>(idx);
    auto *output = ws->Output<CPUBackend>(idx);

    const auto H = input.shape()[0];
    const auto W = input.shape()[1];
    const auto C = input.shape()[2];

    auto *in = input.data<T>();
    auto *out = output->template mutable_data<T>();

    for (Index h = 0; h < H; ++h) {
      for (Index w = 0; w < W; ++w) {
        // calculate displacement for all channels at once
        // vs. per-channel
        if (per_channel_transform) {
          for (Index c = 0; c < C; ++c) {
            // output idx is set by location
            Index out_idx = (h * W + w) * C + c;
            // input idx is calculated by function
            Index in_idx = displace_(h, w, c, H, W, C);

            // copy
            out[out_idx] = augment_(in[in_idx], h, w, c, H, W, C);
          }
        } else {
          // output idx is set by location
          Index out_idx = (h * W + w) * C;
          // input idx is calculated by function
          Index in_idx = displace_(h, w, 0, H, W, C);

          // apply transform uniformly across channels
          for (int c = 0; c < C; ++c) {
            out[out_idx+c] = augment_(in[in_idx + c], h, w, c, H, W, C);
          }
        }
      }
    }
    return true;
  }

  template <typename T>
  bool BatchedGPUKernel(DeviceWorkspace *ws, const int idx) {
    auto &input = ws->Input<GPUBackend>(idx);
    auto *output = ws->Output<GPUBackend>(idx);

    const auto N = input.ntensor();
    auto shape = input.tensor_shape(0);

    output->ResizeLike(input);

    DisplacementKernel<T, Displacement><<<N, 256, 0, ws->stream()>>>(
        input.template data<T>(), output->template mutable_data<T>(),
        input.ntensor(), shape[0], shape[1], shape[2],
        displace_, augment_);
    return true;
  }

  USE_OPERATOR_MEMBERS();

 private:
  Displacement displace_;
  Augment augment_;
};

}  // namespace ndll

#endif  // NDLL_PIPELINE_OPERATORS_DISPLACEMENT_FILTER_H_
