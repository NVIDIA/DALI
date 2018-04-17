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

template <typename T, bool per_channel_transform, class Displacement, class Color>
__global__
void DisplacementKernel(const T *in, T* out,
                        const int N, const Index * shapes, const Index pitch,
                        Displacement displace, Color color) {
  // block per image
  for (int n = blockIdx.x; n < N; n += gridDim.x) {
    const int H = shapes[n * pitch + 0];
    const int W = shapes[n * pitch + 1];
    const int C = shapes[n * pitch + 2];
    const Index offset = shapes[n * pitch + 3];
    // thread per pixel
    const T *image_in = in + offset;
    T *image_out = out + offset;
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

template <bool per_channel_transform, int nThreads, class Displacement, class Color>
__global__
void DisplacementKernel_C3u8_a4(const uint8_t *in, uint8_t* out,
                        const int N, const Index * shapes, const Index pitch,
                        Displacement displace, Color color) {
  __shared__ uint8_t scratch[nThreads * 12];
  // block per image
  for (int n = blockIdx.x; n < N; n += gridDim.x) {
    const int H = shapes[n * pitch + 0];
    const int W = shapes[n * pitch + 1];
    const int C = 3;
    const Index offset = shapes[n * pitch + 3];
    // thread per pixel
    const uint8_t * const image_in = in + offset;
    uint32_t * const image_out = reinterpret_cast<uint32_t*>(out + offset);
    const int nElements = (H * W) >> 2;
    const int loopCount = nElements / nThreads;
    uint8_t * const my_scratch = scratch + threadIdx.x * 12;
    uint32_t * const scratch_32 = reinterpret_cast<uint32_t*>(scratch);

    for (int lidx = 0; lidx < loopCount; ++lidx) {
      const int hw0 = (lidx * nThreads + threadIdx.x) * 4;
      uint32_t * const current_image_out = image_out + lidx * nThreads * 3;
      if (per_channel_transform) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const int hw = hw0 + j;
          const int w = hw % W;
          const int h = hw / W;
#pragma unroll
          for (int c = 0; c < 3; ++c) {
            auto tmp_idx = displace(h, w, c, H, W, C);
            my_scratch[j * 3 + c] = color(image_in[tmp_idx], h, w, c, H, W, C);
          }
        }
      } else {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const int hw = hw0 + j;
          const int w = hw % W;
          const int h = hw / W;
          auto tmp_idx = displace(h, w, 0, H, W, C);
          my_scratch[j * 3 + 0] = color(image_in[tmp_idx + 0], h, w, 0, H, W, C);
          my_scratch[j * 3 + 1] = color(image_in[tmp_idx + 1], h, w, 1, H, W, C);
          my_scratch[j * 3 + 2] = color(image_in[tmp_idx + 2], h, w, 2, H, W, C);
        }
      }
      __syncthreads();

#pragma unroll
      for (int i = 0; i < 3; ++i) {
        current_image_out[threadIdx.x + i * nThreads] = scratch_32[threadIdx.x + i * nThreads];
      }
    }

    // The rest, that was not aligned to block boundary
    const int myId = threadIdx.x + loopCount * nThreads;
    if (myId < nElements) {
      const int hw0 = myId * 4;
      if (per_channel_transform) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const int hw = hw0 + j;
          const int w = hw % W;
          const int h = hw / W;
#pragma unroll
          for (int c = 0; c < 3; ++c) {
            auto tmp_idx = displace(h, w, c, H, W, C);
            out[offset + h * W * C + w * C + c] =
              color(image_in[tmp_idx], h, w, c, H, W, C);
          }
        }
      } else {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const int hw = hw0 + j;
          const int w = hw % W;
          const int h = hw / W;
          auto tmp_idx = displace(h, w, 0, H, W, C);
          out[offset + h * W * C + w * C + 0] =
            color(image_in[tmp_idx + 0], h, w, 0, H, W, C);
          out[offset + h * W * C + w * C + 1] =
            color(image_in[tmp_idx + 1], h, w, 1, H, W, C);
          out[offset + h * W * C + w * C + 2] =
            color(image_in[tmp_idx + 2], h, w, 2, H, W, C);
        }
      }
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
class DisplacementFilter : public Operator {
 public:
  explicit DisplacementFilter(const OpSpec &spec)
    : Operator(spec),
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

  static const int nDims = 3;

  template <typename T>
  bool BatchedGPUKernel(DeviceWorkspace *ws, const int idx) {
    auto &input = ws->Input<GPUBackend>(idx);
    auto *output = ws->Output<GPUBackend>(idx);

    const auto N = input.ntensor();
    const int pitch = nDims + 1;  // shape and offset

    meta_cpu.Resize({N, pitch});
    Index * meta = meta_cpu.template mutable_data<Index>();
    meta_gpu.ResizeLike(meta_cpu);
    meta_gpu.template mutable_data<Index>();

    Index offset = 0;
    for (int i = 0; i < N; ++i) {
      const auto& shape = input.tensor_shape(i);
      NDLL_ENFORCE(shape.size() == nDims,
          "All augmented tensors need to have the same number of dimensions");
      Index current_size = nDims != 0 ? 1 : 0;
      for (int j = 0; j < nDims; ++j) {
        meta[i * pitch + j] = shape[j];
        current_size *= shape[j];
      }
      meta[i * pitch + nDims] = offset;
      offset += current_size;
    }

    output->ResizeLike(input);

    NDLL_ENFORCE(pitch == 4,
            "DisplacementKernel requires pitch to be 4.");

    meta_gpu.Copy(meta_cpu, ws->stream());
    // Find if C is the same for all images and
    // what is the maximum power of 2 dividing
    // all H*W
    int C = meta[nDims - 1];  // First element
    uint64_t maxPower2 = (uint64_t)-1;
    for (int i = 0; i < N; ++i) {
      if (C != meta[i * pitch + nDims - 1]) {
        C = -1;  // Not all C are the same
      }
      uint64_t HW = 1;
      for (int j = 0; j < nDims - 1; ++j) {
        HW *= meta[i * pitch + j];
      }
      uint64_t power2 = HW & (-HW);
      maxPower2 = maxPower2 > power2 ? power2 : maxPower2;
    }

    DisplacementKernelLauncher(ws, input.template data<T>(),
                               output->template mutable_data<T>(),
                               input.ntensor(), pitch, C, maxPower2);
    return true;
  }

  template <typename U>
  void DisplacementKernelLauncher(DeviceWorkspace * ws,
                                  const U* in, U* out,
                                  const int N, const int pitch,
                                  const int C, const uint64_t maxPower2) {
    DisplacementKernel<U,
                       per_channel_transform,
                       Displacement>
                      <<<N, 256, 0, ws->stream()>>>(
        in, out, N, meta_gpu.template mutable_data<Index>(),
        pitch, displace_, augment_);
  }

  void DisplacementKernelLauncher(DeviceWorkspace * ws,
                                  const uint8_t* in, uint8_t* out,
                                  const int N, const int pitch,
                                  const int C, const uint64_t maxPower2) {
    if (C == 3 && maxPower2 >= 4) {
      DisplacementKernel_C3u8_a4<per_channel_transform,
                                 256,
                                 Displacement,
                                 Augment>
                                <<<N, 256, 0, ws->stream()>>>(
                                    in, out, N,
                                    meta_gpu.template mutable_data<Index>(),
                                    pitch, displace_, augment_);
    } else {
      DisplacementKernel<uint8_t,
                         per_channel_transform,
                         Displacement>
                        <<<N, 256, 0, ws->stream()>>>(
                            in, out, N,
                            meta_gpu.template mutable_data<Index>(),
                            pitch, displace_, augment_);
    }
  }

  USE_OPERATOR_MEMBERS();

 private:
  Displacement displace_;
  Augment augment_;

  Tensor<CPUBackend> meta_cpu;
  Tensor<GPUBackend> meta_gpu;
};

}  // namespace ndll

#endif  // NDLL_PIPELINE_OPERATORS_DISPLACEMENT_FILTER_H_
