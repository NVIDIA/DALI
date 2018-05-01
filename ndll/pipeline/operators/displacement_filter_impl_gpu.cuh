// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#ifndef NDLL_PIPELINE_OPERATORS_DISPLACEMENT_FILTER_IMPL_GPU_CUH_
#define NDLL_PIPELINE_OPERATORS_DISPLACEMENT_FILTER_IMPL_GPU_CUH_

#include <random>

#include "ndll/common.h"
#include "ndll/pipeline/operator.h"

namespace ndll {

template <typename T, bool per_channel_transform, class Displacement, NDLLInterpType interp_type>
__global__
void DisplacementKernel(const T *in, T* out,
                        const int N, const Index * shapes,
                        const bool * mask, const Index pitch,
                        Displacement displace) {
  // block per image
  for (int n = blockIdx.x; n < N; n += gridDim.x) {
    const int H = shapes[n * pitch + 0];
    const int W = shapes[n * pitch + 1];
    const int C = shapes[n * pitch + 2];
    const Index offset = shapes[n * pitch + 3];
    const bool m = mask[n];
    // thread per pixel
    const T *image_in = in + offset;
    T *image_out = out + offset;
    for (int out_idx = threadIdx.x; out_idx < H * W * C; out_idx += blockDim.x) {
      if (m) {
        const int c = out_idx % C;
        const int w = (out_idx / C) % W;
        const int h = (out_idx / W / C);

        if (interp_type == NDLL_INTERP_NN) {
          // NN interpolation

          // calculate input idx from function
          Point<Index> p = displace.template operator()<Index>(h, w, c, H, W, C);
          auto in_idx = (p.y * W + p.x) * C + c;

          image_out[out_idx] = image_in[in_idx];
        } else {
          // LINEAR interpolation
          Point<float> p = displace.template operator()<float>(h, w, c, H, W, C);
          T inter_values[4];
          const Index x = p.x;
          const Index y = p.y;
          const Index xp = x < W - 1 ? x + 1 : x;
          const Index yp = y < H - 1 ? y + 1 : y;
          // 0, 0
          inter_values[0] = __ldg(&in[(y * W + x) * C + c]);
          // 1, 0
          inter_values[1] = __ldg(&in[(y * W + xp) * C + c]);
          // 0, 1
          inter_values[2] = __ldg(&in[(yp * W + x) * C + c]);
          // 1, 1
          inter_values[3] = __ldg(&in[(yp * W + xp) * C + c]);
          const float rx = p.x - x;
          const float ry = p.y - y;
          const float mrx = 1 - rx;
          const float mry = 1 - ry;
          image_out[out_idx] = static_cast<T>(
              inter_values[0] * mrx * mry +
              inter_values[1] * rx * mry +
              inter_values[2] * mrx * ry +
              inter_values[3] * rx * ry);
        }
      } else {
        image_out[out_idx] = image_in[out_idx];
      }
    }
  }
}

template <typename T, int C, bool per_channel_transform,
          int nThreads, class Displacement, NDLLInterpType interp_type>
__global__
void DisplacementKernel_aligned32bit(const T *in, T* out,
                        const int N, const Index * shapes,
                        const bool * mask, const Index pitch,
                        Displacement displace) {
  constexpr int nPixelsPerThread = sizeof(uint32_t)/sizeof(T);
  __shared__ T scratch[nThreads * C * nPixelsPerThread];
  // block per image
  for (int n = blockIdx.x; n < N; n += gridDim.x) {
    const int H = shapes[n * pitch + 0];
    const int W = shapes[n * pitch + 1];
    const Index offset = shapes[n * pitch + 3];
    const bool m = mask[n];
    if (m) {
      // thread per pixel
      const T * const image_in = in + offset;
      uint32_t * const image_out = reinterpret_cast<uint32_t*>(out + offset);
      const int nElements = (H * W) / nPixelsPerThread;
      const int loopCount = nElements / nThreads;
      T * const my_scratch = scratch + threadIdx.x * C * nPixelsPerThread;
      uint32_t * const scratch_32 = reinterpret_cast<uint32_t*>(scratch);

      for (int lidx = 0; lidx < loopCount; ++lidx) {
        const int hw0 = (lidx * nThreads + threadIdx.x) * nPixelsPerThread;
        uint32_t * const current_image_out = image_out + lidx * nThreads * C;
        if (per_channel_transform) {
#pragma unroll
          for (int j = 0; j < nPixelsPerThread; ++j) {
            const int hw = hw0 + j;
            const int w = hw % W;
            const int h = hw / W;
#pragma unroll
            for (int c = 0; c < C; ++c) {
              if (interp_type == NDLL_INTERP_NN) {
                Point<Index> p = displace.template operator()<Index>(h, w, c, H, W, C);
                auto tmp_idx = (p.y * W + p.x) * C + c;
                my_scratch[j * C + c] = image_in[tmp_idx];
              } else {
                Point<float> p = displace.template operator()<float>(h, w, c, H, W, C);
                T inter_values[4];
                const Index x = p.x;
                const Index y = p.y;
                const Index xp = x < W - 1 ? x + 1 : x;
                const Index yp = y < H - 1 ? y + 1 : y;
                // 0, 0
                inter_values[0] = __ldg(&image_in[(y * W + x) * C + c]);
                // 1, 0
                inter_values[1] = __ldg(&image_in[(y * W + xp) * C + c]);
                // 0, 1
                inter_values[2] = __ldg(&image_in[(yp * W + x) * C + c]);
                // 1, 1
                inter_values[3] = __ldg(&image_in[(yp * W + xp) * C + c]);
                const float rx = p.x - x;
                const float ry = p.y - y;
                const float mrx = 1 - rx;
                const float mry = 1 - ry;
                my_scratch[j * C + c] = static_cast<T>(
                    inter_values[0] * mrx * mry +
                    inter_values[1] * rx * mry +
                    inter_values[2] * mrx * ry +
                    inter_values[3] * rx * ry);
              }
            }
          }
        } else {
#pragma unroll
          for (int j = 0; j < nPixelsPerThread; ++j) {
            const int hw = hw0 + j;
            const int w = hw % W;
            const int h = hw / W;
            if (interp_type == NDLL_INTERP_NN) {
              Point<Index> p = displace.template operator()<Index>(h, w, 0, H, W, C);
              auto tmp_idx = (p.y * W + p.x) * C;
              for (int c = 0; c < C; ++c) {
                my_scratch[j * C + c] = image_in[tmp_idx + c];
              }
            } else {
              Point<float> p = displace.template operator()<float>(h, w, 0, H, W, C);
              T inter_values[4];
              const Index x = p.x;
              const Index y = p.y;
              const Index xp = x < W - 1 ? x + 1 : x;
              const Index yp = y < H - 1 ? y + 1 : y;
              const float rx = p.x - x;
              const float ry = p.y - y;
              const float mrx = 1 - rx;
              const float mry = 1 - ry;
              for (int c = 0; c < C; ++c) {
                // 0, 0
                inter_values[0] = __ldg(&image_in[(y * W + x) * C + c]);
                // 1, 0
                inter_values[1] = __ldg(&image_in[(y * W + xp) * C + c]);
                // 0, 1
                inter_values[2] = __ldg(&image_in[(yp * W + x) * C + c]);
                // 1, 1
                inter_values[3] = __ldg(&image_in[(yp * W + xp) * C + c]);
                my_scratch[j * C + c] = static_cast<T>(
                    inter_values[0] * mrx * mry +
                    inter_values[1] * rx * mry +
                    inter_values[2] * mrx * ry +
                    inter_values[3] * rx * ry);
              }
            }
          }
        }
        __syncthreads();

#pragma unroll
        for (int i = 0; i < C; ++i) {
          current_image_out[threadIdx.x + i * nThreads] = scratch_32[threadIdx.x + i * nThreads];
        }
      }

      // The rest, that was not aligned to block boundary
      const int myId = threadIdx.x + loopCount * nThreads;
      if (myId < nElements) {
        const int hw0 = myId * nPixelsPerThread;
        if (per_channel_transform) {
#pragma unroll
          for (int j = 0; j < nPixelsPerThread; ++j) {
            const int hw = hw0 + j;
            const int w = hw % W;
            const int h = hw / W;
#pragma unroll
            for (int c = 0; c < C; ++c) {
              if (interp_type == NDLL_INTERP_NN) {
                Point<Index> p = displace.template operator()<Index>(h, w, c, H, W, C);
                auto tmp_idx = (p.y * W + p.x) * C + c;
                out[offset + h * W * C + w * C + c] = image_in[tmp_idx];
              } else {
                // LINEAR interpolation
                Point<float> p = displace.template operator()<float>(h, w, c, H, W, C);
                T inter_values[4];
                const Index x = p.x;
                const Index y = p.y;
                const Index xp = x < W - 1 ? x + 1 : x;
                const Index yp = y < H - 1 ? y + 1 : y;
                // 0, 0
                inter_values[0] = __ldg(&image_in[(y * W + x) * C + c]);
                // 1, 0
                inter_values[1] = __ldg(&image_in[(y * W + xp) * C + c]);
                // 0, 1
                inter_values[2] = __ldg(&image_in[(yp * W + x) * C + c]);
                // 1, 1
                inter_values[3] = __ldg(&image_in[(yp * W + xp) * C + c]);
                const float rx = p.x - x;
                const float ry = p.y - y;
                const float mrx = 1 - rx;
                const float mry = 1 - ry;
                out[offset + h * W * C + w * C + c] = static_cast<T>(
                    inter_values[0] * mrx * mry +
                    inter_values[1] * rx * mry +
                    inter_values[2] * mrx * ry +
                    inter_values[3] * rx * ry);
              }
            }
          }
        } else {
#pragma unroll
          for (int j = 0; j < nPixelsPerThread; ++j) {
            const int hw = hw0 + j;
            const int w = hw % W;
            const int h = hw / W;
            if (interp_type == NDLL_INTERP_NN) {
              Point<Index> p = displace.template operator()<Index>(h, w, 0, H, W, C);
              auto tmp_idx = (p.y * W + p.x) * C;
#pragma unroll
              for (int c = 0; c < C; ++c) {
                out[offset + h * W * C + w * C + c] = image_in[tmp_idx + c];
              }
            } else {
              // LINEAR interpolation
              Point<float> p = displace.template operator()<float>(h, w, 0, H, W, C);
              T inter_values[4];
              const Index x = p.x;
              const Index y = p.y;
              const Index xp = x < W - 1 ? x + 1 : x;
              const Index yp = y < H - 1 ? y + 1 : y;
              const float rx = p.x - x;
              const float ry = p.y - y;
              const float mrx = 1 - rx;
              const float mry = 1 - ry;
              for (int c = 0; c < C; ++c) {
                // 0, 0
                inter_values[0] = __ldg(&image_in[(y * W + x) * C + c]);
                // 1, 0
                inter_values[1] = __ldg(&image_in[(y * W + xp) * C + c]);
                // 0, 1
                inter_values[2] = __ldg(&image_in[(yp * W + x) * C + c]);
                // 1, 1
                inter_values[3] = __ldg(&image_in[(yp * W + xp) * C + c]);
                out[offset + h * W * C + w * C + c] = static_cast<T>(
                    inter_values[0] * mrx * mry +
                    inter_values[1] * rx * mry +
                    inter_values[2] * mrx * ry +
                    inter_values[3] * rx * ry);
              }
            }
          }
        }
      }
    } else {
      const uint32_t * const image_in = reinterpret_cast<const uint32_t *>(in + offset);
      uint32_t * const image_out = reinterpret_cast<uint32_t *>(out + offset);
      const int nElements = (H * W * C) / nPixelsPerThread;
      for (int i = threadIdx.x; i < nElements; i += nThreads) {
        image_out[i] = image_in[i];
      }
    }
  }
}

template <class Displacement,
          bool per_channel_transform>
class DisplacementFilter<GPUBackend, Displacement,
                         per_channel_transform> : public Operator<GPUBackend> {
 public:
  explicit DisplacementFilter(const OpSpec &spec) :
      Operator(spec),
      displace_(spec),
      interp_type_(spec.GetArgument<NDLLInterpType>("interp_type")),
      rand_gen_(spec.GetArgument<int>("seed")),
      dis(spec.GetArgument<float>("probability")) {
    NDLL_ENFORCE(interp_type_ == NDLL_INTERP_NN || interp_type_ == NDLL_INTERP_LINEAR,
        "Unsupported interpolation type, only NN and LINEAR are supported for this operation");
  }

  virtual ~DisplacementFilter() {
     displace_.Cleanup();
  }

  void RunImpl(DeviceWorkspace* ws, const int idx) override {
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

  virtual void DataDependentSetup(DeviceWorkspace *ws, const int idx) {
    // check input is valid, resize output
    auto &input = ws->Input<GPUBackend>(idx);
    auto *output = ws->Output<GPUBackend>(idx);
    output->ResizeLike(input);
  }

  void SetupSharedSampleParams(DeviceWorkspace *ws) override {
    mask_.Resize({batch_size_});
    mask_.mutable_data<bool>();

    for (int i = 0; i < batch_size_; ++i) {
      mask_.mutable_data<bool>()[i] = dis(rand_gen_);
    }
    mask_gpu_.ResizeLike(mask_);
    mask_gpu_.template mutable_data<bool>();
    mask_gpu_.Copy(mask_, ws->stream());
  }

  USE_OPERATOR_MEMBERS();

 private:
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

    switch (interp_type_) {
      case NDLL_INTERP_NN:
        DisplacementKernelLauncher<T, NDLL_INTERP_NN>(ws, input.template data<T>(),
            output->template mutable_data<T>(),
            input.ntensor(), pitch, C, maxPower2);
        break;
      case NDLL_INTERP_LINEAR:
        DisplacementKernelLauncher<T, NDLL_INTERP_LINEAR>(ws, input.template data<T>(),
            output->template mutable_data<T>(),
            input.ntensor(), pitch, C, maxPower2);
        break;
      default:
        NDLL_FAIL("Unsupported interpolation type,"
            " only NN and LINEAR are supported for this operation");
    }

    return true;
  }

  template <typename U, NDLLInterpType interp_type>
  void DisplacementKernelLauncher(DeviceWorkspace * ws,
                                  const U* in, U* out,
                                  const int N, const int pitch,
                                  const int C, const uint64_t maxPower2) {
    if (maxPower2 >= sizeof(uint32_t)/sizeof(U)) {
      switch (C) {
        case 1:
          DisplacementKernel_aligned32bit<U, 1, per_channel_transform,
            256, Displacement, interp_type>
              <<<N, 256, 0, ws->stream()>>>(
                  in, out, N,
                  meta_gpu.template mutable_data<Index>(),
                  mask_gpu_.template mutable_data<bool>(),
                  pitch, displace_);
          return;
        case 3:
          DisplacementKernel_aligned32bit<U, 3, per_channel_transform,
            256, Displacement, interp_type>
              <<<N, 256, 0, ws->stream()>>>(
                  in, out, N,
                  meta_gpu.template mutable_data<Index>(),
                  mask_gpu_.template mutable_data<bool>(),
                  pitch, displace_);
          return;
        default:
          break;
      }
    }
    DisplacementKernel<U, per_channel_transform, Displacement, interp_type>
      <<<N, 256, 0, ws->stream()>>>(
          in, out, N,
          meta_gpu.template mutable_data<Index>(),
          mask_gpu_.template mutable_data<bool>(),
          pitch, displace_);
  }


  Displacement displace_;
  NDLLInterpType interp_type_;

  Tensor<CPUBackend> meta_cpu;
  Tensor<GPUBackend> meta_gpu;

  std::mt19937 rand_gen_;
  Tensor<CPUBackend> mask_;
  Tensor<GPUBackend> mask_gpu_;
  std::bernoulli_distribution dis;
};
}  // namespace ndll

#endif  // NDLL_PIPELINE_OPERATORS_DISPLACEMENT_FILTER_IMPL_GPU_CUH_
