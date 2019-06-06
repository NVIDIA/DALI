// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef DALI_PIPELINE_OPERATORS_DISPLACEMENT_DISPLACEMENT_FILTER_IMPL_GPU_CUH_
#define DALI_PIPELINE_OPERATORS_DISPLACEMENT_DISPLACEMENT_FILTER_IMPL_GPU_CUH_

#include "dali/core/common.h"
#include "dali/pipeline/operators/displacement/displacement_filter.h"

namespace dali {

template <typename T, class Displacement, DALIInterpType interp_type>
__device__ inline T GetPixelValueSingleC(const Index h, const Index w, const Index c,
                                         const Index H, const Index W, const Index C,
                                         const T * input,
                                         Displacement& displace, const T fill_value) {  // NOLINT
  T ret;
  if (interp_type == DALI_INTERP_NN) {
    // NN interpolation

    // calculate input idx from function
    Point<Index> p = displace.template operator()<Index>(h, w, c, H, W, C);
    if (p.x >= 0 && p.y >= 0) {
      auto in_idx = (p.y * W + p.x) * C + c;

      ret = input[in_idx];
    } else {
      ret = fill_value;
    }
  } else {
    // LINEAR interpolation
    Point<float> p = displace.template operator()<float>(h, w, c, H, W, C);
    if (p.x >= 0 && p.y >= 0) {
      T inter_values[4];
      const Index x = p.x;
      const Index y = p.y;
      const Index xp = x < W - 1 ? x + 1 : x;
      const Index yp = y < H - 1 ? y + 1 : y;
      // 0, 0
      inter_values[0] = __ldg(&input[(y * W + x) * C + c]);
      // 1, 0
      inter_values[1] = __ldg(&input[(y * W + xp) * C + c]);
      // 0, 1
      inter_values[2] = __ldg(&input[(yp * W + x) * C + c]);
      // 1, 1
      inter_values[3] = __ldg(&input[(yp * W + xp) * C + c]);
      const float rx = p.x - x;
      const float ry = p.y - y;
      const float mrx = 1 - rx;
      const float mry = 1 - ry;
      ret = static_cast<T>(
          inter_values[0] * mrx * mry +
          inter_values[1] * rx * mry +
          inter_values[2] * mrx * ry +
          inter_values[3] * rx * ry);
    } else {
      ret = fill_value;
    }
  }

  return ret;
}

template <typename T, class Displacement, DALIInterpType interp_type>
__device__ inline void GetPixelValueMultiC(const Index h, const Index w,
                                           const Index H, const Index W, const Index C,
                                           const T * input, T * output,
                                           Displacement& displace, const T fill_value) {  // NOLINT
  if (interp_type == DALI_INTERP_NN) {
    // NN interpolation

    // calculate input idx from function
    Point<Index> p = displace.template operator()<Index>(h, w, 0, H, W, C);
    if (p.x >= 0 && p.y >= 0) {
      auto in_idx = (p.y * W + p.x) * C;

      for (int c = 0; c < C; ++c) {
        output[c] = input[in_idx + c];
      }
    } else {
      for (int c = 0; c < C; ++c) {
        output[c] = fill_value;
      }
    }
  } else {
    // LINEAR interpolation
    Point<float> p = displace.template operator()<float>(h, w, 0, H, W, C);
    if (p.x >= 0 && p.y >= 0) {
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
        inter_values[0] = __ldg(&input[(y * W + x) * C + c]);
        // 1, 0
        inter_values[1] = __ldg(&input[(y * W + xp) * C + c]);
        // 0, 1
        inter_values[2] = __ldg(&input[(yp * W + x) * C + c]);
        // 1, 1
        inter_values[3] = __ldg(&input[(yp * W + xp) * C + c]);
        output[c] = static_cast<T>(
            inter_values[0] * mrx * mry +
            inter_values[1] * rx * mry +
            inter_values[2] * mrx * ry +
            inter_values[3] * rx * ry);
      }
    } else {
      for (int c = 0; c < C; ++c) {
        output[c] = fill_value;
      }
    }
  }
}

template <class Displacement, bool has_param = HasParam<Displacement>::value>
struct PrepareParam {
 public:
  __device__ __host__ inline void operator() (Displacement &, const void *, const int) {}
};

template <class Displacement>
struct PrepareParam<Displacement, true> {
 public:
  __device__ __host__ inline void operator() (Displacement &displace,  // NOLINT(*)
                                              const void * raw_params, const int n) {
    const typename Displacement::Param * const params =
      reinterpret_cast<const typename Displacement::Param *>(raw_params);
    displace.param = params[n];
  }
};

template <typename T, bool per_channel_transform, class Displacement, DALIInterpType interp_type>
__global__
void DisplacementKernel(const T *in, T* out,
                        const int N, const Index * shapes, const bool has_mask,
                        const int * mask, const void * raw_params, const Index pitch,
                        const T fill_value,
                        Displacement displace) {
  // block per image
  for (int n = blockIdx.x; n < N; n += gridDim.x) {
    const int H = shapes[n * pitch + 0];
    const int W = shapes[n * pitch + 1];
    const int C = shapes[n * pitch + 2];
    const Index offset = shapes[n * pitch + 3];
    const bool m = !has_mask || mask[n];
    PrepareParam<Displacement> pp;
    pp(displace, raw_params, n);
    // thread per pixel
    const T *image_in = in + offset;
    T *image_out = out + offset;
    for (int out_idx = threadIdx.x; out_idx < H * W * C; out_idx += blockDim.x) {
      if (m) {
        const int c = out_idx % C;
        const int w = (out_idx / C) % W;
        const int h = (out_idx / W / C);

        image_out[out_idx] = GetPixelValueSingleC<T, Displacement, interp_type>(h, w, c, H, W, C,
            image_in, displace, fill_value);
      } else {
        image_out[out_idx] = image_in[out_idx];
      }
    }
  }
}

template <typename T, int C, bool per_channel_transform,
          int nThreads, class Displacement, DALIInterpType interp_type>
__global__
void DisplacementKernel_aligned32bit(const T *in, T* out,
                        const size_t N, const Index * shapes,
                        const bool has_mask, const int * mask,
                        const void * raw_params,
                        const Index pitch,
                        const T fill_value,
                        Displacement displace) {
  constexpr int nPixelsPerThread = sizeof(uint32_t)/sizeof(T);
  __shared__ T scratch[nThreads * C * nPixelsPerThread];
  // block per image
  for (size_t n = blockIdx.x; n < N; n += gridDim.x) {
    const int H = shapes[n * pitch + 0];
    const int W = shapes[n * pitch + 1];
    const Index offset = shapes[n * pitch + 3];
    const bool m = !has_mask || mask[n];
    PrepareParam<Displacement> pp;
    pp(displace, raw_params, n);
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
              my_scratch[j * C + c] = GetPixelValueSingleC<T, Displacement, interp_type>(h, w, c,
                  H, W, C,
                  image_in, displace, fill_value);
            }
          }
        } else {
#pragma unroll
          for (int j = 0; j < nPixelsPerThread; ++j) {
            const int hw = hw0 + j;
            const int w = hw % W;
            const int h = hw / W;
            GetPixelValueMultiC<T, Displacement, interp_type>(h, w, H, W, C,
                image_in, my_scratch + j * C, displace, fill_value);
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
              out[offset + h * W * C + w * C + c] =
                GetPixelValueSingleC<T, Displacement, interp_type>(
                  h, w, c,
                  H, W, C,
                  image_in, displace, fill_value);
            }
          }
        } else {
#pragma unroll
          for (int j = 0; j < nPixelsPerThread; ++j) {
            const int hw = hw0 + j;
            const int w = hw % W;
            const int h = hw / W;
            GetPixelValueMultiC<T, Displacement, interp_type>(h, w, H, W, C,
                image_in, out + offset + h * W * C + w * C, displace, fill_value);
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
      interp_type_(spec.GetArgument<DALIInterpType>("interp_type")) {
    has_mask_ = spec.HasTensorArgument("mask");
    DALI_ENFORCE(interp_type_ == DALI_INTERP_NN || interp_type_ == DALI_INTERP_LINEAR,
        "Unsupported interpolation type, only NN and LINEAR are supported for this operation");

    if (!spec.TryGetArgument<float>(fill_value_, "fill_value")) {
      int int_value = 0;
      if (!spec.TryGetArgument<int>(int_value, "fill_value")) {
        DALI_FAIL("Invalid type of argument \"fill_value\". Expected int or float");
      }
      fill_value_ = int_value;
    }
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
      DALI_FAIL("Unexpected input type " + input.type().name());
    }
  }

  virtual void DataDependentSetup(DeviceWorkspace *ws, const int idx) {
    // check input is valid, resize output
    auto &input = ws->Input<GPUBackend>(idx);
    auto &output = ws->Output<GPUBackend>(idx);
    output.ResizeLike(input);
  }

  template <typename U = Displacement>
  typename std::enable_if<HasParam<U>::value>::type PrepareDisplacement(DeviceWorkspace *ws) {
    params_.Resize({batch_size_});
    params_.mutable_data<typename U::Param>();

    for (int i = 0; i < batch_size_; ++i) {
      typename U::Param &p = params_.mutable_data<typename U::Param>()[i];
      displace_.Prepare(&p, spec_, ws, i);
    }
    params_gpu_.ResizeLike(params_);
    params_gpu_.Copy(params_, ws->stream());
  }

  template <typename U = Displacement>
  typename std::enable_if<!HasParam<U>::value>::type PrepareDisplacement(DeviceWorkspace *) {}

  void SetupSharedSampleParams(DeviceWorkspace *ws) override {
    if (has_mask_) {
      const auto &mask = ws->ArgumentInput("mask");
      mask_gpu_.ResizeLike(mask);
      mask_gpu_.template mutable_data<int>();
      mask_gpu_.Copy(mask, ws->stream());
    }
    PrepareDisplacement(ws);
  }

  USE_OPERATOR_MEMBERS();
  using Operator<GPUBackend>::RunImpl;

 private:
  static const size_t nDims = 3;

  template <typename T>
  bool BatchedGPUKernel(DeviceWorkspace *ws, const int idx) {
    auto &input = ws->Input<GPUBackend>(idx);
    auto &output = ws->Output<GPUBackend>(idx);

    const auto N = input.ntensor();
    const int pitch = nDims + 1;  // shape and offset

    meta_cpu.Resize({static_cast<int>(N), pitch});
    Index * meta = meta_cpu.template mutable_data<Index>();
    meta_gpu.ResizeLike(meta_cpu);
    meta_gpu.template mutable_data<Index>();

    Index offset = 0;
    for (size_t i = 0; i < N; ++i) {
      const auto& shape = input.tensor_shape(i);
      DALI_ENFORCE(shape.size() == nDims,
          "All augmented tensors need to have the same number of dimensions");
      Index current_size = nDims != 0 ? 1 : 0;
      for (size_t j = 0; j < nDims; ++j) {
        meta[i * pitch + j] = shape[j];
        current_size *= shape[j];
      }
      meta[i * pitch + nDims] = offset;
      offset += current_size;
    }

    output.ResizeLike(input);

    DALI_ENFORCE(pitch == 4,
            "DisplacementKernel requires pitch to be 4.");

    meta_gpu.Copy(meta_cpu, ws->stream());
    // Find if C is the same for all images and
    // what is the maximum power of 2 dividing
    // all H*W
    int C = meta[nDims - 1];  // First element
    uint64_t maxPower2 = (uint64_t)-1;
    for (size_t i = 0; i < N; ++i) {
      if (C != meta[i * pitch + nDims - 1]) {
        C = -1;  // Not all C are the same
      }
      uint64_t HW = 1;
      for (size_t j = 0; j < nDims - 1; ++j) {
        HW *= meta[i * pitch + j];
      }
      uint64_t power2 = HW & (-HW);
      maxPower2 = maxPower2 > power2 ? power2 : maxPower2;
    }

    switch (interp_type_) {
      case DALI_INTERP_NN:
        DisplacementKernelLauncher<T, DALI_INTERP_NN>(ws, input.template data<T>(),
            output.template mutable_data<T>(),
            input.ntensor(), pitch, C, maxPower2);
        break;
      case DALI_INTERP_LINEAR:
        DisplacementKernelLauncher<T, DALI_INTERP_LINEAR>(ws, input.template data<T>(),
            output.template mutable_data<T>(),
            input.ntensor(), pitch, C, maxPower2);
        break;
      default:
        DALI_FAIL("Unsupported interpolation type,"
            " only NN and LINEAR are supported for this operation");
    }

    return true;
  }

  template <typename U, DALIInterpType interp_type>
  void DisplacementKernelLauncher(DeviceWorkspace * ws,
                                  const U* in, U* out,
                                  const size_t N, const int pitch,
                                  const int C, const uint64_t maxPower2) {
    void * param_ptr = params_gpu_.capacity() > 0 ? params_gpu_.raw_mutable_data() : nullptr;
    if (maxPower2 >= sizeof(uint32_t)/sizeof(U)) {
      switch (C) {
        case 1:
          DisplacementKernel_aligned32bit<U, 1, per_channel_transform,
            256, Displacement, interp_type>
              <<<N, 256, 0, ws->stream()>>>(
                  in, out, N,
                  meta_gpu.template mutable_data<Index>(),
                  has_mask_,
                  mask_gpu_.template mutable_data<int>(),
                  param_ptr,
                  pitch, fill_value_, displace_);
          return;
        case 3:
          DisplacementKernel_aligned32bit<U, 3, per_channel_transform,
            256, Displacement, interp_type>
              <<<N, 256, 0, ws->stream()>>>(
                  in, out, N,
                  meta_gpu.template mutable_data<Index>(),
                  has_mask_,
                  mask_gpu_.template mutable_data<int>(),
                  param_ptr,
                  pitch, fill_value_, displace_);
          return;
        default:
          break;
      }
    }
    DisplacementKernel<U, per_channel_transform, Displacement, interp_type>
      <<<N, 256, 0, ws->stream()>>>(
          in, out, N,
          meta_gpu.template mutable_data<Index>(),
          has_mask_,
          mask_gpu_.template mutable_data<int>(),
          param_ptr,
          pitch, fill_value_, displace_);
  }

  Displacement displace_;
  DALIInterpType interp_type_;
  float fill_value_;

  Tensor<CPUBackend> meta_cpu;
  Tensor<GPUBackend> meta_gpu;

  bool has_mask_;
  Tensor<GPUBackend> mask_gpu_;

  Tensor<CPUBackend> params_;
  Tensor<GPUBackend> params_gpu_;
};
}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_DISPLACEMENT_DISPLACEMENT_FILTER_IMPL_GPU_CUH_
