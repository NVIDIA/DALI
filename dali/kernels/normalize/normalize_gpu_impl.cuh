// Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef DALI_KERNELS_NORMALIZE_NORMALIZE_GPU_IMPL_CUH_
#define DALI_KERNELS_NORMALIZE_NORMALIZE_GPU_IMPL_CUH_

#include <cuda_runtime.h>
#include <utility>
#include <string>
#include "dali/core/convert.h"
#include "dali/core/cuda_rt_utils.h"
#include "dali/core/format.h"
#include "dali/core/small_vector.h"
#include "dali/core/tensor_shape_print.h"
#include "dali/kernels/reduce/reduce_drop_dims.h"
#include "dali/kernels/kernel.h"

namespace dali {
namespace kernels {

namespace normalize_impl {

using reduce_impl::DropDims;

template <int max_dims, typename Out, typename In, typename Base, typename Scale>
struct NormalizeNonScalar {
  Out *__restrict__ out;
  const In *__restrict__ in;
  int64_t size;
  const Base *__restrict__ base;
  const Scale *__restrict__ scale;
  DropDims<max_dims> dd;

  DALI_HOST_DEV DALI_FORCEINLINE
  void apply(int64_t offset, float global_scale, float global_shift) {
    int64_t param_offset = dd.reindex(offset);
  #ifdef __CUDA_ARCH__
    float sub = __ldg(base + param_offset);
    float mul = __ldg(scale + param_offset) * global_scale;
  #else
    float sub = base[param_offset];
    float mul = scale[param_offset] * global_scale;
  #endif
    out[offset] = ConvertSat<Out>(fmaf(in[offset] - sub, mul, global_shift));
  }
};

template <int max_dims, typename Out, typename In, typename Base>
struct NormalizeScalarScale {
  Out *__restrict__ out;
  const In *__restrict__ in;
  int64_t size;
  const Base *__restrict__ base;
  DropDims<max_dims> dd;
  float scale;

  DALI_HOST_DEV DALI_FORCEINLINE
  void apply(int64_t offset, float global_scale, float global_shift) {
    int64_t param_offset = dd.reindex(offset);
  #ifdef __CUDA_ARCH__
    float sub = __ldg(base + param_offset);
  #else
    float sub = base[param_offset];
  #endif
    float mul = scale * global_scale;
    out[offset] = ConvertSat<Out>(fmaf(in[offset] - sub, mul, global_shift));
  }
};

template <int max_dims, typename Out, typename In, typename Scale>
struct NormalizeScalarBase {
  Out *__restrict__ out;
  const In *__restrict__ in;
  int64_t size;
  const Scale *__restrict__ scale;
  DropDims<max_dims> dd;
  float base;

  DALI_HOST_DEV DALI_FORCEINLINE
  void apply(int64_t offset, float global_scale, float global_shift) {
    int64_t param_offset = dd.reindex(offset);
  #ifdef __CUDA_ARCH__
    float mul = __ldg(scale + param_offset) * global_scale;
  #else
    float mul = scale[param_offset] * global_scale;
  #endif
    out[offset] = ConvertSat<Out>(fmaf(in[offset] - base, mul, global_shift));
  }
};

template <typename Out, typename In>
struct NormalizeScalar {
  Out *__restrict__ out;
  const In *__restrict__ in;
  int64_t size;
  float base, scale;

  DALI_HOST_DEV DALI_FORCEINLINE
  void apply(int64_t offset, float global_scale, float global_shift) {
    out[offset] = ConvertSat<Out>(fmaf(in[offset] - base, scale * global_scale, global_shift));
  }
};

/**
 * @brief This variant is used when standard deviation is externally provided and needs to
 *        be regularized and inversed.
 *
 * The output elements are calculated as:
 * mul = 1 / sqrt(square(stddev[param_offset]) + epsilon)
 * (in[offset] - mean[param_offset]) * mul * scale + shift
 */
template <int max_dims, typename Out, typename In, typename Mean, typename StdDev>
struct NormalizeInvStdDevNonScalar {
  Out *__restrict__ out;
  const In *__restrict__ in;
  int64_t size;
  const Mean *__restrict__ base;
  const StdDev *__restrict__ scale;
  DropDims<max_dims> dd;

  DALI_HOST_DEV DALI_FORCEINLINE
  void apply(int64_t offset, float epsilon, float global_scale, float global_shift) {
    int64_t param_offset = dd.reindex(offset);
  #ifdef __CUDA_ARCH__
    float mean = __ldg(base + param_offset);
    float stddev = __ldg(scale + param_offset);
  #else
    float mean = base[param_offset];
    float stddev = scale[param_offset];
  #endif
    float x = fmaf(stddev, stddev, epsilon);
    float mul = x ? rsqrt(x) * global_scale : 0;
    out[offset] = ConvertSat<Out>(fmaf(in[offset] - mean, mul, global_shift));
  }
};

/**
 * @brief This variant is used when standard deviation is externally provided and needs to
 *        be regularized and inversed.
 *
 * The output elements are calculated as:
 * mul = 1 / sqrt(square(stddev[param_offset]) + epsilon)
 * (in[offset] - mean[param_offset]) * mul * scale + shift
 */
template <int max_dims, typename Out, typename In, typename StdDev>
struct NormalizeInvStdDevScalarMean {
  Out *__restrict__ out;
  const In *__restrict__ in;
  int64_t size;
  const StdDev *__restrict__ scale;
  DropDims<max_dims> dd;
  float base;

  DALI_HOST_DEV DALI_FORCEINLINE
  void apply(int64_t offset, float epsilon, float global_scale, float global_shift) {
    int64_t param_offset = dd.reindex(offset);
  #ifdef __CUDA_ARCH__
    float stddev = __ldg(scale + param_offset);
  #else
    float stddev = scale[param_offset];
  #endif
    float x = fmaf(stddev, stddev, epsilon);
    float mul = x ? rsqrt(x) * global_scale : 0;
    out[offset] = ConvertSat<Out>(fmaf(in[offset] - base, mul, global_shift));
  }
};

template <typename NormalizeParams>
__global__ void NormalizeKernel(const NormalizeParams *sample_params,
                                float scale, float shift) {
  auto params = sample_params[blockIdx.y];
  int64_t start_ofs = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t grid_stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t ofs = start_ofs; ofs < params.size; ofs += grid_stride) {
    params.apply(ofs, scale, shift);
  }
}

template <typename NormalizeParams>
__global__ void NormalizeInvStdDevKernel(const NormalizeParams *sample_params,
                                         float epsilon, float scale, float shift) {
  auto params = sample_params[blockIdx.y];
  int64_t start_ofs = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t grid_stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t ofs = start_ofs; ofs < params.size; ofs += grid_stride) {
    params.apply(ofs, epsilon, scale, shift);
  }
}

/**
 * @brief Normalizes values according to base and scale given as tensors or scalars.
 *
 * The output is calculated as either:
 * ```
 * out[data_idx] = (in[data_idx] - base[param_idx]) * scale[param_idx] * global_scale + shift
 * ```
 * or
 * ```
 * scale = global_scale / sqrt(stddev[param_idx]^2 + epsilon)
 * out[data_idx] = (in[data_idx] - base[param_idx]) * scale + shift
 * ```
 * Optionally, scale and/or base can be scalar values.
 *
 * @tparam Out    output element type
 * @tparam In     input element type
 * @tparam Base   element type of base/mean tensor
 * @tparam Scale  element type of scale/stddev tensor
 */
template <typename Out, typename In, typename Base, typename Scale>
class NormalizeImplGPU {
  static constexpr const int kMaxDims = 4;

  using Op_NonScalar = NormalizeNonScalar<kMaxDims, Out, In, Base, Scale>;
  using Op_ScalarBase = NormalizeScalarBase<kMaxDims, Out, In, Scale>;
  using Op_ScalarScale = NormalizeScalarScale<kMaxDims, Out, In, Base>;
  using Op_Scalar = NormalizeScalar<Out, In>;
  using Op_InvStdDevNonScalar = NormalizeInvStdDevNonScalar<kMaxDims, Out, In, Base, Scale>;
  using Op_InvStdDevScalarBase = NormalizeInvStdDevScalarMean<kMaxDims, Out, In, Scale>;

 public:
  KernelRequirements Setup(KernelContext &ctx,
                           const TensorListShape<> &data_shape,
                           const TensorListShape<> &param_shape,
                           bool scalar_base,
                           bool scalar_scale,
                           bool scale_is_stddev) {
    ndim_ = data_shape.sample_dim();
    num_samples_ = data_shape.num_samples();
    if (!scalar_base || !scalar_scale) {
      if (param_shape.sample_dim() != ndim_) {
        throw std::invalid_argument("Normalization parameters must have the same "
          "dimensionality as the data");
      }
      if (param_shape.num_samples() != data_shape.num_samples() && param_shape.num_samples() != 1) {
        throw std::invalid_argument("Normalization parameters must have either the same number "
          "of samples as the input or just one sample.");
      }

      axes_.clear();
      for (int d = 0; d < ndim_; d++) {
        if (is_degenerate_dim(param_shape, d))
          axes_.push_back(d);
      }
    } else {
      axes_.resize(ndim_);
      for (int d = 0; d < ndim_; d++)
        axes_[d] = d;
    }
    return Setup(ctx, data_shape, make_span(axes_), scalar_base, scalar_scale, scale_is_stddev);
  }

  KernelRequirements Setup(KernelContext &ctx,
                           const TensorListShape<> &data_shape,
                           span<const int> axes,
                           bool scalar_base,
                           bool scalar_scale,
                           bool scale_is_stddev) {
    ndim_ = data_shape.sample_dim();
    num_samples_ = data_shape.num_samples();

    // this condition is false when the other Setup overload was used
    if (axes_.data() != axes.data())
      axes_ = { axes.begin(), axes.end() };
    axis_mask_ = to_bit_mask<uint64_t>(axes_);
    if (scalar_base && scalar_scale) {
      assert(axis_mask_ == (1_u64 << ndim_) - 1 &&
             "Scalar parameters imply that all axes are reduced.");
    }

    scale_is_stddev_ = scale_is_stddev;
    scalar_base_ = scalar_base;
    scalar_scale_ = scalar_scale;

    KernelRequirements req;
    req.output_shapes = { data_shape };
    return req;
  }

  void Run(KernelContext &ctx,
           const OutListGPU<Out> &out, const InListGPU<In> &in,
           const InListGPU<Base> &base, const InListGPU<Scale> &scale,
           float global_scale = 1, float shift = 0, float epsilon = 0) {
    CheckDataShape(out.shape, in.shape);

    if (scalar_base_ || scalar_scale_)
      throw std::logic_error("Normalize was set up for use with scalar arguments.");

    CheckParamShape(in.shape, base.shape);
    CheckParamShape(in.shape, scale.shape);

    if (scale_is_stddev_)
      RunInvStdDev<Op_InvStdDevNonScalar>(ctx, out, in, base, scale, epsilon, global_scale, shift);
    else
      RunScale<Op_NonScalar>(ctx, out, in, base, scale, global_scale, shift);
  }


  void Run(KernelContext &ctx,
           const OutListGPU<Out> &out, const InListGPU<In> &in,
           const InListGPU<Base> &base, float scale,
           float global_scale = 1, float shift = 0, float epsilon = 0) {
    CheckDataShape(out.shape, in.shape);

    if (scalar_base_ || !scalar_scale_)
      throw std::logic_error("Normalize was not set up for use with scalar scale.");

    CheckParamShape(in.shape, base.shape);

    if (scale_is_stddev_)
      scale = epsilon ? rsqrt(scale * scale + epsilon) : 1 / scale;
    RunScale<Op_ScalarScale>(ctx, out, in, base, scale, global_scale, shift);
  }


  void Run(KernelContext &ctx,
           const OutListGPU<Out> &out, const InListGPU<In> &in,
           float base, const InListGPU<Scale> &scale,
           float global_scale = 1, float shift = 0, float epsilon = 0) {
    CheckDataShape(out.shape, in.shape);

    if (!scalar_base_ || scalar_scale_)
      throw std::logic_error("Normalize was not set up for use with scalar base.");

    CheckParamShape(in.shape, scale.shape);

    if (scale_is_stddev_)
      RunInvStdDev<Op_InvStdDevScalarBase>(ctx, out, in, base, scale, epsilon, global_scale, shift);
    else
      RunScale<Op_ScalarBase>(ctx, out, in, base, scale, global_scale, shift);
  }

  void Run(KernelContext &ctx,
           const OutListGPU<Out> &out, const InListGPU<In> &in,
           float base, float scale,
           float global_scale = 1, float shift = 0, float epsilon = 0) {
    CheckDataShape(out.shape, in.shape);

    if (!scalar_base_ || !scalar_scale_)
      throw std::logic_error("Normalize was not set up for use with scalar arguments.");

    if (scale_is_stddev_)
      scale = epsilon ? rsqrt(scale * scale + epsilon) : 1 / scale;
    RunScale<Op_Scalar>(ctx, out, in, base, scale, global_scale, shift);
  }

 private:
  void CheckParamShape(const TensorListShape<> &data_shape, const TensorListShape<> &param_shape) {
    bool broadcast_param = param_shape.num_samples() == 1;
    int D = data_shape.sample_dim();
    int N = data_shape.num_samples();
    for (int i = 0, p = 0; i < N; i++, p += !broadcast_param) {
      auto dshape = data_shape[i];
      auto pshape = param_shape[p];
      for (int d = 0; d < D; d++) {
        if (axis_mask_ & (1_u64 << d)) {
          if (pshape[d] != 1) {
            throw std::invalid_argument(make_string("Parameter tensor must have extent 1 "
              "in reduced axes. Got:"
              "\nparameter shape: ", pshape,
              "\nreduced axes:    ", axes_str()));
          }
        } else {
          if (pshape[d] != dshape[d]) {
            throw std::invalid_argument(make_string("Parameter tensor must have extent "
              "equal to input shape in non reduced axes. Got:"
              "\nparameter shape: ", pshape,
              "\ndata shape: ", dshape,
              "\nreduced axes:    ", axes_str()));
          }
        }
      }
    }
  }

  void CheckDataShape(const TensorListShape<> &out_shape, const TensorListShape<> &in_shape) {
    if (out_shape != in_shape)
      throw std::invalid_argument("Output and input must have the same shape");
    if (in_shape.sample_dim() != ndim_)
      throw std::invalid_argument("The input tensor list has different dimensionality than the "
        "shape passed to Setup");
    if (in_shape.num_samples() != num_samples_)
      throw std::invalid_argument("The input tensor list has different number of samples than the "
        "shape passed to Setup");
  }

  template <typename Desc>
  std::pair<dim3, dim3> GetLaunchParams(const TensorListShape<> &data_shape, int max_block) const {
    assert(max_block > 0);
    int optimum_block = std::is_same<Desc, Op_Scalar>::value ? 1024 : 256;
    int64_t block = std::min(max_block, optimum_block);
    int64_t max_size = 0;
    for (int i = 0; i < data_shape.num_samples(); i++) {
      int64_t v = volume(data_shape.tensor_shape_span(i));
      if (v > max_size)
        max_size = v;
    }
    if (max_size < block)
      block = max_size;
    int max_blocks_per_sample = max_size == 0 ? 0 : div_ceil(max_size, block);
    dim3 grid(std::min(max_blocks_per_sample, std::max(32, 2048 / num_samples_)), num_samples_);
    return { grid, dim3(block) };
  }

  template <typename Desc, typename BaseParam, typename ScaleParam>
  void RunScale(KernelContext &ctx,
                const OutListGPU<Out> &out, const InListGPU<In> &in,
                const BaseParam &base, const ScaleParam &scale,
                float global_scale, float shift) {
    Desc *cpu_descs = ctx.scratchpad->AllocatePinned<Desc>(num_samples_);
    FillDescs(cpu_descs, out, in, base, scale);
    Desc *gpu_descs = ctx.scratchpad->ToGPU(ctx.gpu.stream, make_span(cpu_descs, num_samples_));
    dim3 grid, block;
    int max_block = MaxThreadsPerBlockStatic(NormalizeKernel<Desc>);
    std::tie(grid, block) = GetLaunchParams<Desc>(in.shape, max_block);
    if (grid.x > 0) {
      NormalizeKernel<<<grid, block, 0, ctx.gpu.stream>>>(gpu_descs, global_scale, shift);
      CUDA_CALL(cudaGetLastError());
    }
  }

  template <typename Desc, typename BaseParam, typename ScaleParam>
  void RunInvStdDev(KernelContext &ctx,
                    const OutListGPU<Out> &out, const InListGPU<In> &in,
                    const BaseParam &base, const ScaleParam &scale,
                    float epsilon, float global_scale, float shift) {
    Desc *cpu_descs = ctx.scratchpad->AllocatePinned<Desc>(num_samples_);
    FillDescs(cpu_descs, out, in, base, scale);
    Desc *gpu_descs = ctx.scratchpad->ToGPU(ctx.gpu.stream, make_span(cpu_descs, num_samples_));
    dim3 grid, block;
    int max_block = MaxThreadsPerBlockStatic(NormalizeInvStdDevKernel<Desc>);
    std::tie(grid, block) = GetLaunchParams<Desc>(in.shape, max_block);
    if (grid.x > 0) {
      NormalizeInvStdDevKernel<<<grid, block, 0, ctx.gpu.stream>>>(gpu_descs, epsilon, global_scale,
                                                                   shift);
      CUDA_CALL(cudaGetLastError());
    }
  }

  std::string axes_str() const {
    std::stringstream ss;
    ss << "{";
    bool first = true;
    for (auto a : axes_) {
      if (first)
        first = false;
      else
        ss << ", ";
      ss << a;
    }
    ss << "}";
    return ss.str();
  }

  template <typename Desc>
  void FillDescs(Desc *descs,
                 const OutListGPU<Out> &out,
                 const InListGPU<In> &in,
                 const InListGPU<Base> &base,
                 const InListGPU<Scale> &scale) {
    int base_idx_delta = base.num_samples() == 1 ? 0 : 1;
    int scale_idx_delta = scale.num_samples() == 1 ? 0 : 1;
    for (int i = 0, b = 0, s = 0; i < num_samples_;
         i++, b += base_idx_delta, s += scale_idx_delta) {
      auto sample_shape = in.shape.tensor_shape_span(i);
      auto &desc = descs[i];
      desc.out = out.data[i];
      desc.in = in.data[i];
      desc.size = volume(sample_shape);
      desc.scale = scale.data[s];
      desc.base = base.data[b];
      desc.dd = DropDims<kMaxDims>(sample_shape, axis_mask_);
    }
  }

  template <typename Desc>
  void FillDescs(Desc *descs,
                 const OutListGPU<Out> &out,
                 const InListGPU<In> &in,
                 float base,
                 const InListGPU<Scale> &scale) {
    int scale_idx_delta = scale.num_samples() == 1 ? 0 : 1;
    for (int i = 0, s = 0; i < num_samples_; i++, s += scale_idx_delta) {
      auto sample_shape = in.shape.tensor_shape_span(i);
      auto &desc = descs[i];
      desc.out = out.data[i];
      desc.in = in.data[i];
      desc.size = volume(sample_shape);
      desc.scale = scale.data[s];
      desc.base = base;
      desc.dd = DropDims<kMaxDims>(sample_shape, axis_mask_);
    }
  }

  template <typename Desc>
  void FillDescs(Desc *descs,
                 const OutListGPU<Out> &out,
                 const InListGPU<In> &in,
                 const InListGPU<Base> &base,
                 float scale) {
    int base_idx_delta = base.num_samples() == 1 ? 0 : 1;
    for (int i = 0, b = 0; i < num_samples_; i++, b += base_idx_delta) {
      auto sample_shape = in.shape.tensor_shape_span(i);
      auto &desc = descs[i];
      desc.out = out.data[i];
      desc.in = in.data[i];
      desc.size = volume(sample_shape);
      desc.scale = scale;
      desc.base = base.data[b];
      desc.dd = DropDims<kMaxDims>(sample_shape, axis_mask_);
    }
  }

  template <typename Desc>
  void FillDescs(Desc *descs,
                 const OutListGPU<Out> &out,
                 const InListGPU<In> &in,
                 float base,
                 float scale) {
    for (int i = 0; i < num_samples_; i++) {
      auto sample_shape = in.shape.tensor_shape_span(i);
      auto &desc = descs[i];
      desc.out = out.data[i];
      desc.in = in.data[i];
      desc.size = volume(sample_shape);
      desc.scale = scale;
      desc.base = base;
    }
  }


  SmallVector<int, DynamicTensorShapeContainer::static_size> axes_;
  uint64_t axis_mask_;
  int ndim_, num_samples_;
  bool scale_is_stddev_;
  bool scalar_base_;
  bool scalar_scale_;
};

}  // namespace normalize_impl
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_NORMALIZE_NORMALIZE_GPU_IMPL_CUH_
