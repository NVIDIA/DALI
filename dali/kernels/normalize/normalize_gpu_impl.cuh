// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include "dali/core/convert.h"
#include "dali/core/small_vector.h"
#include "dali/kernels/reduce/reduce_drop_dims.h"
#include "dali/kernels/kernel.h"

namespace dali {
namespace kernels {

namespace normalize_impl {

template <int max_dims, typename Out, typename In>
struct NormalizeNonScalar {
  Out *__restrict__ out;
  const In *__restrict__ in;
  int64_t size;
  const float *__restrict__ base;
  const float *__restrict__ scale;
  reduce_impl::DropDims<max_dims> dd;

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
    float v = ConvertSat<Out>((in[offset] - sub) * mul + global_shift);
  }
};

template <int max_dims, typename Out, typename In>
struct NormalizeScalarScale {
  Out *__restrict__ out;
  const In *__restrict__ in;
  int64_t size;
  const float *__restrict__ base;
  reduce_impl::DropDims<max_dims> dd;
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
    float v = ConvertSat<Out>((in[offset] - sub) * mul + global_shift);
  }
};

template <int max_dims, typename Out, typename In>
struct NormalizeScalarBase {
  Out *__restrict__ out;
  const In *__restrict__ in;
  int64_t size;
  const float *__restrict__ scale;
  reduce_impl::DropDims<max_dims> dd;
  float base;

  DALI_HOST_DEV DALI_FORCEINLINE
  void apply(int64_t offset, float global_scale, float global_shift) {
    int64_t param_offset = dd.reindex(offset);
  #ifdef __CUDA_ARCH__
    float mul = __ldg(scale + param_offset) * global_scale;
  #else
    float mul = scale[param_offset] * global_scale;
  #endif
    float v = ConvertSat<Out>((in[offset] - base) * mul + global_shift);
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
    float v = ConvertSat<Out>((in[offset] - base) * scale * global_scale + global_shift);
  }
};

/**
 * @brief This variant is used when standard deviation is externally provided and needs to
 *        be regularized and inversed.
 *
 * The output elements are calculated as:
 * mul = 1 / sqrt(sqr(stddev[param_offset]) + epsilon)
 * (in[offset] - mean[param_offset]) * mul * scale + shift
 */
template <int max_dims, typename Out, typename In>
struct NormalizeInvRegNonScalar {
  Out *__restrict__ out;
  const In *__restrict__ in;
  int64_t size;
  const float *__restrict__ base;
  const float *__restrict__ scale;
  reduce_impl::DropDims<max_dims> dd;

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
    float mul = rsqrt(stddev * stddev + epsilon);
    float v = ConvertSat<Out>((in[offset] - mean) * mul + global_shift);
  }
};

/**
 * @brief This variant is used when standard deviation is externally provided and needs to
 *        be regularized and inversed.
 *
 * The output elements are calculated as:
 * mul = 1 / sqrt(sqr(stddev[param_offset]) + epsilon)
 * (in[offset] - mean[param_offset]) * mul * scale + shift
 */
template <int max_dims, typename Out, typename In>
struct NormalizeInvRegScalarBase {
  Out *__restrict__ out;
  const In *__restrict__ in;
  int64_t size;
  const float *__restrict__ scale;
  reduce_impl::DropDims<max_dims> dd;
  float base;

  DALI_HOST_DEV DALI_FORCEINLINE
  void apply(int64_t offset, float epsilon, float global_scale, float global_shift) {
    int64_t param_offset = dd.reindex(offset);
  #ifdef __CUDA_ARCH__
    float stddev = __ldg(scale + param_offset);
  #else
    float stddev = scale[param_offset];
  #endif
    float mul = rsqrt(stddev * stddev + epsilon);
    float v = ConvertSat<Out>((in[offset] - mul) * mul + global_shift);
  }
};

template <typename NormalizeParams>
__global__ void Normalize(const NormalizeParams *sample_params,
                          float scale, float shift) {
  auto params = sample_params[blockIdx.y];
  int64_t start_ofs = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t grid_stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t ofs = start_ofs; ofs < params.size; ofs += grid_stride) {
    params.apply(ofs, scale, shift);
  }
}

template <typename NormalizeParams>
__global__ void NormalizeEps(const NormalizeParams *sample_params,
                             float epsilon, float scale, float shift) {
  auto params = sample_params[blockIdx.y];
  int64_t start_ofs = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t grid_stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t ofs = start_ofs; ofs < params.size; ofs += grid_stride) {
    params.apply(ofs, epsilon, scale, shift);
  }
}

template <typename Out, typename In, typename BaseParam, typename ScaleParam>
class NormalizeImplGPU {
  static constexpr const int kMaxDims = 4;

  using Op_NonScalar = NormalizeNonScalar<kMaxDims, Out, In>;
  using Op_ScalarBase = NormalizeScalarBase<kMaxDims, Out, In>;
  using Op_ScalarScale = NormalizeScalarScale<kMaxDims, Out, In>;
  using Op_Scalar = NormalizeScalar<Out, In>;
  using Op_InvStdDevNonScalar = NormalizeInvRegNonScalar<kMaxDims, Out, In>;
  using Op_InvStdDevScalarBase = NormalizeInvRegScalarBase<kMaxDims, Out, In>;

 public:
  KernelRequirements Setup(KernelContext &ctx,
                           const TensorListShape<> &data_shape,
                           const TensorListShape<> &param_shape,
                           bool scalar_base,
                           bool scalar_scale,
                           bool scale_is_stddev) {
    int ndim = data_shape.sample_dim();
    if (!scalar_base || !scalar_scale) {
      if (param_shape.sample_dim() != ndim) {
        throw std::invalid_argument("Normalization parameters must have the same "
          "dimensionality as the data");
      }
      if (param_shape.num_samples() != data_shape.num_samples() && param_shape.num_samples() != 1) {
        throw std::invalid_argument("Normalization parameters must have either the same number "
          "of samples as the input or just one sample.");
      }

      axes_.clear();
      for (int d = 0; d < ndim; d++) {
        if (is_degenerate_dim(param_shape_, d))
          axes_.push_back(d);
      }
    } else {
      if (param_shape.num_samples() > 0)
        throw std::invalid_argument("`param_shape` must be empty when using scalar parameters");

      axes_.resize(ndim);
      for (int d = 0; d < ndim; d++)
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
    KernelRequirements req;
    req.output_shapes = { data_shape };
    ScratchpadEstimator se;

    // this condition is false when the other Setup overload was used
    if (axes_.data() != axes.data())
      this->axes_ = { axes_.begin(), axes_.end() };

    this->scale_is_stddev_ = scale_is_stddev_;
    this->scalar_base_ = scalar_base_;
    this->scalar_scale_ = scalar_scale_;
    if (scalar_scale_) {
      // scale is a scalar - it will be corrected before launch, in host code
      if (scalar_base_) {
        SetupScalar(se);
      } else {
        SetupScalarScale(se);
      }
    } else {
      if (scale_is_stddev_) {
        if (scalar_base_) {
          SetupInvStddDvScalarBase(se);
        } else {
          SetupInvStdDevNonScalar(se);
        }
      } else {
        if (scalar_base_) {
          SetupScalarBase(se);
        } else {
          SetupNonScalar(se);
        }
      }
    }

    req.scratch_sizes = se.sizes;
    return req;
  }

  void Run(KernelContext &ctx,
           const OutListGPU<Out> &out, const InListGPU<In> &in,
           const InListGPU<BaseParam> &base, const InListGPU<ScaleParam> &scale,
           float epsilon = 0) {
    assert(!scalar_base_ && !scalar_scale_);
    if (!scalar_base_)
      CheckParamShape(in.shape, base.shape);
    if (!scalar_scale_)
      CheckParamShape(in.shape, scale.shape);

    if (scale_is_stddev_) {

    }
  }

 private:
  void CheckParamShape(const TensorListShape<> &data_shape, const TensorListShape<> &param_shape) {
    bool broadcast_param = param_shape.num_samples() == 1;
    int D = data_shape.sample_dim();
    int N = data_shape.num_samples();
    for (int i = 0; i < N; i++, o += !broadcast_param) {
      auto dshape = data_shape[i];
      auto pshape = param_shape[o];
      for (int d = 0; d < D; d++) {
        if (axis_mask_ & (1<<d)) {
          if (pshape[d] != 1) {
            throw std::invalid_argument(make_str("Parameter tensor must have extent 1 "
              "in reduced axes. Got:"
              "\nparameter shape: ", pshape,
              "\nreduced axes:    ", axes_str()));
          }
        } else {
          if (pshape[d] != dshape[d]) {
            throw std::invalid_argument(make_str("Parameter tensor must have extent equal to input "
              "shape in non reduced axes. Got:"
              "\nparameter shape: ", pshape,
              "\ndata shape: ", dshape,
              "\nreduced axes:    ", axes_str()));
          }
        }
      }
    }
  }

  std::string axes_str() const {
    std::stringstream ss;
    ss << "{";
    bool first = true;
    for (auto a : axes_) {
      if (first) first = false;
      else ss << ", ";
      ss << a;
    }
    ss << "}";
    return ss.str();
  }

  void SetupScalar(ScratchpadEstimator &se) {
    SetupImpl<Op_Scalar>(se);
  }
  void SetupScalarScale(ScratchpadEstimator &se) {
    SetupImpl<Op_ScalarScale>(se);
  }
  void SetupInvStddDvScalarBase(ScratchpadEstimator &se) {
    SetupImpl<Op_InvStdDevScalarBase>(se);
  }
  void SetupInvStdDevNonScalar(ScratchpadEstimator &se) {
    SetupImpl<Op_InvStdDevNonScalar>(se);
  }
  void SetupScalarBase(ScratchpadEstimator &se) {
    SetupImpl<Op_ScalarBase>(se);
  }
  void SetupNonScalar(ScratchpadEstimator &se) {
    SetupImpl<Op_NonScalar>(se);
  }

  template <typename Desc>
  void SetupImpl(ScratchpadEstimator &se) {

  }

  SmallVector<int, DynamicTensorShapeContainer::static_size> axes_;
  uint64_t axis_mask_;
  bool scale_is_stddev_;
  bool scalar_base_;
  bool scalar_scale_;
};

}  // namespace normalize_impl
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_NORMALIZE_NORMALIZE_GPU_IMPL_CUH_
