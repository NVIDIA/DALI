// Copyright (c) 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/kernels/normalize/normalize_gpu.h"  // NOLINT
#include "dali/kernels/normalize/normalize_gpu_impl.cuh"  // NOLINT
#include <gtest/gtest.h>
#include <cmath>
#include <initializer_list>
#include <iostream>
#include <random>
#include <utility>
#include "dali/core/cuda_event.h"
#include "dali/kernels/kernel_manager.h"
#include "dali/test/device_test.h"
#include "dali/test/test_tensors.h"
#include "dali/test/tensor_test_utils.h"

namespace dali {
namespace kernels {

template <bool calc_inv_stddev, typename Out, typename In>
void RefNormalize(
    const OutTensorCPU<Out> &out,
    const InTensorCPU<In> &in, const InTensorCPU<float> &base,
    const InTensorCPU<float> &scale,
    float epsilon, float global_scale, float shift,
    TensorShape<> &data_pos, TensorShape<> &base_pos, TensorShape<> &scale_pos, int dim) {

  int db = 0, ds = 0;
  int64_t extent = 0;
  if (dim < in.dim()) {
    db = base.shape[dim] > 1 ? 1 : 0;
    ds = scale.shape[dim] > 1 ? 1 : 0;
    extent = in.shape[dim];
  }
  if (dim >= in.dim() - 1) {  // handles both last dimension and degenerate case
    Out *optr = out(data_pos);
    const In *iptr = in(data_pos);
    const float *sptr = scale(scale_pos);
    const float *bptr = base(base_pos);
    for (int64_t i = 0, b = 0, s = 0; i < extent; i++, b += db, s += ds) {
      float mul;
      if (calc_inv_stddev) {
        float x = sptr[s] * sptr[s] + epsilon;
        mul = x ? rsqrt(x) * global_scale : 0;
      } else {
        mul = sptr[s] * global_scale;
      }
      optr[i] = ConvertSat<Out>(std::fma(iptr[i] - bptr[b], mul, shift));
    }
  } else {
    for (int64_t i = 0, b = 0, s = 0; i < extent; i++, b += db, s += ds) {
      data_pos[dim] = i;
      base_pos[dim] = b;
      scale_pos[dim] = s;
      RefNormalize<calc_inv_stddev>(out, in, base, scale, epsilon, global_scale, shift,
                                    data_pos, base_pos, scale_pos, dim + 1);
    }
  }
}


/**
 * @brief Reference normalization of a single tensor
 *
 * If base/scale has an extent of 1 in any given dimension, it's broadcast along this axis.
 *
 * @param calc_inv_stddev if true, `scale` is assumed to contain standard deviation, which
 *                        is subsequently regularized using given epsilon value
 */
template <typename Out, typename In>
void RefNormalize(
    const OutTensorCPU<Out> &out,
    const InTensorCPU<In> &in, const InTensorCPU<float> &base,
    const InTensorCPU<float> &scale,
    float global_scale, float shift,
    bool calc_inv_stddev, float epsilon) {
  TensorShape<> data_pos, base_pos, scale_pos;
  int D = in.dim();
  data_pos.resize(D);
  base_pos.resize(D);
  scale_pos.resize(D);
  if (calc_inv_stddev) {
    RefNormalize<true>(out, in, base, scale, epsilon, global_scale, shift,
                       data_pos, base_pos, scale_pos, 0);
  } else {
    RefNormalize<false>(out, in, base, scale, epsilon, global_scale, shift,
                        data_pos, base_pos, scale_pos, 0);
  }
}

/**
 * @brief Reference implementation of normalization
 *
 * Goes over all input samples and normalizes them using given base and scale tensor lists.
 * If base/scale TL has 1 element, it is reused for normalization of all samples.
 * If base/scale has an extent of 1 in any given dimension, it's broadcast along this axis.
 *
 * @param calc_inv_stddev if true, `scale` is assumed to contain standard deviation, which
 *                        is subsequently regularized using given epsilon value
 */
template <typename Out, typename In>
void RefNormalize(
    const OutListCPU<Out> &out, const TensorListView<StorageCPU, In> &in,
    const InListCPU<float> &base, const InListCPU<float> &scale,
    float global_scale, float shift,
    bool calc_inv_stddev = false, float epsilon = 0) {
  assert(out.shape == in.shape);
  int N = in.num_samples();
  int db = base.num_samples() > 1;
  int ds = scale.num_samples() > 1;
  for (int i = 0, b = 0, s = 0; i < N; i++, b += db, s += ds) {
    RefNormalize<Out, In>(out[i], in[i], base[b], scale[s],
                          global_scale, shift, calc_inv_stddev, epsilon);
  }
}

template <typename RNG>
TensorListShape<>
RandomDataShape(int num_samples, int ndim, int64_t max_volume,
                uint64_t reduced_axes, bool reduce_batch, RNG &rng) {
  assert(max_volume >= 1);
  TensorListShape<> sh;
  sh.resize(num_samples, ndim);

  int64_t extent_range = std::ceil(pow(max_volume, 1.0 / ndim));
  std::uniform_int_distribution<int64_t> shape_dist(1, extent_range);

  for (int i = 0; i < num_samples; i++) {
    auto sample_shape = sh.tensor_shape_span(i);
    do {
      for (int d = 0; d < ndim; d++) {
        // when reducing samples in the batch, the non-reduced extents must be uniform
        // across all samples
        sample_shape[d] = reduced_axes & (1_u64 << d) || !reduce_batch || i == 0
            ? shape_dist(rng)
            : sh.tensor_shape_span(0)[d];
      }
    } while (volume(sample_shape) > max_volume);
  }
  return sh;
}

/**
 * @brief Creates a tensor list which contains a repeated scalar
 *
 * If ndim > 0, then the tensor list will contain 1x1x...x1 tensors with given dimensionality
 */
template <typename T>
TensorListView<StorageCPU, T> ScalarTLV(T &scalar, int num_samples, int ndim = 0) {
  TensorListView<StorageCPU, T> tlv;
  TensorShape<> ts;
  ts.resize(ndim);
  for (int d = 0; d < ndim; d++)
    ts[d] = 1;

  tlv.shape = uniform_list_shape(num_samples, ts);
  tlv.data.resize(num_samples);
  for (int i = 0 ; i < num_samples; i++)
    tlv.data[i] = &scalar;
  return tlv;
}

template <typename Params>
class NormalizeImplGPUTest;

template <typename Out, typename In>
class NormalizeImplGPUTest<std::pair<Out, In>> : public ::testing::Test {
 public:
  // this will test both the top-level pImpl class and the internal implementation class
  using Kernel = std::conditional_t<std::is_same<Out, In>::value,
    NormalizeGPU<Out, In>,
    normalize_impl::NormalizeImplGPU<Out, In, float, float>
  >;

  void Init(int num_samples, int ndim, int64_t max_sample_volume,
            std::initializer_list<int> reduced_axes, bool reduce_batch,
            bool scalar_base, bool scalar_scale, bool scale_is_stddev) {
    Init(num_samples, ndim, max_sample_volume,
         { reduced_axes.begin(), reduced_axes.end() }, reduce_batch,
         scalar_base, scalar_scale, scale_is_stddev);
  }

  void Init(int num_samples, int ndim, int64_t max_sample_volume,
            span<const int> reduced_axes, bool reduce_batch,
            bool scalar_base, bool scalar_scale, bool scale_is_stddev) {
    In lo = 0, hi = 100;
    use_scalar_base_ = scalar_base;
    use_scalar_scale_ = scalar_scale;
    axis_mask_ = to_bit_mask(reduced_axes);
    reduced_axes_ = { begin(reduced_axes), end(reduced_axes) };
    reduce_batch_ = reduce_batch;
    scale_is_stddev_ = scale_is_stddev;

    data_shape_ = RandomDataShape(num_samples, ndim, max_sample_volume,
                                  axis_mask_, reduce_batch_, rng_);
    in_.reshape(data_shape_);
    UniformRandomFill(in_.cpu(), rng_, lo, hi);

    if (!scalar_base || !scalar_scale) {
      int param_samples = reduce_batch ? 1 : num_samples;
      param_shape_.resize(param_samples, ndim);
      for (int i = 0; i < param_samples; i++) {
        for (int d = 0; d < ndim; d++) {
          bool reduced = axis_mask_ & (1_u64 << d);
          param_shape_.tensor_shape_span(i)[d] = reduced ? 1 : data_shape_.tensor_shape_span(i)[d];
        }
      }
    } else {
      param_shape_.resize(1, 0);
    }

    auto scale_dist = uniform_distribution(0.1f, 10.0f);
    if (scalar_scale) {
      scalar_scale_ = scale_dist(rng_);
    } else {
      scale_.reshape(param_shape_);
      UniformRandomFill(scale_.cpu(), rng_, scale_dist.a(), scale_dist.b());
    }

    if (scalar_base) {
      scalar_base_ = uniform_distribution(lo, hi)(rng_);
    } else {
      base_.reshape(param_shape_);
      UniformRandomFill(base_.cpu(), rng_, lo, hi);
    }

    if (std::is_integral<Out>::value) {
      global_scale_ = std::exp2f(7 * sizeof(Out)) / hi;  // scale to half range
      if (std::is_unsigned<Out>::value)
        shift_ = global_scale_;  // shift half range up
    }
  }

  void RunTest() {
    kmgr_.Resize<Kernel>(1);
    KernelContext ctx;
    ctx.gpu.stream = 0;
    for (int iter = 0; iter < 3; iter++) {
      auto req = kmgr_.Setup<Kernel>(0, ctx, data_shape_, param_shape_,
                                     use_scalar_base_, use_scalar_scale_, scale_is_stddev_);
      ASSERT_EQ(req.output_shapes.size(), 1u);
      ASSERT_EQ(req.output_shapes[0], data_shape_);
      out_.reshape(data_shape_);
      ref_.reshape(data_shape_);

      Launch(ctx);

      int param_samples = param_shape_.num_samples();
      auto ref_base  = use_scalar_base_
                      ? ScalarTLV(scalar_base_,  param_samples, data_shape_.sample_dim())
                      : base_.cpu();
      auto ref_scale = use_scalar_scale_
                      ? ScalarTLV(scalar_scale_, param_samples, data_shape_.sample_dim())
                      : scale_.cpu();
      RefNormalize(ref_.cpu(), in_.cpu(), ref_base, ref_scale,
                  global_scale_, shift_, scale_is_stddev_, epsilon_);

      if (scale_is_stddev_ && !std::is_integral<Out>::value)
        Check(out_.cpu(), ref_.cpu(), EqualEpsRel(1e-6, 1e-6));
      else
        Check(out_.cpu(), ref_.cpu(), EqualUlp(4));
    }
  }

  void RunPerf() {
    kmgr_.Resize<Kernel>(1);
    KernelContext ctx;
    ctx.gpu.stream = 0;
    auto req = kmgr_.Setup<Kernel>(0, ctx, data_shape_, param_shape_,
                                   use_scalar_base_, use_scalar_scale_, scale_is_stddev_);
    ASSERT_EQ(req.output_shapes.size(), 1u);
    ASSERT_EQ(req.output_shapes[0], data_shape_);
    out_.reshape(data_shape_);

    CUDAEvent start = CUDAEvent::CreateWithFlags(0);
    CUDAEvent end = CUDAEvent::CreateWithFlags(0);


    auto out_gpu = out_.gpu();
    CUDA_CALL(
      cudaMemsetAsync(out_gpu.data[0], 0, sizeof(Out) * out_gpu.num_elements(), ctx.gpu.stream));
    Launch(ctx);
    CUDA_CALL(cudaEventRecord(start, ctx.gpu.stream));
    Launch(ctx);
    CUDA_CALL(cudaEventRecord(end, ctx.gpu.stream));
    float time;
    CUDA_CALL(cudaDeviceSynchronize());
    CUDA_CALL(cudaEventElapsedTime(&time, start, end));
    time *= 1e+6f;  // convert to nanoseconds
    int64_t out_size = data_shape_.num_elements() * sizeof(Out);
    int64_t in_size  = data_shape_.num_elements() * sizeof(In);
    int64_t base_size  = scalar_base_  ? 0 : param_shape_.num_elements() * sizeof(float);
    int64_t scale_size = scalar_scale_ ? 0 : param_shape_.num_elements() * sizeof(float);
    int64_t data_size = out_size + in_size + base_size + scale_size;
    std::cerr << "Throughput: " << data_size / time << " GB/s\n";
  }

  void Launch(KernelContext &ctx) {
    if (use_scalar_base_) {
      if (use_scalar_scale_) {
        kmgr_.Run<Kernel>(0, ctx, out_.gpu(), in_.gpu(), scalar_base_, scalar_scale_,
                          global_scale_, shift_, epsilon_);
      } else {
        kmgr_.Run<Kernel>(0, ctx, out_.gpu(), in_.gpu(), scalar_base_, scale_.gpu(),
                          global_scale_, shift_, epsilon_);
      }
    } else {
      if (use_scalar_scale_) {
        kmgr_.Run<Kernel>(0, ctx, out_.gpu(), in_.gpu(), base_.gpu(), scalar_scale_,
                          global_scale_, shift_, epsilon_);
      } else {
        kmgr_.Run<Kernel>(0, ctx, out_.gpu(), in_.gpu(), base_.gpu(), scale_.gpu(),
                          global_scale_, shift_, epsilon_);
      }
    }
  }

 protected:
  KernelManager kmgr_;
  TestTensorList<In> in_;
  TestTensorList<Out> out_;
  TestTensorList<float> ref_;
  TestTensorList<float> base_, scale_;
  TensorListShape<> data_shape_, param_shape_;
  SmallVector<int, 6> reduced_axes_;
  uint64_t axis_mask_;
  bool reduce_batch_ = false;
  bool use_scalar_base_ = false;
  bool use_scalar_scale_ = false;
  bool scale_is_stddev_ = false;

  float scalar_base_ = 0, scalar_scale_ = 1;
  float global_scale_ = 1.25f, shift_ = 0.1f, epsilon_ = 0.2f;

  std::mt19937_64 rng_;
};

using NormalizeTestTypes = ::testing::Types<
  std::pair<int16_t, uint8_t>,
  std::pair<float, uint16_t>,
  std::pair<float, float>>;

TYPED_TEST_SUITE(NormalizeImplGPUTest, NormalizeTestTypes);

TYPED_TEST(NormalizeImplGPUTest, NonScalar) {
  this->Init(10, 4, 10000, { 1, 3 }, false, false, false, false);
  this->RunTest();
  this->Init(10, 3, 10000, { 0, 2 }, true, false, false, false);
  this->RunTest();
}

TYPED_TEST(NormalizeImplGPUTest, ScalarBase) {
  this->Init(10, 4, 10000, { 1, 3 }, false, true, false, false);
  this->RunTest();
  this->Init(10, 3, 10000, { 0, 2 }, true, true, false, false);
  this->RunTest();
}

TYPED_TEST(NormalizeImplGPUTest, ScalarScale) {
  this->Init(10, 4, 10000, { 1, 3 }, false, false, true, false);
  this->RunTest();
  this->Init(10, 3, 10000, { 0, 2 }, true, false, true, false);
  this->RunTest();
}

TYPED_TEST(NormalizeImplGPUTest, ScalarParams) {
  this->Init(10, 4, 10000, {}, false, true, true, false);
  this->RunTest();
  this->Init(10, 3, 10000, {}, true, true, true, false);
  this->RunTest();
}

TYPED_TEST(NormalizeImplGPUTest, NonScalar_InvStdDev) {
  this->Init(10, 4, 10000, { 1, 3 }, false, false, false, true);
  this->RunTest();
  this->Init(10, 3, 10000, { 0, 2 }, true, false, false, true);
  this->RunTest();
}

TYPED_TEST(NormalizeImplGPUTest, ScalarBase_InvStdDev) {
  this->Init(10, 4, 10000, { 1, 3 }, false, true, false, false);
  this->RunTest();
  this->Init(10, 3, 10000, { 0, 2 }, true, true, false, false);
  this->RunTest();
}

TYPED_TEST(NormalizeImplGPUTest, ScalarScale_InvStdDev) {
  this->Init(10, 4, 10000, { 1, 3 }, false, false, true, true);
  this->RunTest();
  this->Init(10, 3, 10000, { 0, 2 }, true, false, true, true);
  this->RunTest();
}

TYPED_TEST(NormalizeImplGPUTest, ScalarParams_InvStdDev) {
  this->Init(10, 4, 10000, {}, false, true, true, true);
  this->RunTest();
  this->Init(10, 3, 10000, {}, true, true, true, true);
  this->RunTest();
}

TYPED_TEST(NormalizeImplGPUTest, Perf_NonScalar5D) {
  this->Init(64, 5, 1<<20, { 1, 3 }, false, false, false, false);
  this->RunPerf();
}

TYPED_TEST(NormalizeImplGPUTest, Perf_NonScalar3D_Reduce01) {
  this->Init(64, 3, 1<<20, { 0, 1 }, false, false, false, false);
  this->RunPerf();
}

TYPED_TEST(NormalizeImplGPUTest, Perf_NonScalar3D_Reduce12) {
  this->Init(64, 3, 1<<20, { 1, 2 }, false, false, false, false);
  this->RunPerf();
}

TYPED_TEST(NormalizeImplGPUTest, Perf_ScalarParams) {
  this->Init(64, 3, 1<<20, {}, false, true, true, false);
  this->RunPerf();
}

TYPED_TEST(NormalizeImplGPUTest, Perf_NonScalar5D_InvStdDev) {
  this->Init(64, 5, 1<<20, { 1, 3 }, false, false, false, true);
  this->RunPerf();
}

TYPED_TEST(NormalizeImplGPUTest, Perf_NonScalar3D_Reduce01_InvStdDev) {
  this->Init(64, 3, 1<<20, { 0, 1 }, false, false, false, true);
  this->RunPerf();
}

TYPED_TEST(NormalizeImplGPUTest, Perf_NonScalar3D_Reduce12_InvStdDev) {
  this->Init(64, 3, 1<<20, { 1, 2 }, false, false, false, true);
  this->RunPerf();
}

TYPED_TEST(NormalizeImplGPUTest, Perf_ScalarParams_InvStdDev) {
  this->Init(64, 3, 1<<20, {}, false, true, true, true);
  this->RunPerf();
}

}  // namespace kernels
}  // namespace dali
