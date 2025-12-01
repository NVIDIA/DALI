// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include <gtest/gtest.h>
#include <random>
#include <utility>
#include "dali/kernels/reduce/mean_stddev_gpu_impl.cuh"
#include "dali/test/test_tensors.h"
#include "dali/test/tensor_test_utils.h"
#include "dali/core/tensor_shape_print.h"
#include "dali/kernels/reduce/reduce_test.h"
#include "dali/kernels/reduce/reduce_gpu_test.h"

namespace dali {
namespace kernels {
namespace reduce_impl {

template <typename Out, typename In, typename Mean>
void CenterAndSquare(const OutTensorCPU<Out> &out,
                     const InTensorCPU<In> &in,
                     const InTensorCPU<Mean> mean,
                     TensorShape<> &in_pos,
                     TensorShape<> &mean_pos,
                     int dim = 0) {
  int extent = in.shape[dim];
  int dj = mean.shape[dim] > 1 ? 1 : 0;
  if (dim == in.dim() - 1) {
    const Mean *mean_ptr = mean(mean_pos);
    const In *in_ptr = in(in_pos);
    Out *out_ptr = out(in_pos);
    for (int i = 0, j = 0; i < extent; i++, j += dj) {
      double d = in_ptr[i] - mean_ptr[j];
      out_ptr[i] = static_cast<Out>(d * d);
    }
  } else {
    for (int i = 0, j = 0; i < extent; i++, j += dj) {
      in_pos[dim] = i;
      mean_pos[dim] = j;
      CenterAndSquare(out, in, mean, in_pos, mean_pos, dim + 1);
    }
  }
}

template <typename Out = float, typename In, typename Mean>
TestTensorList<Out> CenterAndSquare(const InListCPU<In> &in,
                                    const InListCPU<Mean> &mean) {
  TestTensorList<Out> out_tl;
  out_tl.reshape(in.shape);
  auto out = out_tl.cpu();
  int N = in.num_samples();
  for (int i = 0; i < N; i++) {
    auto in_tv = in[i];
    auto out_tv = out[i];
    auto mean_tv = mean.num_samples() > 1 ? mean[i] : mean[0];
    TensorShape<> in_pos, mean_pos;
    in_pos.resize(in_tv.shape.size());
    mean_pos.resize(in_tv.shape.size());
    CenterAndSquare(out_tv, in_tv, mean_tv, in_pos, mean_pos);
  }
  return out_tl;
}

template <typename Out = float, typename In, typename Mean>
TestTensorList<Out> RefStdDev(const TensorListView<StorageCPU, In> &in,
                              const TensorListView<StorageCPU, Mean> &mean,
                              int ddof = 0,
                              double reg = 0, bool inv = false) {
  SmallVector<int, 6> axes;
  for (int d = 0; d < mean.sample_dim(); d++) {
    for (int i = 0; i < mean.num_samples(); i++) {
      if (mean.tensor_shape_span(i)[d] > 1)
        goto non_reduced;
    }
    axes.push_back(d);
  non_reduced:;  // NOLINT
  }

  bool reduce_batch = mean.num_samples() == 1 && in.num_samples() > 1;

  using tmp_t = decltype(In() - Mean());
  auto centered_squared = CenterAndSquare<tmp_t, In, Mean>(in, mean);
  auto centered_squared_cpu = centered_squared.cpu();
  TestTensorList<Out> reduced_samples;

  const auto &out_shape = mean.shape;

  int N = in.num_samples();

  if (reduce_batch) {
    assert(is_uniform(out_shape));
    TensorListShape<> reduced_sample_shapes = uniform_list_shape(in.num_samples(), mean.shape[0]);
    reduced_samples.reshape(reduced_sample_shapes);
    auto reduced_samples_cpu = reduced_samples.cpu();

    for (int i = 0; i < N; i++) {
      RefReduce(reduced_samples_cpu[i], centered_squared_cpu[i],
                make_span(axes), true, reductions::sum());
    }

    TestTensorList<Out> out_tl;
    out_tl.reshape(out_shape);
    auto out = out_tl.cpu();
    int64_t n = out_shape.num_elements();
    double ratio = in.num_elements() / n - ddof;
    for (int j = 0; j < n; j++) {
      double sum = 0;
      for (int i = 0; i < N; i++)
        sum += reduced_samples_cpu.data[i][j];
      out.data[0][j] = inv ? rsqrt(sum / ratio + reg) : std::sqrt(sum / ratio + reg);
    }
    return out_tl;
  } else {
    reduced_samples.reshape(out_shape);
    auto out = reduced_samples.cpu();
    for (int i = 0; i < N; i++) {
      int64_t n = out[i].num_elements();
      int64_t n_in = in[i].num_elements();
      double ratio = n_in / n - ddof;
      RefReduce(out[i], centered_squared_cpu[i], make_span(axes), true, reductions::sum());
      for (int j = 0; j < n; j++) {
        double x = out.data[i][j] / ratio + reg;
        out.data[i][j] = inv ? rsqrt(x) : std::sqrt(x);
      }
    }
    return reduced_samples;
  }
}

template <typename Acc, typename Out, typename In>
void RefMean(const TensorListView<StorageCPU, Out> &out,
             const TensorListView<StorageCPU, In> &in,
             span<const int> axes, bool keep_dims, bool batch) {
  TestTensorList<Acc> sum;
  TensorListShape<> out_shape;
  CalculateReducedShape(out_shape, in.shape, axes, keep_dims, batch);
  sum.reshape(out_shape);
  auto sum_cpu = sum.cpu();
  RefReduce(sum_cpu, in, axes, keep_dims, batch, reductions::sum());
  assert(out.shape == out_shape);
  if (batch) {
    int64_t nin = in.num_elements();
    int64_t nout = out.num_elements();
    double ratio = nin / nout;  // should be an integer, no cast required
    auto *optr = out.data[0];
    auto *sptr = sum_cpu.data[0];
    for (int i = 0; i < nout; i++)
      optr[i] = sptr[i] / ratio;
  } else {
    for (int s = 0; s < in.num_samples(); s++) {
      auto in_tv = in[s];
      auto out_tv = out[s];
      auto sum_tv = sum_cpu[s];

      int64_t nin = in_tv.num_elements();
      int64_t nout = out_tv.num_elements();
      double ratio = nin / nout;  // should be an integer, no cast required
      auto *optr = out_tv.data;
      auto *sptr = sum_tv.data;
      for (int i = 0; i < nout; i++)
        optr[i] = sptr[i] / ratio;
    }
  }
}

TEST(MeanImplGPU, SplitStage) {
  TensorListShape<> in_shape = {{
    { 32, 2, 64000 },
    { 15, 4, 128000 },
    { 72000, 1, 7 }
  }};
  TensorListShape<> ref_out_shape = {{
    { 1, 2, 1 },
    { 1, 4, 1 },
    { 1, 1, 1 }
  }};
  int axes[] = { 0, 2 };

  testing::ReductionKernelTest<MeanImplGPU<float, uint8_t, uint64_t>, float, uint8_t> test;
  for (int iter = 0; iter < 3; iter++) {
    test.Setup(in_shape, ref_out_shape, make_span(axes), true, false);
    EXPECT_GE(test.kernel.GetNumStages(), 4);  // both reduced axes must be split
    test.FillData(0, 255);
    test.Run();
    RefMean<int64_t>(test.ref.cpu(), test.in.cpu(), make_span(axes), true, false);
    test.Check(EqualEpsRel(1e-5, 1e-6));
  }
}



TEST(MeanImplGPU, BatchMean) {
  TensorListShape<> in_shape = {{
    { 32, 3, 64000 },
    { 15, 3, 128000 },
    { 72000, 3, 7 }
  }};
  TensorListShape<> ref_out_shape = {{
    TensorShape<>{3}
  }};
  int axes[] = { 0, 2 };

  testing::ReductionKernelTest<MeanImplGPU<float, uint8_t, uint64_t>, float, uint8_t> test;
  for (int iter = 0; iter < 3; iter++) {
    test.Setup(in_shape, ref_out_shape, make_span(axes), false, true);
    EXPECT_GE(test.kernel.GetNumStages(), 4);  // both reduced axes must be split
    test.FillData(0, 255);
    test.Run();

    RefMean<int64_t>(test.ref.cpu(), test.in.cpu(), make_span(axes), false, true);
    test.Check(EqualEpsRel(1e-5, 1e-6));
  }
}

TEST(StdDevImplGPU, Outer_Inner_SplitStage) {
  TensorListShape<> in_shape = {{
    { 32, 2, 64000 },
    { 15, 4, 128000 },
    { 72000, 1, 7 }
  }};
  TensorListShape<> ref_out_shape = {{
    { 1, 2, 1 },
    { 1, 4, 1 },
    { 1, 1, 1 }
  }};
  int axes[] = { 0, 2 };

  TestTensorList<float> fake_mean;
  fake_mean.reshape(ref_out_shape);
  auto mean_cpu = fake_mean.cpu();
  *mean_cpu[0](0, 0, 0) = 10;
  *mean_cpu[0](0, 1, 0) = 20;
  *mean_cpu[1](0, 0, 0) = 30;
  *mean_cpu[1](0, 1, 0) = 40;
  *mean_cpu[1](0, 2, 0) = 50;
  *mean_cpu[1](0, 3, 0) = 60;
  *mean_cpu[2](0, 0, 0) = 70;

  testing::ReductionKernelTest<StdDevImplGPU<float, int16_t>, float, int16_t> test;
  for (int iter = 0; iter < 3; iter++) {
    test.Setup(in_shape, ref_out_shape, make_span(axes), true, false);
    EXPECT_GE(test.kernel.GetNumStages(), 4);  // both reduced axes must be split
    test.FillData(-100, 100);

    test.Run(fake_mean.gpu());

    test.ref = RefStdDev(test.in.cpu(), mean_cpu);
    test.Check(EqualEpsRel(1e-5, 1e-6));
  }
}


TEST(StdDevImplGPU, Middle_Inner_Sample) {
  TensorListShape<> in_shape = {{
    { 4, 32, 1, 6400 },
    { 3, 15, 2, 12800 },
    { 2, 7200, 3, 7 }
  }};
  TensorListShape<> ref_out_shape = {{
    { 4, 1, 1, 1 },
    { 3, 1, 2, 1 },
    { 2, 1, 3, 1 }
  }};
  int axes[] = { 1, 3 };

  TestTensorList<float> fake_mean;
  fake_mean.reshape(ref_out_shape);
  auto mean_cpu = fake_mean.cpu();
  for (int i = 0, n = mean_cpu.num_elements(); i < n; i++) {
    mean_cpu.data[0][i] = 10 * (i+1);
  }

  testing::ReductionKernelTest<StdDevImplGPU<float, int16_t>, float, int16_t> test;
  for (int iter = 0; iter < 3; iter++) {
    test.Setup(in_shape, ref_out_shape, make_span(axes), true, false);
    EXPECT_GE(test.kernel.GetNumStages(), 2);  // both reduced axes must be split
    test.FillData(-100, 100);

    test.Run(fake_mean.gpu());

    test.ref = RefStdDev(test.in.cpu(), mean_cpu);
    test.Check(EqualEpsRel(1e-5, 1e-6));
  }
}

TEST(StdDevImplGPU, Middle_Inner_Batch) {
  TensorListShape<> in_shape = {{
    { 2, 32, 3, 6400 },
    { 2, 15, 3, 12800 },
    { 2, 7200, 3, 7 }
  }};
  TensorListShape<> ref_out_shape = {{
    { 2, 1, 3, 1 }
  }};
  int axes[] = { 1, 3 };

  TestTensorList<float> fake_mean;
  fake_mean.reshape(ref_out_shape);
  auto mean_cpu = fake_mean.cpu();
  *mean_cpu[0](0, 0, 0, 0) = 10;
  *mean_cpu[0](0, 0, 1, 0) = 20;
  *mean_cpu[0](0, 0, 2, 0) = 30;
  *mean_cpu[0](1, 0, 0, 0) = 40;
  *mean_cpu[0](1, 0, 1, 0) = 50;
  *mean_cpu[0](1, 0, 2, 0) = 60;

  testing::ReductionKernelTest<StdDevImplGPU<float, int16_t>, float, int16_t> test;
  for (int iter = 0; iter < 3; iter++) {
    test.Setup(in_shape, ref_out_shape, make_span(axes), true, true);
    EXPECT_GE(test.kernel.GetNumStages(), 2);  // both reduced axes must be split
    test.FillData(-100, 100);

    test.Run(fake_mean.gpu());

    test.ref = RefStdDev(test.in.cpu(), mean_cpu);
    test.Check(EqualEpsRel(1e-5, 1e-6));
  }
}


TEST(InvStdDevImplGPU, Outer_Batch_Regularized) {
  TensorListShape<> in_shape = {{
    { 480, 640, 3 },
    { 720, 1280, 3 },
    { 1080, 1920, 3 }
  }};
  TensorListShape<> ref_out_shape = {{
    { 1, 1, 3 }
  }};
  int axes[] = { 0, 1 };

  TestTensorList<float> fake_mean;
  fake_mean.reshape(ref_out_shape);
  auto mean_cpu = fake_mean.cpu();
  for (int i = 0, n = mean_cpu.num_elements(); i < n; i++) {
    mean_cpu.data[0][i] = 10 * (i+1);
  }

  testing::ReductionKernelTest<InvStdDevImplGPU<float, int16_t>, float, int16_t> test;
  for (int iter = 0; iter < 3; iter++) {
    test.Setup(in_shape, ref_out_shape, make_span(axes), true, true);
    EXPECT_GE(test.kernel.GetNumStages(), 2);  // both reduced axes must be split
    test.FillData(-100, 100);

    test.Run(fake_mean.gpu(), 1, 12000);

    test.ref = RefStdDev(test.in.cpu(), mean_cpu, 1, 12000, true);
    test.Check(EqualEpsRel(1e-5, 1e-6));
  }
}


}  // namespace reduce_impl
}  // namespace kernels
}  // namespace dali
