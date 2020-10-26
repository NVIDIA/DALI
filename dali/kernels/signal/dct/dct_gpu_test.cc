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

#include <tuple>
#include "dali/kernels/signal/dct/dct_gpu.h"
#include "dali/kernels/kernel_manager.h"
#include "dali/kernels/common/utils.h"
#include "dali/test/tensor_test_utils.h"
#include "dali/test/test_tensors.h"
#include "dali/kernels/signal/dct/dct_test.h"

namespace dali {
namespace kernels {
namespace signal {
namespace dct {
namespace test {

constexpr int Dims = 4;

class Dct1DGpuTest : public ::testing::TestWithParam<std::tuple<int, std::vector<int>>> {
 public:
  Dct1DGpuTest()
      : batch_size_(std::get<0>(GetParam()))
      , axes_(std::get<1>(GetParam()))
      , in_shape_(batch_size_) {}

  ~Dct1DGpuTest() override = default;

 protected:
  void PrepareInput() {
    args_.clear();
    std::mt19937_64 rng{1223};
    std::uniform_int_distribution<> dim_dist(1, 3);
    auto rand_shape = [&]() {
      TensorShape<Dims> shape;
      for (int i = 0; i < Dims; ++i)
        shape[i] = dim_dist(rng) * 10;
      return shape;
    };
    for (int s = 0; s < batch_size_; ++s) {
      auto i = arg_idx_;
      in_shape_.set_tensor_shape(s, rand_shape());
      args_.emplace_back(get_args());
    }
    ttl_in_.reshape(in_shape_);
    UniformRandomFill(ttl_in_.cpu(), rng, 0., 1.);
  }

  TensorListShape<Dims> OutputShape(int axis) {
    TensorListShape<Dims> out_shape(batch_size_);
    for (int s = 0; s < batch_size_; ++s) {
      auto in_shape = in_shape_[s];
      int ndct = args_[s].ndct;
      if (ndct > 0)
        in_shape[axis] = ndct;
      out_shape.set_tensor_shape(s, in_shape);
    }
    return out_shape;
  }

  DctArgs get_args() {
    auto i = arg_idx_;
    if (normalize[(i / 3) % 2] && dct_type[(i / 6) % 4] == 1) {
      i += 3;
      arg_idx_ += 4;
    } else {
      arg_idx_ += 1;
    }
    return DctArgs{dct_type[(i / 6) % 4], normalize[(i / 3) % 2], ndct[i % 3]};
  }

  int batch_size_;
  std::vector<int> axes_;
  TensorListShape<Dims> in_shape_;
  TestTensorList<float, Dims> ttl_in_;
  TestTensorList<float, Dims> ttl_out_;
  std::vector<DctArgs> args_;
  int arg_idx_ = 0;
  const std::array<int, 4> dct_type = {{1, 2, 3, 4}};
  const std::array<bool, 2> normalize = {{false, true}};
  const std::array<int, 3> ndct = {{-1, 10, 20}};
};


TEST_P(Dct1DGpuTest, DctTest) {
  using Kernel = Dct1DGpu<float, float, 4>;
  KernelContext ctx;
  ctx.gpu.stream = 0;
  KernelManager kmgr;
  kmgr.Initialize<Kernel>();
  kmgr.Resize<Kernel>(1, 1);

  for (auto axis : axes_) {
    PrepareInput();
    auto out_shape = OutputShape(axis);
    auto in_view = ttl_in_.gpu();
    auto req = kmgr.Setup<Kernel>(0, ctx, in_view, make_cspan(args_), axis);
    ASSERT_EQ(out_shape, req.output_shapes[0]);
    ttl_out_.reshape(out_shape);
    auto out_view = ttl_out_.gpu();
    cudaMemsetAsync(out_view.data[0], 0, batch_size_*volume(out_view.shape[0])*sizeof(float), 0);
    kmgr.Run<Kernel>(0, 0, ctx, out_view, in_view, make_cspan(args_), axis);
    cudaStreamSynchronize(ctx.gpu.stream);
    auto cpu_in_view = ttl_in_.cpu();
    auto cpu_out_view = ttl_out_.cpu();
    for (int s = 0; s < batch_size_; ++s) {
      auto in = cpu_in_view.tensor_data(s);
      auto out = cpu_out_view.tensor_data(s);
      auto in_sample_shape = in_shape_[s];
      auto out_sample_shape = out_shape[s];
      int64_t n_outer = volume(&in_sample_shape[0], &in_sample_shape[axis]);
      int64_t n_inner = volume(&in_sample_shape[axis + 1], &in_sample_shape[Dims]);
      int64_t n_frames = n_outer * n_inner;
      int64_t inner_stride = (axis < Dims - 1)
                              ? volume(&in_sample_shape[axis + 1], &in_sample_shape[Dims])
                              : 1;
      int64_t axis_stride = volume(&in_sample_shape[axis + 1], &in_sample_shape[Dims]);
      int64_t in_stride = axis_stride * in_sample_shape[axis];
      int64_t out_stride = axis_stride * out_sample_shape[axis];
      int64_t n = in_sample_shape[axis];
      int j = 0;
      for (int64_t outer = 0; outer < n_outer; ++outer) {
        for (int64_t inner = 0; inner < n_inner; ++inner) {
          int64_t in_idx = outer * in_stride + inner;
          int64_t out_idx = outer * out_stride + inner;

          LOG_LINE << "Sample " << s << " / " << batch_size_ << "\n";
          LOG_LINE << "Frame " << j << " / " << n_frames << "\n";
          std::vector<float> in_buf(n, 0);
          LOG_LINE << "Input: ";
          for (int64_t i = 0; i < n; i++) {
            in_buf[i] = in[in_idx];
            in_idx += inner_stride;
          }
          LOG_LINE << "\n";
          int ndct = args_[s].ndct > 0 ? args_[s].ndct : in_shape_[s][axis];
          std::vector<float> ref(ndct, 0);
          ReferenceDct(args_[s].dct_type, make_span(ref), make_cspan(in_buf), args_[s].normalize);
          LOG_LINE << "DCT (type " << args_[s].dct_type << "):";
          for (int k = 0; k < ndct; k++) {
            EXPECT_NEAR(ref[k], out[out_idx], 1e-3);
            out_idx += inner_stride;
          }
          LOG_LINE << "\n";
          ++j;
        }
      }
    }
  }
}

INSTANTIATE_TEST_SUITE_P(Dct1DGpuTest, Dct1DGpuTest, testing::Combine(
    testing::Values(1, 6, 12),  // batch_size
    testing::Values(std::vector<int>{1},
                    std::vector<int>{0, Dims-1, 1},
                    std::vector<int>(5, 1))  // axes
  ));  // NOLINT

}  // namespace test
}  // namespace dct
}  // namespace signal
}  // namespace kernels
}  // namespace dali
