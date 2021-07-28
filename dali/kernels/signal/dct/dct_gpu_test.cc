// Copyright (c) 2020-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <string>
#include "dali/kernels/signal/dct/dct_gpu.h"
#include "dali/kernels/kernel_manager.h"
#include "dali/kernels/common/utils.h"
#include "dali/test/tensor_test_utils.h"
#include "dali/test/test_tensors.h"
#include "dali/kernels/signal/dct/dct_test.h"
#include "dali/core/dev_buffer.h"

namespace dali {
namespace kernels {
namespace signal {
namespace dct {
namespace test {

class Dct1DGpuTest : public ::testing::TestWithParam<
  std::tuple<int, float, std::pair<int, std::vector<int>>>> {
 public:
  Dct1DGpuTest()
      : batch_size_(std::get<0>(GetParam()))
      , lifter_(std::get<1>(GetParam()))
      , dims_(std::get<2>(GetParam()).first)
      , axes_(std::get<2>(GetParam()).second)
      , in_shape_(batch_size_, dims_) {
        if (lifter_) {
          FillLifter();
          lifter_coeffs_gpu_buffer.resize(max_ndct);
          lifter_coeffs_gpu_ = make_tensor_gpu<1>(lifter_coeffs_gpu_buffer.data(), {max_ndct});
          CUDA_CALL(cudaMemcpy(lifter_coeffs_gpu_.data, lifter_coeffs_.data(),
                               lifter_coeffs_.size() * sizeof(float), cudaMemcpyHostToDevice));
        }
        while (args_.size() < static_cast<size_t>(batch_size_) * axes_.size()) {
          for (auto dct : dct_type) {
            for (auto norm : normalize) {
              for (auto n : ndct) {
                if (norm && dct == 1) continue;
                args_.push_back(DctArgs{dct, norm, n});
              }
            }
          }
        }
      }

  ~Dct1DGpuTest() override = default;

 protected:
  void FillLifter() {
    lifter_coeffs_.resize(max_ndct);
    for (int i = 0; i < max_ndct; ++i) {
      lifter_coeffs_[i] = 1.0 + lifter_ / 2 * std::sin(M_PI / lifter_ * (i + 1));
    }
  }

  void PrepareInput() {
    std::mt19937_64 rng{12345};
    std::uniform_int_distribution<> dim_dist(1, 3);
    auto rand_shape = [&]() {
      TensorShape<> shape;
      shape.resize(dims_);
      for (int i = 0; i < dims_; ++i)
        shape[i] = dim_dist(rng) * 10;
      return shape;
    };
    for (int s = 0; s < batch_size_; ++s) {
      in_shape_.set_tensor_shape(s, rand_shape());
    }
    args_span_ = span<DctArgs>(args_.data() + args_idx_, batch_size_);
    args_idx_ += batch_size_;
    ttl_in_.reshape(in_shape_);
    UniformRandomFill(ttl_in_.cpu(), rng, 0., 1.);
  }

  TensorListShape<> OutputShape(int axis) {
    assert(axis < dims_);
    TensorListShape<> out_shape(batch_size_, dims_);
    for (int s = 0; s < batch_size_; ++s) {
      auto in_shape = in_shape_[s];
      int ndct = args_span_[s].ndct;
      if (ndct > 0)
        in_shape[axis] = ndct;
      out_shape.set_tensor_shape(s, in_shape);
    }
    return out_shape;
  }

  int batch_size_;
  float lifter_;
  int dims_;
  std::vector<int> axes_;
  TensorListShape<> in_shape_;
  TestTensorList<float> ttl_in_;
  TestTensorList<float> ttl_out_;
  std::vector<DctArgs> args_;
  std::vector<float> lifter_coeffs_;
  DeviceBuffer<float> lifter_coeffs_gpu_buffer;
  OutTensorGPU<float, 1> lifter_coeffs_gpu_{};
  int args_idx_ = 0;
  span<const DctArgs> args_span_;
  const std::array<int, 4> dct_type = {{1, 2, 3, 4}};
  const std::array<bool, 2> normalize = {{false, true}};
  const std::array<int, 3> ndct = {{-1, 10, 20}};
  const int max_ndct = 40;
};

TEST_P(Dct1DGpuTest, DctTest) {
  using Kernel = Dct1DGpu<float>;
  KernelContext ctx;
  ctx.gpu.stream = 0;
  KernelManager kmgr;
  kmgr.Initialize<Kernel>();
  kmgr.Resize<Kernel>(1, 1);

  for (auto axis : axes_) {
    PrepareInput();
    auto out_shape = OutputShape(axis);
    auto in_view = ttl_in_.gpu();
    auto req = kmgr.Setup<Kernel>(0, ctx, in_view, args_span_, axis);
    ASSERT_EQ(out_shape, req.output_shapes[0]);
    ttl_out_.reshape(out_shape);
    auto out_view = ttl_out_.gpu();
    kmgr.Run<Kernel>(0, 0, ctx, out_view, in_view, lifter_coeffs_gpu_);
    CUDA_CALL(cudaStreamSynchronize(ctx.gpu.stream));
    auto cpu_in_view = ttl_in_.cpu();
    auto cpu_out_view = ttl_out_.cpu();
    for (int s = 0; s < batch_size_; ++s) {
      auto in = cpu_in_view.tensor_data(s);
      auto out = cpu_out_view.tensor_data(s);
      auto in_sample_shape = in_shape_.tensor_shape_span(s);
      auto out_sample_shape = out_shape.tensor_shape_span(s);
      int64_t n_outer = volume(in_sample_shape.begin(), in_sample_shape.begin() + axis);
      int64_t n_inner = volume(in_sample_shape.begin() + axis + 1, in_sample_shape.end());
      int64_t n_frames = n_outer * n_inner;
      int64_t inner_stride = (axis < dims_ - 1)
                              ? volume(in_sample_shape.begin() + axis + 1, in_sample_shape.end())
                              : 1;
      int64_t in_stride = inner_stride * in_sample_shape[axis];
      int64_t out_stride = inner_stride * out_sample_shape[axis];
      int64_t n = in_sample_shape[axis];
      int j = 0;
      DctArgs args = args_span_[s];
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
          int ndct = args.ndct > 0 ? args.ndct : in_shape_[s][axis];
          std::vector<float> ref(ndct, 0);
          ReferenceDct(args.dct_type, make_span(ref), make_cspan(in_buf), args.normalize, lifter_);
          LOG_LINE << "DCT (type " << args.dct_type << "):";
          for (int k = 0; k < ndct; k++) {
            EXPECT_NEAR(ref[k], out[out_idx], 1e-5);
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
    testing::Values(1, 12),  // batch_size
    testing::Values(0.f, 0.5f),  // lifter
    testing::Values(std::make_pair(2, std::vector<int>{1}),
                    std::make_pair(4, std::vector<int>{0, 3, 1}),
                    std::make_pair(1, std::vector<int>{0, 0}))  // dims, axes
  ));  // NOLINT


class Dct1DGpuPerfTest : public ::testing::TestWithParam<bool> {
 protected:
  Dct1DGpuPerfTest(): inner_(GetParam()) {}

  void SetUp() override {
    DctArgs args;
    args.dct_type = 2;
    args.normalize = false;
    args.ndct = ndct_;
    args_batch_ = std::vector<DctArgs>(batch_size_, args);
    const int64_t nframes = 20000;
    for (int s = 0; s < batch_size_; ++s) {
      if (inner_) {
        in_shape_.set_tensor_shape(s, {1, nframes, 60 + s % n_tables_});
        out_shape_.set_tensor_shape(s, {1, nframes, args.ndct});
      } else {
        in_shape_.set_tensor_shape(s, {1, 60 + s % n_tables_, nframes});
        out_shape_.set_tensor_shape(s, {1, args.ndct, nframes});
      }
    }
    if (inner_) {
      layout_ = "interleaved";
    } else {
      layout_ = "planar";
    }
  }

  const int batch_size_ = 128;
  const int n_tables_ = 10;
  const int ndct_ = 64;
  TensorListShape<3> in_shape_{batch_size_};
  TensorListShape<3> out_shape_{batch_size_};
  std::vector<DctArgs> args_batch_;
  bool inner_;
  std::string layout_;
};

TEST_P(Dct1DGpuPerfTest, DISABLED_PerfTest) {
  using Kernel = Dct1DGpu<float>;
  const int batch_size = 64;
  const int n_tables = 5;
  const int n_iters = 6;
  TestTensorList<float, 3> input;
  TestTensorList<float, 3> output;
  int64_t out_elems = out_shape_.num_elements();
  int64_t in_elems = in_shape_.num_elements();
  int64_t cosine_elems = 0;
  for (int i = 0; i < n_tables; ++i) {
    cosine_elems += static_cast<int64_t>(ndct_) * (60 + i);
  }
  int64_t mem_size = sizeof(float) * (out_elems + in_elems + cosine_elems);
  input.reshape(in_shape_);
  output.reshape(out_shape_);
  std::mt19937_64 rng{12345};
  KernelContext ctx;
  ctx.gpu.stream = 0;
  KernelManager kmgr;
  kmgr.Initialize<Kernel>();
  kmgr.Resize<Kernel>(1, 1);
  CUDAEvent start = CUDAEvent::CreateWithFlags(0);
  CUDAEvent end = CUDAEvent::CreateWithFlags(0);
  double tops = 0;
  double mem = 0;
  std::cout << "DCT GPU Perf test.\nLayout: " << layout_
            << "\nInput size: " << static_cast<double>(sizeof(float) * in_elems) * 1e-9 << " GB"
            << std::endl;
  for (int i = 0; i < n_iters; ++i) {
    std::cout << "ITER " << i+1 << '/' << n_iters << std::endl;
    input.reshape(in_shape_);
    output.reshape(out_shape_);
    UniformRandomFill(input.cpu(), rng, 0., 1.);
    auto in_view_gpu = input.gpu();
    auto out_view_gpu = output.gpu();
    int axis = inner_ ? 2 : 1;
    auto req = kmgr.Setup<Kernel>(0, ctx, in_view_gpu, make_cspan(args_batch_), axis);
    CUDA_CALL(cudaEventRecord(start));
    kmgr.Run<Kernel>(0, 0, ctx, out_view_gpu, in_view_gpu, InTensorGPU<float, 1>{});
    CUDA_CALL(cudaEventRecord(end));
    CUDA_CALL(cudaDeviceSynchronize());
    float time;
    CUDA_CALL(cudaEventElapsedTime(&time, start, end));
    if (i > 0) {
      tops += in_elems * ndct_ / static_cast<double>(time) * 1e-6 / (n_iters - 1);
      mem += static_cast<double>(mem_size) / (static_cast<double>(time) * 1e6) / (n_iters - 1);
    }
  }
  std::cout << "---- AVERAGE ----" << std::endl;
  std::cout << "Compute: " << tops << " GFLOPS" << std::endl;
  std::cout << "Memory: " << mem  << "GB/s" << std::endl;
}

INSTANTIATE_TEST_SUITE_P(Dct1DGpuPerfTest, Dct1DGpuPerfTest,
  testing::Values(false, true)  // interleaved layout
);  // NOLINT

}  // namespace test
}  // namespace dct
}  // namespace signal
}  // namespace kernels
}  // namespace dali
