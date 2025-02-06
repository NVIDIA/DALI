// Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <string>
#include <vector>
#include "dali/core/cuda_stream_pool.h"
#include "dali/core/cuda_event.h"
#include "dali/kernels/dynamic_scratchpad.h"
#include "dali/kernels/slice/slice_cpu.h"
#include "dali/kernels/slice/slice_flip_normalize_gpu.h"
#include "dali/kernels/slice/slice_flip_normalize_permute_pad_gpu.h"
#include "dali/kernels/slice/slice_kernel_test.h"
#include "dali/test/dali_test_config.h"
#include "dali/util/file.h"
#include "dali/util/numpy.h"

namespace dali {
namespace kernels {
namespace slice_flip_normalize {
namespace test {

template <typename Out, typename In, int spatial_ndim, int channel_dim>
class SliceFlipNormalizeGPUTest : public ::testing::Test {
 public:
  static constexpr int ndim = spatial_ndim + (channel_dim >= 0);
  static constexpr int d_dim = spatial_ndim < 3 ? -1 : channel_dim == 0 ? 1 : 0;
  static constexpr int h_dim = 0 + (channel_dim == 0) + (spatial_ndim == 3);
  static constexpr int w_dim = h_dim + 1;
  using Kernel = SliceFlipNormalizeGPU<Out, In, spatial_ndim, channel_dim>;

  void RunKernel(TestTensorList<Out, ndim> &output, TestTensorList<In, ndim> &input,
                 const typename Kernel::Args &args) {
    KernelContext ctx;
    DynamicScratchpad scratchpad;
    ctx.scratchpad = &scratchpad;
    ctx.gpu.stream = 0;

    Kernel kernel;

    auto in_gpu = input.gpu();
    auto req = kernel.Setup(ctx, in_gpu.shape, args);

    auto out_sh = req.output_shapes[0].template to_static<ndim>();
    output.reshape(out_sh);

    kernel.Run(ctx, output.gpu(), input.gpu(), args);

    CUDA_CALL(cudaStreamSynchronize(0));
  }

  template <typename T, int ndim>
  void FeedData(TestTensorList<T, ndim> &data,
                const std::vector<Tensor<CPUBackend>>& samples) {
    TensorListShape<ndim> sh;
    sh.resize(samples.size());
    int nsamples = samples.size();
    for (int s = 0; s < nsamples; s++) {
      sh.set_tensor_shape(s, samples[s].shape());
    }
    data.reshape(sh);
    auto data_cpu = data.cpu();
    for (int s = 0; s < nsamples; s++) {
      auto &t = samples[s];
      std::memcpy(data_cpu.tensor_data(s), t.template data<T>(), volume(t.shape()) * sizeof(T));
    }
  }

  void LoadTensor(Tensor<CPUBackend> &tensor, const std::string& path_npy) {
    auto stream = FileStream::Open(path_npy);
    tensor = ::dali::numpy::ReadTensor(stream.get(), true);
  }

  template <typename T, int ndim>
  void LoadBatch(TestTensorList<T, ndim> &data, const std::string& dir,
                 const std::vector<std::string>& filenames) {
    std::vector<Tensor<CPUBackend>> samples;
    samples.resize(filenames.size());
    for (size_t i = 0; i < filenames.size(); i++) {
      LoadTensor(samples[i], dir + "/" + filenames[i]);
    }
    FeedData(data, samples);
  }
};

class SliceFlipNormalizeGPUTest_uint8_uint8_2D_HWC
    : public SliceFlipNormalizeGPUTest<uint8_t, uint8_t, 2, 2> {};
class SliceFlipNormalizeGPUTest_float_uint8_2D_HWC
    : public SliceFlipNormalizeGPUTest<float, uint8_t, 2, 2> {};

TEST_F(SliceFlipNormalizeGPUTest_uint8_uint8_2D_HWC, only_crop) {
  auto tc_dir = testing::dali_extra_path() + "/db/test_data/crop_mirror_normalize";

  TestTensorList<uint8_t, 3> input;
  TestTensorList<uint8_t, 3> ref;
  this->LoadBatch(input, tc_dir,  {"input0.npy", "input1.npy", "input2.npy"});
  this->LoadBatch(ref, tc_dir, {"output0_c.npy", "output1_c.npy", "output2_c.npy"});

  TestTensorList<uint8_t, 3> output;
  typename Kernel::Args args;
  args.perm = {0, 1, 2};
  args.sample_args = {
    { Roi<2>{{8, 2}, {19, 8}} },
    { Roi<2>{{2, 22}, {4, 28}} },
    { Roi<2>{{4, 55}, {5, 100}} }
  };

  this->RunKernel(output, input, args);

  Check(ref.cpu(), output.cpu(), EqualEps(1e-6));
}

TEST_F(SliceFlipNormalizeGPUTest_uint8_uint8_2D_HWC, crop_mirror) {
  auto tc_dir = testing::dali_extra_path() + "/db/test_data/crop_mirror_normalize";

  TestTensorList<uint8_t, 3> input;
  TestTensorList<uint8_t, 3> ref;
  this->LoadBatch(input, tc_dir,  {"input0.npy", "input1.npy", "input2.npy"});
  this->LoadBatch(ref, tc_dir, {"output0_cm.npy", "output1_cm.npy", "output2_cm.npy"});

  TestTensorList<uint8_t, 3> output;
  typename Kernel::Args args;
  args.perm = {0, 1, 2};
  args.sample_args = {
    { Roi<2>{{8, 2}, {19, 8}}, {true, false} },
    { Roi<2>{{2, 22}, {4, 28}}, {true, true} },
    { Roi<2>{{4, 55}, {5, 100}}, {false, false} }
  };

  this->RunKernel(output, input, args);

  Check(ref.cpu(), output.cpu(), EqualEps(1e-6));
}

TEST_F(SliceFlipNormalizeGPUTest_float_uint8_2D_HWC, crop_mirror_normalize) {
  auto tc_dir = testing::dali_extra_path() + "/db/test_data/crop_mirror_normalize";

  TestTensorList<uint8_t, 3> input;
  TestTensorList<float, 3> ref;
  this->LoadBatch(input, tc_dir,  {"input0.npy", "input1.npy", "input2.npy"});
  this->LoadBatch(ref, tc_dir, {"output0_cmn.npy", "output1_cmn.npy", "output2_cmn.npy"});

  TestTensorList<float, 3> output;
  typename Kernel::Args args;
  args.perm = {0, 1, 2};

  args.sample_args = {
    { Roi<2>{{8, 2}, {19, 8}}, {true, false},
      {255 * 0.485f, 255 * 0.456f, 255 * 0.406f},
      {1.0f / (255 * 0.229f), 1.0f / (255 * 0.224f), 1.0f / (255 * 0.225f)} },
    { Roi<2>{{2, 22}, {4, 28}}, {true, true},
      {255 * 0.455f, 255 * 0.436f, 255 * 0.416f},
      {1.0f / (255 * 0.225f), 1.0f / (255 * 0.224f), 1.0f / (255 * 0.221f)} },
    { Roi<2>{{4, 55}, {5, 100}}, {false, false},
      {255 * 0.495f, 255 * 0.466f, 255 * 0.396f},
      {1.0f / (255 * 0.226f), 1.0f / (255 * 0.229f), 1.0f / (255 * 0.222f)} }
  };

  this->RunKernel(output, input, args);

  Check(ref.cpu(), output.cpu(), EqualEps(1e-6));
}

TEST_F(SliceFlipNormalizeGPUTest_float_uint8_2D_HWC, crop_mirror_normalize_transpose) {
  auto tc_dir = testing::dali_extra_path() + "/db/test_data/crop_mirror_normalize";

  TestTensorList<uint8_t, 3> input;
  TestTensorList<float, 3> ref;
  this->LoadBatch(input, tc_dir, {"input0.npy", "input1.npy", "input2.npy"});
  this->LoadBatch(ref, tc_dir, {"output0_cmnt.npy", "output1_cmnt.npy", "output2_cmnt.npy"});

  TestTensorList<float, 3> output;
  typename Kernel::Args args;
  args.perm = {2, 0, 1};

  args.sample_args = {
    { Roi<2>{{8, 2}, {19, 8}}, {true, false},
      {255 * 0.485f, 255 * 0.456f, 255 * 0.406f},
      {1.0f / (255 * 0.229f), 1.0f / (255 * 0.224f), 1.0f / (255 * 0.225f)} },
    { Roi<2>{{2, 22}, {4, 28}}, {true, true},
      {255 * 0.455f, 255 * 0.436f, 255 * 0.416f},
      {1.0f / (255 * 0.225f), 1.0f / (255 * 0.224f), 1.0f / (255 * 0.221f)} },
    { Roi<2>{{4, 55}, {5, 100}}, {false, false},
      {255 * 0.495f, 255 * 0.466f, 255 * 0.396f},
      {1.0f / (255 * 0.226f), 1.0f / (255 * 0.229f), 1.0f / (255 * 0.222f)} }
  };

  this->RunKernel(output, input, args);

  Check(ref.cpu(), output.cpu(), EqualEps(1e-6));
}

TEST_F(SliceFlipNormalizeGPUTest_float_uint8_2D_HWC, pad_normalize) {
  auto tc_dir = testing::dali_extra_path() + "/db/test_data/crop_mirror_normalize";

  TestTensorList<uint8_t, 3> input;
  TestTensorList<float, 3> ref;
  this->LoadBatch(input, tc_dir, {"input0.npy", "input1.npy", "input2.npy"});
  this->LoadBatch(ref, tc_dir, {"output0_pn.npy", "output1_pn.npy", "output2_pn.npy"});

  TestTensorList<float, 3> output;
  typename Kernel::Args args;
  args.perm = {};

  args.sample_args = {
    { Roi<2>{{0, -2}, {19, 8}}, {false, false},
      {255 * 0.485f, 255 * 0.456f, 255 * 0.406f},
      {1.0f / (255 * 0.229f), 1.0f / (255 * 0.224f), 1.0f / (255 * 0.225f)},
      {255.0f, 128.0f, 64.0f, 32.0f}
    },
    { Roi<2>{{-4, 0}, {4, 28}}, {false, false},
      {255 * 0.455f, 255 * 0.436f, 255 * 0.416f},
      {1.0f / (255 * 0.225f), 1.0f / (255 * 0.224f), 1.0f / (255 * 0.221f)},
      {255.0f + 1, 128.0f + 1, 64.0f + 1, 32.0f + 1}
    },
    { Roi<2>{{0, 0}, {5, 120}}, {false, false},
      {255 * 0.495f, 255 * 0.466f, 255 * 0.396f},
      {1.0f / (255 * 0.226f), 1.0f / (255 * 0.229f), 1.0f / (255 * 0.222f)},
      {255.0f + 2, 128.0f + 2, 64.0f + 2, 32.0f + 2}
    }
  };

  this->RunKernel(output, input, args);

  Check(ref.cpu(), output.cpu(), EqualEps(1e-6));
}


TEST(SliceFlipNormalizeGPUTest, Benchmark) {
  using Kernel = SliceFlipNormalizeGPU<float, uint8_t, 2, 2>;
  typename Kernel::Args args;
  Kernel kernel;

  int nsamples = 64;
  using InType = uint8_t;
  using OutType = float;
  TestTensorList<InType, 3> in;
  TensorListShape<3> in_sh = uniform_list_shape(nsamples, TensorShape<3>{1280, 900, 3});
  in.reshape(in_sh);

  args.perm = {2, 0, 1};
  args.sample_args.resize(nsamples);
  for (int i = 0; i < nsamples; i++) {
    args.sample_args[i] = {Roi<2>{{400, 300}, {400 + 256, 300 + 256}},
         {false, false},
         {255 * 0.495f, 255 * 0.466f, 255 * 0.396f},
         {1.0f / (255 * 0.226f), 1.0f / (255 * 0.229f), 1.0f / (255 * 0.222f)}};
  }

  auto bench = [&]() {
    auto stream = CUDAStreamPool::instance().Get();

    KernelContext ctx;
    ctx.gpu.stream = stream;
    DynamicScratchpad scratch;
    ctx.scratchpad = &scratch;

    TestTensorList<OutType, 3> out;
    auto req = kernel.Setup(ctx, in_sh, args);
    out.reshape(req.output_shapes[0].to_static<3>());
    auto out_view = out.gpu();
    auto in_view = in.gpu();

    // warm-up
    CUDAEvent started  = CUDAEvent::CreateWithFlags(0);  // timing enabled
    CUDAEvent finished = CUDAEvent::CreateWithFlags(0);
    CUDA_CALL(cudaStreamSynchronize(ctx.gpu.stream));
    kernel.Run(ctx, out_view, in_view, args);
    int iters = 100;
    CUDA_CALL(cudaEventRecord(started, ctx.gpu.stream));
    for (int i = 0; i < iters; i++)
      kernel.Run(ctx, out_view, in_view, args);
    CUDA_CALL(cudaEventRecord(finished, ctx.gpu.stream));
    CUDA_CALL(cudaStreamSynchronize(ctx.gpu.stream));
    float time_ms = 0;
    CUDA_CALL(cudaEventElapsedTime(&time_ms, started, finished));
    time_ms /= iters;
    // note: using out_view.num_elements() twice in purpose (we are not reading all input)
    int64_t data_size =
        out_view.num_elements() * sizeof(InType) + out_view.num_elements() * sizeof(OutType);
    std::cout << data_size * 1e-6 / time_ms << " GB/s" << std::endl;
  };

  std::cout << "with transpose: ";
  args.perm = {2, 0, 1};
  bench();
  std::cout << "no transpose: ";
  args.perm = {0, 1, 2};
  bench();
}

TEST(SliceFlipNormalizeGPUTest, BenchmarkOld) {
  using Kernel = SliceFlipNormalizePermutePadGpu<float, uint8_t, 3>;
  Kernel kernel;
  std::vector<SliceFlipNormalizePermutePadArgs<3>> args;
  int nsamples = 64;
  args.reserve(nsamples);
  for (int i = 0; i < nsamples; i++) {
    args.emplace_back(TensorShape<3>{256, 256, 3}, TensorShape<3>{1280, 900, 3});
    auto &a = args.back();
    a.anchor[0] = 400;
    a.anchor[1] = 300;
    a.flip[0] = a.flip[1] = false;
    a.permuted_dims[0] = 2;
    a.permuted_dims[1] = 0;
    a.permuted_dims[2] = 1;
    a.mean.resize(3);
    a.mean[0] = 255 * 0.495f;
    a.mean[1] = 255 * 0.466f;
    a.mean[2] = 255 * 0.396f;
    a.inv_stddev.resize(3);
    a.inv_stddev[0] = 1.0f / (255 * 0.226f);
    a.inv_stddev[1] = 1.0f / (255 * 0.229f);
    a.inv_stddev[2] = 1.0f / (255 * 0.222f);
    a.fill_values.resize(3, 0.0f);
    a.channel_dim = 2;
  }

  using InType = uint8_t;
  using OutType = float;
  TensorListShape<3> in_sh = uniform_list_shape(nsamples, TensorShape<3>{1280, 900, 3});
  TestTensorList<InType, 3> in;
  in.reshape(in_sh);
  auto in_view = in.gpu();

  auto bench = [&]() {
    auto stream = CUDAStreamPool::instance().Get();

    KernelContext ctx;
    ctx.gpu.stream = stream;
    DynamicScratchpad scratch;
    ctx.scratchpad = &scratch;

    TestTensorList<OutType, 3> out;
    KernelRequirements req = kernel.Setup(ctx, in_view, args);
    out.reshape(req.output_shapes[0].to_static<3>());
    auto out_view = out.gpu();

    // warm-up
    CUDAEvent started  = CUDAEvent::CreateWithFlags(0);  // timing enabled
    CUDAEvent finished = CUDAEvent::CreateWithFlags(0);
    CUDA_CALL(cudaStreamSynchronize(ctx.gpu.stream));
    kernel.Run(ctx, out_view, in_view, args);
    int iters = 100;
    CUDA_CALL(cudaEventRecord(started, ctx.gpu.stream));
    for (int i = 0; i < iters; i++)
      kernel.Run(ctx, out_view, in_view, args);
    CUDA_CALL(cudaEventRecord(finished, ctx.gpu.stream));
    CUDA_CALL(cudaStreamSynchronize(ctx.gpu.stream));
    float time_ms = 0;
    CUDA_CALL(cudaEventElapsedTime(&time_ms, started, finished));
    time_ms /= iters;
    // note: using out_view.num_elements() twice in purpose (we are not reading all input)
    int64_t data_size =
        out_view.num_elements() * sizeof(InType) + out_view.num_elements() * sizeof(OutType);
    std::cout << data_size * 1e-6 / time_ms << " GB/s" << std::endl;
  };


  std::cout << "with transpose: ";
  for (int i = 0; i < nsamples; i++) {
    auto &a = args[i];
    a.permuted_dims[0] = 2;
    a.permuted_dims[1] = 0;
    a.permuted_dims[2] = 1;
  }
  bench();

  std::cout << "no transpose: ";
  for (int i = 0; i < nsamples; i++) {
    auto &a = args[i];
    a.permuted_dims[0] = 0;
    a.permuted_dims[1] = 1;
    a.permuted_dims[2] = 2;
  }
  bench();
}


}  // namespace test
}  // namespace slice_flip_normalize
}  // namespace kernels
}  // namespace dali
