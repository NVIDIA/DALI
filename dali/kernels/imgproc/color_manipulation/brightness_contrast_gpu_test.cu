// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
#include <vector>
#include <tuple>
#include "dali/kernels/scratch.h"
#include "dali/kernels/tensor_shape.h"
#include "dali/kernels/common/copy.h"
#include "dali/kernels/test/tensor_test_utils.h"
#include "dali/kernels/imgproc/color_manipulation/brightness_contrast_gpu.h"

namespace dali {
namespace kernels {
namespace brightness_contrast {
namespace test {

namespace {

static constexpr size_t kNdims = 3;


template <class In, class Out>
std::enable_if_t<std::is_integral<Out>::value, Out> custom_round(float val) {
  return std::round(val);//TODO round to nearest even
}


template <class In, class Out>
std::enable_if_t<!std::is_integral<Out>::value, Out> custom_round(float val) {
  return val;
}


}  // namespace

class BrightnessContrastGpuTest : public ::testing::Test {
//  using In = typename InputOutputTypes::In;
//  using Out = typename InputOutputTypes::Out;
  using In=float;
  using Out=float;

 public:
  BrightnessContrastGpuTest() {
    input_host_.resize(dataset_size());
  }


  void SetUp() final {
    std::mt19937_64 rng;
    UniformRandomFill(input_host_, rng, 0., 10.);
    calc_output(0);
    CUDA_CALL(cudaMalloc(&input_device_, sizeof(In) * dataset_size()));
    CUDA_CALL(cudaMemcpy(input_device_, input_host_.data(), input_host_.size() * sizeof(In),
                         cudaMemcpyDefault));
    CUDA_CALL(cudaMallocManaged(&output_, dataset_size() * sizeof(Out)));
    cudaDeviceSynchronize();

    verify_test();
  }


  In *input_device_;
  Out *output_;
  std::vector<In> input_host_;
  std::vector<Out> ref_output_;
  std::vector<TensorShape<kNdims>> shapes_ = {{2, 4, 3}};
  std::vector<float> brightness_ = {4};
  std::vector<float> contrast_ = {3};


  void verify_test() {
    assert(shapes_.size() == brightness_.size());
    assert(brightness_.size() == contrast_.size());
    assert(dataset_size() == input_host_.size());
    assert(dataset_size() == ref_output_.size());
  }


  void calc_output(int idx) {
    for (auto in : input_host_) {
      ref_output_.push_back(custom_round<In, Out>(in * contrast_[idx] + brightness_[idx]));
    }
  }


  size_t dataset_size() {
    int ret = 0;
    for (auto sh : shapes_) {
      ret += volume(sh);
    }
    return ret;
  }
};

//using TestTypes = std::tuple<uint8_t>;
//using TestTypes = std::tuple<uint8_t, int8_t, uint16_t, int16_t, int32_t, float>;
//INPUT_OUTPUT_TYPED_TEST_SUITE(BrightnessContrastCudaKernelTest, TestTypes);

namespace {
using TheKernel = BrightnessContrastGpu<float, float, kNdims>;
}  // namespace

TEST_F(BrightnessContrastGpuTest, check_kernel) {
  check_kernel<TheKernel>();
}


TEST_F(BrightnessContrastGpuTest, setup_test) {
  TheKernel kernel;
  KernelContext ctx;
  InListGPU<float, kNdims> in(this->input_device_, this->shapes_);
  auto reqs = kernel.Setup(ctx, in, this->brightness_, this->contrast_);
}


TEST_F(BrightnessContrastGpuTest, run_test) {
  TheKernel kernel;
  KernelContext c;
  InListGPU<float, kNdims> in(this->input_device_, this->shapes_);
  OutListGPU<float, kNdims> out(output_, TensorListShape<kNdims>(this->shapes_));

  auto reqs = kernel.Setup(c, in, this->brightness_, this->contrast_);

  ScratchpadAllocator sa;
  sa.Reserve(reqs.scratch_sizes);
  auto scratchpad = sa.GetScratchpad();
  c.scratchpad = &scratchpad;
  kernel.Run(c, out, in, this->brightness_, this->contrast_);
  cudaDeviceSynchronize();

  auto res = copy<AllocType::Host>(out[0]);
  ASSERT_EQ(static_cast<int>(ref_output_.size()), res.first.num_elements());
  for (size_t i = 0; i < ref_output_.size(); i++) {
    EXPECT_EQ(ref_output_[i], res.second.get()[i]);
  }

}


TEST_F(BrightnessContrastGpuTest, roi_to_TensorListShape) {
  using Rois = std::vector<Box<2, int>>;
  constexpr int nchannels = 3;
  Box<2, int> box1{0, 3};
  Box<2, int> box2{{0, 2},
                   {5, 6}};
  Box<2, int> box3{{0, 0},
                   {0, 0}};
  Rois rois = {box1, box2, box3};

  std::vector<TensorShape<-1>> ref = {{3, 3 * nchannels},
                                      {4, 5 * nchannels},
                                      {0, 0 * nchannels}};
  auto shs = RoiToShape(rois, nchannels);
  EXPECT_TRUE(shs == TensorListShape<-1>(ref));
}


TEST_F(BrightnessContrastGpuTest, adjust_empty_rois) {
  constexpr size_t ndims = 3;
  std::vector<Roi> rois;
  std::vector<TensorShape<ndims>> ts = {{2, 3, 4},
                                        {5, 6, 7}};
  TensorListShape<ndims> tls = ts;
  std::vector<Roi> ref = {
          {{0, 0}, {3, 2}},
          {{0, 0}, {6, 5}},
  };
  auto res = AdjustRois(rois, tls);
  ASSERT_EQ(ref.size(), res.size());
  for (size_t i = 0; i < ref.size(); i++) {
    EXPECT_EQ(ref[i], res[i]);
  }

}


TEST_F(BrightnessContrastGpuTest, adjust_rois) {
  constexpr size_t ndims = 3;

  std::vector<Roi> rois = {
          {{1, 2}, {3, 4}},
          {{5, 6}, {7, 8}},
  };
  std::vector<TensorShape<ndims>> ts = {{9,  10, 11},
                                        {12, 13, 14}};
  TensorListShape<ndims> tls = ts;
  std::vector<Roi> ref = {
          {{1, 2}, {3, 4}},
          {{5, 6}, {7, 8}},
  };
  auto res = AdjustRois(rois, tls);
  ASSERT_EQ(ref.size(), res.size());
  for (size_t i = 0; i < ref.size(); i++) {
    EXPECT_EQ(ref[i], res[i]);
  }
}


TEST_F(BrightnessContrastGpuTest, sample_descriptors) {
  {
    InListGPU<float, kNdims> in(this->input_device_, this->shapes_);
    OutListGPU<float, kNdims> out(output_, TensorListShape<3>(this->shapes_));
    auto res = CreateSampleDescriptors(in, out, this->brightness_, this->contrast_);
    EXPECT_EQ(this->input_device_, res[0].in);
    EXPECT_EQ(this->output_, res[0].out);
    ivec<kNdims - 1> ref_pitch = {2, 12};
    EXPECT_EQ(ref_pitch, res[0].in_pitch);
    EXPECT_EQ(ref_pitch, res[0].out_pitch);
    EXPECT_EQ(brightness_[0], res[0].brightness);
    EXPECT_EQ(contrast_[0], res[0].contrast);
  }

  {
    constexpr int ndims = 7;
    std::vector<TensorShape<ndims>> vts = {{7, 2, 4, 6, 1, 8, 4}};
    TensorListShape<ndims> tls(vts);
    InListGPU<float, ndims> in(this->input_device_, tls);
    OutListGPU<float, ndims> out(output_, tls);
    auto res = CreateSampleDescriptors(in, out, this->brightness_, this->contrast_);
    ivec<ndims - 1> ref = {7, 2, 4, 6, 1, 32};
    EXPECT_EQ(ref, res[0].in_pitch);
  }

}


}  // namespace test
}  // namespace brightness_contrast
}  // namespace kernels
}  // namespace dali
