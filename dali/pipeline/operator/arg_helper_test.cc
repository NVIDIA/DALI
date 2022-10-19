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

#include "dali/pipeline/operator/arg_helper.h"
#include <gtest/gtest.h>
#include "dali/pipeline/operator/op_spec.h"
#include "dali/pipeline/workspace/workspace.h"

namespace dali {

DALI_SCHEMA(ArgHelperTestOp)
  .DocStr(R"(Dummy op schema)")
  .AddOptionalArg<vector<float>>("arg", R"(dummy vector<float> argument)",
    nullptr,  // no default value
    true)
  .AddOptionalArg<float>("scalar", R"(dummy float argument)",
    nullptr,  // no default value
    true);


static constexpr int kNumSamples = 5;

template <int ndim>
void SetupData(TensorList<CPUBackend> &tv, TensorListShape<ndim> sh) {
  tv.set_pinned(false);
  tv.Resize(sh, DALI_FLOAT);
  for (int i = 0; i < tv.num_samples(); i++) {
    float *data = tv.mutable_tensor<float>(i);
    for (int j = 0; j < volume(sh[i]); j++) {
      data[j] = 100 * i + j;
    }
  }
}

template <int ndim, typename... AcquireArgs>
void ArgValueTestTensorInput(TensorListShape<ndim> ts, AcquireArgs... args) {
  OpSpec spec("ArgHelperTestOp");
  ArgumentWorkspace ws;
  auto arg_data = std::make_shared<TensorList<CPUBackend>>();
  SetupData(*arg_data, ts);
  ws.AddArgumentInput("arg", arg_data);
  spec.AddArgumentInput("arg", "arg");

  ArgValue<float, ndim> arg("arg", spec);
  arg.Acquire(spec, ws, kNumSamples, args...);

  ASSERT_TRUE(arg.HasArgumentInput());
  ASSERT_EQ(kNumSamples, arg.size());
  for (int i = 0; i < kNumSamples; i++) {
    auto sh = ts[i];
    ASSERT_EQ(sh, arg[i].shape);
    for (int j = 0; j < volume(sh); j++) {
      float *ptr = arg_data->mutable_tensor<float>(i);
      ASSERT_EQ(ptr[j], arg[i].data[j]);
    }
  }
}

template <int ndim, typename... AcquireArgs>
void ArgValueTestAllowEmpty(TensorListShape<ndim> expected_sh, AcquireArgs... args) {
  ASSERT_TRUE(is_uniform(expected_sh));  // the test assumes that
  auto sh = expected_sh;

  int empty_sample_idx = 2;
  sh.tensor_shape_span(empty_sample_idx)[0] = 0;  // this makes the sample empty

  EXPECT_THROW(ArgValueTestTensorInput<ndim>(sh, expected_sh, ArgValue_Default),
               std::runtime_error);
  ArgValueTestTensorInput<ndim>(sh, expected_sh, ArgValue_AllowEmpty);

  auto expected_sample_sh = sh[0];
  EXPECT_THROW(ArgValueTestTensorInput<ndim>(sh, expected_sample_sh, ArgValue_Default),
               std::runtime_error);
  ArgValueTestTensorInput<ndim>(sh, expected_sample_sh, ArgValue_AllowEmpty);

  OpSpec spec("ArgHelperTestOp");
  ArgumentWorkspace ws;
  auto arg_data = std::make_shared<TensorList<CPUBackend>>();
  SetupData(*arg_data, sh);
  ws.AddArgumentInput("arg", arg_data);
  spec.AddArgumentInput("arg", "arg");
  ArgValue<float, ndim> arg("arg", spec);
  arg.Acquire(spec, ws, kNumSamples, expected_sample_sh, ArgValue_AllowEmpty);

  EXPECT_EQ(kNumSamples, arg.size());
  EXPECT_TRUE(arg.HasValue());
  EXPECT_TRUE(arg);

  for (int i = 0; i < kNumSamples; i++)
    EXPECT_EQ(i == empty_sample_idx, arg.IsEmpty(i));

  // All empty
  OpSpec spec2("ArgHelperTestOp");
  spec2.AddArg("arg", std::vector<float>{});
  ArgumentWorkspace ws2;
  ArgValue<float, ndim> arg2("arg", spec2);
  arg2.Acquire(spec2, ws2, kNumSamples, expected_sample_sh, ArgValue_AllowEmpty);

  EXPECT_EQ(kNumSamples, arg2.size());
  EXPECT_TRUE(arg2.HasValue());
  EXPECT_TRUE(arg2);
  for (int i = 0; i < kNumSamples; i++)
    EXPECT_TRUE(arg2.IsEmpty(i));

  // Not provided
  OpSpec spec3("ArgHelperTestOp");
  ArgumentWorkspace ws3;
  ArgValue<float, ndim> arg3("arg", spec3);
  EXPECT_FALSE(arg3.HasValue());
}


TEST(ArgValue, TensorInput_0D) {
  TensorShape<0> sample_sh{};
  auto sh = uniform_list_shape(kNumSamples, sample_sh);
  ArgValueTestTensorInput<0>(sh, ArgValue_EnforceUniform);
  ArgValueTestTensorInput<0>(sh, sh);
  ArgValueTestTensorInput<0>(sh, sample_sh);
}

TEST(ArgValue, TensorInput_1D) {
  TensorShape<1> sample_sh{3};
  auto sh = uniform_list_shape(kNumSamples, sample_sh);
  ArgValueTestTensorInput<1>(sh, ArgValue_EnforceUniform);
  ArgValueTestTensorInput<1>(sh, sh);
  ArgValueTestTensorInput<1>(sh, sample_sh);
}

TEST(ArgValue, TensorInput_2D) {
  TensorShape<2> sample_sh{3, 2};
  auto sh = uniform_list_shape(kNumSamples, sample_sh);
  ArgValueTestTensorInput<2>(sh, ArgValue_EnforceUniform);
  ArgValueTestTensorInput<2>(sh, sh);
  ArgValueTestTensorInput<2>(sh, sample_sh);
}

TEST(ArgValue, TensorInput_3D) {
  TensorShape<3> sample_sh{3, 2, 2};
  auto sh = uniform_list_shape(kNumSamples, sample_sh);
  ArgValueTestTensorInput<3>(sh, ArgValue_EnforceUniform);
  ArgValueTestTensorInput<3>(sh, sh);
  ArgValueTestTensorInput<3>(sh, sample_sh);
}

TEST(ArgValue, TensorInput_3D_per_sample) {
  TensorListShape<3> sh({{37, 23, 3}, {12, 22, 3}, {42, 42, 3}, {8, 8, 3}, {64, 32, 3}});
  ASSERT_EQ(kNumSamples, sh.size());  // just in case (as it is used inside the test)
  ArgValueTestTensorInput<3>(sh, sh);
  EXPECT_THROW(ArgValueTestTensorInput<3>(sh, ArgValue_EnforceUniform),
               std::runtime_error);
  EXPECT_THROW(ArgValueTestTensorInput<3>(sh, sh[0]), std::runtime_error);
  EXPECT_THROW(ArgValueTestTensorInput<3>(sh, uniform_list_shape(kNumSamples, sh[0])),
               std::runtime_error);
}

TEST(ArgValueTests, Constant_0D) {
  int nsamples = 5;
  auto spec = OpSpec("ArgHelperTestOp").AddArg("scalar", 0.123f);
  ArgValue<float, 0> arg("scalar", spec);
  Workspace ws;
  arg.Acquire(spec, ws, nsamples);
  ASSERT_TRUE(arg.HasExplicitConstant());
  ASSERT_EQ(TensorShape<0>{}, arg[0].shape);
  ASSERT_EQ(0.123f, *arg[0].data);

  // Passing a vector to a scalar ArgValue
  auto spec2 = OpSpec("ArgHelperTestOp").AddArg("scalar", vector<float>{0.1f, 0.2f});
  ArgValue<float, 0> arg2("scalar", spec2);
  EXPECT_THROW(arg2.Acquire(spec2, ws, nsamples), std::runtime_error);
}


TEST(ArgValueTests, Constant_1D) {
  int nsamples = 5;
  std::vector<float> data{0.1f, 0.2f, 0.3f};
  TensorShape<1> expected_shape{3};
  auto spec = OpSpec("ArgHelperTestOp").AddArg("arg", data);
  ArgValue<float, 1> arg("arg", spec);
  Workspace ws;
  arg.Acquire(spec, ws, nsamples, ArgValue_EnforceUniform);
  ASSERT_TRUE(arg.HasExplicitConstant());
  for (int i = 0; i < kNumSamples; i++) {
    ASSERT_EQ(expected_shape, arg[i].shape);
    for (int j = 0; j < expected_shape.num_elements(); j++) {
      ASSERT_EQ(data[j], arg[i].data[j]);
    }
  }
}

TEST(ArgValueTests, Constant_2D) {
  int nsamples = 5;
  auto data = std::vector<float>{0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f};
  TensorShape<2> expected_shape{2, 3};
  auto spec =
    OpSpec("ArgHelperTestOp")
      .AddArg("arg", data);
  Workspace ws;

  auto shape_from_size =
    [](int64_t size) {
      int64_t mat_ndim = sqrt(size);
      assert(mat_ndim > 0);
      DALI_ENFORCE(size == mat_ndim * (mat_ndim + 1),
          make_string("Cannot form an affine transform matrix with ", size, " elements"));
      return TensorShape<2>{mat_ndim, mat_ndim + 1};
    };

  ArgValueFlags flags = ArgValue_EnforceUniform;
  ArgValue<float, 2> err("arg", spec);
  EXPECT_THROW(err.Acquire(spec, ws, nsamples, flags), std::logic_error);  // can't infer shape

  ArgValue<float, 2> arg1("arg", spec);
  arg1.Acquire(spec, ws, nsamples, flags, shape_from_size);

  ArgValue<float, 2> arg2("arg", spec);
  arg2.Acquire(spec, ws, nsamples, expected_shape);

  for (auto &arg : {arg1, arg2}) {
    ASSERT_TRUE(arg.HasExplicitConstant());
    for (int i = 0; i < kNumSamples; i++) {
      ASSERT_EQ(expected_shape, arg[i].shape);
      for (int j = 0; j < expected_shape.num_elements(); j++) {
        ASSERT_EQ(data[j], arg[i].data[j]);
      }
    }
  }
}

TEST(ArgValue, TensorInput_1D_ExpectedShape_AllowEmpty) {
  ArgValueTestAllowEmpty<1>(uniform_list_shape(kNumSamples, TensorShape<1>{3}));
}

TEST(ArgValue, TensorInput_3D_ExpectedShape_AllowEmpty) {
  ArgValueTestAllowEmpty<3>(uniform_list_shape(kNumSamples, TensorShape<3>{10, 10, 3}));
}


}  // namespace dali
