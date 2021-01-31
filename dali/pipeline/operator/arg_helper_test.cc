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

#include "dali/pipeline/operator/arg_helper.h"
#include <gtest/gtest.h>
#include "dali/pipeline/operator/op_spec.h"
#include "dali/pipeline/util/backend2workspace_map.h"
#include "dali/pipeline/workspace/workspace.h"

namespace dali {

static constexpr int kNumSamples = 5;

template <int ndim>
void SetupData(TensorVector<CPUBackend> &tv,
               TensorListShape<ndim> sh) {
  tv.set_pinned(false);
  tv.Resize(sh, TypeTable::GetTypeInfo(DALI_FLOAT));
  for (size_t i = 0; i < tv.size(); i++) {
    float *data = tv[i].mutable_data<float>();
    for (int j = 0; j < volume(sh[i]); j++) {
      data[j] = 100 * i + j;
    }
  }
}

template <int ndim>
void ArgValueTestTensorInput(TensorListShape<ndim> ts) {
  // using a real operator to avoid registering a new one just for this test
  OpSpec spec("MTTransformAttr");
  ArgumentWorkspace ws;
  auto arg_data = std::make_shared<TensorVector<CPUBackend>>();
  SetupData(*arg_data, ts);
  ws.AddArgumentInput("M", arg_data);
  spec.AddArgumentInput("M", "M");

  ArgValue<float, ndim> arg("M", spec);
  arg.Acquire(spec, ws, kNumSamples, true);

  ASSERT_TRUE(arg.IsArgInput());
  ASSERT_EQ(kNumSamples, arg.size());
  for (int i = 0; i < kNumSamples; i++) {
    auto sh = ts[i];
    ASSERT_EQ(sh, arg[i].shape);
    for (int j = 0; j < volume(sh); j++) {
      float *ptr = (*arg_data)[i].mutable_data<float>();
      ASSERT_EQ(ptr[j], arg[i].data[j]);
    }
  }
}

TEST(ArgValue, TensorInput_0D) {
  ArgValueTestTensorInput<0>(
    uniform_list_shape(kNumSamples, TensorShape<0>{}));
}

TEST(ArgValue, TensorInput_1D) {
  ArgValueTestTensorInput<1>(
    uniform_list_shape(kNumSamples, TensorShape<1>{3}));
}

TEST(ArgValue, TensorInput_2D) {
  ArgValueTestTensorInput<2>(
    uniform_list_shape(kNumSamples, TensorShape<2>{3, 2}));
}

TEST(ArgValue, TensorInput_3D) {
  ArgValueTestTensorInput<3>(
    uniform_list_shape(kNumSamples, TensorShape<3>{3, 2, 2}));
}


TEST(ArgValueTests, Constant_0D) {
  int nsamples = 5;
  auto spec = OpSpec("Erase").AddArg("shape", 0.123f);
  ArgValue<float, 0> arg("shape", spec);
  workspace_t<CPUBackend> ws;
  arg.Acquire(spec, ws, nsamples, true);
  ASSERT_TRUE(arg.IsConstant());
  ASSERT_EQ(TensorShape<0>{}, arg[0].shape);
  ASSERT_EQ(0.123f, *arg[0].data);

  // Passing a vector to a scalar ArgValue
  auto spec2 = OpSpec("Erase").AddArg("shape", vector<float>{0.1f, 0.2f});
  ArgValue<float, 0> arg2("shape", spec2);
  EXPECT_THROW(arg2.Acquire(spec2, ws, nsamples, true), std::runtime_error);
}


TEST(ArgValueTests, Constant_1D) {
  int nsamples = 5;
  std::vector<float> data{0.1f, 0.2f, 0.3f};
  TensorShape<1> expected_shape{3};
  auto spec = OpSpec("Erase").AddArg("shape", data);
  ArgValue<float, 1> arg("shape", spec);
  workspace_t<CPUBackend> ws;
  arg.Acquire(spec, ws, nsamples, true);
  ASSERT_TRUE(arg.IsConstant());
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
    OpSpec("MTTransformAttr")
      .AddArg("M", data);
  workspace_t<CPUBackend> ws;

  auto shape_from_size =
    [](int64_t size) {
      int64_t mat_ndim = sqrt(size);
      assert(mat_ndim > 0);
      DALI_ENFORCE(size == mat_ndim * (mat_ndim + 1),
          make_string("Cannot form an affine transform matrix with ", size, " elements"));
      return TensorShape<2>{mat_ndim, mat_ndim + 1};
    };

  ArgValue<float, 2> err("M", spec);
  EXPECT_THROW(err.Acquire(spec, ws, nsamples, true), std::logic_error);  // can't infer shape

  ArgValue<float, 2> arg1("M", spec);
  arg1.Acquire(spec, ws, nsamples, true, shape_from_size);

  ArgValue<float, 2> arg2("M", spec);
  arg2.Acquire(spec, ws, nsamples, expected_shape);

  for (auto &arg : {arg1, arg2}) {
    ASSERT_TRUE(arg.IsConstant());
    for (int i = 0; i < kNumSamples; i++) {
      ASSERT_EQ(expected_shape, arg[i].shape);
      for (int j = 0; j < expected_shape.num_elements(); j++) {
        ASSERT_EQ(data[j], arg[i].data[j]);
      }
    }
  }
}

}  // namespace dali
