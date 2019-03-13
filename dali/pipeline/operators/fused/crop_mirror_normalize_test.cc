// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

#include "dali/test/dali_test_matching.h"
#include "dali/test/dali_test_utils.h"

namespace dali {

static const char *opName = "CropMirrorNormalize";

template <typename ImgType>
class CropMirrorNormalizePermuteTest : public GenericMatchingTest<ImgType> {
 protected:
  void RunTestByDevice(const char *deviceName = "gpu",
                       bool doMirroring = false) {
    const int batch_size = this->jpegs_.nImages();
    this->SetBatchSize(batch_size);
    this->SetNumThreads(1);

    TensorList<CPUBackend> data;
    this->MakeJPEGBatch(&data, batch_size);
    this->SetExternalInputs({{"jpegs", &data}});

    string device(deviceName);
    this->SetOpType(device == "gpu" ? OpType::GPU : OpType::CPU);
    OpSpec spec = OpSpec(opName)
                      .AddArg("device", device)
                      .AddInput("images", device)
                      .AddOutput("cropped1", device)
                      .AddInput("images2", device)
                      .AddOutput("cropped2", device)
                      .AddArg("crop", vector<float>{64, 64})
                      .AddArg("mean", vector<float>(this->c_, 0.))
                      .AddArg("std", vector<float>(this->c_, 1.))
                      .AddArg("image_type", this->img_type_)
                      .AddArg("num_input_sets", 2);

    shared_ptr<dali::Pipeline> pipe = this->GetPipeline();
    // Decode the images
    pipe->AddOperator(OpSpec("HostDecoder")
                          .AddArg("output_type", this->img_type_)
                          .AddInput("jpegs", "cpu")
                          .AddOutput("images", "cpu"));

    pipe->AddOperator(OpSpec("HostDecoder")
                          .AddArg("output_type", this->img_type_)
                          .AddInput("jpegs", "cpu")
                          .AddOutput("images2", "cpu"));

    if (doMirroring) {
      pipe->AddOperator(OpSpec("CoinFlip")
                            .AddArg("device", "support")
                            .AddArg("probability", 0.5f)
                            .AddOutput("mirror", "cpu"));

      spec.AddArgumentInput("mirror", "mirror");
    }

    // CropMirrorNormalizePermute + crop multiple sets of images
    DeviceWorkspace ws;
    this->RunOperator(spec, 1e-4, &ws);
  }
};

const float eps = 50000;
const bool addImageType = true;

static const OpArg vector_params[] = {{"crop", "224, 256", DALI_FLOAT_VEC},
                                      {"mean", "0.", DALI_FLOAT_VEC},
                                      {"std", "1.", DALI_FLOAT_VEC}};
const bool doMirroring = true;

typedef ::testing::Types<RGB, BGR, Gray> Types;

TYPED_TEST_SUITE(CropMirrorNormalizePermuteTest, Types);

TYPED_TEST(CropMirrorNormalizePermuteTest, DISABLED_MultipleDataGPU) {
  this->RunTestByDevice("gpu", !doMirroring);
}

TYPED_TEST(CropMirrorNormalizePermuteTest, DISABLED_MultipleDataGPU_Mirror) {
  this->RunTestByDevice("gpu", doMirroring);
}

TYPED_TEST(CropMirrorNormalizePermuteTest, MultipleDataCPU) {
  this->RunTestByDevice("cpu", !doMirroring);
}

TYPED_TEST(CropMirrorNormalizePermuteTest, MultipleDataCPU_Mirror) {
  this->RunTestByDevice("cpu", doMirroring);
}

TYPED_TEST(CropMirrorNormalizePermuteTest, CropVector) {
  this->RunTest(opName, vector_params,
                sizeof(vector_params) / sizeof(vector_params[0]), addImageType,
                eps);
}

TYPED_TEST(CropMirrorNormalizePermuteTest, Layout_DALI_NCHW) {
  static const OpArg params[] = {{"crop", "224, 224", DALI_FLOAT_VEC},
                                 {"mean", "0.", DALI_FLOAT_VEC},
                                 {"std", "1.", DALI_FLOAT_VEC},
                                 {"output_layout", EnumToString(DALI_NCHW), DALI_INT32}};

  this->RunTest(opName, params, sizeof(params) / sizeof(params[0]),
                addImageType, eps);
}

TYPED_TEST(CropMirrorNormalizePermuteTest, Layout_DALI_NHWC) {
  static const OpArg params[] = {{"crop", "224, 224", DALI_FLOAT_VEC},
                                 {"mean", "0.", DALI_FLOAT_VEC},
                                 {"std", "1.", DALI_FLOAT_VEC},
                                 {"output_layout", EnumToString(DALI_NHWC), DALI_INT32}};

  this->RunTest(opName, params, sizeof(params) / sizeof(params[0]),
                addImageType, eps);
}

TYPED_TEST(CropMirrorNormalizePermuteTest, Layout_DALI_SAME) {
  static const OpArg params[] = {{"crop", "224, 224", DALI_FLOAT_VEC},
                                 {"mean", "0.", DALI_FLOAT_VEC},
                                 {"std", "1.", DALI_FLOAT_VEC},
                                 {"output_layout", EnumToString(DALI_SAME), DALI_INT32}};

  this->RunTest(opName, params, sizeof(params) / sizeof(params[0]),
                addImageType, eps);
}

TYPED_TEST(CropMirrorNormalizePermuteTest, Output_DALI_NO_TYPE) {
  static const OpArg params[] = {{"crop", "224, 224", DALI_FLOAT_VEC},
                                 {"mean", "0.", DALI_FLOAT_VEC},
                                 {"std", "1.", DALI_FLOAT_VEC},
                                 {"output_dtype", "-1", DALI_INT32}};

  this->RunTest(opName, params, sizeof(params) / sizeof(params[0]),
                addImageType, eps);
}

TYPED_TEST(CropMirrorNormalizePermuteTest, Output_DALI_UINT8) {
  static const OpArg params[] = {{"crop", "224, 224", DALI_FLOAT_VEC},
                                 {"mean", "0.", DALI_FLOAT_VEC},
                                 {"std", "1.", DALI_FLOAT_VEC},
                                 {"output_dtype", "0", DALI_INT32}};

  this->RunTest(opName, params, sizeof(params) / sizeof(params[0]),
                addImageType, eps);
}

TYPED_TEST(CropMirrorNormalizePermuteTest, Output_DALI_INT16) {
  static const OpArg params[] = {{"crop", "224, 224", DALI_FLOAT_VEC},
                                 {"mean", "0.", DALI_FLOAT_VEC},
                                 {"std", "1.", DALI_FLOAT_VEC},
                                 {"output_dtype", "1", DALI_INT32}};

  this->RunTest(opName, params, sizeof(params) / sizeof(params[0]),
                addImageType, eps);
}

TYPED_TEST(CropMirrorNormalizePermuteTest, Output_DALI_INT32) {
  static const OpArg params[] = {{"crop", "224, 224", DALI_FLOAT_VEC},
                                 {"mean", "0.", DALI_FLOAT_VEC},
                                 {"std", "1.", DALI_FLOAT_VEC},
                                 {"output_dtype", "2", DALI_INT32}};

  this->RunTest(opName, params, sizeof(params) / sizeof(params[0]),
                addImageType, eps);
}

TYPED_TEST(CropMirrorNormalizePermuteTest, Output_DALI_INT64) {
  static const OpArg params[] = {{"crop", "224, 224", DALI_FLOAT_VEC},
                                 {"mean", "0.", DALI_FLOAT_VEC},
                                 {"std", "1.", DALI_FLOAT_VEC},
                                 {"output_dtype", "3", DALI_INT32}};

  this->RunTest(opName, params, sizeof(params) / sizeof(params[0]),
                addImageType, eps);
}

TYPED_TEST(CropMirrorNormalizePermuteTest, Output_DALI_FLOAT16) {
  static const OpArg params[] = {{"crop", "224, 224", DALI_FLOAT_VEC},
                                 {"mean", "0.", DALI_FLOAT_VEC},
                                 {"std", "1.", DALI_FLOAT_VEC},
                                 {"output_dtype", "4", DALI_INT32}};

  this->RunTest(opName, params, sizeof(params) / sizeof(params[0]),
                addImageType, eps);
}

TYPED_TEST(CropMirrorNormalizePermuteTest, Output_DALI_FLOAT) {
  static const OpArg params[] = {{"crop", "224, 224", DALI_FLOAT_VEC},
                                 {"mean", "0.", DALI_FLOAT_VEC},
                                 {"std", "1.", DALI_FLOAT_VEC},
                                 {"output_dtype", "5", DALI_INT32}};

  this->RunTest(opName, params, sizeof(params) / sizeof(params[0]),
                addImageType, eps);
}

}  // namespace dali
