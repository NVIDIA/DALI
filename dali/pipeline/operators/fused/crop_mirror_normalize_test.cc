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

namespace dali {

static const char *opName = "CropMirrorNormalize";

template <typename ImgType>
class CropMirrorNormalizePermuteTest : public GenericMatchingTest<ImgType> {
 protected:
  void RunTest(const char *deviceName = "gpu", bool doMirroring = false) {
    const int batch_size = this->Imgs(t_jpegImgType).nImages();
    this->SetBatchSize(batch_size);
    this->SetNumThreads(1);

    TensorList<CPUBackend> data;
    this->MakeJPEGBatch(&data, batch_size);
    this->SetExternalInputs({{"jpegs", &data}});

    string device(deviceName);
    this->setOpType(device == "gpu"? DALI_GPU : DALI_CPU);
    OpSpec spec = OpSpec(opName)
                    .AddArg("device", device)
                    .AddInput("images", device)
                    .AddOutput("cropped1", device)
                    .AddInput("images2", device)
                    .AddOutput("cropped2", device)
                    .AddArg("crop", vector<int>{64, 64})
                    .AddArg("mean", vector<float>(this->c_, 0.))
                    .AddArg("std", vector<float>(this->c_, 1.))
                    .AddArg("image_type", this->img_type_)
                    .AddArg("num_input_sets", 2);

    shared_ptr<dali::Pipeline> pipe = this->GetPipeline();
    // Decode the images
    pipe->AddOperator(
      OpSpec("HostDecoder")
        .AddArg("output_type", this->img_type_)
        .AddInput("jpegs", "cpu")
        .AddOutput("images", "cpu"));

    pipe->AddOperator(
      OpSpec("HostDecoder")
        .AddArg("output_type", this->img_type_)
        .AddInput("jpegs", "cpu")
        .AddOutput("images2", "cpu"));

    if (doMirroring) {
      pipe->AddOperator(
        OpSpec("CoinFlip")
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

const bool doMirroring = true;

typedef ::testing::Types<RGB, BGR, Gray> Types;

TYPED_TEST_CASE(CropMirrorNormalizePermuteTest, Types);

TYPED_TEST(CropMirrorNormalizePermuteTest, MultipleDataGPU) {
  this->RunTest("gpu", !doMirroring);
}

TYPED_TEST(CropMirrorNormalizePermuteTest, MultipleDataGPU_Mirror) {
  this->RunTest("gpu", doMirroring);
}

TYPED_TEST(CropMirrorNormalizePermuteTest, MultipleDataCPU) {
  this->RunTest("cpu", !doMirroring);
}

TYPED_TEST(CropMirrorNormalizePermuteTest, MultipleDataCPU_Mirror) {
  this->RunTest("cpu", doMirroring);
}


const bool addImageType = true;
static const char *meanValues[] = {"0.", "0., 0., 0."};
static const char *stdValues[]  = {"1.", "1., 1., 1."};

template <typename ImgType>
class CropMirrorNormalizePermuteTestMatch : public NormalizePermuteMatch<ImgType> {
 protected:
  bool MirroringNeeded() const override   { return m_bMirroring; }

  bool m_bMirroring = false;
};


TYPED_TEST_CASE(CropMirrorNormalizePermuteTestMatch, Types);

#define CROP_MIRROR_TEST(testName, mirroring, ...)                          \
        CONFORMITY_NORMALIZE_TEST_DEF(CropMirrorNormalizePermuteTestMatch,  \
               testName, this->m_bMirroring = mirroring, __VA_ARGS__)

CROP_MIRROR_TEST(CropNumber,           !doMirroring, {{"crop",          "224", DALI_INT32}})
CROP_MIRROR_TEST(CropVector,           !doMirroring, {{"crop",     "224, 256", DALI_INT_VEC}})
CROP_MIRROR_TEST(Layout_DALI_NCHW,     !doMirroring, {{"crop",          "224", DALI_INT32},   \
                                                      {"output_layout",   "0", DALI_INT32}})
CROP_MIRROR_TEST(Layout_DALI_NHWC,     !doMirroring, {{"crop",          "224", DALI_INT32},   \
                                                      {"output_layout",   "1", DALI_INT32}})
CROP_MIRROR_TEST(Layout_DALI_SAME,     !doMirroring, {{"crop",          "224", DALI_INT32},   \
                                                      {"output_layout",   "2", DALI_INT32}})
CROP_MIRROR_TEST(Output_DALI_NO_TYPE,  !doMirroring, {{"crop",          "224", DALI_INT32},   \
                                                      {"output_dtype",   "-1", DALI_INT32}})
CROP_MIRROR_TEST(Output_DALI_UINT8,    !doMirroring, {{"crop",          "224", DALI_INT32},   \
                                                      {"output_dtype",    "0", DALI_INT32}})
CROP_MIRROR_TEST(Output_DALI_INT16,    !doMirroring, {{"crop",          "224", DALI_INT32},   \
                                                      {"output_dtype",    "1", DALI_INT32}})
CROP_MIRROR_TEST(Output_DALI_INT32,    !doMirroring, {{"crop",          "224", DALI_INT32},   \
                                                     {"output_dtype",    "2", DALI_INT32}})
CROP_MIRROR_TEST(Output_DALI_INT64,    !doMirroring, {{"crop",          "224", DALI_INT32},   \
                                                      {"output_dtype",    "3", DALI_INT32}})
CROP_MIRROR_TEST(Output_DALI_FLOAT16,  !doMirroring, {{"crop",          "224", DALI_INT32},   \
                                                      {"output_dtype",    "4", DALI_INT32}})
CROP_MIRROR_TEST(Output_DALI_FLOAT,    !doMirroring, {{"crop",          "224", DALI_INT32},   \
                                                      {"output_dtype",    "5", DALI_INT32}})

}  // namespace dali


