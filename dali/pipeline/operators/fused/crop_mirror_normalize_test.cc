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

#include "dali/test/dali_test_resize.h"

namespace dali {

template <typename ImgType>
class CropMirrorNormalizePermuteTest : public GenericResizeTest<ImgType> {
 protected:
  virtual vector<TensorList<CPUBackend>*>
  Reference(const vector<TensorList<CPUBackend>*> &inputs, DeviceWorkspace *ws) {
    return this->CopyToHost(*ws->Output<GPUBackend>(1));
  }

  void RunTest() {
    const int batch_size = this->jpegs_.nImages();
    this->SetBatchSize(batch_size);
    this->SetNumThreads(1);

    TensorList<CPUBackend> data;
    this->MakeJPEGBatch(&data, batch_size);
    this->SetExternalInputs({{"jpegs", &data}});

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

    // CropMirrorNormalizePermute + crop multiple sets of images
    DeviceWorkspace ws;
    this->RunOperator(OpSpec("CropMirrorNormalize")
                        .AddArg("device", "gpu")
                        .AddInput("images", "gpu")
                        .AddOutput("cropped1", "gpu")
                        .AddInput("images2", "gpu")
                        .AddOutput("cropped2", "gpu")
                        .AddArg("crop", vector<int>{64, 64})
                        .AddArg("mean", vector<float>(this->c_, 0.))
                        .AddArg("std", vector<float>(this->c_, 1.))
                        .AddArg("image_type", this->img_type_)
                        .AddArg("num_input_sets", 2), 1e-4, &ws);

#if DALI_DEBUG
    WriteCHWBatch<float>(*ws.Output<GPUBackend>(0), 0., 1, "img0");
    WriteCHWBatch<float>(*ws.Output<GPUBackend>(1), 0., 1, "img1");
#endif
  }
};

typedef ::testing::Types<RGB, BGR, Gray> Types;
TYPED_TEST_CASE(CropMirrorNormalizePermuteTest, Types);

TYPED_TEST(CropMirrorNormalizePermuteTest, MultipleData) {
  this->RunTest();
}

}  // namespace dali


