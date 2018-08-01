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

#include "dali/test/dali_test_single_op.h"

namespace dali {

typedef struct {
  string opName;
  string paramName;
  float paramVal;
  double epsVal;
} opDescr;

template <typename ImgType>
class ColorTest : public DALISingleOpTest<ImgType> {
 protected:
    void RunTest(const opDescr &descr) {
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
          .AddOutput("input", "cpu"));

      // Launching the same color transformation on CPU (outputIdx 0) and GPU (outputIdx 1)
      this->AddOperatorWithOutput(OpSpec(descr.opName)
                                    .AddArg(descr.paramName, descr.paramVal)
                                    .AddInput("input", "cpu")
                                    .AddOutput("outputCPU", "cpu"));

      this->RunOperator(this->DefaultSchema(descr.opName)
                              .AddArg(descr.paramName, descr.paramVal), descr.epsVal);
    }

    virtual vector<TensorList<CPUBackend>*>
    Reference(const vector<TensorList<CPUBackend>*> &inputs, DeviceWorkspace *ws) {
      return this->CopyToHost(*ws->Output<GPUBackend>(1));
    }

    uint8 GetTestCheckType() const  override {
      return t_checkColorComp + t_checkElements;  // + t_checkAll + t_checkNoAssert;
    }
};

typedef ::testing::Types<RGB> Types;
TYPED_TEST_CASE(ColorTest, Types);

TYPED_TEST(ColorTest, Brightness) {
  this->RunTest({"Brightness", "brightness", 3.f, 1e-4});
}

TYPED_TEST(ColorTest, Contrast) {
  this->RunTest({"Contrast", "contrast", 1.3f, 0.46});
}

TYPED_TEST(ColorTest, Saturation) {
  this->RunTest({"Saturation", "saturation", 3.f, 0.65});
}

TYPED_TEST(ColorTest, Hue) {
  this->RunTest({"Hue", "hue", 31.456f, 0.68});
}

}  // namespace dali
