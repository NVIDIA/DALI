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

#include "dali/test/dali_test_matching.h"

namespace dali {

template <typename ImgType, typename T>
class ElementExtractTest : public GenericMatchingTest<ImgType> {
 protected:
  void PrepareInput(TensorList<CPUBackend>& data,
                    int ntensors = 2,
                    int F = 10,
                    int H = 720,
                    int W = 1280,
                    int C = 3) {
    std::vector<Dims> shape;
    for (int i = 0; i < ntensors; i++) {
        shape.push_back({F, W, H, C});
    }
    data.set_type(TypeInfo::Create<T>());
    data.SetLayout(DALITensorLayout::DALI_NFHWC);
    data.Resize(shape);

    const auto frame_size = W*H*C;
    for (int i = 0; i < ntensors; i++) {
        T *raw_data = static_cast<T*>(
            data.raw_mutable_tensor(i));
        for (int f = 0; f < F; f++) {
            T *frame_data = &raw_data[f*frame_size];
            T value = f % 256;
            for (int k = 0; k < frame_size; k++)
                frame_data[k] = value;
        }
    }
  }

  uint32_t GetTestCheckType() const override {
    return t_checkColorComp;  // + t_checkAll + t_checkNoAssert;
  }

  void RunTestImpl(const opDescr &descr) override {
    const int batch_size = 2;
    this->SetBatchSize(batch_size);
    this->SetNumThreads(1);

    TensorList<CPUBackend> data;
    PrepareInput(data);
    this->SetExternalInputs({{"input", &data}});

    // Launching the same transformation on CPU (outputIdx 0) and GPU (outputIdx 1)
    this->AddOperatorWithOutput(descr);
    this->RunOperator(descr);
  }
};

typedef ::testing::Types<RGB, BGR, Gray, YCbCr> Types;

template <typename ImgType>
class ElementExtractTestFloat : public ElementExtractTest<ImgType, float> {};

TYPED_TEST_CASE(ElementExtractTestFloat, Types);
TYPED_TEST(ElementExtractTestFloat, Test1) {
    this->RunTest({"ElementExtract", {"element_map", "1,2,3", DALI_INT_VEC}, 0.0});
}

template <typename ImgType>
class ElementExtractTestUint8 : public ElementExtractTest<ImgType, uint8_t> {};

TYPED_TEST_CASE(ElementExtractTestUint8, Types);
TYPED_TEST(ElementExtractTestUint8, Test1) {
    this->RunTest({"ElementExtract", {"element_map", "1,2,3", DALI_INT_VEC}, 0.0});
}


}  // namespace dali
