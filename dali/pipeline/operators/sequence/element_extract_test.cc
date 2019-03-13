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

#include <functional>
#include "dali/test/dali_operator_test.h"
#include "dali/pipeline/data/tensor.h"

namespace dali {
namespace testing {

template <typename Backend, typename T>
class ElementExtractTest : public DaliOperatorTest {
 protected:
    GraphDescr GenerateOperatorGraph() const noexcept override {
        return {"ElementExtract"};
    }

 public:
    ElementExtractTest(
        int ntensors = 2,
        int F = 10,
        int H = 720,
        int W = 1280,
        int C = 3,
        std::vector<int> element_map = {0})
        : ntensors_(ntensors)
        , F_(F)
        , H_(H)
        , W_(W)
        , C_(C)
        , element_map_(element_map) {
    }

    std::unique_ptr<TensorList<CPUBackend>>
    GetSequenceData() {
        std::unique_ptr<TensorList<CPUBackend>> data(
            new TensorList<CPUBackend>);
        std::vector<Dims> shape(ntensors_, {F_, H_, W_, C_});
        data->set_type(TypeInfo::Create<T>());
        data->SetLayout(DALITensorLayout::DALI_NFHWC);
        data->Resize(shape);

        const auto frame_size = W_*H_*C_;
        for (int i = 0; i < ntensors_; i++) {
            T *raw_data = static_cast<T*>(
                data->raw_mutable_tensor(i));
            for (int f = 0; f < F_; f++) {
                T *frame_data = &raw_data[f*frame_size];
                T value = f % 256;
                for (int k = 0; k < frame_size; k++)
                    frame_data[k] = value;
            }
        }
        return data;
    }

    void Verify(const TensorListWrapper& input,
                const TensorListWrapper& output,
                const Arguments& args) {
        auto output_tl = output.CopyTo<CPUBackend>();
        ASSERT_NE(nullptr, output_tl);
        auto nouttensors = output_tl->ntensor();
        int element_map_size = element_map_.size();
        EXPECT_EQ(ntensors_ * element_map_size, nouttensors);
        for (int in_idx = 0; in_idx < ntensors_; in_idx++) {
            for (int k = 0; k < element_map_size; k++) {
                auto idx = in_idx * element_map_size + k;
                auto element_idx = element_map_[k];
                const Dims shape = output_tl->tensor_shape(idx);
                const auto *data = output_tl->tensor<T>(idx);
                ASSERT_NE(nullptr, data);
                Dims expected_shape{H_, W_, C_};
                EXPECT_EQ(expected_shape, shape);
                for (int i = 0; i < H_; i++)
                    for (int j = 0; j < W_; j++)
                        EXPECT_EQ(element_idx, data[i*W_+j]);
            }
        }
    }

    void Run(const std::vector<int>& element_map) {
        element_map_ = element_map;
        Arguments args;
        args.emplace("element_map", element_map_);
        args.emplace("device", detail::BackendStringName<Backend>());
        TensorListWrapper tlout;
        auto tlin = GetSequenceData();
        this->RunTest(
            tlin.get(), tlout, args,
            std::bind(&ElementExtractTest::Verify, this,
                std::placeholders::_1,
                std::placeholders::_2,
                std::placeholders::_3));
    }

    int ntensors_, F_, H_, W_, C_;
    std::vector<int> element_map_;
};

template <typename T>
class ElementExtractCPUTest : public ElementExtractTest<CPUBackend, T> {};

typedef ::testing::Types<uint8_t, float> Types;
TYPED_TEST_SUITE(ElementExtractCPUTest, Types);

TYPED_TEST(ElementExtractCPUTest, ExtractFirstElement) {
    this->Run({0});
}

TYPED_TEST(ElementExtractCPUTest, ExtractLastElement) {
    this->Run({this->F_-1});
}

std::vector<int> EvenNumbersTill(int N) {
    std::vector<int> numbers;
    for (int n = 0; n < N; n += 2)
        numbers.push_back(n);
    return numbers;
}

// TODO(janton): Enable multiple output tests once it is supported by DaliOperatorTest
TYPED_TEST(ElementExtractCPUTest, DISABLED_ExtractEvenElement) {
    this->Run(EvenNumbersTill(this->F_));
}

template <typename T>
class ElementExtractGPUTest : public ElementExtractTest<GPUBackend, T> {};

typedef ::testing::Types<uint8_t, float> Types;
TYPED_TEST_SUITE(ElementExtractGPUTest, Types);

TYPED_TEST(ElementExtractGPUTest, ExtractFirstElement) {
    this->Run({0});
}

TYPED_TEST(ElementExtractGPUTest, ExtractLastElement) {
    this->Run({this->F_-1});
}

// TODO(janton): Enable multiple output tests once it is supported by DaliOperatorTest
TYPED_TEST(ElementExtractGPUTest, DISABLED_ExtractEvenElement) {
    this->Run(EvenNumbersTill(this->F_));
}


}  // namespace testing
}  // namespace dali
