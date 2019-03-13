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

template <typename Backend_, typename T_,
          Index N_ = 2, Index F_ = 10, Index H_ = 1280, Index W_ = 800, Index C_ = 3,
          Index crop_H_ = 224, Index crop_W_ = 256>
struct CropSequenceTestArgs {
  using Backend = Backend_;
  using T = T_;
  enum { N = N_ };
  enum { F = F_ };
  enum { H = H_ };
  enum { W = W_ };
  enum { C = C_ };
  enum { crop_H = crop_H_ };
  enum { crop_W = crop_W_ };
};

template <typename TestArgs>
class CropSequenceTest : public DaliOperatorTest {
 protected:
    GraphDescr GenerateOperatorGraph() const noexcept override {
        return {"Crop"};
    }

 public:
    std::unique_ptr<TensorList<CPUBackend>>
    GetSequenceData() {
        std::unique_ptr<TensorList<CPUBackend>> data(
            new TensorList<CPUBackend>);
        std::vector<Dims> shape(TestArgs::N, {TestArgs::F, TestArgs::W, TestArgs::H, TestArgs::C});
        data->set_type(TypeInfo::Create<typename TestArgs::T>());
        data->SetLayout(DALITensorLayout::DALI_NFHWC);
        data->Resize(shape);

        const auto frame_size = TestArgs::W * TestArgs::H * TestArgs::C;
        for (int i = 0; i < TestArgs::N; i++) {
            auto *raw_data = static_cast<typename TestArgs::T*>(
                data->raw_mutable_tensor(i));
            for (int f = 0; f < TestArgs::F; f++) {
                auto *frame_data = &raw_data[f*frame_size];
                auto value = f % 256;
                for (int k = 0; k < frame_size; k++)
                    frame_data[k] = value;
            }
        }
        return data;
    }

    void Verify(const TensorListWrapper& input,
                const TensorListWrapper& output,
                const Arguments& args) {
        auto input_tl = input.CopyTo<CPUBackend>();
        ASSERT_NE(nullptr, input_tl);
        auto output_tl = output.CopyTo<CPUBackend>();
        ASSERT_NE(nullptr, output_tl);
        int nintensors = input_tl->ntensor();
        int nouttensors = output_tl->ntensor();
        ASSERT_EQ(nintensors, nouttensors);
        for (int idx = 0; idx < nouttensors; idx++) {
            const Dims shape = output_tl->tensor_shape(idx);
            const auto *data = output_tl->tensor<typename TestArgs::T>(idx);
            ASSERT_EQ(TestArgs::F, shape[0]);
            ASSERT_EQ(TestArgs::crop_H, shape[1]);
            ASSERT_EQ(TestArgs::crop_W, shape[2]);
            ASSERT_EQ(TestArgs::C, shape[3]);

            auto size_frame = TestArgs::crop_H * TestArgs::crop_W * TestArgs::C;
            for (int f = 0; f < TestArgs::F; f++) {
                for (int k = f*size_frame; k < (f+1)*size_frame; k++) {
                    ASSERT_EQ(f%256, (int)data[k]);
                }
            }
        }
    }

    void Run() {
        Arguments args;
        args.emplace("crop", std::vector<float>{1.0f*TestArgs::crop_H, 1.0f*TestArgs::crop_W});
        args.emplace("device", detail::BackendStringName<typename TestArgs::Backend>());
        TensorListWrapper tlout;
        auto tlin = GetSequenceData();
        this->RunTest(
            tlin.get(), tlout, args,
            std::bind(&CropSequenceTest::Verify, this,
                std::placeholders::_1,
                std::placeholders::_2,
                std::placeholders::_3));
    }
};

template <typename Backend>
using ValidCropArgs = ::testing::Types<
    CropSequenceTestArgs<Backend, uint8_t, 2, 10, 1280, 800, 3, 224, 256>,
    CropSequenceTestArgs<Backend, uint8_t, 1, 10, 1280, 800, 3, 224, 256>,
    CropSequenceTestArgs<Backend, uint8_t, 2, 1,  1280, 800, 3, 224, 256>,
    CropSequenceTestArgs<Backend, uint8_t, 2, 10, 1280, 800, 3, 1,   1>,
    CropSequenceTestArgs<Backend, uint8_t, 2, 10, 1280, 800, 3, 1,   256>>;

using GPU_ValidCropArgs = ValidCropArgs<GPUBackend>;
using CPU_ValidCropArgs = ValidCropArgs<CPUBackend>;

template < typename T>
using CropSequenceTest_GPU_Valid = CropSequenceTest<T>;
TYPED_TEST_SUITE(CropSequenceTest_GPU_Valid, GPU_ValidCropArgs);

TYPED_TEST(CropSequenceTest_GPU_Valid, test_valid_crop_gpu) {
    this->Run();
}

template < typename T>
using CropSequenceTest_CPU_Valid = CropSequenceTest<T>;
TYPED_TEST_SUITE(CropSequenceTest_CPU_Valid, CPU_ValidCropArgs);

TYPED_TEST(CropSequenceTest_CPU_Valid, test_valid_crop_cpu) {
    this->Run();
}

template <typename Backend>
using InvalidCropArgs = ::testing::Types<
    CropSequenceTestArgs<Backend, uint8_t, 2, 10, 1, 1, 3, 224, 256>,
    CropSequenceTestArgs<Backend, uint8_t, 2, 10, 1, 1, 3, -1, -1>>;

using GPU_InvalidCropArgs = InvalidCropArgs<GPUBackend>;
using CPU_InvalidCropArgs = InvalidCropArgs<CPUBackend>;

template < typename T>
using CropSequenceTest_GPU_Invalid = CropSequenceTest<T>;
TYPED_TEST_SUITE(CropSequenceTest_GPU_Invalid, GPU_InvalidCropArgs);

TYPED_TEST(CropSequenceTest_GPU_Invalid, invalid_arguments) {
    EXPECT_THROW(
        this->Run(),
        std::runtime_error);
}

template < typename T>
using CropSequenceTest_CPU_Invalid = CropSequenceTest<T>;
TYPED_TEST_SUITE(CropSequenceTest_CPU_Invalid, CPU_InvalidCropArgs);

TYPED_TEST(CropSequenceTest_CPU_Invalid, invalid_arguments) {
    EXPECT_THROW(
        this->Run(),
        std::runtime_error);
}

}  // namespace testing
}  // namespace dali
