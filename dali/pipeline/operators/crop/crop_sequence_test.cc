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

struct CropSequenceTestArgs {
    int ntensors = 2;
    int F = 10;
    int H = 1280;
    int W = 800;
    int C = 3;
    int crop_H = 224;
    int crop_W = 256;
};

template <typename Backend, typename T>
class CropSequenceTest : public DaliOperatorTest {
 protected:
    GraphDescr GenerateOperatorGraph() const noexcept override {
        return {"Crop"};
    }

 public:
    CropSequenceTest(){
    }

    std::unique_ptr<TensorList<CPUBackend>>
    GetSequenceData() {
        std::unique_ptr<TensorList<CPUBackend>> data(
            new TensorList<CPUBackend>);
        std::vector<Dims> shape(args_.ntensors, {args_.F, args_.W, args_.H, args_.C});
        data->set_type(TypeInfo::Create<T>());
        data->SetLayout(DALITensorLayout::DALI_NFHWC);
        data->Resize(shape);

        const auto frame_size = args_.W * args_.H * args_.C;
        for (int i = 0; i < args_.ntensors; i++) {
            T *raw_data = static_cast<T*>(
                data->raw_mutable_tensor(i));
            for (int f = 0; f < args_.F; f++) {
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
        auto input_tl = input.CopyTo<CPUBackend>();
        ASSERT_NE(nullptr, input_tl);
        auto output_tl = output.CopyTo<CPUBackend>();
        ASSERT_NE(nullptr, output_tl);
        int nintensors = input_tl->ntensor();
        int nouttensors = output_tl->ntensor();
        ASSERT_EQ(nintensors, nouttensors);
        for (int idx = 0; idx < nouttensors; idx++) {
            const Dims shape = output_tl->tensor_shape(idx);
            const auto *data = output_tl->tensor<T>(idx);
            ASSERT_EQ(args_.F, shape[0]);
            ASSERT_EQ(args_.crop_H, shape[1]);
            ASSERT_EQ(args_.crop_W, shape[2]);
            ASSERT_EQ(args_.C, shape[3]);

            auto size_frame = args_.crop_H * args_.crop_W * args_.C;
            for (int f = 0; f < args_.F; f++) {
                for (int k = f*size_frame; k < (f+1)*size_frame; k++ ) {
                    ASSERT_EQ(f%256, (int)data[k]);
                }
            }
        }
    }

    void Run(CropSequenceTestArgs test_args) {
        args_ = test_args;

        Arguments args;
        args.emplace("crop", std::vector<float>{1.0f*args_.crop_H, 1.0f*args_.crop_W});
        args.emplace("device", detail::BackendStringName<Backend>());
        TensorListWrapper tlout;
        auto tlin = GetSequenceData();
        this->RunTest(
            tlin.get(), tlout, args,
            std::bind(&CropSequenceTest::Verify, this,
                std::placeholders::_1,
                std::placeholders::_2,
                std::placeholders::_3));
    }

    CropSequenceTestArgs args_;
};

template <typename T>
class CropSequenceGPUTest : public CropSequenceTest<GPUBackend, T> {};

typedef ::testing::Types<uint8_t/*, float */> Types;
TYPED_TEST_CASE(CropSequenceGPUTest, Types);

TYPED_TEST(CropSequenceGPUTest, 1_frame) {
    CropSequenceTestArgs args;
    args.F = 1;
    this->Run(args);
}

TYPED_TEST(CropSequenceGPUTest, 10_frame) {
    CropSequenceTestArgs args;
    args.F = 10;
    this->Run(args);
}

TYPED_TEST(CropSequenceGPUTest, 1_by_1_cropping_window) {
    CropSequenceTestArgs args;
    args.crop_H = 1;
    args.crop_W = 1;
    this->Run(args);
}

TYPED_TEST(CropSequenceGPUTest, 224_by_256_cropping_window) {
    CropSequenceTestArgs args;
    args.crop_H = 224;
    args.crop_W = 256;
    this->Run(args);
}

}  // namespace testing
}  // namespace dali
