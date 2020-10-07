// Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
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

#include <gtest/gtest.h>

#include "dali/pipeline/data/tensor.h"
#include "dali/test/dali_operator_test.h"
#include "dali/test/dali_operator_test_utils.h"

namespace dali {

namespace testing {

class ReduceTest : public testing::DaliOperatorTest {
    GraphDescr GenerateOperatorGraph() const override {
        GraphDescr graph("Reduce");
        return graph;
    }
};

template <typename DataType>
void ReduceVerify(TensorListWrapper input, TensorListWrapper output, Arguments args) {
    auto output_d = output.CopyTo<CPUBackend>();

    for (size_t i = 0; i < output_d->ntensor(); ++i) {
        auto out_tensor = output_d->tensor<DataType>(i);
        cout << out_tensor[0] << "    ";
    }
    cout << endl;
}

using input_t = unique_ptr<TensorList<CPUBackend>>;

template <typename DataType>
input_t PrepareInput() {
    input_t input(new TensorList<CPUBackend>());
    input->set_type(TypeTable::GetTypeInfoFromStatic<DataType>());

    int batch_size = 5;
    vector<Index> sample_shape { 4 };
    vector<vector<Index>> shape;

    for ( int sample = 0; sample < batch_size; ++sample) {
        shape.push_back(sample_shape); 
    }

    input->Resize(TensorListShape<>(shape));
    for (int sample = 0; sample < batch_size; ++sample) {
        auto sample_data = input->mutable_tensor<DataType>(sample);
        for (int elem = 0; elem < volume(sample_shape); ++elem) {
            sample_data[elem] = sample;
        }
    }

    return input;
}


TEST_F(ReduceTest, Int16) {
    input_t input = PrepareInput<int16_t>();
    
    TensorListWrapper output;
    this->RunTest(input.get(), output, Arguments {}, ReduceVerify<int16_t>);
}

TEST_F(ReduceTest, Int32) {
    input_t input = PrepareInput<int32_t>();
    
    TensorListWrapper output;
    this->RunTest(input.get(), output, Arguments {}, ReduceVerify<int32_t>);
}

TEST_F(ReduceTest, float) {
    input_t input = PrepareInput<float>();
    
    TensorListWrapper output;
    this->RunTest(input.get(), output, Arguments {}, ReduceVerify<float>);
}

}  // namespace testing
}  // namespace dali