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

#include <gtest/gtest.h>
#include <vector>
#include "dali/test/tensor_test_utils.h"
#include "dali/test/test_tensors.h"
#include "dali/kernels/math/transform_points.h"
#include "dali/kernels/math/transform_points.cuh"

namespace dali {
namespace kernels {

struct TransformPointsTest : ::testing::Test {
    const int in_dim = 3;
    const int out_dim = 2;

    void Init() {
        std::vector<i16vec<in_dim>> in_data;
        mat<out_dim, in_dim> M;
        vec<out_dim> T;
        for (int i = 0, k = 1; i < out_dim; i++)
            for (int j = 0; j < in_dim; j++, k++)
                M(i, j) = k;
        for (int i = 0; i < out_dim; i++)
            T[i] = i - 10;

        TensorListShape<3> shape = {{
            { 480, 640, in_dim },
            { 100, 120, in_dim }
        }};

        TensorListShape<3> out_shape = {{
            { 480, 640, in_dim },
            { 100, 120, in_dim }
        }};

        in_data.reshape(shape);
        out_data.reshape(out_shape);

        UniformRandomFill(in_data);
        UniformRandomFill(out_data);

    }

    void RunGPU() {
    }

    TestTensorList<int8_t> in_data;
    TestTensorList<int16_t> out_data;

    std::mt19937_64 rng{1234};

};

TEST_F(TransformPointsTest, CPU) {
    RunCPU();
}

TEST_F(TransformPointsTest, GPU) {
    RunGPU();
}

}  // namespace kernels
}  // namespace dali
