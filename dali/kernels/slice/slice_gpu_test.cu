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

#include <gtest/gtest.h>
#include "dali/kernels/slice/slice_kernel_test.h"

namespace dali {
namespace kernels {

using SLICE_GPU_TEST_TYPES = ::testing::Types<
    SliceGPUTestArgs<int, int, 3, 1, 2, SliceArgsGenerator_WholeTensor<3>>,
    SliceGPUTestArgs<int, int, 4, 1, 2, SliceArgsGenerator_HalfAllDims<4>>,
    SliceGPUTestArgs<int, int, 3, 1, 2, SliceArgsGenerator_HalfOneDim<3, 0>>,
    SliceGPUTestArgs<int, int, 3, 1, 2, SliceArgsGenerator_HalfOneDim<3, 1>>,
    SliceGPUTestArgs<int, int, 3, 1, 2, SliceArgsGenerator_HalfOneDim<3, 2>>,
    SliceGPUTestArgs<float, float, 3, 1, 2, SliceArgsGenerator_HalfOneDim<3, 2>>,
    SliceGPUTestArgs<int, float, 3, 1, 2, SliceArgsGenerator_HalfOneDim<3, 2>>,
    SliceGPUTestArgs<float, int, 3, 1, 2, SliceArgsGenerator_HalfOneDim<3, 2>>,
    SliceGPUTestArgs<int, int, 3, 10, 2, SliceArgsGenerator_HalfOneDim<3, 2>>,
    SliceGPUTestArgs<int, int, 10, 1, 2, SliceArgsGenerator_HalfAllDims<10>>,
    SliceGPUTestArgs<unsigned char, unsigned char, 3, 1, 2, SliceArgsGenerator_HalfAllDims<3>>,
    SliceGPUTestArgs<unsigned char, unsigned char, 1, 1, 2, SliceArgsGenerator_HalfAllDims<1>>,
    SliceGPUTestArgs<unsigned char, unsigned char, 2, 1, 1024, SliceArgsGenerator_HalfAllDims<2>>,
    SliceGPUTestArgs<int, int, 2, 1, 3, SliceArgsGenerator_ExtractCenterElement<2>>
>;

TYPED_TEST_SUITE(SliceGPUTest, SLICE_GPU_TEST_TYPES);

TYPED_TEST(SliceGPUTest, All) {
  this->Run();
}

}  // namespace kernels
}  // namespace dali
