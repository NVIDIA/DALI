// Copyright (c) 2018-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_TEST_DALI_TEST_UTILS_H_
#define DALI_TEST_DALI_TEST_UTILS_H_

#include <string>
#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/data/tensor.h"
#include "dali/pipeline/workspace/device_workspace.h"


template <typename Enum>
std::string EnumToString(Enum value) {
  return std::to_string(static_cast<std::underlying_type_t<Enum>>(value));
}

namespace dali {
namespace test {

std::string CurrentExecutableDir();

/**
 * @brief Produces a batch of ND random data with variable shapes
 *
 * @param data output data
 * @param N number of samples
 * @param ndim number of dimensions
 * @param min_extent minimum extent in the shape
 * @param max_extent maximum extent in the shape
 */
void MakeRandomBatch(TensorList<CPUBackend> &data, int N, int ndim = 3,
                     int64_t min_extent = 10, int64_t max_extent = 20);

/**
 * @brief Compares the outputs of the pipeline for the i-th iteration, with the
 *        appropriate sample in the dataset, assuming wrap-around behavior.
 *
 * @param ws workspace. Should contain single GPU output
 * @param batch_size batch size
 * @param i index of the iteration in the pipeline
 * @param data dataset used to drive the pipeline, the output of the pipeline should
 *             match those samples explicitly, and should wrap-around when reaching
 *             the end.
 */
void CheckResults(DeviceWorkspace ws, int batch_size, int i,
                  TensorList<CPUBackend> &data);

}  // namespace test
}  // namespace dali

#endif  // DALI_TEST_DALI_TEST_UTILS_H_
