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


#include <vector>
#include <random>

#include "dali/pipeline/operators/resize/random_resized_crop.h"
#include "dali/pipeline/operators/common.h"
#include "dali/util/ocv.h"

namespace dali {

DALI_SCHEMA(RandomResizedCrop)
  .DocStr("Perform a crop with randomly chosen area and aspect ratio,"
      " then resize it to given size.")
  .NumInput(1)
  .NumOutput(1)
  .AllowMultipleInputSets()
  .AddArg("size",
      R"code(Size of resized image.)code",
      DALI_INT_VEC)
  .AddParent("RandomCropAttr")
  .AddParent("ResamplingFilterAttr")
  .EnforceInputLayout(DALI_NHWC);

template<>
void RandomResizedCrop<CPUBackend>::BackendInit() {
  Initialize(num_threads_);
  out_shape_.resize(num_threads_);
}

template<>
void RandomResizedCrop<CPUBackend>::RunImpl(SampleWorkspace * ws, const int idx) {
  auto &input = ws->Input<CPUBackend>(idx);
  DALI_ENFORCE(input.ndim() == 3, "Operator expects 3-dimensional image input.");
  DALI_ENFORCE(IsType<uint8>(input.type()), "Expected input data as uint8.");

  const int W = input.shape()[1];
  const int C = input.shape()[2];

  const int newH = size_[0];
  const int newW = size_[1];

  auto &output = ws->Output<CPUBackend>(idx);

  RunCPU(output, input, ws->thread_idx());
}

template<>
void RandomResizedCrop<CPUBackend>::SetupSharedSampleParams(SampleWorkspace *ws) {
  auto &input = ws->Input<CPUBackend>(0);
  vector<Index> input_shape = input.shape();
  DALI_ENFORCE(input_shape.size() == 3,
      "Expects 3-dimensional image input.");

  int H = input_shape[0];
  int W = input_shape[1];
  int id = ws->data_idx();

  crops_[id] = GetCropWindowGenerator(id)(H, W);
  resample_params_[ws->thread_idx()] = CalcResamplingParams(id);
}

DALI_REGISTER_OPERATOR(RandomResizedCrop, RandomResizedCrop<CPUBackend>, CPU);

}  // namespace dali
