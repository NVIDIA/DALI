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

#include "dali/operators/image/resize/random_resized_crop.h"
#include "dali/pipeline/operator/common.h"
#include "dali/util/ocv.h"

namespace dali {

DALI_SCHEMA(RandomResizedCrop)
  .DocStr("Perform a crop with randomly chosen area and aspect ratio,"
" then resize it to given size. Expects a 3-dimensional input with samples"
" in HWC layout `(height, width, channels)`.")
  .NumInput(1)
  .NumOutput(1)
  .AddArg("size",
      R"code(Size of resized image.)code",
      DALI_INT_VEC)
  .AddParent("RandomCropAttr")
  .AddParent("ResamplingFilterAttr")
  .InputLayout("HWC");

template<>
void RandomResizedCrop<CPUBackend>::BackendInit() {
  InitializeCPU(num_threads_);
}

template<>
void RandomResizedCrop<CPUBackend>::RunImpl(HostWorkspace &ws) {
  auto &input = ws.InputRef<CPUBackend>(0);
  auto &output = ws.OutputRef<CPUBackend>(0);

  DALI_ENFORCE(input.ndim() == 3, "Operator expects 3-dimensional image input.");

  RunResize(output, input);
  output.SetLayout(input.GetLayout());
}

template<>
bool RandomResizedCrop<CPUBackend>::SetupImpl(const HostWorkspace &ws) {
  auto &input = ws.InputRef<CPUBackend>(0);
  auto &input_shape = input.shape();
  DALI_ENFORCE(input_shape.sample_dim() == 3,
      "Expects 3-dimensional image input.");

  int N = input.num_samples();

  resample_params_.resize(N);

  for (int sample_idx = 0; sample_idx < N; i++) {
    auto sample_shape = input_shape.tensor_shape_span(sample_idx);
    int H = sample_shape[height_idx];
    int W = sample_shape[width_idx];
    crops_[sample_idx] = GetCropWindowGenerator(id)({H, W}, "HW");
    resample_params_[sample_idx] = CalcResamplingParams(sample_idx);
  }

  SetupResize(output_shape, input, resample_params_, output_type_);
  return true;
}

DALI_REGISTER_OPERATOR(RandomResizedCrop, RandomResizedCrop<CPUBackend>, CPU);

}  // namespace dali
