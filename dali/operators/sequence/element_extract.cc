// Copyright (c) 2019-2021, NVIDIA CORPORATION. All rights reserved.
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

#include "dali/operators/sequence/element_extract.h"
#include "dali/core/error_handling.h"

namespace dali {

DALI_SCHEMA(ElementExtract)
    .DocStr(R"code(Extracts one or more elements from input sequence.

The outputs are slices in the first (outermost) dimension of the input.
There are as many outputs as the elements provided in the ``element_map``.

For example, for ``element_map = [2, 0, 3]`` there will be three outputs, containing
2nd, 0th and 3rd element of the input sequences respectively.

The input layout, if provided, must begin with ``F`` dimension. The outputs will have one less
dimension than the input, that is for ``FHWC`` inputs, the outputs will be ``HWC`` elements.
)code")
    .NumInput(1)
    .NumOutput(1)
    .SequenceOperator()
    .AddArg("element_map",
        R"code(Indices of the elements to extract.)code",
        DALI_INT_VEC)
    .AdditionalOutputsFn(
        [](const OpSpec& spec) {
            auto element_map = spec.GetRepeatedArgument<int>("element_map");
            DALI_ENFORCE(element_map.size() >= 1);
            auto additional_outputs = element_map.size() - 1;
            return additional_outputs;
        });

template <>
void ElementExtract<CPUBackend>::RunImpl(HostWorkspace &ws) {
  auto &input = ws.InputRef<CPUBackend>(0);
  auto element_layout = VideoLayoutInfo::GetFrameLayout(input.GetLayout());
  int elements_per_sample = element_map_.size();
  auto data_type = input.type();
  auto &tp = ws.GetThreadPool();
  for (int k = 0; k < elements_per_sample; k++) {
    int element = element_map_[k];
    auto &output = ws.OutputRef<CPUBackend>(k);

    for (unsigned int i = 0; i < input.ntensor(); i++) {
      auto tensor_shape = input.tensor_shape(i);
      auto element_size = volume(tensor_shape.begin() + 1, tensor_shape.end());
      auto input_offset_bytes = element * element_size * data_type.size();
      tp.AddWork(
          [out_ptr = output.raw_mutable_tensor(i),
           in_ptr = static_cast<const uint8_t *>(input.raw_tensor(i)) + input_offset_bytes,
           element_size, data_type](int thread_id) {
            data_type.Copy<CPUBackend, CPUBackend>(out_ptr, in_ptr, element_size, 0);
          },
          element_size);
    }
    output.SetLayout(element_layout);
  }
  tp.RunAll();
}

DALI_REGISTER_OPERATOR(ElementExtract, ElementExtract<CPUBackend>, CPU);

}  // namespace dali
