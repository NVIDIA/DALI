// Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

#include "dali/pipeline/operators/reader/parser/sequence_parser.h"

namespace dali {

void SequenceParser::Parse(const TensorSequence& data, SampleWorkspace* ws) {
  auto* sequence = ws->Output<CPUBackend>(0);
  auto* metadata = ws->Output<CPUBackend>(1);

  metadata->Resize({static_cast<Index>(data.tensors.size() + 1)});
  auto* metadata_ptr = metadata->mutable_data<Index>();

  metadata_ptr[0] = data.tensors.size();

  Index total_size = 0;

  for (const auto& t : data.tensors) {
    total_size += t.size();
  }

  sequence->Resize({total_size});
  auto* sequence_ptr = sequence->mutable_data<uint8_t>();

  for (size_t i = 0; i < data.tensors.size(); i++) {
    std::memcpy(sequence_ptr, data.tensors[i].raw_data(), data.tensors[i].size());
    sequence_ptr += data.tensors[i].size();
    metadata_ptr[i + 1] = data.tensors[i].size();
  }
}

}  // namespace dali
