// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "dali/operators/reader/fits_reader_gpu_op.h"

#include <string>
#include <vector>

namespace dali {

void FitsReaderGPU::RunImpl(Workspace &ws) {
  int num_outputs = ws.NumOutput();
  int batch_size = GetCurrBatchSize();

  for (int output_idx = 0; output_idx < num_outputs; output_idx++) {
    auto &output = ws.Output<GPUBackend>(output_idx);
    for (int sample_id = 0; sample_id < batch_size; ++sample_id) {
      MemCopy(ouput.raw_mutable_tensor(sample_id), sample.data[output_idx].raw_data(),
              sample.data[output_idx].nbytes(), ws.stream());
    }
  }
}

DALI_REGISTER_OPERATOR(experimental__readers__Fits, FitsReaderGPU, GPU);

}  // namespace dali
