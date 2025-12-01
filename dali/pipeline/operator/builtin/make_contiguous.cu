// Copyright (c) 2017-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/core/nvtx.h"
#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/operator/builtin/make_contiguous.h"
#include "dali/pipeline/workspace/workspace.h"

namespace dali {

void MakeContiguousMixed::RunImpl(Workspace &ws) {
  const auto& input = ws.Input<CPUBackend>(0);
  int sample_dim = input.shape().sample_dim();
  size_t batch_size = input.num_samples();
  DALIDataType type = input.type();
  size_t type_size = input.type_info().size();

  for (int i = 0; i < input.num_samples(); ++i) {
    auto sample = ws.Input<CPUBackend>(0)[i];
    size_t sample_bytes = sample.shape().num_elements() * type_size;
    if (coalesced && sample_bytes > COALESCE_THRESHOLD)
      coalesced = false;
    DALI_ENFORCE(type == sample.type(), "Inconsistent types in "
        "input batch. Cannot copy to contiguous device buffer.");
    DALI_ENFORCE(sample_dim == sample.shape().sample_dim(), "Inconsistent sample dimensions "
        "in input batch. Cannot copy to contiguous device buffer.");
  }
  if (ws.OutputIsType<CPUBackend>(0)) {
    auto &output = ws.Output<CPUBackend>(0);
    DomainTimeRange tr("[DALI][MakeContiguousMixed] H2H non coalesced", DomainTimeRange::kGreen);
    if (pass_through_) {
      AccessOrder out_order = output.order();
      // A call to ShareData may synchronize the orders and we don't want that.
      // TODO(michalz): Find a less hacky solution.
      output.set_order(input.order(), false);
      output.ShareData(input);
      output.set_order(out_order, false);
    } else {
      output.Copy(input);
    }
  } else {
    assert(!IsPassThrough() &&
           "Copy between backends is needed, executor cannot mark this MakeContiguous as "
           "PassThrough node.");
    auto &output = ws.Output<GPUBackend>(0);
    if (coalesced) {
      DomainTimeRange tr("[DALI][MakeContiguousMixed] H2D coalesced", DomainTimeRange::kBlue);
      cpu_output_buff.Copy(input);
      output.Copy(cpu_output_buff, ws.stream());
    } else {
      DomainTimeRange tr("[DALI][MakeContiguousMixed] H2D non coalesced", DomainTimeRange::kGreen);
      output.Copy(input, ws.stream());
    }
    coalesced = true;
  }
}

void MakeContiguousGPU::RunImpl(Workspace &ws) {
  const auto& input = ws.Input<GPUBackend>(0);
  auto& output = ws.Output<GPUBackend>(0);
  DomainTimeRange tr("[DALI][MakeContiguousGPU] D2D", DomainTimeRange::kGreen);
  if (pass_through_) {
    output.ShareData(input);
  } else {
    output.Copy(input);
  }
}


DALI_REGISTER_OPERATOR(MakeContiguous, MakeContiguousMixed, Mixed);
DALI_REGISTER_OPERATOR(MakeContiguous, MakeContiguousGPU, GPU);

}  // namespace dali
