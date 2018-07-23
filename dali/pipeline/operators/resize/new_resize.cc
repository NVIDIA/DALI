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


#include "dali/pipeline/operators/resize/new_resize.h"

namespace dali {

template <>
void NewResize<CPUBackend>::RunImpl(SampleWorkspace *ws, const int idx) {
  const auto &input = ws->Input<CPUBackend>(idx);
  const auto &output = ws->Output<CPUBackend>(idx);

  const auto &input_shape = input.shape();
  DALISize out_size, input_size;
  SetSize(&input_size, input_shape, 0, &out_size);

  const int C = input_shape[2];

  ResizeGridParam resizeParam[N_GRID_PARAMS] = {};
  ResizeMappingTable resizeTbl;
  PrepareCropAndResize(&input_size, &out_size, idx, C, resizeParam, &resizeTbl);

  const int H0 = input_size.height;
  const int W0 = input_size.width;
  const int H1 = out_size.height;
  const int W1 = out_size.width;
  MirroringInfo mirrorInfo;
  MirrorNeeded(&mirrorInfo);

  DataDependentSetupCPU(input, output, "NewResize", NULL, NULL, NULL, &out_size);
  const auto pResizeMapping = RESIZE_MAPPING_CPU(resizeTbl.resizeMappingCPU);
  const auto pMapping = RESIZE_MAPPING_CPU(resizeTbl.resizeMappingSimpleCPU);
  const auto pPixMapping = PIX_MAPPING_CPU(resizeTbl.pixMappingCPU);

  ResizeFunc(W0, H0, input.template data<uint8>(),
             W1, H1, static_cast<uint8 *>(output->raw_mutable_data()), C,
             resizeParam, &mirrorInfo, 0, 0, 1, 0, 1,
             pMapping, pResizeMapping, pPixMapping);
}

template <>
void NewResize<CPUBackend>::SetupSharedSampleParams(SampleWorkspace *ws) {
  per_sample_meta_[ws->thread_idx()] = GetTransfomMeta(ws, spec_);
}


}  // namespace dali

