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

#ifndef DALI_PIPELINE_OPERATORS_CROP_SEQUENCE_CROP_H_
#define DALI_PIPELINE_OPERATORS_CROP_SEQUENCE_CROP_H_

#include <vector>

#include "dali/pipeline/operators/crop/kernel/crop_kernel.h"
#include "dali/pipeline/operators/crop/crop.h"

namespace dali {

class SequenceCrop : public Crop<CPUBackend> {
 public:
  explicit SequenceCrop(const OpSpec &spec) : Crop<CPUBackend>(spec) {}

 protected:
  void RunImpl(Workspace<CPUBackend> *ws, const int idx) override;
  // void SetupSharedSampleParams(Workspace<CPUBackend> *ws) override;
  /**
   * @brief Verify that all input frames have the same sizes. Sequences can be arbitary.
   *        Called in SetupSharedSampleParams.
   *
   * @param ws
   * @return const vector<Index> representing shape of frame (TODO(klecki) compability with regular
   * Crop)
   */
  const std::vector<Index> CheckShapes(const SampleWorkspace *ws) override;
  // using Crop<CPUBackend>::CheckParam;
  using Crop<CPUBackend>::per_sample_crop_;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_CROP_SEQUENCE_CROP_H_
