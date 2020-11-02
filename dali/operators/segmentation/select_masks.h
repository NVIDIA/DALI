// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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


#ifndef DALI_OPERATORS_SEGMENTATION_SELECT_MASKS_H_
#define DALI_OPERATORS_SEGMENTATION_SELECT_MASKS_H_

#include <vector>
#include <unordered_map>
#include "dali/core/common.h"
#include "dali/core/span.h"
#include "dali/core/tensor_shape.h"
#include "dali/pipeline/operator/operator.h"

namespace dali {


class SelectMasksCPU : public Operator<CPUBackend> {
 public:
  explicit SelectMasksCPU(const OpSpec &spec)
      : Operator<CPUBackend>(spec), reindex_masks_(spec.GetArgument<bool>("reindex_masks")) {}

  ~SelectMasksCPU() override = default;
  DISABLE_COPY_MOVE_ASSIGN(SelectMasksCPU);

 protected:
  bool CanInferOutputs() const override {
    return true;
  }

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const workspace_t<CPUBackend> &ws) override;
  void RunImpl(workspace_t<CPUBackend> &ws) override;

 private:
  template <typename T>
  void RunImplTyped(workspace_t<CPUBackend> &ws);

  struct PolygonDesc {
    int new_mask_id = -1;
    int start_vertex = -1;
    int end_vertex = -1;
  };

  struct SampleDesc {
    span<const int> selected_masks;
    std::unordered_map<int, PolygonDesc> polygons;
    void clear() {
      selected_masks = {};
      polygons.clear();
    }
  };
  std::vector<SampleDesc> samples_;

  bool reindex_masks_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_SEGMENTATION_SELECT_MASKS_H_
