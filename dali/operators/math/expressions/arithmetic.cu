// Copyright (c) 2019-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/operators/math/expressions/arithmetic.h"
#include <vector>
#include "dali/kernels/type_tag.h"

namespace dali {
namespace expr {

template <>
void ArithmeticGenericOp<GPUBackend>::RunImpl(Workspace &ws) {
  PrepareSamplesPerTask<GPUBackend>(samples_per_task_, exec_order_, ws, constant_storage_, spec_);
  ws.Output<GPUBackend>(0).SetLayout(result_layout_);
  std::tie(tile_cover_, tile_range_) = GetTiledCover(result_shape_, kTileSize, kTaskSize);
  if (tile_range_.size() == 0) {
    assert(result_shape_.num_elements() == 0);
    return;  // nothing to do
  }
  assert(tile_range_.size() == 1 && "Expected to cover whole GPU execution by 1 task");
  auto tiles = make_cspan(tile_cover_);
  for (size_t i = 0; i < exec_order_.size(); i++) {
    // call impl for whole batch
    exec_order_[i].impl->Execute(exec_order_[i].ctx, make_cspan(samples_per_task_[i]), tiles);
  }
}

}  // namespace expr

DALI_REGISTER_OPERATOR(ArithmeticGenericOp, expr::ArithmeticGenericOp<GPUBackend>, GPU);

}  // namespace dali
