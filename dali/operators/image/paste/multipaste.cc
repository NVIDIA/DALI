// Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/operators/image/paste/multipaste.h"
#include "dali/kernels/imgproc/paste/paste.h"
#include "dali/core/tensor_view.h"

namespace dali {

DALI_SCHEMA(MultiPaste)
    .DocStr(R"code(Performs multiple pastes from image batch to each of the outputs.

If the `in_ids` is specified, the operator expects exactly one input batch.
In that case, for each output sample, `in_ids` describes which samples
from the input batch should be pasted to the corresponding sample
in the output batch.

If the `in_ids` argument is omitted, the operator accepts multiple inputs.
In that case, the i-th sample from each input batch will be pasted to the
i-th sample of the output batch. All the input batches must have the same
type and device placement.

If the input shapes are uniform and no explicit `output_size` is provided,
the operator assumes the same output shape (the output canvas size).
Otherwise, the `output_size` must be specified.

This operator can also change the type of data.)code")
    .NumInput(1, 64)
    .AllowSequences()
    .InputLayout({"HWC", "FHWC"})
    .AddOptionalArg<std::vector<int>>("in_ids", R"code(Indices of the inputs to paste data from.

If specified, the operator accepts exactly one batch as an input.)code",
                                      {}, true)
    .AddOptionalArg<int>("in_anchors", R"code(Absolute coordinates of the top-left corner
of the source region.

The anchors are represented as 2D tensors where the first dimension is equal to the
number of pasted regions and the second one is 2 (for the H and W extents).

If neither `in_anchors` nor `in_anchors_rel` are provided, all anchors are zero.)code",
                         nullptr, true, true)
    .AddOptionalArg<float>("in_anchors_rel", R"code(Relative coordinates of the top-left corner
of the source region.

The argument works like `in_anchors`, but the values should be floats in `[0, 1]` range,
describing the anchor placement relative to the input sample shape.)code",
                           nullptr, true, true)
    .AddOptionalArg<int>("shapes", R"code(Shape of the paste regions.

The shapes are represented as 2D tensors where the first dimension is equal to the
number of pasted regions and the second one is 2 (for the H and W extents).

If neither `shapes` nor `shapes_rel` are provided, the shape is calculated so that
the region spans from the input anchor until the end of the input image.)code",
                         nullptr, true, true)
    .AddOptionalArg<float>("shapes_rel", R"code(Relative shape of the paste regions.

Works like `shape` argument, but the values should be floats in `[0, 1]` range,
describing the paste region shape relative to the input shape.)code",
                           nullptr, true, true)
    .AddOptionalArg<int>("out_anchors", R"code(Absolute coordinates of the top-left corner
of the pasted region in the output canvas.

The anchors are represented as 2D tensors where the first dimension is equal to the
number of pasted regions and the second one is 2 (for the H and W extents).

If neither `out_anchors` nor `out_anchors_rel` are provided, all anchors are zero,
making all the pasted regions start at the top-left corner of the output canvas.)code",
                         nullptr, true, true)
    .AddOptionalArg<float>("out_anchors_rel", R"code(Relative coordinates of the top-left corner
of the pasted region in the output canvas.

Works like `out_anchors` argument, but the values should be floats in `[0, 1]` range,
describing the top-left corner of the pasted region relative to the output canvas size.)code",
                           nullptr, true, true)
    .AddOptionalArg<std::vector<int>>(
        "output_size",
        R"code(A tuple (H, W) describing the output shape
(i.e. the size of the canvas for the output pastes).

Can be omitted if the operator is run with inputs of uniform shape. In that case,
the same shape is used as the canvas size.)code",
        {}, true)
    .AddOptionalTypeArg("dtype", R"code(Output data type. If not set, the input type is used.)code")
    .NumOutput(1);

template <typename OutputType, typename InputType>
void MultiPasteCPU::SetupTyped(const Workspace & /*ws*/,
                               const TensorListShape<> & /*out_shape*/) {
  using Kernel = kernels::PasteCPU<OutputType, InputType>;
  kernel_manager_.Initialize<Kernel>();
}

template <typename OutputType, typename InputType>
void MultiPasteCPU::CopyPatch(TensorListView<StorageCPU, OutputType, 3> &out_view,
                              std::vector<TensorListView<StorageCPU, const InputType, 3>> &in_views,
                              int out_sample_idx, int paste_idx) {
  using Kernel = kernels::PasteCPU<OutputType, InputType>;
  kernels::KernelContext ctx;

  bool hasInIdx = in_idx_.HasExplicitValue();
  int input_idx = hasInIdx ? 0 : paste_idx;
  int in_sample_idx = hasInIdx ? in_idx_[out_sample_idx].data[paste_idx] : out_sample_idx;
  auto tvin = in_views[input_idx][in_sample_idx];
  auto tvout = out_view[out_sample_idx];

  Coords region_shape_view{&region_shapes_data_[out_sample_idx][paste_idx][0], coords_sh_};
  Coords in_anchor_view{&in_anchors_data_[out_sample_idx][paste_idx][0], coords_sh_};
  Coords out_anchor_view{&out_anchors_data_[out_sample_idx][paste_idx][0], coords_sh_};
  kernel_manager_.Run<Kernel>(out_sample_idx, ctx, tvout, tvin, in_anchor_view, region_shape_view,
                              out_anchor_view);
}

template <typename OutputType, typename InputType>
void MultiPasteCPU::RunTyped(Workspace &ws) {
  auto &output = ws.Output<CPUBackend>(0);

  output.SetLayout(ws.Input<CPUBackend>(0).GetLayout());
  auto out_shape = output.shape();

  auto& tp = ws.GetThreadPool();

  auto batch_size = output.shape().num_samples();

  int num_inputs = ws.NumInput();
  std::vector<TensorListView<StorageCPU, const InputType, 3>> in_views;
  in_views.reserve(num_inputs);
  for (int i = 0; i < num_inputs; i++) {
    in_views.push_back(view<const InputType, 3>(ws.Input<CPUBackend>(i)));
  }
  auto out_view = view<OutputType, 3>(output);

  for (int i = 0; i < batch_size; i++) {
    auto paste_count = GetPasteCount(ws, i);
    memset(out_view[i].data, 0, out_view[i].num_elements() * sizeof(OutputType));

    if (!HasIntersections(ws, i)) {
      for (int iter = 0; iter < paste_count; iter++) {
        tp.AddWork(
          [&, i, iter](int thread_id) {
            CopyPatch(out_view, in_views, i, iter);
          },
          out_shape.tensor_size(i));
      }
    } else {
      tp.AddWork(
        [&, i, paste_count](int thread_id) {
          for (int iter = 0; iter < paste_count; iter++) {
            CopyPatch(out_view, in_views, i, iter);
          }
        },
        paste_count * out_shape.tensor_size(i));
    }
  }
  tp.RunAll();
}

DALI_REGISTER_OPERATOR(MultiPaste, MultiPasteCPU, CPU)

}  // namespace dali
