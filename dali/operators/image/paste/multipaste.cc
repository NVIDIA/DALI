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
In that case, for each output sample, `in_ids` describe which samples
from the input batch should be pasted to the corresponding sample
in the output batch.

If the `in_ids` argument is omitted, the operator accepts multiple inputs.
In that case, the i-th sample from each input batch will be pasted to the
i-th sample of the output batch. All the input batches must have the same
type and device placement.

If the input shapes are uniform and no explicit `output_size` is provided,
the operator assumes the same output shape. Otherwise, the `output_size`
must be specified.

This operator can also change the type of data.)code")
    .NumInput(1, 64)
    .AllowSequences()
    .InputLayout({"HWC", "FHWC"})
    .AddOptionalArg<std::vector<int>>("in_ids", R"code(Indices of the inputs to paste data from.

If specified, the operator accepts exactly one batch as an input.)code",
                                      {}, true)
    .AddOptionalArg<int>("in_anchors", R"code(Absolute coordinates of LU corner
of the source region.

The anchors are represented as 2D tensors where the first dimension corresponds to the
number of pasted regions and the second one is equal to the number of dimensions of the
data, excluding channels.

If neither `in_anchors` nor `in_anchors_rel` are provided, all anchors are zero.)code",
                         nullptr, true, true)
    .AddOptionalArg<float>("in_anchors_rel", R"code(Relative coordinates of LU corner
of the source region.

The argument works like `in_anchors`, but the values should be floats in `[0, 1]` range,
describing the anchor placement relative to the input sample shape.)code",
                           nullptr, true, true)
    .AddOptionalArg<int>("shapes", R"code(Shape of the paste regions.

The shapes are represented as 2D tensors where the first dimension corresponds to the
number of pasted regions and the second one is equal to the number of dimensions of the
data, excluding channels.

If neither `shapes` nor `shapes_rel` are provided, the shape is calculated so that
the region goes from the region anchor in the input image until the end
of the input image.)code",
                         nullptr, true, true)
    .AddOptionalArg<float>("shapes_rel", R"code(Relative shape of the paste regions.

Works like `shape` argument, but the values should be floats in `[0, 1]` range,
describing the paste region shape relative to the input shape.)code",
                           nullptr, true, true)
    .AddOptionalArg<int>("out_anchors", R"code(Absolute coordinates of LU corner
of the destination region.

The anchors are represented as 2D tensors where the first dimension corresponds to the
number of pasted regions and the second one is equal to the number of dimensions of the
data, excluding channels.

In neither `out_anchors` nor `out_anchors_rel` are provided, all anchors are zero.)code",
                         nullptr, true, true)
    .AddOptionalArg<float>("out_anchors_rel", R"code(Absolute coordinates of LU corner
of the destination region.

Works like `out_anchors` argument, but the values should be floats in `[0, 1]` range,
describing the LU corner of the pasted region relative to the output shape.)code",
                           nullptr, true, true)
    .AddOptionalArg<std::vector<int>>("output_size",
                                      R"code(A tuple (HW) describing the output shape.

Can be omitted if the operator is run with inputs of uniform shape. In that case,
the same shape is used as the output shape.)code",
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
void MultiPasteCPU::RunTyped(Workspace &ws) {
  auto &output = ws.Output<CPUBackend>(0);

  output.SetLayout(ws.Input<CPUBackend>(0).GetLayout());
  auto out_shape = output.shape();

  auto& tp = ws.GetThreadPool();

  auto batch_size = output.shape().num_samples();

  using Kernel = kernels::PasteCPU<OutputType, InputType>;

  int num_inputs = ws.NumInput();
  std::vector<TensorListView<StorageCPU, const InputType, 3>> in_views;
  in_views.reserve(num_inputs);
  for (int i = 0; i < num_inputs; i++) {
    in_views.push_back(view<const InputType, 3>(ws.Input<CPUBackend>(i)));
  }
  auto out_view = view<OutputType, 3>(output);

  bool hasInIdx = in_idx_.HasExplicitValue();
  for (int i = 0; i < batch_size; i++) {
    auto paste_count = GetPasteCount(ws, i);
    memset(out_view[i].data, 0, out_view[i].num_elements() * sizeof(OutputType));
    auto out_sample_shape = out_shape[i];

    if (!HasIntersections(ws, i)) {
      for (int iter = 0; iter < paste_count; iter++) {
        tp.AddWork(
          [&, i, iter, out_sample_shape](int thread_id) {
            kernels::KernelContext ctx;

            int sample_input = hasInIdx ? 0 : iter;
            int sample_idx = hasInIdx ? in_idx_[i].data[iter] : i;
            auto tvin = in_views[sample_input][sample_idx];
            auto tvout = out_view[i];

            Coords region_shape_view{&region_shapes_data_[i][iter][0], coords_sh_};
            Coords in_anchor_view{&in_anchors_data_[i][iter][0], coords_sh_};
            Coords out_anchor_view{&out_anchors_data_[i][iter][0], coords_sh_};
            kernel_manager_.Run<Kernel>(
                    i, ctx, tvout, tvin,
                    in_anchor_view, region_shape_view, out_anchor_view);
          },
          out_shape.tensor_size(i));
      }
    } else {
      tp.AddWork(
        [&, i, paste_count, out_sample_shape](int thread_id) {
          for (int iter = 0; iter < paste_count; iter++) {

            kernels::KernelContext ctx;

            int sample_input = hasInIdx ? 0 : iter;
            int sample_idx = hasInIdx ? in_idx_[i].data[iter] : i;
            auto tvin = in_views[sample_input][sample_idx];
            auto tvout = out_view[i];

            Coords region_shape_view{&region_shapes_data_[i][iter][0], coords_sh_};
            Coords in_anchor_view{&in_anchors_data_[i][iter][0], coords_sh_};
            Coords out_anchor_view{&out_anchors_data_[i][iter][0], coords_sh_};
            kernel_manager_.Run<Kernel>(
                    i, ctx, tvout, tvin,
                    in_anchor_view, region_shape_view, out_anchor_view);
          }
        },
        paste_count * out_shape.tensor_size(i));
    }
  }
  tp.RunAll();
}

DALI_REGISTER_OPERATOR(MultiPaste, MultiPasteCPU, CPU)

}  // namespace dali
