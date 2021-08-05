// Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
.DocStr(R"code(Performs multiple pastes from image batch to each of outputs

This operator can also change the type of data.)code")
.NumInput(1)
.InputDox(0, "images", "3D TensorList", R"code(Batch of input images.

Assumes HWC layout.)code")
.AddArg("in_ids", R"code(Indices of the inputs to paste data from.)code",
        DALI_INT_VEC, true)
.AddOptionalArg<int>("in_anchors", R"code(Absolute coordinates of LU corner
of the source region.

The anchors are represented as 2D tensors where the first dimension corresponds to the
elements of ``in_ids`` and the second one is equal to the number of dimensions of the
data, excluding channels.

If not provided, all anchors are zero.)code", nullptr, true)
.AddOptionalArg<int>("shapes", R"code(Shape of the paste regions.

The shapes are represented as 2D tensors where the first dimension corresponds to the
elements of ``in_ids`` and the second one is equal to the number of dimensions of the
data, excluding channels.

If not provided, the shape is calculated so that the region goes from the region anchor
 in the input image until the end of the input image.)code", nullptr, true)
.AddOptionalArg<int>("out_anchors", R"code(Absolute coordinates of LU corner
of the destination region.

The anchors are represented as 2D tensors where the first dimension corresponds to the
elements of ``in_ids`` and the second one is equal to the number of dimensions of the
data, excluding channels.

If not provided, all anchors are zero.)code", nullptr, true)
.AddArg("output_size",
R"code(Shape of the output.)code", DALI_INT_VEC, true)
.AddOptionalArg("dtype",
R"code(Output data type. If not set, the input type is used.)code", DALI_NO_TYPE)
.NumOutput(1);

template <typename OutputType, typename InputType>
void MultiPasteCPU::SetupTyped(const workspace_t<CPUBackend> & /*ws*/,
                               const TensorListShape<> & /*out_shape*/) {
  using Kernel = kernels::PasteCPU<OutputType, InputType>;
  kernel_manager_.Initialize<Kernel>();
}


template <typename OutputType, typename InputType>
void MultiPasteCPU::RunTyped(workspace_t<CPUBackend> &ws) {
  const auto &images = ws.template InputRef<CPUBackend>(0);
  auto &output = ws.template OutputRef<CPUBackend>(0);

  output.SetLayout(images.GetLayout());
  auto out_shape = output.shape();

  auto& tp = ws.GetThreadPool();

  auto batch_size = output.shape().num_samples();

  using Kernel = kernels::PasteCPU<OutputType, InputType>;
  auto in_view = view<const InputType, 3>(images);
  auto out_view = view<OutputType, 3>(output);
  for (int i = 0; i < batch_size; i++) {
    auto paste_count = in_idx_[i].shape[0];
    memset(out_view[i].data, 0, out_view[i].num_elements() * sizeof(OutputType));

    if (no_intersections_[i]) {
      for (int iter = 0; iter < paste_count; iter++) {
        int from_sample = in_idx_[i].data[iter];
        int to_sample = i;

        tp.AddWork(
          [&, i, iter, from_sample, to_sample, in_view, out_view](int thread_id) {
            kernels::KernelContext ctx;
            auto tvin = in_view[from_sample];
            auto tvout = out_view[to_sample];

            auto in_sh_view = GetInputShape(from_sample);
            auto in_anchor_view = GetInAnchors(i, iter);
            auto out_anchor_view = GetOutAnchors(i, iter);
            auto region_shape = GetShape(i, iter, in_sh_view, in_anchor_view);
            Coords region_shape_view{region_shape.data(), coords_sh_};
            kernel_manager_.Run<Kernel>(
                    thread_id, to_sample, ctx, tvout, tvin,
                    in_anchor_view, region_shape_view, out_anchor_view);
          },
          out_shape.tensor_size(to_sample));
      }
    } else {
      tp.AddWork(
        [&, i, paste_count, in_view, out_view](int thread_id) {
          for (int iter = 0; iter < paste_count; iter++) {
            int from_sample = in_idx_[i].data[iter];
            int to_sample = i;

            kernels::KernelContext ctx;
            auto tvin = in_view[from_sample];
            auto tvout = out_view[to_sample];

            auto in_sh_view = GetInputShape(from_sample);
            auto in_anchor_view = GetInAnchors(i, iter);
            auto out_anchor_view = GetOutAnchors(i, iter);
            auto region_shape = GetShape(i, iter, in_sh_view, in_anchor_view);
            Coords region_shape_view{region_shape.data(), coords_sh_};

            kernel_manager_.Run<Kernel>(
                    thread_id, to_sample, ctx, tvout, tvin,
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
