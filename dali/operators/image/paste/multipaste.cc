// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
namespace {

template <typename Out, typename In>
using TheKernel = kernels::PasteCpu<Out, In>;

}  // namespace


DALI_SCHEMA(MultiPaste)
.DocStr(R"code(Performs multiple (K * batch_size) pastes from image batch to an output

This operator can also change the type of data.)code")

.NumInput(5, 6)
.InputDox(0, "images", "3D TensorList", R"code(Batch of input images.

Assumes HWC layout.)code")
.InputDox(1, "in_ids", "1D TesorList of shape [K] and type int",
R"code(Indexes from what inputs to paste data in each iteration.)code")
.InputDox(2, "in_anchors", "2D TesorList of shape [K, 2] and type int",
R"code(Absolute values of LU corner of the selection for each iteration.)code")
.InputDox(3, "in_shapes", "2D TesorList of shape [K, 2] and type int",
R"code(Absolute values of size of the selection for each iteration.)code")
.InputDox(4, "out_anchors", "2D TesorList of shape [K, 2] and type int",
R"code(Absolute values of LU corner of the paste for each iteration.)code")
.InputDox(5, "out_ids", "1D TesorList of shape [K] and type int",
R"code(Indexes to what outputs to paste data in each iteration.
If ommitted, i-th tensor pastes to i-th output. )code")
.NumOutput(1)
.AddArg("output_width",
R"code(Output width.)code", DALI_INT32, true)

.AddArg("output_height",
R"code(Output height.)code", DALI_INT32, true)
.AddOptionalArg("dtype",
R"code(Output data type.

If not set, the input type is used.)code", DALI_NO_TYPE)
.AddOptionalArg("input_out_ids",
R"code(If true, the operator takes the last, out_ids input.)code", false)
.AddOptionalArg("no_intersections",
R"code(If true, the operator assumes paste regions do not intersect.

This allows for better multithreading, but might produce weird artefacts if
the assertion is false.)code", false);

DALI_REGISTER_OPERATOR(MultiPaste, MultiPasteCpu, CPU)

bool MultiPasteCpu::SetupImpl(std::vector<OutputDesc> &output_desc,
                                      const workspace_t<CPUBackend> &ws) {
  AcquireArguments(ws);

  const auto &images = ws.template InputRef<CPUBackend>(0);
  const auto &in_idx = ws.template InputRef<CPUBackend>(1);
  const auto &in_anchors = ws.template InputRef<CPUBackend>(2);
  const auto &in_shapes = ws.template InputRef<CPUBackend>(3);
  const auto &out_anchors = ws.template InputRef<CPUBackend>(4);

  // If input_out_ids_ is false, we initialize this as a reference to other Input,
  // as a reference has to be set, but we do not use it anywhere
  const auto &out_idx = ws.template InputRef<CPUBackend>(input_out_ids_ ? 5 : 1);

  const auto &output = ws.template OutputRef<CPUBackend>(0);
  output_desc.resize(1);

  const int n_samples = in_anchors.shape().num_samples();

  for (int i = 0; i < n_samples; i++) {
    const int n_paste = in_anchors[i].shape()[0];

    DALI_ENFORCE(in_anchors[i].shape()[0] == n_paste,
                 "in_anchors must be same length as in_idx");
    DALI_ENFORCE(in_shapes[i].shape()[0] == n_paste,
                 "in_shapes must be same length as in_idx");
    DALI_ENFORCE(out_anchors[i].shape()[0] == n_paste,
                 "out_anchors must be same length as in_idx");

    if (input_out_ids_) {
      DALI_ENFORCE(out_idx[i].shape()[0] == n_paste,
                   "out_idx must be same length as in_idx");
    }
  }

  TYPE_SWITCH(images.type().id(), type2id, InputType, (uint8_t, int16_t, int32_t, float), (
      TYPE_SWITCH(output_type_, type2id, OutputType, (uint8_t, int16_t, int32_t, float), (
          {
            using Kernel = TheKernel<OutputType, InputType>;
            kernel_manager_.Initialize<Kernel>();
            TensorListShape<> sh = images.shape();
            TensorListShape<> shapes(sh.num_samples(), 3);
            for (int i = 0; i < sh.num_samples(); i++) {
              const TensorShape<> &out_sh =
                dali::TensorShape<>(output_height_[i], output_width_[i], sh[i][2]);
              shapes.set_tensor_shape(i, out_sh);
            }

            TypeInfo type;
            type.SetType<OutputType>(output_type_);
            output_desc[0] = {shapes, type};
          }
      ), DALI_FAIL(make_string("Unsupported output type: ", output_type_)))  // NOLINT
  ), DALI_FAIL(make_string("Unsupported input type: ", images.type().id())))  // NOLINT
  return true;
}


void MultiPasteCpu::RunImpl(workspace_t<CPUBackend> &ws) {
  const auto &images = ws.template InputRef<CPUBackend>(0);
  const auto &in_idx = ws.template InputRef<CPUBackend>(1);
  const auto &in_anchors = ws.template InputRef<CPUBackend>(2);
  const auto &in_shapes = ws.template InputRef<CPUBackend>(3);
  const auto &out_anchors = ws.template InputRef<CPUBackend>(4);
  const auto &out_idx = ws.template InputRef<CPUBackend>(input_out_ids_ ? 5 : 1);

  auto &output = ws.template OutputRef<CPUBackend>(0);

  output.SetLayout(images.GetLayout());
  auto out_shape = output.shape();

  auto& tp = ws.GetThreadPool();

  auto batch_size = in_idx.shape().num_samples();

  TYPE_SWITCH(images.type().id(), type2id, InputType, (uint8_t, int16_t, int32_t, float), (
      TYPE_SWITCH(output_type_, type2id, OutputType, (uint8_t, int16_t, int32_t, float), (
          {
            using Kernel = TheKernel<OutputType, InputType>;
            if (no_intersections_) {
              for (int i = 0; i < batch_size; i++) {
                auto paste_count = in_idx[i].shape()[0];

                for (int iter = 0; iter < paste_count; iter++) {
                  int from_sample = in_idx[i].template data<int>()[iter];
                  int to_sample = input_out_ids_ ? out_idx[i].template data<int>()[iter] : i;

                  tp.AddWork(
                      [&, i, iter, from_sample, to_sample](int thread_id) {
                        kernels::KernelContext ctx;
                        auto tvin = view<const InputType, 3>(images[from_sample]);
                        auto tvout = view<OutputType, 3>(output[to_sample]);

                        auto in_anchor_view = subtensor(view<const int, 2>(in_anchors[i]), iter);
                        auto in_shape_view = subtensor(view<const int, 2>(in_shapes[i]), iter);
                        auto out_anchor_view = subtensor(view<const int, 2>(out_anchors[i]), iter);

                        kernel_manager_.Run<Kernel>(thread_id, to_sample, ctx, tvout, tvin,
                                                  in_anchor_view, in_shape_view, out_anchor_view);
                      },
                      out_shape.tensor_size(to_sample));
                }
              }
            } else if (!input_out_ids_) {
              for (int i = 0; i < batch_size; i++) {
                auto paste_count = in_idx[i].shape()[0];

                tp.AddWork(
                    [&, i](int thread_id) {
                       for (int iter = 0; iter < paste_count; iter++) {
                         int from_sample = in_idx[i].template data<int>()[iter];
                         int to_sample = input_out_ids_ ? out_idx[i].template data<int>()[iter] : i;

                         kernels::KernelContext ctx;
                         auto tvin = view<const InputType, 3>(images[from_sample]);
                         auto tvout = view<OutputType, 3>(output[to_sample]);
                         int s_size = 2;
                         auto in_anchor_view = subtensor(view<const int, 2>(in_anchors[i]), iter);
                         auto in_shape_view = subtensor(view<const int, 2>(in_shapes[i]), iter);
                         auto out_anchor_view = subtensor(view<const int, 2>(out_anchors[i]), iter);

                         kernel_manager_.Run<Kernel>(thread_id, to_sample, ctx, tvout, tvin,
                                                  in_anchor_view, in_shape_view, out_anchor_view);
                       }
                    }, paste_count);
              }
            } else {
              tp.AddWork(
                  [&](int thread_id) {
                    for (int i = 0; i < batch_size; i++) {
                      auto paste_count = in_idx[i].shape()[0];

                      for (int iter = 0; iter < paste_count; iter++) {
                        int from_sample = in_idx[i].template data<int>()[iter];
                        int to_sample = input_out_ids_ ? out_idx[i].template data<int>()[iter] : i;

                        kernels::KernelContext ctx;
                        auto tvin = view<const InputType, 3>(images[from_sample]);
                        auto tvout = view<OutputType, 3>(output[to_sample]);
                        int s_size = 2;
                        auto in_anchor_view = subtensor(view<const int, 2>(in_anchors[i]), iter);
                        auto in_shape_view = subtensor(view<const int, 2>(in_shapes[i]), iter);
                        auto out_anchor_view = subtensor(view<const int, 2>(out_anchors[i]), iter);

                        kernel_manager_.Run<Kernel>(thread_id, to_sample, ctx, tvout, tvin,
                                                  in_anchor_view, in_shape_view, out_anchor_view);
                      }
                    }
                  }, 1);
            }
          }
      ), DALI_FAIL(make_string("Unsupported output type: ", output_type_)))  // NOLINT
  ), DALI_FAIL(make_string("Unsupported input type: ", images.type().id())))  // NOLINT
  tp.RunAll();
}

}  // namespace dali
