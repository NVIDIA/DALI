// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

namespace dali {
namespace {

template <typename Out, typename In>
using TheKernel = kernels::PasteCpu<Out, In>;

}  // namespace


DALI_SCHEMA(MultiPaste)
.DocStr(R"code(Performs multiple pastes from image batch to an output

This operator can also change the type of data.)code")

.NumInput(6)  // images, in_ids, out_ids, in_anchors, in_shapes, out_anchors
.NumOutput(1)
.AddOptionalArg("dtype",
R"code(Output data type.

If not set, the input type is used.)code", DALI_NO_TYPE)

.AddOptionalArg("output_width",
R"code(Output width.

If not set, this is calculated using sum of ROI widths.)code", -1, true)

.AddOptionalArg("output_height",
R"code(Output height.

If not set, this is calculated using sum of ROI heights.)code", -1, true);

DALI_REGISTER_OPERATOR(MultiPaste, MultiPasteCpu, CPU)

bool MultiPasteCpu::SetupImpl(std::vector<OutputDesc> &output_desc,
                                      const workspace_t<CPUBackend> &ws) {
  AcquireArguments(ws);

  const auto &images = ws.template InputRef<CPUBackend>(0);
  const auto &in_idx = ws.template InputRef<CPUBackend>(1);
  const auto &out_idx = ws.template InputRef<CPUBackend>(2);
  const auto &in_anchors = ws.template InputRef<CPUBackend>(3);
  const auto &in_shapes = ws.template InputRef<CPUBackend>(4);
  const auto &out_anchors = ws.template InputRef<CPUBackend>(5);
  const auto &output = ws.template OutputRef<CPUBackend>(0);
  output_desc.resize(1);

  const int n_paste = in_anchors.shape().num_samples();
  DALI_ENFORCE(out_idx.shape().num_samples() == n_paste,
               "out_idx must be same length as in_idx");
  DALI_ENFORCE(in_anchors.shape().num_samples() == n_paste,
               "in_anchors must be same length as in_idx");
  DALI_ENFORCE(in_shapes.shape().num_samples() == n_paste,
               "in_shapes must be same length as in_idx");
  DALI_ENFORCE(out_anchors.shape().num_samples() == n_paste,
               "out_anchors must be same length as in_idx");

  TYPE_SWITCH(images.type().id(), type2id, InputType, (uint8_t, int16_t, int32_t, float), (
      TYPE_SWITCH(output_type_, type2id, OutputType, (uint8_t, int16_t, int32_t, float), (
          {
            using Kernel = TheKernel<OutputType, InputType>;
            kernel_manager_.Initialize<Kernel>();
            TensorListShape<> sh = images.shape();
            TensorListShape<> shapes(sh.num_samples(), 3);
            for (int i = 0; i < sh.num_samples(); i++) {
              const TensorListShape<> &out_sh = dali::TensorListShape<>(output_width_[i],
                                                                      output_height_[i]);
              shapes.set_tensor_shape(i, out_sh.tensor_shape(0));
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
  const auto &out_idx = ws.template InputRef<CPUBackend>(2);
  const auto &in_anchors = ws.template InputRef<CPUBackend>(3);
  const auto &in_shapes = ws.template InputRef<CPUBackend>(4);
  const auto &out_anchors = ws.template InputRef<CPUBackend>(5);
  auto &output = ws.template OutputRef<CPUBackend>(0);

  output.SetLayout(images.GetLayout());
  auto out_shape = output.shape();

  auto& tp = ws.GetThreadPool();

  auto paste_count = in_idx.shape().num_samples();

  TYPE_SWITCH(images.type().id(), type2id, InputType, (uint8_t, int16_t, int32_t, float), (
      TYPE_SWITCH(output_type_, type2id, OutputType, (uint8_t, int16_t, int32_t, float), (
          {
            using Kernel = TheKernel<OutputType, InputType>;
            for (int paste = 0; paste < paste_count; paste++) {
              int from_sample = in_idx[paste].template data<int>()[0];
              int to_sample = out_idx[paste].template data<int>()[0];

              tp.AddWork([&, paste](int thread_id) {
                kernels::KernelContext ctx;
                auto tvin = view<const InputType, 3>(images[from_sample]);
                auto tvout = view<OutputType, 3>(output[to_sample]);

                auto in_anchor_view = view<const int, 1>(in_anchors[from_sample]);
                auto in_shape_view = view<const int, 1>(in_shapes[from_sample]);
                auto out_anchor_view = view<const int, 1>(out_anchors[from_sample]);

                kernel_manager_.Run<Kernel>(thread_id, to_sample, ctx, tvout, tvin,
                                            in_anchor_view, in_shape_view, out_anchor_view);
              }, out_shape.tensor_size(to_sample));
            }
          }
      ), DALI_FAIL(make_string("Unsupported output type: ", output_type_)))  // NOLINT
  ), DALI_FAIL(make_string("Unsupported input type: ", images.type().id())))  // NOLINT
  tp.RunAll();
}

}  // namespace dali
