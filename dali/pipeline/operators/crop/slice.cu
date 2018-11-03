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

#include "dali/pipeline/operators/crop/slice.h"

namespace dali {

template <>
void Slice<GPUBackend>::DataDependentSetup(DeviceWorkspace *ws, unsigned int idx) {
  auto &images = ws->Input<GPUBackend>(idx);
  DALI_ENFORCE(IsType<uint8>(images.type()), "Expected input data as uint8.");

  auto &anchors = ws->Input<GPUBackend>(idx + 1);
  auto &sizes = ws->Input<GPUBackend>(idx + 2);

  std::vector<Dims> output_shape(static_cast<unsigned long>(batch_size_));

  for (int i = 0; i < batch_size_; i++) {
    const auto input_shape = images.tensor_shape(i);
    DALI_ENFORCE(input_shape.size() == 3, "Expects 3-dimensional image input.");

    const auto H = static_cast<unsigned int>(input_shape[1]);
    DALI_ENFORCE(H > 0, "Expected Height > 0. Received: " + std::to_string(H));
    const auto W = static_cast<unsigned int>(input_shape[1]);
    DALI_ENFORCE(W > 0, "Expected Width > 0. Received: " + std::to_string(W));
    const auto C = static_cast<unsigned int>(input_shape[2]);
    DALI_ENFORCE(W > 0, "Expected Channels == 1 || Channels == 3. Received: " + std::to_string(C));

    auto anchor = anchors.template tensor<float>(i);
    const auto x = static_cast<unsigned int>(anchor[0]) * W;
    const auto y = static_cast<unsigned int>(anchor[1]) * H;

    input_strides_.template mutable_data<int>()[i] = static_cast<int>(C * W);

    const unsigned int crop_stride = (y * W + x) * C;

    input_ptrs_.template mutable_data<const uint8 *>()[i] =
        images.template tensor<uint8>(i) + crop_stride;

    auto size = sizes.template tensor<float>(i);

    crop_width_[i] = static_cast<int>(size[0]);
    crop_height_[i] = static_cast<int>(size[1]);

    DALITensorLayout outLayout;
    output_shape[i] = GetOutShape(images.GetLayout(), &outLayout, i);
  }

  auto output = ws->Output<GPUBackend>(idx);

  output->Resize(output_shape);
  output->SetLayout(output_layout_ == DALI_SAME ? images.GetLayout() : output_layout_);

  input_ptrs_gpu_.Copy(input_ptrs_, ws->stream());
  input_strides_gpu_.Copy(input_strides_, ws->stream());

  crop_width_gpu_.Copy(crop_width_, ws->stream());
  crop_height_gpu_.Copy(crop_height_, ws->stream());
}

template <>
void Slice<GPUBackend>::RunImpl(DeviceWorkspace *ws, int idx) {
  DataDependentSetup(ws, static_cast<unsigned int>(idx));

  if (output_type_ == DALI_FLOAT16)
    Crop<GPUBackend>::RunHelper<float16>(ws, idx);
  else
    Crop<GPUBackend>::CallRunHelper(ws, idx);
}

template <>
void Slice<GPUBackend>::SetupSharedSampleParams(DeviceWorkspace *ws) {
  DALI_ENFORCE(ws->NumInput() == 3, "Expected 3 inputs. Received: " +
                                        std::to_string(ws->NumInput()));

  const auto &input = ws->Input<GPUBackend>(0);

  if (output_type_ == DALI_NO_TYPE) output_type_ = input.type().id();
}

DALI_REGISTER_OPERATOR(Slice, Slice<GPUBackend>, GPU);

}  // namespace dali
