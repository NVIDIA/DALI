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
void Slice<GPUBackend>::DataDependentSetup(DeviceWorkspace *ws,
                                           unsigned int idx) {
  // Assumes xywh
  const auto &input = ws->Input<GPUBackend>(idx);

  TensorList<CPUBackend> begin;
  begin.Copy(ws->Input<GPUBackend>(idx + 1), ws->stream());

  TensorList<CPUBackend> size;
  size.Copy(ws->Input<GPUBackend>(idx + 2), ws->stream());

  std::vector<Dims> output_shape(batch_size_);

  for (int i = 0; i < batch_size_; i++) {
    auto H = input.tensor_shape(i)[0];
    auto W = input.tensor_shape(i)[1];
    auto C = input.tensor_shape(i)[2];

    crop_width_[i] = static_cast<int>(size.template data<float>()[0]);
    crop_height_[i] = static_cast<int>(size.template data<float>()[1]);

    per_sample_dimensions_[i] = std::make_pair(H, W);

    auto crop_y = static_cast<int>(begin.template data<float>()[1]);
    auto crop_x = static_cast<int>(begin.template data<float>()[0]);

    per_sample_crop_[i] = std::make_pair(crop_y, crop_x);

    input_strides_.template mutable_data<int>()[i] = static_cast<int>(W * C);
    crop_offsets_[i] = static_cast<int>((crop_y * W + crop_x) * C);

    DALITensorLayout outLayout;
    output_shape[i] = GetOutShape(input.GetLayout(), &outLayout, idx);
  }

  auto output = ws->Output<GPUBackend>(idx);

  output->Resize(output_shape);
  output->SetLayout(output_layout_ == DALI_SAME
                        ? ws->Input<GPUBackend>(idx).GetLayout()
                        : output_layout_);

  // Calculate input pointers and copy to gpu
  for (int i = 0; i < batch_size_; ++i) {
    input_ptrs_.template mutable_data<const uint8 *>()[i] =
        input.template tensor<uint8>(i) + crop_offsets_[i];
  }

  input_ptrs_gpu_.Copy(input_ptrs_, ws->stream());
  input_strides_gpu_.Copy(input_strides_, ws->stream());

  crop_width_gpu_.Copy(crop_width_, ws->stream());
  crop_height_gpu_.Copy(crop_height_, ws->stream());
}

template <>
void Slice<GPUBackend>::RunImpl(DeviceWorkspace *ws, int idx) {
  DataDependentSetup(ws, idx);

  switch (output_type_) {
    case DALI_FLOAT16:
      RunHelper<float16>(ws, idx);
      break;
    case DALI_FLOAT:
      RunHelper<float>(ws, idx);
      break;
    case DALI_UINT8:
      RunHelper<unsigned char>(ws, idx);
      break;
    case DALI_INT16:
      RunHelper<int16>(ws, idx);
      break;
    case DALI_INT32:
      RunHelper<int>(ws, idx);
      break;
    case DALI_INT64:
      RunHelper<int64>(ws, idx);
      break;
    default:
      DALI_FAIL("Unsupported output type.");
  }
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
