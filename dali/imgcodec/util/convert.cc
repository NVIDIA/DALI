// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cassert>
#include <stdexcept>
#include "dali/core/static_switch.h"
#include "dali/imgcodec/util/convert.h"
#include "dali/kernels/slice/slice_cpu.h"
#include "dali/kernels/common/utils.h"

#define IMG_CONVERT_TYPES uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, float, float16

namespace dali {
namespace imgcodec {

void Convert(SampleView<CPUBackend> out, TensorLayout out_layout, DALIImageType out_format,
             ConstSampleView<CPUBackend> in, TensorLayout in_layout, DALIImageType in_format,
             TensorShape<> roi_start, TensorShape<> roi_end) {
  const auto &out_shape = out.shape();
  const auto &in_shape = in.shape();
  assert(in_shape.sample_dim() == out_shape.sample_dim());
  int ndim = out_shape.sample_dim();
  int spatial_ndim = ndim - 1;
  int in_channel_dim = ImageLayoutInfo::ChannelDimIndex(in_layout);
  int out_channel_dim = ImageLayoutInfo::ChannelDimIndex(out_layout);

  DALI_ENFORCE(in_channel_dim == spatial_ndim,
    "Not implemented: currently only channels-last layout is supported");
  DALI_ENFORCE(in_layout == out_layout,
    "Not implemented: currently layout transposition is not supported");

  if (!roi_start.empty() && roi_start.sample_dim() != spatial_ndim)
    throw std::invalid_argument(
      "ROI start must be empty or have the dimensionality equal to the number of "
      "spatial dimensions of the data");

  if (!roi_end.empty() && roi_end.sample_dim() != spatial_ndim)
    throw std::invalid_argument(
      "ROI end must be empty or have the dimensionality equal to the number of "
      "spatial dimensions of the data");

  if (roi_start.empty())
    roi_start.resize(spatial_ndim);

  if (roi_end.empty()) {
    roi_end = in_shape.first(spatial_ndim);  // assumes channels-last
  }

  for (int d = 0; d < spatial_ndim; d++) {
    if (roi_end[d] - roi_start[d] != out_shape[d])
      throw std::logic_error("The requested ROI size does not match the output size");
  }

  auto UnsupportedType = [](const char *which_type, DALIDataType type_id) {
    DALI_FAIL(make_string("Unsupported ", which_type, " type: ", type_id,
                          ListTypeNames<IMG_CONVERT_TYPES>()));
  };

  TensorShape<> out_strides = kernels::GetStrides(out_shape);
  TensorShape<> in_strides = kernels::GetStrides(in_shape);
  ptrdiff_t in_offset = 0;
  for (int d = 0; d < roi_start.size(); d++)
    in_offset += in_strides[d] * roi_start[d];

  TYPE_SWITCH(out.type(), type2id, Out, (IMG_CONVERT_TYPES),
    TYPE_SWITCH(in.type(), type2id, In, (IMG_CONVERT_TYPES),
      (Convert(out.mutable_data<Out>(), out_strides.data(), out_channel_dim, out_format,
               in.data<In>() + in_offset, in_strides.data(), in_channel_dim, in_format,
               out_shape.data(), ndim)),
      (UnsupportedType("input", in.type());)),
    (UnsupportedType("output", out.type());));
}

}  // namespace imgcodec
}  // namespace dali
