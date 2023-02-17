// Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "dali/core/tensor_shape_print.h"

namespace dali {
namespace imgcodec {

void Convert(SampleView<CPUBackend> out, TensorLayout out_layout, DALIImageType out_format,
             ConstSampleView<CPUBackend> in, TensorLayout in_layout, DALIImageType in_format,
             ROI roi, Orientation orientation) {
  auto out_shape = out.shape();
  const auto &in_shape = in.shape();
  assert(in_shape.sample_dim() == out_shape.sample_dim());
  int ndim = out_shape.sample_dim();
  int spatial_ndim = ndim - 1;
  int in_channel_dim = ImageLayoutInfo::ChannelDimIndex(in_layout);
  int out_channel_dim = ImageLayoutInfo::ChannelDimIndex(out_layout);
  int h_dim = ImageLayoutInfo::DimIndex(out_layout, 'H');
  int w_dim = ImageLayoutInfo::DimIndex(out_layout, 'W');
  DALI_ENFORCE(h_dim >= 0 && w_dim >= 0,
               "Output layout has to contain at least H and W dimensions.");

  DALI_ENFORCE(in_layout == out_layout,
    "Not implemented: currently layout transposition is not supported");

  if (!roi.begin.empty() && roi.begin.sample_dim() != spatial_ndim)
    throw std::invalid_argument(
      "ROI start must be empty or have the dimensionality equal to the number of "
      "spatial dimensions of the data");

  if (!roi.end.empty() && roi.end.sample_dim() != spatial_ndim)
    throw std::invalid_argument(
      "ROI end must be empty or have the dimensionality equal to the number of "
      "spatial dimensions of the data");

  if (roi.begin.empty())
    roi.begin.resize(spatial_ndim);

  auto out_shape_no_channel = detail::RemoveDim(out_shape, out_channel_dim);
  if (roi.end.empty()) {
    roi.end = out_shape_no_channel;
  }
  for (int d = 0; d < spatial_ndim; d++) {
    if (roi.end[d] - roi.begin[d] != out_shape_no_channel[d])
      throw std::logic_error("The requested ROI size does not match the output size");
  }

  auto UnsupportedOutputType = [](DALIDataType type_id) {
    DALI_FAIL(make_string("Unsupported output type: ", type_id,
                          ListTypeNames<IMGCODEC_TYPES>()));
  };

  auto UnsupportedInputType = [](DALIDataType type_id) {
    DALI_FAIL(make_string("Unsupported intput type: ", type_id,
                          ListTypeNames<IMGCODEC_IN_TYPES>()));
  };

  TensorShape<> out_strides = kernels::GetStrides(out_shape);
  TensorShape<> in_strides = kernels::GetStrides(in_shape);
  ptrdiff_t in_offset = 0;

  auto in_strides_no_channel = detail::RemoveDim(in_strides, in_channel_dim);

  for (int d = 0; d < roi.begin.size(); d++) {
    in_offset += in_strides_no_channel[d] * roi.begin[d];
  }

  TYPE_SWITCH(out.type(), type2id, Out, (IMGCODEC_TYPES),
    TYPE_SWITCH(in.type(), type2id, In, (IMGCODEC_IN_TYPES),
      auto out_ptr = out.mutable_data<Out>();
      ApplyOrientation(orientation, out_ptr,
                       out_strides[w_dim], out_shape[w_dim],
                       out_strides[h_dim], out_shape[h_dim]);
      (Convert(out_ptr, out_strides.data(), out_channel_dim, out_format,
               in.data<In>() + in_offset, in_strides.data(), in_channel_dim, in_format,
               out_shape.data(), ndim)),
      (UnsupportedInputType(in.type());)),
    (UnsupportedOutputType(out.type());));
}

}  // namespace imgcodec
}  // namespace dali
