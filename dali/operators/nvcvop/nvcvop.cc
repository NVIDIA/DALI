// Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/operators/nvcvop/nvcvop.h"

#include <string>

namespace dali::nvcvop {

NVCVBorderType GetBorderMode(std::string_view border_mode) {
  if (border_mode == "constant") {
    return NVCV_BORDER_CONSTANT;
  } else if (border_mode == "replicate") {
    return NVCV_BORDER_REPLICATE;
  } else if (border_mode == "reflect") {
    return NVCV_BORDER_REFLECT;
  } else if (border_mode == "reflect_101") {
    return NVCV_BORDER_REFLECT101;
  } else if (border_mode == "wrap") {
    return NVCV_BORDER_WRAP;
  } else {
    DALI_FAIL("Unknown border mode: " + std::string(border_mode));
  }
}

NVCVInterpolationType GetInterpolationType(DALIInterpType interpolation_type) {
  switch (interpolation_type) {
    case DALIInterpType::DALI_INTERP_NN:
      return NVCV_INTERP_NEAREST;
    case DALIInterpType::DALI_INTERP_LINEAR:
      return NVCV_INTERP_LINEAR;
    case DALIInterpType::DALI_INTERP_CUBIC:
      return NVCV_INTERP_CUBIC;
    default:
      DALI_FAIL(make_string("Unknown interpolation type: ", static_cast<int>(interpolation_type)));
  }
}

nvcv::DataKind GetDataKind(DALIDataType dtype) {
  switch (dtype) {
    case DALI_UINT8:
    case DALI_UINT16:
    case DALI_UINT32:
    case DALI_UINT64:
      return nvcv::DataKind::UNSIGNED;
    case DALI_INT8:
    case DALI_INT16:
    case DALI_INT32:
    case DALI_INT64:
      return nvcv::DataKind::SIGNED;
    case DALI_FLOAT16:
    case DALI_FLOAT:
    case DALI_FLOAT64:
      return nvcv::DataKind::FLOAT;
    default:
      DALI_FAIL("Unsupported data type");
  }
}

nvcv::Swizzle GetSimpleSwizzle(int num_channels) {
  if (num_channels == 1) {
      return nvcv::Swizzle::S_X000;
  } else if (num_channels == 2) {
      return nvcv::Swizzle::S_XY00;
  } else if (num_channels == 3) {
      return nvcv::Swizzle::S_XYZ0;
  } else if (num_channels == 4) {
      return nvcv::Swizzle::S_XYZW;
  } else {
      DALI_FAIL("Unsupported number of channels");
  }
}

nvcv::Packing GetPacking(DALIDataType dtype, const nvcv::Swizzle &swizzle) {
  nvcv::PackingParams packing_params{};
  packing_params.byteOrder = nvcv::ByteOrder::MSB;
  packing_params.swizzle = swizzle;
  for (int c = 0; c < nvcv::GetNumChannels(swizzle); ++c) {
    packing_params.bits[c] = TypeTable::GetTypeInfo(dtype).size() * 8;
  }
  return nvcv::MakePacking(packing_params);
}

nvcv::ImageFormat GetImageFormat(DALIDataType dtype, int num_channels) {
  nvcv::MemLayout mem_layout = nvcv::MemLayout::PITCH_LINEAR;
  nvcv::DataKind data_kind = GetDataKind(dtype);
  nvcv::Swizzle swizzle = GetSimpleSwizzle(num_channels);
  auto packing = GetPacking(dtype, swizzle);
  return nvcv::ImageFormat(mem_layout, data_kind, swizzle, packing);
}

nvcv::DataType GetDataType(DALIDataType dtype, int num_channels) {
  auto swizzle = GetSimpleSwizzle(num_channels);
  return nvcv::DataType(GetDataKind(dtype), GetPacking(dtype, swizzle));
}

/**
 * @brief Wrap tensor data buffer into nvcv::Image.
 *
 * @param data pointer to beginning of the buffer
 * @param shape underlying tensor shape
 * @param format image format
 */
nvcv::Image WrapImage(void *data, const TensorShape<> &shape, const nvcv::ImageFormat &format) {
  DALI_ENFORCE(format.numPlanes() == 1);
  auto pixel_size = format.planeBitsPerPixel(0) / 8;
  nvcv::ImageDataStridedCuda::Buffer buf{};
  buf.numPlanes = 1;
  buf.planes[0].basePtr = static_cast<NVCVByte*>(data);
  buf.planes[0].height = shape[0];
  buf.planes[0].width  = shape[1];
  buf.planes[0].rowStride = shape[1] * pixel_size;
  nvcv::ImageDataStridedCuda img_data(format, buf);
  return nvcv::ImageWrapData(img_data);
}

nvcv::Image AsImage(SampleView<GPUBackend> sample, const nvcv::ImageFormat &format) {
  return AsImage(ConstSampleView<GPUBackend>(sample), format);
}

nvcv::Image AsImage(ConstSampleView<GPUBackend> sample, const nvcv::ImageFormat &format) {
  auto &shape = sample.shape();
  auto data = const_cast<void*>(sample.raw_data());
  return WrapImage(data, shape, format);
}

void AllocateImagesLike(nvcv::ImageBatchVarShape &output, const TensorList<GPUBackend> &t_list,
                        kernels::Scratchpad &scratchpad) {
  auto channel_dim = t_list.GetLayout().find('C');
  uint8_t *buffer = scratchpad.AllocateGPU<uint8_t>(t_list.nbytes());
  size_t offset = 0;
  for (int s = 0; s < t_list.num_samples(); ++s) {
    auto tensor = t_list[s];
    auto num_channels = (channel_dim >= 0) ? tensor.shape()[channel_dim] : 1;
    auto format = GetImageFormat(t_list.type(), num_channels);
    auto image = WrapImage(buffer + offset, tensor.shape(), format);
    output.pushBack(image);
    offset += volume(tensor.shape()) * t_list.type_info().size();
  }
}

void PushImagesToBatch(nvcv::ImageBatchVarShape &batch, const TensorList<GPUBackend> &t_list) {
  auto channel_dim = t_list.GetLayout().find('C');
  for (int s = 0; s < t_list.num_samples(); ++s) {
    auto num_channels = (channel_dim >= 0) ? t_list[s].shape()[channel_dim] : 1;
    auto format = GetImageFormat(t_list.type(), num_channels);
    auto image = AsImage(t_list[s], format);
    batch.pushBack(image);
  }
}

nvcv::Tensor AsTensor(const Tensor<GPUBackend> &tensor, TensorLayout layout,
                      const std::optional<TensorShape<>> &reshape) {
  auto orig_shape = tensor.shape();
  auto dtype = GetDataType(tensor.type(), 1);

  TensorShape<> shape;
  if (reshape.has_value()) {
    DALI_ENFORCE(volume(*reshape) == volume(orig_shape),
                 make_string("Cannot reshape from ", orig_shape, " to ", *reshape, "."));
    shape = reshape.value();
  } else {
    shape = orig_shape;
  }

  nvcv::TensorDataStridedCuda::Buffer inBuf;
  inBuf.basePtr = reinterpret_cast<NVCVByte*>(const_cast<void*>(tensor.raw_data()));
  inBuf.strides[shape.size() - 1] = dtype.strideBytes();
  for (int d = shape.size() - 2; d >= 0; --d) {
    inBuf.strides[d] = shape[d + 1] * inBuf.strides[d + 1];
  }
  TensorLayout out_layout = layout.empty() ? tensor.GetLayout() : layout;
  DALI_ENFORCE(out_layout.empty() || out_layout.size() == shape.size(),
               make_string("Layout ", out_layout,
                           " does not match the number of dimensions: ", shape.size()));
  nvcv::TensorShape out_shape(shape.data(), shape.size(),
                              nvcv::TensorLayout(out_layout.c_str()));
  nvcv::TensorDataStridedCuda inData(out_shape, dtype, inBuf);
  return nvcv::TensorWrapData(inData);
}

}  // namespace dali::nvcvop
