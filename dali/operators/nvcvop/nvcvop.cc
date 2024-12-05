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
#include <utility>

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
  TensorShape<> shape;
  if (reshape.has_value()) {
    DALI_ENFORCE(volume(*reshape) == volume(orig_shape),
                 make_string("Cannot reshape from ", orig_shape, " to ", *reshape, "."));
    shape = reshape.value();
  } else {
    shape = orig_shape;
  }

  TensorLayout out_layout = layout.empty() ? tensor.GetLayout() : layout;

  return AsTensor(const_cast<void *>(tensor.raw_data()), shape, tensor.type(), out_layout);
}

nvcv::Tensor AsTensor(SampleView<GPUBackend> sample, TensorLayout layout,
                      const std::optional<TensorShape<>> &reshape) {
  auto orig_shape = sample.shape();
  TensorShape<> shape;
  if (reshape.has_value()) {
    DALI_ENFORCE(volume(*reshape) == volume(orig_shape),
                 make_string("Cannot reshape from ", orig_shape, " to ", *reshape, "."));
    shape = reshape.value();
  } else {
    shape = orig_shape;
  }

  return AsTensor(sample.raw_mutable_data(), shape, sample.type(), layout);
}

nvcv::Tensor AsTensor(ConstSampleView<GPUBackend> sample, TensorLayout layout,
                      const std::optional<TensorShape<>> &reshape) {
  auto orig_shape = sample.shape();
  TensorShape<> shape;
  if (reshape.has_value()) {
    DALI_ENFORCE(volume(*reshape) == volume(orig_shape),
                 make_string("Cannot reshape from ", orig_shape, " to ", *reshape, "."));
    shape = reshape.value();
  } else {
    shape = orig_shape;
  }

  return AsTensor(const_cast<void *>(sample.raw_data()), shape, sample.type(), layout);
}

nvcv::Tensor AsTensor(void *data, const TensorShape<> &shape, DALIDataType daliDType,
                      TensorLayout layout) {
  auto dtype = GetDataType(daliDType, 1);
  nvcv::TensorDataStridedCuda::Buffer inBuf;
  inBuf.basePtr = static_cast<NVCVByte *>(const_cast<void *>(data));
  inBuf.strides[shape.size() - 1] = dtype.strideBytes();
  for (int d = shape.size() - 2; d >= 0; --d) {
    inBuf.strides[d] = shape[d + 1] * inBuf.strides[d + 1];
  }
  DALI_ENFORCE(
      layout.empty() || layout.size() == shape.size(),
      make_string("Layout ", layout, " does not match the number of dimensions: ", shape.size()));
  nvcv::TensorShape out_shape(shape.data(), shape.size(), nvcv::TensorLayout(layout.c_str()));
  nvcv::TensorDataStridedCuda inData(out_shape, dtype, inBuf);
  return nvcv::TensorWrapData(inData);
}

nvcv::Tensor AsTensor(const void *data, span<const int64_t> shape_data, const nvcv::DataType &dtype,
                      const nvcv::TensorLayout &layout) {
  int ndim = shape_data.size();
  nvcv::TensorDataStridedCuda::Buffer inBuf;
  inBuf.basePtr = static_cast<NVCVByte *>(const_cast<void *>(data));
  inBuf.strides[ndim - 1] = dtype.strideBytes();
  for (int d = ndim - 2; d >= 0; --d) {
    inBuf.strides[d] = shape_data[d + 1] * inBuf.strides[d + 1];
  }
  nvcv::TensorShape out_shape(shape_data.data(), ndim, layout);
  nvcv::TensorDataStridedCuda inData(out_shape, dtype, inBuf);
  return nvcv::TensorWrapData(inData);
}

int64_t calc_num_frames(const TensorShape<> &shape, int first_spatial_dim) {
  return (first_spatial_dim > 0) ?
          volume(&shape[0], &shape[first_spatial_dim]) :
          1;
}

void PushFramesToBatch(nvcv::TensorBatch &batch, const TensorList<GPUBackend> &t_list,
                       int first_spatial_dim, int64_t starting_sample, int64_t frame_offset,
                       int64_t num_frames, const TensorLayout &layout) {
  int ndim = layout.ndim();
  auto nvcv_layout = nvcv::TensorLayout(layout.c_str());
  auto dtype = GetDataType(t_list.type());

  std::vector<nvcv::Tensor> tensors;
  tensors.reserve(num_frames);

  const auto &input_shape = t_list.shape();
  int64_t sample_id = starting_sample - 1;
  auto type_size = dtype.strideBytes();
  std::vector<int64_t> frame_shape(ndim, 1);

  auto frame_stride = 0;
  int sample_nframes = 0;
  const uint8_t *data = nullptr;

  for (int64_t i = 0; i < num_frames; ++i) {
    if (frame_offset == sample_nframes) {
      frame_offset = 0;
      do {
        ++sample_id;
        auto sample_shape = input_shape[sample_id];
        DALI_ENFORCE(sample_id < t_list.num_samples());
        std::copy(&sample_shape[first_spatial_dim], &sample_shape[input_shape.sample_dim()],
                  frame_shape.begin());
        frame_stride = volume(frame_shape) * type_size;
        sample_nframes = calc_num_frames(sample_shape, first_spatial_dim);
      } while (sample_nframes * frame_stride == 0);  // we skip empty samples
      data =
          static_cast<const uint8_t *>(t_list.raw_tensor(sample_id)) + frame_stride * frame_offset;
    }
    tensors.push_back(AsTensor(data, make_span(frame_shape), dtype, nvcv_layout));
    data += frame_stride;
    frame_offset++;
  }
  batch.pushBack(tensors.begin(), tensors.end());
}


cvcuda::Workspace NVCVOpWorkspace::Allocate(const cvcuda::WorkspaceRequirements &reqs,
                                            kernels::Scratchpad &scratchpad) {
  auto *hostBuffer = scratchpad.AllocateHost<uint8_t>(reqs.hostMem.size, reqs.hostMem.alignment);
  auto *pinnedBuffer =
      scratchpad.AllocatePinned<uint8_t>(reqs.pinnedMem.size, reqs.pinnedMem.alignment);
  auto *gpuBuffer = scratchpad.AllocateGPU<uint8_t>(reqs.cudaMem.size, reqs.cudaMem.alignment);

  workspace_.hostMem.data = hostBuffer;
  workspace_.hostMem.req = reqs.hostMem;
  workspace_.pinnedMem.data = pinnedBuffer;
  workspace_.pinnedMem.req = reqs.pinnedMem;
  workspace_.cudaMem.data = gpuBuffer;
  workspace_.cudaMem.req = reqs.cudaMem;
  return workspace_;
}

nvcv::Allocator GetScratchpadAllocator(kernels::Scratchpad &scratchpad) {
  auto hostAllocator = nvcv::CustomHostMemAllocator(
      [&](int64_t size, int32_t align) { return scratchpad.AllocateHost<uint8_t>(size, align); },
      [](void *, int64_t, int32_t) {});

  auto pinnedAllocator = nvcv::CustomHostPinnedMemAllocator(
      [&](int64_t size, int32_t align) { return scratchpad.AllocatePinned<uint8_t>(size, align); },
      [](void *, int64_t, int32_t) {});

  auto gpuAllocator = nvcv::CustomCudaMemAllocator(
      [&](int64_t size, int32_t align) { return scratchpad.AllocateGPU<uint8_t>(size, align); },
      [](void *, int64_t, int32_t) {});

  return nvcv::CustomAllocator(std::move(hostAllocator), std::move(pinnedAllocator),
                               std::move(gpuAllocator));
}

}  // namespace dali::nvcvop
