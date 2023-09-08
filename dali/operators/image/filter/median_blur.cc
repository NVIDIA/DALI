// Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <optional>
#include <nvcv/Image.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <cvcuda/OpMedianBlur.hpp>
#include "dali/core/dev_buffer.h"
#include "dali/kernels/dynamic_scratchpad.h"
#include "dali/core/static_switch.h"
#include "dali/kernels/common/utils.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/operator/arg_helper.h"

namespace dali {


DALI_SCHEMA(experimental__MedianBlur)
  .DocStr(R"doc(
Median blur performs smoothing of an image or sequence of images by replacing each pixel
with the median color of a surrounding rectangular region.
  )doc")
  .NumInput(1)
  .InputDox(0, "input", "TensorList",
            "Input data. Must be images in HWC or CHW layout, or a sequence of those.")
  .NumOutput(1)
  .InputLayout({"HWC", "FHWC", "CHW", "FCHW"})
  .AddOptionalArg("window_size",
    "The size of the window over which the smoothing is performed",
    std::vector<int>({3, 3}),
    true);


class MedianBlur : public Operator<GPUBackend> {
 public:
  explicit MedianBlur(const OpSpec &spec)
  : Operator<GPUBackend>(spec) {}

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) override {
    auto &in = ws.Input<GPUBackend>(0);
    int num_images = NumImages(in);
    if (num_images > effective_batch_size_) {
      effective_batch_size_ = std::max(effective_batch_size_ * 2, num_images);
      input_batch_ = nvcv::ImageBatchVarShape(effective_batch_size_);
      output_batch_ = nvcv::ImageBatchVarShape(effective_batch_size_);
      ksize_buffer_.reserve(effective_batch_size_ * 2, ws.stream());
      impl_.emplace(effective_batch_size_);
    } else {
      input_batch_.clear();
      output_batch_.clear();
    }
    nvcv_format_ = GetImageFormat(in);
    GetImages(input_batch_, in, in.GetLayout(), &batch_map_);

    ksize_arg_.Acquire(spec_, ws, in.num_samples(), TensorShape<1>{2});
    if (ksize_arg_.HasArgumentInput() ||
        ksize_tensor_.empty() ||
        num_images != ksize_tensor_.shape()[0]) {
      SetupKSizeTensor(ws, input_batch_.numImages());
    }
    output_desc.resize(1);
    output_desc[0].type = in.type();
    output_desc[0].shape = in.shape();
    return true;
  }

  void RunImpl(Workspace &ws) override {
    auto &input = ws.Input<GPUBackend>(0);
    auto &output = ws.Output<GPUBackend>(0);
    GetImages(output_batch_, output, input.GetLayout());
    (*impl_)(ws.stream(), input_batch_, output_batch_, ksize_tensor_);
  }

  bool CanInferOutputs() const override {
    return true;
  }

 protected:
  void SetupKSizeTensor(const Workspace &ws, int num_samples) {
    kernels::DynamicScratchpad scratchpad({}, AccessOrder(ws.stream()));
    auto ksize_cpu = scratchpad.AllocatePinned<int32_t>(num_samples * 2);
    for (int64_t i = 0; i < num_samples; ++i) {
      auto dst = ksize_cpu + i * 2;
      auto s = batch_map_[i];
      auto src = ksize_arg_[s].data;
      memcpy(dst, src, ksize_arg_[s].num_elements() * sizeof(int32_t));
    }
    ksize_buffer_.resize(num_samples * 2);
    MemCopy(ksize_buffer_.data(), ksize_cpu, ksize_buffer_.size_bytes(), ws.stream());

    nvcv::TensorDataStridedCuda::Buffer inBuf;
    inBuf.strides[1] = sizeof(int);
    inBuf.strides[0] = 2 * sizeof(int);
    auto src = ksize_buffer_.data();
    inBuf.basePtr = reinterpret_cast<NVCVByte*>(src);

    int64_t shape_data[]{num_samples, 2};
    nvcv::TensorShape shape(shape_data, 2, "NW");

    nvcv::TensorDataStridedCuda inData(shape, nvcv::DataType{NVCV_DATA_TYPE_S32}, inBuf);
    ksize_tensor_ = nvcv::TensorWrapData(inData);
  }

  int NumImages(const TensorList<GPUBackend> &input) {
    const auto &shape = input.shape();
    int cdim = input.GetLayout().find('C');
    assert(cdim >= 0);
    bool channel_last = cdim == shape.sample_dim() - 1;
    int fdim = input.GetLayout().find('F');
    int num_images = 0;
    for (int i = 0; i < shape.num_samples(); ++i) {
      int c = (channel_last) ? 1 : shape.tensor_shape(i)[cdim];
      int f = (fdim != -1) ? shape.tensor_shape(i)[fdim] : 1;
      num_images += c * f;
    }
    return num_images;
  }

  nvcv::ImageFormat GetImageFormat(const TensorList<GPUBackend> &input) {
    int cdim = input.GetLayout().find('C');
    ValidateChannels(input, cdim);
    int channels = (cdim == input.sample_dim() - 1) ? input.tensor_shape(0)[cdim] : 1;
    switch (input.type()) {
      case DALIDataType::DALI_UINT8:
        switch (channels) {
          case 1: return nvcv::FMT_U8;
          case 3: return nvcv::FMT_RGB8;
          case 4: return nvcv::FMT_RGBA8;
        }
      case DALIDataType::DALI_UINT16:
        return nvcv::FMT_U16;
      case DALIDataType::DALI_FLOAT:
        switch (channels) {
          case 1: return nvcv::FMT_F32;
          case 3: return nvcv::FMT_RGBf32;
          case 4: return nvcv::FMT_RGBAf32;
        }
      default:
        DALI_FAIL(make_string("Unsupported input type in MedianBlur operator: ",
                              input.type_info().name(),
                              ". Supported types are: UINT8, UINT16 and FLOAT."));
    }
  }

  void ValidateChannels(const TensorList<GPUBackend> &input, int cdim) const {
    assert(cdim >= 0);
    if (input.num_samples() == 0) return;
    auto channels = input.tensor_shape(0)[cdim];
    DALI_ENFORCE(channels == 1 || channels == 3 || channels == 4,
                 make_string("MedianBlur operator suupports the following number of channels: "
                             "1, 3, 4. The provided input has ", channels, " channels."));

    DALI_ENFORCE(input.type() != DALIDataType::DALI_UINT16 || channels == 1,
                 make_string("MedianBlur operator supports only single-channel images of type "
                             "uint16. Provided image with ", channels, " channels."));
    for (int64_t idx = 1; idx < input.num_samples(); ++idx) {
      DALI_ENFORCE(input.tensor_shape(idx)[cdim] == channels,
                   make_string("MedianBlur operator requires all the samples to have the same "
                               "number of channels. In the provided input, the sample at index 0 "
                               "has ", channels, " channels and the sample at index ", idx,
                               " has ", channels, " channels."));
    }
  }

  void GetImages(nvcv::ImageBatchVarShape &images,
                 int outer_dims,
                 const int64_t *byte_strides,
                 const int64_t *shape,
                 int64_t offset,
                 void *data,
                 int64_t sample_id,
                 std::vector<int64_t> *batch_map) {
    if (outer_dims == 0) {
      nvcv::ImageDataStridedCuda::Buffer buf{};
      buf.numPlanes = 1;
      buf.planes[0].basePtr = static_cast<NVCVByte*>(data) + offset;
      buf.planes[0].rowStride = byte_strides[0];
      buf.planes[0].height = shape[0];
      buf.planes[0].width  = shape[1];
      nvcv::ImageDataStridedCuda img_data(nvcv_format_, buf);
      images.pushBack(nvcv::ImageWrapData(img_data));
      if (batch_map) batch_map->push_back(sample_id);
    } else {
      int extent = shape[0];
      for (int i = 0; i < extent; i++) {
        GetImages(images,
                  outer_dims - 1,
                  byte_strides + 1,
                  shape + 1,
                  offset + byte_strides[0] * i,
                  data,
                  sample_id,
                  batch_map);
      }
    }
  }

  void GetImages(nvcv::ImageBatchVarShape &images,
                 const ConstSampleView<GPUBackend> &sample,
                 bool channels_last,
                 int64_t sample_id,
                 std::vector<int64_t> *batch_map) {
    const auto &shape = sample.shape();
    int channels = channels_last ? shape[shape.sample_dim() - 1] : 0;

    size_t type_size = TypeTable::GetTypeInfo(sample.type()).size();
    SmallVector<int64_t, 6> byte_strides;
    kernels::CalcStrides(byte_strides, shape);
    for (auto &s : byte_strides)
      s *= type_size;

    int image_dims = channels_last ? 3 : 2;
    int ndim = sample.shape().sample_dim();

    void *data = const_cast<void*>(sample.raw_data());
    GetImages(images, ndim - image_dims, byte_strides.data(), shape.data(), 0,
              data, sample_id, batch_map);
  }


  void GetImages(nvcv::ImageBatchVarShape &images, const TensorList<GPUBackend> &tl,
                 const TensorLayout &layout, std::vector<int64_t> *batch_map = nullptr) {
    images.clear();
    if (batch_map) {
      batch_map->clear();
    }
    auto &shape = tl.shape();
    int nsamples = tl.num_samples();
    int ndim = tl.sample_dim();
    int cdim = layout.find('C');
    bool channels_last = cdim == ndim - 1;

    for (int64_t i = 0; i < nsamples; i++) {
      GetImages(images, tl[i], channels_last, i, batch_map);
    }
  }


 private:
  ArgValue<int, 1> ksize_arg_{"window_size", spec_};
  DeviceBuffer<int32_t> ksize_buffer_{};
  nvcv::Tensor ksize_tensor_{};
  int effective_batch_size_ = 0;
  std::optional<cvcuda::MedianBlur> impl_;
  nvcv::ImageBatchVarShape input_batch_{};
  nvcv::ImageBatchVarShape output_batch_{};
  std::vector<int64_t> batch_map_;  //< index of a sample the image on given position comes from
  nvcv::ImageFormat nvcv_format_{};
};

DALI_REGISTER_OPERATOR(experimental__MedianBlur, MedianBlur, GPU);

}  // namespace dali
