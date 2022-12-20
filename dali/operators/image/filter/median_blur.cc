// Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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
#include <nvcv/ImageBatch.hpp>
#include <operators/OpMedianBlur.hpp>
#include "dali/core/static_switch.h"
#include "dali/kernels/common/utils.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/operator/arg_helper.h"

#define MEDIAN_BLUR_TYPES uint8_t, uint16_t, int, float

namespace dali {

namespace nvcv = nv::cv;
namespace nvcvop = nv::cvop;

DALI_SCHEMA(MedianBlur)
  .NumInput(1)
  .NumOutput(1)
  .InputLayout({"HWC", "FHWC", "CHW", "FCHW"})
  .AddOptionalArg("window_size",
    "The size of the window over which the smoothing is performed",
    std::vector<int>({3, 3}),
    true);


class MedianBlur : public Operator<GPUBackend> {
 public:
  MedianBlur(const OpSpec &spec)
  : Operator<GPUBackend>(spec) {
    if (ksize_.HasExplicitConstant()) {
      int ksize = *ksize_[0].data;
      DALI_ENFORCE(ksize > 1 && (ksize&1),
                   "The window size must be an odd integer greater than one.");
    }
  }

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) override {
    auto &in = ws.Input<GPUBackend>(0);
    output_desc.resize(1);
    output_desc[0].type = in.type();
    output_desc[0].shape = in.shape();
    return true;
  }

  void GetImages(int outer_dims, int64_t offset,
                 const int64_t *byte_strides,
                 void *data,
                 const TensorShape<> &shape,
                 nvcv::ImageFormat format) {
    if (outer_dims == 0) {
      nvcv::ImageDataPitchDevice::Buffer buf;
      buf.numPlanes = 1;
      buf.planes[0].buffer = static_cast<char*>(data) + offset;
      buf.planes[0].pitchBytes = byte_strides[0];
      buf.planes[0].height = shape[d];
      buf.planes[0].width  = shape[d + 1];
    } else {
      int extent = shape[d];
      for (int i = 0; i < extent; i++) {
        GetImages(outer_dims - 1,
                  byte_strides + 1,
                  offset + byte_strides[0] * i,
                  data, shape, format);
      }
    }
  }

  void GetImages(const ConstSampleView<GPUBackend> &sample,
                 bool channels_last) {
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
    GetImages(ndim - image_dims, byte_strides.data(), data, ToNVCVFormat(pos, ));
  }


  int GetImages(const TensorList<GPUBackend> &tl, const TensorLayout &layout) {
    auto &shape = tl.shape();
    int nsamples = tl.num_samples();
    int ndim = tl.sample_ndim();
    int cdim = layout.find('C');
    bool channels_last = cdim == ndim - 1;
    
    images_.clear();
    for (int i = 0; i < num_samples; i++) {
      GetImages(tl[i], channels_last);
    }
    return images_.size();
  }

  void RunImpl(Workspace &ws) {
    auto &input = ws.Input<CPUBackend>(0);
    int effective_batch_size = GetImages(input.shape(), input.GetLayout());
    if (effective_batch_size > effective_batch_size_) {
      effective_batch_size_ = std::max(2*effective_batch_size_, effective_batch_size);
      impl_.reset();
      batch_.reset();

      impl_ = nvcvop::MedianBlur(effective_batch_size_);
      batch_.emplace(effective_batch_size_);
    } else {
      batch_->clear();
    }
  }

  bool CanInferOutputs() const override {
    return true;
  }

private:
  ArgValue<int> ksize_{"window_size", spec_};

  int effective_batch_size_ = 0;
  std::optional<nvcvop::MedianBlur> impl_;
  std::optional<nvcv::ImageBatchVarShape> batch_;
  std::vector<std::unique_ptr<nvcv::ImageWrapData>> images_;
};

DALI_REGISTER_OPERATOR(experimental__MedianBlur, MedianBlur, GPU);

}  // namespace dali
