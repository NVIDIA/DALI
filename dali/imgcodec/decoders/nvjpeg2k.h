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

#ifndef DALI_IMGCODEC_DECODERS_NVJPEG2K_H_
#define DALI_IMGCODEC_DECODERS_NVJPEG2K_H_

#include <nvjpeg.h>
#include <memory>
#include <vector>
#include "dali/imgcodec/image_decoder.h"
#include "dali/imgcodec/decoders/decoder_parallel_impl.h"
#include "dali/imgcodec/decoders/nvjpeg/nvjpeg2k_helper.h"

namespace dali {
namespace imgcodec {

/**
 * @brief Decoder interface for nvjpeg2k library
 */
class DLL_PUBLIC NvJpeg2kDecoderInstance : public ImageDecoderImpl {
 public:
  NvJpeg2kDecoderInstance(int device_id, ThreadPool *tp)
  : ImageDecoderImpl(device_id, tp) {}

  using ImageDecoderImpl::Decode;
  DecodeResult Decode(SampleView<GPUBackend> out,
                      ImageSource *in,
                      DecodeParams opts,
                      const ROI &roi) override {
    return Decode({out}, {in}, opts, {roi});
  }

  std::vector<DecodeResult> Decode(span<SampleView<GPUBackend>> out,
                                   cspan<ImageSource *> in,
                                   DecodeParams opts,
                                   cspan<ROI> rois) override;

 private:
  struct ProcessingInfo {
    uint8_t bpp;
    TensorShape<> shape;
    int req_nchannels;
    bool needs_processing;
    DALIDataType pixel_type;
    size_t pixel_size;
    int64_t pixels_count;
    int64_t comp_size;
  };

  bool ParseJpeg2000Info(int id, ProcessingInfo *info, ImageSource *in, DecodeParams opts);
  bool DecodeImpl(int id, ProcessingInfo *info, ImageSource *in, uint8_t *out, DecodeParams opts);
  bool ConvertData(int id, ProcessingInfo *info, uint8_t *in, uint8_t *out, DecodeParams opts);

  NvJpeg2kHandle nvjpeg2k_handle_;
  NvJpeg2kDecodeState nvjpeg2k_decoder_;
  std::vector<NvJpeg2kStream> nvjpeg2k_streams_;
  DeviceBuffer<uint8_t> nvjpeg2k_intermediate_buffer_;
  cudaStream_t nvjpeg2k_cu_stream_;
  cudaEvent_t nvjpeg2k_decode_event_;
  nvjpeg2kDeviceAllocator_t nvjpeg2k_dev_alloc_;
  nvjpeg2kPinnedAllocator_t nvjpeg2k_pin_alloc_;
};

class NvJpeg2kDecoder : public ImageDecoder {
 public:
  ImageDecoderProperties GetProperties() const override {
    static const auto props = []() {
      ImageDecoderProperties props;
      props.supported_input_kinds = InputKind::HostMemory;
      props.supports_partial_decoding = false;  // roi support requires decoding the whole file
      props.fallback = true;
      return props;
    }();
    return props;
  }

  bool IsSupported(int device_id) const override {
    return device_id >= 0;
  }

  std::shared_ptr<ImageDecoderInstance> Create(int device_id, ThreadPool &tp) const override {
    return std::make_shared<NvJpeg2kDecoderInstance>(device_id, &tp);
  }
};

}  // namespace imgcodec
}  // namespace dali

#endif  // DALI_IMGCODEC_DECODERS_NVJPEG2K_H_
