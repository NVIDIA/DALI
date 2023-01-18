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

#ifndef DALI_IMGCODEC_DECODERS_NVJPEG_LOSSLESS_NVJPEG_LOSSLESS_H_
#define DALI_IMGCODEC_DECODERS_NVJPEG_LOSSLESS_NVJPEG_LOSSLESS_H_

#include <nvjpeg.h>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "dali/core/cuda_event.h"
#include "dali/core/cuda_stream_pool.h"
#include "dali/imgcodec/decoders/decoder_batched_api_impl.h"
#include "dali/imgcodec/decoders/nvjpeg/nvjpeg_memory.h"
#include "dali/pipeline/data/buffer.h"

namespace dali {
namespace imgcodec {

class DLL_PUBLIC NvJpegLosslessDecoderInstance : public BatchedApiDecoderImpl {
 public:
  explicit NvJpegLosslessDecoderInstance(int device_id, const std::map<std::string, any> &params);
  ~NvJpegLosslessDecoderInstance();

  using ImageDecoderImpl::CanDecode;
  bool CanDecode(DecodeContext ctx, ImageSource *in, DecodeParams opts, const ROI &roi) override;

  FutureDecodeResults ScheduleDecode(DecodeContext ctx, span<SampleView<GPUBackend>> out,
                                     cspan<ImageSource *> in, DecodeParams opts,
                                     cspan<ROI> rois = {}) override;

 private:
  void Postprocess(DecodeResultsPromise &promise, DecodeContext ctx,
                   span<SampleView<GPUBackend>> out, DecodeParams opts, cspan<ROI> rois);

  nvjpegHandle_t nvjpeg_handle_;
  nvjpegJpegStream_t jpeg_stream_;
  CUDAEvent event_;
  nvjpegJpegState_t state_;

  struct SampleMeta {
    bool needs_processing;
    Orientation orientation;
    float dyn_range_multiplier;
  };

  std::vector<SampleMeta> sample_meta_;
  std::vector<const unsigned char*> encoded_;
  std::vector<size_t> encoded_len_;
  std::vector<nvjpegImage_t> decoded_;
};

class NvJpegLosslessDecoderFactory : public ImageDecoderFactory {
 public:
  ImageDecoderProperties GetProperties() const override {
    ImageDecoderProperties props;
    props.supports_partial_decoding = false;
    props.supported_input_kinds = InputKind::HostMemory;
    props.gpu_output = true;
    props.fallback = true;
    return props;
  }

  bool IsSupported(int device_id) const override {
    return device_id >= 0;
  }

  std::shared_ptr<ImageDecoderInstance>
  Create(int device_id, const std::map<std::string, any> &params = {}) const override {
    return std::make_shared<NvJpegLosslessDecoderInstance>(device_id, params);
  }
};

}  // namespace imgcodec
}  // namespace dali

#endif  // DALI_IMGCODEC_DECODERS_NVJPEG_LOSSLESS_NVJPEG_LOSSLESS_H_
