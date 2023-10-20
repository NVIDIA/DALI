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
#include "dali/core/version_util.h"
#include "dali/imgcodec/decoders/decoder_batched_api_impl.h"
#include "dali/imgcodec/decoders/nvjpeg/nvjpeg_memory.h"
#include "dali/kernels/dynamic_scratchpad.h"
#include "dali/pipeline/data/buffer.h"

namespace dali {
namespace imgcodec {

class DLL_PUBLIC NvJpegLosslessDecoderInstance : public BatchedApiDecoderImpl {
 public:
  explicit NvJpegLosslessDecoderInstance(int device_id,
                                         const std::map<std::string, std::any> &params);
  ~NvJpegLosslessDecoderInstance();

  using ImageDecoderImpl::CanDecode;
  bool CanDecode(DecodeContext ctx, ImageSource *in, DecodeParams opts, const ROI &roi) override;

  FutureDecodeResults ScheduleDecode(DecodeContext ctx, span<SampleView<GPUBackend>> out,
                                     cspan<ImageSource *> in, DecodeParams opts,
                                     cspan<ROI> rois = {}) override;

 private:
  // Parses encoded streams and populates SampleMeta, and batch_sz_
  void Parse(DecodeResultsPromise &promise, DecodeContext ctx, cspan<ImageSource *> in,
             DecodeParams opts, cspan<ROI> rois);

  // Invokes nvJPEG decoding (sample_meta_ and batch_sz_ to be populated)
  void RunDecode(kernels::DynamicScratchpad &s, DecodeContext ctx, span<SampleView<GPUBackend>> out,
                 cspan<ImageSource *> in, DecodeParams opts, cspan<ROI> rois = {});

  void Postprocess(DecodeResultsPromise &promise, DecodeContext ctx,
                   span<SampleView<GPUBackend>> out, DecodeParams opts, cspan<ROI> rois);

  nvjpegHandle_t nvjpeg_handle_;

  struct PerThreadResources {
    explicit PerThreadResources(nvjpegHandle_t handle);
    PerThreadResources(PerThreadResources&& other);
    ~PerThreadResources();
    nvjpegJpegStream_t jpeg_stream;
  };
  std::vector<PerThreadResources> per_thread_resources_;
  CUDAEvent event_;
  nvjpegJpegState_t state_;

  int batch_sz_ = 0;  // number of samples to be decoded by nvJPEG
  struct SampleMeta {
    bool can_decode;
    int idx_in_batch;  // only relevant if can_decode == true
    bool needs_processing;
    Orientation orientation;
    float dyn_range_multiplier;
  };
  std::vector<SampleMeta> sample_meta_;
  std::vector<const unsigned char*> encoded_;
  std::vector<size_t> encoded_len_;
  std::vector<nvjpegImage_t> decoded_;
  bool is_initialized_ = false;
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
    if (device_id < 0)
      return false;

    cudaDeviceProp props;
    CUDA_CALL(cudaGetDeviceProperties(&props, device_id));

    int major = -1;
    int minor = -1;
    int patch = -1;
    GetVersionProperty(nvjpegGetProperty, &major, MAJOR_VERSION, NVJPEG_STATUS_SUCCESS);
    GetVersionProperty(nvjpegGetProperty, &minor, MINOR_VERSION, NVJPEG_STATUS_SUCCESS);
    GetVersionProperty(nvjpegGetProperty, &patch, PATCH_LEVEL, NVJPEG_STATUS_SUCCESS);
    auto NvJpegVersion = MakeVersionNumber(major, minor, patch);

    return props.major >= 6 && NvJpegVersion >= 12020;
  }

  std::shared_ptr<ImageDecoderInstance>
  Create(int device_id, const std::map<std::string, std::any> &params = {}) const override {
    assert(IsSupported(device_id));
    return std::make_shared<NvJpegLosslessDecoderInstance>(device_id, params);
  }
};

}  // namespace imgcodec
}  // namespace dali

#endif  // DALI_IMGCODEC_DECODERS_NVJPEG_LOSSLESS_NVJPEG_LOSSLESS_H_
