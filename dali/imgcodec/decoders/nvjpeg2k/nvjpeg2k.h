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

#ifndef DALI_IMGCODEC_DECODERS_NVJPEG2K_NVJPEG2K_H_
#define DALI_IMGCODEC_DECODERS_NVJPEG2K_NVJPEG2K_H_

#include <nvjpeg.h>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "dali/imgcodec/decoders/decoder_parallel_impl.h"
#include "dali/imgcodec/decoders/nvjpeg2k/nvjpeg2k_helper.h"
#include "dali/core/cuda_stream_pool.h"
#include "dali/core/cuda_event.h"

namespace dali {
namespace imgcodec {

/**
 * @brief Decoder interface for nvjpeg2k library
 */
class DLL_PUBLIC NvJpeg2000DecoderInstance : public BatchParallelDecoderImpl {
 public:
  explicit NvJpeg2000DecoderInstance(int device_id, const std::map<std::string, std::any> &params);
  ~NvJpeg2000DecoderInstance();

  DecodeResult DecodeImplTask(int thread_idx,
                              cudaStream_t stream,
                              SampleView<GPUBackend> out,
                              ImageSource *in,
                              DecodeParams opts,
                              const ROI &roi) override;

  FutureDecodeResults ScheduleDecode(DecodeContext ctx,
                                     span<SampleView<GPUBackend>> out,
                                     cspan<ImageSource *> in,
                                     DecodeParams opts,
                                     cspan<ROI> rois = {}) override {
    ctx.tp = tp_.get();
    return BatchParallelDecoderImpl::ScheduleDecode(std::move(ctx), out, in, std::move(opts), rois);
  }

  bool SetParam(const char *name, const std::any &value) override {
    if (strcmp(name, "nvjpeg2k_device_memory_padding") == 0) {
      nvjpeg2k_device_memory_padding_ = std::any_cast<size_t>(value);
      return true;
    } else if (strcmp(name, "nvjpeg2k_host_memory_padding") == 0) {
      nvjpeg2k_host_memory_padding_ = std::any_cast<size_t>(value);
      return true;
    } else {
      return false;
    }
  }

  std::any GetParam(const char *name) const override {
    if (strcmp(name, "nvjpeg2k_device_memory_padding") == 0) {
      return nvjpeg2k_device_memory_padding_;
    } else if (strcmp(name, "nvjpeg2k_host_memory_padding") == 0) {
      return nvjpeg2k_host_memory_padding_;
    } else {
      return {};
    }
  }

  struct PerThreadResources {
    PerThreadResources() = default;
    PerThreadResources(const NvJpeg2kHandle &nvjpeg2k_handle,
                       size_t device_memory_padding, int device_id)
    : nvjpeg2k_decode_state(nvjpeg2k_handle)
    , nvjpeg2k_stream(NvJpeg2kStream::Create())
    , decode_event(CUDAEvent::Create(device_id))
    , cuda_stream(CUDAStreamPool::instance().Get(device_id)) {
    }

    NvJpeg2kDecodeState nvjpeg2k_decode_state;
    NvJpeg2kStream nvjpeg2k_stream;
    CUDAEvent decode_event;
    CUDAStreamLease cuda_stream;
    NvJpeg2kDecodeParams params;
  };

  /**
   * @brief Context for image decoding, one per picture.
   */
  struct Context {
    Context(DecodeParams opts, const ROI &roi, PerThreadResources &res)
    : opts(opts)
    , roi(roi)
    , nvjpeg2k_decode_state(res.nvjpeg2k_decode_state)
    , nvjpeg2k_stream(res.nvjpeg2k_stream)
    , decode_event(res.decode_event)
    , cuda_stream(res.cuda_stream)
    , params(res.params) {}

    nvjpeg2kImageInfo_t image_info;
    /** @brief Bits per pixel */
    uint8_t bpp = 0;
    /** @brief Data type nvJPEG2000 decodes into, either uint8 or uint16 */
    DALIDataType pixel_type = DALI_NO_TYPE;
    TensorShape<> shape;
    TensorShape<> orig_shape;

    DecodeParams opts;
    const ROI &roi;

    const NvJpeg2kDecodeState &nvjpeg2k_decode_state;
    const NvJpeg2kStream &nvjpeg2k_stream;
    const CUDAEvent &decode_event;
    const CUDAStreamLease &cuda_stream;
    const NvJpeg2kDecodeParams& params;
  };

 private:
  bool ParseJpeg2000Info(ImageSource *in, Context &ctx);
  bool DecodeJpeg2000(ImageSource *in, void *out, const TensorShape<> &sh, const Context &ctx,
                      const ROI &roi = {});

  // TODO(staniewzki): remove default values
  size_t nvjpeg2k_device_memory_padding_ = 256;
  size_t nvjpeg2k_host_memory_padding_ = 256;

  std::unique_ptr<ThreadPool> tp_;
  NvJpeg2kHandle nvjpeg2k_handle_{};
  nvjpeg2kDeviceAllocator_t nvjpeg2k_dev_alloc_;
  nvjpeg2kPinnedAllocator_t nvjpeg2k_pin_alloc_;
  std::vector<PerThreadResources> per_thread_resources_;
};

class NvJpeg2000DecoderFactory : public ImageDecoderFactory {
 public:
  ImageDecoderProperties GetProperties() const override {
    static const auto props = []() {
      ImageDecoderProperties props;
      props.supported_input_kinds = InputKind::HostMemory;
      props.supports_partial_decoding = false;  // roi support requires decoding the whole file
      props.gpu_output = true;
      props.fallback = true;
      return props;
    }();
    return props;
  }

  bool IsSupported(int device_id) const override {
    return device_id >= 0;
  }

  std::shared_ptr<ImageDecoderInstance> Create(
        int device_id, const std::map<std::string, std::any> &params = {}) const override {
    return std::make_shared<NvJpeg2000DecoderInstance>(device_id, params);
  }
};

}  // namespace imgcodec
}  // namespace dali

#endif  // DALI_IMGCODEC_DECODERS_NVJPEG2K_NVJPEG2K_H_
