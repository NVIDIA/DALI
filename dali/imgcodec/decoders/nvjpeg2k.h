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

#define NVJPEG2K_ENABLED true  // using dirty hack for now

#include <nvjpeg.h>
#include <memory>
#include <vector>
#include "dali/imgcodec/decoders/decoder_parallel_impl.h"
#include "dali/imgcodec/decoders/nvjpeg/nvjpeg2k_helper.h"
#include "dali/imgcodec/decoders/nvjpeg/nvjpeg_memory.h"
#include "dali/core/dev_buffer.h"
#include "dali/core/cuda_stream_pool.h"
#include "dali/core/cuda_event.h"

namespace dali {
namespace imgcodec {

/**
 * @brief Decoder interface for nvjpeg2k library
 */
class DLL_PUBLIC NvJpeg2000DecoderInstance : public BatchParallelDecoderImpl {
 public:
  NvJpeg2000DecoderInstance(int device_id, ThreadPool *tp);

  using BatchParallelDecoderImpl::CanDecode;
  bool CanDecode(ImageSource *in, DecodeParams opts, const ROI &roi) {
    return !roi && (opts.dtype == DALI_UINT8);
  }

  using BatchParallelDecoderImpl::DecodeImplTask;
  DecodeResult DecodeImplTask(int thread_idx,
                              cudaStream_t stream,
                              SampleView<GPUBackend> out,
                              ImageSource *in,
                              DecodeParams opts,
                              const ROI &roi) override;

  void SetParam(const char *name, const any &value) {
    if (strcmp(name, "nvjpeg2k_device_memory_padding") == 0) {
      nvjpeg2k_device_memory_padding_ = any_cast<size_t>(value);
    } else if (strcmp(name, "nvjpeg2k_host_memory_padding") == 0) {
      nvjpeg2k_host_memory_padding_ = any_cast<size_t>(value);
    }
  }

  any GetParam(const char *name) const override {
    if (strcmp(name, "nvjpeg2k_device_memory_padding") == 0) {
      return nvjpeg2k_device_memory_padding_;
    } else if (strcmp(name, "nvjpeg2k_host_memory_padding") == 0) {
      return nvjpeg2k_host_memory_padding_;
    } else {
      DALI_FAIL("Unrecognized param name");
    }
  }

 private:
  struct Context {
    uint8_t bpp;
    TensorShape<> shape;
    int req_nchannels;
    bool needs_processing;
    DALIDataType pixel_type;
    size_t pixel_size;
    int64_t pixels_count;
    int64_t comp_size;

    NvJpeg2kDecodeState *nvjpeg2k_decode_state;
    NvJpeg2kStream *nvjpeg2k_stream;
    CUDAEvent *decode_event;
    CUDAStreamLease *cuda_stream;
  };

  size_t nvjpeg2k_device_memory_padding_ = 8;
  size_t nvjpeg2k_host_memory_padding_ = 8;

  bool ParseJpeg2000Info(ImageSource *in, DecodeParams opts, Context *ctx);
  bool DecodeJpeg2000(ImageSource *in, uint8_t *out, DecodeParams opts, Context *ctx);
  bool ConvertData(void *in, uint8_t *out, DecodeParams opts, Context *ctx);

  NvJpeg2kHandle nvjpeg2k_handle_{};
  nvjpeg2kDeviceAllocator_t nvjpeg2k_dev_alloc_;
  nvjpeg2kPinnedAllocator_t nvjpeg2k_pin_alloc_;

  std::vector<NvJpeg2kDecodeState> nvjpeg2k_decode_states_;
  std::vector<DeviceBuffer<uint8_t>> intermediate_buffers_;
  std::vector<NvJpeg2kStream> nvjpeg2k_streams_;
  std::vector<CUDAEvent> decode_events_;
  std::vector<CUDAStreamLease> cuda_streams_;
};

class NvJpeg2000Decoder : public ImageDecoder {
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
    return std::make_shared<NvJpeg2000DecoderInstance>(device_id, &tp);
  }
};

}  // namespace imgcodec
}  // namespace dali

#endif  // DALI_IMGCODEC_DECODERS_NVJPEG2K_H_
