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

#ifndef DALI_IMGCODEC_DECODERS_NVJPEG_H_
#define DALI_IMGCODEC_DECODERS_NVJPEG_H_

#include <nvjpeg.h>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "dali/core/cuda_event.h"
#include "dali/core/cuda_stream_pool.h"
#include "dali/core/dev_buffer.h"
#include "dali/imgcodec/decoders/decoder_parallel_impl.h"
#include "dali/imgcodec/decoders/nvjpeg/nvjpeg_memory.h"

namespace dali {
namespace imgcodec {

class DLL_PUBLIC NvJpegDecoderInstance : public BatchParallelDecoderImpl {
 public:
  explicit NvJpegDecoderInstance(int device_id, const std::map<std::string, any> &params);

  DecodeResult DecodeImplTask(int thread_idx,
                              cudaStream_t stream,
                              SampleView<GPUBackend> out,
                              ImageSource *in,
                              DecodeParams opts,
                              const ROI &roi) override;
  ~NvJpegDecoderInstance();

  bool SetParam(const char *name, const any &value) override;
  any GetParam(const char *name) const override;

 private:
  nvjpegHandle_t nvjpeg_handle_;

  size_t device_memory_padding_ = 0;
  size_t host_memory_padding_ = 0;
  nvjpegDevAllocator_t device_allocator_;
  nvjpegPinnedAllocator_t pinned_allocator_;

  struct DecoderData {
    nvjpegJpegDecoder_t decoder = nullptr;
    nvjpegJpegState_t state = nullptr;
  };

  struct PerThreadResources {
    DecoderData decoder_data;

    nvjpegBufferDevice_t device_buffer;
    nvjpegBufferPinned_t pinned_buffer;

    nvjpegJpegStream_t jpeg_stream;
    CUDAStreamLease stream;
    CUDAEvent decode_event;

    PerThreadResources(nvjpegHandle_t, nvjpegDevAllocator_t*, nvjpegPinnedAllocator_t*,
                       int device_id);
    PerThreadResources(PerThreadResources&&);
    ~PerThreadResources();
  };

  std::vector<PerThreadResources> resources_;
  std::unique_ptr<ThreadPool> tp_;

  struct DecodingContext {
    PerThreadResources& resources;

    nvjpegDecodeParams_t params;
    nvjpegChromaSubsampling_t subsampling;

    TensorShape<> shape;
  };

  void ParseJpegSample(ImageSource& in, DecodeParams opts, DecodingContext& ctx);
  void DecodeJpegSample(ImageSource& in, uint8_t *out, DecodeParams opts, DecodingContext &ctx);
};

class NvJpegDecoderFactory : public ImageDecoderFactory {
 public:
  ImageDecoderProperties GetProperties() const override {
    ImageDecoderProperties props;
    props.supports_partial_decoding = false;
    props.supported_input_kinds = InputKind::HostMemory;
    props.fallback = true;

    return props;
  }

  bool IsSupported(int device_id) const override {
    return device_id >= 0;
  }

  std::shared_ptr<ImageDecoderInstance>
  Create(int device_id, const std::map<std::string, any> &params = {}) const override {
    return std::make_shared<NvJpegDecoderInstance>(device_id, params);
  }
};

}  // namespace imgcodec
}  // namespace dali

#endif  // DALI_IMGCODEC_DECODERS_NVJPEG_H_
