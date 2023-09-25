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

#ifndef DALI_IMGCODEC_DECODERS_NVJPEG_NVJPEG_H_
#define DALI_IMGCODEC_DECODERS_NVJPEG_NVJPEG_H_

#include <nvjpeg.h>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "dali/core/cuda_event.h"
#include "dali/core/cuda_stream_pool.h"
#include "dali/imgcodec/decoders/decoder_parallel_impl.h"
#include "dali/imgcodec/decoders/nvjpeg/nvjpeg_memory.h"
#include "dali/pipeline/data/buffer.h"

namespace dali {
namespace imgcodec {

class DLL_PUBLIC NvJpegDecoderInstance : public BatchParallelDecoderImpl {
 public:
  explicit NvJpegDecoderInstance(int device_id, const std::map<std::string, std::any> &params);

  using BatchParallelDecoderImpl::CanDecode;
  bool CanDecode(DecodeContext ctx, ImageSource *in, DecodeParams opts, const ROI &roi) override;

  // NvjpegDecoderInstance has to operate on its own thread pool instead of the
  // one passed by the DecodeContext. Overriding thread pool pointer caried in
  // the context argument of this variant of
  // BatchParallelDecoderImpl::ScheduleDecode is enough to cover all the Decode
  // and ScheduleDecode functions. This is because all that functions
  // eventually call this variant of ScheduleDecode.
  FutureDecodeResults ScheduleDecode(DecodeContext ctx,
                                     span<SampleView<GPUBackend>> out,
                                     cspan<ImageSource *> in,
                                     DecodeParams opts,
                                     cspan<ROI> rois = {}) override {
    ctx.tp = tp_.get();
    return BatchParallelDecoderImpl::ScheduleDecode(ctx, out, in, opts, rois);
  }

  DecodeResult DecodeImplTask(int thread_idx,
                              cudaStream_t stream,
                              SampleView<GPUBackend> out,
                              ImageSource *in,
                              DecodeParams opts,
                              const ROI &roi) override;
  ~NvJpegDecoderInstance();

  bool SetParam(const char *name, const std::any &value) override;
  std::any GetParam(const char *name) const override;

 private:
  nvjpegHandle_t nvjpeg_handle_;

  int num_threads_ = 1;
  size_t device_memory_padding_ = 0;
  size_t host_memory_padding_ = 0;
  nvjpegDevAllocator_t device_allocator_;
  nvjpegPinnedAllocator_t pinned_allocator_;
  bool use_jpeg_fancy_upsampling_ = false;

  struct DecoderData {
    nvjpegJpegDecoder_t decoder = nullptr;
    nvjpegJpegState_t state = nullptr;
  };

  struct PerThreadResources {
    DecoderData decoder_data;

    nvjpegBufferDevice_t device_buffer;
    std::array<nvjpegBufferPinned_t, 2> pinned_buffers;
    int which_buffer = 0;
    int current_buffer_idx() const { return which_buffer; }
    int next_buffer_idx() const { return 1 - which_buffer; }
    void swap_buffers() { which_buffer = 1 - which_buffer; }

    nvjpegBufferPinned_t pinned_buffer() const {
      return pinned_buffers[which_buffer];
    }
    CUDAEvent &decode_event() {
      return decode_events[which_buffer];
    }

    nvjpegJpegStream_t jpeg_stream;
    CUDAStreamLease stream;
    std::array<CUDAEvent, 2> decode_events;
    nvjpegDecodeParams_t params;

    Buffer<GPUBackend> intermediate_buffer;

    PerThreadResources(nvjpegHandle_t, nvjpegDevAllocator_t*, nvjpegPinnedAllocator_t*,
                       int device_id);
    PerThreadResources(PerThreadResources&&);
    ~PerThreadResources();
  };

  std::vector<PerThreadResources> resources_;
  std::unique_ptr<ThreadPool> tp_;

  struct DecodingContext {
    PerThreadResources& resources;

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
    props.gpu_output = true;
    props.fallback = true;

    return props;
  }

  bool IsSupported(int device_id) const override {
    return device_id >= 0;
  }

  std::shared_ptr<ImageDecoderInstance>
  Create(int device_id, const std::map<std::string, std::any> &params = {}) const override {
    return std::make_shared<NvJpegDecoderInstance>(device_id, params);
  }
};

}  // namespace imgcodec
}  // namespace dali

#endif  // DALI_IMGCODEC_DECODERS_NVJPEG_NVJPEG_H_
