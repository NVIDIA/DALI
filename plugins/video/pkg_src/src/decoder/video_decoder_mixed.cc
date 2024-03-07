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

#include "decoder/video_decoder_mixed.h"
#include "dali/core/tensor_shape.h"

namespace dali_video {

class MemoryVideoFile : public FFmpegDemuxer::DataProvider {
 public:
  MemoryVideoFile(const void *data, int64_t size)
    : data_(static_cast<const uint8_t*>(data)), size_(size), position_(0) {}


  int GetData(uint8_t *buffer, int buffer_size) override {
    int left_in_file = size_ - position_;
    if (left_in_file == 0) {
      return AVERROR_EOF;
    }

    int to_read = std::min(left_in_file, buffer_size);
    std::copy(data_ + position_, data_ + position_ + to_read, buffer);
    position_ += to_read;
    return to_read;
  }

 private:
  const uint8_t *data_;
  const int64_t size_;
  int64_t position_;
};

bool VideoDecoderMixed::SetupImpl(
  std::vector<dali::OutputDesc> &output_desc, const dali::Workspace &ws) {
  ValidateInput(ws);
  const auto &input = ws.Input<dali::CPUBackend>(0);
  int batch_size = input.num_samples();
  auto stream = ws.stream();

  samples_.resize(batch_size);
  dali::TensorListShape<> sh(batch_size, 4);
  for (int i = 0; i < batch_size; i++) {
    auto& sample = samples_[i];
    sample.data_provider_ = std::make_unique<MemoryVideoFile>(input.raw_tensor(i), input[i].shape().num_elements());
    sample.demuxer_ = std::make_unique<FFmpegDemuxer>(sample.data_provider_.get());
    sample.current_packet_ = std::make_unique<PacketData>();
    std::cout << "Sample #" << i << " {10x" << sample.demuxer_->GetHeight() << "x" << sample.demuxer_->GetWidth() << "x" << 3 << "}\n";
    sh.set_tensor_shape(i, dali::TensorShape<>(10, sample.demuxer_->GetHeight(), sample.demuxer_->GetWidth(), 3));
  }
  output_desc.resize(1);
  output_desc[0].shape = sh;
  output_desc[0].type = dali::DALI_UINT8;
  return true;
}

void VideoDecoderMixed::Run(dali::Workspace &ws) {
  auto &output = ws.Output<dali::GPUBackend>(0);
  const auto &input = ws.Input<dali::CPUBackend>(0);
  int batch_size = input.num_samples();
  int s = 0;

  cuInit(0);
  int nGpu = 0;
  cuDeviceGetCount(&nGpu);

  bool m_bDestroyContext = false;
  CUcontext cuContext = nullptr;
  CUstream cuStream = nullptr;

  for (int i = 0; i < batch_size; i++) {
    cuCtxGetCurrent(&cuContext);
    if(!cuContext) {
      createCudaContext(&cuContext, device_id_, 0);
      m_bDestroyContext = true;
    }
    cuCtxPopCurrent(&cuContext);
  }

  if(!cuContext)
    throw std::runtime_error("Failed to create a cuda context");

  cuCtxPushCurrent(cuContext);
  cuStreamCreate(&cuStream, 0);
  cuCtxPopCurrent(nullptr);

  cuCtxPushCurrent(cuContext);
  
  for (int i = 0; i < batch_size; i++) {
    auto& sample = samples_[i];
    sample.decoder_ = std::make_unique<NvDecoder>(
        cuStream, cuContext, true, FFmpeg2NvCodecId(sample.demuxer_->GetVideoCodec()), false,
        false /*_enableasyncallocations*/, false);


    uint8_t* pVideo = NULL;
    int nVideoBytes = 0;
    while (sample.demuxer_->Demux(&pVideo, &nVideoBytes)) {
      if (nVideoBytes) {
        auto vecTupFrame = sample.decoder_->Decode(pVideo, nVideoBytes);
      }
    }

  cuCtxPopCurrent(&cuContext);

  // uint8_t* data = nullptr;
  // int data_size = 0;

  // data_provider_ = std::make_unique<HostMemDataProvider>(data, data_size);
  // demuxer_ = std::make_unique<FFmpegDemuxer>(data_provider_.get());
  // current_packet_ = std::make_unique<PacketData>();

  // int nVideoBytes = 0, nFrameReturned = 0, nFrame = 0;
  // uint8_t* pVideo = NULL, * pFrame;
  // memset(current_packet_.get(), 0, sizeof(PacketData));

  // while (demuxer_->Demux(&pVideo, &nVideoBytes)) {
  //   if (nVideoBytes) {
  //     current_packet_->bsl_data = (uintptr_t)pVideo;
  //     current_packet_->bsl = nVideoBytes;
  //   }
  // }
  }
}



DALI_SCHEMA(plugin__video__decoders__Video)
    .DocStr(
        R"code(Decodes a video file from a memory buffer (e.g. provided by external source).

The video streams can be in most of the container file formats. FFmpeg is used to parse video
 containers and returns a batch of sequences of frames with shape (F, H, W, C) where F is the
 number of frames in a sequence and can differ for each sample.)code")
    .NumInput(1)
    .NumOutput(1)
    .InputDox(0, "buffer", "TensorList", "Data buffer with a loaded video file.")
    .AddOptionalArg("affine",
    R"code(Applies only to the mixed backend type.

If set to True, each thread in the internal thread pool will be tied to a specific CPU core.
 Otherwise, the threads can be reassigned to any CPU core by the operating system.)code", true);



DALI_REGISTER_OPERATOR(plugin__video__decoders__Video, VideoDecoderMixed, dali::Mixed);

}  // namespace dali_video
