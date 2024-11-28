// Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "decoder/color_space.h"

namespace dali_video {

class MemoryVideoFile : public FFmpegDemuxer::DataProvider {
 public:
  MemoryVideoFile(const void *data, int64_t size)
      : data_(static_cast<const uint8_t *>(data)), size_(size), position_(0) {}


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

bool VideoDecoderMixed::SetupImpl(std::vector<dali::OutputDesc> &output_desc,
                                  const dali::Workspace &ws) {
  ValidateInput(ws);
  const auto &input = ws.Input<dali::CPUBackend>(0);
  int batch_size = input.num_samples();
  auto stream = ws.stream();

  samples_.resize(batch_size);
  dali::TensorListShape<> sh(batch_size, 4);
  for (int i = 0; i < batch_size; i++) {
    auto &sample = samples_[i];
    sample.data_provider_ =
        std::make_unique<MemoryVideoFile>(input.raw_tensor(i), input[i].shape().num_elements());
    sample.demuxer_ = std::make_unique<FFmpegDemuxer>(sample.data_provider_.get());
    sample.current_packet_ = std::make_unique<PacketData>();
    sh.set_tensor_shape(i, dali::TensorShape<>(end_frame_, sample.demuxer_->GetHeight(),
                                               sample.demuxer_->GetWidth(), 3));
  }
  output_desc.resize(1);
  output_desc[0].shape = sh;
  output_desc[0].type = DALI_UINT8;
  return true;
}

void VideoDecoderMixed::RunImpl(dali::Workspace &ws) {
  auto &output = ws.Output<dali::GPUBackend>(0);
  const auto &input = ws.Input<dali::CPUBackend>(0);
  int batch_size = input.num_samples();

  CUcontext cuContext = nullptr;
  CUstream cuStream = ws.stream();
  cuCtxGetCurrent(&cuContext);

  if (!cuContext) {
    throw std::runtime_error("Failed to create a cuda context");
  }

  for (int i = 0; i < batch_size; i++) {
    auto output_sample = output[i];
    uint8_t *output_data = output_sample.template mutable_data<uint8_t>();

    auto &sample = samples_[i];
    sample.decoder_ = std::make_unique<NvDecoder>(
        cuStream, cuContext, true, FFmpeg2NvCodecId(sample.demuxer_->GetVideoCodec()), false,
        false /*_enableasyncallocations*/, false);


    uint8_t *pVideo = NULL, *pFrame = nullptr;
    int nVideoBytes = 0, nFrameReturned = 0, nFrame = 0;

    int num_frames = 0;

    do {
      sample.demuxer_->Demux(&pVideo, &nVideoBytes);
      nFrameReturned = sample.decoder_->Decode(pVideo, nVideoBytes);

      for (int i = 0; i < nFrameReturned; i++) {
        pFrame = sample.decoder_->GetFrame();
        CUDA_CALL(cudaStreamSynchronize(cuStream));

        uint8_t *dpFrame = output_data + num_frames * sample.demuxer_->GetHeight() *
                                             sample.demuxer_->GetWidth() * 3;
        int nWidth = sample.decoder_->GetWidth();
        int nPitch = sample.decoder_->GetWidth();
        int iMatrix =
            sample.decoder_->GetVideoFormatInfo().video_signal_description.matrix_coefficients;
        bool full_range =
            sample.decoder_->GetVideoFormatInfo().video_signal_description.video_full_range_flag;


        yuv_to_rgb(pFrame, nPitch, reinterpret_cast<uint8_t *>(dpFrame),
                   sample.decoder_->GetWidth() * 3, sample.decoder_->GetWidth(),
                   sample.decoder_->GetHeight(), full_range, cuStream);

        ++num_frames;
        if (end_frame_ > 0 && num_frames >= end_frame_) {
          return;
        }
      }
    } while (nVideoBytes);
  }
}


DALI_SCHEMA(plugin__video__Decoder)
    .DocStr(
        R"code(Decodes a video file from a memory buffer (e.g. provided by external source).

The video streams can be in most of the container file formats. FFmpeg is used to parse video
 containers and returns a batch of sequences of frames with shape (F, H, W, C) where F is the
 number of frames in a sequence and can differ for each sample.)code")
    .NumInput(1)
    .NumOutput(1)
    .InputDox(0, "buffer", "TensorList", "Data buffer with a loaded video file.")
    .AddOptionalArg("end_frame", R"code(Index of the end frame to be decoded.)code", 0)
    .AddOptionalArg("affine",
                    R"code(Applies only to the mixed backend type.

If set to True, each thread in the internal thread pool will be tied to a specific CPU core.
 Otherwise, the threads can be reassigned to any CPU core by the operating system.)code",
                    true);


DALI_REGISTER_OPERATOR(plugin__video__Decoder, VideoDecoderMixed, dali::Mixed);

}  // namespace dali_video
