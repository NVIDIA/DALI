// Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_VIDEO_FRAMES_DECODER_CPU_H_
#define DALI_OPERATORS_VIDEO_FRAMES_DECODER_CPU_H_

#include "dali/operators/video/frames_decoder_base.h"
#include <string>
#include <string_view>

namespace dali {

class DLL_PUBLIC FramesDecoderCpu : public FramesDecoderBase {
 public:
  /**
   * @brief Construct a new FramesDecoder object.
   *
   * @param filename Path to a video file.
   * @param image_type Image type of the video.
   */
   explicit FramesDecoderCpu(const std::string &filename, DALIImageType image_type = DALI_RGB);

  /**
   * @brief Construct a new FramesDecoder object.
   *
   * @param memory_file Pointer to memory with video file data.
   * @param memory_file_size Size of memory_file in bytes.
   * @param source_info Source info of the video file.
   * @param image_type Image type of the video.
   *
   * @note This constructor assumes that the `memory_file` and
   * `memory_file_size` arguments cover the entire video file, including the header.
   */
  FramesDecoderCpu(const char *memory_file, size_t memory_file_size, std::string_view = {}, DALIImageType image_type = DALI_RGB);

  FramesDecoderCpu(FramesDecoderCpu&&) = default;

  bool ReadNextFrame(uint8_t *data) override;
  void CopyFrame(uint8_t *dst, const uint8_t *src) override;
  void Reset() override;
  void Flush() override;

 protected:
  bool SelectVideoStream(int stream_id = -1) override;

 private:
  void CopyToOutput(uint8_t *data);
  bool ReadRegularFrame(uint8_t *data);
  bool ReadFlushFrame(uint8_t *data);
  bool flush_state_ = false;

  const AVCodec *codec_ = nullptr;
  std::unique_ptr<SwsContext, decltype(&sws_freeContext)> sws_ctx_{
    nullptr, sws_freeContext};

  std::vector<uint8_t> tmp_buffer_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_VIDEO_FRAMES_DECODER_CPU_H_
