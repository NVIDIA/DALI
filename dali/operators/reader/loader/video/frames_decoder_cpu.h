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

#ifndef DALI_OPERATORS_READER_LOADER_VIDEO_FRAMES_DECODER_CPU_H_
#define DALI_OPERATORS_READER_LOADER_VIDEO_FRAMES_DECODER_CPU_H_

#include "dali/operators/reader/loader/video/frames_decoder_base.h"
#include <string>
#include <string_view>

namespace dali {

class DLL_PUBLIC FramesDecoderCpu : public FramesDecoderBase {
 public:
  /**
   * @brief Construct a new FramesDecoder object.
   *
   * @param filename Path to a video file.
   * @param build_index If set to false index will not be build and some features are unavailable.
   */
  explicit FramesDecoderCpu(const std::string &filename, bool build_index = true);

  /**
 * @brief Construct a new FramesDecoder object.
 *
 * @param memory_file Pointer to memory with video file data.
 * @param memory_file_size Size of memory_file in bytes.
 * @param build_index If set to false index will not be build and some features are unavailable.
 * @param num_frames If set, number of frames in the video.
 *
 * @note This constructor assumes that the `memory_file` and
 * `memory_file_size` arguments cover the entire video file, including the header.
 */
  FramesDecoderCpu(const char *memory_file, size_t memory_file_size, bool build_index = true,
                   int num_frames = -1, std::string_view = {});

  FramesDecoderCpu(FramesDecoderCpu&&) = default;

  /**
   * @brief Check if a codec is supported by the particular implementation.
   *
   * @param codec_id Codec ID to check.
   * @return True if the codec is supported, false otherwise.
   */
  bool CanDecode(AVCodecID codec_id) const;
};

}  // namespace dali

#endif  // DALI_OPERATORS_READER_LOADER_VIDEO_FRAMES_DECODER_CPU_H_
