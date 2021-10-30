// Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_READER_LOADER_VIDEO_FRAMES_DECODER_GPU_H_
#define DALI_OPERATORS_READER_LOADER_VIDEO_FRAMES_DECODER_GPU_H_

#include "dali/operators/reader/loader/video/frames_decoder.h"

namespace dali {

class DLL_PUBLIC FramesDecoderGpu : public FramesDecoder {
 public:
  /**
   * @brief Construct a new FramesDecoder object.
   * 
   * @param filename Path to a video file.
   */
  explicit FramesDecoderGpu(const std::string &filename);

 private:
  bool DecodeFrame(uint8_t *data, bool copy_to_output = true) override;
};

}  // namespace dali

#endif  // DALI_OPERATORS_READER_LOADER_VIDEO_FRAMES_DECODER_GPU_H_
