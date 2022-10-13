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

#ifndef DALI_OPERATORS_READER_LOADER_VIDEO_FRAMES_DECODER_GPU_H_
#define DALI_OPERATORS_READER_LOADER_VIDEO_FRAMES_DECODER_GPU_H_

#include "dali/operators/reader/loader/video/frames_decoder.h"

extern "C" {
#include <libavcodec/bsf.h>
}

#include <string>
#include <memory>
#include <queue>
#include <vector>

#include "dali/operators/reader/loader/video/nvdecode/cuviddec.h"
#include "dali/operators/reader/loader/video/nvdecode/nvcuvid.h"

#include "dali/core/dev_buffer.h"

namespace dali {
struct NvDecodeState {
  CUvideodecoder decoder;
  CUvideoparser parser;

  CUVIDSOURCEDATAPACKET packet = { 0 };

  uint8_t *decoded_frame_yuv;

  ~NvDecodeState();
};

struct BufferedFrame {
  DeviceBuffer<uint8_t> frame_;
  int pts_;
};

class DLL_PUBLIC FramesDecoderGpu : public FramesDecoder {
 public:
  /**
   * @brief Construct a new FramesDecoder object.
   *
   * @param filename Path to a video file.
   * @param stream Stream used for decode processing.
   */
  explicit FramesDecoderGpu(const std::string &filename, cudaStream_t stream = 0);

  /**
 * @brief Construct a new FramesDecoder object.
 *
 * @param memory_file Pointer to memory with video file data.
 * @param memory_file_size Size of memory_file in bytes.
 * @param build_index If set to false index will not be build and some features are unavailable.
 *
 * @note This constructor assumes that the `memory_file` and
 * `memory_file_size` arguments cover the entire video file, including the header.
 */
  FramesDecoderGpu(
    const char *memory_file,
    int memory_file_size,
    cudaStream_t stream = 0,
    bool build_index = true);

  bool ReadNextFrame(uint8_t *data, bool copy_to_output = true) override;

  void SeekFrame(int frame_id) override;

  void Reset() override;

  int NextFramePts() { return Index(NextFrameIdx()).pts; }

  int ProcessPictureDecode(void *user_data, CUVIDPICPARAMS *picture_params);

  FramesDecoderGpu(FramesDecoderGpu&&) = default;

  ~FramesDecoderGpu();

  static bool SupportsHevc();

 private:
  std::unique_ptr<NvDecodeState> nvdecode_state_;
  uint8_t *current_frame_output_ = nullptr;
  bool current_copy_to_output_ = false;
  bool frame_returned_ = false;
  bool flush_ = false;
  bool more_frames_to_decode_ = true;

  AVBSFContext *bsfc_ = nullptr;
  AVPacket *filtered_packet_ = nullptr;

  // TODO(awolant): This value is an approximation. Make it set dynamically
  const int num_decode_surfaces_ = 8;

  std::vector<BufferedFrame> frame_buffer_;

  std::queue<int> piped_pts_;

  cudaStream_t stream_;

  void SendLastPacket(bool flush = false);

  BufferedFrame& FindEmptySlot();

  bool HasEmptySlot() const;

  bool IsBufferEmpty() const;

  void InitBitStreamFilter();

  cudaVideoCodec GetCodecType();

  void InitGpuDecoder();

  bool ReadNextFrameWithIndex(uint8_t *data, bool copy_to_output);

  bool ReadNextFrameWithoutIndex(uint8_t *data, bool copy_to_output);

  bool SendFrameToParser();
};

}  // namespace dali

#endif  // DALI_OPERATORS_READER_LOADER_VIDEO_FRAMES_DECODER_GPU_H_
