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
#include <utility>

#include "dali/operators/reader/loader/video/nvdecode/cuviddec.h"
#include "dali/operators/reader/loader/video/nvdecode/nvcuvid.h"
#include "dali/operators/reader/nvdecoder/dynlink_nvcuvid.h"
#include "dali/core/unique_handle.h"

#include "dali/core/dev_buffer.h"

namespace dali {

namespace frame_dec_gpu_impl {

struct CUvideoparserHandle : public UniqueHandle<CUvideoparser, CUvideoparserHandle> {
  DALI_INHERIT_UNIQUE_HANDLE(CUvideoparser, CUvideoparserHandle);

  CUvideoparserHandle() = default;

  explicit CUvideoparserHandle(CUVIDPARSERPARAMS &parser_info) {
    CUDA_CALL(cuvidCreateVideoParser(&handle_, &parser_info));
  }

  static constexpr CUvideoparser null_handle() { return nullptr; }

  static void DestroyHandle(CUvideoparser handle) {
    cuvidDestroyVideoParser(handle);
  }
};

struct DecInstance {
  CUvideodecoder decoder = {};
  cudaVideoCodec codec_type = {};
  cudaVideoChromaFormat chroma_format = {};
  unsigned height = 0;
  unsigned width = 0;
  unsigned num_decode_surfaces = 0;
  unsigned max_height = 0;
  unsigned max_width = 0;
  unsigned int bit_depth_luma_minus8 = 0;
  bool used = false;
  int device_id = 0;
};

class NVDECCache;

class NVDECLease {
 public:
    constexpr NVDECLease() = default;

    explicit NVDECLease(NVDECCache *owner, DecInstance *dec) : owner(owner), decoder(dec) {
    }

    ~NVDECLease() {
      Return();
    }

    void Return();

    NVDECLease(NVDECLease &&other) {
      *this = std::move(other);
    }

    NVDECLease &operator=(NVDECLease &&other) {
      std::swap(owner, other.owner);
      std::swap(decoder, other.decoder);
      other.Return();
      return *this;
    }

    operator CUvideodecoder() const & noexcept {
      return decoder->decoder;
    }

    operator CUvideodecoder() && = delete;

    explicit operator bool() const noexcept {
      return decoder != nullptr;
    }

 private:
    NVDECCache *owner = nullptr;
    DecInstance *decoder = nullptr;
};

}  // namespace frame_dec_gpu_impl

struct NvDecodeState {
  frame_dec_gpu_impl::NVDECLease decoder = {};
  frame_dec_gpu_impl::CUvideoparserHandle parser = {};

  CUVIDSOURCEDATAPACKET packet = { 0 };

  uint8_t *decoded_frame_yuv = nullptr;
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
 * @param num_frames If set, number of frames in the video.
 *
 * @note This constructor assumes that the `memory_file` and
 * `memory_file_size` arguments cover the entire video file, including the header.
 */
  FramesDecoderGpu(
    const char *memory_file,
    int memory_file_size,
    cudaStream_t stream = 0,
    bool build_index = true,
    int num_frames = -1);

  bool ReadNextFrame(uint8_t *data, bool copy_to_output = true) override;

  void SeekFrame(int frame_id) override;

  void Reset() override;

  int NextFramePts() { return Index(NextFrameIdx()).pts; }

  int ProcessPictureDecode(CUVIDPICPARAMS *picture_params);

  int HandlePictureDisplay(CUVIDPARSERDISPINFO *picture_display_info);

  FramesDecoderGpu(FramesDecoderGpu&&) = default;

  ~FramesDecoderGpu();

  static bool SupportsHevc();

  void InitGpuDecoder(CUVIDEOFORMAT *video_format);

 private:
  std::unique_ptr<NvDecodeState> nvdecode_state_;
  uint8_t *current_frame_output_ = nullptr;
  bool current_copy_to_output_ = false;
  bool frame_returned_ = false;
  bool flush_ = false;
  bool more_frames_to_decode_ = true;

  // This is used to order the frames, if there is no pts
  int frame_index_if_no_pts_ = 0;

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

  void InitGpuParser();

  bool ReadNextFrameWithIndex(uint8_t *data, bool copy_to_output);

  bool ReadNextFrameWithoutIndex(uint8_t *data, bool copy_to_output);

  bool SendFrameToParser();

  unsigned int NumEmptySpots() const;
};

}  // namespace dali

#endif  // DALI_OPERATORS_READER_LOADER_VIDEO_FRAMES_DECODER_GPU_H_
