// Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_VIDEO_FRAMES_DECODER_GPU_H_
#define DALI_OPERATORS_VIDEO_FRAMES_DECODER_GPU_H_

#include "dali/operators/video/frames_decoder_base.h"

extern "C" {
#include <libavcodec/bsf.h>
}

#include <string>
#include <memory>
#include <queue>
#include <vector>
#include <utility>

#include "dali/operators/video/dynlink_nvcuvid/cuviddec.h"
#include "dali/operators/video/dynlink_nvcuvid/dynlink_nvcuvid.h"
#include "dali/core/cuda_event.h"
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
  // Size of the coded frame (CUVIDEOFORMAT::coded_*), which the decoder is created for. It is
  // aligned to the macroblock/CTB size, e.g. 1920x1088 for a 1920x1080 H.264 stream.
  unsigned coded_height = 0;
  unsigned coded_width = 0;
  // Output size and origin of the display area (CUVIDEOFORMAT::display_area), which the decoder
  // crops the coded frame to, e.g. 1920x1080 at (0, 0).
  unsigned height = 0;
  unsigned width = 0;
  int display_left = 0;
  int display_top = 0;
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

    int device_id() const noexcept {
      return decoder->device_id;
    }

    explicit operator bool() const noexcept {
      return decoder != nullptr;
    }

 private:
    NVDECCache *owner = nullptr;
    DecInstance *decoder = nullptr;
};

/**
 * @brief Returns how many NVDEC decoders were created so far for a device, i.e. how many times
 *        the decoder cache could not provide a matching one.
 *
 * The count includes decoders that were evicted from the cache since.
 *
 * @param device_id CUDA device ordinal or -1 for the current device.
 */
DLL_PUBLIC size_t NVDECDecodersCreated(int device_id = -1);

/**
 * @brief Returns how many NVDEC output surfaces are mapped by FramesDecoderGpu objects in this
 *        process. Meant for tests: it is 0 when no decoder is alive.
 */
DLL_PUBLIC int64_t NVDECFramesMapped();

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

class DLL_PUBLIC FramesDecoderGpu : public FramesDecoderBase {
 public:
  /**
   * @brief Construct a new FramesDecoder object.
   *
   * @param filename Path to a video file.
   * @param stream CUDA stream to use for decoding.
   * @param image_type Image type of the video.
   */
  explicit FramesDecoderGpu(const std::string &filename, cudaStream_t stream = 0,
                            DALIImageType image_type = DALI_RGB);

  /**
   * @brief Construct a new FramesDecoder object.
   *
   * @param memory_file Pointer to memory with video file data.
   * @param memory_file_size Size of memory_file in bytes.
   * @param source_info Source info of the video file.
   * @param stream CUDA stream to use for decoding.
   * @param image_type Image type of the video.
   * @note This constructor assumes that the `memory_file` and
   * `memory_file_size` arguments cover the entire video file, including the header.
   */
  FramesDecoderGpu(const char *memory_file, size_t memory_file_size,
                   std::string_view source_info = {}, cudaStream_t stream = 0,
                   DALIImageType image_type = DALI_RGB);

  bool ReadNextFrame(uint8_t *data) override;

  void SeekFrame(int frame_id) override;

  void Reset() override;

  void Flush() override;

  int ProcessPictureDecode(CUVIDPICPARAMS *picture_params);

  int HandlePictureDisplay(CUVIDPARSERDISPINFO *picture_display_info);

  // not movable: the parser calls back into this object (and it may hold a mapped surface)
  FramesDecoderGpu(FramesDecoderGpu&&) = delete;

  ~FramesDecoderGpu();

  static bool SupportsHevc();

  /**
   * @brief Number of NVDEC engines of the device; 0 if unknown.
   */
  static int NumEngines(int device_id);

  static bool SupportsCodec(AVCodecID codec_id, uint8_t bit_depth = 8);

  void InitGpuDecoder(CUVIDEOFORMAT *video_format);

  void CopyFrame(uint8_t *dst, const uint8_t *src) override;

 protected:
  bool SelectVideoStream(int stream_id = -1) override;

 private:
  std::unique_ptr<NvDecodeState> nvdecode_state_;

  /**
   * @brief Unmaps the output surface of the last converted picture, once its conversion is done.
   *
   * The surface stays mapped after the conversion is launched, so that the host does not wait for
   * each picture; it is released before NVDEC is used again (the next picture, a flush), which is
   * when NVDEC could overwrite it.
   */
  void ReleaseMappedFrame();

  CUdeviceptr mapped_frame_ = 0;      // the mapped output surface, 0 if none
  CUDAEvent conversion_done_;         // recorded on stream_ after mapped_frame_ is converted
  bool conversion_recorded_ = false;  // false: the conversion failed, wait for the whole stream

  void *current_frame_output_ = nullptr;
  bool current_copy_to_output_ = false;
  bool frame_returned_ = false;
  bool flush_ = false;
  bool more_frames_to_decode_ = true;

  // This is used to order the frames, if there is no pts
  int frame_index_if_no_pts_ = 0;

  AVUniquePtr<AVBSFContext> bsfc_;
  AVUniquePtr<AVPacket> filtered_packet_;

  // TODO(awolant): This value is an approximation. Make it set dynamically
  const int num_decode_surfaces_ = 8;

  std::vector<BufferedFrame> frame_buffer_;

  std::queue<int64_t> piped_pts_;
  int64_t current_pts_ = AV_NOPTS_VALUE;

  cudaStream_t stream_ = 0;

  VideoColorSpaceConversionType conversion_type_ = VIDEO_COLOR_SPACE_CONVERSION_TYPE_YUV_TO_RGB;

  void SendLastPacket(bool flush = false);

  BufferedFrame& FindEmptySlot();

  bool HasEmptySlot() const;

  bool IsBufferEmpty() const;

  void InitBitStreamFilter();

  cudaVideoCodec GetCodecType(AVCodecID codec_id) const;

  bool IsFullRange(CUVIDEOFORMAT *video_format) const;

  void InitGpuParser();

  bool ReadNextFrameWithIndex(uint8_t *data);

  bool ReadNextFrameWithoutIndex(uint8_t *data);

  bool SendFrameToParser();

  unsigned int NumEmptySpots() const;

  unsigned int NumBufferedFrames() const;
};

}  // namespace dali

#endif  // DALI_OPERATORS_VIDEO_FRAMES_DECODER_GPU_H_
