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

#include <exception>
#include <string>
#include <memory>
#include <queue>
#include <vector>
#include <utility>

#include "dali/operators/video/dynlink_nvcuvid/cuviddec.h"
#include "dali/operators/video/dynlink_nvcuvid/dynlink_nvcuvid.h"
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

}  // namespace frame_dec_gpu_impl

struct NvDecodeState {
  frame_dec_gpu_impl::NVDECLease decoder = {};
  frame_dec_gpu_impl::CUvideoparserHandle parser = {};

  CUVIDSOURCEDATAPACKET packet = { 0 };

  uint8_t *decoded_frame_yuv = nullptr;
};

/**
 * @brief A place for a frame that the parser displayed before it was read.
 *
 * The state changes only through SetTimestamp, MarkPixelsValid and Clear, so that a slot never
 * keeps the flag of the frame that it held before.
 */
struct BufferedFrame {
  DeviceBuffer<uint8_t> frame_;  // may be empty when the slot keeps no frame data
  int64_t pts_ = -1;             // -1: the slot is empty
  bool has_data_ = false;        // false: the frame is never read with its data, only pts_ is kept

  /** @brief Takes a frame in, with no pixels yet (see MarkPixelsValid). */
  void SetTimestamp(int64_t pts) {
    pts_ = pts;
    has_data_ = false;
  }

  /** @brief The frame is mapped and converted into frame_: it can be read with its data. */
  void MarkPixelsValid() {
    has_data_ = true;
  }

  /** @brief Empties the slot. It keeps frame_ for the next frame. */
  void Clear() {
    pts_ = -1;
    has_data_ = false;
  }
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

  /**
   * @brief Construct a new FramesDecoder object.
   *
   * @param input_stream Input stream with video file data; the decoder takes ownership of it.
   * @param source_info Source info of the video file.
   * @param stream CUDA stream to use for decoding.
   * @param image_type Image type of the video.
   * @note This constructor assumes that the input stream covers the entire video file,
   * including the header.
   */
  explicit FramesDecoderGpu(std::unique_ptr<InputStream> input_stream,
                            std::string_view source_info = {}, cudaStream_t stream = 0,
                            DALIImageType image_type = DALI_RGB);

  bool ReadNextFrame(uint8_t *data) override;

  void SeekFrame(int frame_id) override;

  void Reset() override;

  void Flush() override;

  int ProcessPictureDecode(CUVIDPICPARAMS *picture_params);

  int HandlePictureDisplay(CUVIDPARSERDISPINFO *picture_display_info);

  // not movable: the parser calls back into this object (pUserData)
  FramesDecoderGpu(FramesDecoderGpu&&) = delete;

  ~FramesDecoderGpu();

  static bool SupportsHevc();

  /**
   * @brief Checks if the codec can be decoded by this decoder, as it is done when the video stream
   *        is selected: it is supported by the decoder and by NVDEC on this platform
   *        (8-bit 4:2:0). It warns if it is not, unless `warn` is false.
   *
   * It queries NVDEC of the current device (once per codec, cached per process), so the calling
   * thread must have a current CUDA context.
   */
  static bool IsCodecSupported(AVCodecID codec_id, bool warn = true);

  /**
   * @brief The codecs for which `IsCodecSupported` is true: a codec is supported if and only if it
   *        is on this list. The list is made on the calling thread (with the same requirement as
   *        `IsCodecSupported`), and can then be used from any thread.
   */
  static std::vector<AVCodecID> SupportedCodecs();

  static bool SupportsCodec(AVCodecID codec_id, uint8_t bit_depth = 8);

  /**
   * @brief Gets the NVDEC decoder for the sequence. Throws if the size of the decoded pictures is
   *        not the size of the output.
   */
  void InitGpuDecoder(CUVIDEOFORMAT *video_format);

  /**
   * @brief How many pictures the parser keeps decoded before it passes them to the display
   *        callback, on top of what reordering needs. That many pictures are queued on NVDEC
   *        while one is mapped and converted.
   */
  static constexpr int kDisplayDelay = 1;

  /**
   * @brief Decode surfaces on top of the minimum that the stream needs and of the display delay:
   *        the picture being mapped and converted must not be decoded over, and the minimum does
   *        not guarantee the best performance (nvcuvid.h).
   */
  static constexpr int kSpareDecodeSurfaces = 2;

  /**
   * @brief The number of decode surfaces of the decoder for a stream. It is also what the sequence
   *        callback returns to the parser, which uses it from then on.
   */
  static int NumDecodeSurfaces(const CUVIDEOFORMAT &video_format);

  /** @brief Stores an exception thrown in a parser callback, to be rethrown by ParseVideoData. */
  void SetCallbackException(std::exception_ptr e) {
    callback_exception_ = std::move(e);
  }

  void CopyFrame(uint8_t *dst, const uint8_t *src) override;

  /**
   * @brief Number of frame buffers (device memory for one frame each) held by the decoder, for
   *        tests.
   */
  int NumFrameBuffers() const;

  /**
   * @brief Number of places for a buffered frame (with or without a frame buffer), for tests.
   */
  int NumBufferSlots() const {
    return static_cast<int>(frame_buffer_.size());
  }

  /**
   * @brief Number of pictures mapped and converted so far (for tests and diagnostics).
   *
   * Pictures that cannot be read with their data when they are displayed (MayNeedPixelsLater is
   * false), or that are displayed as the next frame while it is only skipped, are not.
   */
  int64_t MappedPictures() const {
    return mapped_pictures_;
  }

 protected:
  bool SelectVideoStream(int stream_id = -1) override;

  /**
   * @brief Which of the buffered frames are mapped and converted.
   *
   * The pictures are mapped in display order, when the parser displays them (HandlePictureDisplay).
   * A picture displayed when it is not the next frame is buffered: a frame shown after the next
   * one, which the parser displays in the same cuvidParseVideoData call as the next one, or at the
   * end of the stream, when it displays all the pictures it holds; or a frame that is never read
   * (e.g. one hidden by an edit list, or one shown before the next frame after a seek), which
   * kIfRead drops instead (see IsAmongNextFrames).
   */
  enum class BufferedFrameData {
    // those that may be read with their data (see MayNeedPixelsLater)
    kIfRead,
    // all of them; every picture is then decoded (see ShouldDecodePicture)
    kAll,
    // none of them; an H.264 non-reference picture is then decoded only if it is the next frame,
    // read with its data, when it is decoded
    kNone,
  };

  /**
   * @brief For tests: maps all the buffered frames and decodes every picture (to compare the frames
   *        read with the default) or maps none of them (to check that reading a frame without its
   *        data fails).
   */
  void SetBufferedFrameDataForTesting(BufferedFrameData data) {
    buffered_frame_data_ = data;
  }

  /**
   * @brief Whether a frame with this timestamp may still be read with its pixels, until the
   *        decoder is flushed. False for a frame that is only skipped or never read.
   */
  bool MayNeedPixelsLater(int64_t pts) const;

  /**
   * @brief Whether a frame with this timestamp is among the frames from the next one on, which are
   *        the only ones looked up in the buffer until the decoder is flushed (true without an
   *        index, or past the last frame).
   */
  bool IsAmongNextFrames(int64_t pts) const;

  /**
   * @brief Whether the parser displays a frame with this timestamp, that is not the next one, with
   *        its pixels (mapped and converted) in the buffer, or only its timestamp.
   *
   * It follows MayNeedPixelsLater, unless SetBufferedFrameDataForTesting says otherwise.
   */
  bool ShouldBufferPixels(int64_t pts) const;

  /**
   * @brief Whether the picture with this timestamp, about to be decoded, is decoded at all.
   *
   * False only if its pixels are never read (see ShouldBufferPixels), which lets the decoder skip
   * a picture that no other picture refers to.
   */
  bool ShouldDecodePicture(int64_t pts) const;

 private:
  /**
   * @brief Sends a packet to the NVDEC parser. Exceptions thrown by the parser callbacks are
   *        rethrown here, once the parser has returned.
   */
  void ParseVideoData(CUVIDSOURCEDATAPACKET *packet);

  std::unique_ptr<NvDecodeState> nvdecode_state_;

  int64_t mapped_pictures_ = 0;  // see MappedPictures
  BufferedFrameData buffered_frame_data_ = BufferedFrameData::kIfRead;

  void *current_frame_output_ = nullptr;
  bool current_copy_to_output_ = false;
  bool frame_returned_ = false;
  bool flush_ = false;
  std::exception_ptr callback_exception_;
  bool more_frames_to_decode_ = true;

  // This is used to order the frames, if there is no pts
  int frame_index_if_no_pts_ = 0;

  // The stream's timestamps are in decoding order (HasDecodeOrderTimestamps). A picture is then
  // labelled with the timestamp that the parser passes to the display callback, instead of its
  // packet's: the parser hands the timestamps fed in decoding order out in display order.
  bool labels_from_parser_ = false;
  // With labels_from_parser_: a copy of the last frame read, which a frame without a picture of its
  // own (e.g. the not-coded VOP of a DivX packed stream) repeats. Unknown after a flush: such a
  // frame then takes the next picture shown.
  DeviceBuffer<uint8_t> last_frame_;
  bool has_last_frame_ = false;

  AVUniquePtr<AVBSFContext> bsfc_;
  AVUniquePtr<AVPacket> filtered_packet_;

  // TODO(awolant): This value is an approximation. Make it set dynamically
  const int num_decode_surfaces_ = 8;

  std::vector<BufferedFrame> frame_buffer_;

  // pts of the packets sent to the parser, whose pictures were not decoded yet (decode order)
  std::queue<int64_t> piped_pts_;
  // pts of the picture last decoded to each surface (CUVIDPICPARAMS::CurrPicIdx)
  std::vector<int64_t> surface_pts_;
  // whether the picture decoded to each surface is still waiting for the display callback
  std::vector<bool> surface_to_display_;
  // whether cuvidDecodePicture was called for the picture last assigned to each surface (see
  // ShouldDecodePicture); a surface whose picture was not decoded must not be mapped
  std::vector<bool> surface_decoded_;
  int64_t current_pts_ = AV_NOPTS_VALUE;

  cudaStream_t stream_ = 0;

  VideoColorSpaceConversionType conversion_type_ = VIDEO_COLOR_SPACE_CONVERSION_TYPE_YUV_TO_RGB;

  void SendLastPacket(bool flush = false);

  BufferedFrame& FindEmptySlot();

  // Bytes of a frame in the output type
  size_t FrameBytes() const;

  // The buffer for the last frame read (see last_frame_), allocated for the output type
  uint8_t *LastFrame();

  // Keeps a copy of the frame read (see last_frame_), unless it was converted into LastFrame()
  void KeepAsLastFrame(const void *frame);

  // Returns the frame wanted, which has no picture, as a repeat of the last frame read, into `data`
  // (nullptr: skipped, which needs no last frame)
  void RepeatLastFrame(void *data);

  /**
   * @brief Gives the slot a frame buffer, taking a spare one from another slot if there is any.
   */
  void AttachFrameBuffer(BufferedFrame &slot);

  bool HasEmptySlot() const;

  bool IsBufferEmpty() const;

  void InitBitStreamFilter(AVUniquePtr<AVBSFContext> &bsfc);

  /**
   * @brief Asks the NVDEC parser for the size of the pictures in the stream and, if it differs
   *        from the one in the container, uses it as the size of the output.
   *
   * The parser applies the right and bottom crop of the SPS, but ignores the left and top one: a
   * stream with a left or top crop is decoded to the full frame (a pure left/top crop gives the
   * uncropped size) and, with a mixed crop, to a frame cropped on the right and the bottom only.
   * The position of the file is restored to the first packet.
   */
  void ProbeDisplaySize();

  /**
   * @brief Reads the next packet of the selected stream. Returns false at the end of the file.
   */
  bool ReadSelectedStreamPacket(AVPacket *packet);

  /**
   * @brief Seeks back to the start of the file and checks that the first packet is the next one
   *        read. Returns false if the seek failed or landed elsewhere.
   */
  bool RewindToFirstPacket(const AVPacket *first);

  static cudaVideoCodec GetCodecType(AVCodecID codec_id);

  bool IsFullRange(CUVIDEOFORMAT *video_format) const;
  bool IsBT709(CUVIDEOFORMAT *video_format) const;

  void InitGpuParser();

  bool ReadNextFrameWithIndex(uint8_t *data);

  bool ReadNextFrameWithoutIndex(uint8_t *data);

  bool SendFrameToParser();

  unsigned int NumEmptySpots() const;

  unsigned int NumBufferedFrames() const;
};

}  // namespace dali

#endif  // DALI_OPERATORS_VIDEO_FRAMES_DECODER_GPU_H_
