// Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_READER_LOADER_VIDEO_FRAMES_DECODER_H_
#define DALI_OPERATORS_READER_LOADER_VIDEO_FRAMES_DECODER_H_

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libswscale/swscale.h>
}

#include <vector>
#include <string>
#include <memory>

#include "dali/core/common.h"

namespace dali {
struct IndexEntry {
  int64_t pts;
  int last_keyframe_id;
  bool is_keyframe;
  bool is_flush_frame;
};

struct AvState {
  AVFormatContext *ctx_ = nullptr;
  const AVCodec *codec_ = nullptr;
  AVCodecParameters *codec_params_ = nullptr;
  AVCodecContext *codec_ctx_ = nullptr;
  AVFrame *frame_ = nullptr;
  AVPacket *packet_ = nullptr;
  SwsContext  *sws_ctx_ = nullptr;
  int stream_id_ = -1;

  ~AvState() {
    sws_freeContext(sws_ctx_);
    if (packet_ != nullptr) {
      av_packet_unref(packet_);
      av_packet_free(&packet_);
    }
    if (frame_ != nullptr) {
      av_frame_unref(frame_);
      av_frame_free(&frame_);
    }
    avcodec_free_context(&codec_ctx_);
    avformat_close_input(&ctx_);
    avformat_free_context(ctx_);

    ctx_ = nullptr;
    codec_ = nullptr;
    codec_params_ = nullptr;
    codec_ctx_ = nullptr;
    frame_ = nullptr;
    packet_ = nullptr;
    sws_ctx_ = nullptr;
  }
};

/**
 * @brief Object representing a video file. Allows access to frames and seeking.
 * 
 */
class DLL_PUBLIC FramesDecoder {
 public:
  /**
   * @brief Construct a new FramesDecoder object.
   * 
   * @param filename Path to a video file.
   */
  explicit FramesDecoder(const std::string &filename);

  /**
   * @brief Number of frames in the video
   * 
   */
  int64_t NumFrames() const {
    return index_.size();
  }

  /**
   * @brief Width of a video frame in pixels
   * 
   */
  int Width() const {
    return av_state_->codec_params_->width;
  }

  /**
   * @brief Height of a video frame in pixels
   * 
   */
  int Height() const {
    return av_state_->codec_params_->height;
  }

  /**
   * @brief Number of channels in a video
   * 
   */
  int Channels() const {
      return channels_;
  }

  /**
   * @brief Total number of values in a frame (width * height * channels)
   * 
   */
  int FrameSize() const {
    return Channels() * Width() * Height();
  }

  /**
   * @brief Reads next frame of the video and copies it to the provided buffer, if copy_to_output is True.
   * 
   * @param data Output buffer to copy data to.
   * @param copy_to_output Whether copy the data to the output. 
   * @return Boolean indicating whether the frame was read or not. False means no more frames in the decoder.
   */
  virtual bool ReadNextFrame(uint8_t *data, bool copy_to_output = true);

  /**
   * @brief Seeks to the frame given by id. Next call to ReadNextFrame will return this frame
   * 
   * @param frame_id Id of the frame to seek to
   */
  virtual void SeekFrame(int frame_id);

  /**
   * @brief Seeks to the first frame
   * 
   */
  virtual void Reset();

  /**
   * @brief Returns index of the frame that will be returned by the next call of ReadNextFrame
   * 
   * @return int Index of the next frame to be read
   */
  int NextFrameIdx() { return next_frame_idx_; }

  FramesDecoder(FramesDecoder&&) = default;

  virtual ~FramesDecoder() = default;

 protected:
  std::unique_ptr<AvState> av_state_;

  std::vector<IndexEntry> index_;

  int next_frame_idx_ = 0;

 private:
   /**
   * @brief Gets the packet from the decoder and reads a frame from it to provided buffer. Returns 
   * boolean indicating, if the frame was succesfully read.
   * After this method returns false, there might be more frames to read. Call `ReadFlushFrame` until
   * it returns false, to get all of the frames from the video file.
   * 
   * @param data Output buffer to copy data to. If `copy_to_output` is false, this value is ignored.
   * @param copy_to_output Whether copy the frame to provided output.
   * 
   * @returns True, if the read was succesful, or false, when all regular farmes were consumed.
   * 
   */
  bool ReadRegularFrame(uint8_t *data, bool copy_to_output = true);

  /**
   * @brief Reads frames from the last packet. This packet can hold
   * multiple frames. This method will read all of them one by one.
   * 
   * @param data Output buffer to copy data to. If `copy_to_output` is false, this value is ignored.
   * @param copy_to_output Whether copy the frame to provided output.
   * 
   * @returns True, if the read was succesful, or false, when ther are no more frames in last the packet.
   */
  bool ReadFlushFrame(uint8_t *data, bool copy_to_output = true);

  void CopyToOutput(uint8_t *data);

  void BuildIndex();

  void InitAvState();

  void FindVideoStream();

  void LazyInitSwContext();

  int channels_ = 3;
  bool flush_state_ = false;
  std::string filename_;
};
}  // namespace dali

#endif  // DALI_OPERATORS_READER_LOADER_VIDEO_FRAMES_DECODER_H_
