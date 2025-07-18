// Copyright (c) 2017-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/operators/video/legacy/reader/video_loader.h"

#include <dirent.h>
#include <unistd.h>

#include <iomanip>
#include <memory>
#include <string>
#include <utility>
#include <fstream>
#include <limits>
#include <sstream>


inline int gcd(int a, int b) {
  while (b) {
    int tmp = b;
    b = a % b;
    a = tmp;
  }
  return a;
}

inline std::ostream &operator<<(std::ostream &os, AVRational r) {
  if (!r.num) {
    os << "0";
  } else {
    if (r.den < 0) {
      r.den = -r.den;
      r.num = -r.num;
    }
    int cd = gcd(r.num, r.den);
    if (cd != 1) {
      r.num /= cd;
      r.den /= cd;
    }
    os << r.num;
    if (r.den != 1)
      os << "/" << r.den;
  }
  return os;
}

namespace dali {

namespace {
#undef av_err2str
std::string av_err2str(int errnum) {
  char errbuf[AV_ERROR_MAX_STRING_SIZE];
  av_strerror(errnum, errbuf, AV_ERROR_MAX_STRING_SIZE);
  return std::string{errbuf};
}
}

#if HAVE_AVSTREAM_CODECPAR
auto codecpar(AVStream* stream) -> decltype(stream->codecpar) {
  return stream->codecpar;
}
#else
auto codecpar(AVStream* stream) -> decltype(stream->codec) {
  return stream->codec;
}
#endif

// Are these good numbers? Allow them to be set?
static constexpr auto frames_used_warning_ratio = 3.0f;
static constexpr auto frames_used_warning_minimum = 1000;
static constexpr auto frames_used_warning_interval = 10000;

// Source: http://en.cppreference.com/w/cpp/types/numeric_limits/epsilon
template<class T>
typename std::enable_if<!std::numeric_limits<T>::is_integer, bool>::type
    almost_equal(T x, T y, int ulp) {
    if (x == y) return true;
    // the machine epsilon has to be scaled to the magnitude of the values used
    // and multiplied by the desired precision in ULPs (units in the last place)
    return std::abs(x-y) <= std::numeric_limits<T>::epsilon() * std::abs(x+y) * ulp
        // unless the result is subnormal
        || std::abs(x-y) < std::numeric_limits<T>::min();
}

static int read_packet(void *opaque, uint8_t *buf, int buf_size) {
  VideoFileDesc *file_data = static_cast<VideoFileDesc*>(opaque);

  // open file if needed
  if (!file_data->file_stream) {
    file_data->file_stream = fopen(file_data->filename.c_str(), "r");
    DALI_ENFORCE(file_data->file_stream, make_string("Failed to open video file ",
                                                     file_data->filename));
    int ret = fseek(file_data->file_stream, file_data->file_position, SEEK_SET);
    DALI_ENFORCE(ret == 0, make_string("Failed to open video file ", file_data->filename));
  }
  auto ret = fread(buf, 1, buf_size, file_data->file_stream);
  if (ret == 0 && std::feof(file_data->file_stream)) {
    return AVERROR_EOF;
  } else {
    return ret;
  }
}

static int64_t seek_file(void *opaque, int64_t offset, int whence) {
  VideoFileDesc *file_data = static_cast<VideoFileDesc*>(opaque);

  // open file if needed
  if (!file_data->file_stream) {
    file_data->file_stream = fopen(file_data->filename.c_str(), "r");
    DALI_ENFORCE(file_data->file_stream, make_string("Failed to open video file ",
                                                     file_data->filename));
    int ret = fseek(file_data->file_stream, file_data->file_position, SEEK_SET);
    DALI_ENFORCE(ret == 0, make_string("Failed to open video file ", file_data->filename));
  }
  int ret = -1;
  switch (whence) {
    case SEEK_SET:
    case SEEK_CUR:
    case SEEK_END:
      ret = fseek(file_data->file_stream, offset, whence);
      break;
    default:
      break;
  }
  if (ret == 0) {
    return ftell(file_data->file_stream);
  } else {
    // fseek may return > 0 but it is still an error
    return -abs(ret);
  }
}

void clean_avformat_context(AVFormatContext **p) {
  if (*p && (*p)->flags & AVFMT_FLAG_CUSTOM_IO) {
    if (*p && (*p)->pb) av_freep(&(*p)->pb->buffer);
    avio_context_free(&(*p)->pb);
  }
  avformat_close_input(p);
}

VideoFile& VideoLoader::get_or_open_file(const std::string &filename) {
  static VideoFile empty_file = {};
  auto& file = open_files_[filename];

  if (file.empty()) {
    file.file_desc.filename = filename;
    LOG_LINE << "Opening file " << filename << std::endl;

    auto raw_fmt_ctx = std::unique_ptr<AVFormatContext, void(*)(AVFormatContext*)>(
                                       avformat_alloc_context(), avformat_free_context);

    DALI_ENFORCE(raw_fmt_ctx, "Failed to allocate avformat_alloc_context");

    size_t avio_ctx_buffer_size = 4096;
    auto avio_ctx_buffer = std::unique_ptr<uint8_t, void(*)(void*)>(
                                       static_cast<uint8_t*>(av_malloc(avio_ctx_buffer_size)),
                                       av_freep);

    DALI_ENFORCE(avio_ctx_buffer, "Failed to allocate avio_ctx_buffer");

    AVIOContext *avio_ctx = avio_alloc_context(avio_ctx_buffer.get(), avio_ctx_buffer_size,
                                               0, &file.file_desc, &read_packet, NULL, &seek_file);
    if (!avio_ctx) {
      DALI_FAIL("Failed to allocate avio_alloc_context");
    }
    // avio_alloc_context succeeded and avio_ctx owns this memory now, so release from unique_ptr
    avio_ctx_buffer.release();

    raw_fmt_ctx->pb = avio_ctx;
    AVFormatContext *tmp_raw_fmt_ctx = raw_fmt_ctx.release();
    // if avformat_open_input fails it frees raw_fmt_ctx so we can release it from unique_ptr
    int ret = avformat_open_input(&tmp_raw_fmt_ctx, NULL, NULL, NULL);
    if (ret < 0) {
      // avio_ctx->ctx_ is nullified so we need to free the memory here instead of the
      // AVFormatContext destructor which cannot access it through avio_ctx->ctx_ anymore
      av_freep(&avio_ctx->buffer);
      avio_context_free(&avio_ctx);
      DALI_WARN(make_string("Failed to open video file ", filename, " because of ",
                            av_err2str(ret)));
      open_files_.erase(filename);
      return empty_file;
    }
    file.fmt_ctx_ = make_unique_av<AVFormatContext>(tmp_raw_fmt_ctx, clean_avformat_context);
    LOG_LINE << "File open " << filename << std::endl;

    LOG_LINE << "File info fetched for " << filename << std::endl;

    if (file.fmt_ctx_->nb_streams > 1) {
      LOG_LINE << "There are " << file.fmt_ctx_->nb_streams << " streams in "
                  << filename << " which will degrade performance. "
                  << "Consider removing all but the main video stream."
                  << std::endl;
    }

    file.vid_stream_idx_ = av_find_best_stream(file.fmt_ctx_.get(), AVMEDIA_TYPE_VIDEO,
                                  -1, -1, nullptr, 0);
    if (file.vid_stream_idx_ < 0) {
      if (avformat_find_stream_info(file.fmt_ctx_.get(), nullptr) < 0) {
        DALI_WARN(make_string("Could not find stream information in ", filename));
        open_files_.erase(filename);
        return empty_file;
      }
      file.vid_stream_idx_ = av_find_best_stream(file.fmt_ctx_.get(), AVMEDIA_TYPE_VIDEO,
                                  -1, -1, nullptr, 0);
      if (file.vid_stream_idx_ < 0) {
        DALI_WARN(make_string("Could not find a valid video stream in a file in ", filename));
        open_files_.erase(filename);
        return empty_file;
      }
    }
    LOG_LINE << "Best stream " << file.vid_stream_idx_ << " found for "
              << filename << std::endl;

    auto stream = file.fmt_ctx_->streams[file.vid_stream_idx_];
    int width = codecpar(stream)->width;
    int height = codecpar(stream)->height;
    auto codec_id = codecpar(stream)->codec_id;

    if (width == 0 || height == 0) {
      if (avformat_find_stream_info(file.fmt_ctx_.get(), nullptr) < 0) {
        DALI_WARN(make_string("Could not find stream information in ", filename));
        open_files_.erase(filename);
        return empty_file;
      }

      width = codecpar(stream)->width;
      height = codecpar(stream)->height;
      codec_id = codecpar(stream)->codec_id;
    }

    if (max_width_ == 0) {  // first file to open
      max_width_ = width;
      max_height_ = height;
      codec_id_ = codec_id;

    } else {  // already opened a file
      DALI_ENFORCE(codec_id_ == codec_id, "File " + filename +
                   " is not the same codec as previous files");

      if (NVCUVID_API_EXISTS(cuvidReconfigureDecoder)) {
        if (max_width_ < width) max_width_ = width;
        if (max_height_ < height) max_height_ = height;

      } else {
        if (max_width_ != width ||
            max_height_ != height) {
            std::stringstream err;
            err << "File " << filename << " does not have the same resolution as previous files. ("
                << width << "x" << height
                << " instead of "
                << max_width_ << "x" << max_height_ << "). "
                << "Install Nvidia driver version >=396 (x86) or >=415 (Power PC) to decode"
                   " multiple resolutions";
            DALI_WARN(err.str());
            throw unsupported_exception("Decoder reconfigure feature not supported");
        }
      }
    }
    file.stream_base_ = stream->time_base;
    // 1/frame_rate is duration of each frame (or time base of frame_num)
    file.frame_base_ = AVRational{stream->avg_frame_rate.den,
                                  stream->avg_frame_rate.num};
    file.start_time_ = stream->start_time;
    if (file.start_time_ == AV_NOPTS_VALUE)
      file.start_time_ = 0;
    LOG_LINE
      << "\nStart time is: " << file.start_time_
      << "\nStream base: " << file.stream_base_
      << "\nFrame base: " << file.frame_base_ << "\n";

    // This check is based on heuristic FFMPEG API
    AVPacket pkt = AVPacket{};
    while ((ret = av_read_frame(file.fmt_ctx_.get(), &pkt)) >= 0) {
      if (pkt.stream_index == file.vid_stream_idx_) break;
      av_packet_unref(&pkt);
    }
    auto pkt_duration = pkt.duration;
    av_packet_unref(&pkt);

    if (ret < 0) {
      DALI_WARN(make_string("Unable to read frame from file :", filename));
      open_files_.erase(filename);
      return empty_file;
    }

    DALI_ENFORCE(skip_vfr_check_ ||
      almost_equal(av_q2d(file.frame_base_), pkt_duration * av_q2d(file.stream_base_), 2),
      "Variable frame rate videos are unsupported. This heuristic can yield false positives. "
      "The check can be disabled via the skip_vfr_check flag. Check failed for file: " + filename);
    // empty the read buffer from av_read_frame to save the memory usage
    avformat_flush(file.fmt_ctx_.get());

    auto duration = stream->duration;
    // if no info in the stream check the container
    if (duration == AV_NOPTS_VALUE)
      duration = file.fmt_ctx_->duration / 1000;

    file.frame_count_ = av_rescale_q(duration,
                                     stream->time_base,
                                     file.frame_base_);

    if (codec_id == AV_CODEC_ID_H264 || codec_id == AV_CODEC_ID_HEVC ||
        codec_id == AV_CODEC_ID_MPEG4 || codec_id == AV_CODEC_ID_VP8 ||
        codec_id == AV_CODEC_ID_VP9 || codec_id == AV_CODEC_ID_MJPEG) {
      const char* filtername = nullptr;
      if (codec_id == AV_CODEC_ID_H264) {
        filtername = "h264_mp4toannexb";
      } else if (codec_id == AV_CODEC_ID_MPEG4 && !strcmp(file.fmt_ctx_->iformat->name, "avi")) {
        filtername = "mpeg4_unpack_bframes";
      } else if (codec_id == AV_CODEC_ID_HEVC) {
        filtername = "hevc_mp4toannexb";
      }

#if HAVE_AVBSFCONTEXT
      AVBSFContext* raw_bsf_ctx_ = nullptr;
      if (filtername != nullptr) {
        auto bsf = av_bsf_get_by_name(filtername);
        if (!bsf) {
          DALI_FAIL("Error finding bit stream filter.");
        }
        if (av_bsf_alloc(bsf, &raw_bsf_ctx_) < 0) {
          DALI_FAIL("Error allocating bit stream filter context.");
        }
      } else {
        if (av_bsf_get_null_filter(&raw_bsf_ctx_) < 0) {
          DALI_FAIL("Error creating pass-through filter.");
        }
      }

      file.bsf_ctx_ = make_unique_av<AVBSFContext>(raw_bsf_ctx_, av_bsf_free);

      if (avcodec_parameters_copy(file.bsf_ctx_->par_in, codecpar(stream)) < 0) {
        DALI_FAIL("Error setting BSF parameters.");
      }

      if (av_bsf_init(file.bsf_ctx_.get()) < 0) {
        DALI_FAIL("Error initializing BSF.");
      }

      if (avcodec_parameters_copy(codecpar(stream), file.bsf_ctx_->par_out) < 0) {
        DALI_FAIL("Cannot save BSF parameters.");
      }
#else
      auto raw_bsf_ctx_ = av_bitstream_filter_init(filtername);
      if (!raw_bsf_ctx_) {
        DALI_FAIL("Error finding h264_mp4toannexb bit stream filter.");
      }
      file.bsf_ctx_ = VideoFile::bsf_ptr{raw_bsf_ctx_};
      file.codec = stream->codec;
#endif
    } else {
      std::stringstream err;
      err << "Unhandled codec " << codec_id << " in " << filename;
      DALI_FAIL(err.str());
    }
  } else {
    /* Flush the bitstream filter handle when using mpeg4_unpack_bframes filter.
     * When mpeg4_unpack_bframe is used the filter handle stores information
     * about packed B frames. When we seek in a stream this can confuse the filter
     * and cause it to drop B-Frames.
     */
    auto stream = file.fmt_ctx_->streams[file.vid_stream_idx_];
    if (codecpar(stream)->codec_id ==  AV_CODEC_ID_MPEG4 &&
        !strcmp(file.fmt_ctx_->iformat->name, "avi")) {
      av_bsf_flush(file.bsf_ctx_.get());
    }
  }
  // close the previous file if there was any open
  if (last_opened_.size() && last_opened_ != filename) {
    auto& old_file = open_files_[last_opened_];
    if (old_file.file_desc.file_stream) {
      old_file.file_desc.file_position = ftell(old_file.file_desc.file_stream);
      fclose(old_file.file_desc.file_stream);
      old_file.file_desc.file_stream = nullptr;
    }
  }
  last_opened_ = filename;
  return file;
}

void VideoLoader::seek(VideoFile& file, int frame) {
    auto seek_time = av_rescale_q(frame, file.frame_base_, file.stream_base_) + file.start_time_;
    LOG_LINE << "Seeking to frame " << frame << " timestamp " << seek_time << std::endl;

    auto ret = av_seek_frame(file.fmt_ctx_.get(), file.vid_stream_idx_,
                             seek_time, AVSEEK_FLAG_BACKWARD);

    if (ret < 0) {
      LOG_LINE << "Unable to skip to ts " << seek_time
                << ": " << av_err2str(ret) << std::endl;
    }
    /* Flush the bitstream filter handle when using mpeg4_unpack_bframes filter.
     * When mpeg4_unpack_bframe is used the filter handle stores information
     * about packed B frames. When we seek in a stream this can confuse the filter
     * and cause it to drop B-Frames.
     */
    av_bsf_flush(file.bsf_ctx_.get());

    // todo this seek may be unreliable and will sometimes start after
    // the promised time step.  So we need to calculate the end_time
    // after we actually get a frame to see where we are really
    // starting.
}

void VideoLoader::read_file() {
  // av_packet_unref is unlike the other libav free functions
  using pkt_ptr = std::unique_ptr<AVPacket, decltype(&av_packet_unref)>;
  AVPacket raw_pkt = {};

  auto req = send_queue_.pop();

  LOG_LINE << "Got a request for " << req.filename << " frame " << req.frame
            << " count " << req.count << " send_queue_ has " << send_queue_.size()
            << " frames left" << std::endl;



  auto& file = get_or_open_file(req.filename);
  if (file.empty()) {
    if (vid_decoder_) {
      vid_decoder_->finish();
    }
    DALI_FAIL(make_string("Cannot open video file for file: ", req.filename));
  }
  auto stream = file.fmt_ctx_->streams[file.vid_stream_idx_];
  req.frame_base = file.frame_base_;

  if (vid_decoder_) {
      vid_decoder_->push_req(req);
  } else {
      DALI_FAIL(make_string("No video decoder even after opening a file: ", req.filename));
  }

  // we want to seek each time because even if we ended on the
  // correct key frame, we've flushed the decoder, so it needs
  // another key frame to start decoding again
  int seek_hack = 1;
  seek(file, req.frame);

  auto nonkey_frame_count = 0;

  bool is_first_frame = true;
  int last_key_frame = -1;
  int previous_last_key_frame = -1;
  bool previous_last_key_frame_updated = false;
  bool key = false;
  bool seek_must_succeed = false;
  // how many key frames following the last requested frames we saw so far
  int key_frames_count = 0;
  // how many key frames following the last requested frames we need to see before we stop
  // feeding the decoder
  const int key_frames_treshold = 2;
  int frames_send = 0;
  VidReqStatus dec_status = VidReqStatus::REQ_IN_PROGRESS;

  int read_frame_ret = 0;
  while ((read_frame_ret = av_read_frame(file.fmt_ctx_.get(), &raw_pkt)) >= 0 ||
          dec_status == VidReqStatus::REQ_NOT_STARTED) {
    pkt_ptr pkt = {nullptr, av_packet_unref};
    int64_t frame = 0;

    if (read_frame_ret >= 0) {
      pkt = pkt_ptr(&raw_pkt, av_packet_unref);

      stats_.bytes_read += pkt->size;
      stats_.packets_read++;

      if (pkt->stream_index != file.vid_stream_idx_) {
          continue;
      }

      frame = av_rescale_q(pkt->pts - file.start_time_,
                                file.stream_base_,
                                file.frame_base_);
      LOG_LINE << "Frame candidate " << frame << " (for " << req.frame  <<" )...\n";

      file.last_frame_ = frame;
      key = (pkt->flags & AV_PKT_FLAG_KEY) != 0;
    }

    // if decoding hasn't produced any frames after providing kStartupFrameThreshold frames,
    // or we are at next key frame
    if (last_key_frame != -1 &&
        ((key && last_key_frame != frame && last_key_frame != 0) ||
          frames_send > kStartupFrameThreshold ||
          read_frame_ret < 0) &&
        dec_status == VidReqStatus::REQ_NOT_STARTED) {
        if (last_key_frame <= 0) {
          if (vid_decoder_) {
            vid_decoder_->finish();
          }
          DALI_FAIL(make_string("Decoding not started when starting from frame 0, "
                    "no point in seeking to preceding keyframe which is "
                    "already 0 for file: ", req.filename));
        }

      if (previous_last_key_frame < 0) {
        previous_last_key_frame = last_key_frame - 1;
        previous_last_key_frame_updated = true;
      }
      if (previous_last_key_frame_updated == false) {
        --previous_last_key_frame;
      }
      LOG_LINE << "Decoding not started, seek to preceding key frame, "
               << " frames_send vs kStartupFrameThreshold: "
               << frames_send << " / " << kStartupFrameThreshold
               << ", av_read_frame result: "
               << read_frame_ret
               << ", current frame " << frame
               << ", look for a key frame before " << previous_last_key_frame
               << ", is_key " << key << std::endl;
      seek(file, previous_last_key_frame);
      previous_last_key_frame_updated = false;
      frames_send = 0;
      last_key_frame = -1;
      continue;
    }

    DALI_ENFORCE(pkt, make_string("Failed to read frames for file: ", req.filename));

    if (key) {
      last_key_frame = frame;
      if (frame < previous_last_key_frame) {
        previous_last_key_frame = frame;
        previous_last_key_frame_updated = true;
      }
    }

    int pkt_frames = 1;
    if (pkt->duration) {
      pkt_frames = av_rescale_q(pkt->duration, file.stream_base_, file.frame_base_);
      LOG_LINE << "Duration: " << pkt->duration
                << "\nPacket contains " << pkt_frames << " frames\n";
    }

    // Or we didn't get to a key frame as the first one or we went too far
    if ((!key && is_first_frame) ||
        (key && frame > req.frame && is_first_frame)) {
        LOG_LINE << device_id_ << ": We got ahead of ourselves! "
                  << frame << " > " << req.frame << " + "
                  << nonkey_frame_count
                  << " seek_hack = " << seek_hack << std::endl;
        if (seek_must_succeed) {
          std::stringstream ss;
          ss << device_id_ << ": failed to seek frame "
              << req.frame << " for file: " << req.filename;
          if (vid_decoder_) {
            vid_decoder_->finish();
          }
          DALI_FAIL(ss.str());
        }
        if (req.frame > seek_hack) {
          seek(file, req.frame - seek_hack);
          frames_send = 0;
          seek_hack *= 2;
          last_key_frame = -1;
        } else {
          seek_must_succeed = true;
          seek(file, 0);
          frames_send = 0;
          last_key_frame = -1;
        }
        continue;
    }

    LOG_LINE << device_id_ << ": Sending " << (key ? "  key " : "nonkey")
                << " frame " << frame << " to the decoder."
                << " size = " << pkt->size
                << " req.frame = " << req.frame
                << " req.count = " << req.count
                << " nonkey_frame_count = " << nonkey_frame_count
                << std::endl;

    stats_.bytes_decoded += pkt->size;
    stats_.packets_decoded++;

    if (file.bsf_ctx_ && pkt->size > 0) {
      int ret;
#if HAVE_AVBSFCONTEXT
      auto raw_filtered_pkt = AVPacket{};

      if ((ret = av_bsf_send_packet(file.bsf_ctx_.get(), pkt.release())) < 0) {
        if (vid_decoder_) {
          vid_decoder_->finish();
        }
        DALI_FAIL(make_string("BSF send packet failed: ", av_err2str(ret),
                              " for file: ", req.filename));
      }
      while ((ret = av_bsf_receive_packet(file.bsf_ctx_.get(), &raw_filtered_pkt)) == 0) {
        auto fpkt = pkt_ptr(&raw_filtered_pkt, av_packet_unref);
        ++frames_send;
        dec_status = vid_decoder_->decode_packet(fpkt.get(), file.start_time_, file.stream_base_,
                                    codecpar(stream));
      }
      if (ret != AVERROR(EAGAIN)) {
        if (vid_decoder_) {
          vid_decoder_->finish();
        }
        DALI_FAIL(make_string("BSF receive packet failed: ", av_err2str(ret),
                              " for file: ", req.filename));
      }
#else
      AVPacket fpkt;
      for (auto bsf = file.bsf_ctx_.get(); bsf; bsf = bsf->next) {
        fpkt = *pkt.get();
        ret = av_bitstream_filter_filter(bsf, file.codec, nullptr,
                                          &fpkt.data, &fpkt.size,
                                          pkt->data, pkt->size,
                                          !!(pkt->flags & AV_PKT_FLAG_KEY));
        if (ret < 0) {
            if (vid_decoder_) {
              vid_decoder_->finish();
            }
            DALI_FAIL(make_string("BSF error: ", av_err2str(ret), " for file: ", req.filename));
        }
        if (ret == 0 && fpkt.data != pkt->data) {
          // fpkt is an offset into pkt, copy the smaller portion to the start
          if ((ret = av_copy_packet(&fpkt, pkt.get())) < 0) {
            av_free(fpkt.data);
            if (vid_decoder_) {
              vid_decoder_->finish();
            }
            DALI_FAIL(make_string("av_copy_packet error: ", av_err2str(ret),
                                  " for file: ", req.filename));
          }
          ret = 1;
        }
        if (ret > 0) {
          /* free the buffer in pkt and replace it with the newly
          created buffer in fpkt */
          av_free_packet(pkt.get());
          fpkt.buf = av_buffer_create(fpkt.data, fpkt.size, av_buffer_default_free,
                                      nullptr, 0);
          if (!fpkt.buf) {
              av_free(fpkt.data);
              if (vid_decoder_) {
                vid_decoder_->finish();
              }
              DALI_FAIL(make_string("Unable to create buffer during bsf for file: ", req.filename));
          }
        }
        *pkt.get() = fpkt;
      }
      ++frames_send;
      dec_status = vid_decoder_->decode_packet(pkt.get(), file.start_time_, file.stream_base_,
                                  codecpar(stream));
#endif
    } else {
      ++frames_send;
      dec_status = vid_decoder_->decode_packet(pkt.get(), file.start_time_, file.stream_base_,
                                  codecpar(stream));
    }
    is_first_frame = false;
    if (dec_status == VidReqStatus::REQ_READY) {
      LOG_LINE << "Request completed" << std::endl;
      break;
    } else if (dec_status == VidReqStatus::REQ_ERROR) {
      LOG_LINE << "Request failed" << std::endl;
      if (vid_decoder_) {
          vid_decoder_->finish();
        }
      DALI_FAIL(make_string("The decoder returned a frame that is past the expected one. ",
                "The most likely cause is variable frame rate video. ",
                "Filename: ", req.filename));
    }
  }

  // flush the decoder
  vid_decoder_->decode_packet(nullptr, 0, {0}, 0);
  // empty the read buffer from av_read_frame to save the memory usage
  avformat_flush(file.fmt_ctx_.get());
}

void VideoLoader::push_sequence_to_read(std::string filename, int frame, int count) {
    int total_count = 1 + (count - 1) * stride_;
    auto req = FrameReq{std::move(filename), frame, total_count, stride_, {0, 0}};
    // give both reader thread and decoder a copy of what is coming
    send_queue_.push(std::move(req));
}

void VideoLoader::receive_frames(SequenceWrapper& sequence) {
  auto startup_timeout = 1000;
  while (!vid_decoder_) {
    usleep(500);
    if (startup_timeout-- == 0) {
      DALI_FAIL("Timeout waiting for a valid decoder");
    }
  }
  vid_decoder_->receive_frames(sequence);

  // Stats code
  stats_.frames_used += sequence.count;

  static auto frames_since_warn = 0;
  static auto frames_used_warned = false;
  frames_since_warn += sequence.count;
  auto ratio_used = static_cast<float>(stats_.packets_decoded) / stats_.frames_used;
  if (ratio_used > frames_used_warning_ratio &&
      frames_since_warn > (frames_used_warned ? frames_used_warning_interval :
                            frames_used_warning_minimum)) {
    frames_since_warn = 0;
    frames_used_warned = true;
    LOG_LINE << "\e[1mThe video loader is performing suboptimally due to reading "
             << std::setprecision(2) << ratio_used << "x as many packets as "
             << "frames being used.\e[0m  Consider reencoding the video with a "
             << "smaller key frame interval (GOP length).";
  }
}

void VideoLoader::PrepareEmpty(SequenceWrapper &tensor) {}

void VideoLoader::ReadSample(SequenceWrapper& tensor) {
    // TODO(spanev) remove the async between the 2 following methods?
    auto& seq_meta = frame_starts_[current_frame_idx_];
    tensor.initialize(seq_meta.length, count_, seq_meta.height, seq_meta.width, channels_, dtype_);

    tensor.read_sample_f = [this,
                            file_name = file_info_[seq_meta.filename_idx].filename,
                            index = seq_meta.frame_idx, count = seq_meta.length, &tensor] () {
      thread_file_reader_.DoWork([this]() {
        read_file();
      });
      push_sequence_to_read(file_name, index, count);
      receive_frames(tensor);
      thread_file_reader_.WaitForWork();
    };
    ++current_frame_idx_;

    tensor.label = seq_meta.label;
    tensor.first_frame_idx = seq_meta.frame_idx;
    tensor.sequence.SetSourceInfo(file_info_[seq_meta.filename_idx].filename);
    MoveToNextShard(current_frame_idx_);
}

void VideoLoader::Skip() {
  MoveToNextShard(++current_frame_idx_);
}

Index VideoLoader::SizeImpl() {
    return static_cast<Index>(frame_starts_.size());
}

}  // namespace dali
