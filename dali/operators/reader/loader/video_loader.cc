// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
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
#include "dali/operators/reader/loader/video_loader.h"

#include <dirent.h>
#include <unistd.h>

#include <iomanip>
#include <memory>
#include <string>
#include <utility>
#include <fstream>
#include <limits>

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

inline void assemble_video_list(const std::string& path, const std::string& curr_entry, int label,
                        std::vector<std::pair<std::string, int>> &file_label_pairs) {
  std::string curr_dir_path = path + "/" + curr_entry;
  DIR *dir = opendir(curr_dir_path.c_str());
  DALI_ENFORCE(dir != nullptr, "Directory " + curr_dir_path + " could not be opened");

  struct dirent *entry;

  while ((entry = readdir(dir))) {
    std::string full_path = curr_dir_path + "/" + std::string{entry->d_name};
#ifdef _DIRENT_HAVE_D_TYPE
    /*
     * Regular files and symlinks supported. If FS returns DT_UNKNOWN,
     * filename is validated.
     */
    if (entry->d_type != DT_REG && entry->d_type != DT_LNK &&
        entry->d_type != DT_UNKNOWN) {
      continue;
    }
#endif
    file_label_pairs.push_back(std::make_pair(full_path, label));
  }
  closedir(dir);
}

vector<std::pair<string, int>> filesystem::get_file_label_pair(
    const std::string& file_root,
    const std::vector<std::string>& filenames,
    const std::string& file_list) {
  // open the root
  std::vector<std::pair<std::string, int>> file_label_pairs;
  std::vector<std::string> entry_name_list;

  if (!file_root.empty()) {
    DIR *dir = opendir(file_root.c_str());

    DALI_ENFORCE(dir != nullptr,
        "Directory " + file_root + " could not be opened.");

    struct dirent *entry;

    while ((entry = readdir(dir))) {
      struct stat s;
      std::string entry_name(entry->d_name);
      std::string full_path = file_root + "/" + entry_name;
      int ret = stat(full_path.c_str(), &s);
      DALI_ENFORCE(ret == 0,
          "Could not access " + full_path + " during directory traversal.");
      if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) continue;
      if (S_ISDIR(s.st_mode)) {
        entry_name_list.push_back(entry_name);
      }
    }
    closedir(dir);
    // sort directories to preserve class alphabetic order, as readdir could
    // return unordered dir list. Otherwise file reader for training and validation
    // could return directories with the same names in completely different order
    std::sort(entry_name_list.begin(), entry_name_list.end());
    for (unsigned dir_count = 0; dir_count < entry_name_list.size(); ++dir_count) {
        assemble_video_list(file_root, entry_name_list[dir_count], dir_count, file_label_pairs);
    }

    // sort file names as well
    std::sort(file_label_pairs.begin(), file_label_pairs.end());
  } else if (!file_list.empty()) {
    // load (path, label) pairs from list
    std::ifstream s(file_list);
    DALI_ENFORCE(s.is_open(), file_list + " could not be opened.");

    string video_file;
    int label;
    while (s >> video_file >> label) {
      file_label_pairs.push_back(std::make_pair(video_file, label));
    }

    DALI_ENFORCE(s.eof(), "Wrong format of file_list.");
    s.close();
  } else {
    for (unsigned file_count = 0; file_count < filenames.size(); ++file_count)
        file_label_pairs.push_back(std::make_pair(filenames[file_count], 0));
  }

  LOG_LINE << "read " << file_label_pairs.size() << " files from "
              << entry_name_list.size() << " directories\n";

  return file_label_pairs;
}

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

OpenFile& VideoLoader::get_or_open_file(const std::string &filename) {
  auto& file = open_files_[filename];

  if (!file.open) {
    LOG_LINE << "Opening file " << filename << std::endl;

    AVFormatContext* raw_fmt_ctx = nullptr;
    int ret = avformat_open_input(&raw_fmt_ctx, filename.c_str(), NULL, NULL);
    if (ret < 0) {
      DALI_FAIL(std::string("Could not open file ") + filename + " because of " + av_err2str(ret));
    }
    file.fmt_ctx_ = make_unique_av<AVFormatContext>(raw_fmt_ctx, avformat_close_input);

    LOG_LINE << "File open " << filename << std::endl;

    // is this needed?
    if (avformat_find_stream_info(file.fmt_ctx_.get(), nullptr) < 0) {
      DALI_FAIL(std::string("Could not find stream information in ")
                                + filename);
    }

    LOG_LINE << "File info fetched for " << filename << std::endl;

    if (file.fmt_ctx_->nb_streams > 1) {
      LOG_LINE << "There are " << file.fmt_ctx_->nb_streams << " streams in "
                  << filename << " which will degrade performance. "
                  << "Consider removing all but the main video stream."
                  << std::endl;
    }

    file.vid_stream_idx_ = av_find_best_stream(file.fmt_ctx_.get(), AVMEDIA_TYPE_VIDEO,
                                  -1, -1, nullptr, 0);
    LOG_LINE << "Best stream " << file.vid_stream_idx_ << " found for "
              << filename << std::endl;
    if (file.vid_stream_idx_ < 0) {
      DALI_FAIL(std::string("Could not find video stream in ") + filename);
    }


    auto stream = file.fmt_ctx_->streams[file.vid_stream_idx_];
    int width = codecpar(stream)->width;
    int height = codecpar(stream)->height;
    auto codec_id = codecpar(stream)->codec_id;
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
    }

    DALI_ENFORCE(ret >=0, "Unable to read frame from file :" + filename);

    DALI_ENFORCE(skip_vfr_check_ ||
      almost_equal(av_q2d(file.frame_base_), pkt.duration * av_q2d(file.stream_base_), 2),
      "Variable frame rate videos are unsupported. Check failed for file: " + filename);

    file.frame_count_ = av_rescale_q(stream->duration,
                                     stream->time_base,
                                     file.frame_base_);

    if (codec_id == AV_CODEC_ID_H264 || codec_id == AV_CODEC_ID_HEVC ||
        codec_id == AV_CODEC_ID_MPEG4 || codec_id == AV_CODEC_ID_VP9) {
      const char* filtername = nullptr;
      if (codec_id == AV_CODEC_ID_H264) {
        filtername = "h264_mp4toannexb";
      } else if (codec_id == AV_CODEC_ID_MPEG4 && !strcmp(file.fmt_ctx_->iformat->name, "avi")) {
        filtername = "mpeg4_unpack_bframes";
      } else if (codec_id == AV_CODEC_ID_HEVC) {
        filtername = "hevc_mp4toannexb";
      } else {
        filtername = "null";
      }

#if HAVE_AVBSFCONTEXT
      auto bsf = av_bsf_get_by_name(filtername);
      if (!bsf) {
        DALI_FAIL("Error finding bit stream filter.");
      }
      AVBSFContext* raw_bsf_ctx_ = nullptr;
      if (av_bsf_alloc(bsf, &raw_bsf_ctx_) < 0) {
        DALI_FAIL("Error allocating bit stream filter context.");
      }
      file.bsf_ctx_ = make_unique_av<AVBSFContext>(raw_bsf_ctx_, av_bsf_free);

      if (avcodec_parameters_copy(file.bsf_ctx_->par_in, codecpar(stream)) < 0) {
        DALI_FAIL("Error setting BSF parameters.");
      }

      if (av_bsf_init(file.bsf_ctx_.get()) < 0) {
        DALI_FAIL("Error initializing BSF.");
      }

      avcodec_parameters_copy(codecpar(stream), file.bsf_ctx_->par_out);
#else
      auto raw_bsf_ctx_ = av_bitstream_filter_init(filtername);
      if (!raw_bsf_ctx_) {
        DALI_FAIL("Error finding h264_mp4toannexb bit stream filter.");
      }
      file.bsf_ctx_ = OpenFile::bsf_ptr{raw_bsf_ctx_};
      file.codec = stream->codec;
#endif
    } else {
      std::stringstream err;
      err << "Unhandled codec " << codec_id << " in " << filename;
      DALI_FAIL(err.str());
    }
    file.open = true;
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
  return file;
}

void VideoLoader::seek(OpenFile& file, int frame) {
    auto seek_time = av_rescale_q(frame, file.frame_base_, file.stream_base_) + file.start_time_;
    LOG_LINE << "Seeking to frame " << frame << " timestamp " << seek_time << std::endl;

    auto ret = av_seek_frame(file.fmt_ctx_.get(), file.vid_stream_idx_,
                             seek_time, AVSEEK_FLAG_BACKWARD);

    if (ret < 0) {
      LOG_LINE << "Unable to skip to ts " << seek_time
                << ": " << av_err2str(ret) << std::endl;
    }

    // todo this seek may be unreliable and will sometimes start after
    // the promised time step.  So we need to calculate the end_time
    // after we actually get a frame to see where we are really
    // starting.
}

void VideoLoader::read_file() {
  // av_packet_unref is unlike the other libav free functions
  using pkt_ptr = std::unique_ptr<AVPacket, decltype(&av_packet_unref)>;
  AVPacket raw_pkt = {};
  int seek_hack = 1;

  while (!stop_) {
    if (stop_) {
      break;
    }

    auto req = send_queue_.pop();

    LOG_LINE << "Got a request for " << req.filename << " frame " << req.frame
             << " count " << req.count << " send_queue_ has " << send_queue_.size()
             << " frames left" << std::endl;

    if (stop_) {
      break;
    }

    auto& file = get_or_open_file(req.filename);
    req.frame_base = file.frame_base_;

    if (vid_decoder_) {
        vid_decoder_->push_req(req);
    } else {
        DALI_FAIL("No video decoder even after opening a file");
    }

    // we want to seek each time because even if we ended on the
    // correct key frame, we've flushed the decoder, so it needs
    // another key frame to start decoding again
    seek(file, req.frame);

    auto nonkey_frame_count = 0;
    int frames_left = req.count;
    std::vector<bool> frames_read(frames_left, false);

    bool is_first_frame = true;
    bool key = false;
    bool seek_must_succeed = false;

    while (av_read_frame(file.fmt_ctx_.get(), &raw_pkt) >= 0) {
      auto pkt = pkt_ptr(&raw_pkt, av_packet_unref);

      stats_.bytes_read += pkt->size;
      stats_.packets_read++;

      if (pkt->stream_index != file.vid_stream_idx_) {
          continue;
      }

      auto frame = av_rescale_q(pkt->pts - file.start_time_,
                                file.stream_base_,
                                file.frame_base_);
      LOG_LINE << "Frame candidate " << frame << " (for " << req.frame  <<" )...\n";

      file.last_frame_ = frame;
      key = (pkt->flags & AV_PKT_FLAG_KEY) != 0;
      int pkt_frames = 1;
      if (pkt->duration) {
        pkt_frames = av_rescale_q(pkt->duration, file.stream_base_, file.frame_base_);
        LOG_LINE << "Duration: " << pkt->duration
                 << "\nPacket contains " << pkt_frames << " frames\n";
      }

      if (frame > req.frame) {
        if (key) {
          if (frames_left <= 0)
            break;
          if (is_first_frame) {
            LOG_LINE << device_id_ << ": We got ahead of ourselves! "
                          << frame << " > " << req.frame << " + "
                          << nonkey_frame_count
                          << " seek_hack = " << seek_hack << std::endl;
            if (seek_must_succeed) {
              std::stringstream ss;
              ss << device_id_ << ": failed to seek frame "
                  << req.frame;
              DALI_FAIL(ss.str());
            }
            if (req.frame > seek_hack) {
              seek(file, req.frame - seek_hack);
              seek_hack *= 2;
            } else {
              seek_must_succeed = true;
              seek(file, 0);
            }
            continue;
          } else {
            nonkey_frame_count = 0;
          }
          seek_must_succeed = false;
        } else {
          nonkey_frame_count += pkt_frames;
          // A heuristic so we don't go way over... what should "20" be?
          if (frames_left <= 0 && frame > req.frame + req.count + 20) {
            break;
          }
        }
      }

      if (frame >= req.frame && frame < req.frame + req.count) {
        if (frames_read[frame - req.frame]) {
          ERROR_LOG << "Frame " << frame << " appeared twice\n";
        } else {
          frames_read[frame - req.frame] = true;
          frames_left--;
          LOG_LINE << "Frames left: " << frames_left << "\n";
        }
      } else {
        LOG_LINE << "Frame " << frame << " not in the interesting range.\n";
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
          DALI_FAIL(std::string("BSF send packet failed:") + av_err2str(ret));
        }
        while ((ret = av_bsf_receive_packet(file.bsf_ctx_.get(), &raw_filtered_pkt)) == 0) {
          auto fpkt = pkt_ptr(&raw_filtered_pkt, av_packet_unref);
          vid_decoder_->decode_packet(fpkt.get(), file.start_time_);
        }
        if (ret != AVERROR(EAGAIN)) {
          DALI_FAIL(std::string("BSF receive packet failed:") + av_err2str(ret));
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
              DALI_FAIL(std::string("BSF error:") + av_err2str(ret));
          }
          if (ret == 0 && fpkt.data != pkt->data) {
            // fpkt is an offset into pkt, copy the smaller portion to the start
            if ((ret = av_copy_packet(&fpkt, pkt.get())) < 0) {
              av_free(fpkt.data);
              DALI_FAIL(std::string("av_copy_packet error:") + av_err2str(ret));
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
                DALI_FAIL(std::string("Unable to create buffer during bsf"));
            }
          }
          *pkt.get() = fpkt;
        }
        vid_decoder_->decode_packet(pkt.get(), file.start_time_);
#endif
      } else {
        vid_decoder_->decode_packet(pkt.get(), file.start_time_);
      }
      is_first_frame = false;
    }

    // flush the decoder
    vid_decoder_->decode_packet(nullptr, 0);
  }  // while not done

  if (vid_decoder_) {
    // stop decoding
    vid_decoder_->decode_packet(nullptr, 0);
  }
  LOG_LINE << "Leaving read_file" << std::endl;
}

void VideoLoader::push_sequence_to_read(std::string filename, int frame, int count) {
    int total_count = 1 + (count - 1) * stride_;
    auto req = FrameReq{std::move(filename), frame, total_count, stride_, {0, 0}};
    // give both reader thread and decoder a copy of what is coming
    send_queue_.push(req);
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
  // We have to wait for all kernel recorded in sequence's event are completed
  LOG_LINE << "Waiting for sequence..";
  sequence.wait();
  LOG_LINE << ".got sequence\n";
}

void VideoLoader::PrepareEmpty(SequenceWrapper &tensor) {
  tensor.sequence.Resize({tensor_init_bytes_});
}

void VideoLoader::ReadSample(SequenceWrapper& tensor) {
    // TODO(spanev) remove the async between the 2 following methods?
    auto& seq_meta = frame_starts_[current_frame_idx_];
    tensor.initialize(count_, seq_meta.height, seq_meta.width, channels_);

    push_sequence_to_read(file_label_pair_[seq_meta.filename_idx].first,
                          seq_meta.frame_idx, count_);
    receive_frames(tensor);
    ++current_frame_idx_;

    tensor.label = seq_meta.label;
    MoveToNextShard(current_frame_idx_);
}

Index VideoLoader::SizeImpl() {
    return static_cast<Index>(frame_starts_.size());
}

}  // namespace dali
