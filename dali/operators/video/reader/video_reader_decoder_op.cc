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

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <exception>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "dali/core/boundary.h"
#include "dali/core/cuda_event.h"
#include "dali/core/cuda_stream_pool.h"
#include "dali/core/device_guard.h"
#include "dali/core/exec/thread_pool_base.h"
#include "dali/core/nvtx.h"
#include "dali/core/small_vector.h"
#include "dali/core/span.h"

#include "dali/operators/reader/reader_op.h"
#include "dali/operators/video/frames_decoder_base.h"
#include "dali/operators/video/frames_decoder_cpu.h"
#include "dali/operators/video/frames_decoder_gpu.h"
#include "dali/operators/video/reader/decode_lane_planner.h"
#include "dali/operators/video/reader/frame_index_cache.h"
#include "dali/operators/video/video_utils.h"
#include "dali/pipeline/util/new_thread_pool.h"
#if NVML_ENABLED
#include "dali/util/nvml.h"
#endif

#include "libavutil/rational.h"

namespace dali {

/**
 * @brief What the second phase of PrepareMetadataImpl needs to know about a video file.
 *
 * It is small and does not keep the file open, so that the records of all the files of a list can
 * be alive at the same time.
 */
struct VideoFileIndex {
  bool valid = false;          // false if the file is not a valid video (it is skipped)
  AVCodecID codec_id = AV_CODEC_ID_NONE;  // of the video stream, if valid
  int height = 0, width = 0;   // of the video stream, if valid
  AVPixelFormat pix_fmt = AV_PIX_FMT_NONE;  // of the video stream, if valid (may be unknown)
  std::exception_ptr error;    // set if indexing the file failed with an exception
  // set if valid and the codec is supported (otherwise, the file is skipped), and there's no error
  std::shared_ptr<const FrameIndex> index;
};

/**
 * @brief Indexes one video file. It does not throw: the exceptions are returned in the record.
 *
 * The file is closed before this function returns.
 *
 * The index is taken from FrameIndexCache, or built and cached there. It is not built for the
 * files whose codec is not in `supported_codecs` (made by the decoder's `SupportedCodecs`), which
 * are skipped, as with the decoder. It does not warn about them: the caller does, in the order of
 * the file list. It does not touch CUDA, so it can run on any thread.
 */
static VideoFileIndex IndexVideoFile(const std::string &filename, DALIImageType image_type,
                                     span<const AVCodecID> supported_codecs) {
  VideoFileIndex result;
  try {
    // the version of the file that the indexer opens (see FrameIndexCache::GetOrBuild)
    auto stamp = FrameIndexCache::FileStamp::Of(filename);
    // The index needs no decoder: no codec context, parser, frame buffers nor reopening of
    // the file.
    FramesIndexer indexer(filename, image_type);
    if (!indexer.IsValid()) {
      LOG_LINE << "Invalid video file: " << filename << std::endl;
      return result;
    }
    result.valid = true;
    result.codec_id = indexer.CodecId();
    result.height = indexer.Height();
    result.width = indexer.Width();
    result.pix_fmt = static_cast<AVPixelFormat>(indexer.CodecParameters().format);
    if (std::find(supported_codecs.begin(), supported_codecs.end(), result.codec_id) ==
        supported_codecs.end()) {
      LOG_LINE << "Unsupported codec in video file: " << filename << std::endl;
      return result;
    }
    result.index = FrameIndexCache::instance().GetOrBuild(filename, stamp, [&]() {
      LOG_LINE << "Building index for " << filename << std::endl;
      indexer.BuildIndex();
      return indexer.GetIndex();
    });
  } catch (...) {
    result.error = std::current_exception();
  }
  return result;
}

struct VideoSampleDesc {
  VideoSampleDesc(const VideoFileMeta *video_file_meta = nullptr, int start = -1, int end = -1, int stride = -1)
      : video_file_meta_(video_file_meta), start_(start), end_(end), stride_(stride), frame_idxs_() {}
  const VideoFileMeta *video_file_meta_;
  int start_;
  int end_;
  int stride_;
  span<const int> frame_idxs_;  // non-empty → uniform sampling; use these instead of start_/end_/stride_
};

/** The per-sample values of the metadata outputs. */
struct SampleMeta {
  std::string source_info;
  int32_t label = 0;
  int32_t start = 0;
  std::vector<int32_t> frame_idx;
  std::vector<double> timestamps;
};

template <typename Backend>
struct VideoSample : public VideoSampleDesc {
  VideoSample(const VideoFileMeta *video_file_meta = nullptr, int start = -1, int end = -1, int stride = -1)
      : VideoSampleDesc{video_file_meta, start, end, stride} {
    data_.set_pinned(std::is_same_v<Backend, GPUBackend>);
  }

  VideoSample(const VideoSampleDesc &other)
      : VideoSampleDesc(other) {
    data_.set_pinned(std::is_same_v<Backend, GPUBackend>);
  }

  // the decoded frames, filled by Prefetch (the CPU backend, and the GPU one with the old copying
  // path)
  Tensor<Backend> data_;
  // Without decoding into the output batch (see VideoReaderDecoder::zero_copy_), the metadata of
  // the frames in data_
  SampleMeta meta_;
  // Set when the reader plans the sample's decoding then. With pad_last_batch, the loader returns
  // the last sample again, also in the next batch, which must not decode it again while the
  // previous batch may still be read.
  bool planned = false;
};

enum class FileListFormat {
  kFrames,          // Use exact frame numbers (0-based). Negative values count from end
  kTimestamps,      // Use timestamps in seconds
};

enum class FileListRounding {
  kStartDownEndUp,  // Round start down and end up (default)
  kStartUpEndDown,  // Round start up and end down
  kAllUp,           // Round both up
  kAllDown          // Round both down
};

struct FileListOptions {
  FileListFormat format = FileListFormat::kTimestamps;
  FileListRounding rounding = FileListRounding::kStartDownEndUp;
  bool include_end = false;

  bool should_round_down_start() const {
    return rounding == FileListRounding::kStartDownEndUp || rounding == FileListRounding::kAllDown;
  }

  int64_t round_start(double value) const {
    return should_round_down_start() ? std::floor(value) : std::ceil(value);
  }

  bool should_round_down_end() const {
    return rounding == FileListRounding::kStartUpEndDown || rounding == FileListRounding::kAllDown;
  }

  int64_t round_end(double value) const {
    return should_round_down_end() ? std::floor(value) : std::ceil(value);
  }
};

inline std::string make_string(FileListRounding rounding) {
  switch (rounding) {
    case FileListRounding::kStartDownEndUp:
      return "start_down_end_up";
    case FileListRounding::kStartUpEndDown:
      return "start_up_end_down";
    case FileListRounding::kAllUp:
      return "all_up";
    case FileListRounding::kAllDown:
      return "all_down";
    default:
      DALI_FAIL("Invalid file_list_rounding");
  }
}

inline std::string make_string(FileListFormat format) {
  switch (format) {
    case FileListFormat::kFrames:
      return "frames";
    case FileListFormat::kTimestamps:
      return "timestamps";
    default:
      DALI_FAIL("Invalid file_list_format");
  }
}

inline std::string make_string(FileListOptions options) {
  return make_string(options.format) + ", " + make_string(options.rounding) + ", " +
         (options.include_end ? "include_end" : "exclude_end");
}

namespace detail {

/**
 * @brief The boundary type of the reader: ``pad_mode``, or the legacy ``pad_sequences``
 * (True is ``pad_mode="constant"``, False is ``pad_mode="none"``).
 */
inline boundary::BoundaryType GetReaderBoundaryType(const OpSpec &spec) {
  if (spec.HasArgument("pad_sequences")) {
    DALI_ENFORCE(!spec.HasArgument("pad_mode"),
                 "The legacy argument `pad_sequences` cannot be used together with `pad_mode`. "
                 "Use `pad_mode` only.");
    return spec.GetArgument<bool>("pad_sequences") ? boundary::BoundaryType::CONSTANT
                                                    : boundary::BoundaryType::ISOLATED;
  }
  return GetBoundaryType(spec);
}

/**
 * @brief The options of the ``file_list``, from ``file_list_format``, ``file_list_rounding`` and
 * ``file_list_include_end``, or from the legacy ``file_list_frame_num`` and
 * ``file_list_include_preceding_frame`` that stand for the first two. They are checked also when
 * there is no ``file_list``.
 */
inline FileListOptions GetFileListOptions(const OpSpec &spec) {
  FileListOptions opts;

  if (spec.HasArgument("file_list_frame_num")) {
    DALI_ENFORCE(!spec.HasArgument("file_list_format"),
                 "The legacy argument `file_list_frame_num` cannot be used together with "
                 "`file_list_format`. Use `file_list_format` only.");
    opts.format = spec.GetArgument<bool>("file_list_frame_num") ? FileListFormat::kFrames
                                                                : FileListFormat::kTimestamps;
  } else {
    auto format_str = spec.GetArgument<std::string>("file_list_format");
    if (format_str == "frames") {
      opts.format = FileListFormat::kFrames;
    } else if (format_str == "timestamps") {
      opts.format = FileListFormat::kTimestamps;
    } else {
      DALI_FAIL(make_string("Invalid file_list_format: ", format_str));
    }
  }

  if (spec.HasArgument("file_list_include_preceding_frame")) {
    DALI_ENFORCE(!spec.HasArgument("file_list_rounding"),
                 "The legacy argument `file_list_include_preceding_frame` cannot be used together "
                 "with `file_list_rounding`. Use `file_list_rounding` only.");
    // As in the legacy reader, it only applies to the times: the frame numbers are not rounded
    // by it.
    if (opts.format == FileListFormat::kTimestamps) {
      opts.rounding = spec.GetArgument<bool>("file_list_include_preceding_frame")
                          ? FileListRounding::kStartDownEndUp
                          : FileListRounding::kStartUpEndDown;
    }
  } else {
    auto rounding_str = spec.GetArgument<std::string>("file_list_rounding");
    if (rounding_str == "start_down_end_up") {
      opts.rounding = FileListRounding::kStartDownEndUp;
    } else if (rounding_str == "start_up_end_down") {
      opts.rounding = FileListRounding::kStartUpEndDown;
    } else if (rounding_str == "all_up") {
      opts.rounding = FileListRounding::kAllUp;
    } else if (rounding_str == "all_down") {
      opts.rounding = FileListRounding::kAllDown;
    } else {
      DALI_FAIL(make_string("Invalid file_list_rounding: ", rounding_str));
    }
  }

  opts.include_end = spec.GetArgument<bool>("file_list_include_end");
  return opts;
}

}  // namespace detail

template <typename Backend, typename FramesDecoderImpl, typename Sample = VideoSample<Backend>>
class VideoLoaderDecoder : public Loader<Backend, Sample, true> {
 public:
  explicit inline VideoLoaderDecoder(const OpSpec &spec)
      : Loader<Backend, Sample, true>(spec),
        file_root_(spec.GetArgument<std::string>("file_root")),
        file_list_(spec.GetArgument<std::string>("file_list")),
        filenames_(spec.GetRepeatedArgument<std::string>("filenames")),
        sequence_len_(spec.GetArgument<int>("sequence_length")),
        stride_(spec.GetArgument<int>("stride")),
        step_(spec.GetArgument<int>("step")),
        image_type_(spec.GetArgument<DALIImageType>("image_type")),
        boundary_type_(detail::GetReaderBoundaryType(spec)),
        uniform_sample_(spec.GetArgument<bool>("uniform_sample")),
        num_threads_(std::max(1, spec.GetArgument<int>("num_threads"))) {
    // Like legacy readers.video, count non-empty sources: an explicitly passed empty
    // `file_list=""`, `file_root=""` or `filenames=[]` means "not provided".
    int num_sources = !file_list_.empty() + !file_root_.empty() + !filenames_.empty();
    if (num_sources != 1) {
      throw std::runtime_error(
          make_string("Exactly one of the following arguments must be provided (and "
                      "non-empty): ``file_list``, ``file_root``, ``filenames``. Got ",
                      num_sources, "."));
    }
    // TryGetRepeatedArgument returns true for an explicitly passed empty list too. As in legacy
    // readers.video, `labels=[]` means sequential 0-based labels (generated by GetVideoFiles).
    bool has_labels = spec.TryGetRepeatedArgument(labels_, "labels");
    // As in legacy readers.video, `labels` is meaningful only for `filenames`: `file_root` and
    // `file_list` provide their own labels.
    if (has_labels && filenames_.empty()) {
      throw std::runtime_error(
          "The argument ``labels`` is valid only when file paths "
          "are provided with `filenames` argument.");
    }
    if (has_labels && !labels_.empty()) {
      DALI_ENFORCE(
          labels_.size() == filenames_.size(),
          make_string(
              "Number of provided files and labels should match. Provided ",
              filenames_.size(), " files and ", labels_.size(), " labels."));
    }

    video_files_info_ = GetVideoFiles(file_root_, filenames_, has_labels, labels_, file_list_);
    DALI_ENFORCE(!video_files_info_.empty(), "No files were read.");

    file_list_opts_ = detail::GetFileListOptions(spec);

    // Both feed the default step (stride_ * sequence_len_); zero would make the sample loop in
    // PrepareMetadataImpl never advance.
    DALI_ENFORCE(sequence_len_ >= 1,
                 make_string("sequence_length must be at least 1, got ", sequence_len_, "."));
    if (!uniform_sample_) {
      DALI_ENFORCE(stride_ > 0,
                   make_string("stride must be greater than 0, got ", stride_, "."));
      // Frame indices (and the sample's start_/end_) are int: make sure the sequence extent
      // (stride_ * sequence_len_, which is also the default step) fits, otherwise e.g. the
      // default step would wrap to 0 or a negative value and the sample loop in
      // PrepareMetadataImpl would never terminate.
      int64_t full_seq_stride = static_cast<int64_t>(stride_) * sequence_len_;
      DALI_ENFORCE(full_seq_stride <= std::numeric_limits<int>::max(),
                   make_string("stride * sequence_length must not exceed ",
                               std::numeric_limits<int>::max(), ", got ", stride_, " * ",
                               sequence_len_, " = ", full_seq_stride, "."));
      // `step` <= 0 (including the default, -1) means "use stride * sequence_length".
      if (step_ <= 0) {
        step_ = static_cast<int>(full_seq_stride);
      }
    }
    if (uniform_sample_) {
      if (spec.HasArgument("stride")) {
        DALI_WARN("uniform_sample=True: the `stride` argument is ignored.");
      }
      if (spec.HasArgument("step")) {
        DALI_WARN("uniform_sample=True: the `step` argument is ignored.");
      }
      if (spec.HasArgument("pad_mode") || spec.HasArgument("pad_sequences")) {
        DALI_WARN("uniform_sample=True: the `pad_mode` argument is ignored. "
                  "Frames are repeated when sequence_length exceeds the number of available frames.");
      }
    }
  }

  void PrepareEmpty(Sample &sample) {
    sample = Sample();
  }

  void ReadSample(Sample &sample) override {
    sample = Sample(samples_[current_index_]);
    MoveToNextShard(++current_index_);
  }

  void Skip() override {
    MoveToNextShard(++current_index_);
  }

  const FrameIndex *FileIndex(const std::string &filename) const {
    auto it = indices_.find(filename);
    return it != indices_.end() ? it->second.get() : nullptr;
  }

  Index SizeImpl() override {
    return samples_.size();
  }

  /**
   * @brief Indexes the given distinct files, one record per file, in the same order.
   *
   * The indices are taken from (and added to) FrameIndexCache, which is thread-safe, so the files
   * can be indexed concurrently.
   *
   * Every error is fatal (the files whose codec is not supported are skipped before they are
   * indexed), so the files after the first one that fails are not needed: they are not started,
   * and their records are left empty. With several threads, some of them may have been started
   * already. The files before it are all indexed, so the error reported in the second phase is
   * that of the first failing file in the list, whatever the number of threads.
   */
  std::vector<VideoFileIndex> BuildFileIndices(const std::vector<std::string> &filenames) {
    std::vector<VideoFileIndex> results(filenames.size());
    const int num_workers = std::min<int>(num_threads_, filenames.size());
    // Starting the threads costs about a millisecond: more than it saves only for a few very
    // small files, while for longer ones the parallel build is faster already at a few files.
    const bool parallel = num_workers > 1;

    // Decided here, on the calling thread, before the files are indexed: the codec check of the
    // GPU decoder queries NVDEC of the current device (once per codec, cached per process), which
    // needs a current CUDA context; this thread has the device of the reader, the threads of the
    // pool have none. Then the indexing does not touch CUDA.
    const std::vector<AVCodecID> supported_codecs = FramesDecoderImpl::SupportedCodecs();

    std::atomic<size_t> first_error{filenames.size()};  // the lowest failing slot so far
    auto index_file = [&](size_t i) {
      if (i > first_error.load(std::memory_order_relaxed))
        return;  // a file before this one failed: the second phase does not get to this one
      results[i] = IndexVideoFile(filenames[i], image_type_, make_cspan(supported_codecs));
      if (results[i].error) {
        size_t prev = first_error.load(std::memory_order_relaxed);
        while (i < prev && !first_error.compare_exchange_weak(prev, i, std::memory_order_relaxed)) {
        }
      }
    };

    if (parallel) {
      // A transient pool, destroyed on return. The loader cannot use the pool of the pipeline:
      // this runs in the constructor of the operator or, with lazy_init, on the first use of the
      // reader. The threads need no device. The tasks write only to their own slot, and
      // IndexVideoFile does not throw.
      NewThreadPool pool(num_workers, std::nullopt, false, "VideoIndex");
      Job job;
      for (size_t i = 0; i < filenames.size(); ++i)
        job.AddTask([&index_file, i]() { index_file(i); });
      job.Run(pool, true);
    } else {
      for (size_t i = 0; i < filenames.size(); ++i)
        index_file(i);
    }
    return results;
  }

  /**
   * @brief Builds the list of samples of the reader, in two phases.
   *
   * 1. Every distinct file of the list is indexed once (see BuildFileIndices), possibly in
   *    parallel.
   * 2. The list is walked in order, sequentially. For every entry, the frame range is resolved
   *    (from `file_list`, if given) and the samples (or, with `uniform_sample`, the frame indices
   *    of the sample) are made from the index of its file. The entries that are not valid videos,
   *    whose codec is not supported or whose frame range is empty are skipped (with a warning,
   *    except for the invalid ones). An error that occurred while indexing is thrown here, so the
   *    one reported is that of the first failing file in the list, however many threads there are.
   *
   * Finally, the samples are shuffled (if `shuffle`) and the reader is set to the start of its
   * shard.
   */
  void PrepareMetadataImpl() override {
    LOG_LINE << "Starting PrepareMetadataImpl" << std::endl;
    // With lazy_init, this may run on any thread that asks for the size of the reader (e.g.
    // Pipeline::reader_meta before the first run): the codec checks of the GPU decoder query
    // NVDEC of the current device, which must be the reader's device, with its context current.
    DeviceGuard dg(std::is_same_v<Backend, GPUBackend> ? this->device_id_ : -1);
    samples_.clear();
    all_frame_idxs_.clear();
    if (uniform_sample_)
      all_frame_idxs_.reserve(video_files_info_.size());

    // Phase 1: index each distinct file once, in the order of first appearance in the list.
    std::vector<std::string> filenames;
    std::vector<size_t> file_slot(video_files_info_.size());
    {
      std::unordered_map<std::string, size_t> slots;
      for (size_t i = 0; i < video_files_info_.size(); ++i) {
        const std::string &filename = video_files_info_[i].filename;
        auto [it, inserted] = slots.try_emplace(filename, filenames.size());
        if (inserted)
          filenames.push_back(filename);
        file_slot[i] = it->second;
      }
    }
    std::vector<VideoFileIndex> file_indices = BuildFileIndices(filenames);

    // Phase 2: build the samples, in the order of the list.
    for (size_t i = 0; i < video_files_info_.size(); ++i) {
      auto& entry = video_files_info_[i];
      LOG_LINE << "Processing video file " << i << ": " << entry.filename << std::endl;
      const VideoFileIndex &file_index = file_indices[file_slot[i]];
      if (file_index.error)
        std::rethrow_exception(file_index.error);  // the first failing file, in the list order
      if (!file_index.valid)
        continue;
      if (!file_index.index) {
        // The codec is not supported: skipped, as with the decoder. The check is made again (its
        // result is cached) to warn here, once per entry and in the order of the list, not from
        // the indexing threads.
        FramesDecoderImpl::IsCodecSupported(file_index.codec_id);
        continue;
      }
      // Whether the decoder can decode the format of the stream (for MJPEG, its chroma format and
      // bit depth depend on the GPU), so that the files that it would reject are skipped here,
      // and don't make the reader fail later. It needs CUDA (cached), so it is done on this thread.
      if (!FramesDecoderImpl::IsStreamSupported(file_index.codec_id, file_index.pix_fmt))
        continue;
      const FrameIndex &index = *file_index.index;
      indices_[entry.filename] = file_index.index;  // kept for the reader's lifetime
      // the decoder reads the same codec parameters: the reader sizes its output with these
      entry.height = file_index.height;
      entry.width = file_index.width;
      int64_t num_frames = index.size();
      // The range is resolved in 64 bits, and stored in the (int) fields of the entry only once
      // it is within the frames of the video.
      int64_t start_frame = 0;
      int64_t end_frame = num_frames;
      LOG_LINE << "Total frames in video: " << num_frames << std::endl;
      if (entry.start != 0.0f || entry.end != 0.0f) {
        LOG_LINE << "Processing range [" << entry.start << ", " << entry.end
                 << "], file_list_format: " << make_string(file_list_opts_) << std::endl;
        switch (file_list_opts_.format) {
          case FileListFormat::kFrames: {
            // Negative values count from the end of the video. An end of 0 (or an omitted one) is
            // the end of the video.
            double start = entry.start;
            double end = entry.end;
            if (start < 0)
              start += num_frames;
            if (end <= 0)
              end += num_frames;
            if (start > num_frames) {
              throw std::runtime_error(make_string(
                  "The start frame ", entry.start, " of the file_list entry for \"", entry.filename,
                  "\" is past the end of the video (", num_frames, " frames)."));
            }
            if (file_list_opts_.include_end)
              end += 1;
            // Clamped before the conversion: the values may be too large for an integer.
            start = std::clamp(start, 0.0, static_cast<double>(num_frames));
            end = std::clamp(end, 0.0, static_cast<double>(num_frames));
            start_frame = file_list_opts_.round_start(start);
            end_frame = file_list_opts_.round_end(end);
            break;
          }
          case FileListFormat::kTimestamps: {
            // The times are relative to the first frame. Negative values (and an end of 0, or an
            // omitted one) count from the end of the video.
            const bool round_down_start = file_list_opts_.should_round_down_start();
            const bool round_down_end = file_list_opts_.should_round_down_end();
            const int64_t video_start = index.StartPts();
            const int64_t video_end = index.EndPts();
            int64_t start_ts = (entry.start < 0 ? video_end : video_start) +
                               SecondsToTimestamp(index.timebase, entry.start, round_down_start);
            int64_t end_ts = (entry.end <= 0 ? video_end : video_start) +
                             SecondsToTimestamp(index.timebase, entry.end, round_down_end);
            if (start_ts > video_end) {
              throw std::runtime_error(make_string(
                  "The start time ", entry.start, " of the file_list entry for \"", entry.filename,
                  "\" is past the end of the video (", TimestampToSeconds(index.timebase,
                  video_end - video_start), " s)."));
            }
            start_frame = index.GetFrameIdxByTimestamp(start_ts, round_down_start);
            end_frame = index.GetFrameIdxByTimestamp(end_ts, round_down_end);
            // The end is exclusive: the frame at exactly that time is included on request.
            if (file_list_opts_.include_end && end_frame < num_frames &&
                index[end_frame].pts == end_ts) {
              end_frame++;
            }
            break;
          }
          default:
            DALI_FAIL("Invalid file_list_format");
        }
        LOG_LINE << "Frame range after conversion: [" << start_frame << ", " << end_frame << "]"
                 << std::endl;
      }

      // The padding (see FramesDecoderBase::HandleBoundary) reflects and clamps within
      // [start_frame, end_frame), so the range has to be a subset of the frames of the file.
      // Entries like "-60 10" (a start before the first frame) or "45 60" (an end past the last
      // one) would otherwise map the padded indices to frames outside of the range.
      entry.start_frame = static_cast<int>(std::clamp<int64_t>(start_frame, 0, num_frames));
      entry.end_frame = static_cast<int>(std::clamp<int64_t>(end_frame, 0, num_frames));

      if (entry.start_frame >= entry.end_frame) {
        DALI_WARN(make_string("Empty frame range [", entry.start_frame, ", ", entry.end_frame,
                              ") for file ", entry.filename, ". Skipping."));
        continue;
      }

      if (uniform_sample_) {
        int total_frames = entry.end_frame - entry.start_frame;
        VideoSampleDesc s(&entry, entry.start_frame, entry.end_frame, 1);
        all_frame_idxs_.emplace_back(sequence_len_);
        auto& idxs = all_frame_idxs_.back();
        for (int i = 0; i < sequence_len_; ++i) {
          double t = (sequence_len_ > 1) ? (double)i / (sequence_len_ - 1) : 0.0;
          idxs[i] = entry.start_frame + static_cast<int>(std::round(t * (total_frames - 1)));
        }
        // Moving a std::vector preserves data(), so this span remains valid even if
        // all_frame_idxs_ is reallocated later (inner vectors are moved, not copied).
        s.frame_idxs_ = make_cspan(idxs);
        LOG_LINE << "Adding uniform sample for " << entry.filename
                 << " with " << sequence_len_ << " frames" << std::endl;
        samples_.emplace_back(std::move(s));
      } else {
        // Computed in int64: start + step_ / start + full_seq_stride can exceed INT_MAX even
        // though each term fits in int (the constructor guarantees stride_ * sequence_len_ and
        // step_ do).
        int64_t full_seq_stride = static_cast<int64_t>(stride_) * sequence_len_;
        auto add_sample = [&](int64_t start) {
          int64_t end = start + full_seq_stride;
          DALI_ENFORCE(end <= std::numeric_limits<int>::max(),
                       make_string("Sequence starting at frame ", start, " of \"", entry.filename,
                                   "\" would end at frame ", end, ", which exceeds the supported "
                                   "frame index range. Reduce stride or sequence_length."));
          samples_.emplace_back(&entry, static_cast<int>(start), static_cast<int>(end), stride_);
        };
        // A sequence only needs its last frame to exist: it spans
        // (sequence_length - 1) * stride + 1 frames, not stride * sequence_length (as in the legacy
        // reader). The end of the sample (see add_sample) can be up to stride - 1 frames past the
        // last frame of the range, which the frames decoder does not read.
        int64_t seq_span = static_cast<int64_t>(sequence_len_ - 1) * stride_ + 1;
        int64_t start = entry.start_frame;
        for (; start + seq_span <= entry.end_frame; start += step_) {
          LOG_LINE << "Adding sample with start=" << start << ", end=" << start + full_seq_stride
                   << ", stride=" << stride_ << std::endl;
          add_sample(start);
        }

        // With padding, the sequences that do not fit are padded: there is a sample at every step
        // that remains in the range, as in the legacy reader, with fewer and fewer frames of the
        // range in each.
        if (boundary_type_ != boundary::BoundaryType::ISOLATED) {
          for (; start < entry.end_frame; start += step_) {
            LOG_LINE << "Adding padded sample starting at frame " << start
                     << ", end=" << entry.end_frame << ", stride=" << stride_ << std::endl;
            add_sample(start);
          }
        }
      }
    }

    LOG_LINE << "Created " << samples_.size() << " total samples" << std::endl;

    if (shuffle_) {
      LOG_LINE << "Shuffling samples" << std::endl;
      // seeded with hardcoded value to get
      // the same sequence on every shard
      std::mt19937 g(kDaliDataloaderSeed);
      std::shuffle(std::begin(samples_), std::end(samples_), g);
    }

    // set the initial index for each shard
    Reset(true);
    LOG_LINE << "Finished PrepareMetadataImpl" << std::endl;
  }

  void Reset(bool wrap_to_shard) override {
    current_index_ = wrap_to_shard ? start_index(virtual_shard_id_, num_shards_, SizeImpl()) : 0;
  }

 protected:
  using Base = Loader<Backend, Sample, true>;
  using Base::shard_id_;
  using Base::virtual_shard_id_;
  using Base::num_shards_;
  using Base::stick_to_shard_;
  using Base::shuffle_;
  using Base::dont_use_mmap_;
  using Base::initial_buffer_fill_;
  using Base::copy_read_data_;
  using Base::read_ahead_;
  using Base::IsCheckpointingEnabled;
  using Base::PrepareEmptyTensor;
  using Base::MoveToNextShard;
  using Base::ShouldSkipImage;

  std::string file_root_;
  std::string file_list_;
  std::vector<std::string> filenames_;
  std::vector<int> labels_;

  Index current_index_ = 0;

  int sequence_len_;
  int stride_;
  int step_;
  DALIImageType image_type_;
  boundary::BoundaryType boundary_type_;
  bool uniform_sample_;
  int num_threads_;
  FileListOptions file_list_opts_;

  std::vector<VideoFileMeta> video_files_info_;
  // the frame indices of the valid files, built (or found in FrameIndexCache) by
  // PrepareMetadataImpl
  std::unordered_map<std::string, FrameIndexCache::IndexPtr> indices_;
  std::vector<std::vector<int>> all_frame_idxs_;  // owns frame index data; samples_ hold spans into these
  std::vector<VideoSampleDesc> samples_;
  CUDAStreamLease cuda_stream_;
};

// The labels output is produced for `filenames` with `labels`, and for `file_list` and
// `file_root` (an empty `file_list` or `file_root` counts as not provided).
static bool HasLabelsOutput(const OpSpec &spec) {
  return spec.HasArgument("labels") ||
         !spec.GetArgument<std::string>("file_list").empty() ||
         !spec.GetArgument<std::string>("file_root").empty();
}

template <typename Backend>
class VideoReaderDecoder
    : public DataReader<Backend, VideoSample<Backend>, VideoSample<Backend>, true> {
 public:
  using FramesDecoderImpl =
      std::conditional_t<std::is_same_v<Backend, GPUBackend>, FramesDecoderGpu, FramesDecoderCpu>;
  using VideoLoaderImpl = VideoLoaderDecoder<Backend, FramesDecoderImpl>;
  using Base = DataReader<Backend, VideoSample<Backend>, VideoSample<Backend>, true>;
  using Base::curr_batch_producer_;
  using Base::GetCurrBatchSize;
  using Base::GetSample;
  using Base::loader_;
  using Base::Prefetch;
  using Base::prefetched_batch_queue_;

  explicit VideoReaderDecoder(const OpSpec &spec)
      : Base(spec),
        frame_num_policy_(ParseFrameNumPolicy(spec.GetArgument<std::string>("enable_frame_num"))),
        has_timestamps_(spec.GetArgument<bool>("enable_timestamps")),
        boundary_type_(detail::GetReaderBoundaryType(spec)),
        image_type_(spec.GetArgument<DALIImageType>("image_type")) {
    // Test aid only: a lane's decoder that is not where the plan expects it fails the batch
    const char *strict_plans = std::getenv("DALI_VIDEO_READER_STRICT_PLANS");
    strict_plans_ = strict_plans && *strict_plans && std::string_view(strict_plans) != "0";
    loader_ = InitLoader<VideoLoaderImpl>(spec);
    this->SetInitialSnapshot();

    has_labels_ = HasLabelsOutput(spec);

    auto fill_values = spec.GetRepeatedArgument<int>("fill_value");
    fill_value_.clear();
    fill_value_.reserve(fill_values.size());
    for (auto value : fill_values) {
      DALI_ENFORCE(value >= 0 && value <= 255, "fill_value must be in range [0, 255]");
      fill_value_.push_back(static_cast<uint8_t>(value));
    }
    DALI_ENFORCE(fill_value_.size() >= 1, "fill_value must contain at least one value");

    StreamInitialization(spec);
    DALI_ENFORCE(image_type_ == DALI_RGB || image_type_ == DALI_YCbCr,
                 make_string("Invalid image_type: ", image_type_));

    InitLanes(spec);

    slots_.resize(this->prefetch_queue_depth_);
    if constexpr (std::is_same_v<Backend, GPUBackend>) {
      const char *copy_env = std::getenv("DALI_VIDEO_READER_COPY_OUTPUTS");
      bool copy_outputs = copy_env && std::atoi(copy_env) != 0;
      // without a stream of its own, the reader decodes on the default stream (old drivers)
      zero_copy_ = cuda_stream_ && !copy_outputs;
      if (zero_copy_) {
        for (auto &slot : slots_)
          slot.ready = CUDAEvent::CreateWithFlags(cudaEventDisableTiming, this->device_id_);
      }
    }
  }

  ~VideoReaderDecoder() override {
    LOG_LINE << "VideoReaderDecoder destructor" << std::endl;
    cancel_ = true;  // the lanes skip the samples they have not started
    Base::StopPrefetchThread();  // a Prefetch in progress throws once its batch is done
    StopLanes();
    if constexpr (std::is_same_v<Backend, GPUBackend>) {
      // the lanes may have work on the streams that writes to the buffers of the pending batches
      for (auto &lane : lanes_) {
        if (lane.stream)
          (void)cudaStreamSynchronize(lane.stream.get());
      }
    }
    pending_.clear();  // the samples go back to the loader, which is still alive
    LOG_LINE << "VideoReaderDecoder destructor done" << std::endl;
  }

  void StreamInitialization(const OpSpec &spec) {
    if constexpr (std::is_same_v<Backend, GPUBackend>) {
#if NVML_ENABLED
      auto nvml_handle = nvml::NvmlInstance::CreateNvmlInstance();
      static float driver_version = nvml::GetDriverVersion();
      if (driver_version > 460 && driver_version < 470.21) {
        DALI_WARN_ONCE("Warning: Decoding on a default stream. Performance may be affected.");
        return;
      }
#else
      int driver_cuda_version = 0;
      CUDA_CALL(cuDriverGetVersion(&driver_cuda_version));
      if (driver_cuda_version >= 11030 && driver_cuda_version < 11040) {
        DALI_WARN_ONCE("Warning: Decoding on a default stream. Performance may be affected.");
        return;
      }
#endif
      int device_id = spec.GetArgument<int>("device_id");
      // the batch stream: the output batch is allocated and complete in its order (see zero_copy_)
      cuda_stream_ = CUDAStreamPool::instance().Get(device_id);
      use_streams_ = true;
    }
  }

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) override {
    Base::SetupImpl(output_desc, ws);
    FillOutputDesc(output_desc, CurrSlot());
    // When decoding into the output batch (zero_copy_), RunImpl provides all the outputs.
    return !zero_copy_;
  }

  void RunImpl(Workspace &ws) override {
    auto &slot = CurrSlot();
    if constexpr (std::is_same_v<Backend, GPUBackend>) {
      if (zero_copy_)
        PublishBatch(ws, slot);
    }
    if (!zero_copy_) {
      auto &video_output = ws.Output<Backend>(0);
      int batch_size = GetCurrBatchSize();
      video_output.SetLayout("FHWC");
      AccessOrder order =
          std::is_same_v<Backend, GPUBackend> ? ws.stream() : AccessOrder::host();
      for (int sample_id = 0; sample_id < batch_size; ++sample_id) {
        auto &sample = GetSample(sample_id);
        video_output.CopySample(sample_id, sample.data_, order);
      }
    }

    // Copy optional metadata outputs
    int out_index = 1;
    if (has_labels_) {
      OutputMetadata<int32_t>(ws, out_index++, slot, [](const SampleMeta &m) {
        return make_cspan(&m.label, 1);
      });
    }
    if (frame_num_policy_ == FrameNumPolicy::Scalar) {
      OutputMetadata<int32_t>(ws, out_index++, slot, [](const SampleMeta &m) {
        return make_cspan(&m.start, 1);
      });
    } else if (frame_num_policy_ == FrameNumPolicy::Sequence) {
      OutputMetadata<int32_t>(ws, out_index++, slot, [](const SampleMeta &m) {
        return make_cspan(m.frame_idx);
      });
    }
    if (has_timestamps_) {
      OutputMetadata<double>(ws, out_index++, slot, [](const SampleMeta &m) {
        return make_cspan(m.timestamps);
      });
    }
  }

  bool HasContiguousOutputs() const override {
    return true;
  }

  const FrameIndex *FileIndex(const std::string &filename) const {
    return static_cast<const VideoLoaderImpl &>(*loader_).FileIndex(filename);
  }

  void Prefetch() override {
    DomainTimeRange tr("[DALI][VideoReaderDecoder] Prefetch #" + to_string(curr_batch_producer_),
                       DomainTimeRange::kRed);
    auto &samples = prefetched_batch_queue_[curr_batch_producer_];
    samples.clear();  // first: a consumed batch's samples go back to the loader
    while (static_cast<int>(pending_.size()) <= decode_ahead_ && !read_failed_)
      StartBatch();
    auto batch = std::move(pending_.front());
    pending_.pop_front();
    WaitForBatch(*batch);  // after this, no lane uses the batch
    if (cancel_)  // the reader is being destroyed: nothing is published any more
      throw std::runtime_error("The video reader is being destroyed.");
    if constexpr (std::is_same_v<Backend, GPUBackend>) {
      if (zero_copy_) {
        // The batch stream is ordered after the lanes (no host synchronization). Also when the
        // batch failed: its buffer must not be released while a lane is still writing to it.
        for (size_t l = 0; l < batch->done_recorded.size(); l++) {
          if (batch->done_recorded[l])
            CUDA_CALL(cudaStreamWaitEvent(cuda_stream_.get(), batch->done[l], 0));
        }
      }
    }
    if (auto error = FirstError(*batch)) {
      cancel_ = true;  // the lanes skip the batches after it
      std::rethrow_exception(error);
    }
    FinishBatch(*batch, slots_[curr_batch_producer_]);
    samples = std::move(batch->samples);
    if (this->IsCheckpointingEnabled())
      this->SaveLoaderSnapshot(batch->snapshot);
    LOG_LINE << "Prefetch done" << std::endl;
  }

 protected:
  bool ShouldSyncBeforeRecycle() const override {
    // With zero_copy_, nothing that RunImpl issues on the workspace's stream reads the samples.
    return !zero_copy_;
  }

 private:
  /**
   * @brief The state of one batch of the prefetch queue.
   *
   * It is kept here, not in the samples, because a sample object can be in two batches of the
   * queue at once (a batch made only of the padding of the shard repeats the last sample of the
   * previous batch).
   */
  struct BatchSlot {
    std::vector<SampleMeta> meta;
    TensorListShape<4> shape;
    std::vector<int> src_idx;  // the index of the sample's first occurrence in the batch
    // zero_copy_ only: the output batch, decoded on cuda_stream_ and complete at `ready`
    TensorList<GPUBackend> data;
    CUDAEvent ready;
    bool ready_recorded = false;
  };

  BatchSlot &CurrSlot() {
    return slots_[this->curr_batch_consumer_];
  }

  int64_t NumFrames(const VideoSampleDesc &sample) const {
    return sample.frame_idxs_.empty()
               ? (static_cast<int64_t>(sample.end_) - sample.start_ + sample.stride_ - 1) /
                     sample.stride_
               : static_cast<int64_t>(sample.frame_idxs_.size());
  }

  void FillOutputDesc(std::vector<OutputDesc> &output_desc, const BatchSlot &slot) const {
    output_desc.clear();
    output_desc.reserve(4);
    int batch_size = slot.shape.num_samples();
    output_desc.push_back({slot.shape, DALI_UINT8});

    if (has_labels_) {
      TensorListShape<1> label_shape = uniform_list_shape<1>(batch_size, {1});
      output_desc.push_back({label_shape, DALI_INT32});
    }

    if (frame_num_policy_ == FrameNumPolicy::Scalar) {
      TensorListShape<1> frame_idx_shape = uniform_list_shape<1>(batch_size, {1});
      output_desc.push_back({frame_idx_shape, DALI_INT32});
    } else if (frame_num_policy_ == FrameNumPolicy::Sequence) {
      TensorListShape<1> frame_idx_shape(batch_size);
      for (int sample_id = 0; sample_id < batch_size; ++sample_id)
        frame_idx_shape.set_tensor_shape(sample_id, {slot.shape.tensor_shape_span(sample_id)[0]});
      output_desc.push_back({frame_idx_shape, DALI_INT32});
    }

    if (has_timestamps_) {
      TensorListShape<1> timestamps_shape(batch_size);
      for (int sample_id = 0; sample_id < batch_size; ++sample_id)
        timestamps_shape.set_tensor_shape(sample_id,
                                          {slot.shape.tensor_shape_span(sample_id)[0]});
      output_desc.push_back({timestamps_shape, DALI_FLOAT64});
    }
  }

  template <typename T>
  void OutputMetadata(Workspace &ws, int out_idx, const BatchSlot &slot,
                      span<const T> (*get_values)(const SampleMeta &)) {
    auto &output = ws.Output<Backend>(out_idx);
    int batch_size = output.num_samples();
    DALI_ENFORCE(output.IsContiguousInMemory(), "Output must be contiguous in memory");
    auto output_as_tensor = output.AsTensor();

    auto copy_data = [&](T* data) {
      for (int sample_id = 0; sample_id < batch_size; ++sample_id)
        for (auto &elem : get_values(slot.meta[sample_id]))
          *data++ = elem;
    };

    if constexpr (std::is_same_v<Backend, GPUBackend>) {
      auto data = mm::alloc_raw_async_unique<T, mm::memory_kind::pinned>(
        output_as_tensor.shape().num_elements(), nullptr, ws.stream());
      copy_data(data.get());
      CUDA_CALL(cudaMemcpyAsync(output_as_tensor.template mutable_data<T>(), data.get(),
                                output_as_tensor.shape().num_elements() * sizeof(T),
                                cudaMemcpyHostToDevice, ws.stream()));
    } else {
      copy_data(output_as_tensor.template mutable_data<T>());
    }
  }

  /**
   * @brief Allocates the batch that Prefetch decodes into and RunImpl returns as the output.
   *
   * The shapes come from the files' metadata, so that the batch is allocated before decoding.
   * A new buffer every time: a published batch is owned by its consumers, and the buffer that
   * the executor swapped in was released in its own order (never written to).
   */
  void AllocateBatchBuffer(BatchSlot &slot,
                           const std::vector<std::shared_ptr<VideoSample<Backend>>> &batch) {
    for (int i = 0; i < static_cast<int>(batch.size()); i++) {
      const auto &sample = *batch[i];
      const auto *meta = sample.video_file_meta_;
      slot.shape.set_tensor_shape(i, TensorShape<4>{NumFrames(sample), meta->height, meta->width,
                                                    kNumChannels});
    }
    slot.data.Reset();
    slot.data.set_device_id(this->device_id_);
    slot.data.set_order(AccessOrder(cuda_stream_.get()));
    slot.data.Resize(slot.shape, DALI_UINT8, BatchContiguity::Contiguous);
    slot.data.SetLayout("FHWC");
  }

  /**
   * @brief Returns the decoded batch as output 0, without copying it, and sizes the metadata
   *        outputs (SetupImpl did not request them).
   */
  void PublishBatch(Workspace &ws, BatchSlot &slot) {
    DALI_ENFORCE(slot.ready_recorded, "Internal error: the batch was not decoded.");
    auto &video_output = ws.Output<GPUBackend>(0);
    CUDA_CALL(cudaStreamWaitEvent(ws.stream(), slot.ready, 0));
    std::swap(video_output, slot.data);
    // From now on, the batch is used (and eventually freed) in the order of the consumers.
    video_output.set_order(ws.output_order(), false);
    video_output.SetLayout("FHWC");
    for (int i = 0; i < video_output.num_samples(); i++)
      video_output.SetSourceInfo(i, slot.meta[i].source_info);
    // What the executor gave us is not written to: it is released in its own order.
    slot.data.Reset();
    slot.ready_recorded = false;

    std::vector<OutputDesc> output_desc;
    FillOutputDesc(output_desc, slot);
    for (int o = 1; o < ws.NumOutput(); o++) {
      ws.Output<GPUBackend>(o).Resize(output_desc[o].shape, output_desc[o].type,
                                      BatchContiguity::Contiguous);
    }
  }

  /**
   * @brief A sample to decode, with the frames it decodes (for planning only).
   */
  struct SampleTask {
    VideoSample<Backend> *sample;
    int batch_idx;  // first position of the sample in the batch
    int first;      // smallest frame index decoded; -1 if none (padding only)
    int last;       // largest frame index decoded; -1 if none
    int end;        // where the sample leaves its decoder (see EndFrame)
    SmallVector<int, 32> frames;  // the frames the decoder seeks to, in order (see SampleFrames)
  };

  /**
   * @brief Where a lane's decoder is, as planned: its file (empty: no decoder) and the frame it
   * returns next (see Normalize).
   */
  struct LanePosition {
    std::string filename;
    int next_frame = -1;
  };

  /**
   * @brief Samples of one file that one decoder decodes in a row, without seeking.
   */
  struct DecodeRun {
    const std::string *filename;
    std::vector<int> tasks;  // indices into the batch's tasks, in decoding order
    int first;
    int last;
    int batch_idx;  // the smallest position of its samples in the batch
  };

  /**
   * @brief An error of a sample, at its position in the batch (INT_MAX: of no sample), and the
   * lane that decoded it.
   */
  struct SampleError {
    int batch_idx;
    int lane;
    std::exception_ptr error;
  };

  /**
   * @brief A batch, from when it is read until it is published: its samples and their decoding.
   */
  struct PendingBatch {
    typename Base::BatchQueueElement samples;
    LoaderStateSnapshot snapshot;  // the loader's state after reading the batch (checkpointing)
    std::vector<SampleTask> tasks;
    std::vector<DecodeRun> runs;
    std::vector<LanePlan> plans;  // the runs of each lane, in decoding order
    std::vector<LanePosition> lane_start;  // where the plan expects each lane's decoder
    // The batch's metadata and, with zero_copy_, its output buffer, which the lanes decode into
    // and which is moved to the batch's slot when the batch is published
    BatchSlot output;
    // zero_copy_ only: `allocated` is recorded on the batch stream once `output.data` is allocated
    // (the lanes' streams wait for it); done[l] is recorded on the stream of lane l when the lane
    // has decoded its part (the batch stream waits for it, see Prefetch)
    CUDAEvent allocated;
    std::vector<CUDAEvent> done;
    std::vector<char> done_recorded;  // written by the lane's thread, read once the batch is done
    // Reading, planning or queuing the batch, or recording a lane's errors, failed (once the batch
    // is queued, guarded by lanes_mutex_)
    std::exception_ptr error;
    // Guarded by lanes_mutex_: the lanes that have yet to do their part, and the errors
    int lanes_left = 0;
    std::vector<SampleError> errors;
  };

  /**
   * @brief A decoder with its own thread (with more than one lane) and stream.
   *
   * A lane keeps its decoder while consecutive samples come from the same file, so that they
   * are decoded without seeking, and decodes on its own stream, so that lanes do not wait for
   * each other's work. Which samples a lane decodes is decided before decoding starts, from the
   * samples alone, so a decoder's history (and with it, the output) never depends on thread
   * timing. A lane does its parts of the batches in the order they are read.
   */
  struct DecodeLane {
    CUDAStreamLease stream;  // GPU only; empty: the default stream
    std::unique_ptr<FramesDecoderImpl> decoder;  // after `stream`: destroyed (and synced) first
    Tensor<Backend> constant_frame;
    std::vector<int> frame_idxs;  // scratch
    // Planning: where the decoder is once the lane is done with the batches planned so far. The
    // plans are made from it, not from the decoder.
    LanePosition planned;
    int64_t backlog = 0;  // planning: see LaneStart::backlog
    // Whether decoding a sample failed (and reset the decoder) since the lane last opened one:
    // the decoder is then not where the plan expects it
    bool failed = false;
    // Diagnostics: the pictures the lane decodes in the current batch
    int64_t decoded_pictures = 0;
    std::deque<PendingBatch *> queue;  // guarded by lanes_mutex_: the batches to do, in order
    std::thread thread;  // with more than one lane
  };

  // Planning costs, in pictures decoded: opening a decoder (or resetting it, which reopens the
  // file) costs about as much as decoding a few pictures
  static constexpr int64_t kOpenCost = 16;

  // GPU: decode lanes per NVDEC engine. More sessions than engines keep an engine busy while a
  // lane does its host work between pictures; many more only add memory.
  static constexpr int kLanesPerEngine = 2;

  // With more than one lane: the batches read and decoded ahead of the one being published, so that
  // a lane that is done with its part of a batch goes on to the next one, instead of waiting for
  // the slowest lane. Each holds one more batch of samples (of the loader's 2 *
  // prefetch_queue_depth batches).
  static constexpr int kDecodeAhead = 1;

  // SampleTask::end of a sample that decodes no frame: its decoder stays where it was
  static constexpr int kNoFrames = -2;

  /**
   * @brief A decoder's next frame as the planner sees it: CanReachWithoutSeek and resetting
   * decoders (see FramesDecoderBase::SeekFrame) treat any frame < 0 or >= num_frames as the end
   * of the file, so these are all -1.
   */
  static int Normalize(int next_frame, int num_frames) {
    return next_frame < 0 || next_frame >= num_frames ? -1 : next_frame;
  }

  /**
   * @brief Where decoding a sample leaves its decoder: after the largest frame it decodes
   * (Normalized), or kNoFrames.
   *
   * DecodeFrames maps the frames into [0, num_frames) once more, which differs from the sample's
   * own range (see SampleFrames) when it ends past the end of the file. It decodes the frames in
   * order, so that the largest one is read last.
   */
  int EndFrame(const SampleTask &task, int num_frames) const {
    int last = -1;
    for (int frame : task.frames) {
      last = std::max(last, FramesDecoderBase::HandleBoundary(boundary_type_, frame, 0,
                                                              num_frames));
    }
    return last < 0 ? kNoFrames : Normalize(last + 1, num_frames);
  }

  /**
   * @brief Where a lane's decoder is once it decodes `runs` of `batch`, in order, from `pos`.
   */
  static LanePosition PredictEnd(const PendingBatch &batch, span<const int> runs,
                                 LanePosition pos) {
    for (int r : runs) {
      for (int t : batch.runs[r].tasks) {
        const auto &task = batch.tasks[t];
        const auto &filename = task.sample->video_file_meta_->filename;
        if (pos.filename != filename)
          pos = {filename, 0};  // a newly opened decoder
        if (task.end != kNoFrames)
          pos.next_frame = task.end;
      }
    }
    return pos;
  }

  /**
   * @brief Whether a lane's decoder is at `pos`.
   */
  static bool IsAt(DecodeLane &lane, const LanePosition &pos) {
    if (!lane.decoder)
      return pos.filename.empty();
    return lane.decoder->Filename() == pos.filename &&
           Normalize(lane.decoder->NextFrameIdx(), lane.decoder->NumFrames()) == pos.next_frame;
  }

  /**
   * @brief Checks that a lane's decoder is at `start`, where the plan of its batch expects it.
   *
   * The plans are made from where decoding the samples leaves the decoders, which is where they
   * are as long as they decode every frame they are asked for. If one is somewhere else, the lane
   * still decodes the right frames (decoding seeks from where the decoder is), only the plan may
   * be worse. With DALI_VIDEO_READER_STRICT_PLANS set (for tests), it is an error.
   */
  void CheckPlan(DecodeLane &lane, const LanePosition &start) {
    if (lane.failed || IsAt(lane, start))
      return;
    LOG_LINE << "The decoder of a lane is at "
             << (lane.decoder ? lane.decoder->Filename() : "no file") << ", frame "
             << (lane.decoder ? lane.decoder->NextFrameIdx() : -1) << "; planned: "
             << start.filename << ", frame " << start.next_frame << std::endl;
    DALI_ENFORCE(!strict_plans_, make_string(
        "Internal error: a video decoder is not where the plan expects it, at frame ",
        start.next_frame, " of \"", start.filename, "\"; it is at frame ",
        lane.decoder ? lane.decoder->NextFrameIdx() : -1, " of \"",
        lane.decoder ? lane.decoder->Filename() : "", "\"."));
  }

  /**
   * @brief Number of decode lanes: num_threads, but no more than the batch size and, on the GPU,
   * kLanesPerEngine per NVDEC engine.
   */
  int NumLanes(const OpSpec &spec) {
    int lanes = std::min(spec.GetArgument<int>("num_threads"),
                         spec.GetArgument<int>("max_batch_size"));
    if constexpr (std::is_same_v<Backend, GPUBackend>) {
      int engines = FramesDecoderGpu::NumEngines(spec.GetArgument<int>("device_id"));
      if (engines > 0)
        lanes = std::min(lanes, kLanesPerEngine * engines);
    }
    return std::max(lanes, 1);
  }

  void InitLanes(const OpSpec &spec) {
    lanes_.resize(NumLanes(spec));
    decode_ahead_ = lanes_.size() > 1 ? kDecodeAhead : 0;
    // CPU: the lanes' threads do not use the GPU (DeviceGuard does nothing)
    lane_device_id_ = std::is_same_v<Backend, GPUBackend> ? spec.GetArgument<int>("device_id")
                                                          : CPU_ONLY_DEVICE_ID;
    for (auto &lane : lanes_) {
      lane.constant_frame.set_pinned(std::is_same_v<Backend, GPUBackend>);
      if (use_streams_)
        lane.stream = CUDAStreamPool::instance().Get(lane_device_id_);
    }
  }

  /**
   * @brief Starts the lanes' threads, with more than one lane, unless they run already.
   */
  void StartLanes() {
    if (lanes_.size() == 1 || lanes_started_)
      return;
    for (int l = 0; l < static_cast<int>(lanes_.size()); l++)
      lanes_[l].thread = std::thread(&VideoReaderDecoder::LaneMain, this, l);
    lanes_started_ = true;
  }

  void StopLanes() {
    {
      std::lock_guard<std::mutex> lock(lanes_mutex_);
      stop_lanes_ = true;
    }
    lanes_cv_.notify_all();
    for (auto &lane : lanes_) {
      if (lane.thread.joinable())
        lane.thread.join();
    }
  }

  /**
   * @brief Reads the next batch, plans its decoding and queues it to the lanes.
   *
   * If reading or planning it fails, the batch fails when it is due, and no batch is read after
   * it.
   */
  void StartBatch() {
    pending_.push_back(std::make_unique<PendingBatch>());
    auto &batch = *pending_.back();
    try {
      this->ReadBatch(batch.samples);
      if (this->IsCheckpointingEnabled())
        batch.snapshot = loader_->GetStateSnapshot();
      StartLanes();
      PrepareOutput(batch);
      PlanBatch(batch);
      AssignRuns(batch);
    } catch (...) {
      batch.error = std::current_exception();
      read_failed_ = true;
      return;
    }
    QueueBatch(batch);
  }

  /**
   * @brief Groups the batch's samples into runs: samples of one file that one decoder decodes in
   * a row without seeking.
   *
   * A sample repeated in the batch (pad_last_batch) is decoded once. A sample that is in an earlier
   * batch too (the padding repeats the last sample, also in the next batch) is decoded again with
   * zero_copy_, into the output of the batch; otherwise it is not decoded again, which would
   * overwrite the frames of the earlier batch while that may still be in use.
   */
  void PlanBatch(PendingBatch &batch) {
    auto &tasks = batch.tasks;
    auto &runs = batch.runs;
    for (int i = 0; i < static_cast<int>(batch.samples.size()); i++) {
      if (batch.output.src_idx[i] != i)
        continue;
      auto *sample = batch.samples[i].get();
      if (!zero_copy_) {
        if (sample->planned)
          continue;
        sample->planned = true;
      }
      const auto &filename = sample->video_file_meta_->filename;
      auto *index = FileIndex(filename);
      DALI_ENFORCE(index, make_string("Internal error: no frame index for \"", filename, "\"."));
      SampleTask task{sample, i, -1, -1, -1, {}};
      try {
        SampleFrames(sample->frame_idxs_, sample->start_, sample->end_, sample->stride_,
                     boundary_type_, sample->video_file_meta_->start_frame,
                     sample->video_file_meta_->end_frame, task.frames);
        task.end = EndFrame(task, index->size());
      } catch (...) {
        // decoding the sample reports the error, in its lane, and resets the decoder
        task.frames.clear();
        task.end = -1;
      }
      if (!task.frames.empty()) {
        task.first = task.frames.front();
        task.last = task.frames.back();
      }
      tasks.push_back(std::move(task));
    }

    std::vector<int> order(tasks.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
      const auto &ta = tasks[a], &tb = tasks[b];
      const auto &fa = ta.sample->video_file_meta_->filename;
      const auto &fb = tb.sample->video_file_meta_->filename;
      return std::tie(fa, ta.first, ta.batch_idx) < std::tie(fb, tb.first, tb.batch_idx);
    });
    const FrameIndex *index = nullptr;
    size_t file_runs = 0;  // the first run of the current file
    for (int t : order) {
      auto &task = tasks[t];
      const auto &filename = task.sample->video_file_meta_->filename;
      if (runs.empty() || *runs.back().filename != filename) {
        file_runs = runs.size();
        index = FileIndex(filename);
      }
      // join a run of the file that a decoder continues to this sample without seeking
      auto run = std::find_if(runs.begin() + file_runs, runs.end(), [&](const DecodeRun &r) {
        return task.first < 0 || CanReachWithoutSeek(index, r.last + 1, task.first);
      });
      if (run != runs.end()) {
        run->tasks.push_back(t);
        if (run->first < 0)  // opened by a sample that decodes no frame
          run->first = task.first;
        run->last = std::max(run->last, task.last);
        run->batch_idx = std::min(run->batch_idx, task.batch_idx);
      } else {
        runs.push_back({&filename, {t}, task.first, task.last, task.batch_idx});
      }
    }
  }

  /**
   * @brief Assigns the runs to lanes (see PlanLanes), from where the lanes' earlier batches leave
   * their decoders. A lane without runs releases its decoder.
   */
  void AssignRuns(PendingBatch &batch) {
    std::vector<PlannedRun> runs;
    for (const auto &run : batch.runs) {
      runs.push_back({*run.filename, FileIndex(*run.filename), run.batch_idx, run.first, run.last,
                      {}});
      for (int t : run.tasks)
        runs.back().samples.push_back(make_cspan(batch.tasks[t].frames));
    }
    std::vector<LaneStart> starts(lanes_.size());
    for (size_t l = 0; l < lanes_.size(); l++) {
      const auto &pos = lanes_[l].planned;
      if (!pos.filename.empty()) {
        starts[l] = {pos.filename, FramesDecoderBase::PositionAt(*FileIndex(pos.filename),
                                                                 pos.next_frame)};
      }
      starts[l].backlog = lanes_[l].backlog;
    }
    PlanLanes(make_cspan(runs), make_cspan(starts), kOpenCost, batch.plans);
    if (decode_ahead_ > 0) {
      // A lane that is done with its part of a batch goes on to the next one: the work that one
      // lane has more than another carries over to the next batch
      int64_t least = std::numeric_limits<int64_t>::max();
      for (size_t l = 0; l < lanes_.size(); l++) {
        lanes_[l].backlog += batch.plans[l].load;
        least = std::min(least, lanes_[l].backlog);
      }
      for (auto &lane : lanes_)
        lane.backlog -= least;
    }
    batch.lane_start.resize(lanes_.size());
    for (size_t l = 0; l < lanes_.size(); l++) {
      auto &lane = lanes_[l];
      const auto &lane_runs = batch.plans[l].runs;
      batch.lane_start[l] = lane.planned;
      if (lane_runs.empty() && lanes_.size() > 1)
        lane.planned = {};  // the lane releases its decoder (see DecodeLanePart)
      else
        lane.planned = PredictEnd(batch, make_cspan(lane_runs), std::move(lane.planned));
    }
  }

  /**
   * @brief Prepares the output of a batch: where its samples are repeated, its metadata and, with
   * zero_copy_, the buffer that the lanes decode into.
   *
   * A sample can appear more than once in a batch (the padding of the last batch repeats the last
   * sample): it is decoded once, and the others are copies.
   */
  void PrepareOutput(PendingBatch &batch) {
    auto &out = batch.output;
    int batch_size = batch.samples.size();
    out.meta.resize(batch_size);
    out.shape.resize(batch_size);
    out.src_idx.resize(batch_size);
    first_occurrence_.clear();
    for (int i = 0; i < batch_size; i++)
      out.src_idx[i] = first_occurrence_.emplace(batch.samples[i].get(), i).first->second;

    if constexpr (std::is_same_v<Backend, GPUBackend>) {
      if (zero_copy_) {
        AllocateBatchBuffer(out, batch.samples);
        batch.allocated = CUDAEvent::CreateWithFlags(cudaEventDisableTiming, this->device_id_);
        CUDA_CALL(cudaEventRecord(batch.allocated, cuda_stream_.get()));
        batch.done.clear();
        for (size_t l = 0; l < lanes_.size(); l++) {
          batch.done.push_back(
              CUDAEvent::CreateWithFlags(cudaEventDisableTiming, this->device_id_));
        }
      }
    }
    batch.done_recorded.assign(lanes_.size(), 0);
  }

  /**
   * @brief Moves the decoded batch to its slot of the prefetch queue (the lanes are done with it).
   */
  void FinishBatch(PendingBatch &batch, BatchSlot &slot) {
    auto &out = batch.output;
    int batch_size = batch.samples.size();
    if (!zero_copy_) {
      // the frames are in the samples (of this batch or of an earlier one)
      for (int i = 0; i < batch_size; i++) {
        if (out.src_idx[i] != i)
          continue;
        const auto &sample = *batch.samples[i];
        out.meta[i] = sample.meta_;
        out.shape.set_tensor_shape(i, sample.data_.shape());
      }
    }
    for (int i = 0; i < batch_size; i++) {
      int src = out.src_idx[i];
      if (src == i)
        continue;
      out.meta[i] = out.meta[src];
      out.shape.set_tensor_shape(i, out.shape.tensor_shape(src));
      if constexpr (std::is_same_v<Backend, GPUBackend>) {
        if (zero_copy_) {
          // each sample of the output has a buffer of its own
          CUDA_CALL(cudaMemcpyAsync(out.data.raw_mutable_tensor(i), out.data.raw_tensor(src),
                                    out.shape.tensor_size(i), cudaMemcpyDeviceToDevice,
                                    cuda_stream_.get()));
        }
      }
    }
    slot.meta = std::move(out.meta);
    slot.shape = std::move(out.shape);
    slot.src_idx = std::move(out.src_idx);
    slot.ready_recorded = false;
    if constexpr (std::is_same_v<Backend, GPUBackend>) {
      if (zero_copy_) {
        std::swap(slot.data, out.data);
        // RunImpl orders its stream after this event
        CUDA_CALL(cudaEventRecord(slot.ready, cuda_stream_.get()));
        slot.ready_recorded = true;
      }
    }
  }

  /**
   * @brief Has the lanes decode the batch: on their threads, after the batches queued before it,
   * or here, with one lane.
   */
  void QueueBatch(PendingBatch &batch) {
    if (lanes_.size() == 1) {
      batch.lanes_left = 1;
      RunLanePart(0, batch);
      return;
    }
    {
      std::lock_guard<std::mutex> lock(lanes_mutex_);
      for (auto &lane : lanes_) {
        try {
          lane.queue.push_back(&batch);
        } catch (...) {
          batch.error = std::current_exception();
          break;
        }
        batch.lanes_left++;  // only the lanes that do their part
      }
    }
    lanes_cv_.notify_all();
  }

  /**
   * @brief Waits until no lane uses the batch any more.
   */
  void WaitForBatch(PendingBatch &batch) {
    std::unique_lock<std::mutex> lock(lanes_mutex_);
    batch_done_cv_.wait(lock, [&] { return batch.lanes_left == 0; });
  }

  /**
   * @brief The error a batch fails with: of reading or planning it, or of its first failing sample
   * (whatever the lanes' timing).
   */
  static std::exception_ptr FirstError(const PendingBatch &batch) {
    if (batch.error)
      return batch.error;
    const SampleError *first = nullptr;
    for (const auto &error : batch.errors) {
      if (!first || std::tie(error.batch_idx, error.lane) < std::tie(first->batch_idx, first->lane))
        first = &error;
    }
    return first ? first->error : nullptr;
  }

  void LaneMain(int l) {
    // Nothing may throw out of a lane's thread: if setting it up fails, it fails its part of every
    // batch with that error
    std::exception_ptr setup_error;
    std::optional<DeviceGuard> device_guard;
    try {
      SetThreadName(make_string("[DALI][VR lane ", l, "]").c_str());
      device_guard.emplace(lane_device_id_);
    } catch (...) {
      setup_error = std::current_exception();
    }
    auto &lane = lanes_[l];
    std::unique_lock<std::mutex> lock(lanes_mutex_);
    for (;;) {
      lanes_cv_.wait(lock, [&] { return stop_lanes_ || !lane.queue.empty(); });
      if (lane.queue.empty())
        return;  // stopped
      PendingBatch *batch = lane.queue.front();
      lane.queue.pop_front();
      lock.unlock();
      RunLanePart(l, *batch, setup_error);
      lock.lock();
    }
  }

  /**
   * @brief Does a lane's part of a batch, and records that it is done. Never throws: on a lane's
   * thread, that would end the process, and the batch would never be done. With `setup_error`
   * (the lane's thread could not be set up), the part fails with it.
   */
  void RunLanePart(int l, PendingBatch &batch, std::exception_ptr setup_error = nullptr) {
    std::vector<SampleError> errors;
    std::exception_ptr failure = setup_error;  // of the lane's part, not of a sample
    if (!failure) {
      try {
        DecodeLanePart(l, batch, errors);
      } catch (...) {
        failure = std::current_exception();
      }
    }
    std::lock_guard<std::mutex> lock(lanes_mutex_);
    try {
      if (failure)
        errors.push_back({std::numeric_limits<int>::max(), l, failure});
      batch.errors.insert(batch.errors.end(), errors.begin(), errors.end());
    } catch (...) {
      // out of memory: the batch fails all the same
      if (!batch.error)
        batch.error = failure ? failure : std::current_exception();
    }
    if (--batch.lanes_left == 0)
      batch_done_cv_.notify_all();
  }

  /**
   * @brief Decodes a lane's runs of a batch. A sample that fails is recorded in `errors`.
   */
  void DecodeLanePart(int l, PendingBatch &batch, std::vector<SampleError> &errors) {
    auto &lane = lanes_[l];
    const auto &plan = batch.plans[l];
    if (plan.runs.empty()) {
      if (lanes_.size() > 1)
        lane.decoder.reset();  // do not keep its frame buffers and NVDEC decoder idle
      return;
    }
    DomainTimeRange tr(make_string("[DALI][VideoReaderDecoder] lane ", l),
                       DomainTimeRange::kOrange);
    lane.decoded_pictures = 0;
    int first_task = batch.runs[plan.runs[0]].tasks[0];
    bool cancelled = false;
    if constexpr (std::is_same_v<Backend, GPUBackend>) {
      if (zero_copy_) {
        try {
          // the batch buffer was allocated in the order of the batch stream
          CUDA_CALL(cudaStreamWaitEvent(lane.stream.get(), batch.allocated, 0));
        } catch (...) {
          cancelled = true;  // nothing may be decoded into the buffer
          errors.push_back({batch.tasks[first_task].batch_idx, l, std::current_exception()});
        }
      }
    }
    try {
      CheckPlan(lane, batch.lane_start[l]);
    } catch (...) {
      errors.push_back({batch.tasks[first_task].batch_idx, l, std::current_exception()});
    }
    for (size_t i = 0; i < plan.runs.size() && !cancelled; i++) {
      for (int t : batch.runs[plan.runs[i]].tasks) {
        if ((cancelled = cancel_))
          break;
        auto &task = batch.tasks[t];
        try {
          DecodeSample(task.sample, task.batch_idx, batch.output, lane);
        } catch (...) {
          lane.decoder.reset();  // in an unknown state; its destructor waits for its stream
          lane.failed = true;
          errors.push_back({task.batch_idx, l, std::current_exception()});
        }
      }
    }
    if constexpr (std::is_same_v<Backend, GPUBackend>) {
      cudaStream_t stream = lane.stream ? lane.stream.get() : 0;
      if (errors.empty() && zero_copy_) {
        // the batch stream waits for it when the batch is published (Prefetch): the host does not
        try {
          CUDA_CALL(cudaEventRecord(batch.done[l], stream));
          batch.done_recorded[l] = 1;
        } catch (...) {
          errors.push_back({std::numeric_limits<int>::max(), l, std::current_exception()});
        }
      } else if (errors.empty()) {
        // the samples are read on the workspace stream once the batch is consumed
        try {
          CUDA_CALL(cudaStreamSynchronize(stream));
        } catch (...) {
          errors.push_back({std::numeric_limits<int>::max(), l, std::current_exception()});
        }
      } else if (cudaStreamSynchronize(stream) != cudaSuccess) {
        (void)cudaGetLastError();  // already failing: report the sample's error
      }
    }
    // The pictures the lane's decoder decoded (without those it left out), and the planned ones
    LOG_LINE << "Lane " << l << " decoded " << lane.decoded_pictures << " pictures, planned "
             << plan.pictures << " (load " << plan.load << ")" << std::endl;
  }

  /**
   * @brief Decodes one sample of the batch on a lane, into its place in the output batch (with
   * zero_copy_) or into the sample.
   */
  void DecodeSample(VideoSample<Backend> *sample_ptr, int i, BatchSlot &slot, DecodeLane &lane) {
    auto &sample = *sample_ptr;
    LOG_LINE << "Processing sample " << i << " with filename "
             << sample.video_file_meta_->filename << " and previous decoder "
             << lane.decoder.get() << " filename "
             << (lane.decoder ? lane.decoder->Filename() : "none") << std::endl;
    OpenDecoder(lane, sample.video_file_meta_->filename);
    auto &decoder = *lane.decoder;
    int64_t decoded_pictures = decoder.DecodedPictures();
    int64_t num_frames = NumFrames(sample);
    uint8_t *out = nullptr;
    if (zero_copy_) {
      auto expected = slot.shape.tensor_shape(i);
      DALI_ENFORCE(expected == decoder.Shape(num_frames),
                   make_string("The frames of \"", decoder.Filename(), "\" have the shape ",
                               decoder.Shape(num_frames), ", but the shape was ", expected,
                               " when the reader indexed the file. The file was probably "
                               "modified after the reader was created: create the pipeline "
                               "again to index it."));
      out = static_cast<uint8_t *>(slot.data.raw_mutable_tensor(i));
    } else {
      sample.data_.Resize(decoder.Shape(num_frames), DALI_UINT8);
      sample.data_.SetSourceInfo(decoder.Filename());
      sample.data_.SetLayout("FHWC");
      out = sample.data_.template mutable_data<uint8_t>();
    }
    DecodeFrames(sample, num_frames, out, zero_copy_ ? slot.meta[i] : sample.meta_, lane);
    lane.decoded_pictures += decoder.DecodedPictures() - decoded_pictures;
    LOG_LINE << "Decoding frames done" << std::endl;
  }

  /** Makes the decoder of the lane a decoder of `filename`, reusing it if it is of that file. */
  void OpenDecoder(DecodeLane &lane, const std::string &filename) {
    if (!lane.decoder || lane.decoder->Filename() != filename) {
      lane.decoder.reset();  // first: its frame buffers are freed before the new ones
      lane.failed = false;
      if constexpr (std::is_same_v<Backend, CPUBackend>) {
        lane.decoder = std::make_unique<FramesDecoderImpl>(filename, image_type_);
      } else {
        lane.decoder = std::make_unique<FramesDecoderImpl>(filename, lane.stream, image_type_);
      }
      auto &decoder = *lane.decoder;
      LOG_LINE << "Initialized decoder to " << decoder.Filename() << " ptr: " << &decoder
               << " num_frames: " << decoder.NumFrames() << std::endl;
      // the index the batch was planned with (the file is not indexed again, even if it changed)
      auto *index = FileIndex(filename);
      DALI_ENFORCE(index, make_string("Internal error: no frame index for \"", filename, "\"."));
      if (!decoder.HasIndex())
        decoder.SetIndex(*index);
    } else {
      LOG_LINE << "Reusing decoder for " << lane.decoder->Filename() << " ptr: "
               << lane.decoder.get() << " num_frames: " << lane.decoder->NumFrames() << std::endl;
    }
    DALI_ENFORCE(lane.decoder->IsValid(), make_string("Invalid decoder for filename ", filename));
  }


  /** Decodes the frames of `sample` with the lane's decoder to `out`, and fills its metadata. */
  void DecodeFrames(const VideoSampleDesc &sample, int64_t num_frames, uint8_t *out,
                    SampleMeta &meta, DecodeLane &lane) {
    auto &decoder = *lane.decoder;
    meta.source_info = decoder.Filename();
    meta.label = sample.video_file_meta_->label;
    meta.start = sample.start_;
    const uint8_t *constant_frame =
        boundary_type_ == boundary::BoundaryType::CONSTANT ?
            ConstantFrame(lane.constant_frame, decoder.FrameShape(), make_cspan(fill_value_),
                          lane.stream, true) :
            nullptr;
    if (has_timestamps_) {
      meta.timestamps.resize(num_frames);
    } else {
      meta.timestamps.clear();
    }
    if (!sample.frame_idxs_.empty()) {
      LOG_LINE << "Decoding frames (uniform) num_frames=" << num_frames
               << ", frame_idxs=[" << sample.frame_idxs_.front() << ".."
               << sample.frame_idxs_.back() << "]"
               << ", filename=" << sample.video_file_meta_->filename
               << ", label=" << sample.video_file_meta_->label
               << ", boundary_type=" << to_string(boundary_type_) << std::endl;
    } else {
      LOG_LINE << "Decoding frames start=" << sample.start_ << ", end=" << sample.end_
               << ", stride=" << sample.stride_ << ", num_frames=" << num_frames
               << ", filename=" << sample.video_file_meta_->filename
               << ", label=" << sample.video_file_meta_->label
               << ", start=" << sample.video_file_meta_->start_frame
               << ", end=" << sample.video_file_meta_->end_frame
               << ", boundary_type=" << to_string(boundary_type_) << std::endl;
    }
    int roi_start = sample.video_file_meta_->start_frame;
    int roi_end = sample.video_file_meta_->end_frame;
    if (frame_num_policy_ == FrameNumPolicy::Sequence) {
      meta.frame_idx.resize(num_frames);
      if (!sample.frame_idxs_.empty()) {
        // Uniform indices are always within [roi_start, roi_end-1], so HandleBoundary
        // is a no-op here — but we call it for consistency with the stride path below.
        for (int64_t i = 0; i < num_frames; ++i) {
          meta.frame_idx[i] = static_cast<int32_t>(decoder.HandleBoundary(
              boundary_type_, sample.frame_idxs_[i], roi_start, roi_end));
        }
      } else {
        for (int64_t i = 0; i < num_frames; ++i) {
          meta.frame_idx[i] = static_cast<int32_t>(decoder.HandleBoundary(
              boundary_type_,
              static_cast<int>(sample.start_ + i * sample.stride_),
              roi_start, roi_end));
        }
      }
    } else {
      meta.frame_idx.clear();
    }
    if (!sample.frame_idxs_.empty()) {
      // Uniform sampling: explicit frame indices already include ROI offset (start_frame)
      decoder.DecodeFrames(out, make_cspan(sample.frame_idxs_), boundary_type_, constant_frame,
                           make_span(meta.timestamps));
    } else if (roi_start != 0 || roi_end != decoder.NumFrames()) {
      lane.frame_idxs.clear();
      for (int64_t frame_idx = sample.start_; frame_idx < sample.end_;
           frame_idx += sample.stride_) {
        lane.frame_idxs.push_back(decoder.HandleBoundary(
            boundary_type_, static_cast<int>(frame_idx), roi_start, roi_end));
      }
      decoder.DecodeFrames(out, make_cspan(lane.frame_idxs), boundary_type_, constant_frame,
                           make_span(meta.timestamps));
    } else {
      decoder.DecodeFrames(out, sample.start_, sample.end_, sample.stride_, boundary_type_,
                           constant_frame, make_span(meta.timestamps));
    }
  }

  static constexpr int kNumChannels = 3;

  FrameNumPolicy frame_num_policy_;
  bool has_timestamps_;
  boundary::BoundaryType boundary_type_;
  DALIImageType image_type_;
  std::vector<uint8_t> fill_value_;
  bool has_labels_ = false;
  // GPU: decode into the output batch, instead of into the samples and copying them to the
  // output (DALI_VIDEO_READER_COPY_OUTPUTS=1 restores the copy; internal, for comparisons)
  bool zero_copy_ = false;

  // GPU: the batch stream: the output batch (zero_copy_) is allocated and complete in its order
  CUDAStreamLease cuda_stream_;
  bool use_streams_ = false;  // GPU: a stream per lane (false: the default stream, old drivers)
  std::unordered_map<const void *, int> first_occurrence_;
  bool strict_plans_ = false;  // see CheckPlan
  std::vector<DecodeLane> lanes_;
  int lane_device_id_ = CPU_ONLY_DEVICE_ID;  // the device of the lanes' threads
  bool lanes_started_ = false;
  std::mutex lanes_mutex_;
  std::condition_variable lanes_cv_;  // a lane's queue or stop_lanes_ changed
  std::condition_variable batch_done_cv_;  // a batch's lanes_left reached 0
  bool stop_lanes_ = false;  // guarded by lanes_mutex_
  // The batches read, and queued to the lanes, that are not published yet, in reading order
  std::deque<std::unique_ptr<PendingBatch>> pending_;
  int decode_ahead_ = 0;  // the batches read and decoded ahead of the one being published
  bool read_failed_ = false;  // reading or planning a batch failed: no batch is read after it
  // Set by the destructor, or by Prefetch when a batch fails: the lanes skip the samples they have
  // not started, and Prefetch publishes nothing more
  std::atomic<bool> cancel_{false};
  // one per batch of the prefetch queue; declared after cuda_stream_, so that their buffers are
  // released (in that stream's order) before the stream is
  std::vector<BatchSlot> slots_;
};

DALI_SCHEMA(experimental__readers__Video)
    .DocStr(R"code(Loads and decodes video files from disk.

The operator supports most common video container formats using libavformat (FFmpeg).
MPEG program streams (``.mpg``, ``.vob``) are not supported: such files are skipped with a warning.
Remux them to another container, for example with ``ffmpeg -i in.mpg -c copy out.mp4``.
The operator utilizes either libavcodec (FFmpeg) or NVIDIA Video Codec SDK (NVDEC) for decoding the frames.

The following video codecs are supported by both CPU and GPU backends:

* VP8
* VP9
* MJPEG

The following codecs are supported by the GPU backend only:

* AV1
* MPEG-4
* H.264/AVC
* H.265/HEVC

The samples of a batch are decoded in parallel, by up to ``num_threads`` decoders (on the GPU, at
most two per NVDEC engine of the device). Consecutive samples of one file that can be decoded
without seeking are decoded by the same decoder, one after another. Every decoder has its own
frame buffers, so the memory used grows with the number of decoders. With more than one decoder,
the decoders that are done with their part of a batch start on the next batch, so up to one more
batch of decoded samples is held in memory (host memory for the CPU backend).

The outputs of the operator are: video, [labels], [frame_num], [timestamps].

* ``video``: A sequence of frames with shape ``(F, H, W, C)`` where ``F`` is the number of frames in the sequence
  (can vary between samples), ``H`` is the frame height in pixels, ``W`` is the frame width in pixels, and ``C`` is
  the number of color channels.
* ``labels``: Label associated with the sample. Only available when using ``labels`` with ``filenames``, or when
  using ``file_list`` or ``file_root``.
* ``frame_num``: Frame number information. Shape and content depend on ``enable_frame_num``:

  * ``"scalar"`` or ``True``: Index of the first frame in the decoded sequence, shape ``(1,)``.
  * ``"sequence"``: Frame index of each decoded frame, shape ``(F,)``. Padded frames (e.g. when
    using ``pad_mode='constant'``) have index ``-1``.
* ``timestamps``: Time in seconds of each frame in the sequence. Only available when ``enable_timestamps=True``.
)code")
    .NumInput(0)
    .OutputFn([](const OpSpec &spec) {
      bool has_labels = HasLabelsOutput(spec);
      bool has_frame_num =
          ParseFrameNumPolicy(spec.GetArgument<std::string>("enable_frame_num")) !=
          FrameNumPolicy::None;
      return 1 + has_labels + has_frame_num + spec.GetArgument<bool>("enable_timestamps");
    })
    .AddOptionalArg("filenames",
                    R"code(Absolute paths to the video files to load.

Remote files can be specified with URLs, e.g. ``s3://bucket/video.mp4`` or
``gs://bucket/video.mp4``.

This option is mutually exclusive with `file_root` and `file_list`.)code",
                    std::vector<std::string>{})
    .AddOptionalArg("file_root",
                    R"code(Path to a directory that contains the data files.

The files are expected in subdirectories of `file_root`, one per label.

Remote directories can be specified with URLs, e.g. ``s3://bucket/videos`` or
``gs://bucket/videos``.

This option is mutually exclusive with `filenames` and `file_list`.)code",
                    std::string())
    .AddOptionalArg("file_list",
                    R"code(Path to the file with a list of ``file label [start [end]]`` values.

``start`` and ``end`` are optional and can be used to specify the start and end of the video to load.
The values can be interpreted differently depending on the ``file_list_format``.

The ``file`` can be a remote URL, e.g. ``s3://bucket/video.mp4`` or ``gs://bucket/video.mp4``.

This option is mutually exclusive with `filenames` and `file_root`.)code",
                    std::string())
    .AddOptionalArg("file_list_format",
        R"code(How to interpret start/end values in file_list:

* ``frames``: Use exact frame numbers (0-based).
* ``timestamps``: Use timestamps in seconds, relative to the first frame of the video.

Negative start and end values count from the end of the video, and an end of 0 (or an omitted
end) is the end of the video. An end past the end of the video is clamped to it. A start past the
end of the video is an error, and any other empty range (for example, a start equal to the end)
is skipped with a warning.

Default: ``timestamps``.)code",
        "timestamps")
    .AddOptionalArg("file_list_rounding",
        R"code(How to handle non-exact frame matches:

* ``start_down_end_up`` (default): Round start down and end up
* ``start_up_end_down``: Round start up and end down
* ``all_up``: Round both up
* ``all_down``: Round both down

The legacy ``fn.readers.video`` has the ``file_list_include_preceding_frame`` argument instead:
True corresponds to ``start_down_end_up`` and False to ``start_up_end_down``. The legacy reader
truncates the frame numbers of the ``frames`` format, which corresponds to ``all_down``.)code",
        "start_down_end_up")
    .AddOptionalArg("file_list_include_end",
        R"code(If true, the end frame of a ``file_list`` entry is part of its range. Default: false,
which makes the end exclusive, as the end of a Python range.

In the ``frames`` format, the range ends at ``end + 1``. In the ``timestamps`` format, the frame
whose timestamp is exactly ``end`` is included, if there is one.)code",
        false)
    .AddOptionalArg<vector<int>>("labels", R"(Labels associated with the files listed in
`filenames` argument.

If an empty list is provided, sequential 0-based indices are used as labels. If not provided,
no labels will be yielded.

This argument is valid only together with `filenames`. ``file_root`` and ``file_list`` provide
their own labels, so combining either of them with ``labels`` is an error.)",
                                 nullptr)
    .AddArg("sequence_length", R"code(Frames to load per sequence. Must be at least 1.)code",
            DALI_INT32)
    .AddOptionalArg("enable_frame_num",
                    R"code(Determines what frame number information is returned as an additional output.

* ``"none"`` or ``False`` (default): No frame number output.
* ``"scalar"`` or ``True``: Returns the index of the first frame in the decoded sequence, shape ``(1,)``.
* ``"sequence"``: Returns the frame index of each decoded frame, shape ``(F,)``. For padded
  frames (e.g. when using ``pad_mode='constant'``), the index is ``-1``.)code",
                    std::string("none"))
    .AddOptionalArg("enable_timestamps",
                    R"code(If set, returns the timestamp of the frames in the decoded sequence
as an additional output.)code",
                    false)
    .AddOptionalArg("step",
                    R"code(Frame interval between each sequence.

When the value is less than or equal to 0, `step` is set to `stride * sequence_length`.)code",
                    -1)
    .AddOptionalArg("stride", R"code(Distance between consecutive frames in the sequence.

Must be at least 1, and ``stride * sequence_length`` must not exceed ``2**31 - 1``. Both
conditions are not checked when ``uniform_sample`` is True, which ignores the stride.)code", 1u,
                    false)
    .AddOptionalArg("uniform_sample",
        R"code(If set to True, uniformly samples ``sequence_length`` frames from the full video
(or from the video range defined by ``file_list``), regardless of the video length.

The sampled frame indices correspond to ``numpy.linspace(start, end-1, sequence_length)``
rounded to the nearest integer using ``floor(x + 0.5)`` (rounds half away from zero,
matching C++ ``std::round`` — not NumPy's default banker's rounding).

If ``sequence_length`` exceeds the number of frames in the video, frames are repeated
rather than padded. For example, sampling 5 frames from a 3-frame video yields indices
``[0, 1, 1, 2, 2]``. A single-frame video always produces a sequence of identical frames.

When enabled, each video file produces exactly one sample per epoch.
The ``stride``, ``step``, and ``pad_mode`` arguments are ignored.)code",
        false)
    .AddOptionalArg<std::string>(
        "pad_mode",
        R"code(How to handle videos with insufficient frames when using start_frame/sequence_length/stride:

* ``'none'``: No padding: a sequence that would extend past the last frame is not returned, so
  every sample has exactly ``sequence_length`` frames: ABC -> (dropped)
* ``'constant'``: Pad with a fixed value (specified by ``fill_value``): ABC -> ABCPPP
* ``'edge'`` or ``'repeat'``: Repeat the last valid frame: ABC -> ABCCCC
* ``'reflect_1001'`` or ``'symmetric'``: Reflect padding, including the last element: ABC -> ABCCBA
* ``'reflect_101'`` or ``'reflect'``: Reflect padding, not including the last element: ABC -> ABCBA

A sequence only needs its last frame to exist, so a sequence of ``sequence_length`` frames with the
given ``stride`` fits in ``(sequence_length - 1) * stride + 1`` frames. With padding, a sample starts
at every ``step`` position of the video (or of its ``file_list`` range), so the last ones hold
fewer frames of the video than ``sequence_length``, as with the legacy ``fn.readers.video``.

Not relevant when using ``frames`` argument.)code",
        "none", true)
    .AddOptionalArg("pad_sequences",
        R"code(Allows creation of incomplete sequences if there is an insufficient number
of frames at the very end of the video.)code", false)
    .DeprecateArg("pad_sequences", "2.5", false,
        R"code(Use `pad_mode` instead: ``pad_sequences=True`` is ``pad_mode='constant'``, and
``pad_sequences=False`` is ``pad_mode='none'``.)code")
    .AddOptionalArg("file_list_frame_num",
        R"code(Interpret the start and end values of `file_list` as frame numbers, not as
timestamps.)code", false)
    .DeprecateArg("file_list_frame_num", "2.5", false,
        R"code(Use `file_list_format` instead: ``file_list_frame_num=True`` is
``file_list_format='frames'``, and ``False`` is ``file_list_format='timestamps'``.)code")
    .AddOptionalArg("file_list_include_preceding_frame",
        R"code(Changes how the start and end timestamps of `file_list` are translated to frame
numbers.)code", false)
    .DeprecateArg("file_list_include_preceding_frame", "2.5", false,
        R"code(Use `file_list_rounding` instead: ``file_list_include_preceding_frame=True`` is
``file_list_rounding='start_down_end_up'``, and ``False`` is ``'start_up_end_down'``. It applies to
the ``timestamps`` format only. The legacy reader computes the frame numbers in single precision,
so for a time that is exactly the time of a frame, the frame that this operator selects can differ
from the legacy one by one.)code")
    .AddOptionalArg("skip_vfr_check", R"code(Skips the check for variable frame rate videos.)code",
                    false)
    .DeprecateArg("skip_vfr_check", "2.5", true,
        R"code(It has no effect: this operator reads variable frame rate videos.)code")
    .AddOptionalArg("channels", R"code(Number of channels.)code", 3)
    .DeprecateArg("channels", "2.5", true,
        R"code(It has no effect: the output always has 3 channels (see `image_type`).)code")
    .AddOptionalArg("fill_value",
                    R"code(Value(s) used to pad missing frames when ``pad_mode='constant'``.

Each value must be in range [0, 255].
If a single value is provided, it will be used for all channels.
Otherwise, the number of values must match the number of channels in the video.)code",
                    std::vector<int>{
                        0,
                    })
    .AddOptionalArg("image_type", R"(The color space of the output frames (RGB or YCbCr).

YCbCr output contains the decoded samples upsampled to 4:4:4, in the color range and matrix of
the stream (usually limited range); RGB output is full range.)",
                    DALI_RGB)
    .AddParent("LoaderBase")
    .OutputNDim(0, 4)
    .OutputDType(0, DALI_UINT8)
    .OutputLayout(0, "FHWC");


DALI_REGISTER_OPERATOR(experimental__readers__Video, VideoReaderDecoder<CPUBackend>, CPU);
DALI_REGISTER_OPERATOR(experimental__readers__Video, VideoReaderDecoder<GPUBackend>, GPU);

}  // namespace dali
