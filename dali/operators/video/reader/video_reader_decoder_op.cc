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
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <exception>
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
#include <unordered_map>
#include <utility>
#include <vector>

#include "dali/core/boundary.h"
#include "dali/core/cuda_stream_pool.h"
#include "dali/core/device_guard.h"
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
#if NVML_ENABLED
#include "dali/util/nvml.h"
#endif

#include "libavutil/rational.h"

namespace dali {

struct VideoSampleDesc {
  VideoSampleDesc(const VideoFileMeta *video_file_meta = nullptr, int start = -1, int end = -1, int stride = -1)
      : video_file_meta_(video_file_meta), start_(start), end_(end), stride_(stride), frame_idxs_() {}
  const VideoFileMeta *video_file_meta_;
  int start_;
  int end_;
  int stride_;
  span<const int> frame_idxs_;  // non-empty → uniform sampling; use these instead of start_/end_/stride_
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

  // to be filled by Prefetch
  Tensor<Backend> data_;
  std::vector<double> timestamps_;
  std::vector<int32_t> frame_idx_;
  // Set when the reader plans the sample's decoding. With pad_last_batch, the loader returns the
  // last sample again, also in the next batch, which must not decode it again while the previous
  // batch may still be read.
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
        boundary_type_(GetBoundaryType(spec)),
        uniform_sample_(spec.GetArgument<bool>("uniform_sample")) {
    if ((spec.HasArgument("file_list") + spec.HasArgument("file_root") + spec.HasArgument("filenames")) != 1) {
      DALI_FAIL("Only one of the following arguments can be provided: ``file_list``, ``file_root``, ``filenames``");
    }
    bool has_labels = spec.TryGetRepeatedArgument(labels_, "labels");
    if (has_labels) {
      DALI_ENFORCE(
          labels_.size() == filenames_.size(),
          make_string(
              "Number of provided files and labels should match. Provided ",
              filenames_.size(), " files and ", labels_.size(), " labels."));
    }

    video_files_info_ = GetVideoFiles(file_root_, filenames_, has_labels, labels_, file_list_);
    DALI_ENFORCE(!video_files_info_.empty(), "No files were read.");

    if (!file_list_.empty()) {
      auto format_str = spec.GetArgument<std::string>("file_list_format");
      if (format_str == "frames") {
        file_list_opts_.format = FileListFormat::kFrames;
      } else if (format_str == "timestamps") {
        file_list_opts_.format = FileListFormat::kTimestamps;
      } else {
        DALI_FAIL(make_string("Invalid file_list_format: ", format_str));
      }

      auto rounding_str = spec.GetArgument<std::string>("file_list_rounding");
      if (rounding_str == "start_down_end_up") {
        file_list_opts_.rounding = FileListRounding::kStartDownEndUp;
      } else if (rounding_str == "start_up_end_down") {
        file_list_opts_.rounding = FileListRounding::kStartUpEndDown;
      } else if (rounding_str == "all_up") {
        file_list_opts_.rounding = FileListRounding::kAllUp;
      } else if (rounding_str == "all_down") {
        file_list_opts_.rounding = FileListRounding::kAllDown;
      } else {
        DALI_FAIL(make_string("Invalid file_list_rounding: ", rounding_str));
      }

      file_list_opts_.include_end = spec.GetArgument<bool>("file_list_include_end");
    }

    if (step_ <= 0) {
      step_ = stride_ * sequence_len_;
    }
    if (uniform_sample_) {
      DALI_ENFORCE(sequence_len_ >= 1,
                   "sequence_length must be at least 1 when uniform_sample=True.");
      if (spec.HasArgument("stride")) {
        DALI_WARN("uniform_sample=True: the `stride` argument is ignored.");
      }
      if (spec.HasArgument("step")) {
        DALI_WARN("uniform_sample=True: the `step` argument is ignored.");
      }
      if (spec.HasArgument("pad_mode")) {
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

  /**
   * @brief The frame index of one of the reader's files, nullptr if the file was not indexed
   *        (it is not a valid video).
   */
  const FrameIndex *FileIndex(const std::string &filename) const {
    auto it = indices_.find(filename);
    return it != indices_.end() ? it->second.get() : nullptr;
  }

  Index SizeImpl() override {
    return samples_.size();
  }

  void PrepareMetadataImpl() override {
    LOG_LINE << "Starting PrepareMetadataImpl" << std::endl;
    samples_.clear();
    all_frame_idxs_.clear();
    if (uniform_sample_)
      all_frame_idxs_.reserve(video_files_info_.size());
    for (size_t i = 0; i < video_files_info_.size(); ++i) {
      auto& entry = video_files_info_[i];
      LOG_LINE << "Processing video file " << i << ": " << entry.filename << std::endl;
      // the version of the file that the decoder opens (see FrameIndexCache::GetOrBuild)
      auto stamp = FrameIndexCache::FileStamp::Of(entry.filename);
      std::unique_ptr<FramesDecoderImpl> decoder;
      if constexpr(std::is_same_v<Backend, CPUBackend>) {
        decoder = std::make_unique<FramesDecoderImpl>(entry.filename, image_type_);
      } else {
        decoder = std::make_unique<FramesDecoderImpl>(entry.filename, cuda_stream_, image_type_);
      }
      if (!decoder->IsValid()) {
        LOG_LINE << "Invalid video file: " << entry.filename << std::endl;
        continue;
      }
      auto index = FrameIndexCache::instance().GetOrBuild(entry.filename, stamp, [&]() {
        LOG_LINE << "Building index for " << entry.filename << std::endl;
        decoder->BuildIndex();
        return decoder->GetIndex();
      });
      if (!decoder->HasIndex())
        decoder->SetIndex(*index);
      indices_[entry.filename] = index;  // kept for the reader's lifetime
      int64_t num_frames = decoder->NumFrames();
      entry.start_frame = 0;
      entry.end_frame = num_frames;
      LOG_LINE << "Total frames in video: " << num_frames << std::endl;
      if (entry.start != 0.0f || entry.end != 0.0f) {
        LOG_LINE << "Processing range [" << entry.start << ", " << entry.end
                 << "], file_list_format: " << make_string(file_list_opts_) << std::endl;
        switch (file_list_opts_.format) {
          case FileListFormat::kFrames:
            if (entry.start < 0)
              entry.start = num_frames + entry.start;
            if (entry.end < 0)
              entry.end = num_frames + entry.end;
            entry.start_frame = file_list_opts_.round_start(entry.start);
            entry.end_frame = file_list_opts_.round_end(entry.end);
            break;
          case FileListFormat::kTimestamps:
            entry.start_frame = decoder->GetFrameIdxByTimestamp(
                SecondsToTimestamp(decoder->GetTimebase(), entry.start),
                file_list_opts_.should_round_down_start());
            entry.end_frame = decoder->GetFrameIdxByTimestamp(
                SecondsToTimestamp(decoder->GetTimebase(), entry.end),
                file_list_opts_.should_round_down_end());
            break;
          default:
            DALI_FAIL("Invalid file_list_format");
        }
        if (file_list_opts_.include_end) {
          entry.end_frame = std::min<int>(entry.end_frame, num_frames);
        }
        LOG_LINE << "Frame range after conversion: [" << entry.start_frame << ", "
                 << entry.end_frame << "]" << std::endl;
      }

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
        int start = entry.start_frame;
        int full_seq_stride = stride_ * sequence_len_;
        for (; start + full_seq_stride <= entry.end_frame; start += step_) {
          LOG_LINE << "Adding sample with start=" << start << ", end=" << start + full_seq_stride
                   << ", stride=" << stride_ << std::endl;
          samples_.emplace_back(&entry, start, start + full_seq_stride, stride_);
        }

        // if we have a tail that doesn't fit a full sequence and we allow padding, extend the last
        // sequence
        if (boundary_type_ != boundary::BoundaryType::ISOLATED && start < entry.end_frame) {
          LOG_LINE << "Adding padded tail sample starting at frame " << start
                   << ", end=" << entry.end_frame << ", stride=" << stride_ << std::endl;
          samples_.emplace_back(&entry, start, start + full_seq_stride, stride_);
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
  FileListOptions file_list_opts_;

  std::vector<VideoFileMeta> video_files_info_;
  // the frame indices of the valid files, built (or found in FrameIndexCache) by
  // PrepareMetadataImpl
  std::unordered_map<std::string, FrameIndexCache::IndexPtr> indices_;
  std::vector<std::vector<int>> all_frame_idxs_;  // owns frame index data; samples_ hold spans into these
  std::vector<VideoSampleDesc> samples_;
  CUDAStreamLease cuda_stream_;
};

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
        boundary_type_(GetBoundaryType(spec)),
        image_type_(spec.GetArgument<DALIImageType>("image_type")) {
    // Test aid only: a lane's decoder that is not where the plan expects it fails the batch
    const char *strict_plans = std::getenv("DALI_VIDEO_READER_STRICT_PLANS");
    strict_plans_ = strict_plans && *strict_plans && std::string_view(strict_plans) != "0";
    loader_ = InitLoader<VideoLoaderImpl>(spec);
    this->SetInitialSnapshot();

    has_labels_ = spec.HasArgument("labels") ||
                  !spec.GetArgument<std::string>("file_list").empty() ||
                  !spec.GetArgument<std::string>("file_root").empty();

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
  }

  ~VideoReaderDecoder() override {
    LOG_LINE << "VideoReaderDecoder destructor" << std::endl;
    cancel_ = true;  // the lanes skip the samples they have not started
    Base::StopPrefetchThread();  // a Prefetch in progress throws once its batch is done
    StopLanes();
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
      use_streams_ = true;
    }
  }

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) override {
    Base::SetupImpl(output_desc, ws);
    output_desc.reserve(4);
    int batch_size = GetCurrBatchSize();
    TensorListShape<4> video_shape(batch_size);
    for (int sample_id = 0; sample_id < batch_size; ++sample_id) {
      auto &sample = GetSample(sample_id);
      video_shape.set_tensor_shape(sample_id, sample.data_.shape());
    }
    output_desc.push_back({video_shape, DALI_UINT8});

    if (has_labels_) {
      TensorListShape<1> label_shape = uniform_list_shape<1>(batch_size, {1});
      output_desc.push_back({label_shape, DALI_INT32});
    }

    if (frame_num_policy_ == FrameNumPolicy::Scalar) {
      TensorListShape<1> frame_idx_shape = uniform_list_shape<1>(batch_size, {1});
      output_desc.push_back({frame_idx_shape, DALI_INT32});
    } else if (frame_num_policy_ == FrameNumPolicy::Sequence) {
      TensorListShape<1> frame_idx_shape(batch_size);
      for (int sample_id = 0; sample_id < batch_size; ++sample_id) {
        auto num_frames = GetSample(sample_id).data_.shape()[0];
        frame_idx_shape.set_tensor_shape(sample_id, {num_frames});
      }
      output_desc.push_back({frame_idx_shape, DALI_INT32});
    }

    if (has_timestamps_) {
      TensorListShape<1> timestamps_shape(batch_size);
      for (int sample_id = 0; sample_id < batch_size; ++sample_id) {
        auto &sample = GetSample(sample_id);
        auto num_frames = sample.data_.shape()[0];
        timestamps_shape.set_tensor_shape(sample_id, {num_frames});
      }
      output_desc.push_back({timestamps_shape, DALI_FLOAT64});
    }
    return true;
  }

  template <typename T>
  void OutputMetadata(Workspace &ws, int out_idx,
                      std::function<span<const T>(const VideoSample<Backend>&)> get_values) {
    auto &output = ws.Output<Backend>(out_idx);
    int batch_size = output.num_samples();
    DALI_ENFORCE(output.IsContiguousInMemory(), "Output must be contiguous in memory");
    auto output_as_tensor = output.AsTensor();

    auto copy_data = [&](T* data) {
      for (int sample_id = 0; sample_id < batch_size; ++sample_id)
        for (auto &elem : get_values(GetSample(sample_id)))
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

  void RunImpl(Workspace &ws) override {
    auto &video_output = ws.Output<Backend>(0);
    int batch_size = GetCurrBatchSize();

    video_output.SetLayout("FHWC");
    AccessOrder order = std::is_same_v<Backend, GPUBackend> ? ws.stream() : AccessOrder::host();
    for (int sample_id = 0; sample_id < batch_size; ++sample_id) {
      auto &sample = GetSample(sample_id);
      video_output.CopySample(sample_id, sample.data_, order);
    }

    // Copy optional metadata outputs
    int out_index = 1;
    if (has_labels_) {
      OutputMetadata<int32_t>(ws, out_index++, [](auto &s) {
        return make_cspan(&s.video_file_meta_->label, 1);
      });
    }
    if (frame_num_policy_ == FrameNumPolicy::Scalar) {
      OutputMetadata<int32_t>(ws, out_index++, [](auto &s) {
        return make_cspan(&s.start_, 1);
      });
    } else if (frame_num_policy_ == FrameNumPolicy::Sequence) {
      OutputMetadata<int32_t>(ws, out_index++, [](auto &s) {
        return make_cspan(s.frame_idx_);
      });
    }
    if (has_timestamps_) {
      OutputMetadata<double>(ws, out_index++, [](auto &s) {
        return make_cspan(s.timestamps_);
      });
    }
  }

  bool HasContiguousOutputs() const override {
    return true;
  }

  void Prefetch() override {
    DomainTimeRange tr("[DALI][VideoReaderDecoder] Prefetch #" + to_string(curr_batch_producer_),
                       DomainTimeRange::kRed);
    auto &slot = prefetched_batch_queue_[curr_batch_producer_];
    slot.clear();  // first: a consumed batch's samples go back to the loader
    while (static_cast<int>(pending_.size()) <= decode_ahead_ && !read_failed_)
      StartBatch();
    auto batch = std::move(pending_.front());
    pending_.pop_front();
    WaitForBatch(*batch);  // after this, no lane uses the batch
    if (cancel_)  // the reader is being destroyed: nothing is published any more
      throw std::runtime_error("The video reader is being destroyed.");
    if (auto error = FirstError(*batch)) {
      cancel_ = true;  // the lanes skip the batches after it
      std::rethrow_exception(error);
    }
    slot = std::move(batch->samples);
    if (this->IsCheckpointingEnabled())
      this->SaveLoaderSnapshot(batch->snapshot);
    LOG_LINE << "Prefetch done" << std::endl;
  }

 private:
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
   * @brief The frame index of one of the reader's files (see VideoLoaderDecoder::FileIndex).
   */
  const FrameIndex *FileIndex(const std::string &filename) const {
    return static_cast<const VideoLoaderImpl &>(*loader_).FileIndex(filename);
  }

  /**
   * @brief Groups the batch's samples into runs: samples of one file that one decoder decodes in
   * a row without seeking.
   *
   * A sample repeated in the batch (pad_last_batch) is decoded once, and a sample planned in an
   * earlier batch (padding repeats the last sample, also in the next batch) is not decoded again.
   */
  void PlanBatch(PendingBatch &batch) {
    auto &tasks = batch.tasks;
    auto &runs = batch.runs;
    for (int i = 0; i < static_cast<int>(batch.samples.size()); i++) {
      auto *sample = batch.samples[i].get();
      if (sample->planned)
        continue;
      sample->planned = true;
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
    try {
      CheckPlan(lane, batch.lane_start[l]);
    } catch (...) {
      int first_task = batch.runs[plan.runs[0]].tasks[0];
      errors.push_back({batch.tasks[first_task].batch_idx, l, std::current_exception()});
    }
    bool cancelled = false;
    for (size_t i = 0; i < plan.runs.size() && !cancelled; i++) {
      for (int t : batch.runs[plan.runs[i]].tasks) {
        if ((cancelled = cancel_))
          break;
        auto &task = batch.tasks[t];
        try {
          DecodeSample(*task.sample, lane);
          if constexpr (std::is_same_v<Backend, GPUBackend>) {
            // the decoder does not wait for its GPU work: its asynchronous errors (conversions,
            // copies) are reported for this sample
            CUDA_CALL(cudaStreamSynchronize(lane.stream ? lane.stream.get() : 0));
          }
        } catch (...) {
          lane.decoder.reset();  // in an unknown state; its destructor waits for its stream
          lane.failed = true;
          errors.push_back({task.batch_idx, l, std::current_exception()});
        }
      }
    }
    if constexpr (std::is_same_v<Backend, GPUBackend>) {
      // the samples are read on the workspace stream once the batch is consumed
      cudaStream_t stream = lane.stream ? lane.stream.get() : 0;
      if (errors.empty()) {
        try {
          CUDA_CALL(cudaStreamSynchronize(stream));
        } catch (...) {
          errors.push_back({std::numeric_limits<int>::max(), l, std::current_exception()});
        }
      } else if (cudaStreamSynchronize(stream) != cudaSuccess) {
        (void)cudaGetLastError();  // already failing: report the sample's error
      }
    }
    // Planned with the files' indices, the two match (on the GPU, within a picture or two)
    LOG_LINE << "Lane " << l << " decoded " << lane.decoded_pictures << " pictures, planned "
             << plan.pictures << " (load " << plan.load << ")" << std::endl;
  }

  /**
   * @brief Decodes one sample on a lane, reusing the lane's decoder if it has the same file.
   */
  void DecodeSample(VideoSample<Backend> &sample, DecodeLane &lane) {
    const auto &filename = sample.video_file_meta_->filename;
    LOG_LINE << "Processing sample with filename " << filename << " and previous decoder "
             << lane.decoder.get() << " filename "
             << (lane.decoder ? lane.decoder->Filename() : "none") << std::endl;
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
               << lane.decoder.get() << " num_frames: " << lane.decoder->NumFrames()
               << std::endl;
    }
    auto &decoder = *lane.decoder;
    DALI_ENFORCE(decoder.IsValid(), make_string("Invalid decoder for filename ", filename));
    int64_t decoded_pictures = decoder.ParsedPictures();

    int64_t num_frames = sample.frame_idxs_.empty()
        ? (sample.end_ - sample.start_ + sample.stride_ - 1) / sample.stride_
        : static_cast<int64_t>(sample.frame_idxs_.size());
    sample.data_.Resize(
        {num_frames, decoder.Height(), decoder.Width(), decoder.Channels()}, DALI_UINT8);
    sample.data_.SetSourceInfo(decoder.Filename());
    sample.data_.SetLayout("FHWC");

    const uint8_t *constant_frame =
        boundary_type_ == boundary::BoundaryType::CONSTANT ?
            ConstantFrame(lane.constant_frame, decoder.FrameShape(), make_cspan(fill_value_),
                          lane.stream, true) :
            nullptr;
    if (has_timestamps_) {
      sample.timestamps_.resize(num_frames);
    } else {
      sample.timestamps_.clear();
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
      sample.frame_idx_.resize(num_frames);
      if (!sample.frame_idxs_.empty()) {
        // Uniform indices are always within [roi_start, roi_end-1], so HandleBoundary
        // is a no-op here — but we call it for consistency with the stride path below.
        for (int64_t i = 0; i < num_frames; ++i) {
          sample.frame_idx_[i] = static_cast<int32_t>(decoder.HandleBoundary(
              boundary_type_, sample.frame_idxs_[i], roi_start, roi_end));
        }
      } else {
        for (int64_t i = 0; i < num_frames; ++i) {
          sample.frame_idx_[i] = static_cast<int32_t>(decoder.HandleBoundary(
              boundary_type_,
              static_cast<int>(sample.start_ + i * sample.stride_),
              roi_start, roi_end));
        }
      }
    } else {
      sample.frame_idx_.clear();
    }
    if (!sample.frame_idxs_.empty()) {
      // Uniform sampling: explicit frame indices already include ROI offset (start_frame)
      decoder.DecodeFrames(sample.data_.template mutable_data<uint8_t>(),
                           make_cspan(sample.frame_idxs_), boundary_type_, constant_frame,
                           make_span(sample.timestamps_));
    } else if (roi_start != 0 || roi_end != decoder.NumFrames()) {
      lane.frame_idxs.clear();
      for (int frame_idx = sample.start_; frame_idx < sample.end_;
           frame_idx += sample.stride_) {
        lane.frame_idxs.push_back(decoder.HandleBoundary(
            boundary_type_, frame_idx, roi_start, roi_end));
      }
      decoder.DecodeFrames(sample.data_.template mutable_data<uint8_t>(),
                           make_cspan(lane.frame_idxs), boundary_type_, constant_frame,
                           make_span(sample.timestamps_));
    } else {
      decoder.DecodeFrames(sample.data_.template mutable_data<uint8_t>(), sample.start_,
                           sample.end_, sample.stride_, boundary_type_, constant_frame,
                           make_span(sample.timestamps_));
    }
    lane.decoded_pictures += decoder.ParsedPictures() - decoded_pictures;
    LOG_LINE << "Decoding frames done" << std::endl;
  }

  FrameNumPolicy frame_num_policy_;
  bool has_timestamps_;
  boundary::BoundaryType boundary_type_;
  DALIImageType image_type_;
  std::vector<uint8_t> fill_value_;
  bool has_labels_ = false;
  bool strict_plans_ = false;  // see CheckPlan

  bool use_streams_ = false;  // GPU: a stream per lane (false: the default stream, old drivers)
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
};

DALI_SCHEMA(experimental__readers__Video)
    .DocStr(R"code(Loads and decodes video files from disk.

The operator supports most common video container formats using libavformat (FFmpeg).
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
      bool has_labels = spec.HasArgument("labels") || spec.HasArgument("file_list") ||
                        spec.HasArgument("file_root");
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

* ``frames``: Use exact frame numbers (0-based). Negative values count from end.
* ``timestamps``: Use timestamps in seconds.

Default: ``timestamps``.)code",
        "timestamps")
    .AddOptionalArg("file_list_rounding",
        R"code(How to handle non-exact frame matches:

* ``start_down_end_up`` (default): Round start down and end up
* ``start_up_end_down``: Round start up and end down
* ``all_up``: Round both up
* ``all_down``: Round both down)code",
        "start_down_end_up")
    .AddOptionalArg("file_list_include_end",
        R"code(If true, include the end frame in the range. Default: true)code",
        true)
    .AddOptionalArg<vector<int>>("labels", R"(Labels associated with the files listed in
`filenames` argument. If not provided, no labels will be yielded.)",
                                 nullptr)
    .AddArg("sequence_length", R"code(Frames to load per sequence.)code", DALI_INT32)
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

When the value is less than 0, `step` is set to `sequence_length`.)code",
                    -1)
    .AddOptionalArg("stride", R"code(Distance between consecutive frames in the sequence.)code", 1u,
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

* ``'none'``: Return shorter sequences if not enough frames: ABC -> ABC
* ``'constant'``: Pad with a fixed value (specified by ``pad_value``): ABC -> ABCPPP
* ``'edge'`` or ``'repeat'``: Repeat the last valid frame: ABC -> ABCCCC
* ``'reflect_1001'`` or ``'symmetric'``: Reflect padding, including the last element: ABC -> ABCCBA
* ``'reflect_101'`` or ``'reflect'``: Reflect padding, not including the last element: ABC -> ABCBA

Not relevant when using ``frames`` argument.)code",
        "none", true)
    .AddOptionalArg("fill_value",
                    R"code(Value(s) used to pad missing frames when ``pad_mode='constant'``'.

Each value must be in range [0, 255].
If a single value is provided, it will be used for all channels.
Otherwise, the number of values must match the number of channels in the video.)code",
                    std::vector<int>{
                        0,
                    })
    .AddOptionalArg("image_type", R"(The color space of the output frames (RGB or YCbCr).)",
                    DALI_RGB)
    .AddParent("LoaderBase")
    .OutputNDim(0, 4)
    .OutputDType(0, DALI_UINT8)
    .OutputLayout(0, "FHWC");


DALI_REGISTER_OPERATOR(experimental__readers__Video, VideoReaderDecoder<CPUBackend>, CPU);
DALI_REGISTER_OPERATOR(experimental__readers__Video, VideoReaderDecoder<GPUBackend>, GPU);

}  // namespace dali
