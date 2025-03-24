// Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <string>
#include <vector>

#include "dali/core/boundary.h"
#include "dali/core/cuda_stream_pool.h"
#include "dali/core/span.h"

#include "dali/operators/reader/reader_op.h"
#include "dali/operators/video/frames_decoder_base.h"
#include "dali/operators/video/frames_decoder_cpu.h"
#include "dali/operators/video/frames_decoder_gpu.h"
#include "dali/operators/video/video_utils.h"

#include "libavutil/rational.h"

namespace dali {

struct VideoSampleDesc {
  VideoSampleDesc(std::string filename = {}, int label = -1, int start = -1, int end = -1,
                  int stride = -1)
      : filename_(filename), label_(label), start_(start), end_(end), stride_(stride) {}
  std::string filename_;
  int label_ = -1;
  int start_ = -1;
  int end_ = -1;
  int stride_ = -1;
};

template <typename Backend>
struct VideoSample : public VideoSampleDesc {
  VideoSample(std::string filename = {}, int label = -1, int start = -1, int end = -1,
              int stride = -1)
      : VideoSampleDesc{filename, label, start, end, stride} {}

  VideoSample(const VideoSampleDesc &other) noexcept
      : VideoSampleDesc(other) {}

  // to be filled by Prefetch
  Tensor<Backend> data_;
  std::vector<double> timestamps_;
  std::vector<int64_t> frame_idx_;
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

  bool should_round_down_end() const {
    return rounding == FileListRounding::kStartUpEndDown || rounding == FileListRounding::kAllDown;
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
        boundary_type_(GetBoundaryType(spec)) {
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

  Index SizeImpl() override {
    return samples_.size();
  }

  void PrepareMetadataImpl() override {
    LOG_LINE << "Starting PrepareMetadataImpl" << std::endl;
    for (size_t i = 0; i < video_files_info_.size(); ++i) {
      auto& entry = video_files_info_[i];
      LOG_LINE << "Processing video file " << i << ": " << entry.video_file << std::endl;
      std::unique_ptr<FramesDecoderImpl> decoder;
      if constexpr(std::is_same_v<Backend, CPUBackend>) {
        decoder = std::make_unique<FramesDecoderImpl>(entry.video_file, image_type_);
      } else {
        decoder = std::make_unique<FramesDecoderImpl>(entry.video_file, cuda_stream_, image_type_);
      }
      if (!decoder->IsValid()) {
        LOG_LINE << "Invalid video file: " << entry.video_file << std::endl;
        continue;
      } else {
        LOG_LINE << "Building index for " << entry.video_file << std::endl;
        decoder->BuildIndex();
      }
      int64_t num_frames = decoder->NumFrames();
      int start_frame = 0, end_frame = num_frames;
      LOG_LINE << "Total frames in video: " << num_frames << std::endl;
      if (entry.start_time != 0.0f || entry.end_time != 0.0f) {
        LOG_LINE << "Processing time range [" << entry.start_time << ", " << entry.end_time
                 << "], file_list_format: " << make_string(file_list_opts_) << std::endl;
        switch (file_list_opts_.format) {
          case FileListFormat::kFrames:
            if (entry.start_time < 0)
              entry.start_time = num_frames + entry.start_time;
            if (entry.end_time < 0)
              entry.end_time = num_frames + entry.end_time;
            start_frame = file_list_opts_.should_round_down_start() ? std::floor(entry.start_time) :
                                                                      std::ceil(entry.start_time);
            end_frame = file_list_opts_.should_round_down_end() ? std::floor(entry.end_time) :
                                                                  std::ceil(entry.end_time);
            break;
          case FileListFormat::kTimestamps:
            start_frame = decoder->GetFrameIdxByTimestamp(
                SecondsToTimestamp(decoder->GetTimebase(), entry.start_time),
                file_list_opts_.should_round_down_start());
            end_frame = decoder->GetFrameIdxByTimestamp(
                SecondsToTimestamp(decoder->GetTimebase(), entry.end_time),
                file_list_opts_.should_round_down_end());
            break;
          default:
            DALI_FAIL("Invalid file_list_format");
        }
        if (file_list_opts_.include_end) {
          end_frame = std::min<int>(end_frame, num_frames);
        }
        LOG_LINE << "Frame range after conversion: [" << start_frame << ", " << end_frame << "]"
                 << std::endl;
      }

      if (start_frame >= end_frame) {
        DALI_WARN(make_string("Empty frame range [", start_frame, ", ", end_frame, ") for file ",
                              entry.video_file, ". Skipping."));
        continue;
      }

      int start = start_frame;
      int full_seq_stride = stride_ * sequence_len_;
      for (; start + full_seq_stride <= end_frame; start += step_) {
        LOG_LINE << "Adding sample with start=" << start << ", end=" << start + full_seq_stride
                 << ", stride=" << stride_ << std::endl;
        samples_.emplace_back(entry.video_file, entry.label, start, start + full_seq_stride,
                              stride_);
      }

      // if we have a tail that doesn't fit a full sequence and we allow padding, extend the last
      // sequence
      if (boundary_type_ != boundary::BoundaryType::ISOLATED && start < end_frame) {
        LOG_LINE << "Adding padded tail sample starting at frame " << start << ", end=" << end_frame
                 << ", stride=" << stride_ << std::endl;
        samples_.emplace_back(entry.video_file, entry.label, start, start + full_seq_stride,
                              stride_);
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
  FileListOptions file_list_opts_;

  std::vector<VideoFileMeta> video_files_info_;
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
        has_frame_idx_(spec.GetArgument<bool>("enable_frame_num")),
        has_timestamps_(spec.GetArgument<bool>("enable_timestamps")),
        boundary_type_(GetBoundaryType(spec)),
        image_type_(spec.GetArgument<DALIImageType>("image_type")) {
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
      cuda_stream_ = CUDAStreamPool::instance().Get(device_id);
    }
    DALI_ENFORCE(image_type_ == DALI_RGB || image_type_ == DALI_YCbCr,
                 make_string("Invalid image_type: ", image_type_));
  }

  ~VideoReaderDecoder() override {
    LOG_LINE << "VideoReaderDecoder destructor" << std::endl;
    Base::StopPrefetchThread();
    LOG_LINE << "VideoReaderDecoder destructor done" << std::endl;
  }

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) override {
    Base::SetupImpl(output_desc, ws);
    output_desc.reserve(4);
    int batch_size = GetCurrBatchSize();
    TensorListShape<4> video_shape(batch_size);
    TensorListShape<1> timestamps_shape(batch_size);
    TensorListShape<1> frame_idx_shape(batch_size);
    TensorListShape<1> label_shape = uniform_list_shape<1>(batch_size, {1});
    for (int sample_id = 0; sample_id < batch_size; ++sample_id) {
      auto &sample = GetSample(sample_id);
      video_shape.set_tensor_shape(sample_id, sample.data_.shape());
      timestamps_shape.set_tensor_shape(sample_id, {sample.data_.shape()[0]});
    }
    output_desc.push_back({video_shape, DALI_UINT8});
    if (has_labels_) {
      output_desc.push_back({label_shape, DALI_INT32});
    }
    if (has_frame_idx_) {
      output_desc.push_back({label_shape, DALI_INT32});
    }
    if (has_timestamps_) {
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
      Tensor<CPUBackend> tmp;
      tmp.set_pinned(true);
      tmp.Resize(output_as_tensor.shape(), output_as_tensor.type());
      copy_data(tmp.mutable_data<T>());
      output_as_tensor.Copy(tmp, ws.stream());
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
        return make_cspan(&s.label_, 1);
      });
    }
    if (has_frame_idx_) {
      OutputMetadata<int32_t>(ws, out_index++, [](auto &s) {
        return make_cspan(&s.start_, 1);
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
    Base::Prefetch();
    auto &current_batch = prefetched_batch_queue_[curr_batch_producer_];
    size_t i = 0;
    // decoder_.reset();  // TODO(janton): remove this
    for (auto &sample : current_batch) {
      LOG_LINE << "Processing sample " << i++ << " with filename " << sample->filename_
               << " and previous decoder " << decoder_.get() << " filename "
               << (decoder_ ? decoder_->Filename() : "none") << std::endl;
      auto prev_filename = decoder_ ? decoder_->Filename() : "";
      if (prev_filename != sample->filename_) {
        if constexpr (std::is_same_v<Backend, CPUBackend>) {
          decoder_ = std::make_unique<FramesDecoderImpl>(sample->filename_, image_type_);
        } else {
          decoder_ = std::make_unique<FramesDecoderImpl>(sample->filename_, cuda_stream_, image_type_);
        }
        LOG_LINE << "Initialized decoder to " << decoder_->Filename() << " ptr: " << decoder_.get()
                 << " num_frames: " << decoder_->NumFrames() << std::endl;
        decoder_->BuildIndex();
      } else {
        LOG_LINE << "Reusing decoder for " << decoder_->Filename() << " ptr: " << decoder_.get()
                 << " num_frames: " << decoder_->NumFrames() << std::endl;
      }
      DALI_ENFORCE(decoder_->IsValid(),
                   make_string("Invalid decoder for filename ", sample->filename_));

      int64_t num_frames = (sample->end_ - sample->start_ + sample->stride_ - 1) / sample->stride_;
      sample->data_.set_pinned(std::is_same_v<Backend, GPUBackend>);
      sample->data_.Resize(
          {num_frames, decoder_->Height(), decoder_->Width(), decoder_->Channels()}, DALI_UINT8);
      sample->data_.SetSourceInfo(decoder_->Filename());
      sample->data_.SetLayout("FHWC");

      const uint8_t *constant_frame =
          boundary_type_ == boundary::BoundaryType::CONSTANT ?
              ConstantFrame(constant_frame_, decoder_->FrameShape(), make_cspan(fill_value_),
                            cuda_stream_, true) :
              nullptr;
      if (has_timestamps_) {
        sample->timestamps_.resize(num_frames);
      } else {
        sample->timestamps_.clear();
      }
      LOG_LINE << "Decoding frames start=" << sample->start_ << ", end=" << sample->end_
               << ", stride=" << sample->stride_ << ", num_frames=" << num_frames << std::endl;
      decoder_->DecodeFrames(sample->data_.template mutable_data<uint8_t>(), sample->start_,
                            sample->end_, sample->stride_, boundary_type_, constant_frame,
                            make_span(sample->timestamps_));
      LOG_LINE << "Decoding frames done" << std::endl;
    }

    if (cuda_stream_) {
      CUDA_CALL(cudaStreamSynchronize(cuda_stream_.get()));
    }
    LOG_LINE << "Prefetch done" << std::endl;
  }

 private:
  bool has_frame_idx_;
  bool has_timestamps_;
  boundary::BoundaryType boundary_type_;
  DALIImageType image_type_;
  std::vector<uint8_t> fill_value_;
  bool has_labels_ = false;

  Tensor<Backend> constant_frame_;
  CUDAStreamLease cuda_stream_;
  std::unique_ptr<FramesDecoderImpl> decoder_;  // keeping one decoder open.
};

DALI_SCHEMA(experimental__readers__Video)
    .DocStr(R"code(Loads and decodes video files from disk.

The operator supports most common video container formats using libavformat (FFmpeg).
The operator utilizes either libavcodec (FFmpeg) or NVIDIA Video Codec SDK (NVDEC) for decoding the frames.

The following video codecs are supported by both CPU and GPU backends:

* H.264/AVC
* H.265/HEVC
* VP8
* VP9
* MJPEG

The following codecs are supported by the GPU backend only:

* AV1
* MPEG-4

The outputs of the operator are: video, [labels], [frame_idx], [timestamp].

* ``video``: A sequence of frames with shape ``(F, H, W, C)`` where ``F`` is the number of frames in the sequence
  (can vary between samples), ``H`` is the frame height in pixels, ``W`` is the frame width in pixels, and ``C`` is
  the number of color channels.
* ``labels``: Label associated with the sample. Only available when using ``labels`` with ``filenames``, or when
  using ``file_list`` or ``file_root``.
* ``frame_idx``: Index of first frame in sequence. Only available when ``enable_frame_num=True``.
* ``timestamps``: Time in seconds of each frame in the sequence. Only available when ``enable_timestamps=True``.
)code")
    .NumInput(0)
    .OutputFn([](const OpSpec &spec) {
      bool has_labels = spec.HasArgument("labels") || spec.HasArgument("file_list") ||
                        spec.HasArgument("file_root");
      return 1 + has_labels + spec.GetArgument<bool>("enable_frame_num") +
             spec.GetArgument<bool>("enable_timestamps");
    })
    .AddOptionalArg("filenames",
                    R"code(Absolute paths to the video files to load.

This option is mutually exclusive with `file_root` and `file_list`.)code",
                    std::vector<std::string>{})
    .AddOptionalArg("file_root",
                    R"code(Path to a directory that contains the data files.

This option is mutually exclusive with `filenames` and `file_list`.)code",
                    std::string())
    .AddOptionalArg("file_list",
                    R"code(Path to the file with a list of ``file label [start [end]]`` values.

``start`` and ``end`` are optional and can be used to specify the start and end of the video to load.
The values can be interpreted differently depending on the ``file_list_format``.

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
                    R"code(If set, returns the index of the first frame in the decoded sequence
as an additional output.)code",
                    false)
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
    .AddParent("LoaderBase");

DALI_REGISTER_OPERATOR(experimental__readers__Video, VideoReaderDecoder<CPUBackend>, CPU);
DALI_REGISTER_OPERATOR(experimental__readers__Video, VideoReaderDecoder<GPUBackend>, GPU);

}  // namespace dali
