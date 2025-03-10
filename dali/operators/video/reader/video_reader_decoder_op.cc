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
#include "dali/core/span.h"
#include "dali/core/cuda_stream_pool.h"
#include "dali/operators/reader/reader_op.h"
#include "dali/operators/video/frames_decoder_cpu.h"
#include "dali/operators/video/frames_decoder_gpu.h"
#include "dali/operators/video/reader/video_loader_decoder.h"

namespace dali {

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
        has_timestamps_(spec.GetArgument<bool>("enable_timestamps")) {
    loader_ = InitLoader<VideoLoaderImpl>(spec);
    this->SetInitialSnapshot();

    has_labels_ = spec.HasArgument("labels") ||
                  !spec.GetArgument<std::string>("file_list").empty() ||
                  !spec.GetArgument<std::string>("file_root").empty();

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
  }

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) override {
    Base::SetupImpl(output_desc, ws);
    output_desc.reserve(4);
    int batch_size = GetCurrBatchSize();
    TensorListShape<4> video_shape(batch_size);
    TensorListShape<1> timestamps_shape(batch_size);
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
      output_desc.push_back({timestamps_shape, DALI_FLOAT});
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
      OutputMetadata<float>(ws, out_index++, [](auto &s) {
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

    // keeping one decoder open. In case the next sample belongs to the same file,
    // we reuse the same decoder.
    std::unique_ptr<FramesDecoderImpl> decoder;

    int i = 0;
    for (auto &sample : current_batch) {
      auto prev_filename = decoder ? decoder->Filename() : "";
      if (!decoder || decoder->Filename() != sample->filename_) {
        if constexpr (std::is_same_v<Backend, CPUBackend>) {
          decoder = std::make_unique<FramesDecoderImpl>(sample->filename_, true);
        } else {
          decoder = std::make_unique<FramesDecoderImpl>(sample->filename_, true, cuda_stream_);
        }
        LOG_LINE << "Initialized decoder to " << decoder->Filename() << " ptr: " << decoder.get()
                 << " num_frames: " << decoder->NumFrames() << std::endl;
      } else {
        LOG_LINE << "Reusing decoder for " << decoder->Filename() << " ptr: " << decoder.get()
                 << " num_frames: " << decoder->NumFrames() << std::endl;
      }
      DALI_ENFORCE(decoder->IsValid(),
                   make_string("Invalid decoder for filename ", sample->filename_));

      int64_t num_frames = (sample->end_ - sample->start_ + sample->stride_ - 1) / sample->stride_;
      if (has_timestamps_) {
        sample->timestamps_.reserve(num_frames);
      }

      sample->data_.set_pinned(std::is_same_v<Backend, GPUBackend>);
      sample->data_.Resize(
          {num_frames, decoder->Height(), decoder->Width(), decoder->Channels()}, DALI_UINT8);
      sample->data_.SetSourceInfo(decoder->Filename());
      sample->data_.SetLayout("FHWC");

      int64_t frame_size = decoder->FrameSize();
      uint8_t *data = sample->data_.template mutable_data<uint8_t>();
      int64_t pts_0 = decoder->Index(0).pts;
      int64_t pos = sample->start_;
      for (; pos < sample->end_; pos += sample->stride_, data += frame_size) {
        decoder->SeekFrame(pos);
        if (has_timestamps_) {
          sample->timestamps_.push_back(
              TimestampToSeconds(decoder->GetTimebase(), decoder->Index(pos).pts - pts_0));
        }
        decoder->ReadNextFrame(data);
      }
    }
    if (cuda_stream_) {
      CUDA_CALL(cudaStreamSynchronize(cuda_stream_.get()));
    }
  }

 private:
  bool has_labels_ = false;
  bool has_frame_idx_ = false;
  bool has_timestamps_ = false;
  CUDAStreamLease cuda_stream_;
};

DALI_SCHEMA(experimental__readers__Video)
    .DocStr(R"code(Loads and decodes video files from disk.

The operator supports most common video container formats using libavformat (FFmpeg).
The operator utilizes either libavcodec (FFmpeg) or NVIDIA Video Codec SDK (NVDEC) for decoding the frames.

The following video codecs are supported by both CPU and Mixed backends:

* H.264/AVC
* H.265/HEVC
* VP8
* VP9
* MJPEG

The following codecs are supported by the Mixed backend only:

* AV1
* MPEG-4

The outputs of the operator are: video, [labels], [frame_idx], [timestamp].

* ``video``: A sequence of frames with shape ``(F, H, W, C)`` where ``F`` is the number of frames in the sequence
  (can vary between samples), ``H`` is the frame height in pixels, ``W`` is the frame width in pixels, and ``C`` is
  the number of color channels.
* ``labels``: Label associated with the sample. Only available when using ``labels`` with ``filenames``, or when
  using ``file_list`` or ``file_root``.
* ``frame_idx``: Index of first frame in sequence. Only available when ``enable_frame_num=True``.
* ``timestamp``: Time in seconds of first frame in sequence. Only available when ``enable_timestamps=True``.
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
    .AddOptionalArg(
        "file_list_format",
        R"code(Policy to interpret the ``start`` and ``end`` values, when using ``file_list`` argument.

The following policies are supported:

* ``frame_index``: The values are interpreted as the exact frame number to start or end the video.
  In case of negative values, they are interpreted as a number of frames from the end of the video.
  In case of floating point values, the start frame number will be rounded up and the end frame number will be rounded down.
  Frame numbers start from 0.

* ``timestamp``: The values are interpreted as the timestamp of the frame to start or end the video.
  When no exact timestamp is found, the start timestamp is rounded up to the next available frame and the end timestamp is rounded down.

* ``timestamp_inclusive``: The values are interpreted as the timestamp of the frame to start or end the video.
  When no exact timestamp is found, the start timestamp is rounded down to the next available frame and the end timestamp is rounded up.

The default policy is ``timestamp``.)code",
        "timestamp")
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
    .AddParent("LoaderBase");

DALI_REGISTER_OPERATOR(experimental__readers__Video, VideoReaderDecoder<CPUBackend>, CPU);
DALI_REGISTER_OPERATOR(experimental__readers__Video, VideoReaderDecoder<GPUBackend>, GPU);

}  // namespace dali
