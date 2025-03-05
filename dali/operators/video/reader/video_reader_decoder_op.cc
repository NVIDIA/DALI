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
#include "dali/operators/reader/reader_op.h"
#include "dali/operators/video/reader/video_loader_decoder.h"
#include "dali/operators/video/frames_decoder_gpu.h"
#include "dali/operators/video/frames_decoder_cpu.h"
#include "dali/core/cuda_stream_pool.h"

namespace dali {

template <typename Backend>
class VideoReaderDecoder : public DataReader<Backend, VideoSample<Backend>, VideoSample<Backend>, true> {
 public:
  using FramesDecoderImpl = std::conditional_t<std::is_same_v<Backend, GPUBackend>,
                                         FramesDecoderGpu,
                                         FramesDecoderCpu>;
  using VideoLoaderImpl = VideoLoaderDecoder<Backend, FramesDecoderImpl>;
  using Base = DataReader<Backend, VideoSample<Backend>, VideoSample<Backend>, true>;
  using Base::loader_;
  using Base::prefetched_batch_queue_;
  using Base::curr_batch_producer_;
  using Base::Prefetch;
  using Base::GetCurrBatchSize;
  using Base::GetSample;

  explicit VideoReaderDecoder(const OpSpec &spec)
      : Base(spec),
        has_labels_(spec.HasArgument("labels")),
        has_frame_idx_(spec.GetArgument<bool>("enable_frame_num")),
        has_timestamps_(spec.GetArgument<bool>("enable_timestamps")) {
    loader_ = InitLoader<VideoLoaderImpl>(spec);
    this->SetInitialSnapshot();

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
    for (int sample_id = 0; sample_id < batch_size; ++sample_id) {
      auto &sample = GetSample(sample_id);
      video_shape.set_tensor_shape(sample_id, sample.data_.shape());
    }
    output_desc.push_back({video_shape, DALI_UINT8});

    if (has_labels_) {
      output_desc.push_back({uniform_list_shape<1>(batch_size, {1}), DALI_INT32});
    }
    if (has_frame_idx_) {
      output_desc.push_back({uniform_list_shape<1>(batch_size, {1}), DALI_INT32});
    }
    if (has_timestamps_) {
      output_desc.push_back({uniform_list_shape<1>(batch_size, {1}), DALI_INT64});
    }
    return true;
  }

  template <typename T>
  void OutputMetadata(Workspace &ws, int out_idx, std::function<T(const VideoSample<Backend>&)> get_value) {
    auto &output = ws.Output<Backend>(out_idx);
    int batch_size = output.num_samples();
    SmallVector<T, 32> values;
    values.resize(batch_size);
    for (int sample_id = 0; sample_id < batch_size; ++sample_id) {
      values[sample_id] = get_value(GetSample(sample_id));
    }
    MemCopy(output.AsTensor().raw_mutable_data(), values.data(),
            batch_size * sizeof(T), ws.has_stream() ? ws.stream() : 0);
  }

  void RunImpl(Workspace &ws) override {
    auto &video_output = ws.Output<Backend>(0);
    int batch_size = GetCurrBatchSize();

    video_output.SetLayout("FHWC");
    // Copy video data and set source info
    for (int sample_id = 0; sample_id < batch_size; ++sample_id) {
      auto &sample = GetSample(sample_id);
      MemCopy(video_output.raw_mutable_tensor(sample_id), sample.data_.raw_data(),
              sample.data_.size(), ws.has_stream() ? ws.stream() : 0);
      video_output.SetSourceInfo(sample_id, sample.data_.GetSourceInfo());
    }

    // Copy optional metadata outputs
    int out_index = 1;
    if (has_labels_) {
      OutputMetadata<int32_t>(ws, out_index++, [](auto& s) { return s.label_; });
    }
    if (has_frame_idx_) {
      OutputMetadata<int32_t>(ws, out_index++, [](auto& s) { return s.start_; });
    }
    if (has_timestamps_) {
      OutputMetadata<int64_t>(ws, out_index++, [](auto& s) { return s.start_timestamp_; });
    }
  }

  bool HasContiguousOutputs() const override { return true; }

  void Prefetch() override {
    Base::Prefetch();
    auto &current_batch = prefetched_batch_queue_[curr_batch_producer_];
    for (auto &sample : current_batch) {
      if (!decoder_ || decoder_->Filename() != sample->filename_) {
        if constexpr (std::is_same_v<Backend, CPUBackend>) {
          decoder_ = std::make_unique<FramesDecoderImpl>(sample->filename_, true);
        } else {
          decoder_ = std::make_unique<FramesDecoderImpl>(sample->filename_, true);
        }
        LOG_LINE << "Initialized decoder to " << decoder_->Filename()
                 << " ptr: " << decoder_.get() << " num_frames: " << decoder_->NumFrames()
                 << std::endl;
      }
      sample->start_timestamp_ = decoder_->Index(sample->start_).pts;

      int64_t num_frames = (sample->end_ - sample->start_ + sample->stride_ - 1) / sample->stride_;
      sample->data_.Resize(
          {num_frames, decoder_->Height(), decoder_->Width(), decoder_->Channels()},
          DALI_UINT8);
      sample->data_.SetSourceInfo(decoder_->Filename());
      sample->data_.SetLayout("FHWC");

      int64_t frame_size = decoder_->FrameSize();
      uint8_t* data = sample->data_.template mutable_data<uint8_t>();
      for (int64_t pos = sample->start_; pos < sample->end_; pos += sample->stride_) {
        decoder_->SeekFrame(pos);
        decoder_->ReadNextFrame(data);
        data += frame_size;
      }
    }
    if (cuda_stream_) {
      CUDA_CALL(cudaStreamSynchronize(cuda_stream_.get()));
    }
  }

 private:
  bool has_labels_ = false;
  bool has_frame_idx_  = false;
  bool has_timestamps_ = false;

  std::unique_ptr<FramesDecoderImpl> decoder_;

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

Each output sample is a sequence of frames with shape ``(F, H, W, C)`` where:

* ``F`` is the number of frames in the sequence (can vary between samples)
* ``H`` is the frame height in pixels
* ``W`` is the frame width in pixels
* ``C`` is the number of color channels
  
.. note::
  Containers which do not support indexing, like MPEG, require DALI to build the index.
DALI will go through the video and mark keyframes to be able to seek effectively,
even in the constant frame rate scenario.
)code")
  .NumInput(0)
  .OutputFn([](const OpSpec &spec) {
    return 1
        + spec.HasArgument("labels")
        + spec.GetArgument<bool>("enable_frame_num") 
        + spec.GetArgument<bool>("enable_timestamps");
  })
  .AddOptionalArg("filenames",
      R"code(Absolute paths to the video files to load.)code",
      std::vector<std::string>{})
    .AddOptionalArg<vector<int>>("labels", R"(Labels associated with the files listed in
`filenames` argument. If not provided, no labels will be yielded.)", nullptr)
  .AddArg("sequence_length",
      R"code(Frames to load per sequence.)code",
      DALI_INT32)
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
  .AddOptionalArg("stride",
      R"code(Distance between consecutive frames in the sequence.)code", 1u, false)
  .AddParent("LoaderBase");

DALI_REGISTER_OPERATOR(experimental__readers__Video, VideoReaderDecoder<CPUBackend>, CPU);
DALI_REGISTER_OPERATOR(experimental__readers__Video, VideoReaderDecoder<GPUBackend>, GPU);

}  // namespace dali
