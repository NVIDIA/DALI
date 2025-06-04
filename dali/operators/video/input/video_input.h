// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_VIDEO_INPUT_VIDEO_INPUT_H_
#define DALI_OPERATORS_VIDEO_INPUT_VIDEO_INPUT_H_

#include <deque>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "dali/kernels/common/memset.h"
#include "dali/operators/video/frames_decoder_cpu.h"
#include "dali/operators/video/frames_decoder_gpu.h"
#include "dali/pipeline/operator/builtin/input_operator.h"

namespace dali {

namespace detail {

/**
 * Determines, how is the video file split into batches.
 *
 * For a given video file and frames_per_sequence and batch_size parameters, we can determine
 * beforehand, how do the return sequences look like in terms of their length:
 *
 * For instance, for num_frames=67, frames_per_sequence=5 and batch_size=3:
 * --------------- --------------- --------------- --------------- -------
 * [   ][   ][   ] [   ][   ][   ] [   ][   ][   ] [   ][   ][   ] [   ][]
 * --------------- --------------- --------------- --------------- -------
 * 4 full batches of full sequences, and 1 (last) partial batch with 1 full sequence
 * and 2-frame partial sequence at the end.
 *
 * @param num_frames How many frames the video has.
 * @param frames_per_sequence How many frames per sequence user requested.
 * @param batch_size How many sequences make up a single batch.
 * @return Tuple: [Number of full batches,
 *                 Number of full sequences in the last batch,
 *                 Number of frames in the last (incomplete) sequence]
 */
inline auto DetermineBatchOutline(int num_frames, int frames_per_sequence, int batch_size) {
  assert(frames_per_sequence > 0);
  assert(batch_size > 0);
  // Initially, the Operator will return full batches of full sequences.
  const int num_full_batches = num_frames / (frames_per_sequence * batch_size);
  const int remaining_frames = num_frames - num_full_batches * frames_per_sequence * batch_size;
  assert(remaining_frames < frames_per_sequence * batch_size);

  // Then, in the last batch, first few sequences will be full.
  const int num_full_sequences = remaining_frames / frames_per_sequence;
  assert(num_full_sequences < batch_size);

  // And at the end of the last batch, the last sequence will be either partial or padded,
  // depending on user setting.
  const int frames_in_last_sequence = remaining_frames - num_full_sequences * frames_per_sequence;
  assert(frames_in_last_sequence < frames_per_sequence);
  assert(num_full_batches >= 0 && num_full_sequences >= 0 && frames_in_last_sequence >= 0);
  return std::make_tuple(num_full_batches, num_full_sequences, frames_in_last_sequence);
}

}  // namespace detail


template <typename Backend>
using frames_decoder_t = std::conditional_t<std::is_same<Backend, CPUBackend>::value,
                                            FramesDecoderCpu, FramesDecoderGpu>;

static const std::string next_output_data_id_trace_name_ = "next_output_data_id";  // NOLINT


template <typename Backend, typename FramesDecoderImpl = frames_decoder_t<Backend>>
class VideoInput : public InputOperator<Backend> {
 public:
  static constexpr bool is_cpu = std::is_same_v<Backend, CPUBackend>;
  using InBackend = CPUBackend;
  using OutBackend = std::conditional_t<is_cpu, CPUBackend, GPUBackend>;
  static_assert((is_cpu && std::is_same_v<FramesDecoderImpl, FramesDecoderCpu>) ||
                    (!is_cpu && std::is_same_v<FramesDecoderImpl, FramesDecoderGpu>),
                "Incompatible FramesDecoder to a given Backend");

  explicit VideoInput(const OpSpec &spec)
      : InputOperator<Backend>(spec),
        sequence_length_(spec.GetArgument<int>("sequence_length")),
        device_id_(spec.GetArgument<int>("device_id")),
        batch_size_(spec.GetArgument<int>("max_batch_size")),
        last_sequence_policy_(spec.GetArgument<std::string>("last_sequence_policy")) {
    DALI_ENFORCE(
        last_sequence_policy_ == "partial" || last_sequence_policy_ == "pad",
        make_string("Provided `last_sequence_policy` is not supported: ", last_sequence_policy_));
    if constexpr (!is_cpu) {
      thread_pool_.emplace(this->num_threads_, spec.GetArgument<int>("device_id"),
                           spec.GetArgument<bool>("affine"), "VideoInput<MixedBackend>");
    }
  }


  int NextBatchSize() override {
    return batch_size_;
  }


  void Advance() override {}


  const TensorLayout &in_layout() const override {
    return in_layout_;
  }


  int in_ndim() const override {
    return 1;
  }


  DALIDataType in_dtype() const override {
    return DALIDataType::DALI_UINT8;
  }


  bool SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) override;

  /**
   * Awkward RunImpl function for VideoInput operator.
   * @see VideoInputCpu & VideoInputMixed for more information.
   */
  void VideoInputRunImpl(Workspace &ws);


 private:
  void Invalidate() {
    initialized_ = false;
    needs_data_load_ = true;
    data_id_ = std::nullopt;
  }


  void LoadDataFromInputOperator(ThreadPool &thread_pool);

  void SetNextDataIdTrace(Workspace &ws, std::string next_data_id);


  /**
   * Determine the Output Descriptors for a given video file.
   * Fills the output_descs_ queue.
   */
  void DetermineOutputDescs(int num_frames) {
    auto [num_full_batches, num_full_sequences, frames_in_last_sequence] =
        detail::DetermineBatchOutline(num_frames, sequence_length_, batch_size_);
    // Initially, the Operator will return full batches of full sequences.
    for (int i = 0; i < num_full_batches; i++) {
      OutputDesc od;
      od.shape = uniform_list_shape(batch_size_, GetSequenceShape(sequence_length_, 0));
      od.type = DALI_UINT8;
      output_descs_.push_back(std::move(od));
    }

    if (num_full_sequences == 0 && frames_in_last_sequence == 0) {
      return;
    }

    OutputDesc od;
    od.type = DALI_UINT8;
    // Then, in the last batch, first few sequences will be full.
    od.shape = uniform_list_shape(num_full_sequences + (frames_in_last_sequence == 0 ? 0 : 1),
                                  GetSequenceShape(sequence_length_, 0));

    // And at the end of the last batch, the last sequence will be either partial or padded,
    // depending on user setting.
    if (frames_in_last_sequence != 0 && last_sequence_policy_ != "pad") {
      od.shape.set_tensor_shape(od.shape.num_samples() - 1,
                                GetSequenceShape(frames_in_last_sequence, 0));
    }
    output_descs_.push_back(std::move(od));
  }


  TensorShape<4> GetSequenceShape(int frames_per_sequence, int input_sample_idx) {
    TensorShape<4> ret;
    ret[0] = frames_per_sequence;
    ret[1] = frames_decoders_[input_sample_idx]->Height();
    ret[2] = frames_decoders_[input_sample_idx]->Width();
    ret[3] = frames_decoders_[input_sample_idx]->Channels();
    return ret;
  }


  TensorShape<3> GetFrameShape(int input_sample_idx) {
    TensorShape<3> ret;
    ret[0] = frames_decoders_[input_sample_idx]->Height();
    ret[1] = frames_decoders_[input_sample_idx]->Width();
    ret[2] = frames_decoders_[input_sample_idx]->Channels();
    return ret;
  }


  ThreadPool &GetThreadPool(const Workspace &ws) {
    if constexpr (is_cpu) {
      return ws.GetThreadPool();
    } else {
      assert(thread_pool_.has_value());
      return *thread_pool_;
    }
  }

  std::unique_ptr<FramesDecoderImpl> CreateDecoder(const char *data, size_t size, std::string_view source_info,
                                                   cudaStream_t stream = 0) {
    if constexpr (std::is_same_v<Backend, CPUBackend>) {
      return std::make_unique<FramesDecoderImpl>(data, size, source_info);
    } else {
      return std::make_unique<FramesDecoderImpl>(data, size, source_info, stream);
    }
  }

  const int sequence_length_ = {};
  const int device_id_ = {};
  const int batch_size_ = {};
  const std::string last_sequence_policy_;

  /// VideoInput is initialized, when it's ready to return decoded sequences using given input.
  bool initialized_ = false;
  /// VideoInput needs data load, if the current input has been depleted.
  bool needs_data_load_ = true;
  /// Input to the VideoInput. It's a single encoded video file.
  TensorList<InBackend> encoded_video_;
  /// DataId property of the input. @see daliSetExternalInputDataId
  std::optional<std::string> data_id_;

  /// A queue with the Output Descriptors for a given video file.
  std::deque<OutputDesc> output_descs_;

  /// Used for padding incomplete sequences.
  uint8_t pad_frame_value_ = 0;

  /// CPU operators have default Thread Pool inside Workspace. Mixed and GPU ops don't.
  std::optional<ThreadPool> thread_pool_ = std::nullopt;

  std::vector<std::unique_ptr<FramesDecoderImpl>> frames_decoders_;

  TensorLayout in_layout_ = "B";  // Byte stream.
};

template <typename Backend, typename FramesDecoderImpl>
void VideoInput<Backend, FramesDecoderImpl>::LoadDataFromInputOperator(ThreadPool &thread_pool) {
  // By definition, the input batch size of this Operator is always 1.
  static constexpr int input_batch_size = 1;
  assert(needs_data_load_);        // Data shall not be loaded if it's not needed.
  assert(this->HasDataInQueue());  // Data shall not be loaded if there's no data in queue.
  encoded_video_.Reset();
  encoded_video_.set_pinned(device_id_ != CPU_ONLY_DEVICE_ID);
  frames_decoders_.resize(input_batch_size);
  this->ForwardCurrentData(encoded_video_, data_id_, thread_pool);
  needs_data_load_ = false;
}


template <typename Backend, typename FramesDecoderImpl>
void VideoInput<Backend, FramesDecoderImpl>::SetNextDataIdTrace(Workspace &ws,
                                                                std::string next_data_id) {
  ws.SetOperatorTrace(next_output_data_id_trace_name_, std::move(next_data_id));
}

template <typename Backend, typename FramesDecoderImpl>
bool VideoInput<Backend, FramesDecoderImpl>::SetupImpl(std::vector<OutputDesc> &output_desc,
                                                       const Workspace &ws) {
  if (!initialized_) {
    if (needs_data_load_) {
      InputOperator<Backend>::HandleDataAvailability();
      auto &tp = GetThreadPool(ws);
      LoadDataFromInputOperator(tp);
    }

    auto sample = encoded_video_[0];
    const char *data = reinterpret_cast<const char *>(sample.data<uint8_t>());
    int size = sample.shape().num_elements();
    std::string_view source_info{};
    frames_decoders_[0] = CreateDecoder(data, size, source_info, ws.has_stream() ? ws.stream() : 0);
    DALI_ENFORCE(frames_decoders_[0]->IsValid(),
                 "Failed to create video decoder for provided video data");

    assert(output_descs_.empty());
    DetermineOutputDescs(static_cast<int>(frames_decoders_[0]->NumFrames()));

    // This has to be done for every video file, since we need to know the shape of the frames.
    if (last_sequence_policy_ == "pad") {
      pad_frame_value_ = 0;
    }

    initialized_ = true;
  }
  output_desc.resize(1);
  output_desc[0] = output_descs_.front();
  output_descs_.pop_front();
  return true;
}

template <typename Backend, typename FramesDecoderImpl>
void VideoInput<Backend, FramesDecoderImpl>::VideoInputRunImpl(Workspace &ws) {
  auto &output = ws.Output<OutBackend>(0);
  output.SetLayout("FHWC");

  bool full_sequence;
  auto &frames_decoder = *frames_decoders_[0];
  auto stream = ws.has_stream() ? ws.stream() : 0;
  int64_t frame_size = frames_decoder.FrameSize();
  bool input_sample_depleted = false;
  for (int64_t s = 0; s < output.num_samples(); s++) {
    uint8_t *output_data = output.template mutable_tensor<uint8_t>(s);
    int64_t f = 0;
    for (; f < sequence_length_ && frames_decoder.NextFrameIdx() != -1; f++) {
      frames_decoder.ReadNextFrame(output_data + f * frame_size);
    }

    // Handle padding if needed
    if (f < sequence_length_) {
      if (last_sequence_policy_ == "partial") {
        input_sample_depleted = true;
        break;
      } else if (f > 0) {
        assert(last_sequence_policy_ == "pad");
        auto padding_size = frame_size * (sequence_length_ - f);
        kernels::memset<detail::storage_tag_map_t<OutBackend>>(
            output_data + f * frame_size, pad_frame_value_, padding_size, stream);
      }
    }
  }

  // If true, this operator can be run again, after this Run.
  bool will_return_next = true;

  // There won't be any more output using the current input.
  input_sample_depleted |= frames_decoders_[0]->NextFrameIdx() == -1;

  if (input_sample_depleted) {
    Invalidate();
    if (this->HasDataInQueue()) {
      /*
       * Loading the next input (if available).
       * Instead of doing this in Setup, it's done in Run so that operator can assign proper
       * "next_output_data_id" trace.
       */
      LoadDataFromInputOperator(GetThreadPool(ws));
    } else {
      will_return_next = false;
    }
  }

  if (data_id_) {
    SetNextDataIdTrace(ws, *data_id_);
  }
  InputOperator<Backend>::SetDepletedOperatorTrace(ws, !will_return_next);
}

}  // namespace dali

#endif  // DALI_OPERATORS_VIDEO_INPUT_VIDEO_INPUT_H_
