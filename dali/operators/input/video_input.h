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

#ifndef DALI_OPERATORS_INPUT_VIDEO_INPUT_H_
#define DALI_OPERATORS_INPUT_VIDEO_INPUT_H_

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <deque>
#include <string>
#include <utility>
#include <vector>
#include "dali/pipeline/operator/builtin/input_operator.h"
#include "dali/operators/decoder/video/video_decoder_base.h"
#include "dali/operators/reader/loader/video/frames_decoder.h"
#include "dali/operators/reader/loader/video/frames_decoder_gpu.h"

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


/**
 * Helper structure to handle Pad frames.
 *
 * Pad frame is a frame, which will be used for padding incomplete sequences.
 *
 * This structure, given the value and shape, generates a pad frame on a proper device.
 *
 * @tparam OutBackend Backend, on which the pad frame will be allocated.
 * @tparam PadType Type of the pad value.
 */
template<typename OutBackend, typename PadType>
struct PadFrameCreator {
  static constexpr bool is_cpu = std::is_same_v<OutBackend, CPUBackend>;

  template<typename T>
  using container_type =
          std::conditional_t<is_cpu, thrust::host_vector<T>, thrust::device_vector<T>>;

  PadFrameCreator() = default;


  PadFrameCreator(PadType value, TensorShape<3> frame_shape,
                  std::optional<cudaStream_t> stream = std::nullopt) {
    Initialize(value, frame_shape, stream);
  }


  void Initialize(PadType value, TensorShape<3> frame_shape,
                  std::optional<cudaStream_t> stream = std::nullopt) {
    pad_frame_data_ = std::vector<PadType>(frame_shape.num_elements(), value);
    pad_frame_.Resize(frame_shape, TypeTable::GetTypeId<PadType>());
    if (stream) {
      pad_frame_.Copy(pad_frame_data_, *stream);
    } else {
      pad_frame_.Copy(pad_frame_data_);
    }
  }


  /// Buffer with the data used for padding.
  std::vector<PadType> pad_frame_data_;
  /// Pad frame.
  Tensor<OutBackend> pad_frame_;


  SampleView<OutBackend> GetPadFrame() {
    return sample_view(pad_frame_);
  }
};


}  // namespace detail


template<typename Backend>
using frames_decoder_t =
        std::conditional_t<
                std::is_same<Backend, CPUBackend>::value,
                FramesDecoder,
                FramesDecoderGpu
        >;

static const std::string next_output_data_id_trace_name_ = "next_output_data_id";  // NOLINT


template<typename Backend, typename FramesDecoder = frames_decoder_t<Backend>>
class VideoInput : public VideoDecoderBase<Backend, FramesDecoder>, public InputOperator<Backend> {
 public:
  static constexpr bool is_cpu = std::is_same_v<Backend, CPUBackend>;
  using InBackend = CPUBackend;
  using OutBackend = std::conditional_t<is_cpu, CPUBackend, GPUBackend>;
  static_assert((is_cpu && std::is_same_v<FramesDecoder, dali::FramesDecoder>) ||
                (!is_cpu && std::is_same_v<FramesDecoder, FramesDecoderGpu>),
                "Incompatible FramesDecoder to a given Backend");

  using VideoDecoderBase<Backend, FramesDecoder>::frames_decoders_;


  explicit VideoInput(const OpSpec &spec) :
          InputOperator<Backend>(spec),
          sequence_length_(spec.GetArgument<int>("sequence_length")),
          device_id_(spec.GetArgument<int>("device_id")),
          batch_size_(spec.GetArgument<int>("max_batch_size")),
          last_sequence_policy_(spec.GetArgument<std::string>("last_sequence_policy")) {
    DALI_ENFORCE(last_sequence_policy_ == "partial" || last_sequence_policy_ == "pad",
                 make_string("Provided `last_sequence_policy` is not supported: ",
                             last_sequence_policy_));
    if constexpr (!is_cpu) {
      thread_pool_.emplace(this->num_threads_, spec.GetArgument<int>("device_id"),
                           spec.GetArgument<bool>("affine"), "VideoInput<MixedBackend>");
    }
  }


  bool CanInferOutputs() const override {
    return true;
  }


  int NextBatchSize() override {
    return batch_size_;
  }


  void Advance() override {
  }


  const TensorLayout& in_layout() const override {
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

  void CreateDecoder(const Workspace &ws);


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
      output_descs_.push_back(od);
    }

    if (num_full_sequences == 0 && frames_in_last_sequence == 0) {
      return;
    }

    OutputDesc od;
    od.type = DALI_UINT8;
    // Then, in the last batch, first few sequences will be full.
    od.shape = uniform_list_shape(
            num_full_sequences + (frames_in_last_sequence == 0 ? 0 : 1),
            GetSequenceShape(sequence_length_, 0));

    // And at the end of the last batch, the last sequence will be either partial or padded,
    // depending on user setting.
    if (frames_in_last_sequence != 0 && last_sequence_policy_ != "pad") {
      od.shape.set_tensor_shape(od.shape.num_samples() - 1,
                                GetSequenceShape(frames_in_last_sequence, 0));
    }
    output_descs_.push_back(od);
  }


  void InitializePadValue(uint8_t value, std::optional<cudaStream_t> stream) {
    pad_frame_creator_ = {value, GetFrameShape(0), stream};
  }


  TensorShape<4> GetSequenceShape(int frames_per_sequence, int input_sample_idx) {
    TensorShape<4> ret;
    ret[0] = frames_per_sequence;
    ret[1] = this->frames_decoders_[input_sample_idx]->Height();
    ret[2] = this->frames_decoders_[input_sample_idx]->Width();
    ret[3] = this->frames_decoders_[input_sample_idx]->Channels();
    return ret;
  }


  TensorShape<3> GetFrameShape(int input_sample_idx) {
    TensorShape<3> ret;
    ret[0] = this->frames_decoders_[input_sample_idx]->Height();
    ret[1] = this->frames_decoders_[input_sample_idx]->Width();
    ret[2] = this->frames_decoders_[input_sample_idx]->Channels();
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
  detail::PadFrameCreator<OutBackend, uint8_t> pad_frame_creator_;

  /// CPU operators have default Thread Pool inside Workspace. Mixed and GPU ops don't.
  std::optional<ThreadPool> thread_pool_ = std::nullopt;

  TensorLayout in_layout_ = "B";  // Byte stream.
};


/**
 * Checks, if the operator described by a given Schema is a VideoInput operator.
 */
inline bool IsVideoInput(const OpSchema& schema) {
  return schema.name() == "experimental__inputs__Video";
}


template<typename Backend, typename FramesDecoder>
void VideoInput<Backend, FramesDecoder>::LoadDataFromInputOperator(ThreadPool &thread_pool) {
  // By definition, the input batch size of this Operator is always 1.
  static constexpr int input_batch_size = 1;
  assert(needs_data_load_);  // Data shall not be loaded if it's not needed.
  assert(this->HasDataInQueue());  // Data shall not be loaded if there's no data in queue.
  encoded_video_.Reset();
  encoded_video_.set_pinned(device_id_ != CPU_ONLY_DEVICE_ID);
  frames_decoders_.resize(input_batch_size);
  this->ForwardCurrentData(encoded_video_, data_id_, thread_pool);
  needs_data_load_ = false;
}


template<typename Backend, typename FramesDecoder>
void
VideoInput<Backend, FramesDecoder>::SetNextDataIdTrace(Workspace &ws, std::string next_data_id) {
  ws.SetOperatorTrace(next_output_data_id_trace_name_, std::move(next_data_id));
}


template<typename Backend, typename FramesDecoder>
bool VideoInput<Backend, FramesDecoder>::SetupImpl(std::vector<OutputDesc> &output_desc,
                                                   const Workspace &ws) {
  if (!initialized_) {
    if (needs_data_load_) {
      InputOperator<Backend>::HandleDataAvailability();
      auto &tp = GetThreadPool(ws);
      LoadDataFromInputOperator(tp);
    }

    CreateDecoder(ws);

    assert(output_descs_.empty());
    DetermineOutputDescs(static_cast<int>(frames_decoders_[0]->NumFrames()));

    // This has to be done for every video file, since we need to know the shape of the frames.
    if (last_sequence_policy_ == "pad") {
      InitializePadValue(0, !is_cpu ? std::make_optional(ws.stream()) : std::nullopt);
    }

    initialized_ = true;
  }
  output_desc.resize(1);
  output_desc[0] = output_descs_.front();
  output_descs_.pop_front();
  return true;
}


template<typename Backend, typename FramesDecoder>
void VideoInput<Backend, FramesDecoder>::VideoInputRunImpl(Workspace &ws) {
  auto &output = ws.Output<OutBackend>(0);
  output.SetLayout("FHWC");

  bool full_sequence;
  for (int64_t s = 0; s < output.num_samples(); s++) {
    auto pad_value = last_sequence_policy_ == "pad" ? std::optional<SampleView<OutBackend>>(
            pad_frame_creator_.GetPadFrame()) : std::nullopt;
    full_sequence = this->DecodeFrames(output[s], 0, sequence_length_, pad_value,
                                       ws.has_stream() ? std::make_optional(ws.stream())
                                                       : std::nullopt);
    if (!full_sequence) {
      break;
    }
  }

  // If true, this operator can be run again, after this Run.
  bool will_return_next = true;

  // There won't be any more output using the current input.
  bool input_sample_depleted = !full_sequence || frames_decoders_[0]->NextFrameIdx() == -1;

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

#endif  // DALI_OPERATORS_INPUT_VIDEO_INPUT_H_
