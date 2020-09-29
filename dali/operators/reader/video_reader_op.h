// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_OPERATORS_READER_VIDEO_READER_OP_H_
#define DALI_OPERATORS_READER_VIDEO_READER_OP_H_

#include <string>
#include <vector>

#include "dali/operators/reader/reader_op.h"
#include "dali/operators/reader/loader/video_loader.h"


namespace dali {
namespace detail {
inline int VideoReaderOutputFn(const OpSpec &spec) {
    std::string file_root = spec.GetArgument<std::string>("file_root");
    std::string file_list = spec.GetArgument<std::string>("file_list");
    bool enable_frame_num = spec.GetArgument<bool>("enable_frame_num");
    bool enable_timestamps = spec.GetArgument<bool>("enable_timestamps");
    int num_outputs = 1;
    if (!file_root.empty() || !file_list.empty()) {
        num_outputs++;
        if (enable_frame_num) num_outputs++;
        if (enable_timestamps) num_outputs++;
    }
    return num_outputs;
}
}  // namespace detail

class VideoReader : public DataReader<GPUBackend, SequenceWrapper> {
 public:
  explicit VideoReader(const OpSpec &spec)
  : DataReader<GPUBackend, SequenceWrapper>(spec),
    filenames_(spec.GetRepeatedArgument<std::string>("filenames")),
    file_root_(spec.GetArgument<std::string>("file_root")),
    file_list_(spec.GetArgument<std::string>("file_list")),
    enable_frame_num_(spec.GetArgument<bool>("enable_frame_num")),
    enable_timestamps_(spec.GetArgument<bool>("enable_timestamps")),
    count_(spec.GetArgument<int>("sequence_length")),
    channels_(spec.GetArgument<int>("channels")),
    dtype_(spec.GetArgument<DALIDataType>("dtype")) {
    DALIImageType image_type(spec.GetArgument<DALIImageType>("image_type"));

    prefetched_batch_tensors_.resize(prefetch_queue_depth_);

    int arg_count = !filenames_.empty() + !file_root_.empty() + !file_list_.empty();

    DALI_ENFORCE(arg_count == 1,
                 "Only one of `filenames`, `file_root` or `file_list` argument "
                 "must be specified at once");

    DALI_ENFORCE(image_type == DALI_RGB || image_type == DALI_YCbCr,
                 "Image type must be RGB or YCbCr.");

    DALI_ENFORCE(dtype_ == DALI_FLOAT || dtype_ == DALI_UINT8,
                 "Data type must be FLOAT or UINT8.");

    enable_label_output_ = !file_root_.empty() || !file_list_.empty();
    DALI_ENFORCE(enable_label_output_ || !enable_frame_num_,
                "frame numbers can be enabled only when "
                "`file_list` or `file_root` argument is passed");
    DALI_ENFORCE(enable_label_output_ || !enable_timestamps_,
                "timestamps can be enabled only when "
                "`file_list` or `file_root` argument is passed");

    // TODO(spanev): Factor out the constructor body to make VideoReader compatible with lazy_init.
    loader_ = InitLoader<VideoLoader>(spec, filenames_);

    if (enable_label_output_) {
      label_shape_ = uniform_list_shape(batch_size_, {1});

      if (enable_frame_num_)
        frame_num_shape_ = label_shape_;
      if (enable_timestamps_)
        timestamp_shape_ = uniform_list_shape(batch_size_, {count_});
    }
  }

  inline ~VideoReader() {
    // wait for all kernels working on the sequences and the underlying memory is kept in
    // prefetched_batch_tensors_ and it will be destoryied now
    for (auto &batch : prefetched_batch_queue_) {
      for (auto &sample : batch) {
        sample->wait();
      }
    }
  }

 protected:
  virtual void SetOutputShapeType(TensorList<GPUBackend> &output, DeviceWorkspace &ws) {
    output.Resize(prefetched_batch_tensors_[curr_batch_consumer_].shape(),
                  prefetched_batch_tensors_[curr_batch_consumer_].type());
  }

  void PrepareAdditionalOutputs(DeviceWorkspace &ws) {
    if (enable_label_output_) {
      int output_index = 1;
      label_output_ = &ws.Output<GPUBackend>(output_index++);
      label_output_->set_type(TypeTable::GetTypeInfoFromStatic<int>());
      label_output_->Resize(label_shape_);
      if (enable_frame_num_) {
        frame_num_output_ = &ws.Output<GPUBackend>(output_index++);
        frame_num_output_->set_type(TypeTable::GetTypeInfoFromStatic<int>());
        frame_num_output_->Resize(frame_num_shape_);
      }

      if (enable_timestamps_) {
        timestamp_output_ = &ws.Output<GPUBackend>(output_index++);
        timestamp_output_->set_type(TypeTable::GetTypeInfoFromStatic<double>());
        timestamp_output_->Resize(timestamp_shape_);
      }
    }
  }

  virtual void ProcessVideo(
    TensorList<GPUBackend> &video_output,
    TensorList<GPUBackend> &video_batch,
    DeviceWorkspace &ws) {
    video_output.Copy(video_batch, ws.stream());
  }

  void ProcessAdditionalOutputs(
    int data_idx, SequenceWrapper &prefetched_video, cudaStream_t stream) {
    if (enable_label_output_) {
      auto *label = label_output_->mutable_tensor<int>(data_idx);
      CUDA_CALL(cudaMemcpyAsync(
        label, &prefetched_video.label, sizeof(int), cudaMemcpyDefault, stream));
      if (enable_frame_num_) {
        auto *frame_num = frame_num_output_->mutable_tensor<int>(data_idx);
        CUDA_CALL(cudaMemcpyAsync(
          frame_num, &prefetched_video.first_frame_idx, sizeof(int), cudaMemcpyDefault, stream));
      }
      if (enable_timestamps_) {
        auto *timestamp = timestamp_output_->mutable_tensor<double>(data_idx);
        timestamp_output_->type().Copy<GPUBackend, CPUBackend>(
          timestamp,
          prefetched_video.timestamps.data(),
          prefetched_video.timestamps.size(),
          stream);
      }
    }
  }

  void RunImpl(DeviceWorkspace &ws) override {
    auto& video_output = ws.Output<GPUBackend>(0);
    auto& curent_batch = prefetched_batch_tensors_[curr_batch_consumer_];

    SetOutputShapeType(video_output, ws);
    PrepareAdditionalOutputs(ws);

    ProcessVideo(video_output, curent_batch, ws);
    video_output.SetLayout("FHWC");

    for (size_t data_idx = 0; data_idx < curent_batch.ntensor(); ++data_idx) {
      auto& prefetched_video = GetSample(data_idx);
      ProcessAdditionalOutputs(data_idx, prefetched_video, ws.stream());
    }
  }

  // override prefetching here
  void Prefetch() override;

  static constexpr int sequence_dim = 4;
  std::vector<std::string> filenames_;
  std::string file_root_;
  std::string file_list_;
  bool enable_frame_num_;
  bool enable_timestamps_;
  int count_;
  int channels_;

  TensorListShape<> label_shape_;
  TensorListShape<> timestamp_shape_;
  TensorListShape<> frame_num_shape_;

  TensorList<GPUBackend> *label_output_ = NULL;
  TensorList<GPUBackend> *frame_num_output_ = NULL;
  TensorList<GPUBackend> *timestamp_output_ = NULL;

  vector<TensorList<GPUBackend>> prefetched_batch_tensors_;

  DALIDataType dtype_;
  bool enable_label_output_;

  USE_READER_OPERATOR_MEMBERS(GPUBackend, SequenceWrapper);
};

}  // namespace dali

#endif  // DALI_OPERATORS_READER_VIDEO_READER_OP_H_
