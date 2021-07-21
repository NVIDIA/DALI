// Copyright (c) 2017-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <algorithm>

#include "dali/operators/reader/loader/video_loader.h"
#include "dali/operators/reader/reader_op.h"

namespace dali {
namespace detail {
inline int VideoReaderOutputFn(const OpSpec &spec) {
  std::string file_root = spec.GetArgument<std::string>("file_root");
  std::string file_list = spec.GetArgument<std::string>("file_list");
  std::vector<std::string> file_names = spec.GetRepeatedArgument<std::string>("filenames");
  std::vector<int> labels;
  bool has_labels_arg = spec.TryGetRepeatedArgument(labels, "labels");
  bool enable_frame_num = spec.GetArgument<bool>("enable_frame_num");
  bool enable_timestamps = spec.GetArgument<bool>("enable_timestamps");
  int num_outputs = 1;
  if ((!file_names.empty() && has_labels_arg) || !file_root.empty() || !file_list.empty()) {
    ++num_outputs;
  }
  if (!file_list.empty() || !file_names.empty()) {
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

    bool has_labels_arg = spec.TryGetRepeatedArgument(labels_, "labels");

    prefetched_batch_tensors_.resize(prefetch_queue_depth_);

    int arg_count = !filenames_.empty() + !file_root_.empty() + !file_list_.empty();

    DALI_ENFORCE(arg_count == 1,
                 "Only one of `filenames`, `file_root` or `file_list` argument "
                 "must be specified at once");

    DALI_ENFORCE(image_type == DALI_RGB || image_type == DALI_YCbCr,
                 "Image type must be RGB or YCbCr.");

    DALI_ENFORCE(dtype_ == DALI_FLOAT || dtype_ == DALI_UINT8, "Data type must be FLOAT or UINT8.");

    can_use_frames_timestamps_ = !file_list_.empty() || (!filenames_.empty() && has_labels_arg);

    DALI_ENFORCE(can_use_frames_timestamps_ || !enable_frame_num_,
                 "frame numbers can be enabled only when "
                 "`file_list`, or `filenames` with `labels` argument are passed");
    DALI_ENFORCE(can_use_frames_timestamps_ || !enable_timestamps_,
                 "timestamps can be enabled only when "
                 "`file_list`, or `filenames` with `labels` argument are passed");

    DALI_ENFORCE(!(has_labels_arg && filenames_.empty()),
                 "The argument ``labels`` is valid only when file paths "
                 "are provided with ``filenames`` argument.");

    if (!filenames_.empty() && has_labels_arg) {
      DALI_ENFORCE(filenames_.size() == labels_.size() || labels_.empty(), make_string("Provided ",
                  labels_.size(), " labels for ", filenames_.size(), " files."));
    }

    output_labels_ = has_labels_arg || !file_list_.empty() || !file_root_.empty();

    // TODO(spanev): Factor out the constructor body to make VideoReader compatible with lazy_init.
    loader_ = InitLoader<VideoLoader>(spec, filenames_);

    label_shape_ = uniform_list_shape(max_batch_size_, {1});

    if (can_use_frames_timestamps_) {
      if (enable_frame_num_) frame_num_shape_ = label_shape_;
      if (enable_timestamps_) timestamp_shape_ = uniform_list_shape(max_batch_size_, {count_});
    }
  }

  inline ~VideoReader() {
    // stop prefetching so we are not scheduling any more work from here on so we are safe to remove
    // the underlying memory
    DataReader<GPUBackend, SequenceWrapper>::StopPrefetchThread();
    // when this destructor is called some kernels can still be scheduled to work on the memory
    // that is present in the prefetched_batch_tensors_
    // prefetched_batch_queue_ keeps the relevant cuda events recorded that are associated with
    // this memory. Calling wait makes sure that no more work is pending and we can free the GPU
    // memory
    for (auto &batch : prefetched_batch_queue_) {
      for (auto &sample : batch) {
        sample->wait();
      }
    }
  }

 protected:
  virtual void SetOutputShapeType(TensorList<GPUBackend> &output, DeviceWorkspace &ws) {
    auto &curr_batch = prefetched_batch_tensors_[curr_batch_consumer_];
    output.Resize(curr_batch.shape(), curr_batch.type());
  }

  void PrepareAdditionalOutputs(DeviceWorkspace &ws) {
    int output_index = 1;
    if (output_labels_) {
      label_output_ = &ws.Output<GPUBackend>(output_index++);
      label_output_->set_type(TypeTable::GetTypeInfoFromStatic<int>());
      label_output_->Resize(label_shape_);
      if (can_use_frames_timestamps_) {
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
  }

  virtual void ProcessVideo(TensorList<GPUBackend> &video_output,
                            TensorList<GPUBackend> &video_batch, DeviceWorkspace &ws) {
    video_output.Copy(video_batch, ws.stream());
    video_output.SetLayout("FHWC");
  }

  void ProcessAdditionalOutputs(int data_idx, SequenceWrapper &prefetched_video,
                                cudaStream_t stream) {
    if (output_labels_) {
      auto *label = label_output_->mutable_tensor<int>(data_idx);
      CUDA_CALL(
          cudaMemcpyAsync(label, &prefetched_video.label, sizeof(int), cudaMemcpyDefault, stream));
      if (can_use_frames_timestamps_) {
        if (enable_frame_num_) {
          auto *frame_num = frame_num_output_->mutable_tensor<int>(data_idx);
          CUDA_CALL(cudaMemcpyAsync(frame_num, &prefetched_video.first_frame_idx, sizeof(int),
                                    cudaMemcpyDefault, stream));
        }
        if (enable_timestamps_) {
          auto *timestamp = timestamp_output_->mutable_tensor<double>(data_idx);
          if (prefetched_video.timestamps.size() < static_cast<size_t>(count_)) {
            // pad timestamps for shorter sequences
            auto old_size = prefetched_video.timestamps.size();
            prefetched_video.timestamps.resize(count_);
            std::fill(prefetched_video.timestamps.begin() + old_size,
                      prefetched_video.timestamps.end(), -1);
          }
          timestamp_output_->type().Copy<GPUBackend, CPUBackend>(
              timestamp, prefetched_video.timestamps.data(), prefetched_video.timestamps.size(),
              stream);
        }
      }
    }
  }

  void RunImpl(DeviceWorkspace &ws) override {
    auto &video_output = ws.Output<GPUBackend>(0);
    auto &curent_batch = prefetched_batch_tensors_[curr_batch_consumer_];

    SetOutputShapeType(video_output, ws);
    PrepareAdditionalOutputs(ws);

    ProcessVideo(video_output, curent_batch, ws);

    for (size_t data_idx = 0; data_idx < curent_batch.ntensor(); ++data_idx) {
      auto &prefetched_video = GetSample(data_idx);
      ProcessAdditionalOutputs(data_idx, prefetched_video, ws.stream());
    }
  }

  // override prefetching here
  void Prefetch() override;

  static constexpr int sequence_dim = 4;
  std::vector<std::string> filenames_;
  std::vector<int> labels_;
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
  bool can_use_frames_timestamps_ = false;
  bool output_labels_ = false;

  USE_READER_OPERATOR_MEMBERS(GPUBackend, SequenceWrapper);
};

}  // namespace dali

#endif  // DALI_OPERATORS_READER_VIDEO_READER_OP_H_
