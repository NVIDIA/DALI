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
    tl_shape_(batch_size_, sequence_dim),
    dtype_(spec.GetArgument<DALIDataType>("dtype")) {
    DALIImageType image_type(spec.GetArgument<DALIImageType>("image_type"));

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

    // TODO(spanev): support rescale
    // TODO(spanev): Factor out the constructor body to make VideoReader compatible with lazy_init.
      try {
        loader_ = InitLoader<VideoLoader>(spec, filenames_);
      } catch (std::exception &e) {
        DALI_WARN(std::string(e.what()));
        throw;
      }

      if (enable_label_output_) {
        label_shape_ = uniform_list_shape(batch_size_, {1});

        if (enable_frame_num_)
          frame_num_shape_ = label_shape_;
        if (enable_timestamps_)
          timestamp_shape_ = uniform_list_shape(batch_size_, {count_});
      }
  }

  inline ~VideoReader() override = default;

 protected:
  void SetupSharedSampleParams(DeviceWorkspace &ws) override {
  }

  void RunImpl(DeviceWorkspace &ws) override {
    auto& tl_sequence_output = ws.Output<GPUBackend>(0);
    TensorList<GPUBackend> *label_output = NULL;
    TensorList<GPUBackend> *frame_num_output = NULL;
    TensorList<GPUBackend> *timestamp_output = NULL;

    if (dtype_ == DALI_FLOAT) {
      tl_sequence_output.set_type(TypeInfo::Create<float>());
    } else {  // dtype_ == DALI_UINT8
      tl_sequence_output.set_type(TypeInfo::Create<uint8>());
    }

    for (int data_idx = 0; data_idx < batch_size_; ++data_idx) {
      auto sequence_shape = GetSample(data_idx).sequence.shape();
      tl_shape_.set_tensor_shape(data_idx, sequence_shape);
    }

    tl_sequence_output.Resize(tl_shape_);
    tl_sequence_output.SetLayout("FHWC");

    if (enable_label_output_) {
      int output_index = 1;
      label_output = &ws.Output<GPUBackend>(output_index++);
      label_output->set_type(TypeInfo::Create<int>());
      label_output->Resize(label_shape_);
      if (enable_frame_num_) {
        frame_num_output = &ws.Output<GPUBackend>(output_index++);
        frame_num_output->set_type(TypeInfo::Create<int>());
        frame_num_output->Resize(frame_num_shape_);
      }

      if (enable_timestamps_) {
        timestamp_output = &ws.Output<GPUBackend>(output_index++);
        timestamp_output->set_type(TypeInfo::Create<double>());
        timestamp_output->Resize(timestamp_shape_);
      }
    }

    for (int data_idx = 0; data_idx < batch_size_; ++data_idx) {
      auto* sequence_output = tl_sequence_output.raw_mutable_tensor(data_idx);

      auto& prefetched_sequence = GetSample(data_idx);
      tl_sequence_output.type().Copy<GPUBackend, GPUBackend>(sequence_output,
                                  prefetched_sequence.sequence.raw_data(),
                                  prefetched_sequence.sequence.size(),
                                  ws.stream());

        if (enable_label_output_) {
          auto *label = label_output->mutable_tensor<int>(data_idx);
          CUDA_CALL(cudaMemcpyAsync(label, &prefetched_sequence.label, sizeof(int),
                                    cudaMemcpyDefault, ws.stream()));
          if (enable_frame_num_) {
            auto *frame_num = frame_num_output->mutable_tensor<int>(data_idx);
            CUDA_CALL(cudaMemcpyAsync(frame_num, &prefetched_sequence.first_frame_idx,
                                      sizeof(int), cudaMemcpyDefault, ws.stream()));
          }
          if (enable_timestamps_) {
            auto *timestamp = timestamp_output->mutable_tensor<double>(data_idx);
            timestamp_output->type().Copy<GPUBackend, CPUBackend>(timestamp,
                                                   prefetched_sequence.timestamps.data(),
                                                   prefetched_sequence.timestamps.size(),
                                                   ws.stream());
          }
        }
    }
  }


 private:
  static constexpr int sequence_dim = 4;
  std::vector<std::string> filenames_;
  std::string file_root_;
  std::string file_list_;
  bool enable_frame_num_;
  bool enable_timestamps_;
  int count_;
  int channels_;

  TensorListShape<> tl_shape_;
  TensorListShape<> label_shape_;
  TensorListShape<> timestamp_shape_;
  TensorListShape<> frame_num_shape_;

  DALIDataType dtype_;
  bool enable_label_output_;

  USE_READER_OPERATOR_MEMBERS(GPUBackend, SequenceWrapper);
};

}  // namespace dali

#endif  // DALI_OPERATORS_READER_VIDEO_READER_OP_H_
