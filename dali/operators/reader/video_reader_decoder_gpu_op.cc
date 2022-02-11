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
#include "dali/operators/reader/video_reader_decoder_gpu_op.h"

#include <string>
#include <vector>

namespace dali {

VideoReaderDecoderGpu::VideoReaderDecoderGpu(const OpSpec &spec)
    : DataReader<GPUBackend, VideoSampleGpu>(spec),
      has_labels_(spec.HasArgument("labels")) {
      loader_ = InitLoader<VideoLoaderDecoderGpu>(spec);
}

void VideoReaderDecoderGpu::PrepareOutput(TensorList<GPUBackend> &video_output) {
}

void VideoReaderDecoderGpu::RunImpl(DeviceWorkspace &ws) {
  auto &video_output = ws.Output<GPUBackend>(0);
  auto &current_batch = prefetched_batch_queue_[curr_batch_consumer_];
  int batch_size = current_batch.size();

  TensorListShape<4> output_shape;
  output_shape.resize(batch_size);

  for (int sample_id = 0; sample_id < batch_size; ++sample_id) {
    auto &sample = current_batch[sample_id];
    output_shape.set_tensor_shape(sample_id, sample->data_.shape());
  }

  video_output.Resize(output_shape, current_batch[0]->data_.type());

  for (int sample_id = 0; sample_id < batch_size; ++sample_id) {
    auto &sample = current_batch[sample_id];
    MemCopy(
      video_output.raw_mutable_tensor(sample_id),
      sample->data_.raw_mutable_data(),
      sample->data_.size(),
      ws.stream());
  }

  if (!has_labels_) {
    return;
  }

  auto &labels_output = ws.Output<GPUBackend>(1);
  TensorListShape<1> labels_shape;
  labels_shape.resize(batch_size);

  TensorShape<1> one_label_shape = { 1 };

  for (int sample_id = 0; sample_id < batch_size; ++sample_id) {
    labels_shape.set_tensor_shape(sample_id, one_label_shape);
  }

  labels_output.Resize(labels_shape, DALIDataType::DALI_INT32);

  for (int sample_id = 0; sample_id < batch_size; ++sample_id) {
    auto &sample = current_batch[sample_id];
    MemCopy(
      labels_output.raw_mutable_tensor(sample_id),
      &sample->label_,
      sizeof(DALIDataType::DALI_INT32),
      ws.stream());
  }

}

DALI_REGISTER_OPERATOR(experimental__readers__Video, VideoReaderDecoderGpu, GPU);

}  // namespace dali
