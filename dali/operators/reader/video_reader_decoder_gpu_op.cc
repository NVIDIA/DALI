// Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    : DataReader<GPUBackend, VideoSampleGpu, VideoSampleGpu, true>(spec),
      has_labels_(spec.HasArgument("labels")),
      has_frame_idx_(spec.GetArgument<bool>("enable_frame_num")) {
      loader_ = InitLoader<VideoLoaderDecoderGpu>(spec);
      this->SetInitialSnapshot();
}

void VideoReaderDecoderGpu::Prefetch() {
  DataReader<GPUBackend, VideoSampleGpu, VideoSampleGpu, true>::Prefetch();

  auto &current_batch = prefetched_batch_queue_[curr_batch_producer_];
  for (auto &sample : current_batch) {
    sample->Decode();
  }
}

bool VideoReaderDecoderGpu::SetupImpl(
  std::vector<OutputDesc> &output_desc, const Workspace &ws) {
  DataReader<GPUBackend, VideoSampleGpu, VideoSampleGpu, true>::SetupImpl(output_desc, ws);

  output_desc.resize(1 + has_labels_ + has_frame_idx_);
  int batch_size = GetCurrBatchSize();

  TensorListShape<4> video_shape(batch_size);

  for (int sample_id = 0; sample_id < batch_size; ++sample_id) {
    auto &sample = GetSample(sample_id);
    video_shape.set_tensor_shape(sample_id, sample.data_.shape());
  }

  output_desc[0] = { video_shape, DALI_UINT8 };

  int out_index = 1;
  if (has_labels_) {
    output_desc[out_index] = {
      uniform_list_shape<1>(batch_size, {1}),
      DALI_INT32
    };
    out_index++;
  }
  if (has_frame_idx_) {
    output_desc[out_index] = {
      uniform_list_shape<1>(batch_size, {1}),
      DALI_INT32
    };
    out_index++;
  }

  return true;
}

void VideoReaderDecoderGpu::RunImpl(Workspace &ws) {
  auto &video_output = ws.Output<GPUBackend>(0);
  int batch_size = GetCurrBatchSize();

  video_output.SetLayout("FHWC");

  // TODO(awolant): Would struct of arrays work better?
  for (int sample_id = 0; sample_id < batch_size; ++sample_id) {
    auto &sample = GetSample(sample_id);

    MemCopy(
      video_output.raw_mutable_tensor(sample_id),
      sample.data_.raw_data(),
      sample.data_.size(),
      ws.stream());
    video_output.SetSourceInfo(sample_id, sample.data_.GetSourceInfo());
  }

  int out_index = 1;
  if (has_labels_) {
    auto &labels_output = ws.Output<GPUBackend>(out_index);
    SmallVector<int, 32> labels_cpu;

    for (int sample_id = 0; sample_id < batch_size; ++sample_id) {
      auto &sample = GetSample(sample_id);
      labels_cpu[sample_id] = sample.label_;
    }

    MemCopy(
      labels_output.AsTensor().raw_mutable_data(),
      labels_cpu.data(),
      batch_size * sizeof(DALI_INT32),
      ws.stream());
    out_index++;
  }
  if (has_frame_idx_) {
    auto &frame_idx_output = ws.Output<GPUBackend>(out_index);
    SmallVector<int, 32> frame_idx_output_cpu;

    for (int sample_id = 0; sample_id < batch_size; ++sample_id) {
      auto &sample = GetSample(sample_id);
      frame_idx_output_cpu[sample_id] = sample.span_ ? sample.span_->start_ : -1;
    }

    MemCopy(
      frame_idx_output.AsTensor().raw_mutable_data(),
      frame_idx_output_cpu.data(),
      batch_size * sizeof(DALI_INT32),
      ws.stream());
    out_index++;
  }
}

DALI_REGISTER_OPERATOR(experimental__readers__Video, VideoReaderDecoderGpu, GPU);

}  // namespace dali
