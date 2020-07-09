// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_OPERATORS_READER_VIDEO_READER_RESIZE_OP_H_
#define DALI_OPERATORS_READER_VIDEO_READER_RESIZE_OP_H_

#include <string>
#include <vector>

#include "dali/operators/reader/loader/video_loader.h"
#include "dali/operators/reader/reader_op.h"
#include "dali/operators/reader/video_reader_op.h"
#include "dali/operators/image/resize/resize.h"

namespace dali {

class VideoReaderResize : public VideoReader, protected ResizeAttr, protected ResizeBase {
 public:
  explicit VideoReaderResize(const OpSpec &spec)
      : VideoReader(spec),
        ResizeAttr(spec),
        ResizeBase(spec),
        per_video_transform_metas_(batch_size_) {
    DALI_ENFORCE(dtype_ == DALI_UINT8, "Data type must be UINT8.");
    ResizeAttr::SetBatchSize(count_);
    ResizeBase::InitializeGPU(count_, spec_.GetArgument<int>("minibatch_size"));
    resample_params_.resize(count_);
  }

  inline ~VideoReaderResize() override = default;

 protected:
  void SetupSharedSampleParams(DeviceWorkspace &ws) override {}

  void SetOutputShape(TensorList<GPUBackend> &output, DeviceWorkspace &ws) override {
    TensorListShape<> output_shape(batch_size_, sequence_dim);
    for (int data_idx = 0; data_idx < batch_size_; ++data_idx) {
      per_video_transform_metas_[data_idx] = GetTransformMeta(
          spec_, GetSample(data_idx).frame_shape(), &ws, data_idx, ResizeInfoNeeded());
      TensorShape<> sequence_shape{count_, per_video_transform_metas_[data_idx].rsz_h,
                                   per_video_transform_metas_[data_idx].rsz_w, channels_};
      output_shape.set_tensor_shape(data_idx, sequence_shape);
    }
    output.Resize(output_shape);
  }

  void ShareSingleOutputAsTensorList(int data_idx, TensorList<GPUBackend> &batch_output,
                         TensorList<GPUBackend> &single_output) {
    auto *raw_output = batch_output.raw_mutable_tensor(data_idx);
    int64_t after_resize_frame_size = per_video_transform_metas_[data_idx].rsz_h *
                                      per_video_transform_metas_[data_idx].rsz_w * channels_;
    single_output.ShareData(raw_output, sizeof(uint8) * count_ * after_resize_frame_size);
  }

  void ProcessSingleVideo(
      int data_idx,
      TensorList<GPUBackend> &video_output,
      SequenceWrapper &prefetched_video,
      DeviceWorkspace &ws) override {
    std::fill_n(
      resample_params_.begin(),
      count_,
      detail::GetResamplingParams(
        per_video_transform_metas_[data_idx],
        min_filter_,
        mag_filter_));

    TensorList<GPUBackend> input;
    prefetched_video.share_frames(input);

    TensorList<GPUBackend> output;
    ShareSingleOutputAsTensorList(data_idx, video_output, output);

    ResizeBase::RunGPU(output, input, ws.stream());
  }

 private:
  vector<TransformMeta> per_video_transform_metas_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_READER_VIDEO_READER_RESIZE_OP_H_
