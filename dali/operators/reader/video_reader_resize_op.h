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

#include "dali/core/common.h"
#include "dali/core/error_handling.h"
#include "dali/kernels/imgproc/resample.h"
#include "dali/kernels/imgproc/resample/params.h"
#include "dali/kernels/imgproc/resample_cpu.h"
#include "dali/kernels/kernel.h"
#include "dali/kernels/kernel_manager.h"
#include "dali/kernels/scratch.h"
#include "dali/operators/image/resize/resize.h"
#include "dali/operators/image/resize/resize_base.h"
#include "dali/pipeline/data/views.h"
#include "dali/pipeline/operator/op_spec.h"

namespace dali {

class VideoReaderResize : public VideoReader, protected ResizeAttr, protected ResizeBase {
 public:
  explicit VideoReaderResize(const OpSpec &spec)
      : VideoReader(spec),
        ResizeAttr(spec),
        ResizeBase(spec) {

    ResizeAttr::SetBatchSize(count_);
    InitializeGPU(count_, spec_.GetArgument<int>("minibatch_size"));
    resample_params_.resize(count_);
  }

  inline ~VideoReaderResize() override = default;

 protected:
  void SetupSharedSampleParams(DeviceWorkspace &ws) override {
  }

  void SetOutputShape(TensorList<GPUBackend> &output, DeviceWorkspace &ws) override {
    TensorListShape<> output_shape(batch_size_, sequence_dim);
    for (int data_idx = 0; data_idx < batch_size_; ++data_idx) {
      auto transform_meta = GetTransformMeta(
        spec_, GetSample(data_idx).sequence.shape(), &ws, data_idx, ResizeInfoNeeded());
      TensorShape<> sequence_shape{count_, transform_meta.rsz_h, transform_meta.rsz_w, channels_};
      output_shape.set_tensor_shape(data_idx, sequence_shape);
    }
    output.Resize(output_shape);
  }

  void ProcessSingleVideo(
    int data_idx, 
    TensorList<GPUBackend> &video_output, 
    void *single_video_output,
    SequenceWrapper &prefetched_video, 
    DeviceWorkspace &ws) override {
    void *current_sequence = prefetched_video.sequence.raw_mutable_data();

    TensorShape<3> input_tensor_shape(
      prefetched_video.height, prefetched_video.width, prefetched_video.channels);

    resample_params_.clear();

    auto transform_meta = GetTransformMeta(
      spec_, input_tensor_shape, &ws, data_idx, ResizeInfoNeeded());

    for (int i = 0; i < prefetched_video.count; ++i) {
      resample_params_.push_back(GetResamplingParams(transform_meta));
    }

    TensorList<GPUBackend> input;
    TensorList<GPUBackend> output;

    int64_t after_resize_frame_size = transform_meta.rsz_h * transform_meta.rsz_w * channels_;

    input.ShareData(current_sequence, sizeof(uint8) * prefetched_video.count *
                                          prefetched_video.height * prefetched_video.width *
                                          channels_);
    output.ShareData(single_video_output,
                     sizeof(uint8) * prefetched_video.count * after_resize_frame_size);

    TensorListShape<> input_shape;

    input_shape.resize(prefetched_video.count, 3);
    out_shape_.resize(prefetched_video.count, 3);

    
    TensorShape<3> output_tensor_shape(
      transform_meta.rsz_w, transform_meta.rsz_h, prefetched_video.channels);

    for (int i = 0; i < prefetched_video.count; ++i) {
      input_shape.set_tensor_shape(i, input_tensor_shape);
      out_shape_.set_tensor_shape(i, output_tensor_shape);
    }

    input.set_type(TypeInfo::Create<uint8>());
    output.set_type(TypeInfo::Create<uint8>());

    input.Resize(input_shape);
    output.Resize(out_shape_);

    RunGPU(output, input, ws.stream());
  }

 private:
  kernels::ResamplingParams2D GetResamplingParams(const TransformMeta &meta) const {
    kernels::ResamplingParams2D params;
    params[0].output_size = meta.rsz_h;
    params[1].output_size = meta.rsz_w;
    params[0].min_filter = params[1].min_filter = min_filter_;
    params[0].mag_filter = params[1].mag_filter = mag_filter_;
    return params;
  }
};

}  // namespace dali

#endif  // DALI_OPERATORS_READER_VIDEO_READER_RESIZE_OP_H_
