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
#include "dali/operators/image/resize/resize_attr.h"
#include "dali/operators/image/resize/resampling_attr.h"
#include "dali/operators/image/resize/resize_base.h"

namespace dali {

class VideoReaderResize : public VideoReader,
                          protected ResizeBase<GPUBackend> {
 public:
  explicit VideoReaderResize(const OpSpec &spec)
      : VideoReader(spec),
        ResizeBase(spec) {
    ResizeBase::InitializeGPU(spec_.GetArgument<int>("minibatch_size"),
                              spec_.GetArgument<int64_t>("temp_buffer_hint"));
  }

  inline ~VideoReaderResize() override = default;

 protected:
  void SetOutputShapeType(TensorList<GPUBackend> &output, DeviceWorkspace &ws) override {
    input_shape_ = prefetched_batch_tensors_[curr_batch_consumer_].shape();

    resize_attr_.PrepareResizeParams(spec_, ws, input_shape_, "FHWC");
    resampling_attr_.PrepareFilterParams(spec_, ws, max_batch_size_);
    resample_params_.resize(resize_attr_.params_.size());
    resampling_attr_.GetResamplingParams(make_span(resample_params_),
                                         make_cspan(resize_attr_.params_));
    resize_attr_.GetResizedShape(output_shape_, input_shape_);
    output.Resize(output_shape_, prefetched_batch_tensors_[curr_batch_consumer_].type());
  }

  void ShareSingleOutputAsTensorList(int data_idx, TensorList<GPUBackend> &batch_output,
                                     TensorList<GPUBackend> &single_output) {
    auto *raw_output = batch_output.raw_mutable_tensor(data_idx);
    TensorShape<> seq_shape = batch_output.shape()[data_idx];
    TensorListShape<> shape = uniform_list_shape(1, seq_shape);
    const auto &type = batch_output.type();
    single_output.ShareData(raw_output, shape.num_elements() * type.size(), shape, type);
  }

  void ProcessVideo(
    TensorList<GPUBackend> &video_output,
    TensorList<GPUBackend> &video_batch,
    DeviceWorkspace &ws) override {
    TensorListShape<> input_shape(1, sequence_dim);
    for (size_t data_idx = 0; data_idx < video_batch.ntensor(); ++data_idx) {
      TensorList<GPUBackend> input;
      input_shape.set_tensor_shape(0, video_batch.tensor_shape(data_idx));
      input.ShareData(video_batch.raw_mutable_tensor(data_idx),
                      volume(video_batch.tensor_shape(data_idx)) * video_batch.type().size(),
                      input_shape, video_batch.type());

      TensorList<GPUBackend> output;
      ShareSingleOutputAsTensorList(data_idx, video_output, output);
      TensorListShape<> output_shape;
      SetupResize(output_shape, output.type().id(), input.shape(), input.type().id(),
                  make_cspan(&resample_params_[data_idx], 1), resize_attr_.first_spatial_dim_);
      assert(output_shape == output.shape());
      RunResize(ws, output, input);
    }
    video_output.SetLayout("FHWC");
  }

  ResizeAttr resize_attr_;
  ResamplingFilterAttr resampling_attr_;
 private:
  std::vector<kernels::ResamplingParams2D> resample_params_;
  TensorListShape<> input_shape_, output_shape_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_READER_VIDEO_READER_RESIZE_OP_H_
