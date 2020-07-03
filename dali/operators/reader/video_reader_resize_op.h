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

#include "dali/operators/reader/reader_op.h"
#include "dali/operators/reader/loader/video_loader.h"
#include "dali/operators/reader/video_reader_op.h"

#include "dali/core/common.h"
#include "dali/core/error_handling.h"
#include "dali/kernels/scratch.h"
#include "dali/kernels/kernel.h"
#include "dali/kernels/imgproc/resample/params.h"
#include "dali/kernels/kernel_manager.h"
#include "dali/pipeline/operator/op_spec.h"
#include "dali/kernels/imgproc/resample.h"
#include "dali/kernels/imgproc/resample_cpu.h"
#include "dali/operators/image/resize/resize_base.h"
#include "dali/pipeline/data/views.h"

namespace dali {

class VideoReaderResize : public VideoReader {
 public:
  explicit VideoReaderResize(const OpSpec &spec)
  : VideoReader(spec),
    resize_(spec.GetArgument<bool>("resize")),
    resize_x_(spec.GetArgument<float>("resize_x")),
    resize_y_(spec.GetArgument<float>("resize_y")) {

    resampling_type_ = detail::interp2resample(spec_.GetArgument<DALIInterpType>("interp_type"));
  }

  inline ~VideoReaderResize() override = default;

 protected:
  void SetupSharedSampleParams(DeviceWorkspace &ws) override {
  }

  void RunImpl(DeviceWorkspace &ws) override {
    auto& tl_sequence_output = ws.Output<GPUBackend>(0);
    TensorList<GPUBackend> *label_output = NULL;
    TensorList<GPUBackend> *frame_num_output = NULL;
    TensorList<GPUBackend> *timestamp_output = NULL;

    // Setting output type
    if (dtype_ == DALI_FLOAT) {
      tl_sequence_output.set_type(TypeInfo::Create<float>());
    } else {  // dtype_ == DALI_UINT8
      tl_sequence_output.set_type(TypeInfo::Create<uint8>());
    }

    TensorShape<> sequence_shape {count_, resize_y_, resize_x_, channels_};
    for (int data_idx = 0; data_idx < batch_size_; ++data_idx) {
    tl_shape_.set_tensor_shape(data_idx, sequence_shape);
    }

    // Setting output shape and layout
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

        // Process one sample (video) as a batch of images in Resize operator here
        void *current_sequence = prefetched_sequence.sequence.raw_mutable_data();

        TensorList<GPUBackend> input;
        TensorList<GPUBackend> output;

        int64_t after_resize_frame_size = resize_x_*resize_y_*prefetched_sequence.channels;

        input.ShareData(
            current_sequence, 
            sizeof(uint8)*prefetched_sequence.count*prefetched_sequence.height*prefetched_sequence.width*prefetched_sequence.channels);
        output.ShareData(
            sequence_output,  
            sizeof(uint8)*prefetched_sequence.count*after_resize_frame_size);

        TensorListShape<> input_shape;
        TensorListShape<> output_shape;

        input_shape.resize(prefetched_sequence.count, 3);
        output_shape.resize(prefetched_sequence.count, 3);

        TensorShape<3> input_tensor_shape(
            prefetched_sequence.height, prefetched_sequence.width, prefetched_sequence.channels);
        TensorShape<3> output_tensor_shape(
            resize_y_, resize_x_, prefetched_sequence.channels);

        for (int i = 0; i < prefetched_sequence.count; ++i) {
            input_shape.set_tensor_shape(i, input_tensor_shape);
            output_shape.set_tensor_shape(i, output_tensor_shape);
        }

        input.set_type(TypeInfo::Create<uint8>());
        output.set_type(TypeInfo::Create<uint8>());
        
        input.Resize(input_shape);
        output.Resize(output_shape);

        auto in_view = view<const uint8_t, 3>(input);
        auto out_view = view<uint8_t, 3>(output);

        using Kernel = kernels::ResampleGPU<uint8_t, uint8_t>;
        kernels::KernelManager kmgr;
        kmgr.Resize<Kernel>(1, 1);
        kmgr.SetMemoryHint(kernels::AllocType::GPU, 0);
        kmgr.ReserveMaxScratchpad(0);

        kernels::KernelContext context;
        context.gpu.stream = ws.stream();

        std::vector<kernels::ResamplingParams2D> resample_params;
        // resample_params.resize(prefetched_sequence.count);
        for (int i  = 0; i < prefetched_sequence.count; ++i) {
            kernels::ResamplingParams2D params;
            params[0].output_size = resize_y_;
            params[1].output_size = resize_x_;
            params[0].min_filter = params[1].min_filter = resampling_type_;
            params[0].mag_filter = params[1].mag_filter = resampling_type_;
            
            resample_params.push_back(params);
        }

        auto &req = kmgr.Setup<Kernel>(
            0, 
            context,
            in_view, 
            make_span(resample_params.data(), prefetched_sequence.count));

        kmgr.Run<Kernel>(
            0, 
            0, 
            context,
            out_view, 
            in_view, 
            make_span(resample_params.data(), prefetched_sequence.count));
        

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
  bool resize_;
  float resize_x_;
  float resize_y_;
  kernels::ResamplingFilterType resampling_type_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_READER_VIDEO_READER_RESIZE_OP_H_
