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

#ifndef DALI_PIPELINE_OPERATORS_READER_VIDEO_READER_OP_H_
#define DALI_PIPELINE_OPERATORS_READER_VIDEO_READER_OP_H_

#include "dali/pipeline/operators/reader/reader_op.h"
#include "dali/pipeline/operators/reader/loader/video_loader.h"

namespace dali {

class VideoReader : public DataReader<GPUBackend, SequenceWrapper> {
 public:
  explicit VideoReader(const OpSpec &spec)
  : DataReader<GPUBackend, SequenceWrapper>(spec),
    filenames_(spec.GetRepeatedArgument<std::string>("filenames")),
    count_(spec.GetArgument<int>("count")),
    height_(spec.GetArgument<int>("height")),
    width_(spec.GetArgument<int>("width")),
    channels_(spec.GetArgument<int>("channels")) {
    	loader_.reset(new VideoLoader(spec, filenames_));
      try {
        dynamic_cast<VideoLoader*>(loader_.get())->init();
      } catch (std::runtime_error& e) {
        DALI_FAIL(std::string(e.what()));
      }

      std::vector<Index> t_shape({batch_size_, count_, height_, width_, channels_});

      for (int i = 0; i < batch_size_; ++i) {
        tl_shape_.push_back(t_shape);
      }
  }



  virtual inline ~VideoReader() = default;

 protected:
  void RunImpl(DeviceWorkspace *ws, const int idx) override {
    const int data_idx = samples_processed_.load();
    auto* input_sequence = prefetched_batch_[data_idx];
    auto* tl_sequence_output = ws->Output<GPUBackend>(0);
    auto* sequence_output = tl_sequence_output->mutable_tensor<float>(data_idx);
    // TODO(spanev) copy the ouput into sequence_output
    CUDA_CALL(cudaMemcpy(sequence_output,
                         input_sequence->sequence.raw_data(),
                         input_sequence->sequence.size(),
                         cudaMemcpyDeviceToDevice));
  }

  void SetupSharedSampleParams(DeviceWorkspace *ws) {
    auto* tl_sequence_output = ws->Output<GPUBackend>(0);
    tl_sequence_output->set_type(TypeInfo::Create<float>());
    tl_sequence_output->Resize(tl_shape_);
  }


 private:
  std::vector<std::string> filenames_;
  int count_;
  int height_;
  int width_;
  int channels_;

  std::vector<std::vector<Index>> tl_shape_;

  USE_READER_OPERATOR_MEMBERS(GPUBackend, SequenceWrapper);
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_READER_VIDEO_READER_OP_H_