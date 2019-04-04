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

#include <string>
#include <vector>

#include "dali/pipeline/operators/reader/reader_op.h"
#include "dali/pipeline/operators/reader/loader/video_loader.h"

namespace dali {

class VideoReader : public DataReader<GPUBackend, SequenceWrapper> {
 public:
  explicit VideoReader(const OpSpec &spec)
  : DataReader<GPUBackend, SequenceWrapper>(spec),
    filenames_(spec.GetRepeatedArgument<std::string>("filenames")),
    count_(spec.GetArgument<int>("sequence_length")),
    channels_(spec.GetArgument<int>("channels")),
    output_scale_(spec.GetArgument<float>("scale")),
    dtype_(spec.GetArgument<DALIDataType>("dtype")) {
    DALIImageType image_type(spec.GetArgument<DALIImageType>("image_type"));

    DALI_ENFORCE(image_type == DALI_RGB || image_type == DALI_YCbCr,
                 "Image type must be RGB or YCbCr.");

    DALI_ENFORCE(dtype_ == DALI_FLOAT || dtype_ == DALI_UINT8,
                 "Data type must be FLOAT or UINT8.");


    // TODO(spanev): support rescale
      try {
        loader_ = InitLoader<VideoLoader>(spec, filenames_);
        auto w_h = dynamic_cast<VideoLoader*>(loader_.get())->load_width_height(filenames_[0]);
        width_ = static_cast<int>(w_h.first * output_scale_);
        height_ = static_cast<int>(w_h.second * output_scale_);
      } catch (std::runtime_error& e) {
        DALI_FAIL(std::string(e.what()));
      }

      std::vector<Index> t_shape({count_, height_, width_, channels_});

      for (int i = 0; i < batch_size_; ++i) {
        tl_shape_.push_back(t_shape);
      }
  }

  inline ~VideoReader() override = default;

 protected:
  void SetupSharedSampleParams(DeviceWorkspace *ws) override {
  }

  void RunImpl(DeviceWorkspace *ws, const int idx) override {
    auto& tl_sequence_output = ws->Output<GPUBackend>(idx);
    if (dtype_ == DALI_FLOAT) {
      tl_sequence_output.set_type(TypeInfo::Create<float>());
    } else {  // dtype_ == DALI_UINT8
      tl_sequence_output.set_type(TypeInfo::Create<uint8>());
    }

    tl_sequence_output.Resize(tl_shape_);
    tl_sequence_output.SetLayout(DALI_NFHWC);

    for (int data_idx = 0; data_idx < batch_size_; ++data_idx) {
      auto* sequence_output = tl_sequence_output.raw_mutable_tensor(data_idx);

      auto& prefetched_sequence = GetSample(data_idx);
      tl_sequence_output.type().Copy<GPUBackend, GPUBackend>(sequence_output,
                                  prefetched_sequence.sequence.raw_data(),
                                  prefetched_sequence.sequence.size(),
                                  ws->stream());
    }
  }


 private:
  std::vector<std::string> filenames_;
  int count_;
  int height_;
  int width_;
  int channels_;

  float output_scale_;

  std::vector<std::vector<Index>> tl_shape_;

  DALIDataType dtype_;

  USE_READER_OPERATOR_MEMBERS(GPUBackend, SequenceWrapper);
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_READER_VIDEO_READER_OP_H_
