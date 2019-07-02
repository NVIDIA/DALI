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
    file_root_(spec.GetArgument<std::string>("file_root")),
    file_list_(spec.GetArgument<std::string>("file_list")),
    count_(spec.GetArgument<int>("sequence_length")),
    channels_(spec.GetArgument<int>("channels")),
    output_scale_(spec.GetArgument<float>("scale")),
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


    // TODO(spanev): support rescale
    // TODO(spanev): Factor out the constructor body to make VideoReader compatible with lazy_init.
      try {
        loader_ = InitLoader<VideoLoader>(spec, filenames_);
        auto w_h = dynamic_cast<VideoLoader*>(loader_.get())->load_width_height();
        width_ = static_cast<int>(w_h.first * output_scale_);
        height_ = static_cast<int>(w_h.second * output_scale_);
      } catch (std::exception &e) {
        DALI_FAIL(std::string(e.what()));
      }

      tl_shape_ = kernels::uniform_list_shape(batch_size_, {count_, height_, width_, channels_});

      enable_label_output_ = !file_root_.empty() || !file_list_.empty();

      if (enable_label_output_) {
        label_shape_ = kernels::uniform_list_shape(batch_size_, {1});
      }
  }

  inline ~VideoReader() override = default;

 protected:
  void SetupSharedSampleParams(DeviceWorkspace *ws) override {
  }

  void RunImpl(DeviceWorkspace *ws, const int idx) override {
    auto& tl_sequence_output = ws->Output<GPUBackend>(idx);
    TensorList<GPUBackend> *label_output = NULL;

    if (dtype_ == DALI_FLOAT) {
      tl_sequence_output.set_type(TypeInfo::Create<float>());
    } else {  // dtype_ == DALI_UINT8
      tl_sequence_output.set_type(TypeInfo::Create<uint8>());
    }

    tl_sequence_output.Resize(tl_shape_);
    tl_sequence_output.SetLayout(DALI_NFHWC);

    if (enable_label_output_) {
      label_output = &ws->Output<GPUBackend>(idx + 1);
      label_output->set_type(TypeInfo::Create<int>());
      label_output->Resize(label_shape_);
    }

    for (int data_idx = 0; data_idx < batch_size_; ++data_idx) {
      auto* sequence_output = tl_sequence_output.raw_mutable_tensor(data_idx);

      auto& prefetched_sequence = GetSample(data_idx);
      tl_sequence_output.type().Copy<GPUBackend, GPUBackend>(sequence_output,
                                  prefetched_sequence.sequence.raw_data(),
                                  prefetched_sequence.sequence.size(),
                                  ws->stream());

        if (enable_label_output_) {
          auto *label = label_output->mutable_tensor<int>(data_idx);
          CUDA_CALL(cudaMemcpyAsync(label, &prefetched_sequence.label, sizeof(int),
                                    cudaMemcpyDefault, ws->stream()));
        }
    }
  }


 private:
  std::vector<std::string> filenames_;
  std::string file_root_;
  std::string file_list_;
  int count_;
  int height_;
  int width_;
  int channels_;

  float output_scale_;

  kernels::TensorListShape<> tl_shape_;
  kernels::TensorListShape<> label_shape_;

  DALIDataType dtype_;
  bool enable_label_output_;

  USE_READER_OPERATOR_MEMBERS(GPUBackend, SequenceWrapper);
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_READER_VIDEO_READER_OP_H_
