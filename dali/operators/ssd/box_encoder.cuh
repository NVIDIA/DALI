// Copyright (c) 2017-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_SSD_BOX_ENCODER_CUH_
#define DALI_OPERATORS_SSD_BOX_ENCODER_CUH_

#include <utility>
#include <vector>

#include "dali/core/dev_buffer.h"
#include "dali/core/tensor_shape.h"
#include "dali/operators/ssd/box_encoder.h"
#include "dali/pipeline/operator/checkpointing/stateless_operator.h"

namespace dali {


struct BoxEncoderSampleDesc {
  float4 *boxes_out;
  int *labels_out;
  const float4 *boxes_in;
  const int *labels_in;
  int in_box_count;
};

template <>
class BoxEncoder<GPUBackend> : public StatelessOperator<GPUBackend> {
 public:
  static constexpr int BlockSize = 256;
  using BoundingBox = Box<2, float>;

  explicit BoxEncoder(const OpSpec &spec)
      : StatelessOperator<GPUBackend>(spec),
        curr_batch_size_(-1),
        criteria_(spec.GetArgument<float>("criteria")),
        offset_(spec.GetArgument<bool>("offset")),
        scale_(spec.GetArgument<float>("scale")) {
    DALI_ENFORCE(criteria_ >= 0.f,
                 "Expected criteria >= 0, actual value = " + std::to_string(criteria_));
    DALI_ENFORCE(criteria_ <= 1.f,
                 "Expected criteria <= 1, actual value = " + std::to_string(criteria_));

    PrepareAnchors(spec.GetArgument<vector<float>>("anchors"));

    auto means = spec.GetArgument<vector<float>>("means");
    DALI_ENFORCE(means.size() == BoundingBox::size,
      "means size must be a list of 4 values.");

    means_.Resize({BoundingBox::size}, DALI_FLOAT);
    auto means_data = means_.mutable_data<float>();
    MemCopy(means_data, means.data(), BoundingBox::size * sizeof(float));

    auto stds = spec.GetArgument<vector<float>>("stds");
    DALI_ENFORCE(stds.size() == BoundingBox::size,
      "stds size must be a list of 4 values.");
    DALI_ENFORCE(std::find(stds.begin(), stds.end(), 0) == stds.end(),
       "stds values must be != 0.");
    stds_.Resize({BoundingBox::size}, DALI_FLOAT);
    auto stds_data = stds_.mutable_data<float>();
    MemCopy(stds_data, stds.data(), BoundingBox::size * sizeof(float));
    CUDA_CALL(cudaStreamSynchronize(0));
  }

  virtual ~BoxEncoder() = default;

  DISABLE_COPY_MOVE_ASSIGN(BoxEncoder);

 protected:
  bool SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) override {
    curr_batch_size_ = ws.GetInputBatchSize(0);

    best_box_idx_.Resize({curr_batch_size_ * anchor_count_}, DALI_INT32);
    best_box_iou_.Resize({curr_batch_size_ * anchor_count_}, DALI_FLOAT);
    return false;
  }

  void RunImpl(Workspace &ws) override;

 private:
  static constexpr int kBoxesOutputDim = 2;
  static constexpr int kLabelsOutputDim = 1;
  int curr_batch_size_;
  const float criteria_;
  int64_t anchor_count_;
  Tensor<GPUBackend> anchors_;
  Tensor<GPUBackend> anchors_as_center_wh_;
  Tensor<GPUBackend> best_box_idx_;
  Tensor<GPUBackend> best_box_iou_;

  std::vector<BoxEncoderSampleDesc> samples;
  DeviceBuffer<BoxEncoderSampleDesc> samples_dev;

  bool offset_;
  Tensor<GPUBackend> means_;
  Tensor<GPUBackend> stds_;
  float scale_;

  std::pair<int*, float*> ClearBuffers(const cudaStream_t &stream);

  void PrepareAnchors(const vector<float> &anchors);

  void ClearLabels(TensorList<GPUBackend> &labels_out, const cudaStream_t &stream);

  void WriteAnchorsToOutput(TensorList<GPUBackend> &boxes_out, const cudaStream_t &stream);

  void ClearOutput(TensorList<GPUBackend> &boxes_out, const cudaStream_t &stream);

  std::pair<TensorListShape<>, TensorListShape<>> CalculateDims(
    const TensorList<GPUBackend> &boxes_input);

  static const int kBoxesInId = 0;
  static const int kLabelsInId = 1;
  static const int kBoxesOutId = 0;
  static const int kLabelsOutId = 1;
};
}  // namespace dali

#endif  // DALI_OPERATORS_SSD_BOX_ENCODER_CUH_
