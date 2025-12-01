// Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_IMAGE_CROP_RANDOM_CROP_GENERATOR_H_
#define DALI_OPERATORS_IMAGE_CROP_RANDOM_CROP_GENERATOR_H_

#include <vector>
#include <random>
#include <memory>
#include <utility>
#include <string>

#include "dali/core/static_switch.h"
#include "dali/core/tensor_shape.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/operator/op_spec.h"
#include "dali/pipeline/operator/common.h"
#include "dali/operators/image/crop/random_crop_attr.h"
#include "dali/pipeline/operator/checkpointing/snapshot_serializer.h"
#include "dali/pipeline/operator/checkpointing/op_checkpoint.h"

namespace dali {

template <typename Backend>
class RandomCropGeneratorOp : public Operator<Backend> {
 public:
  explicit inline RandomCropGeneratorOp(const OpSpec &spec)
      : Operator<Backend>(spec), crop_attr_(spec) {}

  inline ~RandomCropGeneratorOp() override = default;

  DISABLE_COPY_MOVE_ASSIGN(RandomCropGeneratorOp);

  USE_OPERATOR_MEMBERS();
  using Operator<Backend>::RunImpl;


 protected:
  bool SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) override {
    auto curr_batch_size = ws.GetInputBatchSize(0);
    auto &input = ws.Input<Backend>(0);
    const auto &in_shape = input.shape();
    DALIDataType in_type = input.type();

    int N = in_shape.num_samples();
    for (int sample_idx = 0; sample_idx < N; sample_idx++) {
      auto sample_shape = in_shape.tensor_shape_span(sample_idx);
      DALI_ENFORCE(
          sample_shape.size() == 1 && (sample_shape[0] == 2 || sample_shape[0] == 3),
          "Expected a 1D tensor with 2 or 3 values representing an input shape (HW or HWC)");
    }
    output_desc.resize(2);
    // outputs are anchor and shape
    output_desc[0].type = DALI_INT32;
    output_desc[0].shape = uniform_list_shape<1>(curr_batch_size, TensorShape<1>{2});
    output_desc[1].type = DALI_INT32;
    output_desc[1].shape = uniform_list_shape<1>(curr_batch_size, TensorShape<1>{2});
    return true;
  }

  template <typename T>
  void RunImplTyped(Workspace& ws) {
    auto curr_batch_size = ws.GetInputBatchSize(0);
    auto& input = ws.Input<Backend>(0);
    auto& output_anchor = ws.Output<Backend>(0);
    auto& output_shape = ws.Output<Backend>(1);
    for (int s = 0; s < curr_batch_size; s++) {
      auto img_dims = view<const T, 1>(input[s]);
      int32_t H = img_dims.data[0];
      int32_t W = img_dims.data[1];
      auto crop_win = crop_attr_.GetCropWindowGenerator(s)({H, W}, "HW");

      auto crop_anchor = view<int32_t, 1>(output_anchor[s]);
      crop_anchor.data[0] = crop_win.anchor[0];
      crop_anchor.data[1] = crop_win.anchor[1];

      auto crop_shape = view<int32_t, 1>(output_shape[s]);
      crop_shape.data[0] = crop_win.shape[0];
      crop_shape.data[1] = crop_win.shape[1];
    }
  }

  void RunImpl(Workspace &ws) override {
    auto dtype = ws.Input<Backend>(0).type();
    TYPE_SWITCH(dtype, type2id, T, (uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, int64_t),
      (RunImplTyped<T>(ws);),
      (DALI_FAIL(
          make_string("Only integer types are supported. Got: ", dtype));));
  }

  void SaveState(OpCheckpoint &cpt, AccessOrder order) override {
    cpt.MutableCheckpointState() = crop_attr_.RNGSnapshot();
  }

  void RestoreState(const OpCheckpoint &cpt) override {
    auto &rngs = cpt.CheckpointState<std::vector<std::mt19937>>();
    crop_attr_.RestoreRNGState(rngs);
  }

  std::string SerializeCheckpoint(const OpCheckpoint &cpt) const override {
    const auto &state = cpt.CheckpointState<std::vector<std::mt19937>>();
    return SnapshotSerializer().Serialize(state);
  }

  void DeserializeCheckpoint(OpCheckpoint &cpt, const std::string &data) const override {
    cpt.MutableCheckpointState() =
      SnapshotSerializer().Deserialize<std::vector<std::mt19937>>(data);
  }

 private:
  RandomCropAttr crop_attr_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_IMAGE_CROP_RANDOM_CROP_GENERATOR_H_
