// Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
// limitations under the License.

#include <map>
#include <memory>
#include <string>
#include <vector>
#include "dali/core/tensor_shape.h"
#include "dali/operators/generic/slice/slice_attr.h"
#include "dali/operators/image/crop/crop_attr.h"
#include "dali/operators/image/crop/random_crop_attr.h"
#include "dali/operators/imgcodec/imgcodec.h"
#include "dali/pipeline/operator/checkpointing/snapshot_serializer.h"
#include "dali/pipeline/operator/checkpointing/op_checkpoint.h"
#include "dali/pipeline/operator/common.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/util/crop_window.h"

#ifndef DALI_OPERATORS_IMGCODEC_ROI_IMAGE_DECODER_H_
#define DALI_OPERATORS_IMGCODEC_ROI_IMAGE_DECODER_H_

namespace dali {
namespace imgcodec {

inline ROI RoiFromCropWindowGenerator(const CropWindowGenerator& generator, TensorShape<> shape) {
  shape = shape.first(2);
  auto crop_window = generator(shape, "HW");
  crop_window.EnforceInRange(shape);
  TensorShape<> end = crop_window.anchor;
  for (int d = 0; d < end.size(); d++) {
    end[d] += crop_window.shape[d];
  }
  return {crop_window.anchor, end};
}

template <typename Decoder, typename Backend>
class WithCropAttr : public Decoder, CropAttr {
 public:
  explicit WithCropAttr(const OpSpec &spec) : Decoder(spec), CropAttr(spec) {}

 protected:
  void SetupRoiGenerator(const OpSpec &spec, const Workspace &ws) override {
    CropAttr::ProcessArguments(spec, ws);
  }

  ROI GetRoi(const OpSpec &spec, const Workspace &ws, std::size_t data_idx,
             TensorShape<> shape) override {
    return RoiFromCropWindowGenerator(GetCropWindowGenerator(data_idx), shape);
  }
};

template <typename Decoder, typename Backend>
class WithSliceAttr : public Decoder, SliceAttr {
 public:
  explicit WithSliceAttr(const OpSpec &spec) : Decoder(spec), SliceAttr(spec) {}
 protected:
  void SetupRoiGenerator(const OpSpec &spec, const Workspace &ws) override {
    SliceAttr::ProcessArguments<Backend>(spec, ws);
  }

  ROI GetRoi(const OpSpec &spec, const Workspace &ws, std::size_t data_idx,
             TensorShape<> shape) override {
    return RoiFromCropWindowGenerator(SliceAttr::GetCropWindowGenerator(data_idx), shape);
  }
};

template <typename Decoder, typename Backend>
class WithRandomCropAttr : public Decoder, RandomCropAttr {
 public:
  explicit WithRandomCropAttr(const OpSpec &spec) : Decoder(spec), RandomCropAttr(spec) {}

 protected:
  void SetupRoiGenerator(const OpSpec &spec, const Workspace &ws) override {}

  ROI GetRoi(const OpSpec &spec, const Workspace &ws, std::size_t data_idx,
             TensorShape<> shape) override {
    return RoiFromCropWindowGenerator(RandomCropAttr::GetCropWindowGenerator(data_idx), shape);
  }

  void SaveState(OpCheckpoint &cpt, AccessOrder order) override {
    cpt.MutableCheckpointState() = RNGSnapshot();
  }

  void RestoreState(const OpCheckpoint &cpt) override {
    auto &rngs = cpt.CheckpointState<std::vector<std::mt19937>>();
    RestoreRNGState(rngs);
  }

  std::string SerializeCheckpoint(const OpCheckpoint &cpt) const override {
    const auto &state = cpt.CheckpointState<std::vector<std::mt19937>>();
    return SnapshotSerializer().Serialize(state);
  }

  void DeserializeCheckpoint(OpCheckpoint &cpt, const std::string &data) const override {
    cpt.MutableCheckpointState() =
        SnapshotSerializer().Deserialize<std::vector<std::mt19937>>(data);
  }
};


}  // namespace imgcodec
}  // namespace dali

#endif  // DALI_OPERATORS_IMGCODEC_ROI_IMAGE_DECODER_H_
