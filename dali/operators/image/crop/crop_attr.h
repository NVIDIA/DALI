// Copyright (c) 2019-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_IMAGE_CROP_CROP_ATTR_H_
#define DALI_OPERATORS_IMAGE_CROP_CROP_ATTR_H_

#include <vector>
#include "dali/core/common.h"
#include "dali/pipeline/operator/common.h"
#include "dali/pipeline/workspace/workspace.h"
#include "dali/pipeline/workspace/sample_workspace.h"
#include "dali/core/span.h"
#include "dali/core/tensor_shape.h"
#include "dali/util/crop_window.h"

namespace dali {

/**
 * @brief Crop parameter and input size handling.
 *
 * Responsible for accessing image type, starting points and size of crop area.
 */
class CropAttr {
 public:
  static constexpr int kNoCrop = -1;
  explicit CropAttr(const OpSpec& spec);

  void ProcessArguments(const OpSpec& spec, const ArgumentWorkspace* ws, std::size_t data_idx);

  TensorShape<> CalculateAnchor(const span<float>& anchor_norm, const TensorShape<>& crop_shape,
                                const TensorShape<>& input_shape);

  void ProcessArguments(const OpSpec& spec, const Workspace& ws);
  void ProcessArguments(const OpSpec& spec, const SampleWorkspace& ws);

  const CropWindowGenerator& GetCropWindowGenerator(std::size_t data_idx) const;

  inline bool IsWholeImage() const {
    return is_whole_image_;
  }

  std::vector<int> crop_height_;
  std::vector<int> crop_width_;
  std::vector<int> crop_depth_;
  std::vector<float> crop_x_norm_;
  std::vector<float> crop_y_norm_;
  std::vector<float> crop_z_norm_;
  std::vector<CropWindowGenerator> crop_window_generators_;
  bool is_whole_image_ = false;
  std::function<int64_t(double)> round_fn_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_IMAGE_CROP_CROP_ATTR_H_
