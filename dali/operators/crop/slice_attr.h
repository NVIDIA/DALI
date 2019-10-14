// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_OPERATORS_CROP_SLICE_ATTR_H_
#define DALI_OPERATORS_CROP_SLICE_ATTR_H_

#include <utility>
#include <vector>
#include "dali/core/common.h"
#include "dali/core/error_handling.h"
#include "dali/operators/common.h"
#include "dali/operators/operator.h"
#include "dali/util/crop_window.h"
#include "dali/core/tensor_layout.h"

namespace dali {

/**
 * @brief Crop parameter and input size handling.
 *
 * Responsible for accessing image type, starting points and size of crop area.
 */
class SliceAttr {
 protected:
    explicit inline SliceAttr(const OpSpec &spec)
        : batch_size__(spec.GetArgument<int>("batch_size")) {
        crop_height_.resize(batch_size__, 0.0f);
        crop_width_.resize(batch_size__, 0.0f);
        crop_x_norm_.resize(batch_size__, 0.0f);
        crop_y_norm_.resize(batch_size__, 0.0f);
        crop_window_generators_.resize(batch_size__, {});
    }

    void ProcessArguments(const SampleWorkspace &ws) {
        DALI_ENFORCE(ws.NumInput() == 3,
            "Expected 3 inputs. Received: " + std::to_string(ws.NumInput()));

        const auto &images = ws.Input<CPUBackend>(0);
        const auto &crop_begin = ws.Input<CPUBackend>(1);
        const auto &crop_size = ws.Input<CPUBackend>(2);
        int data_idx = ws.data_idx();
        // Assumes xywh
        ProcessArgumentsHelper(
            data_idx,
            crop_size.data<float>()[0],
            crop_size.data<float>()[1],
            crop_begin.data<float>()[0],
            crop_begin.data<float>()[1]);
    }

    void ProcessArguments(MixedWorkspace &ws) {
        DALI_ENFORCE(ws.NumInput() == 3,
            "Expected 3 inputs. Received: " + std::to_string(ws.NumInput()));
        for (std::size_t data_idx = 0; data_idx < batch_size__; data_idx++) {
            const auto &images = ws.Input<CPUBackend>(0, data_idx);
            const auto &crop_begin = ws.Input<CPUBackend>(1, data_idx);
            const auto &crop_size = ws.Input<CPUBackend>(2, data_idx);
            // Assumes xywh
            ProcessArgumentsHelper(
                data_idx,
                crop_size.data<float>()[0],
                crop_size.data<float>()[1],
                crop_begin.data<float>()[0],
                crop_begin.data<float>()[1]);
        }
    }

  const CropWindowGenerator& GetCropWindowGenerator(std::size_t data_idx) const {
    DALI_ENFORCE(data_idx < crop_window_generators_.size());
    return crop_window_generators_[data_idx];
  }

  std::vector<float> crop_height_;
  std::vector<float> crop_width_;
  std::vector<float> crop_x_norm_;
  std::vector<float> crop_y_norm_;
  std::vector<CropWindowGenerator> crop_window_generators_;

 private:
  void ProcessArgumentsHelper(int data_idx, float crop_w, float crop_h,
                              float crop_x_norm, float crop_y_norm) {
    crop_width_[data_idx] = crop_w;
    crop_height_[data_idx] = crop_h;
    crop_x_norm_[data_idx] = crop_x_norm;
    crop_y_norm_[data_idx] = crop_y_norm;

    DALI_ENFORCE(crop_x_norm + crop_w <= 1.0f,
      "crop_x[" + std::to_string(crop_x_norm) + "] + crop_width["
      + std::to_string(crop_w) + "] must be <= 1.0f");
    DALI_ENFORCE(crop_y_norm + crop_h <= 1.0f,
      "crop_y[" + std::to_string(crop_y_norm) + "] + crop_height["
      + std::to_string(crop_h) + "] must be <= 1.0f");

    crop_window_generators_[data_idx] =
      [this, data_idx](const kernels::TensorShape<>& shape,
                       const TensorLayout& layout) {
        CropWindow crop_window;
        crop_window.anchor[0] = crop_y_norm_[data_idx] * shape[0];
        crop_window.anchor[1] = crop_x_norm_[data_idx] * shape[1];
        crop_window.shape[0] =
          (crop_height_[data_idx] + crop_y_norm_[data_idx]) * shape[0] - crop_window.anchor[0];
        crop_window.shape[1] =
          (crop_width_[data_idx] + crop_x_norm_[data_idx]) * shape[1] - crop_window.anchor[1];
        DALI_ENFORCE(crop_window.IsInRange(shape));
        return crop_window;
    };
  }

  std::size_t batch_size__;
};

}  // namespace dali

#endif  // DALI_OPERATORS_CROP_SLICE_ATTR_H_
