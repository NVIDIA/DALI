// Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_IMAGE_COLOR_DEBAYER_H_
#define DALI_OPERATORS_IMAGE_COLOR_DEBAYER_H_

#include <string>
#include <vector>

#include "dali/core/span.h"
#include "dali/kernels/imgproc/color_manipulation/debayer/debayer.h"
#include "dali/pipeline/operator/checkpointing/stateless_operator.h"
#include "dali/pipeline/operator/common.h"
#include "dali/pipeline/operator/op_spec.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/operator/sequence_operator.h"

#define DEBAYER_SUPPORTED_TYPES_GPU (uint8_t, uint16_t)

namespace dali {
namespace debayer {

using namespace kernels::debayer;  // NOLINT

constexpr static const char *kAlgArgName = "algorithm";
constexpr static const char *kBluePosArgName = "blue_position";

template <int ndim>
TensorListShape<3> infer_output_shape(const TensorListShape<ndim> &input_shapes) {
  int sample_dim = input_shapes.sample_dim();
  int batch_size = input_shapes.num_samples();
  DALI_ENFORCE(
      sample_dim == 2 || sample_dim == 3,
      make_string(
          "The debayer operator expects grayscale images, however the input has sample dim ",
          sample_dim, " which cannot be interpreted as grayscale images."));
  if (sample_dim == 3) {
    for (int sample_idx = 0; sample_idx < batch_size; sample_idx++) {
      int num_channels = input_shapes[sample_idx][2];
      DALI_ENFORCE(
          num_channels == 1,
          make_string("The debayer operator expects grayscale (i.e. single channel) images, "
                      "however the sample at idx ",
                      sample_idx, " has shape ", input_shapes[sample_idx],
                      ", which, assuming HWC layout, implies ", num_channels, " channels. ",
                      "If you are trying to process video or sequences, please make sure to set "
                      "input's layout to `FHW`. You can use `fn.reshape` to set proper layout."));
    }
  }
  TensorListShape<3> out_shapes(batch_size);
  for (int sample_idx = 0; sample_idx < batch_size; sample_idx++) {
    auto height = input_shapes[sample_idx][0];
    auto width = input_shapes[sample_idx][1];
    DALI_ENFORCE(height % 2 == 0 && width % 2 == 0,
                 make_string("The height and width of the image to debayer must be even. However, "
                             "the sample at idx ",
                             sample_idx, " has shape ", input_shapes[sample_idx], "."));
    TensorShape<3> out_shape{height, width, 3};
    out_shapes.set_tensor_shape(sample_idx, out_shape);
  }
  return out_shapes;
}

inline debayer::DALIBayerPattern blue_pos2pattern(span<const int> blue_position) {
  DALI_ENFORCE(
      blue_position.size() == 2,
      make_string(
          "The ", debayer::kBluePosArgName,
          " must be specified with exactly two values to describe a position within 2D tile. Got ",
          blue_position.size(), " values."));
  int y = blue_position[0];
  int x = blue_position[1];
  DALI_ENFORCE(
      0 <= y && y <= 1 && 0 <= x && x <= 1,
      make_string("The `", debayer::kBluePosArgName,
                  "` position must lie within 2x2 tile, got: ", TensorShape<2>{y, x}, "."));
  int bayer_enum_val = 2 * y + x;
  return static_cast<debayer::DALIBayerPattern>(bayer_enum_val);
}

template <typename Backend>
class DebayerImplBase {
 public:
  virtual ~DebayerImplBase() = default;
  virtual void RunImpl(Workspace &ws) = 0;
};

}  // namespace debayer


template <typename Backend>
class Debayer : public SequenceOperator<Backend, StatelessOperator> {
 public:
  using Base = SequenceOperator<Backend, StatelessOperator>;
  explicit Debayer(const OpSpec &spec)
      : Base(spec),
        alg_{debayer::parse_algorithm_name(spec.GetArgument<std::string>(debayer::kAlgArgName))} {
    if (!spec_.HasTensorArgument(debayer::kBluePosArgName)) {
      std::vector<int> blue_pos;
      GetSingleOrRepeatedArg(spec_, blue_pos, debayer::kBluePosArgName, 2);
      static_pattern_ = debayer::blue_pos2pattern(make_span(blue_pos));
    }
  }

 protected:
  DISABLE_COPY_MOVE_ASSIGN(Debayer);
  USE_OPERATOR_MEMBERS();

 protected:
  bool SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) override {
    output_desc.resize(1);
    const auto &input_shape = ws.GetInputShape(0);
    output_desc[0].type = ws.GetInputDataType(0);
    output_desc[0].shape = debayer::infer_output_shape(input_shape);
    AcquirePatternArgument(ws, input_shape.num_samples());
    return true;
  }

  bool CanInferOutputs() const override {
    return true;
  }

  void AcquirePatternArgument(const Workspace &ws, int batch_size) {
    if (!spec_.HasTensorArgument(debayer::kBluePosArgName)) {
      pattern_.resize(batch_size, static_pattern_);
    } else {
      const auto &tv = ws.ArgumentInput(debayer::kBluePosArgName);
      auto blue_positions_view = view<const int>(tv);
      pattern_.clear();
      pattern_.reserve(batch_size);
      for (int sample_idx = 0; sample_idx < batch_size; sample_idx++) {
        const auto &blue_pos = blue_positions_view[sample_idx];
        pattern_.push_back(
            debayer::blue_pos2pattern(make_span(blue_pos.data, blue_pos.num_elements())));
      }
    }
  }

  debayer::DALIBayerPattern static_pattern_ = {};
  debayer::DALIDebayerAlgorithm alg_;
  std::vector<debayer::DALIBayerPattern> pattern_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_IMAGE_COLOR_DEBAYER_H_
