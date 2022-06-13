// Copyright (c) 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_IMAGE_PEEK_SHAPE_PEEK_IMAGE_SHAPE_H_
#define DALI_OPERATORS_IMAGE_PEEK_SHAPE_PEEK_IMAGE_SHAPE_H_

#include <vector>
#include "dali/core/backend_tags.h"
#include "dali/image/image_factory.h"
#include "dali/core/tensor_shape.h"
#include "dali/pipeline/data/types.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/core/static_switch.h"

namespace dali {

class PeekImageShape : public Operator<CPUBackend> {
 public:
  PeekImageShape(const PeekImageShape &) = delete;

  explicit PeekImageShape(const OpSpec &spec) : Operator<CPUBackend>(spec) {
    output_type_ = spec.GetArgument<DALIDataType>("dtype");
    switch (output_type_) {
    case DALI_INT32:
    case DALI_UINT32:
    case DALI_INT64:
    case DALI_UINT64:
    case DALI_FLOAT:
    case DALI_FLOAT64:
      break;
    default:
      {
        DALI_FAIL(make_string(
          "Operator PeekImageShape can return the output as one of the following:\n"
          "int32, uint32, int64, uint64, float or double;\n"
          "requested: ", output_type_));
        break;
      }
    }
  }
  bool CanInferOutputs() const override {
    return true;
  }

 protected:
  bool SetupImpl(std::vector<OutputDesc> &output_desc, const HostWorkspace &ws) override {
    const auto &input = ws.template Input<CPUBackend>(0);
    size_t batch_size = input.num_samples();
    output_desc.resize(1);
    output_desc[0].shape = uniform_list_shape<1>(batch_size, { 3 });
    output_desc[0].type = output_type_;
    return true;
  }

  template <typename type>
  void WriteShape(TensorView<StorageCPU, type, 1> out, const TensorShape<3> &shape) {
    for (int i = 0; i < 3; ++i) {
      out.data[i] = shape[i];
    }
  }

  void RunImpl(HostWorkspace &ws) override {
    auto &thread_pool = ws.GetThreadPool();
    const auto &input = ws.template Input<CPUBackend>(0);
    auto &output = ws.template Output<CPUBackend>(0);
    size_t batch_size = input.num_samples();
    DALI_ENFORCE(input.type() == DALI_UINT8,
                 "The input must be a raw, undecoded file stored as a flat uint8 array.");
    DALI_ENFORCE(input.sample_dim() == 1, "Input must be 1D encoded JPEG bit stream.");

    for (size_t sample_id = 0; sample_id < batch_size; ++sample_id) {
      thread_pool.AddWork([sample_id, &input, &output, this] (int tid) {
        const auto& image = input[sample_id];
        auto img =
            ImageFactory::CreateImage(image.data<uint8>(), image.shape().num_elements(), {});
        auto shape = img->PeekShape();
        TYPE_SWITCH(output_type_, type2id, type,
                (int32_t, uint32_t, int64_t, uint64_t, float, double),
          (WriteShape(view<type, 1>(output[sample_id]), shape);),
          (DALI_FAIL(make_string("Unsupported type for Shapes: ", output_type_))));
      }, 0);
      // the amount of work depends on the image format and exact sample which is unknown here
    }
    thread_pool.RunAll();
  }

  USE_OPERATOR_MEMBERS();
  using Operator<CPUBackend>::RunImpl;

 private:
  DALIDataType output_type_ = DALI_INT64;
};

}  // namespace dali

#endif  // DALI_OPERATORS_IMAGE_PEEK_SHAPE_PEEK_IMAGE_SHAPE_H_
