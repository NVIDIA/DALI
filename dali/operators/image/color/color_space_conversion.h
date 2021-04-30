// Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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


#ifndef DALI_OPERATORS_IMAGE_COLOR_COLOR_SPACE_CONVERSION_H_
#define DALI_OPERATORS_IMAGE_COLOR_COLOR_SPACE_CONVERSION_H_

#include <vector>

#include "dali/pipeline/operator/operator.h"

namespace dali {

template <typename Backend>
class ColorSpaceConversion : public Operator<Backend> {
 public:
  inline explicit ColorSpaceConversion(const OpSpec &spec)
      : Operator<Backend>(spec),
        input_type_(spec.GetArgument<DALIImageType>("image_type")),
        output_type_(spec.GetArgument<DALIImageType>("output_type")),
        in_nchannels_(NumberOfChannels(input_type_)),
        out_nchannels_(NumberOfChannels(output_type_)) {
  }

 protected:
  bool CanInferOutputs() const override { return true; }

  bool SetupImpl(std::vector<OutputDesc> &output_desc,
                 const workspace_t<Backend> &ws) override {
    output_desc.resize(1);
    const auto &input = ws.template InputRef<Backend>(0);
    auto in_sh = input.shape();
    auto ndim = in_sh.sample_dim();
    int nsamples = in_sh.num_samples();
    auto in_layout = input.GetLayout();
    int channel_dim = in_layout.find('C');
    assert(channel_dim == ndim - 1);  // shoulb be enforced by input layouts
    DALI_ENFORCE(IsType<uint8_t>(input.type()), "Color space conversion accept only uint8 tensors");
    auto out_sh = in_sh;
    for (int i = 0; i < in_sh.num_samples(); i++) {
      int c = in_sh.tensor_shape_span(i)[channel_dim];
      DALI_ENFORCE(in_nchannels_ == c, make_string("Expected ", in_nchannels_, ". Got ", c));
      out_sh.tensor_shape_span(i)[channel_dim] = out_nchannels_;
    }
    output_desc[0].type = input.type();
    output_desc[0].shape = out_sh;
    return true;
  }

  void RunImpl(workspace_t<Backend> &ws) override;
  USE_OPERATOR_MEMBERS();
  using Operator<Backend>::RunImpl;

  const DALIImageType input_type_;
  const DALIImageType output_type_;
  const int in_nchannels_;
  const int out_nchannels_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_IMAGE_COLOR_COLOR_SPACE_CONVERSION_H_
