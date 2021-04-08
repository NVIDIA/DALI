// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_OPERATORS_IMAGE_DISTORTION_JPEG_COMPRESSION_DISTORTION_OP_H_
#define DALI_OPERATORS_IMAGE_DISTORTION_JPEG_COMPRESSION_DISTORTION_OP_H_

#include <string>
#include <vector>
#include "dali/pipeline/operator/arg_helper.h"
#include "dali/pipeline/data/views.h"
#include "dali/pipeline/operator/common.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/core/format.h"

namespace dali {

template <typename Backend>
class JpegCompressionDistortion : public Operator<Backend> {
 protected:
  explicit JpegCompressionDistortion(const OpSpec &spec)
      : Operator<Backend>(spec),
        spec_(spec),
        quality_arg_("quality", spec) {
  }

  bool CanInferOutputs() const override {
    return true;
  }

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const workspace_t<Backend> &ws) override {
    const auto &input = ws.template InputRef<Backend>(0);
    output_desc.resize(1);
    const auto &in_sh = input.shape();
    assert(in_sh.sample_dim() == 3);  // should be check by the layout
    for (int s = 0; s < in_sh.num_samples(); s++) {
      DALI_ENFORCE(in_sh.tensor_shape_span(s)[2] == 3,
        make_string("Expected RGB samples with HWC layout, got shape: ", in_sh[s]));
    }

    output_desc[0] = {in_sh, input.type()};
    quality_arg_.Acquire(spec_, ws, in_sh.num_samples(), TensorShape<0>{});
    return true;
  }

  OpSpec spec_;
  ArgValue<int> quality_arg_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_IMAGE_DISTORTION_JPEG_COMPRESSION_DISTORTION_OP_H_
