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


#ifndef DALI_OPERATORS_SEGMENTATION_RANDOM_OBJECT_BBOX_H_
#define DALI_OPERATORS_SEGMENTATION_RANDOM_OBJECT_BBOX_H_

#include <string>
#include <random>
#include <unordered_map>
#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/operator/arg_helper.h"
#include "dali/pipeline/util/batch_rng.h"

namespace dali {

class RandomObjectBBox : public Operator<CPUBackend> {
 public:
  enum OutputFormat {
    Out_AnchorShape,
    Out_StartEnd,
    Out_Box
  };

  explicit RandomObjectBBox(const OpSpec &spec) : Operator<CPUBackend>(spec),
    foreground_prob_("foreground_prob", spec),
    classes_("classes", spec),
    weights_("weights", spec),
    k_largest_("k_largest", spec),
    threshold_("threshold", spec) {
    string format_string = spec.GetArgument("format");
    format_ = ParseOutputFormat
  }

  static OutputFormat ParseOutputFormat(const std::string &format)  {
    if (format == "anchor_shape")
      return Out_AnchorShape;
    else if (format == "start_end")
      return Out_StartEnd;
    else if (format == "box")
      return Out_Box;

    DALI_FAIL(make_string("Invalid output format: \"", format, "\"\n"
      "Possible values: \"anchor_shape\", \"start_end\" and \"box\"."));
  }

  bool CanInferOutputs() const override {
    return true;
  }

  void SetupImpl(

 private:
  std::unordered_map<int, int> label_map_;
  int background_;

  BatchRNG<> rngs_;
  ArgValue<int, 0> background_;
  ArgValue<int, 1> classes_;
  ArgValue<float, 0> foreground_prob_;
  ArgValue<float, 1> weights_;
  ArgValue<int, 0> k_largest_;
  ArgValue<int, 1> threshold_;
  OutputFormat format_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_SEGMENTATION_RANDOM_OBJECT_BBOX_H_
