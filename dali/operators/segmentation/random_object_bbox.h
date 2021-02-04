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
#include <unordered_set>
#include <vector>
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
        rngs_(spec.GetArgument<int>("seed"), max_batch_size_),
        background_("background", spec),
        classes_("classes", spec),
        foreground_prob_("foreground_prob", spec),
        weights_("class_weights", spec),
        threshold_("threshold", spec) {
    format_ = ParseOutputFormat(spec.GetArgument<string>("format"));

    ignore_class_ = spec.GetArgument<bool>("ignore_class");
    if (ignore_class_ && (classes_.IsDefined() || weights_.IsDefined())) {
      DALI_FAIL("Class-related arguments ``classes`` and ``weights`` cannot be used "
                "when ``ignore_class`` is True");
    }
    if (spec.TryGetArgument(k_largest_, "k_largest")) {
      DALI_ENFORCE(k_largest_ >= 1, make_string(
                   "``k_largest`` must be at least 1; got ", k_largest_));
    }
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

  bool SetupImpl(vector<OutputDesc> &out_descs, const HostWorkspace &ws) override;
  void RunImpl(HostWorkspace &ws) override;

 private:
  void AcquireArgs(const HostWorkspace &ws, int N, int ndim);

  void ProcessSample(Context &context);

  template <typename T>
  void ProcessSample(Context &context, const TensorView<StorageCPU, const T> &input);

  template <typename T>
  void FindLabels(std::unordered_set<int> &labels, const T *data, int64_t N);

  template <typename T>
  void FindLabels(std::unordered_set<int> &labels, const TensorView<StorageCPU, const T> &in) {
    FindLabels(labels, in.data, in.num_elements());
  }


  bool  ignore_class_ = false;
  int   k_largest_ = -1;
  BatchRNG<> rngs_;
  ArgValue<int> background_;
  ArgValue<int, 1> classes_;
  ArgValue<float> foreground_prob_;
  ArgValue<float, 1> weights_;
  ArgValue<int, 1> threshold_;
  OutputFormat format_;

  struct Context {
    void Init(int sample_idx, const Tensor<CPUBackend> &in) {
      this->sample_idx = sample_idx;
      input = in;
      auto &shape = input.shape;
      int64_t n = volume(shape);
      filtered_data.resize(2);
      blob_data.resize(2);
      filtered = make_tensor_cpu(filtered_data.data(), shape);
      blobs = make_tensor_cpu(blob_data.data(), shape);
      labels.clear();
    }

    TensorView<StorageCPU, int> out1, out2;
    Tensor<CPUBackend> input;
    int sample_idx;

    vector<int> filtered_data;
    vector<int64_t> blob_data;
    TensorView<StorageCPU, int> filtered;
    TensorView<StorageCPU, int64_t> blobs;
    std::unordered_set<int> labels, tmp_labels;
    vector<int> ordered_labels;
    vector<int> box_data;
  };
  vector<Context> contexts_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_SEGMENTATION_RANDOM_OBJECT_BBOX_H_
