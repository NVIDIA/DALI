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

    bool output_class_ = spec.GetArgument<bool>("output_class");

    if (ignore_class_ && (classes_.IsDefined() || weights_.IsDefined() || output_class_)) {
      DALI_FAIL("Class-related arguments ``classes``, ``weights`` and ``output_class`` "
                "cannot be used when ``ignore_class`` is True");
    }

    // additional class id output goes last, if at all; -1 denotes that it's absent
    class_output_idx_ = output_class_ ? (format_ == Out_Box ? 1 : 2) : -1;


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

  bool HasClassLabelOutput() const {
    return class_output_idx_ >= 0;
  }

  using ClassVec = SmallVector<int, 32>;
  using WeightVec = SmallVector<float, 32>;

  void GetBgFgAndWeights(ClassVec &classes, WeightVec &weights, int &background, int sample_idx);

  struct SampleContext {
    void Init(int sample_idx, const Tensor<CPUBackend> *in) {
      this->sample_idx = sample_idx;
      input = in;
      auto &shape = input->shape();
      int64_t n = volume(shape);
      filtered_data.resize(n);
      blob_data.resize(n);
      filtered = make_tensor_cpu(filtered_data.data(), shape);
      blobs = make_tensor_cpu(blob_data.data(), shape);
      labels.clear();
    }

    /**
     * @brief Calculate CDF from weights
     */
    bool CalculateCDF() {
      int ncls = weights.size();
      cdf.resize(ncls);
      double sum = 0;  // accumulate in double for increased precision
      for (int i = 0; i < ncls; i++) {
        sum += weights[i];
        cdf[i] = static_cast<float>(sum);  // downconvert to float
      }
      return sum > 0;
    }

    /**
     * @brief Pick a random label according to CDF
     *
     * Use binary search to find the label in CDF
     */
    template <typename RNG>
    bool PickClassLabel(RNG &rng) {
      int ncls = cdf.size();
      if (!ncls)
        return false;

      double pos = class_dist(rng) * cdf.back();

      int idx = std::lower_bound(cdf.begin(), cdf.end(), pos) - cdf.begin();
      // the index may be ambiguous if there are zero weights, so we need to skip these
      while (idx < ncls && weights[idx] == 0)
        idx++;
      class_idx = idx >= ncls ? -1 : idx;
      class_label = class_idx >= 0 ? classes[class_idx] : background;
      return class_idx >= 0;
    }

    TensorView<StorageCPU, int> out1, out2;
    const Tensor<CPUBackend> *input = nullptr;
    int sample_idx;

    ClassVec classes;
    WeightVec weights;
    WeightVec cdf;
    int background;
    int class_idx;
    int class_label;

    vector<uint8_t> filtered_data;
    vector<int64_t> blob_data;
    TensorView<StorageCPU, uint8_t> filtered;
    TensorView<StorageCPU, int64_t> blobs;
    std::unordered_set<int> labels;
    std::uniform_real_distribution<double> class_dist{0, 1};
    vector<int> box_data;
    struct {
      SmallVector<int, 6> lo, hi;
    } selected_box;

    void SelectBox(int index) {
      int ndim = blobs.dim();
      selected_box.lo.resize(ndim);
      selected_box.hi.resize(ndim);
      for (int d = 0; d < ndim; d++) {
        selected_box.lo[d] = box_data[2*index*ndim + d];
        selected_box.hi[d] = box_data[2*index*ndim + d + ndim];
      }
    }
  };
  vector<SampleContext> contexts_;

  bool PickForegroundBox(SampleContext &context);

  template <typename T>
  bool PickForegroundBox(SampleContext &context, const TensorView<StorageCPU, const T> &input);

  bool PickBlob(SampleContext &ctx, int nblobs);

  template <int ndim>
  int PickBox(span<Box<ndim, int>> boxes, int sample_idx);

  template <typename T>
  void FindLabels(std::unordered_set<int> &labels, const T *data, int64_t N);

  template <typename T>
  void FindLabels(std::unordered_set<int> &labels, const TensorView<StorageCPU, const T> &in) {
    FindLabels(labels, in.data, in.num_elements());
  }


  bool  ignore_class_ = false;
  int   k_largest_ = -1;          // -1 means no k largest
  int   class_output_idx_ = -1;   // -1 means no class output
  BatchRNG<> rngs_;
  ArgValue<int> background_;
  ArgValue<int, 1> classes_;
  ArgValue<float> foreground_prob_;
  ArgValue<float, 1> weights_;
  ArgValue<int, 1> threshold_;
  OutputFormat format_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_SEGMENTATION_RANDOM_OBJECT_BBOX_H_
