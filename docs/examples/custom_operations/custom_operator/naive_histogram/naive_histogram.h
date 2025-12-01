// Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_NAIVE_HISTOGRAM_H
#define DALI_NAIVE_HISTOGRAM_H

#include <vector>
#include "dali/core/tensor_shape.h"
#include "dali/pipeline/operator/operator.h"

namespace naive_histogram {

template<typename Backend>
class NaiveHistogram : public ::dali::Operator<Backend> {
 public:
  explicit NaiveHistogram(const ::dali::OpSpec &spec) :
          ::dali::Operator<Backend>(spec),
          // We get a value of scalar argument from the OpSpec.
          n_histogram_bins_(spec.GetArgument<int>("n_bins")) {}


  virtual inline ~NaiveHistogram() = default;

  NaiveHistogram(const NaiveHistogram &) = delete;

  NaiveHistogram &operator=(const NaiveHistogram &) = delete;

  NaiveHistogram(NaiveHistogram &&) = delete;

  NaiveHistogram &operator=(NaiveHistogram &&) = delete;

 protected:
  bool SetupImpl(std::vector<::dali::OutputDesc> &output_desc,
                 const ::dali::Workspace &ws) override {
    /*
     * The purpose of the SetupImpl function is to tell the operator, what is the
     * shape of the output. With this information, the operator can preallocate the output.
     * Therefore, user won't need to do it in the RunImpl manually.
     */
    using namespace ::dali;
    const auto &input = ws.Input<Backend>(0);

    // Telling the operator, that there's one output.
    output_desc.resize(1);

    // Telling the operator, that every output sample is a 1-dimensional vector with
    // n_histogram_bins_ values and int32 type.
    output_desc[0] = {uniform_list_shape(input.num_samples(), {n_histogram_bins_}), DALI_INT32};
    return true;
  }


  void RunImpl(::dali::Workspace &ws) override;

 private:
  int n_histogram_bins_;
};

}  // namespace naive_histogram

#endif  // DALI_NAIVE_HISTOGRAM_H
