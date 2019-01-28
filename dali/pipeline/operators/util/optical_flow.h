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

#ifndef DALI_OPTICAL_FLOW_H
#define DALI_OPTICAL_FLOW_H

#include "dali/pipeline/operators/operator.h"
#include "dali/pipeline/data/backend.h"

namespace dali {

template<typename Backend>
class OpticalFlow : public Operator<Backend> {
 public:
  explicit OpticalFlow(const OpSpec &spec) :
          Operator<Backend>(spec),
          quality_factor_(spec.GetArgument<float>("preset")),
          grid_size_(spec.GetArgument<int>("output_format")),
          enable_hints_(spec.GetArgument<bool>("enable_hints")) {

  }


  ~OpticalFlow() = default;
  DISABLE_COPY_MOVE_ASSIGN(OpticalFlow);

 protected:
  void RunImpl(SampleWorkspace *ws, const int idx) override {
    cout << quality_factor_ << endl << grid_size_ << endl << enable_hints_ << endl;
  }


 private:
  const float quality_factor_;
  const int grid_size_;
  const bool enable_hints_;
};

}  // namespace dali

#endif  // DALI_OPTICAL_FLOW_H
