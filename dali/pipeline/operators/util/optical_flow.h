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

#include <dali/pipeline/data/views.h>
#include <dali/pipeline/data/backend.h>
#include <dali/pipeline/operators/operator.h>
#include <dali/aux/optical_flow/optical_flow_stub.h>

namespace dali {

namespace detail {

template<typename Backend>
struct backend_to_compute {
  using type = kernels::ComputeCPU;
};

template<>
struct backend_to_compute<GPUBackend> {
  using type = kernels::ComputeGPU;
};

const std::string kPresetArgName = "preset";   // NOLINT
const std::string kOutputFormatArgName = "output_format";   // NOLINT
const std::string kEnableHintsArgName = "enable_hints";   // NOLINT

}  // namespace detail

template<typename Backend>
class OpticalFlow : public Operator<Backend> {
  using ComputeBackend = typename detail::backend_to_compute<Backend>::type;
 public:
  explicit OpticalFlow(const OpSpec &spec) :
          Operator<Backend>(spec),
          quality_factor_(spec.GetArgument<typename std::remove_const<
                  decltype(this->quality_factor_)>::type>(detail::kPresetArgName)),
          grid_size_(spec.GetArgument<typename std::remove_const<
                  decltype(this->grid_size_)>::type>(detail::kOutputFormatArgName)),
          enable_hints_(spec.GetArgument<typename std::remove_const<
                  decltype(this->enable_hints_)>::type>(detail::kEnableHintsArgName)),
          optical_flow_(std::unique_ptr<optical_flow::OpticalFlowAdapter<ComputeBackend>>(
                  new optical_flow::OpticalFlowStub<ComputeBackend>(of_params_))) {

    // In case hints are enabled, we need 2 inputs
    DALI_ENFORCE((enable_hints_ && spec.NumInput() == 2) || !enable_hints_,
                 "Incorrect number of inputs. Expected: 2, Obtained: " +
                 std::to_string(spec.NumInput()));
    optical_flow::VectorGridSize grid_size;
    if (grid_size_ < 4) {
      grid_size = optical_flow::VectorGridSize::UNDEF;
    } else if (grid_size_ == 4) {
      grid_size = optical_flow::VectorGridSize::SIZE_4;
    } else {
      grid_size = optical_flow::VectorGridSize::MAX;
    }
    of_params_ = {quality_factor_, grid_size, enable_hints_};
  }


  ~OpticalFlow() = default;
  DISABLE_COPY_MOVE_ASSIGN(OpticalFlow);

 protected:
  void RunImpl(Workspace<Backend> *ws, const int) override;


 private:
  const float quality_factor_;
  const int grid_size_;
  const bool enable_hints_;
  optical_flow::OpticalFlowParams of_params_;
  std::unique_ptr<optical_flow::OpticalFlowAdapter<ComputeBackend>> optical_flow_;
};

}  // namespace dali

#endif  // DALI_OPTICAL_FLOW_H
