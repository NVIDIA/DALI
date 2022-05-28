// Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_GENERIC_REDUCE_HISTOGRAM_H_
#define DALI_OPERATORS_GENERIC_REDUCE_HISTOGRAM_H_

#include "dali/kernels/kernel_manager.h"
#include "dali/operators/generic/reduce/axes_helper.h"
#include "dali/pipeline/operator/arg_helper.h"
#include "dali/pipeline/operator/operator.h"

namespace dali {

namespace hist_detail {

struct HistReductionAxesHelper : detail::AxesHelper {
 public:
  explicit HistReductionAxesHelper(const OpSpec &);

  void PrepareReductionAxes(const TensorLayout &layout, int sample_dim, int hist_dim);
  bool IsIdentityTransform() const {
    return is_identity_;
  }
  bool IsSimpleReduction1() const;
  bool NeedsTranspose() const;

 private:
  void PrepareChannelAxisArg(const TensorLayout &layout,
                             const SmallVector<bool, 6> &reduction_axes_mask, int hist_dim,
                             bool implicit_axes);

 public:
  // TODO: make private
  span<int> reduction_axes_;
  span<int> non_reduction_axes_;
  int channel_axis_ = -1;
  SmallVector<int, 6> axes_order_;
  std::string channel_axis_name_;
  bool has_channel_axis_arg_;
  bool has_channel_axis_name_arg_;
  bool is_identity_ = false;
};

}  // namespace hist_detail

class HistogramCPU : public Operator<CPUBackend>, hist_detail::HistReductionAxesHelper {
 public:
  explicit HistogramCPU(const OpSpec &spec);

  bool CanInferOutputs() const override {
    return true;
  }

  ~HistogramCPU() override = default;

 private:
  int ValidateRangeArguments(const workspace_t<CPUBackend> &ws, int num_samples);
  int ValidateUniformRangeArguments(const workspace_t<CPUBackend> &ws, int num_samples);
  int ValidateNonUniformRangeArguments(const workspace_t<CPUBackend> &ws, int num_samples);

  void ValidateBinsArgument(const workspace_t<CPUBackend> &ws, int num_samples, int hist_dim);
  void InferBinsArgument(const workspace_t<CPUBackend> &ws, int num_samples, int hist_dim);

  void PrepareReductionShapes(const TensorListShape<> &input_shapes, OutputDesc &output_desc);
  void SubdivideTensorsShapes(const TensorListShape<> &input_shapes,
                              const TensorListShape<> &output_shapes, OutputDesc &output_desc);

  TensorListShape<> GetBinShapes(int num_samples) const;

 public:
  bool SetupImpl(std::vector<OutputDesc> &output_desc, const workspace_t<CPUBackend> &ws) override;
  void RunImpl(workspace_t<CPUBackend> &ws) override;

 private:
  USE_OPERATOR_MEMBERS();
  TensorListShape<> splited_input_shapes_;
  TensorListShape<> splited_output_shapes_;
  std::vector<int> split_mapping_;
  kernels::KernelManager kmgr_;
  std::vector<std::vector<float>> batch_ranges_;
  std::vector<SmallVector<int, 3>> batch_bins_;
  ArgValue<int, 1> param_num_bins_;
  kernels::ScratchpadAllocator transpose_mem_;
  int hist_dim_ = -1;
  bool needs_transpose_ = false;
  bool uniform_ = true;
};

}  // namespace dali

#endif  // DALI_OPERATORS_GENERIC_REDUCE_HISTOGRAM_H_
