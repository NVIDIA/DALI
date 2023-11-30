// Copyright (c) 2017-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_PIPELINE_OPERATOR_BUILTIN_EXTERNAL_SOURCE_H_
#define DALI_PIPELINE_OPERATOR_BUILTIN_EXTERNAL_SOURCE_H_

#include <string>
#include <vector>
#include "dali/core/common.h"
#include "dali/pipeline/operator/builtin/input_operator.h"

namespace dali {

/**
 * @brief Provides in-graph access to data fed in from outside of dali.
 * For now, we do a copy from the passed in data into our data to avoid
 * potential scoping and data corruption issues.
 * Please note, that it is not allowed to call this concurrently as it
 * may mix the order of inputted data.
 */
template <typename Backend>
class ExternalSource : public InputOperator<Backend> {
  using Operator<Backend>::spec_;

 public:
  void SaveState(OpCheckpoint &cpt, AccessOrder order) override {}

  void RestoreState(const OpCheckpoint &cpt) override {}

  std::string SerializeCheckpoint(const OpCheckpoint &cpt) const override { return {}; }

  void DeserializeCheckpoint(OpCheckpoint &cpt, const std::string &data) const override {
    DALI_ENFORCE(data.empty(),
                 "Provided checkpoint contains non-empty data for a stateless operator. "
                 "The checkpoint might come from another pipeline. ");
  }

  explicit ExternalSource(const OpSpec &spec)
      : InputOperator<Backend>(spec),
        repeats_last_(spec.GetArgument<bool>("repeat_last")),
        previous_dtype_(DALIDataType::DALI_NO_TYPE),
        ndim_(-1),
        layout_() {
    spec.TryGetArgument(dtype_, "dtype");
    if (spec.TryGetArgument(ndim_, "ndim")) {
      DALI_ENFORCE(ndim_ >= 0, make_string("Incorrect number of dimensions (", ndim_,
                   "). Use positive values for tensors or 0 for scalars."));
    }
    spec.TryGetArgument(layout_, "layout");
    InferNdim();
    output_name_ = spec.Output(0);
  }

  virtual ~ExternalSource() = default;

  inline string name() const override {
    return "ExternalSource (" + output_name_ + ")";
  }

  const TensorLayout& in_layout() const override {
    return layout_;
  }

  int in_ndim() const override {
    return ndim_;
  }

  DALIDataType in_dtype() const override {
    return dtype_;
  }

  DISABLE_COPY_MOVE_ASSIGN(ExternalSource);

 protected:
  bool HasNdim() {
    return !layout_.empty() || spec_.HasArgument("ndim");
  }


  void InferNdim() {
    if (!layout_.empty()) {
      if (ndim_ != -1) {
        DALI_ENFORCE(ndim_ == layout_.ndim(), make_string("Number of dimensions in the provided "
                     "layout does not match the ndim argument. The arguments provided:",
                     "\n ndim = ", ndim_, ",",
                     "\n layout: \"", layout_, "\"."));
      } else {
        ndim_ = layout_.ndim();
      }
    }
  }


  bool SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) override {
    InputOperator<Backend>::HandleDataAvailability();
    ValidateInputData(InputOperator<Backend>::PeekCurrentData());
    TensorListShape<> shape;
    output_desc.resize(1);
    output_desc[0].shape = InputOperator<Backend>::PeekCurrentData().shape();
    output_desc[0].type = InputOperator<Backend>::PeekCurrentData().type();
    // unconditionally disabled, still we can provide shape but we don't want to allocate anything
    return false;
  }


  bool CanInferOutputs() const override {
    // shape inference during setup is disabled because it can be calculated during the runtime
    // depending on the input and output
    return false;
  }


  void RunImpl(Workspace &ws) override;


  template <typename SrcBackend>
  void ValidateInputData(const TensorList<SrcBackend> &batch) {
    const bool is_gpu_src = std::is_same<SrcBackend, GPUBackend>::value;
    const bool is_gpu_dst = std::is_same<Backend, GPUBackend>::value;
    if (is_gpu_src && !is_gpu_dst) {
      DALI_WARN(
          "Warning: Loading GPU-originated data into CPU ExternalSource operator is discouraged "
          "and might be inefficient.");
    }

    DALI_ENFORCE(
        OperatorBase::max_batch_size_ >= static_cast<int>(batch.num_samples()),
        make_string("Data list provided to ExternalSource needs to have batch_size <= ",
                    OperatorBase::max_batch_size_, ", found ", batch.num_samples(), " samples."));

    DALI_ENFORCE(batch.num_samples() > 0,
                 "ExternalSource expects non-empty batches to be provided as the input. Got batch "
                 "with 0 samples.");

    DALI_ENFORCE(
        dtype_ == DALI_NO_TYPE || dtype_ == batch.type(),
        make_string("ExternalSource expected data of type ", TypeTable::GetTypeInfo(dtype_).name(),
        " and got: ", batch.type_info().name()));

    DALI_ENFORCE(previous_dtype_ == DALI_NO_TYPE || previous_dtype_ == batch.type(),
      make_string("Type of the data fed to the external source has changed from the "
                  "previous iteration. Type in the previous iteration was ",
                  TypeTable::GetTypeInfo(previous_dtype_).name(),
                  " and the current type is ", batch.type_info().name(), "."));
    previous_dtype_ = batch.type();

    auto input_ndim = batch.shape().sample_dim();
    if (HasNdim()) {
      DALI_ENFORCE(input_ndim == ndim_,
                   make_string("ExternalSource expected data with ", ndim_, " dimensions and got ",
                     input_ndim, " dimensions."));
    } else if (ndim_ != -1) {
      DALI_ENFORCE(input_ndim == ndim_,
                   make_string("Number of dimensions of the data fed to the external source has "
                      "changed from previous iteration. Dimensionality in the previous "
                      "iteration was ", ndim_, " and the current is ", input_ndim, "."));
    }
    ndim_ = input_ndim;

    if (spec_.HasArgument("layout")) {
      if (batch.GetLayout().empty()) {
        layout_ = spec_.template GetArgument<TensorLayout>("layout");
      } else {
        DALI_ENFORCE(layout_ == batch.GetLayout(),
                     make_string("Expected data with layout: \"", layout_,
                     "\" and got: \"", batch.GetLayout(), "\"."));
      }
    } else {
        if (layout_.empty()) {
          layout_ = batch.GetLayout();
        } else  {
          DALI_ENFORCE(layout_ == batch.GetLayout(),
                        make_string("Layout of the data fed to the external source has changed "
                          "from previous iteration. Layout in the previous iteration was \"",
                          layout_, "\" and the current is \"", batch.GetLayout(), "\"."));
        }
    }
  }

  const bool repeats_last_;

  string output_name_;

  DALIDataType dtype_ = DALI_NO_TYPE;
  DALIDataType previous_dtype_ = DALI_NO_TYPE;
  int ndim_;
  TensorLayout layout_;
  std::optional<std::string> data_id_ = std::nullopt;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATOR_BUILTIN_EXTERNAL_SOURCE_H_
