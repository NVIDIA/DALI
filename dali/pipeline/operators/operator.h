// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_PIPELINE_OPERATORS_OPERATOR_H_
#define DALI_PIPELINE_OPERATORS_OPERATOR_H_

#include <algorithm>
#include <string>
#include <utility>
#include <vector>
#include <memory>

#include "dali/common.h"
#include "dali/error_handling.h"
#include "dali/pipeline/workspace/device_workspace.h"
#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/operators/operator_factory.h"
#include "dali/pipeline/operators/op_schema.h"
#include "dali/pipeline/operators/op_spec.h"
#include "dali/pipeline/workspace/sample_workspace.h"
#include "dali/pipeline/util/backend2workspace_map.h"

namespace dali {

template <typename InputType>
inline void CheckInputLayout(const InputType& input, const OpSpec& spec) {
  auto &schema = SchemaRegistry::GetSchema(spec.name());
  if (schema.EnforceInputLayout()) {
    DALI_ENFORCE(input.GetLayout() == schema.InputLayout());
  }
}

template <typename Workspace>
inline void CheckInputLayouts(const Workspace *ws, const OpSpec &spec) {
  for (int i = 0; i < spec.NumRegularInput(); ++i) {
    auto& input = ws->template Input<CPUBackend>(i);
    CheckInputLayout(input, spec);
  }
}

template <>
inline void CheckInputLayouts(const DeviceWorkspace *ws, const OpSpec &spec) {
  for (int i = 0; i < spec.NumRegularInput(); ++i) {
    if (ws->InputIsType<CPUBackend>(i)) {
      auto& input = ws->Input<CPUBackend>(i);
      CheckInputLayout(input, spec);
    } else if (ws->InputIsType<GPUBackend>(i)) {
      auto& input = ws->Input<GPUBackend>(i);
      CheckInputLayout(input, spec);
    } else {
      DALI_FAIL("Input has an unkown backend");
    }
  }
}

/**
 * @brief Baseclass for the basic unit of computation in the pipeline.
 *
 * OperatorBase defines the API used by the pipeline to execute operations.
 */
class DLL_PUBLIC OperatorBase {
 public:
  DLL_PUBLIC inline explicit OperatorBase(const OpSpec &spec) :
    spec_(spec), num_threads_(spec.GetArgument<int>("num_threads")),
    batch_size_(spec.GetArgument<int>("batch_size")),
    input_sets_(spec.GetArgument<int>("num_input_sets")),
    default_cuda_stream_priority_(spec.GetArgument<int>("default_cuda_stream_priority")) {
    DALI_ENFORCE(num_threads_ > 0, "Invalid value for argument num_threads.");
    DALI_ENFORCE(batch_size_ > 0, "Invalid value for argument batch_size.");
  }

  DLL_PUBLIC virtual inline ~OperatorBase() noexcept(false)
  {}

  /**
   * @brief Executes the operator on a single sample on the CPU.
   */
  DLL_PUBLIC virtual void Run(SampleWorkspace *ws) {
    DALI_FAIL("CPU execution is not implemented for this operator!");
  }

  /**
   * @brief Executes the operator on a batch of samples on the GPU.
   */
  DLL_PUBLIC virtual void Run(DeviceWorkspace *ws) {
    DALI_FAIL("GPU execution is not implemented for this operator!");
  }

  /**
   * @brief Used by operators interfacing with both CPU and GPU.
   */
  DLL_PUBLIC virtual void Run(MixedWorkspace *ws) {
    DALI_FAIL("Mixed execution is not implemented for this operator!");
  }

  /**
   * @brief Used by support operators (RNG etc.).
   */
  DLL_PUBLIC virtual void Run(SupportWorkspace *ws) {
    DALI_FAIL(name() + " is not a support operator!");
  }

  /**
   * @brief returns the name of the operator. By default returns
   * the name of the op as specified by the OpSpec it was constructed
   * from.
   */
  DLL_PUBLIC virtual string name() const {
    return spec_.name();
  }

  /**
   * @brief For reader Ops, returns the size of the dataset
   * For all other Ops, returns -1
   */
  DLL_PUBLIC virtual Index epoch_size() const {
    return -1;
  }

  DLL_PUBLIC int GetNumInputSets() const {
    return input_sets_;
  }

  DISABLE_COPY_MOVE_ASSIGN(OperatorBase);

 protected:
  const OpSpec spec_;
  int num_threads_;
  int batch_size_;
  int input_sets_;
  int default_cuda_stream_priority_;
};

#define USE_OPERATOR_MEMBERS()                       \
  using OperatorBase::spec_;                         \
  using OperatorBase::num_threads_;                  \
  using OperatorBase::batch_size_;                   \
  using OperatorBase::default_cuda_stream_priority_

/**
 * @brief Class defining an operator using specific backend.
 *
 * To create a custom operator, derive from this class, implement the
 * RunImpl method and register the operator using the DALI_REGISTER_OPERATOR
 * macro. To define meta-data about the op like the number of inputs
 * it takes, a docstring (for python), etc., use the DALI_OPERATOR_SCHEMA,
 * macro. The op can then be added to a pipeline through its registered
 * name (the first arg to the registration macro).
 */
template <typename Backend>
class Operator : public OperatorBase {
 public:
  inline explicit Operator(const OpSpec &spec) :
    OperatorBase(spec),
    sequences_allowed_(SchemaRegistry::GetSchema(spec.name()).AllowsSequences())
  {}

  inline ~Operator() noexcept(false) override
  {}

  using OperatorBase::Run;
  void Run(Workspace<Backend> *ws) override {
    std::vector<std::vector<int>> seq_sizes;
    if (std::is_same<Backend, GPUBackend>::value) {
        if (sequences_allowed_) {
          Flatten(ws);
        }
    }
    CheckInputLayouts(ws, spec_);
    SetupSharedSampleParams(ws);
    for (int i = 0; i < input_sets_; ++i) {
      if (std::is_same<Backend, GPUBackend>::value) {
        // Before we start working on the next input set, we need
        // to wait until the last one is finished. Otherwise for some ops
        // we risk overwriting data used by the kernel called for previous
        // image. Doing it for all ops is a compromise between performance
        // (which should not be greatly affected) and robustness (guarding
        // against this potential problem for newly added ops)
        SyncHelper(i, ws);
      }
      RunImpl(ws, i);
    }
    if (std::is_same<Backend, GPUBackend>::value) {
      if (sequences_allowed_) {
        Unflatten(ws);
      }
    }
  }

  /**
   * @brief Shared param setup
   */
  virtual void SetupSharedSampleParams(Workspace<Backend> *ws) {}

  /**
   * @brief Implementation of the operator - to be
   * implemented by derived ops.
   */
  virtual void RunImpl(Workspace<Backend> *ws, int idx = 0) = 0;

  int SequenceSize(int idx = 0) {
    DALI_ENFORCE(sequences_allowed_,
      "This operator is not implemented for sequences. "
      "Use AllowSequences() is OpSchema to enable it.");
    DALI_ENFORCE_VALID_INDEX(idx, seq_sizes_.size());
    return std::max(1, seq_sizes_[idx]);
  }

 private:
  // SINFAE for Run is not possible as we want it to be virtual
  template <typename B = Backend>
  typename std::enable_if<std::is_same<B, GPUBackend>::value>::type
  SyncHelper(int i, Workspace<B> *ws) {
    if (i != 0) {
        CUDA_CALL(cudaStreamSynchronize(ws->stream()));
    }
  }

  template <typename B = Backend>
  typename std::enable_if<!std::is_same<B, GPUBackend>::value>::type
  SyncHelper(int /*unused*/, Workspace<B> */*unused*/) {}

  template <typename B = Backend>
  typename std::enable_if<std::is_same<B, GPUBackend>::value>::type
  Flatten(Workspace<Backend> *ws) {
    seq_sizes_.clear();
    seq_sizes_.resize(input_sets_, 0);
    for (int i = 0; i < input_sets_; ++i) {
      auto &input = ws->template MutableInput<Backend>(i);
      const std::vector<Dims>& old_shapes = input.shape();
      DALITensorLayout layout = input.GetLayout();
      if (IsSequence(layout)) {
        // size of seq is the first dim in each tensor
        seq_sizes_[i] = old_shapes[0][0];

        std::vector<Dims> new_shapes;
        for (const Dims &old_shape : old_shapes) {
          // batch of sequences of different size not implemented
          DALI_ENFORCE(seq_sizes_[i] == old_shape[0],
            "Operator " + spec_.name() + " expects a batch of sequences of same length.");
          // getting only the need 3 dimensions
          Dims new_shape(old_shape.begin() + 1, old_shape.end());
          for (int s = 0; s < seq_sizes_[i]; ++s) {
            new_shapes.emplace_back(new_shape);
          }
        }
        input.Resize(new_shapes);
        input.SetLayout(GetElementLayout(input.GetLayout()));
      }
    }
  }

  template <typename B = Backend>
  typename std::enable_if<!std::is_same<B, GPUBackend>::value>::type
  Flatten(Workspace<B> */*unused*/) {}


  template <typename B = Backend>
  typename std::enable_if<std::is_same<B, GPUBackend>::value>::type
  Unflatten(Workspace<Backend> *ws) {
    for (int idx = 0; idx < input_sets_; ++idx) {
      CUDA_CALL(cudaStreamSynchronize(ws->stream()));
      if (seq_sizes_[idx] > 0) {
        auto &input = ws->template MutableInput<Backend>(idx);
        auto &output = ws->template Output<Backend>(idx);
        const std::vector<Dims>& old_shapes_input = input.shape();
        const std::vector<Dims>& old_shapes_output = output.shape();
        std::vector<Dims> new_shapes_input;
        std::vector<Dims> new_shapes_output;
        for (unsigned int i = 0; i < old_shapes_input.size(); i += seq_sizes_[idx]) {
          {
            Dims shape_input;
            shape_input.reserve(old_shapes_input[i].size() + 1);
            shape_input.push_back(static_cast<Index>(seq_sizes_[idx]));
            shape_input.insert(shape_input.end(),
                               old_shapes_input[i].begin(),
                               old_shapes_input[i].end());
            new_shapes_input.push_back(std::move(shape_input));
          }
          {
            Dims shape_output;
            shape_output.reserve(old_shapes_output[i].size() + 1);
            shape_output.push_back(static_cast<Index>(seq_sizes_[idx]));
            shape_output.insert(shape_output.end(),
                                old_shapes_output[i].begin(),
                                old_shapes_output[i].end());
            new_shapes_output.push_back(std::move(shape_output));
          }
        }
        input.Resize(new_shapes_input);
        output.Resize(new_shapes_output);
        input.SetLayout(GetSequenceLayout(input.GetLayout()));
        output.SetLayout(GetSequenceLayout(output.GetLayout()));
      }
    }
  }

  template <typename B = Backend>
  typename std::enable_if<!std::is_same<B, GPUBackend>::value>::type
  Unflatten(Workspace<B> */*unused*/) {}

  bool sequences_allowed_;
  // store size of each sequence for each input set
  std::vector<int> seq_sizes_;
};

template<>
class Operator<MixedBackend> : public OperatorBase {
 public:
  inline explicit Operator(const OpSpec &spec) :
    OperatorBase(spec)
  {}

  inline ~Operator() noexcept(false) override
  {}

  using OperatorBase::Run;
  void Run(MixedWorkspace *ws) override = 0;

  virtual void SetupSharedSampleParams(MixedWorkspace *ws) {}
};

// Create registries for CPU & GPU Operators
DALI_DECLARE_OPTYPE_REGISTRY(CPUOperator, OperatorBase);
DALI_DECLARE_OPTYPE_REGISTRY(GPUOperator, OperatorBase);
DALI_DECLARE_OPTYPE_REGISTRY(MixedOperator, OperatorBase);
DALI_DECLARE_OPTYPE_REGISTRY(SupportOperator, OperatorBase);

// Must be called from .cc or .cu file
#define DALI_REGISTER_OPERATOR(OpName, OpType, device)          \
  int DALI_OPERATOR_SCHEMA_REQUIRED_FOR_##OpName();             \
  static int ANONYMIZE_VARIABLE(OpName) =                       \
    DALI_OPERATOR_SCHEMA_REQUIRED_FOR_##OpName();               \
  DALI_DEFINE_OPTYPE_REGISTERER(OpName, OpType,                 \
      device##Operator, ::dali::OperatorBase, #device)

class ResizeParamDescr;

void DataDependentSetupCPU(const Tensor<CPUBackend> &input, Tensor<CPUBackend> &output,
                           const char *pOpName = NULL,
                           const uint8 **pInRaster = NULL, uint8 **ppOutRaster = NULL,
                           vector<DALISize> *pSizes = NULL, const DALISize *out_size = NULL);
bool DataDependentSetupGPU(const TensorList<GPUBackend> &input, TensorList<GPUBackend> &output,
                           size_t batch_size, bool reshapeBatch = false,
                           vector<const uint8 *> *iPtrs = NULL, vector<uint8 *> *oPtrs = NULL,
                           vector<DALISize> *pSizes = NULL, ResizeParamDescr *pResizeParam = NULL);
void CollectPointersForExecution(size_t batch_size,
                                 const TensorList<GPUBackend> &input, vector<const uint8 *> *inPtrs,
                                 TensorList<GPUBackend> &output, vector<uint8 *> *outPtrs);

DLL_PUBLIC std::unique_ptr<OperatorBase> InstantiateOperator(const OpSpec &spec);

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_OPERATOR_H_
