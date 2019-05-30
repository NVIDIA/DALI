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

#include "dali/core/common.h"
#include "dali/core/error_handling.h"
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
  DLL_PUBLIC inline explicit OperatorBase(const OpSpec &spec)
      : spec_(spec),
        num_threads_(spec.GetArgument<int>("num_threads")),
        batch_size_(spec.GetArgument<int>("batch_size")),
        input_sets_(spec.GetArgument<int>("num_input_sets")),
        default_cuda_stream_priority_(spec.GetArgument<int>("default_cuda_stream_priority")) {
    DALI_ENFORCE(num_threads_ > 0, "Invalid value for argument num_threads.");
    DALI_ENFORCE(batch_size_ > 0, "Invalid value for argument batch_size.");
  }

  DLL_PUBLIC virtual inline ~OperatorBase() noexcept(false) {}

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
class Operator : public OperatorBase {};

template <>
class Operator<SupportBackend> : public OperatorBase {
 public:
  inline explicit Operator(const OpSpec &spec) : OperatorBase(spec) {}

  inline ~Operator() noexcept(false) override {}

  using OperatorBase::Run;
  void Run(SupportWorkspace *ws) override {
    CheckInputLayouts(ws, spec_);
    SetupSharedSampleParams(ws);
    for (int i = 0; i < input_sets_; ++i) {
      RunImpl(ws, i);
    }
  }

  /**
   * @brief Implementation of the operator - to be
   * implemented by derived ops.
   */
  virtual void RunImpl(SupportWorkspace *ws, int idx = 0) = 0;

  /**
   * @brief Shared param setup
   */
  virtual void SetupSharedSampleParams(SupportWorkspace *ws) {}
};

template <>
class Operator<CPUBackend> : public OperatorBase {
 public:
  inline explicit Operator(const OpSpec &spec) : OperatorBase(spec) {}

  inline ~Operator() noexcept(false) override {}

  using OperatorBase::Run;
  void Run(SampleWorkspace *ws) override {
    CheckInputLayouts(ws, spec_);
    SetupSharedSampleParams(ws);
    for (int i = 0; i < input_sets_; ++i) {
      RunImpl(ws, i);
    }
  }

  /**
   * @brief Legacy implementation of CPU operator using per-sample approach
   *
   * Usage of this API will be deprecated.
   */
  virtual void RunImpl(SampleWorkspace *ws, int idx = 0) {}

  /**
   * @brief Implementation of the operator - to be implemented by derived ops.
   */
  virtual void RunImpl(HostWorkspace *ws, int idx = 0) {
    DALI_ENFORCE(false, "Not implemented yet");
  }

  /**
   * @brief Shared param setup. Legacy implementation for per-sample approach
   *
   * Usage of this API will be deprecated.
   */
  virtual void SetupSharedSampleParams(SampleWorkspace *ws) {}

  /**
   * @brief Shared param setup
   */
  virtual void SetupSharedSampleParams(HostWorkspace *ws) {}
};

template <>
class Operator<GPUBackend> : public OperatorBase {
 public:
  inline explicit Operator(const OpSpec &spec) : OperatorBase(spec) {}

  inline ~Operator() noexcept(false) override {}

  using OperatorBase::Run;
  void Run(DeviceWorkspace *ws) override {
    CheckInputLayouts(ws, spec_);
    SetupSharedSampleParams(ws);
    for (int i = 0; i < input_sets_; ++i) {
      // Before we start working on the next input set, we need
      // to wait until the last one is finished. Otherwise for some ops
      // we risk overwriting data used by the kernel called for previous
      // image. Doing it for all ops is a compromise between performance
      // (which should not be greatly affected) and robustness (guarding
      // against this potential problem for newly added ops)
      SyncHelper(i, ws);
      RunImpl(ws, i);
    }
  }

  /**
   * @brief Implementation of the operator - to be
   * implemented by derived ops.
   */
  virtual void RunImpl(DeviceWorkspace *ws, int idx = 0) = 0;

  /**
   * @brief Shared param setup
   */
  virtual void SetupSharedSampleParams(DeviceWorkspace *ws) {}

 private:
  void SyncHelper(int i, DeviceWorkspace *ws) {
    if (i != 0) {
        CUDA_CALL(cudaStreamSynchronize(ws->stream()));
    }
  }
};

template<>
class Operator<MixedBackend> : public OperatorBase {
 public:
  inline explicit Operator(const OpSpec &spec) : OperatorBase(spec) {}

  inline ~Operator() noexcept(false) override {}

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
