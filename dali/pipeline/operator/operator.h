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

#ifndef DALI_PIPELINE_OPERATOR_OPERATOR_H_
#define DALI_PIPELINE_OPERATOR_OPERATOR_H_

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "dali/core/common.h"
#include "dali/core/format.h"
#include "dali/core/error_handling.h"
#include "dali/core/tensor_shape.h"
#include "dali/core/tensor_shape_print.h"
#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/operator/op_schema.h"
#include "dali/pipeline/operator/op_spec.h"
#include "dali/pipeline/operator/operator_factory.h"
#include "dali/pipeline/util/backend2workspace_map.h"
#include "dali/pipeline/workspace/device_workspace.h"
#include "dali/pipeline/workspace/sample_workspace.h"

namespace dali {

/**
 * Names for most commonly used arguments, to keep consistency between arg naming amongst operators.
 */
namespace arg_names {
const std::string kSeed = "seed";            // NOLINT
const std::string kDtype = "dtype";          // NOLINT
}  // namespace arg_names

/**
 * @brief Gets a data layout for the input at given index
 *
 * If the layout is explicitly defined, it's verified against the schema.
 * If the layout is not specified, a default one is taken from the schema
 * based on the input's dimensionality.
 */
template <typename Workspace>
inline TensorLayout GetInputLayout(const Workspace &ws, const OpSchema &schema, int index) {
  if (ws.template InputIsType<CPUBackend>(index)) {
    auto &input = ws.template InputRef<CPUBackend>(index);
    return schema.GetInputLayout(index, input.shape().sample_dim(), input.GetLayout());
  } else if (ws.template InputIsType<GPUBackend>(index)) {
    auto &input = ws.template InputRef<GPUBackend>(index);
    return schema.GetInputLayout(index, input.shape().sample_dim(), input.GetLayout());
  } else {
    DALI_FAIL("Input " + std::to_string(index) + " has an unknown backend");
  }
}

/**
 * @brief Verifies that the inputs in the workspace satisfy the layout
 *        constraints imposed by the schema.
 */
template <typename Workspace>
inline void CheckInputLayouts(const Workspace &ws, const OpSpec &spec) {
  if (spec.NumRegularInput() > 0) {
    auto &schema = spec.GetSchema();
    for (int i = 0; i < spec.NumRegularInput(); ++i) {
      (void)GetInputLayout(ws, schema, i);
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
        default_cuda_stream_priority_(spec.GetArgument<int>("default_cuda_stream_priority")) {
    DALI_ENFORCE(num_threads_ > 0, "Invalid value for argument num_threads.");
    DALI_ENFORCE(batch_size_ > 0, "Invalid value for argument batch_size.");
  }

  DLL_PUBLIC virtual inline ~OperatorBase() {}

  DLL_PUBLIC virtual bool Setup(std::vector<OutputDesc> &output_desc, const HostWorkspace &ws) {
    DALI_FAIL("CPU execution is not implemented for this operator!");
  }

  DLL_PUBLIC virtual bool Setup(std::vector<OutputDesc> &output_desc, const DeviceWorkspace &ws) {
    DALI_FAIL("GPU execution is not implemented for this operator!");
  }

  DLL_PUBLIC virtual bool Setup(std::vector<OutputDesc> &output_desc, const MixedWorkspace &ws) {
    DALI_FAIL("Mixed execution is not implemented for this operator!");
  }

  /**
   * @brief If Operator can infer the output shapes it means that its output would use a single
   * underlying allocation, especailly for CPU TensorVector will use contiguous mode.
   */
  DLL_PUBLIC virtual bool CanInferOutputs() const {
    return false;
  }

  /**
   * @brief Executes the operator on a batch of samples on the CPU.
   */
  DLL_PUBLIC virtual void Run(HostWorkspace &ws) {
    DALI_FAIL("CPU execution is not implemented for this operator!");
  }

  /**
   * @brief Executes the operator on a batch of samples on the GPU.
   */
  DLL_PUBLIC virtual void Run(DeviceWorkspace &ws) {
    DALI_FAIL("GPU execution is not implemented for this operator!");
  }

  /**
   * @brief Used by operators interfacing with both CPU and GPU.
   */
  DLL_PUBLIC virtual void Run(MixedWorkspace &ws) {
    DALI_FAIL("Mixed execution is not implemented for this operator!");
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

  template <typename Workspace>
  TensorLayout InputLayout(const Workspace &ws, int index) const {
    return GetInputLayout(ws, spec_.GetSchema(), index);
  }

  DLL_PUBLIC bool CanBePruned() const {
    const auto &schema = spec_.GetSchema();
    return !spec_.GetArgument<bool>("preserve") && !schema.IsNoPrune();
  }

  DISABLE_COPY_MOVE_ASSIGN(OperatorBase);

 protected:
  /**
   * @brief Fill output vector with per-sample argument values.
   *
   * If there's a tensor argument, then the values are copied from it;
   * otherwise scalar value is replicated.
   *
   * @tparam T Type of the Argument
   * @param output Container for the data. This function will reallocate it.
   * @param argument_name name of the Argument
   * @param ws
   */
  template<typename T>
  void GetPerSampleArgument(std::vector<T> &output, const std::string &argument_name,
                            const ArgumentWorkspace &ws) {
    if (spec_.HasTensorArgument(argument_name)) {
      const auto &arg = ws.ArgumentInput(argument_name);
      decltype(auto) shape = arg.shape();
      int N = shape.num_samples();
      if (N == 1) {
        bool is_valid_shape = shape.tensor_shape(0) == TensorShape<1>{batch_size_};

        DALI_ENFORCE(is_valid_shape,
          make_string("`", argument_name, "` must be a 1xN or Nx1 (N = ", batch_size_,
                     ") tensor list. Got: ", shape));

        output.resize(batch_size_);
        auto *data = arg[0].template data<T>();

        for (int i = 0; i < batch_size_; i++) {
          output[i] = data[i];
        }
      } else {
        bool is_valid_shape = N == batch_size_ &&
                              is_uniform(shape) &&
                              shape.tensor_shape(0) == TensorShape<1>{1};
        DALI_ENFORCE(is_valid_shape,
          make_string("`", argument_name, "` must be a 1xN or Nx1 (N = ", batch_size_,
                     ") tensor list. Got: ", shape));

        output.resize(batch_size_);
        for (int i = 0; i < batch_size_; i++) {
          output[i] = arg[i].template data<T>()[0];
        }
      }
    } else {
      output.resize(batch_size_, spec_.template GetArgument<T>(argument_name));
    }
    assert(output.size() == static_cast<size_t>(batch_size_));
  }

  const OpSpec spec_;
  int num_threads_;
  int batch_size_;
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
class Operator<CPUBackend> : public OperatorBase {
 public:
  inline explicit Operator(const OpSpec &spec) : OperatorBase(spec) {}

  inline ~Operator() override {}

  using OperatorBase::Setup;
  using OperatorBase::Run;

  bool Setup(std::vector<OutputDesc> &output_desc, const HostWorkspace &ws) override {
    return SetupImpl(output_desc, ws);
  }

  void Run(HostWorkspace &ws) override {
    CheckInputLayouts(ws, spec_);
    SetupSharedSampleParams(ws);
    RunImpl(ws);
    ws.GetThreadPool().WaitForWork();
  }

  /**
   * @brief Setup of the operator - to be implemented by derived op.
   *
   * @param output_desc describe the shape and type of the outputs (for the whole batch)
   * @param ws
   * @return true iff the operator specified the output shape and type
   */
  virtual bool SetupImpl(std::vector<OutputDesc> &output_desc, const HostWorkspace &ws) = 0;

  /**
   * @brief Legacy implementation of CPU operator using per-sample approach
   *
   * Usage of this API is deprecated. For CPU Ops `void RunImpl(HostWorkspace &ws)`
   * should be overridden instead.
   */
  virtual void RunImpl(SampleWorkspace &ws) {}

  /**
   * @brief Implementation of the operator - to be implemented by derived ops.
   */
  virtual void RunImpl(HostWorkspace &ws) {
    // This is implemented, as a default, using the RunImpl that accepts SampleWorkspace,
    // allowing for fallback to old per-sample implementations.

    for (int data_idx = 0; data_idx < batch_size_; ++data_idx) {
      auto &thread_pool = ws.GetThreadPool();
      thread_pool.DoWorkWithID([this, &ws, data_idx](int tid) {
        SampleWorkspace sample;
        ws.GetSample(&sample, data_idx, tid);
        this->SetupSharedSampleParams(sample);
        this->RunImpl(sample);
      });
    }
  }

  /**
   * @brief Shared param setup. Legacy implementation for per-sample approach
   *
   * Usage of this API is deprecated. For CPU Ops `void SetupSharedSampleParams(HostWorkspace &ws)`
   * should be used instead.
   */
  virtual void SetupSharedSampleParams(SampleWorkspace &ws) {}

  /**
   * @brief Shared param setup
   */
  virtual void SetupSharedSampleParams(HostWorkspace &ws) {}
};

template <>
class Operator<GPUBackend> : public OperatorBase {
 public:
  inline explicit Operator(const OpSpec &spec) : OperatorBase(spec) {}

  inline ~Operator() override {}

  using OperatorBase::Setup;
  using OperatorBase::Run;

  bool Setup(std::vector<OutputDesc> &output_desc, const DeviceWorkspace &ws) override {
    return SetupImpl(output_desc, ws);
  }

  void Run(DeviceWorkspace &ws) override {
    CheckInputLayouts(ws, spec_);
    SetupSharedSampleParams(ws);
    RunImpl(ws);
  }

  /**
   * @brief Setup of the operator - to be implemented by derived op.
   *
   * @param output_desc describe the shape and type of the outputs (for the whole batch)
   * @param ws
   * @return true iff the operator specified the output shape and type
   */
  virtual bool SetupImpl(std::vector<OutputDesc> &output_desc, const DeviceWorkspace &ws) = 0;

  /**
   * @brief Implementation of the operator - to be
   * implemented by derived ops.
   */
  virtual void RunImpl(DeviceWorkspace &ws) = 0;

  /**
   * @brief Shared param setup
   */
  virtual void SetupSharedSampleParams(DeviceWorkspace &ws) {}
};

template <>
class Operator<MixedBackend> : public OperatorBase {
 public:
  inline explicit Operator(const OpSpec &spec) : OperatorBase(spec) {}

  inline ~Operator() override {}

  using OperatorBase::Setup;
  using OperatorBase::Run;

  bool Setup(std::vector<OutputDesc> &output_desc, const MixedWorkspace &ws) override {
    return SetupImpl(output_desc, ws);
  }

  /**
   * @brief Setup of the operator - to be implemented by derived op.
   *
   * @param output_desc describe the shape and type of the outputs (for the whole batch)
   * @param ws
   * @return true iff the operator specified the output shape and type
   */
  virtual bool SetupImpl(std::vector<OutputDesc> &output_desc, const MixedWorkspace &ws) = 0;

  void Run(MixedWorkspace &ws) override = 0;

  virtual void SetupSharedSampleParams(MixedWorkspace &ws) {}
};

// Create registries for CPU & GPU Operators
DALI_DECLARE_OPTYPE_REGISTRY(CPUOperator, OperatorBase);
DALI_DECLARE_OPTYPE_REGISTRY(GPUOperator, OperatorBase);
DALI_DECLARE_OPTYPE_REGISTRY(MixedOperator, OperatorBase);

// Must be called from .cc or .cu file
#define DALI_REGISTER_OPERATOR(OpName, OpType, device)                                  \
  int DALI_OPERATOR_SCHEMA_REQUIRED_FOR_##OpName();                                     \
  static int ANONYMIZE_VARIABLE(OpName) = DALI_OPERATOR_SCHEMA_REQUIRED_FOR_##OpName(); \
  DALI_DEFINE_OPTYPE_REGISTERER(OpName, OpType, device##Operator, ::dali::OperatorBase, #device)


DLL_PUBLIC std::unique_ptr<OperatorBase> InstantiateOperator(const OpSpec &spec);

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATOR_OPERATOR_H_
