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
#include <unordered_map>

#include "dali/core/any.h"
#include "dali/core/common.h"
#include "dali/core/format.h"
#include "dali/core/error_handling.h"
#include "dali/core/tensor_shape.h"
#include "dali/core/tensor_shape_print.h"
#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/operator/common.h"
#include "dali/pipeline/operator/op_schema.h"
#include "dali/pipeline/operator/op_spec.h"
#include "dali/pipeline/operator/operator_factory.h"
#include "dali/pipeline/util/backend2workspace_map.h"
#include "dali/pipeline/workspace/device_workspace.h"
#include "dali/pipeline/workspace/sample_workspace.h"

namespace dali {

struct DLL_PUBLIC ReaderMeta {
  Index epoch_size = -1;          // raw epoch size
  Index epoch_size_padded = -1;   // epoch size with the padding at the end
  int number_of_shards = -1;      // number of shards
  int shard_id = -1;              // shard id of given reader
  int pad_last_batch = -1;        // if given reader should pad last batch
  int stick_to_shard = -1;        // if given reader should stick to its shard

  DLL_PUBLIC operator bool() const {
    return epoch_size != -1 && epoch_size_padded != -1 && number_of_shards != -1 &&
           shard_id != -1 && pad_last_batch != -1 && stick_to_shard != -1;
  }
};

/**
 * Names for most commonly used arguments, to keep consistency between arg naming amongst operators.
 */
namespace arg_names {
const std::string kSeed = "seed";            // NOLINT
const std::string kDtype = "dtype";          // NOLINT
}  // namespace arg_names

/**
 * @brief Verifies that the inputs in the workspace satisfy the layout
 *        constraints imposed by the schema.
 */
template <typename Workspace>
inline void CheckInputLayouts(const Workspace &ws, const OpSpec &spec) {
  auto &schema = spec.GetSchema();
  for (int i = 0; i < spec.NumRegularInput(); ++i) {
    if (ws.template InputIsType<CPUBackend>(i)) {
      auto &input = ws.template InputRef<CPUBackend>(i);
      (void) schema.GetInputLayout(i, input.shape().sample_dim(), input.GetLayout());
    } else if (ws.template InputIsType<GPUBackend>(i)) {
      auto &input = ws.template InputRef<GPUBackend>(i);
      (void) schema.GetInputLayout(i, input.shape().sample_dim(), input.GetLayout());
    } else {
      DALI_FAIL(make_string("Input ", i, " has an unknown backend"));
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
        max_batch_size_(spec.GetArgument<int>("max_batch_size")),
        default_cuda_stream_priority_(spec.GetArgument<int>("default_cuda_stream_priority")) {
    DALI_ENFORCE(num_threads_ > 0, "Invalid value for argument num_threads.");
    DALI_ENFORCE(max_batch_size_ > 0, "Invalid value for argument max_batch_size.");
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
   * underlying allocation, especially for CPU TensorVector will use contiguous mode.
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
   * @brief For reader Ops, returns the metadata of the reader and dataset,
   * See ReaderMeta strucutre for the data returned
   * For all other Ops, returns -1
   */

  DLL_PUBLIC virtual ReaderMeta GetReaderMeta() const {
    return {};
  }

  DLL_PUBLIC const OpSpec& GetSpec() const {
    return spec_;
  }

  DLL_PUBLIC bool CanBePruned() const {
    const auto &schema = spec_.GetSchema();
    return !spec_.GetArgument<bool>("preserve") && !schema.IsNoPrune();
  }

  DISABLE_COPY_MOVE_ASSIGN(OperatorBase);


  template<typename T>
  T GetDiagnostic(const std::string &name) const {
    try {
      return *any_cast<T *>(diagnostics_.at(name));
    } catch (dali::bad_any_cast &e) {
      DALI_FAIL(make_string("Specified type of diagnostic parameter (`", typeid(T).name(),
                            "`) doesn't match the type that this parameter was registered with. ",
                            e.what()));
    } catch (std::out_of_range &e) {
      DALI_FAIL(make_string("Diagnostic parameter with specified name (`", name,
                            "`) hasn't been registered. ", e.what()));
    } catch (...) {
      DALI_FAIL("Error occured when reading diagnostic parameter.");
    }
  }

  template<typename T>
  void RegisterDiagnostic(std::string name, T *val) {
    using namespace std;  // NOLINT
    static_assert(is_arithmetic_or_half<remove_reference_t<T>>::value || is_enum<T>::value,
                  "The eligible diagnostic entry types are arithmetic types or enum");
    if (!diagnostics_.emplace(move(name), val).second) {
      DALI_FAIL("Diagnostic with given name already exists");
    }
  }


 protected:
  /**
   * @brief Fill output vector with per-sample argument values.
   *
   * If there's a tensor argument, then the values are copied from it;
   * otherwise scalar value is replicated.
   *
   * @tparam T Type of the Argument
   * @param output        Container for the data. This function will reallocate it.
   * @param argument_name name of the Argument
   * @param ws            workspace object, from which ArgumentInputs are taken
   * @param batch_size    number of samples in the batch
   */
  template<typename T>
  void GetPerSampleArgument(std::vector<T> &output, const std::string &argument_name,
                            const ArgumentWorkspace &ws, int batch_size) {
    DALI_ENFORCE(batch_size > 0, "Default batch size (-1) is not supported anymore");
    dali::GetPerSampleArgument(output, argument_name, spec_, ws, batch_size);
  }

  // TODO(mszolucha): remove these two to allow i2i variable batch size, when all ops are ready
  template <typename Backend>
  DLL_PUBLIC void EnforceUniformInputBatchSize(const workspace_t<Backend> &ws) const;

  template <typename Backend>
  DLL_PUBLIC void EnforceUniformOutputBatchSize(const workspace_t<Backend> &ws) const;

  const OpSpec spec_;
  int num_threads_;
  int max_batch_size_;
  int default_cuda_stream_priority_;

  std::unordered_map<std::string, any> diagnostics_;
};

#define USE_OPERATOR_MEMBERS()                       \
  using OperatorBase::spec_;                         \
  using OperatorBase::num_threads_;                  \
  using OperatorBase::max_batch_size_;               \
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

template <typename Workspace>
TensorLayout GetInputLayout(Workspace& ws, int i) {
  if (ws.template InputIsType<CPUBackend>(i)) {
    auto &in = ws.template InputRef<CPUBackend>(i);
    return in.GetLayout();
  }

  assert(ws.template InputIsType<GPUBackend>(i));
  auto &in = ws.template InputRef<GPUBackend>(i);
  return in.GetLayout();
}

template <typename Workspace>
TensorLayout GetOutputLayout(Workspace &ws, int i) {
  if (ws.template OutputIsType<CPUBackend>(i)) {
    auto &out = ws.template OutputRef<CPUBackend>(i);
    return out.GetLayout();
  }

  assert(ws.template OutputIsType<GPUBackend>(i));
  auto &out = ws.template OutputRef<GPUBackend>(i);
  return out.GetLayout();
}

template <>
class Operator<CPUBackend> : public OperatorBase {
 public:
  inline explicit Operator(const OpSpec &spec) : OperatorBase(spec) {}

  inline ~Operator() override {}

  using OperatorBase::Setup;
  using OperatorBase::Run;

  bool Setup(std::vector<OutputDesc> &output_desc, const HostWorkspace &ws) override {
    EnforceUniformInputBatchSize<CPUBackend>(ws);
    return SetupImpl(output_desc, ws);
  }

  void Run(HostWorkspace &ws) override {
    CheckInputLayouts(ws, spec_);
    SetupSharedSampleParams(ws);
    RunImpl(ws);
    ws.GetThreadPool().WaitForWork();
    EnforceUniformOutputBatchSize<CPUBackend>(ws);
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

    auto curr_batch_size = ws.NumInput() > 0 ? ws.GetInputBatchSize(0) : max_batch_size_;
    for (int i = 0; i < ws.NumOutput(); i++) {
      auto &output = ws.OutputRef<CPUBackend>(i);
      output.SetSize(curr_batch_size);
    }
    auto &thread_pool = ws.GetThreadPool();
    for (int data_idx = 0; data_idx < curr_batch_size; ++data_idx) {
      thread_pool.AddWork([this, &ws, data_idx](int tid) {
        SampleWorkspace sample;
        ws.GetSample(&sample, data_idx, tid);
        this->SetupSharedSampleParams(sample);
        this->RunImpl(sample);
      }, -data_idx);  // -data_idx for FIFO order
    }
    thread_pool.RunAll();
  }

  /**
   * @brief Shared param setup. Legacy implementation for per-sample approach
   *
   * Usage of this API is deprecated. For CPU Ops `void SetupImpl(HostWorkspace &ws)`
   * should be used instead.
   */
  virtual void SetupSharedSampleParams(SampleWorkspace &ws) {}

  /**
   * @brief Shared param setup.
   *
   * Usage of this API is deprecated. For CPU Ops `void SSetupImpl(HostWorkspace &ws)`
   * should be used instead.
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
    EnforceUniformInputBatchSize<GPUBackend>(ws);
    return SetupImpl(output_desc, ws);
  }

  void Run(DeviceWorkspace &ws) override {
    CheckInputLayouts(ws, spec_);
    SetupSharedSampleParams(ws);
    RunImpl(ws);
    EnforceUniformOutputBatchSize<GPUBackend>(ws);
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
    EnforceUniformInputBatchSize<MixedBackend>(ws);
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

  /**
   * @brief Shared param setup
   */
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
