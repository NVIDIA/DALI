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

#ifndef DALI_PIPELINE_OPERATOR_OPERATOR_H_
#define DALI_PIPELINE_OPERATOR_OPERATOR_H_

#include <any>
#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <unordered_map>

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
#include "dali/pipeline/operator/checkpointing/op_checkpoint.h"
#include "dali/pipeline/util/batch_utils.h"
#include "dali/pipeline/workspace/workspace.h"
#include "dali/pipeline/workspace/sample_workspace.h"
#include "dali/pipeline/util/thread_pool.h"

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

  DLL_PUBLIC virtual bool Setup(std::vector<OutputDesc> &output_desc, const Workspace &ws) = 0;


  /**
   * @brief If Operator can infer the output shapes it means that its output would use a single
   * underlying allocation, especially for CPU TensorList will use contiguous mode.
   */
  DLL_PUBLIC virtual bool CanInferOutputs() const {
    return false;
  }

  /**
   * @brief Executes the operator on a batch of samples.
   */
  DLL_PUBLIC virtual void Run(Workspace &ws) = 0;

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
      return *std::any_cast<T *>(diagnostics_.at(name));
    } catch (std::bad_any_cast &e) {
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
    if (!diagnostics_.emplace(std::move(name), val).second) {
      DALI_FAIL("Diagnostic with given name already exists");
    }
  }

  /**
   * @brief Saves operator state into a checkpoint.
   *
   * Is called exactly once per epoch.
  */
  virtual void SaveState(OpCheckpoint &cpt, std::optional<cudaStream_t> stream) {
    DALI_FAIL("Checkpointing is not implemented for this operator.");
  }

  /**
   * @brief Restores operator state from checkpoint.
   *
   * Passed OpCheckpoint should have the host access order.
   *
   * Implementation can be blocking, as the performance is not critical.
  */
  virtual void RestoreState(const OpCheckpoint &cpt) {
    DALI_FAIL("Checkpointing is not implemented for this operator.");
  }

  /**
   * @brief Serializes the passed OpCheckpoint, containing state saved by this operator.
   *
   * Passed OpCheckpoint should have the host access order.
  */
  virtual std::string SerializeCheckpoint(const OpCheckpoint &cpt) const {
    DALI_FAIL("Checkpointing is not implemented for this operator.");
  }

  /**
   * @brief Deserializes serialized operator state and sets it in the passed OpCheckpoint.
  */
  virtual void DeserializeCheckpoint(OpCheckpoint &cpt, const std::string &data) const {
    DALI_FAIL("Checkpointing is not implemented for this operator.");
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
  DLL_PUBLIC void EnforceUniformInputBatchSize(const Workspace &ws) const;

  template <typename Backend>
  DLL_PUBLIC void EnforceUniformOutputBatchSize(const Workspace &ws) const;

  const OpSpec spec_;
  int num_threads_;
  int max_batch_size_;
  int default_cuda_stream_priority_;

  std::unordered_map<std::string, std::any> diagnostics_;
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

inline TensorLayout GetInputLayout(const Workspace &ws, int i) {
  if (ws.InputIsType<CPUBackend>(i)) {
    auto &in = ws.Input<CPUBackend>(i);
    return in.GetLayout();
  }

  assert(ws.InputIsType<GPUBackend>(i));
  auto &in = ws.Input<GPUBackend>(i);
  return in.GetLayout();
}

inline TensorLayout GetOutputLayout(const Workspace &ws, int i) {
  if (ws.OutputIsType<CPUBackend>(i)) {
    auto &out = ws.Output<CPUBackend>(i);
    return out.GetLayout();
  }

  assert(ws.OutputIsType<GPUBackend>(i));
  auto &out = ws.Output<GPUBackend>(i);
  return out.GetLayout();
}

template <>
class Operator<CPUBackend> : public OperatorBase {
 public:
  inline explicit Operator(const OpSpec &spec) : OperatorBase(spec) {}

  inline ~Operator() override {}

  using OperatorBase::Setup;
  using OperatorBase::Run;

  bool Setup(std::vector<OutputDesc> &output_desc, const Workspace &ws) override {
    EnforceUniformInputBatchSize<CPUBackend>(ws);
    CheckInputLayouts(ws, spec_);
    return SetupImpl(output_desc, ws);
  }

  void Run(Workspace &ws) override {
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
  virtual bool SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) = 0;

  /**
   * @brief Legacy implementation of CPU operator using per-sample approach
   *
   * Usage of this API is deprecated. For CPU Ops `void RunImpl(Workspace &ws)`
   * should be overridden instead.
   */
  virtual void RunImpl(SampleWorkspace &ws) {}

  /**
   * @brief Implementation of the operator - to be implemented by derived ops.
   */
  virtual void RunImpl(Workspace &ws) {
    // This is implemented, as a default, using the RunImpl that accepts SampleWorkspace,
    // allowing for fallback to old per-sample implementations.

    auto curr_batch_size = ws.NumInput() > 0 ? ws.GetInputBatchSize(0) : max_batch_size_;
    for (int i = 0; i < ws.NumOutput(); i++) {
      auto &output = ws.Output<CPUBackend>(i);
      output.SetSize(curr_batch_size);
    }
    auto &thread_pool = ws.GetThreadPool();
    for (int data_idx = 0; data_idx < curr_batch_size; ++data_idx) {
      thread_pool.AddWork([this, &ws, data_idx](int tid) {
        SampleWorkspace sample;
        MakeSampleView(sample, ws, data_idx, tid);
        this->SetupSharedSampleParams(sample);
        this->RunImpl(sample);
      }, -data_idx);  // -data_idx for FIFO order
    }
    // Run all tasks and wait for them to finish
    thread_pool.RunAll();
    // Propagate metadata from individual samples to the whole batch as working with SampleWorkspace
    // breaks metadata consistency - it sets it only to samples
    FixBatchPropertiesConsistency(ws, CanInferOutputs());
  }

  /**
   * @brief Shared param setup. Legacy implementation for per-sample approach
   *
   * Usage of this API is deprecated. For CPU Ops `void SetupImpl(Workspace &ws)`
   * should be used instead.
   */
  virtual void SetupSharedSampleParams(SampleWorkspace &ws) {}

  /**
   * @brief Shared param setup.
   *
   * Usage of this API is deprecated. For CPU Ops `void SSetupImpl(Workspace &ws)`
   * should be used instead.
   */
  virtual void SetupSharedSampleParams(Workspace &ws) {}
};

template <>
class Operator<GPUBackend> : public OperatorBase {
 public:
  inline explicit Operator(const OpSpec &spec) : OperatorBase(spec) {}

  inline ~Operator() override {}

  using OperatorBase::Setup;
  using OperatorBase::Run;

  bool Setup(std::vector<OutputDesc> &output_desc, const Workspace &ws) override {
    EnforceUniformInputBatchSize<GPUBackend>(ws);
    CheckInputLayouts(ws, spec_);
    return SetupImpl(output_desc, ws);
  }

  void Run(Workspace &ws) override {
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
  virtual bool SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) = 0;

  /**
   * @brief Implementation of the operator - to be
   * implemented by derived ops.
   */
  virtual void RunImpl(Workspace &ws) = 0;

  /**
   * @brief Shared param setup
   */
  virtual void SetupSharedSampleParams(Workspace &ws) {}
};

template <>
class Operator<MixedBackend> : public OperatorBase {
 public:
  inline explicit Operator(const OpSpec &spec) : OperatorBase(spec) {}

  inline ~Operator() override {}

  using OperatorBase::Setup;
  using OperatorBase::Run;

  bool Setup(std::vector<OutputDesc> &output_desc, const Workspace &ws) override {
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
  virtual bool SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) = 0;

  void Run(Workspace &ws) override = 0;

  /**
   * @brief Shared param setup
   */
  virtual void SetupSharedSampleParams(Workspace &ws) {}
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
