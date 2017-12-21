// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
#ifndef NDLL_PIPELINE_OPERATOR_H_
#define NDLL_PIPELINE_OPERATOR_H_

#include <string>

#include "ndll/common.h"
#include "ndll/error_handling.h"
#include "ndll/pipeline/device_workspace.h"
#include "ndll/pipeline/data/backend.h"
#include "ndll/pipeline/operator_factory.h"
#include "ndll/pipeline/op_schema.h"
#include "ndll/pipeline/op_spec.h"
#include "ndll/pipeline/sample_workspace.h"

namespace ndll {

enum NDLLOpType {
  NDLL_GPU = 0,
  NDLL_CPU = 1,
  NDLL_INTERNAL = 2
};

/**
 * @brief Baseclass for the basic unit of computation in the pipeline.
 *
 * Operator defines the API used by the pipeline to execute operations.
 * To create a custom operator, derive from this class, implement the
 * RunPerSampleCPU / RunBatchedGPU methods as desired, and register
 * the operator using the macros NDLL_REGISTER_{CPU,GPU}_OPERATOR.
 * To define meta-data about the op like the min/max number of inputs
 * it takes, a doctstring (for python), etc., use the OPERATOR_SCHEMA,
 * macro. The op can then be added to a pipeline through its registered
 * name (the first arg to the registration macros).
 */
template <typename Backend>
class Operator {
 public:
  inline explicit Operator(const OpSpec &spec) :
    spec_(spec), num_threads_(spec.GetArgument<int>("num_threads", -1)),
    batch_size_(spec.GetArgument<int>("batch_size", -1)) {
    NDLL_ENFORCE(num_threads_ > 0, "Invalid value for argument num_threads.");
    NDLL_ENFORCE(batch_size_ > 0, "Invalid value for argument batch_size.");
  }

  virtual inline ~Operator() = default;

  /**
   * @brief Executes the operator on a single sample on the CPU.
   */
  inline void Run(SampleWorkspace *ws) {
#ifndef NDEBUG
    NDLL_ENFORCE_VALID_INDEX(ws->thread_idx(), num_threads_);
    NDLL_ENFORCE_VALID_INDEX(ws->data_idx(), batch_size_);
#endif
    RunPerSampleCPU(ws);
  }

  /**
   * @brief Executes the operator on a batch of samples on the GPU.
   */
  inline void Run(DeviceWorkspace *ws) {
    RunBatchedGPU(ws);
  }

  /**
   * @brief returns the name of the operator. By default returns
   * the name of the op as specified by the OpSpec it was constructed
   * from.
   */
  virtual string name() const {
    return spec_.name();
  }

  DISABLE_COPY_MOVE_ASSIGN(Operator);

 protected:
  /**
   * @brief Per image CPU computation of the operator to be
   * implemented by derived ops.
   */
  virtual inline void RunPerSampleCPU(SampleWorkspace *ws) {
    NDLL_FAIL("RunPerSampleCPU not implemented");
  }

  /**
   * @brief Batched GPU computation of the operator to be
   * implemented by derived ops.
   */
  virtual inline void RunBatchedGPU(DeviceWorkspace *ws) {
    NDLL_FAIL("RunBatchedGPU not implemented");
  }

  OpSpec spec_;
  int num_threads_;
  int batch_size_;
};

#define USE_OPERATOR_MEMBERS()                  \
  using Operator<Backend>::spec_;               \
  using Operator<Backend>::num_threads_;        \
  using Operator<Backend>::batch_size_

// Create registries for CPU & GPU Operators
NDLL_DECLARE_OPTYPE_REGISTRY(CPUOperator, Operator<CPUBackend>);
NDLL_DECLARE_OPTYPE_REGISTRY(GPUOperator, Operator<GPUBackend>);

// Must be called from .cc or .cu file
#define NDLL_REGISTER_CPU_OPERATOR(OpName, OpType)        \
  int OPERATOR_SCHEMA_REQUIRED_FOR_##OpName();            \
  static int ANONYMIZE_VARIABLE(OpName) =                 \
    OPERATOR_SCHEMA_REQUIRED_FOR_##OpName();              \
  NDLL_DEFINE_OPTYPE_REGISTERER(OpName, OpType,           \
      ndll::CPUOperator, ndll::Operator<CPUBackend>)

#define NDLL_REGISTER_GPU_OPERATOR(OpName, OpType)        \
  int OPERATOR_SCHEMA_REQUIRED_FOR_##OpName();            \
  static int ANONYMIZE_VARIABLE(OpName) =                 \
    OPERATOR_SCHEMA_REQUIRED_FOR_##OpName();              \
  NDLL_DEFINE_OPTYPE_REGISTERER(OpName, OpType,           \
      ndll::GPUOperator, ndll::Operator<GPUBackend>)

}  // namespace ndll

#endif  // NDLL_PIPELINE_OPERATOR_H_
