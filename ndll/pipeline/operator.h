// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#ifndef NDLL_PIPELINE_OPERATOR_H_
#define NDLL_PIPELINE_OPERATOR_H_

#include <string>
#include <vector>

#include "ndll/common.h"
#include "ndll/error_handling.h"
#include "ndll/pipeline/workspace/device_workspace.h"
#include "ndll/pipeline/ndll.pb.h"
#include "ndll/pipeline/data/backend.h"
#include "ndll/pipeline/operator_factory.h"
#include "ndll/pipeline/op_schema.h"
#include "ndll/pipeline/op_spec.h"
#include "ndll/pipeline/workspace/sample_workspace.h"
#include "ndll/pipeline/util/backend2workspace_map.h"

namespace ndll {

enum NDLLOpType {
  NDLL_GPU = 0,
  NDLL_CPU = 1,
  NDLL_MIXED = 2
};

/**
 * @brief Baseclass for the basic unit of computation in the pipeline.
 *
 * OperatorBase defines the API used by the pipeline to execute operations.
 * To create a custom operator, derive from this class, implement the
 * RunPerSampleCPU / RunBatchedGPU methods as desired, and register
 * the operator using the macros NDLL_REGISTER_{CPU,GPU}_OPERATOR.
 * To define meta-data about the op like the min/max number of inputs
 * it takes, a doctstring (for python), etc., use the NDLL_OPERATOR_SCHEMA,
 * macro. The op can then be added to a pipeline through its registered
 * name (the first arg to the registration macros).
 */
class OperatorBase {
 public:
  inline explicit OperatorBase(const OpSpec &spec) :
    spec_(spec), num_threads_(spec.GetArgument<int>("num_threads")),
    batch_size_(spec.GetArgument<int>("batch_size")),
    input_sets_(spec.GetArgument<int>("num_input_sets")) {
    NDLL_ENFORCE(num_threads_ > 0, "Invalid value for argument num_threads.");
    NDLL_ENFORCE(batch_size_ > 0, "Invalid value for argument batch_size.");
  }

  virtual inline ~OperatorBase() = default;

  /**
   * @brief Executes the operator on a single sample on the CPU.
   */
  virtual void Run(SampleWorkspace *ws) {
    NDLL_FAIL("CPU execution is not implemented for this operator!");
  }

  /**
   * @brief Executes the operator on a batch of samples on the GPU.
   */
  virtual void Run(DeviceWorkspace *ws) {
    NDLL_FAIL("GPU execution is not implemented for this operator!");
  }

  /**
   * @brief Used by operators interfacing with both CPU and GPU.
   */
  virtual void Run(MixedWorkspace *ws) {
    NDLL_FAIL("Mixed execution is not implemented for this operator!");
  }

  /**
   * @brief returns the name of the operator. By default returns
   * the name of the op as specified by the OpSpec it was constructed
   * from.
   */
  virtual string name() const {
    return spec_.name();
  }

  /**
   * @brief For reader Ops, returns the size of the dataset
   * For all other Ops, returns -1
   */
  virtual Index epoch_size() const {
    return -1;
  }

  int GetNumInputSets() const {
    return input_sets_;
  }

  DISABLE_COPY_MOVE_ASSIGN(OperatorBase);

 protected:
  OpSpec spec_;
  int num_threads_;
  int batch_size_;
  int input_sets_;
};

#define USE_OPERATOR_MEMBERS()                  \
  using OperatorBase::spec_;               \
  using OperatorBase::num_threads_;        \
  using OperatorBase::batch_size_

template <typename Backend>
class Operator : public OperatorBase {
 public:
  inline explicit Operator(const OpSpec &spec) :
    OperatorBase(spec)
  {}

  virtual inline ~Operator() = default;

  using OperatorBase::Run;
  void Run(Workspace<Backend> *ws) override {
    SetupSharedSampleParams(ws);

    for (int i = 0; i < input_sets_; ++i) {
      RunImpl(ws, i);
    }
  }

 protected:
  /**
   * @brief Shared param setup for CPU computation
   */
  virtual void SetupSharedSampleParams(Workspace<Backend> *ws) {}

  /**
   * @brief Implementation of the operator - to be
   * implemented by derived ops.
   */
  virtual void RunImpl(Workspace<Backend> *ws, int idx = 0) = 0;
};

template<>
class Operator<Mixed> : public OperatorBase {
 public:
  inline explicit Operator(const OpSpec &spec) :
    OperatorBase(spec)
  {}

  virtual inline ~Operator() = default;

  using OperatorBase::Run;
  void Run(MixedWorkspace *ws) override = 0;
};

// Create registries for CPU & GPU Operators
NDLL_DECLARE_OPTYPE_REGISTRY(CPUOperator, OperatorBase);
NDLL_DECLARE_OPTYPE_REGISTRY(GPUOperator, OperatorBase);
NDLL_DECLARE_OPTYPE_REGISTRY(MixedOperator, OperatorBase);

// Must be called from .cc or .cu file
#define NDLL_REGISTER_OPERATOR(OpName, OpType, device)          \
  int NDLL_OPERATOR_SCHEMA_REQUIRED_FOR_##OpName();             \
  static int ANONYMIZE_VARIABLE(OpName) =                       \
    NDLL_OPERATOR_SCHEMA_REQUIRED_FOR_##OpName();               \
  NDLL_DEFINE_OPTYPE_REGISTERER(OpName, OpType,                 \
      device##Operator, ndll::OperatorBase)

#define NDLL_REGISTER_OPERATOR_FOR_DEVICE(OpName, device)       \
        NDLL_REGISTER_OPERATOR(OpName, OpName<device##Backend>, device)

#define NDLL_NOT_IMPLEMENED_OPERATOR  NDLL_FAIL("Not implemented")

class ResizeParamDescr;

void DataDependentSetupCPU(const Tensor<CPUBackend> &input, Tensor<CPUBackend> *output,
                           const char *pOpName = NULL,
                           const uint8 **pInRaster = NULL, uint8 **ppOutRaster = NULL,
                           vector<NDLLSize> *pSizes = NULL, const NDLLSize *out_size = NULL);
bool DataDependentSetupGPU(const TensorList<GPUBackend> &input, TensorList<GPUBackend> *output,
                           size_t batch_size, bool reshapeBatch = false,
                           vector<const uint8 *> *iPtrs = NULL, vector<uint8 *> *oPtrs = NULL,
                           vector<NDLLSize> *pSizes = NULL, ResizeParamDescr *pResizeParam = NULL);
void CollectPointersForExecution(size_t batch_size,
                                 const TensorList<GPUBackend> &input, vector<const uint8 *> *inPtrs,
                                 TensorList<GPUBackend> *output, vector<uint8 *> *outPtrs);

}  // namespace ndll

#endif  // NDLL_PIPELINE_OPERATOR_H_
