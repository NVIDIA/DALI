#ifndef NDLL_PIPELINE_OPERATORS_OPERATOR_H_
#define NDLL_PIPELINE_OPERATORS_OPERATOR_H_

#include <type_traits>

#include "ndll/common.h"
#include "ndll/error_handling.h"
#include "ndll/pipeline/data/backend.h"
#include "ndll/pipeline/op_spec.h"
#include "ndll/pipeline/batch_workspace.h"
#include "ndll/pipeline/sample_workspace.h"

namespace ndll {

/**
 * @brief Baseclass for the basic unit of computation in the pipeline.
 *
 * Operator defines the API used by the pipeline to execute operations,
 * perform shape/type inference, and setup any needed paramters for batched
 * execution. User-defined ops should derive from 'Decoder' or 'Transformer'
 * depending on which category the op fits into.
 *
 * Executing ops on an entire batch often requires the setup of meta-data on
 * GPU. To do this efficiently, the Operator provides methods that the
 * the Pipeline can query to figure out what paramters the op needs on GPU,
 * and then adds the required buffers to the ops Workspace. The op can
 * then implement ParamterSetup{PerSample, Batched}GPU to setup their
 * parameters on the CPU. These parameters are efficiently transfered
 * by the Pipeline to the GPU prior to execution of the op. The GPU versions
 * of the same data setup on the CPU by the op is also available in the
 * Workspace passed in for kernel execution.
 */
template <typename Backend>
class Operator {
public:
  inline explicit Operator(const OpSpec &spec) :
    num_threads_(spec.GetSingleArgument<int>("num_threads", -1)),
    batch_size_(spec.GetSingleArgument<int>("batch_size", -1)) {
    NDLL_ENFORCE(num_threads_ > 0, "Invalid value for argument num_threads.");
    NDLL_ENFORCE(batch_size_ > 0, "Invalid value for argument batch_size.");
  }
  
  virtual inline ~Operator() = default;

  /**
   * @brief Executes the operator on a single sample on the CPU.
   */
  void Run(SampleWorkspace *ws) {
#ifndef NDEBUG
    NDLL_ENFORCE(ws->thread_idx() > 0, "Invalid negative thread idx for cpu work.");
    NDLL_ENFORCE(ws->thread_idx() < num_threads_, "Thread index out of range.");
    NDLL_ENFORCE(ws->data_idx() > 0, "Invalid negative data index for cpu work.");
    NDLL_ENFORCE(ws->data_idx() < batch_size_, "Data index out of range.");
#endif
    RunPerSampleCPU(ws);
  }

  /**
   * @brief Executes the operator on a batch of samples on the GPU.
   */
  void Run(BatchWorkspace *ws) {
#ifndef NDEBUG
    NDLL_ENFORCE(ws->thread_idx() == -1, "GPU op should not be run in thread pool.");
    NDLL_ENFORCE(ws->data_idx() == -1, "GPU op should own whole batch.");
#endif
    RunBatchedGPU(ws);
  }
  
  /**
   * @brief Returns a vector of Tensor shapes, each one refering to the
   * the shape of the Output tensor at position `meta->data_idx()`
   * in the output TensorLists
   */
  virtual vector<Dims> InferOutputShapes(const SampleWorkspace &ws) = 0;

  /**
   * @brief Returns a vector of type meta data, one for each output
   * TensorList.
   */
  virtual vector<TypeInfo> InferOutputTypes(const BatchMeta &meta) = 0;


  /**
   * @brief Returns the number of extra kernel parameter tensors required by
   * the operator. Defaults to 0.
   */
  virtual int NumParameterTensor() const { return 0; }
  
  /**
   * @brief Returns a vector of Tensor shapes, each one refering to the 
   * shape of a different parameter tensor required by the Op for batched
   * gpu execution. By default the sizes are 0, and no space is allocate 
   * for the op's batched parameters. The length of the returned vector
   * must match the value returned by Operator<Backend>#NumParameterTensor.
   */
  virtual vector<Dims> InferParameterShapes(const BatchMeta &meta) {
    return vector<Dims>{};
  }

  /**
   * @brief Returns a vector of type meta data for each of the required
   * gpu paramters buffers. The length of the returned vector must match 
   * the value returned by Operator<Backend>#NumParameterTensor.
   */
  virtual vector<TypeInfo> InferParameterTypes(const BatchMeta &meta) {
    return vector<TypeInfo>{};
  }

  /**
   * @brief Can be implemented by derived ops to setup any needed paramters for
   * the kernel. Paramters are setup on host, and then transfered to the device
   * by the pipeline.
   *
   * Prefer to implement paramter setup per-image (by implementing 
   * Operator<Backend>#ParameterSetupPerSample). This is called in the thread 
   * pool and thus reduces the amount of serial work done by the pipeline.
   */
  virtual void ParameterSetupBatched(BatchWorkspace *ws) {
    // No-op by default
  }
  
  /**
   * @brief Can be implemented by derived ops to setup any needed per-sample 
   * paramters for the kernel. Paramters are setup on host, and then transfered 
   * to the device by the pipeline.
   */
  virtual void ParameterSetupPerSample(SampleWorkspace *ws) {
    // No-op by default
  }
  
  /**
   * @brief returns the name of the operator
   */
  virtual string name() const = 0;

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
  virtual inline void RunBatchedGPU(BatchWorkspace *ws) {
    NDLL_FAIL("RunBatchedGPU not implemented");
  }

  int num_threads_;
  int batch_size_;
};

#define USE_OPERATOR_MEMBERS()                  \
  using Operator<Backend>::num_threads_;        \
  using Operator<Backend>::batch_size_

} // namespace ndll

#endif // NDLL_PIPELINE_OPERATORS_OPERATOR_H_
