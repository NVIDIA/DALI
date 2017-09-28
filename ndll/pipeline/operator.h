#ifndef NDLL_PIPELINE_OPERATORS_OPERATOR_H_
#define NDLL_PIPELINE_OPERATORS_OPERATOR_H_

#include <memory>
#include <type_traits>
#include <utility>

#include "ndll/common.h"
#include "ndll/error_handling.h"
#include "ndll/pipeline/data/backend.h"
#include "ndll/pipeline/data/batch.h"
#include "ndll/pipeline/data/datum.h"
#include "ndll/pipeline/data/sub_tensor.h"
#include "ndll/pipeline/util/stream_pool.h"

namespace ndll {

// Note: The original rationale behind having cpu & gpu ops be the same class was that
// the cpu & gpu implementations share alot of the same code (InferOutputShape, for example),
// but now this sharing of code is starting to stress the abstraction to the point that
// it is a bit hacky. Can we refactor so that shared methods and backend specific methods
// are separate?

/**
 * @brief Baseclass for the basic unit of computation in the pipeline
 */
template <typename Backend>
class Operator {
public:
  inline Operator()
    : num_threads_(-1),
      batch_size_(-1),
      stream_pool_(nullptr) {}
  virtual inline ~Operator() = default;
  
  /**
   * @brief executes the op on a single datum on cpu 
   */
  template <typename T = Backend>
  inline typename std::enable_if<std::is_base_of<CPUBackend, T >::value>::type
  Run(const Datum<Backend> &input, Datum<Backend> *output, int data_idx, int thread_idx) {
#ifndef NDEBUG
    NDLL_ENFORCE(num_threads_ > 0,
        "Num threads must be set before \"Run()\" is called");
    NDLL_ENFORCE(batch_size_ > 0,
        "Batch size must be set before \"Run()\" is called");
#endif
    RunPerDatumCPU(input, output, data_idx, thread_idx);
  }

  /**
   * @brief Executes the op on the whole batch of data on the gpu
   */
  template <typename T = Backend>
  inline typename std::enable_if<std::is_base_of<GPUBackend, T>::value>::type
  Run(const Batch<Backend> &input, Batch<Backend> *output) {
#ifndef NDEBUG
    NDLL_ENFORCE(batch_size_ == input.ndatum(),
        "Batch size must be set before \"Run()\" is called");
    NDLL_ENFORCE(stream_pool_.get() != nullptr,
        "Stream pool must be set before \"Run()\" is called");
#endif
    RunBatchedGPU(input, output);
  }

  /**
   * @brief Returns a vector where each element represents the size of
   * different parameters for 'RunBatchedGPU' that must be setup. By
   * default the sizes are 0, and on space is allocate for the Ops
   * batched parameters.
   *
   * Running operations on whole batches of data on GPU often requires
   * that lots of meta-data be copied to the GPU prior to kernel launch.
   * To do this efficiently, the pipeline manages a mega-buffer to store
   * all batched params for all operators. The parameters can then be
   * moved to device all at once.
   *
   * To take advantage of this feature, operators should override this
   * method to specify the sizes it needs. This method will be called
   * by the executor after the shape inference loop, so any data
   * dependent params should be setup in InferOutputShape. In general,
   * ops should prefer to do work in threaded methods (RunPerDatumCPU, 
   * InferOutputShape) to minimize serial workload in the pipeline
   *
   * If this method is overriden, 'SetBatchedParamBuffers()' must be
   * as well so the operator can be handed the allocate buffers.
   */
  template <typename T = Backend> inline
  typename std::enable_if<std::is_base_of<GPUBackend, T>::value, const vector<size_t>&>::type
  GetBatchedParameterSize() {
    CalculateBatchedParameterSize();
    return batched_param_sizes_;
  }

  template <typename T = Backend>
  inline typename std::enable_if<std::is_base_of<GPUBackend, T>::value>::type
  SetBatchedParameterBuffers(const vector<CPUSubTensor> &buffers,
      const vector<GPUSubTensor> &gpu_buffers) {
    NDLL_ENFORCE(buffers.size() == gpu_buffers.size());
    NDLL_ENFORCE(buffers.size() == batched_param_sizes_.size());
    batched_param_buffers_ = buffers;
    batched_param_gpu_buffers_ = gpu_buffers;
  }

  template <typename T = Backend>
  inline typename std::enable_if<std::is_base_of<GPUBackend, T>::value>::type
  BatchedParameterSetup(const Batch<Backend> &input, Batch<Backend> *output) {
    SerialBatchedParameterSetup(input, output);
  }

  /**
   * @brief Gives Operators a chance to perform batched parameter setup
   * in the executors threads to minimize serial work
   */
  template <typename T = Backend>
  inline typename std::enable_if<std::is_base_of<GPUBackend, T>::value>::type
  BatchedParameterSetupPerDatum(const Batch<Backend> &input, Batch<Backend> *output,
      int data_idx, int thread_idx) {
    ThreadedBatchedParameterSetup(input, output, data_idx, thread_idx);
  }
  
  /**
   * @brief returns the output op shape given the input shape and data
   */
  virtual vector<Index> InferOutputShape(const Datum<Backend> &input,
      int data_idx, int thread_idx) = 0;

  /**
   * @brief sets the type of the input batch based on the input type
   */
  virtual void SetOutputType(Batch<Backend> *output, TypeMeta input_type) = 0;
  
  /**
   * @brief returns a newly allocated exact copy of the operator
   */
  virtual Operator* Clone() const = 0;

  /**
   * @brief returns the name of the operator
   */
  virtual string name() const = 0;

  //
  /// Setters for operator meta-data required to execute the op
  //

  // User can override if they need to setup meta-data
  virtual inline void set_num_threads(int num_threads) {
    num_threads_ = num_threads;
  }

  // User can override if they need to setup meta-data
  virtual inline void set_batch_size(int batch_size) {
    batch_size_ = batch_size;
  }

  inline void set_stream_pool(shared_ptr<StreamPool> stream_pool) {
    stream_pool_ = stream_pool;
  }
  
  DISABLE_COPY_MOVE_ASSIGN(Operator);
protected:
  /**
   * @brief Per image CPU computation of the operator to be 
   * implemented by derived ops.
   */
  virtual inline void RunPerDatumCPU(const Datum<Backend> &input,
      Datum<Backend> *output, int data_idx, int thread_idx) {
    NDLL_FAIL("RunPerDatumCPU not implemented");
  }

  /**
   * @brief Batched GPU computation of the operator to be 
   * implemented by derived ops.
   */
  virtual inline void RunBatchedGPU(const Batch<Backend> &input,
      Batch<Backend> *output) {
    NDLL_FAIL("RunBatchedGPU not implemented");
  }

  /**
   * @brief Performs and serial calculation neccessary for batched parameter
   * size calculation. Ops should do any work possible in the threaded methods
   * to avoid doing unnecessary serial work
   */
  virtual void CalculateBatchedParameterSize() {
    // Default does nothing
  }
  
  /**
   * @brief Performs any serial batched parameter setup that needs to be done 
   * by the op. Ops should do any work possible in the threaded methods to 
   * avoid doing unnecessary serial work
   */
  virtual void SerialBatchedParameterSetup(const Batch<Backend> &input, Batch<Backend> *output) {
    // Default does nothing
  }

  /**
   * Can be overriden by a derive op to perform any needed batched parameter
   * setup in the executors threads to avoid doing unnecessary serial work
   *
   * The input batch is provided for ops that need to set ptr offsets. We
   * cannot provide the output batch for all ops, as the last op in the 
   * pipeline won't have its output batch set until 'RunForward' is called
   */
  virtual void ThreadedBatchedParameterSetup(const Batch<Backend> &input,
      Batch<Backend> *output, int data_idx, int thread_idx) {
    // Default does nothing
  }
  
  int num_threads_;
  int batch_size_;
  std::shared_ptr<StreamPool> stream_pool_;

  vector<size_t> batched_param_sizes_;
  vector<CPUSubTensor> batched_param_buffers_;
  vector<GPUSubTensor> batched_param_gpu_buffers_;
};

// Decoders are special operations that can have data-dependent shape inference.
// For this reason, they are always first in the pipeline.
template <typename Backend>
class Decoder : public Operator<Backend> {
public:
  inline Decoder() {}
  virtual inline ~Decoder() = default;
  
  DISABLE_COPY_MOVE_ASSIGN(Decoder);
protected:
};

template <typename Backend>
class Transformer : public Operator<Backend> {
public:
  inline Transformer() {}
  virtual inline ~Transformer() = default;

  inline vector<Index> InferOutputShape(const Datum<Backend> &input,
      int data_idx, int thread_idx) override final {
#ifndef NDEBUG
    NDLL_ENFORCE(data_idx < this->batch_size_, "data_idx out of range");
#endif
    
    // Transfomers cannot have data dependent output shapes, we override
    // this method and allow the user to define a simpler method that
    // only receives the input shape
    return InferOutputShapeFromShape(input.shape(), data_idx, thread_idx);
  }

  // TODO(tgale): Can we make this not copy another vector? Will it
  // even make two tmps or will the compiler just forward them on
  // through the return statement?
  virtual vector<Index> InferOutputShapeFromShape(
      const vector<Index> &input_shape, int data_idx, int thread_idx) = 0;

  DISABLE_COPY_MOVE_ASSIGN(Transformer);
protected:
};

} // namespace ndll

#endif // NDLL_PIPELINE_OPERATORS_OPERATOR_H_
