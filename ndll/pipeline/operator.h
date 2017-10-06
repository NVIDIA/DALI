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

namespace ndll {

/**
 * @brief Baseclass for the basic unit of computation in the pipeline.
 *
 * Operator defines the API used by the pipeline to execute operations,
 * perform shape inference, and setup any needed paramters for batched
 * execution. User-defined ops should derive from 'Decoder' or 'Transformer'
 * depending on which category the op fits into.
 */
template <typename Backend>
class Operator {
public:
  inline Operator()
    : num_threads_(-1),
      batch_size_(-1),
      has_stream_(false) {}
  virtual inline ~Operator() = default;
  
  /**
   * @brief Executes the op on a single datum on cpu.
   *
   * @param input The input Datum that is to be processed.
   * @param output The output Datum to process the input into.
   * @param data_idx The index of this Datum in the batch.
   * @param thread_idx The id of the calling thread.
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
   * @brief Executes the op on the whole batch of data on the gpu.
   *
   * @param input The input Batch of data to process.
   * @param output The Batch to store the processed input in.
   */
  template <typename T = Backend>
  inline typename std::enable_if<std::is_base_of<GPUBackend, T>::value>::type
  Run(const Batch<Backend> &input, Batch<Backend> *output) {
#ifndef NDEBUG
    NDLL_ENFORCE(batch_size_ == input.ndatum(),
        "Batch size must be set before \"Run()\" is called");
    NDLL_ENFORCE(has_stream_,
        "Stream must be set before \"Run()\" is called");
#endif
    RunBatchedGPU(input, output);
  }

  /**
   * @brief Returns a vector where each element represents the size of
   * different parameters for 'RunBatchedGPU' that must be setup. By
   * default the sizes are 0, and no space is allocate for the op's
   * batched parameters. 
   *
   * @ref ndll::Operator<Backend>#CalculateBatchedParamterSize is called 
   * in this method. See @ref ndll::Operator<Backend>#CalculateBatchedParamterSize 
   * for information regarding how to leverage this feature in a derived class.
   */
  template <typename T = Backend> inline
  typename std::enable_if<std::is_base_of<GPUBackend, T>::value, const vector<size_t>&>::type
  GetBatchedParameterSize() {
    CalculateBatchedParameterSize();
    return batched_param_sizes_;
  }

  /**
   * @brief Saves the input SubTensors for the op to stage its batched 
   * parameters in.
   */
  template <typename T = Backend>
  inline typename std::enable_if<std::is_base_of<GPUBackend, T>::value>::type
  SetBatchedParameterBuffers(const vector<SubTensor<CPUBackend>> &buffers,
      const vector<SubTensor<GPUBackend>> &gpu_buffers) {
    NDLL_ENFORCE(buffers.size() == gpu_buffers.size());
    NDLL_ENFORCE(buffers.size() == batched_param_sizes_.size());
    batched_param_buffers_ = buffers;
    batched_param_gpu_buffers_ = gpu_buffers;
  }

  /**
   * @brief Forwards the input arguments to 
   * @ref ndll::Operator<Backend>#SerialBatchedParamterSetup. See
   * @ref ndll::Operator<Backend>#SerialBatchedParamterSetup for details.
   */
  template <typename T = Backend>
  inline typename std::enable_if<std::is_base_of<GPUBackend, T>::value>::type
  BatchedParameterSetup(const Batch<Backend> &input, Batch<Backend> *output) {
    SerialBatchedParameterSetup(input, output);
  }

  /**
   * @brief Gives Operators a chance to perform batched parameter setup
   * in the executors threads to minimize serial work.
   * 
   * Calls @ref ndll::Operator<Backend>#ThreadedBatchedParamterSetup. See
   * Calls @ref ndll::Operator<Backend>#ThreadedBatchedParamterSetup for
   * details.
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

  /**
   * @brief Setter for the number of threads that will execute the op.
   * Can be overriden by derived classes to perform thread local
   * resource setup.
   */
  virtual inline void set_num_threads(int num_threads) {
    num_threads_ = num_threads;
  }

  /**
   * @brief Setter for the size of the batch that will be processed.
   * Can be overriden by derived classes to perform per-datum resource
   * setup.
   */
  virtual inline void set_batch_size(int batch_size) {
    batch_size_ = batch_size;
  }

  inline void set_stream(cudaStream_t stream) {
    has_stream_ = true;
    stream_ = stream;
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
   * (InferOutputShape) to avoid doing unnecessary serial work.
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
   * ops should prefer to do work in threaded methods (InferOutputShape, 
   * BatchedParamterSetupPerDatum) to minimize serial workload in the 
   * pipeline.
   */
  virtual void CalculateBatchedParameterSize() {
    // Default does nothing
  }
  
  /**
   * @brief Performs any serial batched parameter setup that needs to be done 
   * by the op. Ops should do any work possible in the threaded methods 
   * (BatchedParamterSetupPerDatum) to avoid doing unnecessary serial work.
   */
  virtual void SerialBatchedParameterSetup(const Batch<Backend> &input, Batch<Backend> *output) {
    // Default does nothing
  }

  /**
   * @brief Performs batched paramter setup that needs to be done by the op
   * on a per-datum basis. This method is called in the second Pipeline thread
   * loop and can be overriden by a derive op to perform any needed batched 
   * parametersetup in the executors threads to avoid doing unnecessary serial 
   * work
   */
  virtual void ThreadedBatchedParameterSetup(const Batch<Backend> &input,
      Batch<Backend> *output, int data_idx, int thread_idx) {
    // Default does nothing
  }
  
  int num_threads_;
  int batch_size_;
  cudaStream_t stream_;
  bool has_stream_;
  
  vector<size_t> batched_param_sizes_;
  vector<SubTensor<CPUBackend>> batched_param_buffers_;
  vector<SubTensor<GPUBackend>> batched_param_gpu_buffers_;
};

/**
 * @brief Decoder are special ops that are allowed to have data dependent output 
 * shapes. For this reason, they must appear first in the pipeline and can only 
 * appear once. User-defined decoders should derive from this class.
 */
template <typename Backend>
class Decoder : public Operator<Backend> {
public:
  inline Decoder() {}
  virtual inline ~Decoder() = default;
  
  DISABLE_COPY_MOVE_ASSIGN(Decoder);
protected:
};

/**
 * @brief Transformers are general ops whose output shape depends only on the
 * input shape. User-defined transformations should derive from this class.
 */
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

  /**
   * @brief Returns the output shape that will be produced for the given
   * input shape. User-defined ops must implement this method.
   */
  virtual vector<Index> InferOutputShapeFromShape(
      const vector<Index> &input_shape, int data_idx, int thread_idx) = 0;

  DISABLE_COPY_MOVE_ASSIGN(Transformer);
protected:
};

} // namespace ndll

#endif // NDLL_PIPELINE_OPERATORS_OPERATOR_H_
