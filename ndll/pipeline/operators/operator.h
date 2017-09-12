#ifndef NDLL_PIPELINE_OPERATORS_OPERATOR_H_
#define NDLL_PIPELINE_OPERATORS_OPERATOR_H_

#include <type_traits>
#include <utility>

#include "ndll/common.h"
#include "ndll/error_handling.h"
#include "ndll/pipeline/data/backend.h"
#include "ndll/pipeline/data/batch.h"

namespace ndll {

/**
 * @brief Baseclass for the basic unit of computation in the pipeline
 */
template <typename Backend>
class Operator {
public:
  Operator() {}
  virtual ~Operator() = default;

  // Note: An operator defines a computation that can be performed in the pipeline.
  // Operators can be run per image on the cpu or batched on the gpu. Each execution
  // method takes in slightly different paramters, and only works with certain cases
  // of the 'Backend' template paramter. We wan't to enforce that the execution method
  // for each 'Backend' can only be called when the template parameter is correct. To
  // do this, we use 'enable_if'. If the wrong method is called for the template paramter
  // this will result in a compiler error. An alternative is to check at runtime w/
  // 'dynamic_cast', but this is grosser and moves a check we can do at compile time
  // into run time.
  
  /**
   * @brief executes the op on a single datum on cpu 
   */
  template <typename T = Backend>
  typename std::enable_if<std::is_base_of<CPUBackend, T >::value, int >::type
  Run(const Batch<Backend> &input, Batch<Backend> *output, int data_idx) {
    RunPerDatumCPU(input, output, data_idx);
  }

  /**
   * @brief Executes the op on the whole batch of data on the gpu
   */
  template <typename T = Backend>
  typename std::enable_if<std::is_base_of<GPUBackend, T>::value>::type
  Run(const Batch<Backend> &input, Batch<Backend> *output) {
    RunBatchedGPU(input, output);
  }

  /**
   * @brief Per image CPU computation of the operator to be 
   * implemented by derived ops.
   */
  virtual void RunPerDatumCPU(const Batch<Backend> &input,
      Batch<Backend> *output, int data_idx) {
    NDLL_FAIL("RunPerDatumCPU not implemented");
  }

  /**
   * @brief Batched GPU computation of the operator to be 
   * implemented by derived ops.
   */
  virtual void RunBatchedGPU(const Batch<Backend> &input,
      Batch<Backend> *output) {
    NDLL_FAIL("RunBatchedGPU not implemented");
  }

  /**
   * @brief returns the output op shape given the input shape and data
   */
  virtual vector<Index> InferOutputShape(const Datum<Backend> &input) {
    NDLL_FAIL("InferOutputShape not implemented");
  }
  
  /**
   * Move constructor to allow transfer of ownership of the 
   * op from the user to the pipeline. This must be implemented
   * by any derived op.
   */
  Operator(Operator &&op) noexcept {}
  Operator(const Operator&) = delete;
  Operator& operator=(Operator&&) = delete;
  Operator& operator=(const Operator&) = delete;
};

template <typename Backend>
class Decoder : public Operator<Backend> {
public:
  Decoder() {}
  virtual ~Decoder() = default;
  
  vector<Index> InferOutputShape(const Datum<Backend> &input) {
    return vector<Index>{};
  }

  /**
   * Move constructor to allow transfer of ownership of the
   * decoder from the user to the pipeline. Must be implemented
   * by any derived decoder.
   */
  Decoder(Decoder &&dec) noexcept {}
  Decoder(const Decoder&) = delete;
  Decoder& operator=(Decoder&&) = delete;
  Decoder& operator=(const Decoder&) = delete;

private:
};

} // namespace ndll

#endif // NDLL_PIPELINE_OPERATORS_OPERATOR_H_
