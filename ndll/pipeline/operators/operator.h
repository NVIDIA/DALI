#ifndef NDLL_PIPELINE_OPERATORS_OPERATOR_H_
#define NDLL_PIPELINE_OPERATORS_OPERATOR_H_

#include <type_traits>
#include <utility>

#include "ndll/common.h"
#include "ndll/error_handling.h"
#include "ndll/pipeline/data/backend.h"
#include "ndll/pipeline/data/batch.h"

namespace ndll {

// TODO(tgale): Need to try and define and op that uses some complex internal
// storage (e.g. a Tensor) to make sure defining move & move-assignment ops
// is not a huge burden

/**
 * @brief Baseclass for the basic unit of computation in the pipeline
 */
template <typename Backend>
class Operator {
public:
  inline Operator() {}
  virtual inline ~Operator() = default;

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
  inline typename std::enable_if<std::is_base_of<CPUBackend, T >::value, int >::type
  Run(const Datum<Backend> &input, Datum<Backend> *output, int data_idx) {
    RunPerDatumCPU(input, output, data_idx);
  }

  /**
   * @brief Executes the op on the whole batch of data on the gpu
   */
  template <typename T = Backend>
  inline typename std::enable_if<std::is_base_of<GPUBackend, T>::value>::type
  Run(const Batch<Backend> &input, Batch<Backend> *output) {
    RunBatchedGPU(input, output);
  }

  /**
   * @brief Per image CPU computation of the operator to be 
   * implemented by derived ops.
   */
  virtual inline void RunPerDatumCPU(const Datum<Backend> &input,
      Datum<Backend> *output) {
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
   * @brief returns the output op shape given the input shape and data
   */
  virtual inline vector<Index> InferOutputShape(const Datum<Backend> &input) {
    NDLL_FAIL("InferOutputShape not implemented");
  }
  
  /**
   * Move constructor & move assignment operator to allow transfer 
   * of ownership of the op from the user to the pipeline. This 
   * must be implemented by any derived op.
   */
  inline Operator(Operator &&op) noexcept {}
  inline Operator& operator=(Operator&&) noexcept { return *this; }
  
  Operator(const Operator&) = delete;
  Operator& operator=(const Operator&) = delete;
};

template <typename Backend>
class Decoder : public Operator<Backend> {
public:
  inline Decoder() {}
  virtual inline ~Decoder() = default;

  // TODO(tgale): Make this pure virtual once the pipeline is more refined
  virtual inline vector<Index> InferOutputShape(const Datum<Backend> &input) {
    return vector<Index>{};
  }

  /**
   * Move constructor & move assignment operator to allow transfer 
   * of ownership of the op from the user to the pipeline. This 
   * must be implemented by any derived op.
   */
  inline Decoder(Decoder &&dec) noexcept {}
  inline Decoder& operator=(Decoder&&) noexcept {}

  Decoder(const Decoder&) = delete;
  Decoder& operator=(const Decoder&) = delete;
protected:
};

template <typename Backend>
class Transformer : public Operator<Backend> {
public:
  inline Transformer() {}
  virtual inline ~Transformer() = default;

  inline vector<Index> InferOutputShape(const Datum<Backend> &input) override {
    // Transfomers cannot have data dependent output shapes, we override
    // this method and allow the user to define a simpler method that
    // only receives the input shape
    return InferOutputShapeFromShape(input.shape());
  }

  // TODO(tgale): Can we make this not copy another vector? Will it
  // even make two tmps or will the compiler just forward them on
  // through the return statement?
  virtual inline vector<Index>
  InferOutputShapeFromShape(const vector<Index> &input_shape) = 0;

  /**
   * Move constructor to allow transfer of ownership of the
   * decoder from the user to the pipeline. Must be implemented
   * by any derived decoder.
   */
  inline Transformer(Transformer &&dec) noexcept {}
  Transformer(const Transformer&) = delete;
  Transformer& operator=(Transformer&&) = delete;
  Transformer& operator=(const Transformer&) = delete;
protected:
};

} // namespace ndll

#endif // NDLL_PIPELINE_OPERATORS_OPERATOR_H_
