#ifndef NDLL_PIPELINE_OPERATORS_OPERATOR_H_
#define NDLL_PIPELINE_OPERATORS_OPERATOR_H_

#include <memory>
#include <type_traits>
#include <utility>

#include "ndll/common.h"
#include "ndll/error_handling.h"
#include "ndll/pipeline/data/backend.h"
#include "ndll/pipeline/data/batch.h"
#include "ndll/pipeline/util/stream_pool.h"

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
  inline Operator(int num_threads, std::shared_ptr<StreamPool> stream_pool)
    : num_threads_(num_threads), stream_pool_(stream_pool) {}
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
  inline typename std::enable_if<std::is_base_of<CPUBackend, T >::value>::type
  Run(const Datum<Backend> &input, Datum<Backend> *output) {
    RunPerDatumCPU(input, output);
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
  
  DISABLE_COPY_MOVE_ASSIGN(Operator);
protected:
  int num_threads_;
  std::shared_ptr<StreamPool> stream_pool_;
};

// TODO(tgale): Is there any point to having this? It does not
// change anything from the base Operator class.
template <typename Backend>
class Decoder : public Operator<Backend> {
public:
  inline Decoder(int num_threads, std::shared_ptr<StreamPool> stream_pool)
    : Operator<Backend>(num_threads, stream_pool) {}
  virtual inline ~Decoder() = default;

  DISABLE_COPY_MOVE_ASSIGN(Decoder);
protected:
};

template <typename Backend>
class Transformer : public Operator<Backend> {
public:
  inline Transformer(int num_threads, std::shared_ptr<StreamPool> stream_pool)
    : Operator<Backend>(num_threads, stream_pool) {}
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
  virtual vector<Index> InferOutputShapeFromShape(const vector<Index> &input_shape) = 0;

  DISABLE_COPY_MOVE_ASSIGN(Transformer);
protected:
};

} // namespace ndll

#endif // NDLL_PIPELINE_OPERATORS_OPERATOR_H_
