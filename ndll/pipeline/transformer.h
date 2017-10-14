#ifndef NDLL_PIPELINE_TRANSFORMER_H_
#define NDLL_PIPELINE_TRANSFORMER_H_

#include "ndll/pipeline/operator.h"
#include "ndll/pipeline/operator_factory.h"

namespace ndll {

/**
 * @brief Transformers are general ops whose output shape depends only on the
 * input shape. User-defined transformations should derive from this class.
 */
template <typename Backend>
class Transformer : public Operator<Backend> {
public:
  inline Transformer() {}
  inline explicit Transformer(const OpSpec &spec) : Operator<Backend>(spec) {}
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

// Create registries for CPU & GPU Transformeres
NDLL_DEFINE_OPTYPE_REGISTRY(CPUTransformer, Operator<CPUBackend>);
NDLL_DEFINE_OPTYPE_REGISTRY(GPUTransformer, Operator<CPUBackend>);

// Must be called from .cc or .cu file
#define NDLL_REGISTER_CPU_TRANSFORM(OpName)           \
  NDLL_DEFINE_OPTYPE_REGISTERER(OpName, CPUDecoder,   \
      Operator<CPUBackend>)
#define NDLL_REGISTER_GPU_TRANSFORM(OpName)           \
  NDLL_DEFINE_OPTYPE_REGISTERER(OpName, GPUDecoder,   \
      Operator<GPUBackend>)

} // namespace ndll

#endif // NDLL_PIPELINE_TRANSFORMER_H_
