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

//   inline vector<Dims> InferOutputShapes(const SampleWorkspace &ws) override final {
// #ifndef NDEBUG
//     NDLL_ENFORCE(ws->thread_idx() > 0, "Invalid negative thread idx for cpu work.");
//     NDLL_ENFORCE(ws->thread_idx() < num_threads_, "Thread index out of range.");
//     NDLL_ENFORCE(ws->data_idx() > 0, "Invalid negative data index for cpu work.");
//     NDLL_ENFORCE(ws->data_idx() < batch_size_, "Data index out of range.");
// #endif
//     return InferOutputShapesFromShapes(ws.meta());
//   }
  
  DISABLE_COPY_MOVE_ASSIGN(Transformer);
protected:

  USE_OPERATOR_MEMBERS();
};

// Create registries for CPU & GPU Transformeres
NDLL_DECLARE_OPTYPE_REGISTRY(CPUTransformer, Transformer<CPUBackend>);
NDLL_DECLARE_OPTYPE_REGISTRY(GPUTransformer, Transformer<GPUBackend>);

// Must be called from .cc or .cu file
#define NDLL_REGISTER_CPU_TRANSFORM(OpName, OpType)   \
  NDLL_DEFINE_OPTYPE_REGISTERER(OpName, OpType,       \
      CPUTransformer, Transformer<CPUBackend>) 
#define NDLL_REGISTER_GPU_TRANSFORM(OpName, OpType)   \
  NDLL_DEFINE_OPTYPE_REGISTERER(OpName, OpType,       \
      GPUTransformer, Transformer<GPUBackend>)

} // namespace ndll

#endif // NDLL_PIPELINE_TRANSFORMER_H_
