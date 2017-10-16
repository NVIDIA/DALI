#ifndef NDLL_PIPELINE_DECODER_H_
#define NDLL_PIPELINE_DECODER_H_

#include "ndll/pipeline/operator.h"
#include "ndll/pipeline/operator_factory.h"

namespace ndll {

/**
 * @brief Decoder are special ops that are allowed to have data dependent output 
 * shapes. For this reason, they must appear first in the pipeline and can only 
 * appear once. User-defined decoders should derive from this class.
 */
template <typename Backend>
class Decoder : public Operator<Backend> {
public:
  inline Decoder() {}
  inline explicit Decoder(const OpSpec &spec) : Operator<Backend>(spec) {}
  virtual inline ~Decoder() = default;
  
  DISABLE_COPY_MOVE_ASSIGN(Decoder);
protected:
};

// Create registries for CPU & GPU Decoders
NDLL_DEFINE_OPTYPE_REGISTRY(CPUDecoder, Decoder<CPUBackend>);
NDLL_DEFINE_OPTYPE_REGISTRY(GPUDecoder, Decoder<GPUBackend>);

// Must be called from .cc or .cu file
#define NDLL_REGISTER_CPU_DECODER(OpName, OpType)           \
  NDLL_DEFINE_OPTYPE_REGISTERER(OpName, OpType,             \
      CPUDecoder, Decoder<CPUBackend>)
#define NDLL_REGISTER_GPU_DECODER(OpName, OpType)           \
  NDLL_DEFINE_OPTYPE_REGISTERER(OpName, OpType,             \
      GPUDecoder, Decoder<GPUBackend>)

} // namespace ndll

#endif // NDLL_PIPELINE_DECODER_H_
