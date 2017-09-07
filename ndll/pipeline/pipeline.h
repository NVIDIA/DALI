#ifndef NDLL_PIPELINE_PIPELINE_H_
#define NDLL_PIPELINE_PIPELINE_H_

#include <memory>

#include "ndll/common.h"
#include "ndll/pipeline/buffer.h"
#include "ndll/pipeline/operator.h"
#include "ndll/pipeline/stream_pool.h"
#include "ndll/pipeline/thread_pool.h"

namespace ndll {

/**
 * @brief Organizes and executes the set of operators chosen by the user.
 * Provides optimizations like batched copies to GPU and pre-sizing of
 * buffers.
 */
template <typename CPUBackend, typename GPUBackend>
class Pipeline {
public:
  /**
   * @brief Creates a pipeline with `num_threads` worker threads and 
   * working in `main_stream`. 
   *
   * The max  streams parameter limits the maximum amount of additional 
   * streams that operators in pipeline can use. The pipeline inserts 
   * cudaEvents into the main_stream as needed to ensure expected 
   * synchronization behavior as if all work had been issued in the 
   * main_stream. The non-blocking flag specifies whether additional 
   * streams should be allocated as non-blocking streams.
   */
  Pipeline(int num_threads, cudaStream_t main_stream, int max_streams, bool non_blocking) :
    thread_pool_(num_threads), stream_pool_(main_stream, max_streams, non_blocking) {}
  
  ~Pipeline() = default;

  /**
   * @brief Adds an op to the prefetch stage of the pipeline. The input op is
   * moved into the pipeline and left in a default state
   */
  void AddPrefetchOp(Operator &op) {
    prefetch_ops_.push_back(std::move(op));
  }

  /**
   * @brief Adds an op to the forward stage of the pipeline. The input op is
   * moved into the pipeline and left in a default state
   */
  void AddForwardOp(Operator &op) {
    forward_ops_.push_back(std::move(op));
  }

  /**
   * @brief Pairs CPUBuffers w/ GPUBuffers for transfering data to the GPU
   *
   * TODO(tgale): Make sure this stores the correct references 
   * and pointers that will still be in scope
   */
  void AddCopyBuffers(std::shared_ptr<vector<BufferBase> > cpu_buffers,
      std::shared_ptr<vector<BufferBase> > gpu_buffers) {
    NDLL_ENFORCE(cpu_buffers != nullptr);
    NDLL_ENFORCE(gpu_buffers != nullptr);
    NDLL_ENFORCE(cpu_buffers->size() == gpu_buffers->size());
    cpu_buffers_ = cpu_buffers;
    gpu_buffers_ = gpu_buffers;
  }

  /**
   * @brief Run the prefetch stage of the pipeline
   */
  void RunPrefetch() {
    // Run the decoder to get output dimensions, then do a shape inference pass
    // to let the ops set up their shit and also let the forward ops tell us
    // how much memory they need for their params for the forward pass

    // Run another thread loop to finish all host-side processing.
    // Then do another pass through all the forward ops to let them setup their
    // batched forward params in our mega buffer?
    // Document all assumptions of this execution pattern:
    // 1. Size of the output is assumed to be only dependent on the input data shape.
    //    for things like bounding box crops that would be produced in the decoder,
    //    This is stil ok because we execute the decoder stage first.
  }

  /**
   * @brief Copy the host buffers into their device-side counterparts
   */
  void RunCopy() {
    for (int i = 0; i < cpu_buffers_->size(); ++i) {
      // TODO(tgale): Should we enforce this or resize the
      // device buffer appropriately?
      NDLL_ENFORCE((*cpu_buffers_)[i].shape() == (*gpu_buffers_)[i].shape());
      
      CUDA_ENFORCE(cudaMemcpyAsync((*gpu_buffers_)[i].raw_data(),
              (*cpu_buffers_)[i].raw_data(), (*cpu_buffers_)[i].bytes(),
              cudaMemcpyHostToDevice, stream_pool_.GetStream()));
    }
  }

  /**
   * @brief Run the forward stage of the pipeline
   */
  void RunForward() {
    // Launch all the kernels, then use the stream pool to insert
    // events to enforce synchronization behavior
  }
  
  DISABLE_COPY_ASSIGN(Pipeline);
private:
  StreamPool stream_pool_;
  ThreadPool thread_pool_;
  
  Decoder decoder_;
  vector<Operator> prefetch_ops_;
  vector<Operator> forward_ops_;

  std::shared_ptr<vector<BufferBase> > cpu_buffers_;
  std::shared_ptr<vector<BufferBase> > gpu_buffers_;
};

} // namespace ndll

#endif // NDLL_PIPELINE_PIPELINE_H_
