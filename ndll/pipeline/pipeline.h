#ifndef NDLL_PIPELINE_PIPELINE_H_
#define NDLL_PIPELINE_PIPELINE_H_

#include <functional>
#include <memory>

#include "ndll/common.h"
#include "ndll/pipeline/batch.h"
#include "ndll/pipeline/operator.h"
#include "ndll/pipeline/stream_pool.h"
#include "ndll/pipeline/tensor.h"
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
  inline Pipeline(int num_threads, cudaStream_t main_stream,
      int max_streams,  bool non_blocking) :
    built_(false), decode_location_(DECODE_NONE), thread_pool_(num_threads),
    stream_pool_(main_stream, max_streams, non_blocking) {}
  
  ~Pipeline() = default;

  /**
   * @brief Adds an op to the prefetch stage of the pipeline. The input op is
   * moved into the pipeline and left in a default state
   */
  inline void AddPrefetchOp(Operator<CPUBackend> &op) {
    NDLL_ENFORCE(!built_, "Alterations to the pipeline after "
        "\"Build()\" has been called are not allowed");
    prefetch_ops_.push_back(std::move(op));
  }

  /**
   * @brief Adds an op to the forward stage of the pipeline. The input op is
   * moved into the pipeline and left in a default state
   */
  inline void AddForwardOp(Operator<GPUBackend> &op) {
    NDLL_ENFORCE(!built_, "Alterations to the pipeline after "
        "\"Build()\" has been called are not allowed");
    forward_ops_.push_back(std::move(op));
  }

  /**
   * @brief Inserts the decoder on the front of the prefetch stage of the pipeline.
   * The op is moved into the pipeline and left in a default state
   */
  inline void AddDecoder(Operator<CPUBackend> &dec) {
    NDLL_ENFORCE(!built_, "Alterations to the pipeline after "
            "\"Build()\" has been called are not allowed");
    NDLL_ENFORCE(decode_location_ == DECODE_NONE,
        "A Decoder already exists in the pipeline");
    prefetch_ops_.insert(prefetch_ops_.begin(), std::move(dec));
    decode_location_ = DECODE_PREFETCH;
  }
  
  /**
   * @brief Inserts the decoder on the front of the forward stage of the pipeline.
   * The op is moved into the pipeline and left in a default state.
   *
   * Decoders are special ops that can only appear once in the pipeline, and must
   * appear first. Decoders can also have data dependent shape inference methods
   * Adding the Decoder to forward stage implies that their are no prefetch ops. 
   * If this is not the case, 'Build()' will throw an error.
   */
  inline void AddDecoder(Decoder<GPUBackend> &dec) {
    NDLL_ENFORCE(!built_, "Alterations to the pipeline after "
        "\"Build()\" has been called are not allowed");
    NDLL_ENFORCE(decode_location_ == DECODE_NONE,
        "A Decoder already exists in the pipeline");
    forward_ops_.insert(prefetch_ops_.begin(), std::move(dec));
    decode_location_ = DECODE_FORWARD;
  }

  
  /**
   * @brief Performs some checks on the user-constructed pipeline, setups data
   * for intermediate results, and marks as ready for execution.
   */
  inline void Build() {
    // Make sure the decoder is the first op in the pipeline
    if (decode_location_ == DECODE_FORWARD) {
      NDLL_ENFORCE(prefetch_ops_.size() == 0,
          "The Decoder is set to occur in the forward "
          "pipeline stage but prefetch operators exist");
    }

    // Create buffers for intermediate pipeline results. We need 1
    // buffer for each cpu side op output, and 1 buffer for each
    // gpu side op input. The input to the pipeline and the output
    // to the pipeline are passed into the "RunPrefetch" and
    // "RunForward" methods
    for (int i = 0; i < prefetch_ops_.size(); ++i) {
      BatchPtr<CPUBackend> tmp_cpu;
      cpu_buffers_.push_back(std::move(tmp_cpu));
    }
    for (int i = 0; i < forward_ops_.size(); ++i) {
      BatchPtr<GPUBackend> tmp_gpu;
      gpu_buffers_.push_back(std::move(tmp_gpu));
    }

    // Even though it is not managed by the pipeline, we need to keep track of
    // and resize the output batch for the case where the output shape
    // depends on something decided by one of the operators. For the case with
    // determinitic output shape, this resize should just be a no-op after we
    // resize the output buffer on the first forward pass.
    intermediate_shapes_.resize(cpu_buffers_.size() + gpu_buffers_.size() + 1);
    
    // TODO(tgale): Is it actually worth enforcing this? We need
    // to do this setup before "Run*" is called but we also don't
    // really want to check this flag every time those methods
    // are called. For now we will check.
    built_ = true;
  }

  /**
   * @brief Run the prefetch stage of the pipeline
   */
  inline void RunPrefetch(Batch<CPUBackend> *input) {
    NDLL_ENFORCE(built_,
        "\"Build()\" must be called before the pipeline is executed");
    Index batch_size = input->ndatum();
    NDLL_ENFORCE(batch_size > 0);

    // Size all the intermediate shapes for threads to write into
    for (auto &shape : intermediate_shapes_) {
      shape.resize(batch_size);
    }
    
    for (Index i = 0; i < batch_size; ++i) {
      // TODO(tgale): Save all the dims and resize the intermediate buffers. Look
      // into what temporary objects are created to avoid unescesary cost.
      
      // Run type inference for this image on the whole pipeline
      thread_pool_.DoWorkWithID(std::bind(
              [this, &input] (int data_idx, int tid) {
                Datum<CPUBackend> datum(input, data_idx);
                vector<Index> dims;
                intermediate_shapes_[0][data_idx] =
                  prefetch_ops_[0].InferOutputShape(datum);
                
                for (int j = 1; j < prefetch_ops_.size(); ++j) {
                  datum.Reset(cpu_buffers_[j-1].get(), data_idx);
                  intermediate_shapes_[j][data_idx] =
                    prefetch_ops_[j].InferOutputShape(datum);
                }
              }, i, std::placeholders::_1));
    }
    thread_pool_.WaitForWork();
  }

  /**
   * @brief Run the forward stage of the pipeline
   */
  inline void RunForward() {
    // Launch all the kernels, then use the stream pool to insert
    // events to enforce synchronization behavior
    NDLL_ENFORCE(built_,
        "\"Build()\" must be called before the pipeline is executed");
  }
  
  DISABLE_COPY_MOVE_ASSIGN(Pipeline);
private:
  enum DecodeLocation {
    DECODE_NONE,
    DECODE_PREFETCH,
    DECODE_FORWARD
  };
  DecodeLocation decode_location_;
  bool built_;
  
  StreamPool stream_pool_;
  ThreadPool thread_pool_;

  // TODO(tgale): Add support for GPU decoders by moving
  // the dec into the vectors for simplified execution.
  Decoder<CPUBackend> decoder_;
  vector<Operator<CPUBackend>> prefetch_ops_;
  vector<Operator<GPUBackend>> forward_ops_;

  // Batch objects to store intermediate results of the
  // pipeline. Could probably do this with more efficient
  // memory usage than just 1 buffer per intermediate...
  template <typename T>
  using BatchPtr = std::unique_ptr<Batch<T>>;
  vector<BatchPtr<CPUBackend>> cpu_buffers_;
  vector<BatchPtr<GPUBackend>> gpu_buffers_;

  // Vectors to keep track of the shape of each sample
  // at each stage as collected during the shape inference
  // pass. We pre-allocate these so threads can directly
  // write to the appropriate locations.
  vector<vector<Dims>> intermediate_shapes_;
};

} // namespace ndll

#endif // NDLL_PIPELINE_PIPELINE_H_
