#ifndef NDLL_PIPELINE_PIPELINE_H_
#define NDLL_PIPELINE_PIPELINE_H_

#include <functional>
#include <memory>

#include "ndll/common.h"
#include "ndll/pipeline/data/batch.h"
#include "ndll/pipeline/data/tensor.h"
#include "ndll/pipeline/operators/operator.h"
#include "ndll/pipeline/util/stream_pool.h"
#include "ndll/pipeline/util/thread_pool.h"

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
    decode_location_(DECODE_NONE), built_(false), 
    stream_pool_(main_stream, max_streams, non_blocking),
    thread_pool_(num_threads) {}
  
  ~Pipeline() = default;

  /**
   * @brief Adds an op to the prefetch stage of the pipeline. The input op is
   * moved into the pipeline and left in a default state
   */
  inline void AddPrefetchOp(const Operator<CPUBackend> &op) {
    NDLL_ENFORCE(!built_, "Alterations to the pipeline after "
        "\"Build()\" has been called are not allowed");
    OpPtr<CPUBackend> tmp(op.Clone());
    prefetch_ops_.push_back(std::move(tmp));
  }

  /**
   * @brief Adds an op to the forward stage of the pipeline. The input op is
   * moved into the pipeline and left in a default state
   */
  inline void AddForwardOp(const Operator<GPUBackend> &op) {
    NDLL_ENFORCE(!built_, "Alterations to the pipeline after "
        "\"Build()\" has been called are not allowed");
    OpPtr<GPUBackend> tmp(op.Clone());
    forward_ops_.push_back(std::move(tmp));
  }

  /**
   * @brief Inserts the decoder on the front of the prefetch stage of the pipeline.
   * The op is moved into the pipeline and left in a default state
   */
  inline void AddDecoder(const Operator<CPUBackend> &dec) {
    NDLL_ENFORCE(!built_, "Alterations to the pipeline after "
            "\"Build()\" has been called are not allowed");
    NDLL_ENFORCE(decode_location_ == DECODE_NONE,
        "A Decoder already exists in the pipeline");
    OpPtr<CPUBackend> tmp(dec.Clone());
    prefetch_ops_.insert(prefetch_ops_.begin(), std::move(tmp));
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
  inline void AddDecoder(const Operator<GPUBackend> &dec) {
    NDLL_ENFORCE(!built_, "Alterations to the pipeline after "
        "\"Build()\" has been called are not allowed");
    NDLL_ENFORCE(decode_location_ == DECODE_NONE,
        "A Decoder already exists in the pipeline");
    OpPtr<GPUBackend> tmp(dec.Clone());
    forward_ops_.insert(forward_ops_.begin(), std::move(tmp));
    decode_location_ = DECODE_FORWARD;
  }

  
  /**
   * @brief Performs some checks on the user-constructed pipeline, setups data
   * for intermediate results, and marks as ready for execution. Optionally
   * takes in the input type if the first ops output type is input type dependent
   */
  inline void Build(const TypeMeta *input_type_ptr = nullptr) {
    // Make sure the decoder is the first op in the pipeline
    if (decode_location_ == DECODE_FORWARD) {
      NDLL_ENFORCE(prefetch_ops_.size() == 0,
          "The Decoder is set to occur in the forward "
          "pipeline stage but prefetch operators exist");
    }

    TypeMeta input_type;
    if (input_type_ptr != nullptr) {
      input_type = *input_type_ptr;
    }
    
    // Create buffers for intermediate pipeline results. We need 1
    // buffer for each cpu side op output, and 1 buffer for each
    // gpu side op input. The input to the pipeline and the output
    // to the pipeline are passed into the "RunPrefetch" and
    // "RunForward" methods
    //
    // For the Prefetch ops, we also need to set the output buffer
    // types so that the memory can be allocated prior to wrapping
    // individual samples in 'Datum' objects. The forward ops get
    // the whole batch at once, so we can just call their
    // 'SetOutputType()' methods before launching the kernels
    for (size_t i = 0; i < prefetch_ops_.size(); ++i) {
      BatchPtr<CPUBackend> tmp_cpu(new Batch<CPUBackend>);
      cpu_buffers_.push_back(std::move(tmp_cpu));
      prefetch_ops_[i]->SetOutputType(cpu_buffers_[i].get(), input_type);
      input_type = cpu_buffers_[i]->type();
    }
    for (size_t i = 0; i < forward_ops_.size(); ++i) {
      BatchPtr<GPUBackend> tmp_gpu(new Batch<GPUBackend>);
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
   *
   * TODO(tgale): We also want to be able to run this with the first op reading the data
   * from the database as well. Running in this fashion we may see some slowdown because
   * all the reading of data will be serialized before this function can be called. 
   *
   * We could also add a data reader to this if we want. Some frameworks may serialize
   * data reading to deal with non-thread-safe databases, we'll want to be able to
   * overlap reading with processing for the previous datum, so we'll have to do it in
   * this first loop.
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
                intermediate_shapes_[0][data_idx] =
                  prefetch_ops_[0]->InferOutputShape(datum);
                datum.Resize(intermediate_shapes_[0][data_idx]);
                
                // DEBUG
                // for (auto &val : intermediate_shapes_[0][data_idx]) cout << val << " ";
                // cout << endl;
                
                for (size_t j = 1; j < prefetch_ops_.size(); ++j) {
                  intermediate_shapes_[j][data_idx] =
                    prefetch_ops_[j]->InferOutputShape(datum);
                  datum.Resize(intermediate_shapes_[j][data_idx]);
                }
              }, i, std::placeholders::_1));
    }
    thread_pool_.WaitForWork();

    // TODO(tgale): These resizes set the dims for the buffers but not the types.
    // We need some way to get the types for these buffers from the operators
    // so that when we wrap them with datum the type has already been set and the
    // memory has been allocated. We can do a pass over the ops in the 'Build()'
    // function and call 'data<T>()' on the nullptr buffers to set the type
    
    // Resize the intermidate buffers
    for (size_t i = 0; i < cpu_buffers_.size(); ++i) {
      cpu_buffers_[i]->Resize(intermediate_shapes_[i]);
    }
    for (size_t i = 0; i < gpu_buffers_.size(); ++i) {
      gpu_buffers_[i]->Resize(
          intermediate_shapes_[i + cpu_buffers_.size()]
          );
    }

    // Execute all the prefetch ops
    for (Index i = 0; i < batch_size; ++i) {
      thread_pool_.DoWorkWithID(std::bind(
              [this, &input] (int data_idx, int tid) {
                // We're going to ping-pong back and forth between these Datums
                // So we can cut the number of calls to "Reset()" in half.
                vector<Datum<CPUBackend>> datums(2);
                datums[0].Reset(input, data_idx);
                datums[1].Reset(cpu_buffers_[0].get(), data_idx);
                
                prefetch_ops_[0]->Run(datums[0], &datums[1]);
                for (size_t j = 1; j < prefetch_ops_.size(); ++j) {
                  // Get the other datum to output this ops result into
                  datums[!(j&1)].Reset(cpu_buffers_[j].get(), data_idx);
                  prefetch_ops_[j]->Run(datums[j&1], &datums[!(j&1)]);
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
  // How are we supposed to know where the decoder is and
  // whether to give it data dependent shape infer? Is
  // this guaranteed by the setup we currently have?
  template <typename T>
  using OpPtr = std::unique_ptr<Operator<T>>;
  vector<OpPtr<CPUBackend>> prefetch_ops_;
  vector<OpPtr<GPUBackend>> forward_ops_;

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
