#ifndef NDLL_PIPELINE_PIPELINE_H_
#define NDLL_PIPELINE_PIPELINE_H_

#include <functional>
#include <memory>

#include "ndll/common.h"
#include "ndll/pipeline/data/batch.h"
#include "ndll/pipeline/data/tensor.h"
#include "ndll/pipeline/operators/copy_op.h"
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
  inline Pipeline(int batch_size, int num_threads, cudaStream_t main_stream,
      int max_streams,  bool non_blocking) :
    decode_location_(DECODE_NONE), built_(false), batch_size_(batch_size),
    stream_pool_(new StreamPool(main_stream, max_streams, non_blocking)),
    thread_pool_(num_threads) {
    NDLL_ENFORCE(batch_size_ > 0);
  }
  
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
   * for intermediate results, and marks as ready for execution.
   */
  inline void Build(TypeMeta input_type) {
    // Make sure the decoder is the first op in the pipeline
    if (decode_location_ == DECODE_FORWARD) {
      NDLL_ENFORCE(prefetch_ops_.size() == 0,
          "The Decoder is set to occur in the forward "
          "pipeline stage but prefetch operators exist");
    }
    NDLL_ENFORCE(prefetch_ops_.size() + forward_ops_.size() > 0,
        "Pipeline must have at least one operator");
    
    // Create buffers for intermediate pipeline results. We need 1
    // buffer for each cpu side op output, and 1 buffer for each
    // gpu side op input. The input to the pipeline and the output
    // to the pipeline are passed into the "RunPrefetch" and
    // "RunForward" methods
    //
    // For the Prefetch ops, we also need to set the output buffer
    // types so that the memory can be allocated prior to wrapping
    // individual samples in 'Datum' objects. The forward ops get
    // the whole batch at once, but we call these methods here as
    // well to avoid as much work as possible during training
    for (size_t i = 0; i < prefetch_ops_.size(); ++i) {
      BatchPtr<CPUBackend> tmp_cpu(new Batch<CPUBackend>);
      cpu_buffers_.push_back(std::move(tmp_cpu));
      prefetch_ops_[i]->SetOutputType(cpu_buffers_[i].get(), input_type);
      input_type = cpu_buffers_[i]->type();
    }
    // We always need at least one gpu buffer to copy into
    BatchPtr<GPUBackend> tmp_gpu(new Batch<GPUBackend>);
    gpu_buffers_.push_back(std::move(tmp_gpu));
    gpu_buffers_[0]->set_type(input_type);
    
    for (size_t i = 1; i < forward_ops_.size(); ++i) {
      BatchPtr<GPUBackend> tmp_gpu(new Batch<GPUBackend>);
      gpu_buffers_.push_back(std::move(tmp_gpu));
      forward_ops_[i-1]->SetOutputType(gpu_buffers_[i].get(), input_type);
      input_type = gpu_buffers_[i]->type();
    }

    // If we don't have any user-defined forward ops, add a CopyOp
    // that will copy the data into the output batch
    if (forward_ops_.size() == 0) {
      OpPtr<GPUBackend> tmp(new CopyOp<GPUBackend>);
      forward_ops_.push_back(std::move(tmp));
    }
    
    // Even though it is not managed by the pipeline, we need to keep track of
    // and resize the output batch for the case where the output shape
    // depends on something decided by one of the operators. For the case with
    // determinitic output shape, this resize should just be a no-op after we
    // resize the output buffer on the first forward pass.
    intermediate_shapes_.resize(cpu_buffers_.size() + gpu_buffers_.size() + 1);

    // Size all the intermediate shapes for threads to write into
    for (auto &shape : intermediate_shapes_) {
      shape.resize(batch_size_);
    }

    // Set important meta-data for all the operators in the pipeline
    for (auto &op : prefetch_ops_) {
      op->set_num_threads(num_thread());
      op->set_batch_size(batch_size_);
      op->set_stream_pool(stream_pool_);
    }
    for (auto &op : forward_ops_) {
      op->set_num_threads(num_thread());
      op->set_batch_size(batch_size_);
      op->set_stream_pool(stream_pool_);
    }
    
    // Mark the pipeline as built so we know it is safe to run
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
    NDLL_ENFORCE(input->ndatum() == batch_size_,
        "Calling batch size does not match pipeline parameter");
    
    for (Index i = 0; i < batch_size_; ++i) {
      // Run type inference for this image on the whole pipeline
      thread_pool_.DoWorkWithID(std::bind(
              [this, &input] (int data_idx, int tid) {
                // Get the output shape for the cpu-side results
                Datum<CPUBackend> datum(input, data_idx);
                intermediate_shapes_[0][data_idx] =
                  prefetch_ops_[0]->InferOutputShape(datum, data_idx);
                datum.Resize(intermediate_shapes_[0][data_idx]);
                
                for (size_t j = 1; j < prefetch_ops_.size(); ++j) {
                  intermediate_shapes_[j][data_idx] =
                    prefetch_ops_[j]->InferOutputShape(datum, data_idx);
                  datum.Resize(intermediate_shapes_[j][data_idx]);
                }
                
                // Save the shape of the gpu-side copy buffer
                int offset = prefetch_ops_.size();
                intermediate_shapes_[offset][data_idx] =
                  intermediate_shapes_[offset - 1][data_idx];

                // Get the output shape for the gpu-side results
                ++offset;
                Datum<GPUBackend> datum_gpu(datum.shape());
                for (size_t j = 0; j < forward_ops_.size(); ++j) {
                  intermediate_shapes_[j + offset][data_idx] =
                    forward_ops_[j]->InferOutputShape(datum_gpu, data_idx);
                  datum_gpu.Resize(intermediate_shapes_[j + offset][data_idx]);
                }
              }, i, std::placeholders::_1));
    }
    thread_pool_.WaitForWork();
    
    // Resize the intermidate buffers
    for (size_t i = 0; i < cpu_buffers_.size(); ++i) {
      cpu_buffers_[i]->Resize(intermediate_shapes_[i]);
    }

    // Resize the gpu-size intermediate buffers & set the types
    for (size_t i = 0; i < gpu_buffers_.size(); ++i) {
      gpu_buffers_[i]->Resize(
          intermediate_shapes_[i + cpu_buffers_.size()]);
    }

    // Execute all the prefetch ops
    for (Index i = 0; i < batch_size_; ++i) {
      thread_pool_.DoWorkWithID(std::bind(
              [this, &input] (int data_idx, int tid) {
                // We're going to ping-pong back and forth between these Datums
                // So we can cut the number of calls to "Reset()" in half.
                vector<Datum<CPUBackend>> datums(2);
                datums[0].Reset(input, data_idx);
                datums[1].Reset(cpu_buffers_[0].get(), data_idx);
                
                prefetch_ops_[0]->Run(datums[0], &datums[1], data_idx);
                for (size_t j = 1; j < prefetch_ops_.size(); ++j) {
                  // Get the other datum to output this ops result into
                  datums[!(j&1)].Reset(cpu_buffers_[j].get(), data_idx);
                  prefetch_ops_[j]->Run(datums[j&1], &datums[!(j&1)], data_idx);
                }
              }, i, std::placeholders::_1));
    }
    thread_pool_.WaitForWork();
  }

  /**
   * @brief Copies the result of the prefetch stage into the input 
   * buffer for the forward stage
   */
  inline void RunCopy() {
    Batch<CPUBackend> &src = *cpu_buffers_[cpu_buffers_.size()-1];
    Batch<GPUBackend> *dst = gpu_buffers_[0].get();
    
    // Copy the data to the GPU in the main stream
    CUDA_CALL(cudaMemcpyAsync(
            dst->raw_data(),
            src.raw_data(),
            src.nbytes(),
            cudaMemcpyHostToDevice,
            stream_pool_->GetStream()));
  }
  
  /**
   * @brief Run the forward stage of the pipeline
   */
  inline void RunForward(Batch<GPUBackend> *output) {
    NDLL_ENFORCE(built_,
        "\"Build()\" must be called before the pipeline is executed");

    // TODO(tgale): Having to synchronize here is not ideal. We do this
    // to make sure that the copy from "RunCopy" has completed before
    // ops are free to run their kernels in other streams. Is there a
    // better way to prevent this from happening?
    CUDA_CALL(cudaStreamSynchronize(stream_pool_->GetStream()));

    // TODO(tgale): I think this type setting is redundant...remove it
    
    // Run all the forward ops
    TypeMeta input_type = gpu_buffers_[0]->type();
    for (size_t i = 1; i < forward_ops_.size(); ++i) {
      forward_ops_[i-1]->SetOutputType(gpu_buffers_[i].get(), input_type);
      forward_ops_[i-1]->Run(*gpu_buffers_[i-1], gpu_buffers_[i].get());
    }

    // Run the last op to output into the passed in batch. We are
    // guaranteed To have this op because we insert a CopyOp if
    // there are no user defined ops in the forward stage
    int idx = forward_ops_.size() - 1;
    forward_ops_[idx]->SetOutputType(output, input_type);
    output->Resize(intermediate_shapes_[intermediate_shapes_.size()-1]);
    forward_ops_[idx]->Run(*gpu_buffers_[idx], output);

    // insert cudaEvents and 'cudaStreamWaitEvent()'s into all extra
    // streams used by the ops to ensure proper synchronization
    // behavior with the main stream
    stream_pool_->SetMainStreamEvents();
  }

  // Accessor for the stream pool
  inline std::shared_ptr<StreamPool> stream_pool() {
    return stream_pool_;
  }

  // Accessor for the size of the thread pool
  inline int num_thread() const {
    return thread_pool_.size();
  }

  inline void Print() const {
    // Print all the operators in the pipeline
    cout << "Printing Pipeline Operators: " << endl;
    cout << "[Prefetch Ops]: " << endl;
    for (auto &op : prefetch_ops_) {
      cout << op->name() << endl;
    }
    cout << "[Forward Ops]: " << endl;
    for (auto &op : forward_ops_) {
      cout << op->name() << endl;
    }
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

  int batch_size_;
  std::shared_ptr<StreamPool> stream_pool_;
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
