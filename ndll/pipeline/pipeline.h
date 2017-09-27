#ifndef NDLL_PIPELINE_PIPELINE_H_
#define NDLL_PIPELINE_PIPELINE_H_

#include <functional>
#include <memory>

#include "ndll/common.h"
#include "ndll/pipeline/data_reader.h"
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
      int max_streams,  bool non_blocking, int device_id) :
    decode_location_(DECODE_NONE), built_(false), batch_size_(batch_size),
    stream_pool_(new StreamPool(main_stream, max_streams, non_blocking)),
    thread_pool_(num_threads, device_id), input_datum_(batch_size) {
    NDLL_ENFORCE(batch_size_ > 0);
    // Set the data type for our mega-buffers
    mega_buffer_.template data<uint8>();
    mega_buffer_gpu_.template data<uint8>();
    
    // Note: We do not set device/thread affinity in the pipeline anywhere
    // because frameworks like C2 could have different threads running
    // through this code. We do however require that the device is set
    // to match the input device id on construction before any of the
    // pipeline method are called.
    //
    // In the thread pool that we control, we configure affinity to ensure
    // good NUMA configuration and correct device settings
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
   * @brief Adds the input DataReader to the pipeline. The DataReader will
   * provide access to single data samples during execution, and allow us
   * to overlap the reading of data with the processing of data in the
   * thread pool
   */
  inline void AddDataReader(const DataReaderBase<CPUBackend> &reader) {
    NDLL_ENFORCE(!built_, "Alterations to the pipeline after "
        "\"Build()\" has been called are not allowed");
    data_reader_.reset(reader.Clone());
  }
  
  /**
   * @brief Performs some checks on the user-constructed pipeline, setups data
   * for intermediate results, and marks as ready for execution.
   *
   * @param output the batch to output the results of the pipeline into
   */
  void Build(shared_ptr<Batch<GPUBackend>> output);

  /**
   * @brief Run the prefetch stage of the pipeline
   */
  void RunPrefetch();

  /**
   * @brief Copies the result of the prefetch stage into the input 
   * buffer for the forward stage
   */
  void RunCopy();

  // Note: While RunPrefetch & RunCopy can be run in prefetch threads to
  // overlap with the forward-backward pass, RunForward must be called
  // Before RunPrefetch & RunCopy can be called again. Thus, we do not
  // prefetch more than a single batch at once
  
  /**
   * @brief Run the forward stage of the pipeline into the output buffer
   */
  void RunForward();

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
  // Return the nearest multiple of 8 that is >= base_ptr_offset
  inline size_t round_up_to_8(size_t base_ptr_offset) {
    if (base_ptr_offset & 7) {
      base_ptr_offset = (base_ptr_offset & 7) + 8;
    }
    return base_ptr_offset;
  }
  
  // Helper function to setup mega-buffer and distribute
  // sub-buffers to the ops in the forward pass
  void MegaBufferSetupAndDistribution();
  
  enum DecodeLocation {
    DECODE_NONE,
    DECODE_PREFETCH,
    DECODE_FORWARD
  };
  DecodeLocation decode_location_;
  bool built_;

  int batch_size_;
  shared_ptr<StreamPool> stream_pool_;
  ThreadPool thread_pool_;

  template <typename T>
  using OpPtr = unique_ptr<Operator<T>>;
  vector<OpPtr<CPUBackend>> prefetch_ops_;
  vector<OpPtr<GPUBackend>> forward_ops_;

  // Batch objects to store intermediate results of the
  // pipeline. Could probably do this with more efficient
  // memory usage than just 1 buffer per intermediate...
  //
  // Note: we use shared_ptrs here because our output buffer
  // is a shared_ptr and we want to be able to simply insert
  // it into this vector to make the copy & forward simpler
  template <typename T>
  using BatchPtr = shared_ptr<Batch<T>>;
  vector<BatchPtr<CPUBackend>> cpu_buffers_;
  vector<BatchPtr<GPUBackend>> gpu_buffers_;

  // Batch to output the result of the pipeline to
  shared_ptr<Batch<GPUBackend>> output_buffer_;
  
  // DataReader to query for datum during execution
  unique_ptr<DataReaderBase<CPUBackend>> data_reader_;
  vector<Datum<CPUBackend>> input_datum_;
  
  // Vectors to keep track of the shape of each sample
  // at each stage as collected during the shape inference
  // pass. We pre-allocate these so threads can directly
  // write to the appropriate locations.
  vector<vector<Dims>> intermediate_shapes_;

  // Tensors to store all batched op parameters for ops in
  // the forward pass. Enables single copy of paramters
  // instead of copies per operator
  Tensor<CPUBackend> mega_buffer_;
  Tensor<GPUBackend> mega_buffer_gpu_;
};

template <typename CPUBackend, typename GPUBackend>
void Pipeline<CPUBackend, GPUBackend>::Build(shared_ptr<Batch<GPUBackend>> output) {
  NDLL_ENFORCE(output != nullptr, "Output batch must point to valid Batch object");
  output_buffer_ = output;
  
  // Make sure the decoder is the first op in the pipeline
  if (decode_location_ == DECODE_FORWARD) {
    NDLL_ENFORCE(prefetch_ops_.size() == 0,
        "The Decoder is set to occur in the forward "
        "pipeline stage but prefetch operators exist");
  }
  NDLL_ENFORCE(prefetch_ops_.size() + forward_ops_.size() > 0,
      "Pipeline must have at least one operator");
  NDLL_ENFORCE(data_reader_ != nullptr,
      "The Pipeline must have a data reader to be built");

  // Get the input data type to the pipeline
  TypeMeta input_type = data_reader_->OutputType();
  NDLL_ENFORCE(input_type.id() != NO_TYPE);
    
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

  if (forward_ops_.size() == 0) {
    // Copy directly into the output buffer
    gpu_buffers_.push_back(output_buffer_);
    gpu_buffers_[0]->set_type(input_type);
  } else {
    // Create a buffer to copy into
    BatchPtr<GPUBackend> tmp_gpu(new Batch<GPUBackend>);
    gpu_buffers_.push_back(std::move(tmp_gpu));
    gpu_buffers_[0]->set_type(input_type);
  }

  // Create the rest of the input buffers
  for (size_t i = 1; i < forward_ops_.size(); ++i) {
    BatchPtr<GPUBackend> tmp_gpu(new Batch<GPUBackend>);
    gpu_buffers_.push_back(std::move(tmp_gpu));
    forward_ops_[i-1]->SetOutputType(gpu_buffers_[i].get(), input_type);
    input_type = gpu_buffers_[i]->type();
  }

  // If we do have forward ops, add the output buffer to the
  // gpu_buffers and set its output type from the last op
  if (forward_ops_.size() > 0) {
    gpu_buffers_.push_back(output_buffer_);
    forward_ops_[forward_ops_.size()-1]->
      SetOutputType(output_buffer_.get(), input_type);
  }

  // TODO(tgale): We no longer need the copy op as we have access to the output
  // buffer from the start. If there are not ops in the forward pass, make the
  // copy occur directly into the output batch. This is a bit tricky, as we
  // unfortunately can't just insert the output buffer into the gpu_buffers
  // vector. We can easily do it by adding conditions that check if the
  // forward stage is empty, but this is icky. Find a nice way to do this
  
  // If we don't have any user-defined forward ops, add a CopyOp
  // that will copy the data into the output batch
  // if (forward_ops_.size() == 0) {
  //   OpPtr<GPUBackend> tmp(new CopyOp<GPUBackend>);
  //   forward_ops_.push_back(std::move(tmp));
  // }

  // Vector of shapes for all of the intermediate operator results as
  // well as the output gpu buffer
  intermediate_shapes_.resize(cpu_buffers_.size() + gpu_buffers_.size());

  // Size all the intermediate shapes for threads to write into
  for (auto &shape : intermediate_shapes_) {
    shape.resize(batch_size_);
  }

  // Set important meta-data for all the operators in the pipeline
  for (auto &op : prefetch_ops_) {
    op->set_num_threads(thread_pool_.size());
    op->set_batch_size(batch_size_);
    op->set_stream_pool(stream_pool_);
  }
  for (auto &op : forward_ops_) {
    op->set_num_threads(thread_pool_.size());
    op->set_batch_size(batch_size_);
    op->set_stream_pool(stream_pool_);
  }
    
  // Mark the pipeline as built so we know it is safe to run
  built_ = true;
}

template <typename CPUBackend, typename GPUBackend>
void Pipeline<CPUBackend, GPUBackend>::RunPrefetch() {
  NDLL_ENFORCE(built_, "\"Build()\" must be called before the pipeline is executed");
    
  for (Index i = 0; i < batch_size_; ++i) {
    // Get a Datum from the reader
    data_reader_->Read(&input_datum_[i]);
      
    // Run type inference for this image on the whole pipeline
    thread_pool_.DoWorkWithID(std::bind(
            [this] (int data_idx, int tid) {
              // Get the output shape for the cpu-side results
              intermediate_shapes_[0][data_idx] =
                prefetch_ops_[0]->InferOutputShape(input_datum_[data_idx], data_idx, tid);
                
              Datum<CPUBackend> datum(intermediate_shapes_[0][data_idx]);
              for (size_t j = 1; j < prefetch_ops_.size(); ++j) {
                intermediate_shapes_[j][data_idx] =
                  prefetch_ops_[j]->InferOutputShape(datum, data_idx, tid);
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
                  forward_ops_[j]->InferOutputShape(datum_gpu, data_idx, tid);
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

  // Setup the mega-buffer & distribute sub-buffers to
  // the ops in the forward pass
  MegaBufferSetupAndDistribution();
    
  // Execute all the prefetch ops
  for (Index i = 0; i < batch_size_; ++i) {
    thread_pool_.DoWorkWithID(std::bind(
            [this] (int data_idx, int tid) {
              // We're going to ping-pong back and forth between these Datums
              // So we can cut the number of calls to "Reset()" in half.
              vector<Datum<CPUBackend>> datums(2);
              datums[1].Reset(cpu_buffers_[0].get(), data_idx);
                
              prefetch_ops_[0]->Run(input_datum_[data_idx], &datums[1], data_idx, tid);
              for (size_t j = 1; j < prefetch_ops_.size(); ++j) {
                // Get the other datum to output this ops result into
                datums[!(j&1)].Reset(cpu_buffers_[j].get(), data_idx);
                prefetch_ops_[j]->Run(datums[j&1], &datums[!(j&1)], data_idx, tid);
              }

              // Give the forward ops a chance to set up
              // parameters for their kernel launches
              for (size_t j = 0; j < forward_ops_.size(); ++j) {
                forward_ops_[j]->BatchedParameterSetupPerDatum(
                    *gpu_buffers_[j], gpu_buffers_[j+1].get(), data_idx, tid);
              }
            }, i, std::placeholders::_1));
  }
  thread_pool_.WaitForWork();
}

template <typename CPUBackend, typename GPUBackend>
void Pipeline<CPUBackend, GPUBackend>::RunCopy() {
  Batch<CPUBackend> &src = *cpu_buffers_[cpu_buffers_.size()-1];
  Batch<GPUBackend> *dst = gpu_buffers_[0].get();
    
  // Copy the data to the GPU in the main stream
  CUDA_CALL(cudaMemcpyAsync(
          dst->raw_data(),
          src.raw_data(),
          src.nbytes(),
          cudaMemcpyHostToDevice,
          stream_pool_->GetStream()));

  // Copy the mega-buffer to GPU in the main stream
  CUDA_CALL(cudaMemcpyAsync(
          mega_buffer_gpu_.raw_data(),
          mega_buffer_.raw_data(),
          mega_buffer_.nbytes(),
          cudaMemcpyHostToDevice,
          stream_pool_->GetStream()));
}

template <typename CPUBackend, typename GPUBackend>
void Pipeline<CPUBackend, GPUBackend>::RunForward() {
  NDLL_ENFORCE(built_,
      "\"Build()\" must be called before the pipeline is executed");

  // TODO(tgale): Having to synchronize here is not ideal. We do this
  // to make sure that the copy from "RunCopy" has completed before
  // ops are free to run their kernels in other streams. Is there a
  // better way to prevent this from happening?
  CUDA_CALL(cudaStreamSynchronize(stream_pool_->GetStream()));

  // Run all the forward ops
  for (size_t i = 0; i < forward_ops_.size(); ++i) {
    forward_ops_[i]->Run(*gpu_buffers_[i], gpu_buffers_[i+1].get());
  }

  // Run the last op to output into the passed in batch. We are
  // guaranteed To have this op because we insert a CopyOp if
  // there are no user defined ops in the forward stage
  // int idx = forward_ops_.size() - 1;
  // forward_ops_[idx]->SetOutputType(output, gpu_buffers_[idx]->type());
  // output->Resize(intermediate_shapes_[intermediate_shapes_.size()-1]);
  // forward_ops_[idx]->Run(*gpu_buffers_[idx], output);

  // insert cudaEvents and 'cudaStreamWaitEvent()'s into all extra
  // streams used by the ops to ensure proper synchronization
  // behavior with the main stream
  stream_pool_->SetMainStreamEvents();
}

template <typename CPUBackend, typename GPUBackend>
void Pipeline<CPUBackend, GPUBackend>::MegaBufferSetupAndDistribution() {
  TimeRange _tr("mega-buffer-setup");
  // Query all the forward ops for mega-buffer sizes
  size_t total_bytes = 0;
  vector<size_t> offsets;
  vector<int> num_buff_for_op(forward_ops_.size(), 0);
  for (size_t i = 0; i < forward_ops_.size(); ++i) {
    const vector<size_t>& sizes =
      forward_ops_[i]->GetBatchedParameterSize();
    num_buff_for_op[i] = sizes.size();
    for (auto &num_bytes : sizes) {
      // Align the start of each buffer to 8-bytes
      size_t aligned_num_bytes = round_up_to_8(num_bytes);
      offsets.push_back(total_bytes);
      total_bytes += aligned_num_bytes;
    }
  }
  offsets.push_back(total_bytes);
    
  // Allocate the tensors on host & device
  mega_buffer_.Resize({(Index)total_bytes});
  mega_buffer_gpu_.Resize({(Index)total_bytes});

  // Hand out SubTensors for all the ops batched parameters
  int buffer_id = 0;
  for (size_t i = 0; i < forward_ops_.size(); ++i) {
    vector<CPUSubTensor> cpu_buffers(num_buff_for_op[i]);
    vector<GPUSubTensor> gpu_buffers(num_buff_for_op[i]);
    for (int j = 0; j < num_buff_for_op[i]; ++j) {
      CPUSubTensor sub_buffer(&mega_buffer_,
          offsets[buffer_id], offsets[buffer_id+1]);
      GPUSubTensor sub_buffer_gpu(&mega_buffer_gpu_,
          offsets[buffer_id], offsets[buffer_id+1]);
      cpu_buffers[j] = sub_buffer;
      gpu_buffers[j] = sub_buffer_gpu;
      ++buffer_id;
    }
    forward_ops_[i]->
      SetBatchedParameterBuffers(cpu_buffers, gpu_buffers);
    forward_ops_[i]->
      BatchedParameterSetup(*gpu_buffers_[i], gpu_buffers_[i+1].get());
  }
}

} // namespace ndll

#endif // NDLL_PIPELINE_PIPELINE_H_
