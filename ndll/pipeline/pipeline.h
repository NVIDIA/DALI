#ifndef NDLL_PIPELINE_PIPELINE_H_
#define NDLL_PIPELINE_PIPELINE_H_

#include <map>

#include "ndll/common.h"
#include "ndll/pipeline/data_reader.h"
#include "ndll/pipeline/data/backend.h"
#include "ndll/pipeline/data/batch.h"
#include "ndll/pipeline/data/datum.h"
#include "ndll/pipeline/data/tensor.h"
#include "ndll/pipeline/decoder.h"
#include "ndll/pipeline/operator.h"
#include "ndll/pipeline/operators/copy_op.h"
#include "ndll/pipeline/parser.h"
#include "ndll/pipeline/transformer.h"
#include "ndll/pipeline/util/thread_pool.h"
#include "ndll/util/npp.h"

namespace ndll {

/**
 * @brief Organizes and executes the set of operators chosen by the user.
 * Provides optimizations like batched copies to GPU and pre-sizing of
 * buffers.
 *
 * The Pipeline produces a processed batch of data on the GPU. It is 
 * composed of Operators (@ref ndll::Operator<Backend>), and breaks 
 * its execution into 3 phases: 'prefetch', 'copy', and 'forward'. 
 * Operators are added to a specific phase when the pipeline is built 
 * up. Prefetch operators are executed per-image w/ multiple threads, 
 * while Forward operators are executed on an entire batch at once on 
 * the GPU. We currently don't support running all operations (cpu or gpu) 
 * per-image in the prefetch stage, but it wouldn't be too difficult to 
 * enable.
 * 
 * The pipeline manages all memory used in the pipeline. We currently 
 * maintain buffers for all intermediate results, but this central 
 * management means that we could do some tricks (e.g. just using two 
 * buffers and ping-ponging back and forth) to reduce memory requirements
 *
 * The pipeline also provides a mechanism for combining all batched GPU 
 * operation parameters into a single mega-buffer during the prefetch stage. 
 * Operations in the Forward stage are queried for the amount of batched 
 * paramter storage they need after performing shape inference. They are 
 * then  given a chance to set up their batched paramters (both serially 
 * and threaded, depending on the need of the operation) into their chunk 
 * of the mega-buffer. During the copy stage, we copy the output of the 
 * prefetch stage, as well as the mega-buffer to the GPU.
 */
class Pipeline {
public:
  /**
   * @brief Creates a pipeline with `num_threads` worker threads and 
   * working in `main_stream`. 
   *
   * GPU memory and pinned memory allocations cause implicit synchronization of 
   * the device, resulting in very slow startup times as ndll buffer sizes 
   * stabilize. To avoid this slowdown, we optionally take in an estimated size
   * of each image that will be processed in bytes. This hint is used to
   * pre-size buffers, potentially avoiding slow startup if the hint is close
   * to the true amount of memory that will be needed by the largest image to
   * be processed.
   *
   * @param batch_size the size of the batch that should be produced.
   * @param num_threads the number of threads to use in the prefetch stage.
   * @param stream the stream to operate in.
   * @param device_id id of the GPU to operate on.
   * @param set_affinity indicates whether thread affinity should be
   * configured in the thread pool. Defaults to 'true'.
   * @param pixels_per_image_hint Estimated size of each image to be processed.
   * Defaults to 0.
   */
  inline Pipeline(int batch_size, int num_threads, cudaStream_t stream,
      int device_id, bool set_affinity = true, size_t pixels_per_image_hint = 0) :
    decode_location_(DECODE_NONE), built_(false), batch_size_(batch_size),
    stream_(stream), thread_pool_(num_threads, device_id, set_affinity),
    pixels_per_image_hint_(pixels_per_image_hint), data_reader_(nullptr),
    input_datum_(batch_size), data_parser_(nullptr), parsed_datum_(batch_size) {
    NDLL_ENFORCE(batch_size_ > 0);
    // Set the data type for our mega-buffers
    mega_buffer_.template mutable_data<uint8>();
    mega_buffer_gpu_.template mutable_data<uint8>();

    // TODO(tgale): We need to figure out the best way to ensure that the memory
    // this object allocates is stored on the correct NUMA node that we can
    // force on the frameworks. Frameworks like C2 are tricky because any thread
    // could execute this pipe on any iteration, so we'll need a way to force
    // these specfic allocations to go our way without messing with everything
    // else.
  }

  // inline Pipeline(int batch_size, int num_threads, int64 cuda_stream,
  //     int device_id, bool set_affinity = true) :
  //   decode_location_(DECODE_NONE), built_(false),
  //   batch_size_(batch_size), stream_((cudaStream_t)cuda_stream),
  //   thread_pool_(num_threads, device_id, set_affinity),
  //   data_reader_(nullptr), input_datum_(batch_size),
  //   data_parser_(nullptr), parsed_datum_(batch_size) {
  //   NDLL_ENFORCE(batch_size_ > 0);
  //   // Set the data type for our mega-buffers
  //   mega_buffer_.template mutable_data<uint8>();
  //   mega_buffer_gpu_.template mutable_data<uint8>();

  //   // TODO(tgale): We need to figure out the best way to ensure that the memory
  //   // this object allocates is stored on the correct NUMA node that we can
  //   // force on the frameworks. Frameworks like C2 are tricky because any thread
  //   // could execute this pipe on any iteration, so we'll need a way to force
  //   // these specfic allocations to go our way without messing with everything
  //   // else.
  // }
  
  ~Pipeline() = default;

  // TODO(tgale): Add handling of extra inputs and outputs for ops in all pipeline
  // construction methods. Also add setting of batch size & num threads & stream
  
  /**
   * @brief Adds a Decoder to the pipeline. The decoder is either inserted on 
   * the front of the prefetch stage, or the front of the forward stage depending 
   * on the 'stage' setting in the OpSpec.
   *
   * Decoder are special ops that are allowed to have data dependent output shapes.
   * For this reason, they must appear first in the pipeline and can only appear
   * once.
   */
  inline void AddDecoder(const OpSpec &spec) {
    NDLL_ENFORCE(!built_, "Alterations to the pipeline after "
        "\"Build()\" has been called are not allowed");

    // Add some pipeline meta-data and handle extra input/output tensors
    OpSpec spec_copy = PrepareOpSpec(spec);
    
    // Construct the decoder with the input spec
    if (spec_copy.stage() == "Prefetch") {
      OpPtr<CPUBackend> tmp(
          CPUDecoderRegistry::Registry().Create(spec_copy.name(), spec_copy));
      prefetch_ops_.insert(prefetch_ops_.begin(), std::move(tmp));
      decode_location_ = DECODE_PREFETCH;
    } else if (spec_copy.stage() == "Forward") {
      OpPtr<GPUBackend> tmp(
          GPUDecoderRegistry::Registry().Create(spec_copy.name(), spec_copy));
      forward_ops_.insert(forward_ops_.begin(), std::move(tmp));
      decode_location_ = DECODE_FORWARD;
    } else {
      NDLL_FAIL("Invalid stage argument \"" + spec_copy.stage() +
          "\". Stage must be either \"Prefetch\" or \"Forward\"");
    }
  }

  /**
   * @brief Adds a Transformer to the pipeline. The Transformer is either 
   * inserted in the prefetch stage, or the forward stage depending on the 
   * 'stage' setting in the OpSpec.
   */
  inline void AddTransform(const OpSpec &spec) {
    NDLL_ENFORCE(!built_, "Alterations to the pipeline after "
        "\"Build()\" has been called are not allowed");

    // Add some pipeline meta-data and handle extra input/output tensors
    OpSpec spec_copy = PrepareOpSpec(spec);
    
    // Construct the decoder with the input spec
    if (spec_copy.stage() == "Prefetch") {
      OpPtr<CPUBackend> tmp(
          CPUTransformerRegistry::Registry().Create(spec_copy.name(), spec_copy));
      prefetch_ops_.push_back(std::move(tmp));
    } else if (spec_copy.stage() == "Forward") {
      OpPtr<GPUBackend> tmp(
          GPUTransformerRegistry::Registry().Create(spec_copy.name(), spec_copy));
      forward_ops_.push_back(std::move(tmp));
    } else {
      NDLL_FAIL("Invalid stage argument \"" + spec_copy.stage() +
          "\". Stage must be either \"Prefetch\" or \"Forward\"");
    }
  }
  
  /**
   * @brief Adds the input DataReader to the pipeline. The DataReader will
   * provide access to single data samples during execution, and allow us
   * to overlap the reading of data with the processing of data in the
   * thread pool.
   */
  inline void AddDataReader(const OpSpec &spec) {
    NDLL_ENFORCE(!built_, "Alterations to the pipeline after "
        "\"Build()\" has been called are not allowed");
    NDLL_ENFORCE(data_reader_ == nullptr, "Pipeline already "
        "has a DataReader.");
    NDLL_ENFORCE(spec.stage() == "Prefetch",
        "DataReaders only operator in Prefetch stage.");
    
    // Add some pipeline meta-data and handle extra input/output tensors
    OpSpec spec_copy = PrepareOpSpec(spec);

    // Construct the DataReader with the input spec
    data_reader_ = DataReaderRegistry::Registry().Create(
        spec_copy.name(), spec_copy);
  }

  /**
   * @brief Add the input Parser to the pipeline. The parser will be
   * called on the Datum produced by the DataReader. This allows support
   * for custom data formats without altering the basic ops defined for 
   * the pipeline.
   */
  inline void AddParser(const Parser &parser) {
    NDLL_ENFORCE(!built_, "Alterations to the pipeline after "
        "\"Build()\" has been called are not allowed");
    data_parser_.reset(parser.Clone());
  }
  
  /**
   * @brief Performs some checks on the user-constructed pipeline, setups data
   * for intermediate results, and marks as ready for execution.
   */
  void Build();

  /**
   * @brief Run the prefetch stage of the pipeline.
   *
   * The prefetch stage of the pipeline is broken into three phases.
   * In the first phase, all samples are read from the DataReader
   * and are launched into the thread pool to be parsed. In this thread
   * loop, we pass over all Operators in the pipeline to perform shape
   * inference.
   *
   * In the second phase, we take the shapes that we have calculated 
   * and resize all of our intermediate results. We also pass over all
   * ops in the that will be run in the forward stage of the pipeline
   * and query for the number of bytes they need to store their batched
   * paramters. We then allocate our mega buffer to store all ops
   * batched parameters and distribute SubTensors to all the ops that
   * requested batached paramter storage. Finally, we iterate over the
   * forward stage ops one more time to give them a serial section to
   * setup their batched parameters into the mega buffer.
   *
   * In the third phase, we launch into the thread pool to execute all
   * the prefetch ops. After finishing execution of the prefetch ops,
   * each thread also passes over the forward stage ops to give them
   * a change to setup any per-image batched paramters.
   */
  void RunPrefetch();

  /**
   * @brief Copies the result of the prefetch stage into the input 
   * buffer for the forward stage. Also copied all batched parameters
   * to the GPU for the forward stage of computation.
   */
  void RunCopy();

  /**
   * @brief Run the forward stage of the pipeline into the output buffer.
   *
   * This stage is designed to be extremely light weight to minimize cost
   * on the front of the forward pass. All parameter setup and allocations
   * have been done previously, and we simply iterate over the forward
   * stage ops and launch their kernels.
   *
   * Note: While RunPrefetch & RunCopy can be run in prefetch threads to
   * overlap with the forward-backward pass, RunForward must be called
   * Before RunCopy can be called again so that we do not corrupt the
   * results of the previous execution pass before the forward stage
   * is finished with them.
   */
  void RunForward();

  /**
   * @brief Returns a reference to the batch storing the result of the 
   * pipeline. 
   *
   * @param sync Indicates whether this function should synchronize on
   * the pipeline stream before returning. Defaults to 'true'
   */
  const Batch<GPUBackend>& output_batch(bool sync = true) const {
    if (sync) CUDA_CALL(cudaStreamSynchronize(stream()));
    return *gpu_buffers_.back();
  }

  /**
   * @brief Returns the output CPU Tensor with the given name. Performs
   * no synchonization prior to returning the Tensor.
   */
  const Tensor<CPUBackend>& output_tensor(const string &name) const {
    auto it = extra_tensors_.find(name);
    NDLL_ENFORCE(it != extra_tensors_.end(), "Tensor with name\""
        + name + "\" does not exist.");
    return *(it->second);
  }

  /**
   * @brief Returns the output GPU Tensor with the given name. Performs
   * no synchonization prior to returning the Tensor.
   */
  const Tensor<GPUBackend>& output_gpu_tensor(const string &name) const {
    auto it = extra_gpu_tensors_.find(name);
    NDLL_ENFORCE(it != extra_gpu_tensors_.end(), "Tensor with name\""
        + name + "\" does not exist.");
    return *(it->second);
  }
  
  /**
   * @brief Returns the batch size that will be produced by the pipeline.
   */
  int batch_size() const { return batch_size_; }

  /**
   * @brief Returns the number of threads that are used by the pipeline.
   */
  int num_threads() const { return thread_pool_.size(); }
  
  /**
   * @brief Returns the stream that the pipeline is working in.
   */
  cudaStream_t stream() const { return stream_; }

  /**
   * @brief Prints the name of all operators that are in the pipeline.
   */
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

  // Helper function to resize intermediate buffers and
  // handle data sharing for GPU buffers.
  void IntermediateBufferResizeAndSetup();

  // Helper function to setup mega-buffer and distribute
  // sub-buffers to the ops in the forward pass
  void MegaBufferSetupAndDistribution();

  // Helper to add pipeline meta-data and handle extra
  // input/output tensors.
  OpSpec PrepareOpSpec(const OpSpec &spec);
  
  // Helper function to add extra input/output tensors to an
  // OpSpec based on the requested input/outputs.
  void ExtraTensorSetup(OpSpec *spec);
  
  enum DecodeLocation {
    DECODE_NONE,
    DECODE_PREFETCH,
    DECODE_FORWARD
  };
  DecodeLocation decode_location_;
  bool built_;

  int batch_size_;
  cudaStream_t stream_;
  ThreadPool thread_pool_;
  size_t pixels_per_image_hint_;

  template <typename T>
  using TensorPtr = shared_ptr<Tensor<T>>;
  std::map<string, TensorPtr<CPUBackend>> extra_tensors_;
  std::map<string, TensorPtr<GPUBackend>> extra_gpu_tensors_;
  
  template <typename T>
  using OpPtr = unique_ptr<Operator<T>>;
  vector<OpPtr<CPUBackend>> prefetch_ops_;
  vector<OpPtr<GPUBackend>> forward_ops_;

  // Batch objects to store intermediate results of
  // the pipeline.
  template <typename T>
  using BatchPtr = shared_ptr<Batch<T>>;
  vector<BatchPtr<CPUBackend>> cpu_buffers_;
  vector<BatchPtr<GPUBackend>> gpu_buffers_;

  // The actually GPU allocations we maintain
  vector<BatchPtr<GPUBackend>> gpu_storage_;
  
  // DataReader to query for datum during execution
  unique_ptr<DataReader> data_reader_;
  vector<Datum<CPUBackend>> input_datum_;

  // The parser to handle custom input data formats
  unique_ptr<Parser> data_parser_;
  vector<Datum<CPUBackend>> parsed_datum_;
  
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

} // namespace ndll

#endif // NDLL_PIPELINE_PIPELINE_H_
