#ifndef NDLL_PIPELINE_PIPELINE_H_
#define NDLL_PIPELINE_PIPELINE_H_

#include <map>

#include "ndll/common.h"
#include "ndll/pipeline/data/backend.h"
#include "ndll/pipeline/data/tensor.h"
#include "ndll/pipeline/data/tensor_list.h"
#include "ndll/pipeline/operator.h"
#include "ndll/pipeline/op_graph.h"
#include "ndll/pipeline/util/thread_pool.h"

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
    built_(false), batch_size_(batch_size), stream_(stream),
    thread_pool_(num_threads, device_id, set_affinity),
    pixels_per_image_hint_(pixels_per_image_hint) {
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

  ~Pipeline() = default;

  /**
   * @brief Creates a placeholder for an external input with the given name
   */
  inline void AddExternalInput(const string &name) {
    // Create a spec for an ExternalInput op and add it to our graph
    OpSpec spec =
      OpSpec("ExternalSource")
      .AddArg("device", "cpu")
      .AddArg("inplace", true)
      .AddOutput(name, "cpu");
    graph_.AddOp(PrepareOpSpec(spec));
  }
  
  /**
   * @brief Adds an Operator with the input specification to the pipeline. The
   * 'device' argument in the OpSpec determines whether the CPU or GPU version
   * of the named operator will be added to the pipeline
   */
  inline void AddOperator(const OpSpec &spec) {
    NDLL_ENFORCE(!built_, "Alterations to the pipeline after "
        "\"Build()\" has been called are not allowed");

    // Add some pipeline meta-data
    OpSpec spec_copy = PrepareOpSpec(spec);
    string device = spec.GetArgument<string>("device", "cpu");
    if (device == "cpu") {
      OpPtr<CPUBackend> tmp(
          CPUOperatorRegistry::Registry().Create(spec_copy.name(), spec_copy));
      cpu_ops_.push_back(std::move(tmp));
    } else if (device == "gpu") {
      OpPtr<GPUBackend> tmp(
          GPUOperatorRegistry::Registry().Create(spec_copy.name(), spec_copy));
      gpu_ops_.push_back(std::move(tmp));
    } else {
      NDLL_FAIL("Invalid device argument \"" + device +
          "\". Valid options are \"cpu\" or \"gpu\"");
    }
  }
  
  /**
   * @brief Performs some checks on the user-constructed pipeline, setups data
   * for intermediate results, and marks as ready for execution.
   */
  void Build();

  /**
   * @brief Run the cpu portion of the pipeline.
   */
  void RunCPU();

  /**
   * @brief Copies the result of the prefetch stage into the input 
   * buffer for the forward stage. Also copied all kernel parameters
   * to the GPU for the GPU stage of computation.
   */
  void RunCopy();

  /**
   * @brief Run the gpu portion of the pipeline.
   *
   * This stage is designed to be extremely light weight to minimize cost
   * on the front of the forward pass. All parameter setup and allocations
   * have been done previously, and we simply iterate over the forward
   * stage ops and launch their kernels.
   */
  void RunGPU();

  /**
   * @brief Returns the output TensorList w/ the specified name.
   */
  template <typename Backend>
  const TensorList<Backend>& output(const string &name) const;
  
  /**
   * @brief Returns the batch size that will be produced by the pipeline.
   */
  inline int batch_size() const { return batch_size_; }

  /**
   * @brief Returns the number of threads that are used by the pipeline.
   */
  inline int num_threads() const { return thread_pool_.size(); }
  
  /**
   * @brief Returns the stream that the pipeline is working in.
   */
  inline cudaStream_t stream() const { return stream_; }

  /**
   * @brief Returns a pointer to the Operator w/ the specified name.
   */
  template <typename Backend>
  Operator<Backend>* op(const string &name);
  
  DISABLE_COPY_MOVE_ASSIGN(Pipeline);
private:
  // Return the nearest multiple of 8 that is >= base_ptr_offset
  inline size_t round_up_to_8(size_t base_ptr_offset) {
    if (base_ptr_offset & 7) {
      base_ptr_offset = (base_ptr_offset & ~7) + 8;
    }
    return base_ptr_offset;
  }

  // Helper function to setup mega-buffer and distribute
  // sub-buffers to the ops in the forward pass
  void MegaBufferSetupAndDistribution();

  // Helper to add pipeline meta-data 
  OpSpec PrepareOpSpec(const OpSpec &spec);
  
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
  vector<OpPtr<CPUBackend>> cpu_ops_;
  vector<OpPtr<GPUBackend>> gpu_ops_;
  
  // Tensors to store all batched op parameters for ops in
  // the forward pass. Enables single copy of paramters
  // instead of copies per operator
  Tensor<CPUBackend> mega_buffer_;
  Tensor<GPUBackend> mega_buffer_gpu_;

  OpGraph graph_;
};

} // namespace ndll

#endif // NDLL_PIPELINE_PIPELINE_H_
