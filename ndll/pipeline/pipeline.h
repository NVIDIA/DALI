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
    // Verify that this name is unique and record it
    auto it = edge_names_.find(name);
    NDLL_ENFORCE(it == edge_names_.end(), "External input name '" +
        name + "' conflicts with existing intermediate result name");
    EdgeMeta meta;
    meta.has_cpu = true;
    meta.has_gpu = false;
    meta.has_contiguous = false;
    NDLL_ENFORCE(edge_names_.insert({name, meta}).second,
        "ExternalInput name insertion failure.");
    
    // Create a spec for an ExternalInput op and add it to our graph
    OpSpec spec =
      OpSpec("ExternalSource")
      .AddArg("device", "cpu")
      .AddArg("inplace", true)
      .AddOutput(name, "cpu");
    PrepareOpSpec(&spec);
    graph_.AddOp(spec);
  }
  
  /**
   * @brief Adds an Operator with the input specification to the pipeline. The
   * 'device' argument in the OpSpec determines whether the CPU or GPU version
   * of the named operator will be added to the pipeline
   */
  void AddOperator(OpSpec spec);
  
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
  const vector<const TensorList<Backend>&> output() const;
  
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

  // For testing
  template <typename T>
  friend class PipelineTest;
  
  DISABLE_COPY_MOVE_ASSIGN(Pipeline);
private:
  using EdgeMeta = struct {
    bool has_cpu, has_gpu, has_contiguous;
  };
  
  // Return the nearest multiple of 8 that is >= base_ptr_offset
  inline size_t round_up_to_8(size_t base_ptr_offset) {
    if (base_ptr_offset & 7) {
      base_ptr_offset = (base_ptr_offset & ~7) + 8;
    }
    return base_ptr_offset;
  }
  
  void SetupCPUInput(std::map<string, EdgeMeta>::iterator it,
      int input_idx, OpSpec *spec);

  void SetupGPUInput(std::map<string, EdgeMeta>::iterator it);

  inline EdgeMeta NewEdge(string device) {
    EdgeMeta edge;
    edge.has_cpu = false;
    edge.has_gpu = false;
    edge.has_contiguous = false;
    if (device == "cpu") {
      edge.has_cpu = true;
    } else if (device == "gpu") {
      edge.has_gpu = true;
    } else {
      NDLL_FAIL("Invalid device argument \"" + device + "\". "
          "Valid options are \"cpu\" or \"gpu\"");
    }
    return edge;
  }
      
  // Helper function to setup mega-buffer and distribute
  // sub-buffers to the ops in the forward pass
  void MegaBufferSetupAndDistribution();

  // Helper to add pipeline meta-data 
  void PrepareOpSpec(OpSpec *spec);
  
  bool built_;
  int batch_size_;
  cudaStream_t stream_;
  ThreadPool thread_pool_;
  size_t pixels_per_image_hint_;

  OpGraph graph_;

  std::map<string, EdgeMeta> edge_names_;
};

} // namespace ndll

#endif // NDLL_PIPELINE_PIPELINE_H_
