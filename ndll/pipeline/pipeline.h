#ifndef NDLL_PIPELINE_PIPELINE_H_
#define NDLL_PIPELINE_PIPELINE_H_

#include <map>

#include "ndll/common.h"
#include "ndll/pipeline/data/backend.h"
#include "ndll/pipeline/data/tensor.h"
#include "ndll/pipeline/data/tensor_list.h"
#include "ndll/pipeline/executor.h"
#include "ndll/pipeline/operators/external_source.h"
#include "ndll/pipeline/op_graph.h"

namespace ndll {

// TODO(tgale): Update dox for the pipeline and
// for all other classes that we have changed
// the semantics of.

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
   * @brief Creates a pipeline with `num_threads` worker threads.
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
   * @param bytes_per_sample_hint Estimated size of each sample to be processed.
   * Defaults to 0.
   */
  inline Pipeline(int batch_size, int num_threads, int device_id,
      int queue_depth = 2, size_t bytes_per_sample_hint = 0,
      bool set_affinity = false, int max_num_stream = -1) :
    built_(false), batch_size_(batch_size), num_threads_(num_threads),
    bytes_per_sample_hint_(bytes_per_sample_hint),
    executor_(batch_size, num_threads, device_id, bytes_per_sample_hint,
        set_affinity, queue_depth, max_num_stream) {
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
      .AddOutput(name, "cpu");
    PrepareOpSpec(&spec);
    graph_.AddOp(spec);
  }

  /**
   * @brief Sets the external input with the input name to the 
   * input data.
   */
  inline void SetExternalInput(const string &name,
      const TensorList<CPUBackend> &tl) {
    NodeID node_id = graph_.TensorSourceID(name + "_cpu");
    NDLL_ENFORCE(graph_.NodeType(node_id) == NDLL_CPU,
        "Internal error setting external input data.");

    int op_idx = graph_.NodeIdx(node_id);
    auto *op_ptr = &graph_.cpu_op(op_idx);
    ExternalSource<CPUBackend> *source =
      dynamic_cast<ExternalSource<CPUBackend>*>(op_ptr);
    NDLL_ENFORCE(source != nullptr, "Input name '" +
        name + "' is not marked as an external input.");
    source->SetDataSource(tl);
  }
  
  /**
   * @brief Adds an Operator with the input specification to the pipeline. The
   * 'device' argument in the OpSpec determines whether the CPU or GPU version
   * of the named operator will be added to the pipeline
   */
  void AddOperator(OpSpec spec);
  
  /**
   * @brief Performs some checks on the user-constructed pipeline, setups data
   * for intermediate results, and marks as ready for execution. The input
   * vector specifies the name and device of the desired outputs of the pipeline.
   */
  void Build(vector<std::pair<string, string>> output_names);

  /**
   * @brief Run the cpu portion of the pipeline.
   */
  void RunCPU();

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
   * @brief Fills the input device workspace with the output of the pipeline.
   * This method blocks until the next batch is complete. RunCPU and RunGPU
   * must be called prior to calling this or this method will result in
   * deadlock.
   *
   * TODO(tgale): This seems bad to have a method that can deadline so
   * easily. Is there a different behavior that we would like?
   */
  void Outputs(DeviceWorkspace *ws);
    
  /**
   * @brief Returns the batch size that will be produced by the pipeline.
   */
  inline int batch_size() const { return batch_size_; }

  /**
   * @brief Returns the number of threads used by the pipeline.
   */
  inline int num_threads() const { return num_threads_; }
  
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
  int batch_size_, num_threads_;
  size_t bytes_per_sample_hint_;

  OpGraph graph_;
  Executor executor_;
  
  std::map<string, EdgeMeta> edge_names_;
};

} // namespace ndll

#endif // NDLL_PIPELINE_PIPELINE_H_
