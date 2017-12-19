#ifndef NDLL_PIPELINE_PIPELINED_EXECUTOR_H_
#define NDLL_PIPELINE_PIPELINED_EXECUTOR_H_

#include "ndll/common.h"
#include "ndll/error_handling.h"
#include "ndll/pipeline/executor.h"

namespace ndll {

/**
 * @brief In addition to the functionality provided by Executor, 
 * the PipelinedExecutor enables pipelined execution by queueing
 * the outputs of each stage (that aren't pipeline outputs - these
 * are already queued by the Executor), and increasing the queue
 * depth to 3. Because we have more, and deeper queues, this
 * executor requires more memory than the normal Executor, but can
 * see large performance benefits from pipelining the cpu, internal,
 * and gpu portions of the graph.
 */
class PipelinedExecutor : public Executor {
public:
  inline PipelinedExecutor(int batch_size, int num_thread,
      int device_id, size_t bytes_per_sample_hint,
      bool set_affinity = false, int max_num_stream = -1) :
    Executor(batch_size, num_thread, device_id, bytes_per_sample_hint,
        set_affinity, max_num_stream) {
    Executor::queue_depth_ = 3;
  }

  virtual ~PipelinedExecutor() = default;

  void Build(OpGraph *graph, vector<string> output_names) override;

  DISABLE_COPY_MOVE_ASSIGN(PipelinedExecutor);  

protected:

  void SetupStageOutputsForGraph();

  virtual inline void SetupForIter() {
    cout << "SETTING FOR ITERATION" << endl;
    SetOutputBuffersForIter();
    SetStageOutputsForIter();
    cout << "FINISHED SETTING FOR ITERATION" << endl;
  }
  
  void SetStageOutputsForIter();

  template <typename Backend>
  class TensorVectorPool {
  public:
    inline TensorVectorPool(int size, int batch_size, size_t bytes_hint) {
      tvs_.resize(size);
      for (int i = 0; i < size; ++i) {
        for (int j = 0; j < batch_size; ++j) {
          tvs_[i].push_back(std::make_shared<Tensor<Backend>>());
          tvs_[i].back()->Resize({(Index)bytes_hint});
        }
      }
    }

    inline vector<shared_ptr<Tensor<Backend>>> GetTV(int idx) {
      return tvs_[idx];
    }    
  private:
    vector<vector<shared_ptr<Tensor<Backend>>>> tvs_;
  };
  
  // Note: Pipelining the cpu, internal, and gpu execution
  // can be viewed as prefetching each stage w.r.t. the
  // other stages. Thus, we need to queue the outputs of
  // each stage to avoid overwriting data that could still
  // be in use. To do this, we find all outputs of the
  // cpu & internal stages of the pipeline that aren't
  // outptus requested by the user and setup `queue_depth`
  // extra buffers that we will rotate between. Note that
  // we do not worry about CPU outputs of the internal
  // stage, as these will only be created as outputs
  // requested by the user.
  vector<TensorVectorPool<CPUBackend>> cpu_stage_outputs_;
  vector<TensorListPool<GPUBackend>> internal_stage_outputs_;
  vector<OutputInfo> cpu_stage_output_info_;
  vector<OutputInfo> internal_stage_output_info_;
  
  USE_EXECUTOR_MEMBERS();
};

} // namespace ndll

#endif // NDLL_PIPELINE_PIPELINED_EXECUTOR_H_
