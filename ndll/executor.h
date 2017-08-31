#ifndef NDLL_EXECUTOR_H_
#define NDLL_EXECUTOR_H_

#include "ndll/common.h"
#include "ndll/operator.h"
#include "ndll/stage.h"

namespace ndll {

/**
 * @brief Manages pipeline of stages, auto-tuning and execution of the pipeline
 */
template <typename CPUBackend, GPUBackend>
class Executor {
public:
  Executor() {}
  ~Executor() = default;

  void RunPrefetch() {
    // TODO(tgale): Thread this
    vector<Dims> dims = decoder_.GetDims();

    // Give
    size_t total_bytes = 0;
    vector<size_t> op_param_offsets;
    for (auto op : forward_ops) {
      // TODO(tgale): Do we want to thread this per image? Batched
      // resize would require us to calculate the params which
      // would require an RNG. We should be able to thread this
      size_t bytes = op->GetForwardGpuParamSize(dims);

      // TODO(tgale): Align all pointers to 8-bytes
      op_param_offsets.push_back(total_bytes);
      total_bytes += bytes;      
    }

    // allocate the mega-buffer
    mega_buffer_.Resize(total_bytes);
    d_mega_buffer_.Resize(total_bytes);

    for (int i = 0; i < forward_ops_.size(); ++i) {
      auto op = forward_ops_[i];
      op->SetGpuParamPointer(mega_buffer_.data() + op_param_offsets[i]);
      op->SetGpuParamDevPointer(d_mega_buffer_.data() + op_param_offsets[i]);
    }

    // Run the prefetch stages

    // Set all batched params for forward ops

    // How do we want to organize computation in the two thread loops, and how do
    // we want to expose this to the ops so they can implement what they need where?

  }

  void RunForward() {

  }
  
  DISABLE_COPY_ASSIGN(Executor);
private:
  Decoder decoder_;
  vector<Operator> ops_;
  
  // Note: We need someway of running the op specific method like
  // size getting for the decoder, batch params buffer size for forward
  // stages, and batch params for forward stages
  vector<shared_ptr<Stage>> prefetch_stages_;
  vector<shared_ptr<Stage>> forward_stages_;
};

} // namespace ndll

#endif // NDLL_EXECUTOR_H_
