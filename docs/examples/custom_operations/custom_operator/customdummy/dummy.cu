#include <cuda_runtime_api.h>
#include "dummy.h"

namespace other_ns {

template<>
void Dummy<::dali::GPUBackend>::RunImpl(::dali::Workspace &ws) {
  const auto &input = ws.Input<::dali::GPUBackend>(0);
  const auto &shape = input.shape();
  auto &output = ws.Output<::dali::GPUBackend>(0);
  for (int sample_idx = 0; sample_idx < shape.num_samples(); sample_idx++) {
    CUDA_CALL(cudaMemcpyAsync(
            output.raw_mutable_tensor(sample_idx),
            input.raw_tensor(sample_idx),
            shape[sample_idx].num_elements() * input.type_info().size(),
            cudaMemcpyDeviceToDevice,
            ws.stream()));
  }
}

}  // namespace other_ns

DALI_REGISTER_OPERATOR(CustomDummy, ::other_ns::Dummy<::dali::GPUBackend>,
                       ::dali::GPU);
