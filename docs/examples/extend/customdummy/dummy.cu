#include <cuda_runtime_api.h>
#include "dummy.h"

namespace other_ns {

template<>
void Dummy<::dali::GPUBackend>::RunImpl(::dali::DeviceWorkspace &ws) {
  const auto &input = ws.Input<::dali::GPUBackend>(0);
  auto &output = ws.Output<::dali::GPUBackend>(0);
  CUDA_CALL(cudaMemcpyAsync(
          output.raw_mutable_data(),
          input.raw_data(),
          input.nbytes(),
          cudaMemcpyDeviceToDevice,
          ws.stream()));
}

}  // namespace other_ns

DALI_REGISTER_OPERATOR(CustomDummy, ::other_ns::Dummy<::dali::GPUBackend>, ::dali::GPU);
