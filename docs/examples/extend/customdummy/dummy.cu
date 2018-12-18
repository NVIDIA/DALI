#include <cuda_runtime_api.h>
#include "dummy.h"

namespace other_ns {

template<>
void Dummy<::dali::GPUBackend>::RunImpl(::dali::DeviceWorkspace *ws, const int idx) {
  auto &input = ws->Input<::dali::GPUBackend>(idx);
  auto output = ws->Output<::dali::GPUBackend>(idx);
  output->set_type(input.type());
  output->ResizeLike(input);
  CUDA_CALL(cudaMemcpyAsync(
          output->raw_mutable_data(),
          input.raw_data(),
          input.nbytes(),
          cudaMemcpyDeviceToDevice,
          ws->stream()));
}

}  // namespace other_ns

DALI_REGISTER_OPERATOR(CustomDummy, ::other_ns::Dummy<::dali::GPUBackend>, ::dali::GPU);
