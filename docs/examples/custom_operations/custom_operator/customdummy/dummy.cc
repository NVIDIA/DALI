#include "dummy.h"

namespace other_ns {

template <>
void Dummy<::dali::CPUBackend>::RunImpl(::dali::HostWorkspace &ws) {
  const auto &input = ws.InputRef<::dali::CPUBackend>(0);
  auto &output = ws.OutputRef<::dali::CPUBackend>(0);

  ::dali::TypeInfo type = input.type();
  auto &tp = ws.GetThreadPool();
  const auto &in_shape = input.shape();
  for (int sample_id = 0; sample_id < in_shape.num_samples(); sample_id++) {
    tp.AddWork(
        [&, sample_id](int thread_id) {
          type.Copy<::dali::CPUBackend, ::dali::CPUBackend>(output.raw_mutable_tensor(sample_id),
                                                            input.raw_tensor(sample_id),
                                                            in_shape.tensor_size(sample_id), 0);
        },
        in_shape.tensor_size(sample_id));
  }
  tp.RunAll();
}

}  // namespace other_ns

DALI_REGISTER_OPERATOR(CustomDummy, ::other_ns::Dummy<::dali::CPUBackend>, ::dali::CPU);

DALI_SCHEMA(CustomDummy)
    .DocStr("Make a copy of the input tensor")
    .NumInput(1)
    .NumOutput(1);
