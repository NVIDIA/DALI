#include "dummy.h"

namespace other_ns {

template<>
void Dummy<::dali::CPUBackend>::RunImpl(::dali::SampleWorkspace &ws) {
  const auto &input = ws.Input<::dali::CPUBackend>(0);
  auto &output = ws.Output<::dali::CPUBackend>(0);

  ::dali::TypeInfo type = input.type();
  type.Copy<::dali::CPUBackend, ::dali::CPUBackend>(
      output.raw_mutable_data(),
      input.raw_data(), input.size(), 0);
}

}  // namespace other_ns

DALI_REGISTER_OPERATOR(CustomDummy, ::other_ns::Dummy<::dali::CPUBackend>, ::dali::CPU);

DALI_SCHEMA(CustomDummy)
  .DocStr("Make a copy of the input tensor")
  .NumInput(1)
  .NumOutput(1);
