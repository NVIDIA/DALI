// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include "dali/pipeline/operators/util/dump_image.h"
#include "dali/util/image.h"

namespace dali {

template<>
void DumpImage<GPUBackend>::RunImpl(DeviceWorkspace *ws, int idx) {
  auto &input = ws->Input<GPUBackend>(idx);
  auto output = ws->Output<GPUBackend>(idx);

  WriteHWCBatch(input, suffix_ + "-" + std::to_string(idx));

  // Forward the input
  output->Copy(input, ws->stream());
}

DALI_REGISTER_OPERATOR(DumpImage, DumpImage<GPUBackend>, GPU);

}  // namespace dali
