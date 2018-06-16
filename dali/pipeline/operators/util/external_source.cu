// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include "dali/pipeline/operators/util/external_source.h"

namespace dali {

template<>
void ExternalSource<GPUBackend>::RunImpl(DeviceWorkspace *ws, const int idx) {
  DALI_ENFORCE(data_in_tl_, "Cannot feed non-contiguous data to GPU op.");
  auto output = ws->Output<GPUBackend>(idx);
  output->ShareData(&tl_data_);
}

DALI_REGISTER_OPERATOR(ExternalSource, ExternalSource<GPUBackend>, GPU);

}  // namespace dali
