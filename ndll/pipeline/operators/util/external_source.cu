// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include "ndll/pipeline/operators/util/external_source.h"

namespace ndll {

template<>
void ExternalSource<GPUBackend>::RunImpl(DeviceWorkspace *ws, const int idx) {
  NDLL_ENFORCE(data_in_tl_, "Cannot feed non-contiguous data to GPU op.");
  auto output = ws->Output<GPUBackend>(idx);
  output->ShareData(&tl_data_);
}

NDLL_REGISTER_OPERATOR(ExternalSource, ExternalSource<GPUBackend>, GPU);

}  // namespace ndll
