// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include "ndll/pipeline/operators/external_source.h"

namespace ndll {

template<>
void ExternalSource<CPUBackend>::RunImpl(SampleWorkspace *ws, const int idx) {
  // Wrap the output tensor around our data
  auto output = ws->Output<CPUBackend>(idx);
  if (data_in_tl_) {
    output->ShareData(&tl_data_, ws->data_idx());
  } else {
    NDLL_ENFORCE_VALID_INDEX(ws->data_idx(), t_data_.size());
    auto &data = t_data_[ws->data_idx()];
    output->ShareData(&data);
  }
}

NDLL_REGISTER_OPERATOR(ExternalSource, ExternalSource<CPUBackend>, CPU);

NDLL_OPERATOR_SCHEMA(ExternalSource)
  .DocStr("Allows externally provided data to be passed as an input to "
          "the pipeline")
  .NumInput(0)
  .NumOutput(1);

}  // namespace ndll
