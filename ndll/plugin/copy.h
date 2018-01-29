// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#ifndef NDLL_PLUGIN_COPY_H_
#define NDLL_PLUGIN_COPY_H_

#include "ndll/pipeline/data/tensor_list.h"

namespace ndll {

void CopyToExternalTensor(const TensorList<CPUBackend>& tl, void* ptr);
void CopyToExternalTensor(const TensorList<GPUBackend>& tl, void* ptr);

}  // namespace ndll

#endif  // NDLL_PLUGIN_COPY_H_
