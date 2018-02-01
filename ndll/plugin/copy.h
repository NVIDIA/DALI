// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#ifndef NDLL_PLUGIN_COPY_H_
#define NDLL_PLUGIN_COPY_H_

#include "ndll/pipeline/data/tensor_list.h"
#include "ndll/pipeline/data/tensor.h"

namespace ndll {

void CopyToExternalTensor(TensorList<CPUBackend>* tl, void* ptr);
void CopyToExternalTensor(TensorList<GPUBackend>* tl, void* ptr);
void CopyToExternalTensor(const Tensor<CPUBackend>& tl, void* ptr);
void CopyToExternalTensor(const Tensor<GPUBackend>& tl, void* ptr);

}  // namespace ndll

#endif  // NDLL_PLUGIN_COPY_H_
