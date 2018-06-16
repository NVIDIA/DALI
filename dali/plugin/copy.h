// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#ifndef DALI_PLUGIN_COPY_H_
#define DALI_PLUGIN_COPY_H_

#include "dali/pipeline/data/tensor_list.h"
#include "dali/pipeline/data/tensor.h"

namespace dali {

void CopyToExternalTensor(TensorList<CPUBackend>* tl, void* ptr);
void CopyToExternalTensor(TensorList<GPUBackend>* tl, void* ptr);
void CopyToExternalTensor(const Tensor<CPUBackend>& tl, void* ptr);
void CopyToExternalTensor(const Tensor<GPUBackend>& tl, void* ptr);

}  // namespace dali

#endif  // DALI_PLUGIN_COPY_H_
