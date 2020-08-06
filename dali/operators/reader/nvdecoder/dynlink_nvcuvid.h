// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef DALI_OPERATORS_READER_NVDECODER_DYNLINK_NVCUVID_H_
#define DALI_OPERATORS_READER_NVDECODER_DYNLINK_NVCUVID_H_

#include <string>
#include "dali/operators/reader/nvdecoder/nvcuvid.h"

#define NVCUVID_CALL(arg) CUDA_CALL(arg)
#define NVCUVID_API_EXISTS(arg) (cuvidIsSymbolAvailable(#arg))

bool cuvidInitChecked(void);
bool cuvidIsSymbolAvailable(const char *name);

#endif  // DALI_OPERATORS_READER_NVDECODER_DYNLINK_NVCUVID_H_
