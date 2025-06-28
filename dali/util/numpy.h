// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_UTIL_NUMPY_H_
#define DALI_UTIL_NUMPY_H_

#include <string>
#include <string_view>

#include "dali/core/common.h"
#include "dali/core/tensor_shape.h"
#include "dali/core/stream.h"
#include "dali/pipeline/data/sample_view.h"
#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/data/tensor.h"
#include "dali/core/static_switch.h"
#include "dali/kernels/transpose/transpose.h"

#define NUMPY_ALLOWED_TYPES \
  (bool, uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, float16, \
  double)

namespace dali {
namespace numpy {

class DLL_PUBLIC HeaderData {
 public:
  TensorShape<> shape;
  const TypeInfo *type_info = nullptr;
  bool fortran_order        = false;
  int64_t data_offset       = 0;

  DALIDataType type() const;

  size_t size() const;

  size_t nbytes() const;
};

DLL_PUBLIC void ParseHeader(HeaderData &parsed_header, InputStream *src);

DLL_PUBLIC void ParseODirectHeader(HeaderData &parsed_header, InputStream *src,
                                   size_t o_direct_alignm, size_t o_direct_read_len_alignm);

DLL_PUBLIC void FromFortranOrder(SampleView<CPUBackend> output, ConstSampleView<CPUBackend> input);

DLL_PUBLIC void ParseHeaderContents(HeaderData& target, const std::string_view header);

DLL_PUBLIC Tensor<CPUBackend> ReadTensor(InputStream *src, bool pinned);

}  // namespace numpy
}  // namespace dali

#endif  // DALI_UTIL_NUMPY_H_
