// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include "dali/operators/generic/constant.h"

#include <vector>
#include "dali/core/convert.h"
#include "dali/core/static_switch.h"
#include "dali/pipeline/data/views.h"

namespace dali {

DALI_SCHEMA(Constant)
  .DocStr(R"code(Produces a batch of constant tensors.

The floating point input data should be placed in `fdata` argument and integer data in `idata`.
The data is a flat vector of values or a single scalar. The data is then reshaped according
to the `shape` argument. If the data is scalar, it will be broadcast to fill the entire shape.

The operator only performs meaningful work at first invocation - subsequent calls will return
a reference to the same memory.

The operator can be automatically instantiated in Python with a call to
`dali.types.Constant(value, dtype, shape, layout)`. The value can be a scalar, a tuple, a list or
a numpy array, in which case the `shape` and `dtype` (if not explicitly overridden), will be
taken from that array.

64-bit integer and double precision arrays are not supported and will be silently downgraded
to 32-bit.
)code")
  .NumInput(0)
  .NumOutput(1)
  .AddOptionalArg("shape",
                  "The desired shape of the output. If not set, the data is assumed to be 1D",
                  std::vector<int>())
  .AddOptionalArg("fdata",
                  "Contents of the constant produced (for floating point types). "
                  "`fdata` and `idata` are mutually exclusive and one of them is required",
                  std::vector<float>())
  .AddOptionalArg("idata",
                  "Contents of the constant produced (for integer types). "
                  "`fdata` and `idata` are mutually exclusive and one of them is required",
                  std::vector<int>())
  .AddOptionalArg("dtype",
                  "Type of the output data. If not set, the output is float if `fdata` "
                  "argument is used and int if `idata` is used.",
                  DALI_NO_TYPE)
  .AddOptionalArg("layout",
                  "Layout info. If set and not empty, the layout must match the "
                  "dimensionality of the output.",
                  TensorLayout());

namespace {
template <typename Dst, typename Src>
void FillTensorVector(
  TensorVector<CPUBackend> &dst, const TensorListShape<> &shape, const std::vector<Src> &src) {
  dst.SetContiguous(false);
  dst.Resize(shape);
  assert(is_uniform(shape));
  int64_t n = shape[0].num_elements();
  assert(src.size() == static_cast<size_t>(n) || src.size() == 1);
  Dst *out = dst[0].mutable_data<Dst>();
  if (src.size() == 1) {
    Dst val = ConvertSat<Dst>(src[0]);
    for (int64_t i = 0; i < n; i++) {
      out[i] = val;
    }
  } else {
    for (int64_t i = 0; i < n; i++) {
      out[i] = ConvertSat<Dst>(src[i]);
    }
  }
  for (int i = 1; i < shape.num_samples(); i++) {
    dst[i].ShareData(&dst[0]);
  }
}
}  // namespace

template <>
void Constant<CPUBackend>::RunImpl(HostWorkspace &ws) {
  if (output_.ntensor() == 0) {
    TYPE_SWITCH(output_type_, type2id, type, CONSTANT_OP_SUPPORTED_TYPES,
      (
        if (!fdata_.empty()) {
          FillTensorVector<type>(output_, output_shape_, fdata_);
        } else {
          assert(!idata_.empty());
          FillTensorVector<type>(output_, output_shape_, idata_);
        }
      ), (DALI_FAIL(make_string("Unsupported type: ", to_string(output_type_)))));  // NOLINT
  }
  auto &out = ws.OutputRef<CPUBackend>(0);

  out.ShareData(&output_);
  int N = output_shape_.num_samples();
  for (int i = 0; i < N; i++) {
    assert(out[i].raw_data() == output_[i].raw_data());
  }
  out.SetLayout(layout_);
}

DALI_REGISTER_OPERATOR(Constant, Constant<CPUBackend>, CPU);

}  // namespace dali
