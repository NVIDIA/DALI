// Copyright (c) 2019-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <gtest/gtest.h>

namespace dali {
namespace kernels {

struct Add {
  template <typename Out, typename In0, typename In1>
  void run(Out &out, In0 in0, In1 in1) {
    out = static_cast<Out>(in0 + in1);
  }
};

template <typename Operator, typename Out, typename In0, typename In1, int ndim>
void BinOp(Operator &&op, Out *out, const int64_t *out_shape, const int64_t *out_strides, 
           const In0* in0, const int64_t *in0_shape, const int64_t *in0_strides,
           const In1* in1, const int64_t *in1_shape, const int64_t *in1_strides, 
           std::integral_constant<int, ndim>) {
  static_assert(ndim > 1);
  if (*in0_shape > 1 && *in1_shape > 1) {
    for (int64_t i = 0; i < *out_shape; i++) {
      BinOp(std::forward<Operator>(op), 
            out, out_shape + 1, out_strides + 1,
            in0, in0_shape + 1, in1_strides + 1,
            in1, in1_shape + 1, in1_strides + 1,
            std::integral_constant<int, ndim - 1>());
      in0 += *in0_strides;
      in1 += *in1_strides;
      out += *out_strides;
    }
  } else if (*in0_shape > 1 && *in1_shape == 1) {
    for (int64_t i = 0; i < *out_shape; i++) {
      BinOp(std::forward<Operator>(op),
            out, out_shape + 1, out_strides + 1,
            in0, in0_shape + 1, in1_strides + 1,
            in1, in1_shape + 1, in1_strides + 1,
            std::integral_constant<int, ndim - 1>());
      in0 += *in0_strides;
      out += *out_strides;
    }
  } else if (*in0_shape == 1 && *in1_shape > 1) {
    for (int64_t i = 0; i < *out_shape; i++) {
      BinOp(std::forward<Operator>(op),
            out, out_shape + 1, out_strides + 1,
            in0, in0_shape + 1, in1_strides + 1,
            in1, in1_shape + 1, in1_strides + 1,
            std::integral_constant<int, ndim - 1>());
      in1 += *in1_strides;
      out += *out_strides;
    }
  } else {
    assert(*in0_shape == 1 && *in1_shape == 1);
    BinOp(std::forward<Operator>(op), 
          out, out_shape + 1, out_strides + 1,
          in0, in0_shape + 1, in1_strides + 1,
          in1, in1_shape + 1, in1_strides + 1,
          std::integral_constant<int, ndim - 1>());
  }
}

template <typename Operator, typename Out, typename In0, typename In1>
void BinOp(Operator &&op, Out *out, const int64_t *out_shape, const int64_t *out_strides, 
           const In0* in0, const int64_t *in0_shape, const int64_t *in0_strides,
           const In1* in1, const int64_t *in1_shape, const int64_t *in1_strides, 
           std::integral_constant<int, 1>) {
  if (*in0_shape > 1 && *in1_shape > 1) {
    for (int64_t i = 0; i < *out_shape; i++) {
      op.run(*out, *in0, *in1);
      in0 += *in0_strides;
      in1 += *in1_strides;
      out += *out_strides;
    }
  } else if (*in0_shape > 1 && *in1_shape == 1) {
    for (int64_t i = 0; i < *out_shape; i++) {
      op.run(*out, *in0, *in1);
      in0 += *in0_strides;
      out += *out_strides;
    }
  } else if (*in0_shape == 1 && *in1_shape > 1) {
    for (int64_t i = 0; i < *out_shape; i++) {
      op.run(*out, *in0, *in1);
      in1 += *in1_strides;
      out += *out_strides;
    }
  } else {
    assert(*in0_shape == 1 && *in1_shape == 1);
    op.run(*out, *in0, *in1);
  }
}


TEST(ArithmOpBcastTest, Bcast) {
  uint8_t a[] = {1, 2, 3, 
                 4, 5, 6};
  int64_t a_shape[] = {2, 3};
  int64_t a_strides[] = {3, 1};
  uint8_t b[] = {7, 8, 9};
  int64_t b_shape[] = {1, 3};
  int64_t b_strides[] = {1, 1};

  uint8_t out[] = {0, 0, 0, 
                   0, 0, 0};
  int64_t out_shape[] = {2, 3};
  int64_t out_strides[] = {3, 1};

  BinOp<>(Add{},
          out, out_shape, out_strides,
          a, a_shape, a_strides,
          b, b_shape, b_strides,
          std::integral_constant<int, 2>{});

  auto ptr = out;
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 3; j++) {
      std::cout << " " << (int) *ptr++; 
    }
    std::cout << "\n";
  }
}


}  // namespace kernels
}  // namespace dali
