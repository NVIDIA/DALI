// Copyright (c) 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <string>
#include <vector>
#include "dali/util/numpy.h"
#include "dali/core/stream.h"


namespace dali {
namespace numpy {

TEST(NumpyLoaderTest, ParseHeader) {
  {
    HeaderData target;
    detail::ParseHeaderMetadata(target, "{'descr':'<i2', 'fortran_order':True, 'shape':(4,7),}");
    ASSERT_EQ(target.type(), DALI_INT16);
    ASSERT_EQ(target.fortran_order, true);
    ASSERT_EQ(target.shape, (TensorShape<>{7, 4}));
  }
  {
    HeaderData target;
    detail::ParseHeaderMetadata(target,
        "  {  'descr' : '<f4'   ,   'fortran_order'  : False, 'shape' : (4,)}");
    ASSERT_EQ(target.type(), DALI_FLOAT);
    ASSERT_EQ(target.fortran_order, false);
    ASSERT_EQ(target.shape, (TensorShape<>{4}));
  }
  {
    HeaderData target;
    detail::ParseHeaderMetadata(target, "{'descr':'<f8','fortran_order':False,'shape':(),}");
    ASSERT_EQ(target.type(), DALI_FLOAT64);
    ASSERT_EQ(target.fortran_order, false);
    ASSERT_TRUE(target.shape.empty());
  }
}

TEST(NumpyLoaderTest, ParseHeaderError) {
  HeaderData target;
  std::vector<std::string> wrong = {
    "random_string",
    "{descr:'<f4'}",
    "{'descr':'','fortran_order':False,'shape':(4,7),}",
    "{'descr':'<f4','fortran_order':false,'shape':(4,7),}"
    "{'descr':'<f4','fortran_order':false,'shape':(a, b, c),}"
    "{'descr':'<f4','fortran_order':False,'shape':[4,7],}"
  };
  for (const auto &header : wrong) {
    EXPECT_THROW(detail::ParseHeaderMetadata(target, header), std::runtime_error);
  }
}

}  // namespace numpy
}  // namespace dali

