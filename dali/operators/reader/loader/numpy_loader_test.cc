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

#include <gtest/gtest.h>
#include <string>
#include "dali/operators/reader/loader/numpy_loader.h"


namespace dali {

TEST(NumpyLoaderTest, ParseHeader) {
  {
    NumpyParseTarget target;
    detail::ParseHeaderMetadata(target, "{'descr':'<i2', 'fortran_order':True, 'shape':(4,7),}");
    ASSERT_EQ(target.type_info.id(), DALI_INT16);
    ASSERT_EQ(target.fortran_order, true);
    ASSERT_EQ(target.shape, std::vector<int64_t>({7, 4}));
  }
  {
    NumpyParseTarget target;
    detail::ParseHeaderMetadata(target,
                        "  {  'descr' : '<f4'   ,   'fortran_order'  : False, 'shape' : (4,)}");
    ASSERT_EQ(target.type_info.id(), DALI_FLOAT);
    ASSERT_EQ(target.fortran_order, false);
    ASSERT_EQ(target.shape, std::vector<int64_t>({4}));
  }
  {
    NumpyParseTarget target;
    detail::ParseHeaderMetadata(target, "{'descr':'<f8','fortran_order':False,'shape':(),}");
    ASSERT_EQ(target.type_info.id(), DALI_FLOAT64);
    ASSERT_EQ(target.fortran_order, false);
    ASSERT_TRUE(target.shape.empty());
  }
}

TEST(NumpyLoaderTest, ParseHeaderError) {
  NumpyParseTarget target;
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

}  // namespace dali

