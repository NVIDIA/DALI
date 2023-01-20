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

#include <fitsio.h>
#include <gtest/gtest.h>
#include <stdlib.h>
#include <string.h>


namespace dali {
namespace testing {

class CfitsioTest : public ::testing::Test {
 protected:
  int status, filemode;
  char filename[40];
  fitsfile *fptr;

  void SetUp() final {
    status = 0;
    if (remove("testprog.fit") != 0) {
      std::cerr << "Failed to remove the file.\n";
    }
  }

  void TearDown() final {
    if (remove("testprog.fit") != 0) {
      std::cerr << "Faild to remove the file.\n";
    }
  }
};


TEST_F(CfitsioTest, openNonexistantFile) {
  fits_open_file(&fptr, "testprog.fit", READWRITE, &status);

  EXPECT_TRUE(fptr == nullptr);
  EXPECT_NE(status, 0);

  ffclos(fptr, &status);

  EXPECT_NE(status, 0);
}

TEST_F(CfitsioTest, createEmptyFile) {
  const char force_recreate_filename[] = "!testprog.fit";
  strncpy(filename, force_recreate_filename, sizeof(filename));
  ffinit(&fptr, filename, &status);

  EXPECT_EQ(status, 0);

  filename[0] = '\0';

  ffflnm(fptr, filename, &status);
  EXPECT_EQ(status, 0);
  ffflmd(fptr, &filemode, &status);
  EXPECT_EQ(status, 0);

  EXPECT_STREQ(filename, "testprog.fit");
  EXPECT_EQ(filemode, 1);

  ffclos(fptr, &status);
}


}  // namespace testing
}  // namespace dali
