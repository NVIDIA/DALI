#include <gtest/gtest.h>
#include <libtar.h>

#include <cstdlib>
#include <string>
#include <fcntl.h>
#include <iostream>

#include "dali/operators/reader/loader/filesystem.h"

namespace dali {
namespace testing {

TEST(LibTar, OpenClose) {
  std::string filepath(dali::filesystem::join_path(
      std::getenv("DALI_EXTRA_PATH"),
      "db/webdataset/MNIST/devel-0.tar"
  ));
  
  TAR* archive;
  ASSERT_EQ(tar_open(&archive, filepath.c_str(), NULL, O_RDONLY, 0, TAR_GNU), 0);
  ASSERT_EQ(tar_close(archive), 0);
}

} // namespace testing
} // namespace dali