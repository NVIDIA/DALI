// Copyright (c) 2020-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/util/fits.h"
#include <gtest/gtest.h>
#include <string>
#include <vector>
#include "dali/core/stream.h"
#include "dali/test/dali_test_config.h"
#include "dali/util/odirect_file.h"

namespace dali {
namespace fits {

namespace {

template <typename T>
vector<T> ReadVector(InputStream *src) {
  vector<T> data;
  data.resize(src->Size() / sizeof(T));
  auto ret = src->Read(reinterpret_cast<uint8_t *>(data.data()), src->Size());
  DALI_ENFORCE(ret == src->Size(), "Failed to read numpy file");
  return data;
}

struct test_sample {
  test_sample(std::string img_path, std::string ref_data_path, std::string ref_offset_sizes_path,
              std::string ref_tile_sizes_path)
      : path(img_path),
        ref_undecoded_data(
            ReadVector<uint8_t>(FileStream::Open(ref_data_path).get())),
        ref_offset_sizes(
            ReadVector<int64_t>(FileStream::Open(ref_offset_sizes_path).get())),
        ref_tile_sizes(
            ReadVector<int64_t>(FileStream::Open(ref_tile_sizes_path).get())) {}

  std::string path;
  vector<uint8_t> ref_undecoded_data;
  vector<int64_t> ref_offset_sizes;
  vector<int64_t> ref_tile_sizes;
};

struct TestData {
  TestData() {
    const auto fits_dir =
        make_string(dali::testing::dali_extra_path(), "/db/single/fits/compressed/");
    const auto fits_ref_dir =
        make_string(dali::testing::dali_extra_path(), "/db/single/reference/fits/");

    auto filenames = {"kitty-2948404_640_red_rice", "cat-1046544_640_blue_rice",
                      "domestic-cat-726989_640_green_rice"};

    for (auto filename : filenames) {
      test_samples.emplace_back(make_string(fits_dir, filename, ".fits"),
                                make_string(fits_ref_dir, filename, ".data"),
                                make_string(fits_ref_dir, filename, ".offset_size"),
                                make_string(fits_ref_dir, filename, ".tile_size"));
    }
  }

  void Destroy() {
    test_samples.clear();
  }

  span<test_sample> get() {
    return make_span(test_samples);
  }

 private:
  vector<test_sample> test_samples;
};

static TestData data;  // will initialize once for the whole suite

}  // namespace

TEST(FitsExtractUndecodedTest, ExtractData) {
  int status = 0;
  int64_t rows;
  vector<uint8_t> undecoded_data;
  vector<int64_t> offset_sizes, tile_sizes;

  for (const auto &sample : data.get()) {
    auto fptr = FitsHandle::OpenFile(sample.path.c_str(), READONLY);
    FITS_CALL(fits_movabs_hdu(fptr, 2, nullptr, &status));  // move to the first HDU with data
    FITS_CALL(fits_get_num_rows(fptr, &rows, &status));

    ExtractUndecodedData(fptr, undecoded_data, offset_sizes, tile_sizes, rows, &status);

    ASSERT_EQ(undecoded_data, sample.ref_undecoded_data);
    ASSERT_EQ(offset_sizes, sample.ref_offset_sizes);
    ASSERT_EQ(tile_sizes, sample.ref_tile_sizes);
  }
}


}  // namespace fits
}  // namespace dali
