// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#include <opencv2/imgcodecs.hpp>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>
#include "dali/kernels/test/test_data.h"
#include "dali/test/dali_test_config.h"


namespace dali {
namespace testing {

namespace {

const std::string &base_folder() {
  static const std::string path = testing::dali_extra_path() + "/db/";
  return path;
}

inline size_t file_length(std::ifstream &f) {
  auto pos = f.tellg();
  f.seekg(0, f.end);
  size_t length = f.tellg();
  f.seekg(pos, f.beg);
  return length;
}

struct TestDataImpl {
  span<uint8_t> file(const char *name) {
    auto result = files.insert({ name, {} });
    auto &file = result.first->second;
    if (result.second) {
      std::string path = name[0] == '/' ? name : base_folder() + name;
      std::ifstream f(path, std::ios::binary|std::ios::in);
      if (!f.good())
        throw std::runtime_error("Cannot load file: " + path);
      size_t length = file_length(f);

      file.resize(length);
      f.read(file.data(), length);
    }
    return { reinterpret_cast<uint8_t*>(file.data()), (span_extent_t)file.size() };
  }

  const cv::Mat &image(const char *name, bool color) {
    std::string key = std::string(name)+":"+(color ? "BGR" : "gray");
    auto result = images.insert({ key, {} });
    auto &image = result.first->second;
    if (result.second) {
      std::string path = name[0] == '/' ? name : base_folder() + name;
      image = cv::imread(path, color ? cv::IMREAD_COLOR : cv::IMREAD_GRAYSCALE);
      if (image.empty())
        throw std::runtime_error("Cannot read image: " + path);
    }
    return image;
  }

  std::unordered_map<std::string, std::vector<char> > files;
  std::unordered_map<std::string, cv::Mat> images;
} test_data;

}  // namespace

namespace data {
span<uint8_t> file(const char *name) {
  return test_data.file(name);
}

const cv::Mat &image(const char *name, bool color) {
  return test_data.image(name, color);
}
}  // namespace data

}  // namespace testing
}  // namespace dali
