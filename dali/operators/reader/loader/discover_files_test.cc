// Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <glob.h>
#include <gtest/gtest.h>
#include <fstream>
#include <string>
#include <vector>

#include "dali/core/error_handling.h"
#include "dali/operators/reader/loader/filesystem.h"
#include "dali/operators/reader/loader/discover_files.h"
#include "dali/operators/reader/loader/utils.h"
#include "dali/test/dali_test_config.h"

namespace dali {

class DiscoverFilesTest : public ::testing::Test {
  std::vector<std::pair<std::string, int>> readFileLabelFile() {
    std::vector<std::pair<std::string, int>> image_label_pairs;
    std::string file_list = file_root + "/image_list.txt";
    std::ifstream s(file_list);
    DALI_ENFORCE(s.is_open(), "Cannot open: " + file_list);

    std::vector<char> line_buf(16 << 10);
    char *line = line_buf.data();
    for (int n = 1; s.getline(line, line_buf.size()); n++) {
      int i = strlen(line) - 1;

      for (; i >= 0 && isspace(line[i]); i--) {}

      int label_end = i + 1;

      if (i < 0)
        continue;

      for (; i >= 0 && isdigit(line[i]); i--) {}

      int label_start = i + 1;

      for (; i >= 0 && isspace(line[i]); i--) {}

      int name_end = i + 1;
      DALI_ENFORCE(
          name_end > 0 && name_end < label_start && label_start >= 2 && label_end > label_start,
          make_string("Incorrect format of the list file \"", file_list, "\":", n,
                      " expected file name followed by a label; got: ", line));

      line[label_end] = 0;
      line[name_end] = 0;

      image_label_pairs.emplace_back(line, std::atoi(line + label_start));
    }
    std::sort(image_label_pairs.begin(), image_label_pairs.end());
    DALI_ENFORCE(s.eof(), "Wrong format of file_list: " + file_list);

    return image_label_pairs;
  }

 protected:
  DiscoverFilesTest()
      : file_root(testing::dali_extra_path() + "/db/single/jpeg"),
        file_label_pairs(readFileLabelFile()) {}

  std::vector<std::string> globMatch(std::vector<std::string> &filters, std::string path) {
    std::vector<std::string> correct_match;
    glob_t pglob;
    for (auto &filter : filters) {
      std::string pattern = path + filesystem::dir_sep + '*' + filesystem::dir_sep + filter;
      if (glob(pattern.c_str(), GLOB_TILDE, NULL, &pglob) == 0) {
        for (unsigned int count = 0; count < pglob.gl_pathc; ++count) {
          std::string match(pglob.gl_pathv[count]);
          correct_match.push_back(match.substr(path.length() + 1, std::string::npos));
        }
        globfree(&pglob);
      }
    }
    std::sort(correct_match.begin(), correct_match.end());
    std::unique(correct_match.begin(), correct_match.end());
    return correct_match;
  }

  std::string file_root;
  std::vector<std::pair<std::string, int>> file_label_pairs;
};

TEST_F(DiscoverFilesTest, MatchAllFilter) {
  auto file_label_pairs_filtered =
      discover_files(file_root, {true, false, kKnownExtensionsGlob, {}});
  ASSERT_EQ(this->file_label_pairs.size(), file_label_pairs_filtered.size());
  for (size_t i = 0; i < file_label_pairs_filtered.size(); ++i) {
    ASSERT_EQ(this->file_label_pairs[i].first, file_label_pairs_filtered[i].filename);
  }
}

TEST_F(DiscoverFilesTest, SingleFilter) {
  std::vector<std::string> filters{"dog*.jpg"};
  auto file_label_pairs_filtered =
      discover_files(file_root, {true, false, filters});
  std::vector<std::string> correct_match = globMatch(filters, file_root);


  for (size_t i = 0; i < file_label_pairs_filtered.size(); ++i) {
    ASSERT_EQ(correct_match[i], file_label_pairs_filtered[i].filename);
  }
}

TEST_F(DiscoverFilesTest, MultipleOverlappingFilters) {
  std::vector<std::string> filters{"dog*.jpg", "snail*.jpg", "*_1280.jpg"};
  auto file_label_pairs_filtered =
      discover_files(file_root, {true, false, filters});
  std::vector<std::string> correct_match = globMatch(filters, file_root);

  for (size_t i = 0; i < file_label_pairs_filtered.size(); ++i) {
    EXPECT_EQ(correct_match[i], file_label_pairs_filtered[i].filename);
  }
}

TEST_F(DiscoverFilesTest, CaseSensitiveFilters) {
  std::vector<std::string> filters{"*.jPg"};
  std::string root = (testing::dali_extra_path() + "/db/single/case_sensitive");
  auto file_label_pairs_filtered = discover_files(root, {true, true, filters});
  std::vector<std::string> correct_match = globMatch(filters, root);

  for (size_t i = 0; i < file_label_pairs_filtered.size(); ++i) {
    EXPECT_EQ(correct_match[i], file_label_pairs_filtered[i].filename);
  }
}

TEST_F(DiscoverFilesTest, CaseInsensitiveFilters) {
  std::vector<std::string> filters{"*.jPg"};
  std::vector<std::string> glob_filters{"*.jpg", "*.jpG", "*.jPg", "*.jPG",
                                        "*.Jpg", "*.JpG", "*.JPg", "*.JPG"};
  std::string root = (testing::dali_extra_path() + "/db/single/case_sensitive");
  auto file_label_pairs_filtered = discover_files(root, {true, false, filters});
  std::vector<std::string> correct_match = globMatch(glob_filters, root);

  for (size_t i = 0; i < file_label_pairs_filtered.size(); ++i) {
    EXPECT_EQ(correct_match[i], file_label_pairs_filtered[i].filename);
  }
}

}  // namespace dali
