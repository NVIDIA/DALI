// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

#include <sys/stat.h>
#include <dirent.h>
#include "dali/util/image.h"

namespace dali {

namespace {

inline bool ends_with(const std::string &value, const std::string &ending) {
    if (ending.size() > value.size()) return false;
    return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}

bool is_supported_extension(const std::string &file,
                            const std::vector<std::string> &supported_extensions) {
  for (const auto &extension : supported_extensions) {
    if (ends_with(file, extension))
      return true;
  }
  return false;
}

bool is_empty_file(const std::string &full_path) {
  struct stat stat_record;
  const bool empty_file = (stat(full_path.c_str(), &stat_record) || stat_record.st_size <= 1);
  return empty_file;
}

std::vector<std::string> list_files(
  const std::string& directory,
  const std::vector<std::string> &supported_extensions) {
  std::vector<std::string> image_list;
  struct dirent *dir = nullptr;
  DIR *d = opendir(directory.c_str());
  DALI_ENFORCE(d, "didn't find any files in `" + directory + "`");
  while ((dir=readdir(d)) != nullptr) {
#if !defined(__AARCH64_QNX__)
    if (dir->d_type != DT_REG)
      continue;
#endif
    std::string filename{dir->d_name};
    if (!is_supported_extension(filename, supported_extensions))
      continue;

    std::string full_path = directory + "/" + dir->d_name;
    if (is_empty_file(full_path))
      continue;

    image_list.push_back(full_path);
  }
  return image_list;
}

}  // namespace

void LoadImages(const vector<string> &image_names, ImgSetDescr *imgs) {
  for (auto img_name : image_names) {
    std::ifstream img_file(img_name);
    DALI_ENFORCE(img_file.is_open());

    img_file.seekg(0, std::ios::end);
    int img_size = static_cast<int>(img_file.tellg());
    img_file.seekg(0, std::ios::beg);

    auto data = new uint8[img_size];
    imgs->data_.push_back(data);
    imgs->sizes_.push_back(img_size);
    img_file.read(reinterpret_cast<char*>(data), img_size);
    imgs->filenames_.push_back(img_name);
  }
}

std::vector<std::string> ImageList(const std::string& image_folder,
                                   const std::vector<std::string> &supported_extensions) {
  std::vector<std::string> image_names;
  const string image_list = image_folder  + "/image_list.txt";
  std::ifstream file(image_list);
  if (file.is_open()) {
    std::string filename;
    while (file >> filename) {
      DALI_ENFORCE(!filename.empty());
      std::string full_path = image_folder + "/" + filename;
      if (is_supported_extension(filename, supported_extensions) && !is_empty_file(full_path)) {
        image_names.push_back(full_path);
      }
    }
    return image_names;
  }

  return list_files(image_folder, supported_extensions);
}

void LoadFromFile(const string &file_name, uint8 **image, int *h, int *w, int *c) {
  std::ifstream file(file_name + ".txt");
  DALI_ENFORCE(file.is_open());

  file >> *h;
  file >> *w;
  file >> *c;

  // lol at this multiplication
  int size = (*h)*(*w)*(*c);
  *image = new uint8[size];
  int tmp = 0;
  for (int i = 0; i < size; ++i) {
    file >> tmp;
    (*image)[i] = (uint8)tmp;
  }
}

void WriteHWCImage(const uint8 *img, int h, int w, int c, const string &file_name) {
  WriteImageScaleBias(img, h, w, c, 0.f, 1.0f, file_name, outHWCImage);
}

void WriteBatch(const TensorList<CPUBackend> &tl, const string &suffix, float bias, float scale) {
  const auto type = tl.type();
  const auto layout = tl.GetLayout();

  if (IsType<uint8>(type)) {
    if (layout == DALI_NCHW)
      WriteCHWBatch<uint8>(tl, bias, scale, suffix);
    else
      WriteHWCBatch<uint8>(tl, bias, scale, suffix);
  } else if (IsType<int16>(type)) {
    if (layout == DALI_NCHW)
      WriteCHWBatch<int16>(tl, bias, scale, suffix);
    else
      WriteHWCBatch<int16>(tl, bias, scale, suffix);
  } else if (IsType<int32>(type)) {
    if (layout == DALI_NCHW)
      WriteCHWBatch<int32>(tl, bias, scale, suffix);
    else
      WriteHWCBatch<int32>(tl, bias, scale, suffix);
  } else if (IsType<int64>(type)) {
    if (layout == DALI_NCHW)
      WriteCHWBatch<int64>(tl, bias, scale, suffix);
    else
      WriteHWCBatch<int64>(tl, bias, scale, suffix);
  } else if (IsType<float16>(type)) {
    if (layout == DALI_NCHW)
      WriteCHWBatch<float16>(tl, bias, scale, suffix);
    else
      WriteHWCBatch<float16>(tl, bias, scale, suffix);
  } else if (IsType<float>(type)) {
    if (layout == DALI_NCHW)
      WriteCHWBatch<float>(tl, bias, scale, suffix);
    else
      WriteHWCBatch<float>(tl, bias, scale, suffix);
  }
}

}  // namespace dali
