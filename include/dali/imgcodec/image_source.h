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

#ifndef DALI_IMGCODEC_IMAGE_SOURCE_H_
#define DALI_IMGCODEC_IMAGE_SOURCE_H_

namespace dali {
namespace imgcodec {

enum class InputKind : int {
  None = 0,
  // abstract stream interface that reads data from a custom file-like source
  StreamInterface = 1,
  // bitstream loaded into host memory
  HostMemory = 2,
  // bitstream loaded into device memory
  DeviceMemory = 4,
  // file name
  Filename = 8,
};

constexpr InputKind operator|(InputKind a, InputKind b) {
  return static_cast<InputKind>(static_cast<int>(a) | static_cast<int>(b));
}

constexpr InputKind operator&(InputKind a, InputKind b) {
  return static_cast<InputKind>(static_cast<int>(a) & static_cast<int>(b));
}

class DLL_PUBLIC ImageSource {
 public:
  ImageSource() = default;
  static ImageSource FromFilename(std::string filename);
  static ImageSource FromHostMem(const void *mem, size_t size, std::string source_info = "");
  static ImageSource FromDeviceMem(const void *mem, size_t size, std::string source_info = "");
  static ImageSource FromStream(InputStream *stream, std::string source_info = "");

  InputKind Kind() const { return kind_; }
  template <typename T = void>
  const T *RawData() const { return static_cast<const T *>(data_); }
  size_t Size() const { return size_; }
  InputStream *Stream() const { return stream_; }
  const char *Filename() const {
    if (kind_ != InputKind::Filename)
      throw std::logic_error("This image source doesn't have a filename.");
    return name_.c_str();
  }
  const char *SourceInfo() const { return name_.c_str(); }

  shared_ptr<InputStream> Open() const {
    if (stream_)
      return { stream_, [](void*){} };
    else {
      return {};  // TODO(michalz)
    }
  }
 private:
  InputKind kind_ = InputKind::None;
  const void *data_ = nullptr;
  size_t size_ = 0;
  // storage for filename or source info
  std::string name_;
  InputStream *stream_ = nullptr;
};

}  // namespace imgcodec
}  // namespace dali

#endif  // DALI_IMGCODEC_IMAGE_SOURCE_H_
