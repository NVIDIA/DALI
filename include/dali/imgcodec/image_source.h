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

#include <memory>
#include <string>
#include <utility>
#include "dali/core/api_helper.h"
#include "dali/core/int_literals.h"
#include "dali/core/stream.h"
#include "dali/core/access_order.h"
#include "dali/core/common.h"

namespace dali {
namespace imgcodec {

enum class InputKind : int {
  None = 0,
  // abstract stream interface that reads data from a custom file-like source
  Stream = 1,
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

constexpr bool operator!(InputKind k) {
  return k == InputKind::None;
}

/**
 * @brief A source of data from image parsers and codecs
 */
class DLL_PUBLIC ImageSource {
 public:
  ImageSource() = default;

  /**
   * @brief Creates an image source from a filename
   */
  static ImageSource FromFilename(std::string filename);

  /**
   * @brief Creates an image source from data in host memory
   */
  static ImageSource FromHostMem(const void *mem, size_t size, std::string source_info = "");

  /**
   * @brief Creates an image source from data in device memory
   */
  static ImageSource FromDeviceMem(const void *mem, size_t size, int device_id, AccessOrder order,
                                   std::string source_info = "");

  /**
   * @brief Creates an image source from an InputStream interface
   */
  static ImageSource FromStream(std::shared_ptr<InputStream> stream, std::string source_info = "");

  /**
   * @brief Creates an image source from an InputStream interface
   */
  static ImageSource FromStream(InputStream *stream, std::string source_info = "") {
    return FromStream(std::shared_ptr<InputStream>(stream, [](void*){}), std::move(source_info));
  }

  /**
   * @brief Returns the kind of source
   */
  InputKind Kind() const { return kind_; }

  /**
   * @brief Access the raw data pointer (makes sense for HostMemory and DeviceMemory kinds)
   */
  template <typename T = void>
  const T *RawData() const { return static_cast<const T *>(data_); }

  /**
   * @brief Access the data size (does NOT make sense for Filename kind)
   */
  size_t Size() const {
    if (size_ == -1_uz)
      throw std::logic_error("Unknown size.");
    return size_;
  }

  /**
   * @brief Returns the device id associated with the data.
   *        Only makes sense for DeviceMem kind
   */
  int DeviceId() const {
    return device_id_;
  }

  /**
   * @brief Returns the access order for the memory.
   *        Makes sense for HostMem and DeviceMem kinds
   */
  AccessOrder Order() const {
    return order_;
  }

  /**
   * @brief Access an InputStream associated with the ImageSource.
   *        The stream must be already created (e.g. via Open)
   */
  const std::shared_ptr<InputStream> &Stream() const { return stream_; }

  /**
   * @brief Gets the filename, if the kind of image source is Filename - otherwise throws
   */
  const char *Filename() const {
    if (kind_ != InputKind::Filename)
      throw std::logic_error("This image source doesn't have a filename.");
    return name_.empty() ? nullptr : name_.c_str();
  }

  /**
   * @brief Returns a human-readable unique identifier of the image source
   *
   * The source info is a string that is unique in the system - it can be a filename (path)
   * or some compound identifier, like a name of a container file accompanied by
   * a record id or an offset,
   */
  const char *SourceInfo() const { return name_.empty() ? nullptr : name_.c_str(); }

  /**
   * @brief Opens the image source, returning an input stream
   *
   * If the image source is InputKind::Stream, the stream is rewound and returned directly.
   * Otherwise, a memory stream or a file is opened.
   *
   * @return std::shared_ptr<InputStream>
   */
  std::shared_ptr<InputStream> Open() const;

 private:
  ImageSource(InputKind kind,
              const void *data, size_t size,
              std::string name,
              std::shared_ptr<InputStream> stream = {},
              int device_id = CPU_ONLY_DEVICE_ID,
              AccessOrder order = AccessOrder::host())
  : kind_(kind)
  , data_(data)
  , size_(size)
  , name_(std::move(name))
  , stream_(std::move(stream)) {}

  InputKind kind_ = InputKind::None;
  const void *data_ = nullptr;
  size_t size_ = -1_uz;
  // storage for filename or source info
  std::string name_;
  std::shared_ptr<InputStream> stream_;
  int device_id_ = CPU_ONLY_DEVICE_ID;
  AccessOrder order_ = AccessOrder::host();
};

}  // namespace imgcodec
}  // namespace dali

#endif  // DALI_IMGCODEC_IMAGE_SOURCE_H_
