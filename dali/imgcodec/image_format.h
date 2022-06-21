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

#ifndef DALI_IMGCODEC_IMAGE_FORMAT_H_
#define DALI_IMGCODEC_IMAGE_FORMAT_H_

#include <fstream>
#include <stdexcept>
#include <vector>
#include <memory>
#include "dali/core/span.h"
#include "dali/core/tensor_shape.h"
#include "dali/pipeline/data/sample_view.h"

namespace dali {
namespace imgcodec {

struct ImageInfo {
  TensorShape<> shape;
  struct {
    int rotate;
    bool flip_x, flip_y;
  } orientation;
};

enum class InputKind : int {
  None = 0,
  // abstract interface that reads data from a custom source
  ReadInterface = 1,
  // bitstream loaded into host memory
  HostMemory = 2,
  // bitstream loaded into device memory
  DeviceMemory = 4,
  // file name
  Filename = 8
};

constexpr InputKind operator|(InputKind a, InputKind b) {
  return static_cast<InputKind>(static_cast<int>(a) | static_cast<int>(b));
}

constexpr InputKind operator&(InputKind a, InputKind b) {
  return static_cast<InputKind>(static_cast<int>(a) & static_cast<int>(b));
}

class EncodedImage {
 public:
  virtual void *GetRawData() const = 0;
  virtual size_t GetSize() const = 0;
  virtual const char *GetFilename() const = 0;
  virtual ssize_t Read(void *buf, size_t count) = 0;
  virtual ssize_t Seek(ssize_t pos) = 0;
  virtual InputKind GetKind() = 0;
};

class EncodedImageHostMemory : public EncodedImage {
 public:
  EncodedImageHostMemory(void *data, size_t len) : ptr_(data), size_(len) {}

  void *GetRawData() const override {
    return ptr_;
  }

  size_t GetSize() const override {
    return size_;
  }

  const char *GetFilename() const override {
    throw std::runtime_error("Not supported interface");
  }

  ssize_t Read(void *buf, size_t count) override {
    throw std::runtime_error("Not supported interface");
  }

  ssize_t Seek(ssize_t pos) override {
    throw std::runtime_error("Not supported interface");
  }

  InputKind GetKind() override {
    return InputKind::HostMemory;
  }

 private:
  void *ptr_;
  size_t size_;
};

class ImageParser {
 public:
  virtual ImageInfo GetInfo(EncodedImage *encoded) const = 0;
  virtual bool CanParse(EncodedImage *encoded) const = 0;
};

class ImageParserManager {
 public:
  void RegisterParser(std::shared_ptr<ImageParser> parser, int position = -1) {
    if (position < 0)
      parsers_.push_back(parser);
    else
      parsers_.insert(parsers_.begin() + position, parser);
  }

  ImageInfo Parse(EncodedImage *encoded) const {
    for (auto &parser : parsers_) {
      if (parser->CanParse(encoded)) {
        return parser->GetInfo(encoded);
      }
    }
    return {};  // TODO(janton): error?
  }

 private:
  std::vector<std::shared_ptr<ImageParser>> parsers_;
};

class ImageCodec;

class ImageFormat {
 public:
  ImageFormat(const char *name, shared_ptr<ImageParser> parser);
  bool Matches(EncodedImage *encoded) const;
  ImageParser *Parser() const;
  span<ImageCodec *> Codecs();
  void RegisterCodec(std::shared_ptr<ImageCodec> decoder, float priority);
};

class ImageFormatRegistry {
 public:
  void RegisterFormat(ImageFormat format);
  ImageFormat *GetImageFormat(EncodedImage *image) const;
  span<ImageFormat *> Formats() const;
};

}  // namespace imgcodec
}  // namespace dali

#endif  // DALI_IMGCODEC_IMAGE_FORMAT_H_
