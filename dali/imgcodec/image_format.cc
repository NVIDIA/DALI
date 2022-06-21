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

#include "dali/imgcodec/image_format.h"
#include <string>

namespace dali {
namespace imgcodec {

ImageFormat::ImageFormat(const char *name, shared_ptr<ImageParser> parser)
    : name_(name), parser_(parser) {}

bool ImageFormat::Matches(EncodedImage *encoded) const {
  return parser_->CanParse(encoded);
}

ImageParser *ImageFormat::Parser() const {
  return parser_.get();
}

const std::string &ImageFormat::Name() const {
  return name_;
}

span<ImageCodec *> ImageFormat::Codecs() {
  codecs_ptrs_.reserve(codecs_.size());
  codecs_ptrs_.clear();
  for (auto &codec : codecs_)
    codecs_ptrs_.push_back(codec.get());
  return make_span(codecs_ptrs_);
}

void ImageFormat::RegisterCodec(std::shared_ptr<ImageCodec> codec, float priority) {}

void ImageFormatRegistry::RegisterFormat(std::shared_ptr<ImageFormat> format) {
  formats_.push_back(format);
}

ImageFormat *ImageFormatRegistry::GetImageFormat(EncodedImage *image) const {
  for (auto &format : formats_) {
    if (format->Matches(image)) {
      return format.get();
    }
  }
  return nullptr;
}

span<ImageFormat *> ImageFormatRegistry::Formats() const {
  formats_ptrs_.reserve(formats_.size());
  formats_ptrs_.clear();
  for (auto &ptr : formats_)
    formats_ptrs_.push_back(ptr.get());
  return make_span(formats_ptrs_);
}

}  // namespace imgcodec
}  // namespace dali
