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

bool ImageFormat::Matches(ImageSource *encoded) const {
  return parser_->CanParse(encoded);
}

const ImageParser *ImageFormat::Parser() const {
  return parser_.get();
}

const std::string &ImageFormat::Name() const {
  return name_;
}

span<ImageDecoder *const> ImageFormat::Decoders() const {
  return make_cspan(decoder_ptrs_);
}

void ImageFormat::RegisterDecoder(std::shared_ptr<ImageDecoder> decoder, float priority) {
  auto it = decoders_.emplace(priority, std::move(decoder));
  if (std::next(it) == decoders_.end()) {
    decoder_ptrs_.push_back(it->second.get());
  } else {
    decoder_ptrs_.clear();
    for (auto [priority, decoder] : decoders_) {
      decoder_ptrs_.push_back(decoder.get());
    }
  }
}

void ImageFormatRegistry::RegisterFormat(std::shared_ptr<ImageFormat> format) {
  formats_.push_back(std::move(format));
  format_ptrs_.push_back(formats_.back().get());
}

const ImageFormat *ImageFormatRegistry::GetImageFormat(ImageSource *image) const {
  for (auto &format : formats_) {
    if (format->Matches(image)) {
      return format.get();
    }
  }
  return nullptr;
}

span<ImageFormat* const> ImageFormatRegistry::Formats() const {
  return make_cspan(format_ptrs_);
}

size_t ImageParser::ReadHeader(uint8_t *buffer, ImageSource *encoded, size_t n) const {
  if (encoded->Kind() == InputKind::HostMemory) {
    if (encoded->Size() < n)
      n = encoded->Size();
    std::memcpy(buffer, encoded->RawData(), n);
  } else {
    auto stream = encoded->Open();
    return stream->Read(buffer, n);
  }
  return n;
}

}  // namespace imgcodec
}  // namespace dali
