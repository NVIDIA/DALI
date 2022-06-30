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

#include <stdio.h>
#include <string>
#include "dali/imgcodec/image_source.h"
#include "dali/util/file.h"

namespace dali {
namespace imgcodec {

ImageSource ImageSource::FromFilename(std::string filename) {
  return { InputKind::Filename, nullptr, 0, std::move(filename), nullptr };
}

ImageSource ImageSource::FromHostMem(const void *mem, size_t size, std::string source_info) {
  return { InputKind::HostMemory, mem, size, std::move(source_info), nullptr };
}

ImageSource ImageSource::FromDeviceMem(const void *mem, size_t size, std::string source_info) {
  return { InputKind::DeviceMemory, mem, size, std::move(source_info), nullptr };
}

ImageSource ImageSource::FromStream(std::shared_ptr<InputStream> stream, std::string source_info) {
  return { InputKind::Stream, nullptr, stream->Size(), std::move(source_info), std::move(stream) };
}

std::shared_ptr<InputStream> ImageSource::Open() const {
  switch (kind_) {
    case InputKind::Stream:
      return stream_;
    case InputKind::HostMemory:
      return std::make_shared<MemInputStream>(data_, size_);
    case InputKind::Filename:
      return std::shared_ptr<InputStream>(FileStream::Open(name_, false, false).release());

    case InputKind::DeviceMemory:
    default:
      throw std::logic_error("Not implemented");
  }
}


}  // namespace imgcodec
}  // namespace dali
