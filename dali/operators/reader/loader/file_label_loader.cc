// Copyright (c) 2017-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/operators/reader/loader/file_label_loader.h"
#include <cerrno>
#include <cstdlib>
#include <limits>
#include <memory>
#include <stdexcept>
#include "dali/core/common.h"
#include "dali/operators/reader/loader/filesystem.h"
#include "dali/operators/reader/loader/utils.h"
#include "dali/util/file.h"
#include "dali/util/uri.h"
#include "dali/core/call_at_exit.h"

namespace dali {

using filesystem::dir_sep;

FileLabel MakeFileLabel(int64_t value, DALIDataType dtype) {
  switch (dtype) {
    case DALI_INT32:
      if (value < std::numeric_limits<int32_t>::min() ||
          value > std::numeric_limits<int32_t>::max())
        throw std::invalid_argument(
            make_string("Label ", value, " is out of range of the label type ", dtype, "."));
      return static_cast<int32_t>(value);
    case DALI_INT64:
      return value;
    case DALI_FLOAT:
      return static_cast<float>(value);
    default:
      throw std::invalid_argument(make_string("Unsupported label type: ", dtype, "."));
  }
}

FileLabel MakeFileLabel(float value, DALIDataType dtype) {
  if (dtype != DALI_FLOAT)
    throw std::invalid_argument(
        make_string("Cannot store a floating point label in the label type ", dtype, "."));
  return value;
}

std::optional<FileLabel> ParseFileLabel(const char *str, DALIDataType dtype) {
  char *end = nullptr;
  errno = 0;
  switch (dtype) {
    case DALI_INT32:
    case DALI_INT64: {
      long long value = std::strtoll(str, &end, 10);  // NOLINT(runtime/int)
      if (end == str || *end != '\0' || errno == ERANGE)
        return std::nullopt;
      if (dtype == DALI_INT64)
        return static_cast<int64_t>(value);
      if (value < std::numeric_limits<int32_t>::min() ||
          value > std::numeric_limits<int32_t>::max())
        return std::nullopt;
      return static_cast<int32_t>(value);
    }
    case DALI_FLOAT: {
      float value = std::strtof(str, &end);
      if (end == str || *end != '\0' || errno == ERANGE)
        return std::nullopt;
      return value;
    }
    default:
      throw std::invalid_argument(make_string("Unsupported label type: ", dtype, "."));
  }
}

template<bool checkpointing_supported>
void FileLabelLoaderBase<checkpointing_supported>::PrepareEmpty(ImageLabelWrapper &image_label) {
  PrepareEmptyTensor(image_label.image);
}

template<bool checkpointing_supported>
void FileLabelLoaderBase<checkpointing_supported>::ReadSample(ImageLabelWrapper &image_label) {
  auto entry = file_label_entries_[current_index_++];

  // handle wrap-around
  MoveToNextShard(current_index_);

  // should be cleared by now
  assert(image_label.file_stream == nullptr);

  // copy the label
  image_label.label = GetLabel(entry);
  DALIMeta meta;
  meta.SetSourceInfo(entry.filename);
  meta.SetSkipSample(false);

  // if image is cached, skip loading
  if (ShouldSkipImage(entry.filename)) {
    meta.SetSkipSample(true);
    image_label.image.Reset();
    image_label.image.SetMeta(meta);
    image_label.image.Resize({0}, DALI_UINT8);
    return;
  }

  auto path = filesystem::join_path(file_root_, entry.filename);
  FileStream::Options opts;
  opts.read_ahead = read_ahead_;
  opts.use_mmap = !copy_read_data_;
  opts.use_odirect = false;
  auto uri = URI::Parse(path, URI::ParseOpts::AllowNonEscaped);
  bool local_file = !uri.valid() || uri.scheme() == "file";
  auto current_file = FileStream::Open(path, opts, entry.size);
  auto current_file_cleanup = AtScopeExit([&current_file] {
    if (current_file)
      current_file->Close();
  });
  Index file_size = current_file->Size();

  if (copy_read_data_ || !current_file->CanMemoryMap()) {
    if (image_label.image.shares_data()) {
      image_label.image.Reset();
    }
    if (local_file) {
      // if local file, read right away
      image_label.image.Resize({file_size}, DALI_UINT8);
      int64_t read_nbytes =
          current_file->Read(image_label.image.mutable_data<uint8_t>(), file_size);
      DALI_ENFORCE(read_nbytes == file_size, make_string("Failed to read file: ", entry.filename));
    } else {
      // if URI, defer reading
      image_label.file_stream = std::move(current_file);
    }
  } else {
    auto p = current_file->Get(file_size);
    DALI_ENFORCE(p != nullptr, make_string("Failed to read file: ", entry.filename));
    // Wrap the raw data in the Tensor object.
    image_label.image.ShareData(p, file_size, false, {file_size}, DALI_UINT8, CPU_ONLY_DEVICE_ID);
  }
  image_label.image.SetMeta(meta);
}

template<bool checkpointing_supported>
Index FileLabelLoaderBase<checkpointing_supported>::SizeImpl() {
  return static_cast<Index>(file_label_entries_.size());
}

template class FileLabelLoaderBase<false>;
template class FileLabelLoaderBase<true>;

}  // namespace dali
