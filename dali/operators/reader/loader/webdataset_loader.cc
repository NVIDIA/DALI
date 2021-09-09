// Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/operators/reader/loader/webdataset_loader.h"
#include <cstddef>
#include <cstring>
#include <limits>
#include <memory>
#include <utility>
#include "dali/core/error_handling.h"
#include "dali/pipeline/data/types.h"
#include "dali/util/file.h"

namespace dali {
namespace detail {
namespace wds {

inline MissingExtBehavior ParseMissingExtBehavior(std::string missing_component_behavior) {
  std::transform(missing_component_behavior.begin(), missing_component_behavior.end(),
                 missing_component_behavior.begin(), static_cast<int (*)(int)>(std::tolower));
  if (missing_component_behavior == "") {
    return MissingExtBehavior::Empty;
  } else if (missing_component_behavior == "skip") {
    return MissingExtBehavior::Skip;
  } else if (missing_component_behavior == "empty") {
    return MissingExtBehavior::Empty;
  } else if (missing_component_behavior == "error") {
    return MissingExtBehavior::Raise;
  } else {
    return MissingExtBehavior::Invalid;
  }
}

inline SampleDesc ParseSampleDesc(std::ifstream& index_file, const std::string& index_path,
                                  int64_t line) {
  SampleDesc out;
  DALI_ENFORCE(index_file >> out.start_offset,
               make_string("Malformed index file at ", index_path, " line ", line,
                           " - less components than stated at the beginning of the index file"));

  DALI_ENFORCE(out.start_offset % detail::kBlockSize == 0,
               make_string("Malformed index file at ", index_path, " line ", line, " - offset ",
                           out.start_offset, " not divisible by ", detail::kBlockSize));
  out.end_offset = std::numeric_limits<decltype(out.end_offset)>::max();

  std::string component_metadata;
  std::getline(index_file, component_metadata);

  std::stringstream extensions_stream(component_metadata);
  std::string extension;
  int64_t size;

  while (extensions_stream >> extension) {
    DALI_ENFORCE(extensions_stream >> size,
                 make_string("Malformed index file at ", index_path, " line ", line,
                             " - size corresponding to the extension not found"));
    out.index_file_metadata.push_back(ComponentDesc(std::move(extension), size));
  }
  DALI_ENFORCE(out.index_file_metadata.size() > 0_uz,
               make_string("Malformed index file at ", index_path, " line ", line,
                           " - no extensions provided for the sample"));
  return out;
}

inline std::vector<SampleDesc> ParseIndexFile(MissingExtBehavior missing_component_behavior,
                                              const std::string& index_path,
                                              std::vector<std::set<std::string>>& ext) {
  std::ifstream index_file(index_path);
  int64_t index_size;
  size_t sample_desc_num = 0;  // for index file validity check
  index_file >> index_size >> sample_desc_num;
  DALI_ENFORCE(sample_desc_num > 0_uz,
               "Empty index file at " + index_path);  // index file validity check

  DALI_ENFORCE(index_size % detail::kBlockSize == 0,
               make_string("Malformed index file at ", index_path, " line 0 - final offset ",
                           index_size, " not divisible by ", detail::kBlockSize));

  std::vector<SampleDesc> out;
  out.reserve(sample_desc_num);

  for (size_t sample_index = 0; sample_index < sample_desc_num; sample_index++) {
    SampleDesc new_sample = ParseSampleDesc(index_file, index_path, sample_index + 1);
    DALI_ENFORCE(new_sample.start_offset < index_size,
                 make_string("Malformed index file at ", index_path, " line ", sample_index + 1,
                             " - reported final offset smaller than a sample start offset"));
    if (!out.empty()) {
      out.back().end_offset = std::min(out.back().end_offset, new_sample.start_offset);
      DALI_ENFORCE(out.back().start_offset < out.back().end_offset,
                   make_string("Malformed index file at ", index_path, " line ", sample_index + 1,
                               " - sample offsets not in order"));
    }

    // filtering out samples without the required extensions
    bool discard = false;
    if (!std::all_of(ext.begin(), ext.end(), [&](std::set<std::string> extension_set) {
          for (auto& component_desc : new_sample.index_file_metadata) {
            if (extension_set.count(component_desc.ext)) {
              return true;
            }
          }
          return false;
        })) {
      switch (missing_component_behavior) {
        case MissingExtBehavior::Skip:
          discard = true;
          break;
        case MissingExtBehavior::Raise:
          DALI_FAIL(make_string("Underful sample detected in index file at ", index_path, " line ",
                                sample_index + 1));
        default:
          break;
      }
    }
    if (discard) {
      continue;
    }

    out.push_back(std::move(new_sample));
  }
  if (out.size()) {
    out.back().end_offset = index_size;
  }
  return out;
}

}  // namespace wds
}  // namespace detail

inline std::string SupportedTypesListGen() {
  std::stringstream out;
  for (auto& dtype : detail::wds::kSupportedTypes) {
    out << dtype << ',';
  }
  if (!detail::wds::kSupportedTypes.empty()) {
    out.unget();
  }
  return out.str();
}

WebdatasetLoader::WebdatasetLoader(const OpSpec& spec)
    : Loader(spec),
      uris_(spec.GetRepeatedArgument<std::string>("uris")),
      index_paths_(spec.GetRepeatedArgument<std::string>("index_paths")),
      missing_component_behavior_(detail::wds::ParseMissingExtBehavior(
          spec.GetArgument<std::string>("missing_component_behavior"))) {
  DALI_ENFORCE(uris_.size() == index_paths_.size(),
               "Number of uris does not match the number of index files");
  DALI_ENFORCE(uris_.size() > 0, "No webdataset shards provided");
  DALI_ENFORCE(missing_component_behavior_ != detail::wds::MissingExtBehavior::Invalid,
               "Invalid value for missing_component_behavior");

  std::vector<std::string> samples_exts = spec.GetRepeatedArgument<std::string>("ext");
  ext_.reserve(samples_exts.size());

  // splitting extension bundles by the delimiter
  for (size_t exts_idx = 0; exts_idx < samples_exts.size(); exts_idx++) {
    std::stringstream exts_stream(samples_exts[exts_idx]);
    std::string ext;
    ext_.emplace_back();
    while (std::getline(exts_stream, ext, detail::wds::kExtDelim)) {
      if (!ext_.back().count(ext)) {
        ext_.back().insert(ext);
        ext_map_[ext].push_back(exts_idx);
      }
    }
  }

  dtypes_ = spec.HasArgument("dtypes") ? spec.GetRepeatedArgument<DALIDataType>("dtypes") :
                                         std::vector<DALIDataType>(ext_.size(), DALI_UINT8);
  for (auto& dtype : dtypes_) {
    DALI_ENFORCE(detail::wds::kSupportedTypes.count(dtype),
                 "Unsupported output dtype. Supported types include: " + SupportedTypesListGen());
  }
  DALI_ENFORCE(ext_.size() == dtypes_.size(),
               "Number of extensions does not match the number of types");
}

WebdatasetLoader::~WebdatasetLoader() {}

void WebdatasetLoader::PrepareEmpty(vector<Tensor<CPUBackend>>& empty) {
  empty = std::vector<Tensor<CPUBackend>>(ext_.size());
  for (auto& tensor : empty) {
    tensor.reserve(tensor_init_bytes_);
  }
}

inline std::string GetExtension(const std::string& filepath) {
  const size_t base_name_pos = filepath.find_last_of('/') + 1;
  const size_t dot_pos = filepath.find_first_of('.', base_name_pos);
  return filepath.substr(dot_pos + 1);
}

inline Index WebdatasetLoader::GetCurrentSampleIndex() const {
  return wds_shards_prefixsums_[current_wds_shard_index_] + current_sample_index_;
}

inline void WebdatasetLoader::SetDataPointer(std::vector<Tensor<CPUBackend>>& sample,
                                             bitmask& component_was_set,
                                             const std::string& extension,
                                             const std::string& source_info,
                                             std::shared_ptr<void> data, int64_t size) const {
  DALIMeta meta;
  meta.SetSourceInfo(source_info);
  meta.SetSkipSample(false);

  for (size_t component_index : ext_map_.at(extension)) {
    if (!component_was_set[component_index]) {
      auto dtype_info = TypeTable::GetTypeInfo(dtypes_[component_index]);
      sample[component_index].SetMeta(meta);
      DALI_ENFORCE(size % static_cast<int64_t>(dtype_info.size()) == 0,
                   "Index file at " + index_paths_[current_wds_shard_index_] +
                       " reporting component sizes different to actual");
      sample[component_index].ShareData(
          data, size, {size / static_cast<int64_t>(dtype_info.size())}, dtype_info);
      component_was_set[component_index] = true;
    }
  }
}


inline uint8_t* WebdatasetLoader::ShareDataPointer(std::vector<Tensor<CPUBackend>>& sample,
                                                   bitmask& component_was_set,
                                                   const std::string& extension,
                                                   const std::string& source_info,
                                                   int64_t size) const {
  DALIMeta meta;
  meta.SetSourceInfo(source_info);
  meta.SetSkipSample(false);

  uint8_t* shared_tensor_data = nullptr;
  for (size_t component_index : ext_map_.at(extension)) {
    if (!component_was_set[component_index]) {
      sample[component_index].SetMeta(meta);
      auto dtype_info = TypeTable::GetTypeInfo(dtypes_[component_index]);
      DALI_ENFORCE(size % static_cast<int64_t>(dtype_info.size()) == 0,
                   "Index file at " + index_paths_[current_wds_shard_index_] +
                       " reporting component sizes different to actual");
      if (shared_tensor_data == nullptr) {
        if (sample[component_index].shares_data()) {
          sample[component_index].Reset();
        }
        sample[component_index].reserve(size);
        sample[component_index].Resize({size / static_cast<int64_t>(dtype_info.size())},
                                       dtype_info);
        shared_tensor_data = reinterpret_cast<uint8_t*>(sample[component_index].raw_mutable_data());
      } else {
        sample[component_index].ShareData(shared_tensor_data, size,
                                          {size / static_cast<int64_t>(dtype_info.size())},
                                          dtype_info);
      }
      component_was_set[component_index] = true;
    }
  }
  return shared_tensor_data;
}

inline void WebdatasetLoader::MarkCached(std::vector<Tensor<CPUBackend>>& sample,
                                         bitmask& component_was_set,
                                         const std::string& extension,
                                         const std::string& source_info) const {
  DALIMeta meta;
  meta.SetSourceInfo(source_info);
  meta.SetSkipSample(true);

  for (size_t component_index : ext_map_.at(extension)) {
    if (!component_was_set[component_index]) {
      sample[component_index].Reset();
      sample[component_index].SetMeta(meta);
      sample[component_index].set_type(TypeTable::GetTypeInfo(dtypes_[component_index]));
      sample[component_index].Resize({0});
      component_was_set[component_index] = true;
    }
  }
}

void WebdatasetLoader::ReadSample(vector<Tensor<CPUBackend>>& sample) {
  MoveToNextShard(GetCurrentSampleIndex());
  auto& current_wds_shard = wds_shards_[current_wds_shard_index_];
  auto& current_sample = wds_shards_metadata_[current_wds_shard_index_][current_sample_index_];
  current_wds_shard.SeekArchive(current_sample.start_offset);

  bitmask component_was_set;
  component_was_set.resize(sample.size(), false);
  while (current_wds_shard.TellArchive() < current_sample.end_offset) {
    DALI_ENFORCE(!current_wds_shard.EndOfArchive(),
                 make_string("Index file at ", index_paths_[current_wds_shard_index_],
                             " reporting a file longer than actual (archive reached an offset ",
                             std::to_string(current_wds_shard.TellArchive()),
                             " and the sample is supposed to end at ",
                             std::to_string(current_sample.end_offset), ")"));
    // Check in case of encountering a tar entry that is not a file
    if (current_wds_shard.GetFileType() != detail::TarArchive::ENTRY_FILE) {
      current_wds_shard.NextFile();
      continue;
    }

    // Check in case of encountering an unneeded entry
    const std::string extension = GetExtension(current_wds_shard.GetFileName());
    if (!ext_map_.count(extension)) {
      current_wds_shard.NextFile();
      continue;
    }

    // Check in case skipping sample is necessary
    const std::string source_info = uris_[current_wds_shard_index_] + " at offset " +
                                    std::to_string(current_wds_shard.TellArchive());
    if (ShouldSkipImage(source_info)) {
      MarkCached(sample, component_was_set, extension, source_info);
      continue;
    }

    // Reading the data into the tensors
    int64_t size = static_cast<int64_t>(current_wds_shard.GetFileSize());

    if (!copy_read_data_) {
      auto p = current_wds_shard.ReadFile();
      DALI_ENFORCE(p != nullptr, "Error reading from a file " + uris_[current_wds_shard_index_]);
      SetDataPointer(sample, component_was_set, extension, source_info, p, size);
    } else {
      uint8_t* shared_tensor_data =
          ShareDataPointer(sample, component_was_set, extension, source_info, size);
      if (shared_tensor_data) {
        uint64_t n_read = current_wds_shard.Read(static_cast<uint8_t*>(shared_tensor_data), size);
        DALI_ENFORCE(static_cast<int64_t>(n_read) == size,
                     "Error reading from a file " + uris_[current_wds_shard_index_]);
      }
    }

    current_wds_shard.NextFile();
  }

  // setting empty components:
  int not_set_count = 0;
  for (size_t component_index = 0; component_index < sample.size(); component_index++) {
    if (!component_was_set[component_index]) {
      not_set_count++;
      sample[component_index].Reset();
      sample[component_index].set_type(TypeTable::GetTypeInfo(dtypes_[component_index]));
      sample[component_index].Resize({0});
    }
  }
  if (not_set_count && missing_component_behavior_ != detail::wds::MissingExtBehavior::Empty) {
    DALI_FAIL(make_string("Index file at ", index_paths_[current_wds_shard_index_],
                          " reporting different extensions in a sample to the actual ones"));
  }

  current_sample_index_++;
  while (current_wds_shard_index_ < wds_shards_metadata_.size() &&
         current_sample_index_ >= wds_shards_metadata_[current_wds_shard_index_].size()) {
    current_wds_shard_index_++;
    current_sample_index_ = 0;
  }
}

Index WebdatasetLoader::SizeImpl() {
  return total_size_;
}

void WebdatasetLoader::PrepareMetadataImpl() {
  if (!dont_use_mmap_) {
    mmap_reserver_ = FileStream::MappingReserver(static_cast<unsigned int>(uris_.size()));
  }
  copy_read_data_ = dont_use_mmap_ || !mmap_reserver_.CanShareMappedData();

  // reserving the data in vector fields
  wds_shards_metadata_.reserve(index_paths_.size());
  wds_shards_.reserve(uris_.size());
  wds_shards_prefixsums_.reserve(index_paths_.size());

  // collecting the index files
  for (size_t index_file_index = 0; index_file_index < index_paths_.size(); index_file_index++) {
    wds_shards_metadata_.emplace_back(
        ParseIndexFile(missing_component_behavior_, index_paths_[index_file_index], ext_));
    wds_shards_prefixsums_.push_back(total_size_);
    total_size_ += wds_shards_metadata_.back().size();

    // checking dtype compatibility
    for (auto& sample_desc : wds_shards_metadata_.back()) {
      bitmask was_component_assigned;
      was_component_assigned.resize(ext_.size(), false);
      for (auto& component_desc : sample_desc.index_file_metadata) {
        if (!ext_map_.count(component_desc.ext)) {
          continue;
        }
        for (auto& sample_index : ext_map_[component_desc.ext]) {
          if (!was_component_assigned[sample_index]) {
            DALI_ENFORCE(
                component_desc.size % TypeTable::GetTypeInfo(dtypes_[sample_index]).size() == 0,
                "Component of a tar file '" + uris_[index_file_index] + "' at offset " +
                    std::to_string(sample_desc.start_offset) +
                    " has a size not divisible by the chosen dtype's size of " +
                    std::to_string(TypeTable::GetTypeInfo(dtypes_[sample_index]).size()) +
                    " bytes");
            was_component_assigned[sample_index] = true;
          } else {
            DALI_WARN_ONCE(
                "Several components corresponding to a single output detected. Only the first one "
                "shall be returend.");
          }
        }
      }
    }
  }
  wds_shards_prefixsums_.push_back(total_size_);  // for the last shard when reaches the end

  DALI_ENFORCE(static_cast<int64_t>(total_size_) >= static_cast<int64_t>(num_shards_),
               "Number of shards bigger than the total number of samples");

  // initializing all the readers
  for (auto& uri : uris_) {
    wds_shards_.emplace_back(FileStream::Open(uri, read_ahead_, !dont_use_mmap_));
  }

  // seeking the first wds shard to use
  const size_t first_index_ = start_index(shard_id_, num_shards_, total_size_);

  while (wds_shards_prefixsums_[first_wds_shard_index_] +
             wds_shards_metadata_[first_wds_shard_index_].size() <=
         first_index_) {
    first_wds_shard_index_++;
  }
  first_sample_index_ = first_index_ - wds_shards_prefixsums_[first_wds_shard_index_];

  // initializing the first reader
  current_wds_shard_index_ = first_wds_shard_index_;
  current_sample_index_ = first_sample_index_;
}

void WebdatasetLoader::Reset(bool wrap_to_shard) {
  current_wds_shard_index_ = wrap_to_shard ? first_wds_shard_index_ : 0;
  current_sample_index_ = wrap_to_shard ? first_sample_index_ : 0;
}

}  // namespace dali
