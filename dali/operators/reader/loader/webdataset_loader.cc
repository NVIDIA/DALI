// Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <tuple>
#include <utility>
#include "dali/core/common.h"
#include "dali/core/version_util.h"
#include "dali/core/error_handling.h"
#include "dali/operators/reader/loader/webdataset/tar_utils.h"
#include "dali/pipeline/data/types.h"

namespace dali {

template <typename... Args>
inline std::string IndexFileErrMsg(const std::string& index_path, int64_t line,
                                   const Args&... details) {
  return make_string("Malformed index file at \"", index_path, "\" line ", line, " - ", details...);
}

namespace detail {
namespace wds {

inline MissingExtBehavior ParseMissingExtBehavior(std::string missing_component_behavior) {
  for (auto& c : missing_component_behavior)
    c = std::tolower(static_cast<unsigned char>(c));
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


inline void ParseSampleDesc(std::vector<SampleDesc>& samples_container,
                            std::vector<ComponentDesc>& components_container,
                            std::ifstream& index_file, const std::string& index_path, int64_t line,
                            int index_version) {
  // Preparing the SampleDesc
  samples_container.emplace_back();
  samples_container.back().components =
      VectorRange<ComponentDesc>(components_container, components_container.size());
  samples_container.back().line_number = line;

  // Getting the components data
  std::string components_metadata;
  std::getline(index_file, components_metadata);
  std::stringstream components_stream(components_metadata);

  // Reading consecutive components
  ComponentDesc component;
  while (components_stream >> component.ext) {
    if (index_version == MakeVersionNumber(1, 2)) {
      DALI_ENFORCE(
          components_stream >> component.offset >> component.size >> component.filename,
          IndexFileErrMsg(
              index_path, line,
              "Could not find all necessary component parameters (offset, size or filename). Every "
              "record in the index file should look like: `<ext> <offset> <size> <filename>`."));
    } else {
      DALI_ENFORCE(components_stream >> component.offset >> component.size,
                   IndexFileErrMsg(
                       index_path, line,
                       "Could not find all necessary component parameters (offset or size). Every "
                       "record in the index file should look like: `<ext> <offset> <size>`."));
    }
    DALI_ENFORCE(
        component.offset % kBlockSize == 0,
        IndexFileErrMsg(index_path, line, "tar offset is not a multiple of tar block size (",
                        kBlockSize, "), perhaps the size value is exported before offset?"));
    components_container.emplace_back(std::move(component));
    samples_container.back().components.num++;
  }

  // Finishing up the SampleDesc
  DALI_ENFORCE(samples_container.back().components.num,
               IndexFileErrMsg(index_path, line, "no extensions provided for the sample"));
}

inline int ParseIndexVersion(const string& version_str) {
  const char *s = version_str.c_str();
  assert(*s == 'v');
  s++;
  int major = atoi(s);
  s = strchr(s, '.');
  assert(s);
  s++;
  int minor = atoi(s);
  return MakeVersionNumber(major, minor);
}

inline void ParseIndexFile(std::vector<SampleDesc>& samples_container,
                           std::vector<ComponentDesc>& components_container,
                           const std::string& index_path) {
  std::ifstream index_file(index_path);

  // Index Checking
  std::string global_meta;
  getline(index_file, global_meta);
  std::stringstream global_meta_stream(global_meta);
  std::string index_version_str;
  DALI_ENFORCE(global_meta_stream >> index_version_str,
               IndexFileErrMsg(index_path, 0, "no version signature found"));
  DALI_ENFORCE(
      kSupportedIndexVersions.count(index_version_str) > 0,
      IndexFileErrMsg(index_path, 0,
                      make_string("Unsupported version of the index file (",
                                  index_version_str, ").")));
  int index_version = ParseIndexVersion(index_version_str);

  // Getting the number of samples in the index file
  int64_t sample_desc_num_signed;
  DALI_ENFORCE(global_meta_stream >> sample_desc_num_signed,
               IndexFileErrMsg(index_path, 0, "no sample count found"));
  DALI_ENFORCE(sample_desc_num_signed > 0,
               IndexFileErrMsg(index_path, 0, "sample count must be positive"));

  const size_t sample_desc_num = sample_desc_num_signed;
  samples_container.reserve(samples_container.size() + sample_desc_num);
  for (size_t sample_index = 0; sample_index < sample_desc_num; sample_index++) {
    ParseSampleDesc(samples_container, components_container, index_file, index_path,
                    sample_index + 1, index_version);
  }
}

std::tuple<std::string, std::string> split_name(const std::string& filepath) {
  size_t dot_pos = filepath.find('.', filepath.rfind('/') + 1);
  return {filepath.substr(0, dot_pos), filepath.substr(dot_pos + 1)};
}

inline void ParseTarFile(std::vector<SampleDesc>& samples_container,
                         std::vector<ComponentDesc>& components_container,
                         std::unique_ptr<FileStream>& tar_file) {
  int64_t initial_file_pos = tar_file->TellRead();
  TarArchive tar_archive(std::move(tar_file));

  std::string last_filename;
  // rewind to the first valid entry
  for (; !tar_archive.EndOfArchive(); tar_archive.NextFile()) {
    if (tar_archive.GetFileType() == TarArchive::ENTRY_FILE) {
      std::tie(last_filename, std::ignore) = split_name(tar_archive.GetFileName());
      break;
    }
  }
  size_t last_components_size = components_container.size();
  for (; !tar_archive.EndOfArchive(); tar_archive.NextFile()) {
    if (tar_archive.GetFileType() != TarArchive::ENTRY_FILE) {
      continue;
    }

    std::string basename, ext;
    std::tie(basename, ext) = split_name(tar_archive.GetFileName());

    if (basename.empty()) {
      continue;
    }

    if (basename != last_filename) {
      samples_container.emplace_back();
      samples_container.back().components =
          VectorRange<ComponentDesc>(components_container, last_components_size,
                                     components_container.size() - last_components_size);
      last_filename = basename;
      last_components_size = components_container.size();
    }

    components_container.emplace_back();
    components_container.back().size = tar_archive.GetFileSize();
    components_container.back().offset = tar_archive.TellArchive() + tar_archive.HeaderSize();
    components_container.back().ext = std::move(ext);
    components_container.back().filename = tar_archive.GetFileName();
  }
  samples_container.emplace_back();
  samples_container.back().components =
      VectorRange<ComponentDesc>(components_container, last_components_size,
                                 components_container.size() - last_components_size);

  tar_file = tar_archive.Release();
}

}  // namespace wds
}  // namespace detail

inline std::string SupportedTypesListGen() {
  std::stringstream out;
  for (auto& dtype : detail::wds::kSupportedTypes) {
    out << dtype << ", ";
  }
  std::string out_str = out.str();
  return out_str.substr(0, out_str.size() - 2 * (detail::wds::kSupportedTypes.size() > 0));
}

std::string str_tolower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c){ return std::tolower(c); });
    return s;
}

WebdatasetLoader::WebdatasetLoader(const OpSpec& spec)
    : Loader(spec),
      paths_(spec.GetRepeatedArgument<std::string>("paths")),
      index_paths_(spec.GetRepeatedArgument<std::string>("index_paths")),
      missing_component_behavior_(detail::wds::ParseMissingExtBehavior(
          spec.GetArgument<std::string>("missing_component_behavior"))),
      case_sensitive_extensions_(spec.GetArgument<bool>("case_sensitive_extensions")) {
  DALI_ENFORCE(paths_.size() == index_paths_.size() || index_paths_.size() == 0,
               make_string("The number of index files, if any, must match the number of archives ",
               "in the dataset"));
  DALI_ENFORCE(paths_.size() > 0, "No webdataset archives provided");
  DALI_ENFORCE(missing_component_behavior_ != detail::wds::MissingExtBehavior::Invalid,
               make_string("Invalid value for missing_component_behavior '",
                           spec.GetArgument<std::string>("missing_component_behavior"),
                           "' possible values are: skip, error, empty"));

  std::vector<std::string> samples_exts = spec.GetRepeatedArgument<std::string>("ext");
  ext_.reserve(samples_exts.size());

  // splitting extension bundles by the delimiter
  for (size_t exts_idx = 0; exts_idx < samples_exts.size(); exts_idx++) {
    std::stringstream exts_stream(samples_exts[exts_idx]);
    std::string ext;
    ext_.emplace_back();
    while (std::getline(exts_stream, ext, detail::wds::kExtDelim)) {
      if (!case_sensitive_extensions_) {
        ext = str_tolower(ext);
      }
      if (!ext_.back().count(ext)) {
        ext_.back().insert(ext);
      }
    }
  }

  dtypes_ = spec.HasArgument("dtypes") ? spec.GetRepeatedArgument<DALIDataType>("dtypes")
                                       : std::vector<DALIDataType>(ext_.size(), DALI_UINT8);

  for (DALIDataType dtype : dtypes_) {
    DALI_ENFORCE(detail::wds::kSupportedTypes.count(dtype),
                 make_string("Unsupported output dtype ", dtype,
                             ". Supported types are: ", SupportedTypesListGen()));
  }
  DALI_ENFORCE(ext_.size() == dtypes_.size(),
               "Number of extensions does not match the number of provided types");
}

WebdatasetLoader::~WebdatasetLoader() {}

void WebdatasetLoader::PrepareEmpty(vector<Tensor<CPUBackend>>& empty) {
  empty = std::vector<Tensor<CPUBackend>>(ext_.size());
  for (size_t output_index = 0; output_index < ext_.size(); output_index++) {
    empty[output_index].set_pinned(false);
    empty[output_index].reserve(tensor_init_bytes_);
    empty[output_index].set_type(dtypes_[output_index]);
  }
}

std::string WebdatasetLoader::GetSampleSource(const detail::wds::SampleDesc& sample) {
  if (generate_index_) {
    return make_string("tar file at \"", paths_[sample.wds_shard_index], '"');
  } else {
    return make_string("index file at \"", index_paths_[sample.wds_shard_index], "\" line ",
                       sample.line_number);
  }
}


void WebdatasetLoader::ReadSample(vector<Tensor<CPUBackend>>& sample) {
  MoveToNextShard(sample_index_);
  detail::wds::SampleDesc& current_sample = samples_[sample_index_];
  auto& current_wds_shard = wds_shards_[current_sample.wds_shard_index];

  for (auto& component : current_sample.components) {
    // Checking if the component data from the index file agrees with reality
    DALI_ENFORCE(
        component.offset < static_cast<int64_t>(current_wds_shard->Size()),
        IndexFileErrMsg(index_paths_[current_sample.wds_shard_index], current_sample.line_number,
                        "offset is outside of the archive file"));

    current_wds_shard->SeekRead(component.offset);

    // Skipping cached samples
    const std::string sample_key = make_string_delim(':', paths_[current_sample.wds_shard_index],
                                                     component.offset, component.filename);

    DALIMeta meta;
    meta.SetSourceInfo(sample_key);
    if (ShouldSkipImage(sample_key)) {
      meta.SetSkipSample(true);
      for (auto& output : component.outputs) {
        sample[output].Reset();
        sample[output].SetMeta(meta);
        sample[output].Resize({0}, dtypes_[output]);
      }
      continue;
    }
    // Reading Data
    if (copy_read_data_) {
      uint8_t* shared_tensor_data = nullptr;
      bool shared_tensor_is_pinned = false;
      int device_id = CPU_ONLY_DEVICE_ID;
      for (auto& output : component.outputs) {
        if (!shared_tensor_data) {
          if (sample[output].shares_data()) {
            sample[output].Reset();
          }
          sample[output].Resize(
              {static_cast<int64_t>(component.size / sample[output].type_info().size())},
              dtypes_[output]);
          shared_tensor_data = reinterpret_cast<uint8_t*>(sample[output].raw_mutable_data());
          shared_tensor_is_pinned = sample[output].is_pinned();
          device_id = sample[output].device_id();
        } else {
          sample[output].ShareData(
              shared_tensor_data, component.size, shared_tensor_is_pinned,
              {static_cast<int64_t>(component.size / sample[output].type_info().size())},
              sample[output].type(), device_id);
        }
      }
      DALI_ENFORCE(current_wds_shard->Read(shared_tensor_data, component.size) == component.size,
                   "Error reading from a file " + paths_[current_sample.wds_shard_index]);
    } else {
      auto data = current_wds_shard->Get(component.size);
      for (auto& output : component.outputs) {
        sample[output].SetMeta(meta);
        sample[output].ShareData(
            data, component.size, false,
            {static_cast<int64_t>(component.size / sample[output].type_info().size())},
            sample[output].type(), CPU_ONLY_DEVICE_ID);
      }
    }
  }

  // Setting non-filled outputs
  for (auto& empty_output : current_sample.empty_outputs) {
    sample[empty_output].Reset();
    sample[empty_output].Resize({0}, dtypes_[empty_output]);
  }
  sample_index_++;
}

Index WebdatasetLoader::SizeImpl() {
  return samples_.size();
}

void WebdatasetLoader::PrepareMetadataImpl() {
  if (!dont_use_mmap_) {
    mmap_reserver_ = FileStream::MappingReserver(static_cast<unsigned int>(paths_.size()));
  }
  copy_read_data_ = dont_use_mmap_ || !mmap_reserver_.CanShareMappedData();

  generate_index_ = index_paths_.size() == 0;
  if (generate_index_) {
    DALI_WARN("Index file not provided, it may take some time to infer it from the tar file");
  }

  // initializing all the readers
  wds_shards_.reserve(paths_.size());
  for (auto& uri : paths_) {
    wds_shards_.emplace_back(FileStream::Open(uri, read_ahead_, !copy_read_data_));
  }

  // preparing the map from extensions to outputs
  std::unordered_map<std::string, std::vector<size_t>> ext_map;
  for (size_t output_index = 0; output_index < ext_.size(); output_index++) {
    for (auto& ext : ext_[output_index]) {
      ext_map[ext].push_back(output_index);
    }
  }

  // collecting and filtering the index files
  std::vector<detail::wds::SampleDesc> unfiltered_samples;
  std::vector<detail::wds::ComponentDesc> unfiltered_components;
  bitmask was_output_set;
  was_output_set.resize(ext_.size(), false);
  output_indicies_.reserve(ext_.size());

  std::vector<size_t> dtype_sizes_(dtypes_.size());
  for (size_t i = 0; i < dtypes_.size(); i++) {
    dtype_sizes_[i] = TypeTable::GetTypeInfo(dtypes_[i]).size();
  }

  for (size_t wds_shard_index = 0; wds_shard_index < paths_.size(); wds_shard_index++) {
    unfiltered_samples.resize(0);
    unfiltered_components.resize(0);
    if (generate_index_) {
      detail::wds::ParseTarFile(unfiltered_samples, unfiltered_components,
                                wds_shards_[wds_shard_index]);
    } else {
      detail::wds::ParseIndexFile(unfiltered_samples, unfiltered_components,
                                  index_paths_[wds_shard_index]);
    }

    for (auto& sample : unfiltered_samples) {
      detail::wds::SampleDesc new_sample{
          detail::wds::VectorRange<detail::wds::ComponentDesc>(components_, components_.size()),
          detail::wds::VectorRange<size_t>(empty_outputs_, empty_outputs_.size()), wds_shard_index,
          sample.line_number};

      size_t start_outputs_index = output_indicies_.size();

      for (auto& component : sample.components) {
        component.outputs =
            detail::wds::VectorRange<size_t>(output_indicies_, output_indicies_.size());
        auto ext = component.ext;
        if (!case_sensitive_extensions_) {
          ext = str_tolower(ext);
        }
        for (auto& output : ext_map[ext]) {
          if (!was_output_set[output]) {
            DALI_ENFORCE(component.size % dtype_sizes_[output] == 0,
                         make_string("Error in index file at ", GetSampleSource(new_sample),
                                     " - component size and dtype incompatible"));
            output_indicies_.push_back(output);
            component.outputs.num++;
            was_output_set[output] = true;
          } else {
            std::call_once(multiple_files_single_component, [&]() {
              DALI_WARN(make_string("Multiple components matching output ", output, " at ",
                                    GetSampleSource(new_sample), "."));
            });
          }
        }
        if (component.outputs.num) {
          components_.push_back(std::move(component));
          new_sample.components.num++;
        }
      }

      if (new_sample.components.num < ext_.size()) {
        switch (missing_component_behavior_) {
          case detail::wds::MissingExtBehavior::Empty:
            for (size_t output = 0; output < ext_.size(); output++) {
              if (!was_output_set[output]) {
                empty_outputs_.push_back(output);
                new_sample.empty_outputs.num++;
              }
            }
            samples_.push_back(new_sample);
            break;
          case detail::wds::MissingExtBehavior::Skip:
            components_.resize(new_sample.components.start);
            output_indicies_.resize(start_outputs_index);
            break;
          case detail::wds::MissingExtBehavior::Raise:
            DALI_FAIL(make_string("Underful sample detected at ", GetSampleSource(new_sample)));
            break;
          default:
            break;
        }
      } else {
        samples_.push_back(new_sample);
      }
      was_output_set.fill(false);
    }
  }
  sample_index_ = start_index(shard_id_, num_shards_, samples_.size());
}

void WebdatasetLoader::Reset(bool wrap_to_shard) {
  sample_index_ = wrap_to_shard ? start_index(shard_id_, num_shards_, samples_.size()) : 0;
}

}  // namespace dali
