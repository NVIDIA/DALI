#include "dali/operators/reader/loader/webdataset_loader.h"
#include <cstddef>
#include <cstring>
#include "dali/core/error_handling.h"
#include "dali/pipeline/data/types.h"
#include "dali/util/file.h"

namespace dali {
namespace detail {
namespace wds {

inline MissingExtBehavior Str2MissingExt(std::string missing_component_behavior) {
  std::remove(missing_component_behavior.begin(), missing_component_behavior.end(), '_');
  std::transform(missing_component_behavior.begin(), missing_component_behavior.end(),
                 missing_component_behavior.begin(), static_cast<int (*)(int)>(std::tolower));
  if (missing_component_behavior == "") {
    return MissingExtBehavior::Empty;
  } else if (missing_component_behavior == "skip") {
    return MissingExtBehavior::Skip;
  } else if (missing_component_behavior == "fillempty") {
    return MissingExtBehavior::Empty;
  } else if (missing_component_behavior == "raise") {
    return MissingExtBehavior::Raise;
  } else if (missing_component_behavior == "raiseerror") {
    return MissingExtBehavior::Raise;
  } else {
    return MissingExtBehavior::Invalid;
  }
}

inline SampleConfig ParseSampleConfig(std::ifstream& config, const std::string& config_path) {
  SampleConfig out;
  config >> out.start_offset;
  out.end_offset = std::numeric_limits<decltype(out.end_offset)>::max();

  std::string component_metadata;
  std::getline(config, component_metadata);
  component_metadata += ' ';

  std::stringstream extensions_stream(component_metadata);
  std::string extension;
  int64_t size;

  while (extensions_stream >> extension >> size) {
    DALI_ENFORCE(size >= 0 && !extension.empty(),
                 "Malformed index file at " + config_path);  // config validity check
    out.config_metadata.push_back(ComponentConfig(std::move(extension), size));
    size = 0;
  }
  DALI_ENFORCE(out.config_metadata.size() > 0_uz,
               "Malformed index file at " + config_path);  // config validity check
  return out;
}

inline std::vector<SampleConfig> ParseConfig(MissingExtBehavior missing_component_behavior,
                                             const std::string& config_path,
                                             std::vector<std::set<std::string>>& ext) {
  std::ifstream config(config_path);
  int64_t config_size;
  size_t sample_config_num = 0;  // for config validity check
  config >> config_size >> sample_config_num;
  DALI_ENFORCE(sample_config_num > 0_uz,
               "Empty index file at " + config_path);  // config validity check

  std::vector<SampleConfig> out;
  out.reserve(sample_config_num);

  for (size_t sample_index = 0; sample_index < sample_config_num; sample_index++) {
    SampleConfig new_sample = ParseSampleConfig(config, config_path);
    if (!out.empty()) {
      out.back().end_offset = std::min(out.back().end_offset, new_sample.start_offset);
    }

    // filtering out samples without the required extensions
    if (std::all_of(ext.begin(), ext.end(), [&](std::set<std::string> extension_set) {
          for (auto& component_config : new_sample.config_metadata) {
            if (extension_set.count(component_config.ext)) {
              return true;
            }
          }
          return false;
        })) {
      switch (missing_component_behavior) {
        case MissingExtBehavior::Skip:
          continue;
        case MissingExtBehavior::Raise:
          DALI_ERROR("Underfull sample detected in index file at " + config_path);
        default:
          break;
      };
    }

    out.push_back(std::move(new_sample));
  }
  if (out.size()) {
    out.back().end_offset = std::min(out.back().end_offset, config_size);
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
      configs_(spec.GetRepeatedArgument<std::string>("configs")),
      missing_component_behavior_(detail::wds::Str2MissingExt(
          spec.GetArgument<std::string>("missing_component_behavior"))) {
  DALI_ENFORCE(uris_.size() == configs_.size(),
               "Number of uris does not match the number of config files");
  DALI_ENFORCE(uris_.size() > 0, "No webdataset shards provided");
  DALI_ENFORCE(missing_component_behavior_ != detail::wds::MissingExtBehavior::Invalid,
               "Invalid value for missing_component_behavior");

  for (auto& dtype : dtypes_) {
    DALI_ENFORCE(detail::wds::kSupportedTypes.count(dtype),
                 "Unsupported output dtype. Supported types include: " + SupportedTypesListGen());
  }

  std::vector<std::string> samples_exts = spec.GetRepeatedArgument<std::string>("ext");
  ext_.reserve(samples_exts.size());

  // splitting extension bundles by the delimiter
  for (size_t exts_idx = 0; exts_idx < samples_exts.size(); exts_idx++) {
    std::stringstream exts_stream(samples_exts[exts_idx]);
    std::string ext;
    ext_.emplace_back();
    while (std::getline(exts_stream, ext, detail::wds::kExtDelim)) {
      ext_.back().insert(ext);
      if (ext_map_[ext].empty() || ext_map_[ext].back() != exts_idx) {
        ext_map_[ext].push_back(exts_idx);
      }
    }
  }

  dtypes_ = spec.HasArgument("dtypes") ? spec.GetRepeatedArgument<DALIDataType>("dtypes") :
                                         std::vector<DALIDataType>(ext_.size(), DALI_UINT8);
  DALI_ENFORCE(ext_.size() == dtypes_.size(),
               "Number of extensions does not match the number of types");
}

WebdatasetLoader::~WebdatasetLoader() {}

void WebdatasetLoader::PrepareEmpty(vector<Tensor<CPUBackend>>& empty) {
  empty = std::vector<Tensor<CPUBackend>>(ext_.size());
  for (auto& tensor : empty) {
    tensor.Resize({tensor_init_bytes_});
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
                                             std::vector<char>& sample_was_set,
                                             const std::string& extension,
                                             const std::string& source_info,
                                             std::shared_ptr<void> data, int64_t size) const {
  DALIMeta meta;
  meta.SetSourceInfo(source_info);
  meta.SetSkipSample(false);

  for (size_t component_index : ext_map_.at(extension)) {
    if (!sample_was_set[component_index]) {
      auto dtype_info = TypeTable::GetTypeInfo(dtypes_[component_index]);
      sample[component_index].SetMeta(meta);
      sample[component_index].ShareData(
          data, size, {size / static_cast<int64_t>(dtype_info.size())}, dtype_info);
      sample_was_set[component_index] = true;
    }
  }
}


inline uint8_t* WebdatasetLoader::ShareDataPointer(std::vector<Tensor<CPUBackend>>& sample,
                                                   std::vector<char>& sample_was_set,
                                                   const std::string& extension,
                                                   const std::string& source_info,
                                                   int64_t size) const {
  DALIMeta meta;
  meta.SetSourceInfo(source_info);
  meta.SetSkipSample(false);

  uint8_t* shared_tensor_data = nullptr;
  for (size_t component_index : ext_map_.at(extension)) {
    if (!sample_was_set[component_index]) {
      sample[component_index].SetMeta(meta);
      auto dtype_info = TypeTable::GetTypeInfo(dtypes_[component_index]);
      if (shared_tensor_data == nullptr) {
        if (sample[component_index].shares_data()) {
          sample[component_index].Reset();
        }
        sample[component_index].reserve(size);
        sample[component_index].Resize(size / dtype_info.size(), dtype_info);
        shared_tensor_data = reinterpret_cast<uint8_t*>(sample[component_index].raw_mutable_data());
      } else {
        DALI_ENFORCE(size % static_cast<int64_t>(dtype_info.size()) == 0,
                     "Index file at " + configs_[current_wds_shard_index_] +
                         " reporting component sizes different to actual")
        sample[component_index].ShareData(
            shared_tensor_data, size, {size / static_cast<int64_t>(dtype_info.size())}, dtype_info);
      }
      sample_was_set[component_index] = true;
    }
  }
  return shared_tensor_data;
}

inline void WebdatasetLoader::MarkCached(std::vector<Tensor<CPUBackend>>& sample,
                                         std::vector<char>& sample_was_set,
                                         const std::string& extension,
                                         const std::string& source_info) const {
  DALIMeta meta;
  meta.SetSourceInfo(source_info);
  meta.SetSkipSample(true);

  for (size_t component_index : ext_map_.at(extension)) {
    if (!sample_was_set[component_index]) {
      sample[component_index].Reset();
      sample[component_index].SetMeta(meta);
      sample[component_index].set_type(TypeTable::GetTypeInfo(dtypes_[component_index]));
      sample[component_index].Resize({0});
      sample_was_set[component_index] = true;
    }
  }
}

void WebdatasetLoader::ReadSample(vector<Tensor<CPUBackend>>& sample) {
  MoveToNextShard(GetCurrentSampleIndex());
  auto& current_wds_shard = wds_shards_[current_wds_shard_index_];
  auto& current_sample = wds_shards_metadata_[current_wds_shard_index_][current_sample_index_];
  current_wds_shard.SeekArchive(current_sample.start_offset);

  std::cerr << "Started reading sample with the following config: start_offset = "
            << current_sample.start_offset << " end_offset = " << current_sample.end_offset
            << " current_wds_shard position = " << current_wds_shard.TellArchive()
            << " current_wds_shard EndOfArchive " << current_wds_shard.EndOfArchive() << std::endl;

  vector<char> sample_was_set(sample.size(), false);
  while (current_wds_shard.TellArchive() < current_sample.end_offset) {
    DALI_ENFORCE(!current_wds_shard.EndOfArchive(),
                 "Index file at " + configs_[current_wds_shard_index_] +
                     " reporting a file longer than actual (archive reached an offset " +
                     std::to_string(current_wds_shard.TellArchive()) +
                     " and the sample is supposed to end at " +
                     std::to_string(current_sample.end_offset) ")");
    std::cerr << "Reading a file with a name " << current_wds_shard.GetFileName() << std::endl;
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
      MarkCached(sample, sample_was_set, extension, source_info);
      continue;
    }

    // Reading the data into the tensors
    int64_t size = static_cast<int64_t>(current_wds_shard.GetFileSize());

    if (!copy_read_data_) {
      auto p = current_wds_shard.ReadFile();
      DALI_ENFORCE(p != nullptr, "Error reading from a file " + uris_[current_wds_shard_index_]);
      SetDataPointer(sample, sample_was_set, extension, source_info, p, size);
    } else {
      uint8_t* shared_tensor_data =
          ShareDataPointer(sample, sample_was_set, extension, source_info, size);
      uint64_t n_read = current_wds_shard.Read(static_cast<uint8_t*>(shared_tensor_data), size);
      DALI_ENFORCE(static_cast<int64_t>(n_read) == size,
                   "Error reading from a file " + uris_[current_wds_shard_index_]);
    }

    current_wds_shard.NextFile();
  }

  // setting empty components:
  for (size_t component_index = 0; component_index < sample.size(); component_index++) {
    if (!sample_was_set[component_index]) {
      sample[component_index].Reset();
      sample[component_index].set_type(TypeTable::GetTypeInfo(dtypes_[component_index]));
      sample[component_index].Resize({0});
    }
  }

  current_sample_index_++;
  while (current_wds_shard_index_ < wds_shards_metadata_.size() &&
         current_sample_index_ >= wds_shards_metadata_[current_wds_shard_index_].size()) {
    current_wds_shard_index_++;
    current_sample_index_ = 0;
  }

  std::cerr << "ReadSample Tensor types: ";
  for (auto& t : sample) {
    std::cerr << t.type().id() << ' ';
  }
  std::cerr << std::endl;
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
  wds_shards_metadata_.reserve(configs_.size());
  wds_shards_.reserve(uris_.size());
  wds_shards_prefixsums_.reserve(configs_.size());

  // collecting the config files
  for (size_t config_index = 0; config_index < configs_.size(); config_index++) {
    wds_shards_metadata_.emplace_back(
        ParseConfig(missing_component_behavior_, configs_[config_index], ext_));
    wds_shards_prefixsums_.push_back(total_size_);
    total_size_ += wds_shards_metadata_.back().size();

    // checking dtype compatibility
    for (auto& sample_config : wds_shards_metadata_.back()) {
      std::vector<char> was_sample_assigned(ext_.size(), 0);
      for (auto& component_config : sample_config.config_metadata) {
        if (!ext_map_.count(component_config.ext)) {
          continue;
        }
        for (auto& sample_index : ext_map_[component_config.ext]) {
          if (!was_sample_assigned[sample_index]) {
            DALI_ENFORCE(
                component_config.size % TypeTable::GetTypeInfo(dtypes_[sample_index]).size() == 0,
                "Component of a tar file '" + uris_[config_index] + "' at offset " +
                    std::to_string(sample_config.start_offset) +
                    " has a size not divisible by the chosen dtype's size of " +
                    std::to_string(TypeTable::GetTypeInfo(dtypes_[sample_index]).size()) +
                    " bytes");
            was_sample_assigned[sample_index] = true;
          }
        }
      }
    }
  }
  wds_shards_prefixsums_.push_back(total_size_);  // for the last shard when reaches the end

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
  if (stick_to_shard_) {
    current_wds_shard_index_ = first_wds_shard_index_;
    current_sample_index_ = first_sample_index_;
  }
}

void WebdatasetLoader::Reset(bool wrap_to_shard) {
  current_wds_shard_index_ = wrap_to_shard ? first_wds_shard_index_ : 0;
  current_sample_index_ = wrap_to_shard ? first_sample_index_ : 0;
}

}  // namespace dali