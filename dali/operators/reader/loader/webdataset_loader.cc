#include "dali/operators/reader/loader/webdataset_loader.h"
#include <cstddef>
#include <cstring>
#include "dali/util/file.h"

namespace dali {

const std::set<DALIDataType> WebdatasetLoader::kSupportedTypes = {
    DALI_UINT8, DALI_UINT16, DALI_UINT32,  DALI_UINT64, DALI_INT8,   DALI_INT16,
    DALI_INT32, DALI_INT64,  DALI_FLOAT16, DALI_FLOAT,  DALI_FLOAT64};

inline std::string SupportedTypesListGen() {
  std::stringstream out;
  for (auto& component_dtype : WebdatasetLoader::kSupportedTypes) {
    out << component_dtype << ',';
  }
  if (!WebdatasetLoader::kSupportedTypes.empty()) {
    out.unget();
  }
  return out.str();
}

inline WebdatasetLoader::MissingExt WebdatasetLoader::Str2MissingExt(
    std::string missing_component_behavior) {
  std::remove(missing_component_behavior.begin(), missing_component_behavior.end(), '_');
  std::transform(missing_component_behavior.begin(), missing_component_behavior.end(),
                 missing_component_behavior.begin(), static_cast<int (*)(int)>(std::tolower));
  if (missing_component_behavior == "") {
    return MissingExt::Empty;
  } else if (missing_component_behavior == "skip") {
    return MissingExt::Skip;
  } else if (missing_component_behavior == "fillempty") {
    return MissingExt::Empty;
  } else if (missing_component_behavior == "raise") {
    return MissingExt::Raise;
  } else if (missing_component_behavior == "raiseerror") {
    return MissingExt::Raise;
  } else {
    return MissingExt::Invalid;
  }
}

WebdatasetLoader::WebdatasetLoader(const OpSpec& spec)
    : Loader(spec),
      uris_(spec.GetRepeatedArgument<std::string>("uris")),
      configs_(spec.GetRepeatedArgument<std::string>("configs")),
      missing_component_behavior_(
          Str2MissingExt(spec.GetArgument<std::string>("missing_component_behavior"))) {
  dtype_ = spec.HasArgument("dtype") ? spec.GetRepeatedArgument<DALIDataType>("dtype") :
                                       std::vector<DALIDataType>(uris_.size(), DALI_UINT8);

  DALI_ENFORCE(uris_.size() == configs_.size(),
               "Number of uris does not match the number of config files");
  DALI_ENFORCE(uris_.size() == dtype_.size(), "Number of uris does not match the number of types");
  DALI_ENFORCE(uris_.size() > 0, "No webdataset shards provided");
  DALI_ENFORCE(missing_component_behavior_ != MissingExt::Invalid,
               "Invalid value for missing_component_behavior");

  for (auto& component_dtype : dtype_) {
    DALI_ENFORCE(kSupportedTypes.find(component_dtype) != kSupportedTypes.end(),
                 "Unsupported output dtype. Supported types include: " + SupportedTypesListGen());
  }

  std::vector<std::string> samples_exts = spec.GetRepeatedArgument<std::string>("ext");
  ext_.reserve(samples_exts.size());

  // splitting extension bundles by the delimiter
  for (size_t exts_idx = 0; exts_idx < samples_exts.size(); exts_idx++) {
    std::stringstream exts_stream(samples_exts[exts_idx]);
    std::string ext;
    ext_.emplace_back();
    while (std::getline(exts_stream, ext, kExtDelim)) {
      ext_.back().insert(ext);
      if (ext_map_[ext].empty() || ext_map_[ext].back() != exts_idx) {
        ext_map_[ext].push_back(exts_idx);
      }
    }
  }
}

WebdatasetLoader::~WebdatasetLoader() {}

void WebdatasetLoader::PrepareEmpty(vector<Tensor<CPUBackend>>& empty) {
  empty = std::vector<Tensor<CPUBackend>>(ext_.size());
  for (auto& tensor : empty) {
    tensor.Resize({tensor_init_bytes_});
  }
}

inline std::string GetExtension(std::string filepath) {
  const size_t base_name_pos = filepath.find_last_of('/') + 1;
  const size_t dot_pos = filepath.find_first_of('.', base_name_pos);
  return filepath.substr(dot_pos + 1);
}

inline Index WebdatasetLoader::GetCurrentSampleIndex() const {
  return wds_shards_prefixsums_[current_wds_shard_index_] + current_sample_index_;
}

inline size_t WebdatasetLoader::MaxCommonDtypeSize(const std::string& extension) const {
  size_t dtype_max_size = 1;
  for (auto& component_index : ext_map_.at(extension)) {
    dtype_max_size =
        std::max(dtype_max_size, TypeTable::GetTypeInfo(dtype_[component_index]).size());
  }
  return dtype_max_size;
}

inline void WebdatasetLoader::SetDataPointer(std::vector<Tensor<CPUBackend>>& sample,
                                             std::vector<bool>& sample_was_set,
                                             const std::string& extension,
                                             const std::string& source_info,
                                             std::shared_ptr<void> data, int64_t size) const {
  DALIMeta meta;
  meta.SetSourceInfo(source_info);
  meta.SetSkipSample(false);

  for (size_t component_index : ext_map_.at(extension)) {
    if (!sample_was_set[component_index]) {
      auto component_dtype_info = TypeTable::GetTypeInfo(dtype_[component_index]);
      sample[component_index].SetMeta(meta);
      sample[component_index].ShareData(data, size,
                                        {size / static_cast<int64_t>(component_dtype_info.size())},
                                        component_dtype_info);
      sample_was_set[component_index] = true;
    }
  }
}


inline uint8_t* WebdatasetLoader::ShareDataPointer(std::vector<Tensor<CPUBackend>>& sample,
                                                   std::vector<bool>& sample_was_set,
                                                   const std::string& extension,
                                                   const std::string& source_info, int64_t size,
                                                   size_t dtype_max_size) const {
  DALIMeta meta;
  meta.SetSourceInfo(source_info);
  meta.SetSkipSample(false);

  uint8_t* shared_tensor_data = nullptr;
  for (size_t component_index : ext_map_.at(extension)) {
    if (!sample_was_set[component_index]) {
      sample[component_index].SetMeta(meta);
      auto component_dtype_info = TypeTable::GetTypeInfo(dtype_[component_index]);
      if (shared_tensor_data == nullptr) {
        sample[component_index].reserve(align_up(size, dtype_max_size));
        sample[component_index].Resize(size / component_dtype_info.size(), component_dtype_info);
        shared_tensor_data = reinterpret_cast<uint8_t*>(sample[component_index].raw_mutable_data());
        std::memset(shared_tensor_data + size, 0, align_up(size, dtype_max_size) - size);
      } else {
        sample[component_index].ShareData(
            shared_tensor_data, size, {size / static_cast<int64_t>(component_dtype_info.size())},
            component_dtype_info);
      }
      sample_was_set[component_index] = true;
    }
  }
  return shared_tensor_data;
}

inline void WebdatasetLoader::MarkCached(std::vector<Tensor<CPUBackend>>& sample,
                                         std::vector<bool>& sample_was_set,
                                         const std::string& extension,
                                         const std::string& source_info) const {
  DALIMeta meta;
  meta.SetSourceInfo(source_info);
  meta.SetSkipSample(true);

  for (size_t component_index : ext_map_.at(extension)) {
    if (!sample_was_set[component_index]) {
      sample[component_index].Reset();
      sample[component_index].SetMeta(meta);
      sample[component_index].set_type(TypeTable::GetTypeInfo(dtype_[component_index]));
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

  vector<bool> sample_was_set(sample.size(), false);
  while (current_wds_shard.TellArchive() < current_sample.end_offset) {
    // Check in case of encountering a tar entry that is not a file
    if (current_wds_shard.GetFileType() != detail::TarArchive::ENTRY_FILE) {
      DALI_ENFORCE(current_wds_shard.NextFile(), "Index file reporting a file longer than actual");
      continue;
    }

    // Check in case of encountering an unneeded entry
    const std::string extension = GetExtension(current_wds_shard.GetFileName());
    if (ext_map_.find(extension) == ext_map_.end()) {
      DALI_ENFORCE(current_wds_shard.NextFile(), "Index file reporting a file longer than actual");
      continue;
    }

    // Check in case skipping sample is necessary
    const std::string source_info = uris_[current_wds_shard_index_] + " at offset " +
                                    to_string(current_wds_shard.TellArchive());
    if (ShouldSkipImage(source_info)) {
      MarkCached(sample, sample_was_set, extension, source_info);
      DALI_ENFORCE(current_wds_shard.NextFile(), "Index file reporting a file longer than actual");
      continue;
    }

    // Reading the data into the tensors
    int64_t size = static_cast<int64_t>(current_wds_shard.GetFileSize());
    size_t dtype_max_size = MaxCommonDtypeSize(extension);
    if (!copy_read_data_ && size % dtype_max_size == 0) {
      auto p = current_wds_shard.ReadFile();
      DALI_ENFORCE(p != nullptr, "Error reading from a file " + uris_[current_wds_shard_index_]);
      SetDataPointer(sample, sample_was_set, extension, source_info, p, size);
    } else {
      uint8_t* shared_tensor_data =
          ShareDataPointer(sample, sample_was_set, extension, source_info, size, dtype_max_size);
      uint64_t n_read = current_wds_shard.Read(static_cast<uint8_t*>(shared_tensor_data), size);
      DALI_ENFORCE(static_cast<int64_t>(n_read) == size,
                   "Error reading from a file " + uris_[current_wds_shard_index_]);
    }

    DALI_ENFORCE(current_wds_shard.NextFile(), "Index file reporting a file longer than actual");
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

inline WebdatasetLoader::SampleConfig ParseSampleConfig(std::ifstream& config,
                                                        std::string config_path) {
  WebdatasetLoader::SampleConfig out;
  config >> out.start_offset;
  out.end_offset = std::numeric_limits<decltype(out.end_offset)>::infinity();

  std::string extensions;
  std::getline(config, extensions);

  std::stringstream extensions_stream(extensions);
  std::string extension;
  while (std::getline(extensions_stream, extension)) {
    out.extensions.insert(extension);
  }
  DALI_ENFORCE(out.extensions.size() > 0_uz,
               "Malformed index file at " + config_path);  // config validity check
  return out;
}

template <typename T>
inline bool HasIntersection(std::set<T> set_a, std::set<T> set_b) {
  if (set_b.size() < set_a.size()) {
    return HasIntersection(set_b, set_a);
  }
  for (auto& item : set_a) {
    if (set_b.find(item) != set_b.end()) {
      return true;
    }
  }
  return false;
}

inline std::vector<WebdatasetLoader::SampleConfig> WebdatasetLoader::ParseConfig(
    std::string config_path) {
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
    if (std::all_of(ext_.begin(), ext_.end(), [&](std::set<std::string> extension_set) {
          return HasIntersection(extension_set, new_sample.extensions);
        })) {
      switch (missing_component_behavior_) {
        case MissingExt::Skip:
          continue;
        case MissingExt::Raise:
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
  for (auto& config_path : configs_) {
    wds_shards_metadata_.emplace_back(ParseConfig(config_path));
    wds_shards_prefixsums_.push_back(total_size_);
    total_size_ += wds_shards_metadata_.back().size();
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
         first_sample_index_) {
    first_wds_shard_index_++;
  }
  first_sample_index_ = first_sample_index_ - wds_shards_prefixsums_[first_wds_shard_index_];

  if (stick_to_shard_) {
    current_wds_shard_index_ = first_wds_shard_index_;
    current_sample_index_ = first_sample_index_;
  }

  // initializing the first reader
  if (stick_to_shard_) {
    wds_shards_[first_wds_shard_index_].SeekArchive(
        wds_shards_metadata_[first_wds_shard_index_][first_sample_index_].start_offset);
  }
}

void WebdatasetLoader::Reset(bool wrap_to_shard) {
  current_wds_shard_index_ = wrap_to_shard ? first_wds_shard_index_ : 0;
  current_sample_index_ = wrap_to_shard ? first_sample_index_ : 0;
  for (detail::TarArchive& wds_shard : wds_shards_) {
    wds_shard.SeekArchive(0);
  }
  if (wrap_to_shard) {
    wds_shards_[first_wds_shard_index_].SeekArchive(
        wds_shards_metadata_[first_wds_shard_index_][first_sample_index_].start_offset);
  }
}

}  // namespace dali