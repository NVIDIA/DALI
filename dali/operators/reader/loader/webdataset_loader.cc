#include "dali/operators/reader/loader/webdataset_loader.h"
#include <boost/algorithm/string.hpp>
#include "dali/util/file.h"

namespace dali {

inline WebdatasetLoader::MissingExt WebdatasetLoader::Str2MissingExt(
    std::string missing_component_behavior) {
  std::remove(missing_component_behavior.begin(), missing_component_behavior.end(), '_');
  std::transform(missing_component_behavior.begin(), missing_component_behavior.end(),
                 missing_component_behavior.begin(), static_cast<int(*)(int)>(std::tolower));
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
          Str2MissingExt(spec.GetArgument<std::string>("missing_component_behavior"))),
      dtype_(spec.GetArgument<DALIDataType>("dtype")) {
  DALI_ENFORCE(uris_.size() == configs_.size(),
               "Number of uris does not match the number of config files");
  DALI_ENFORCE(uris_.size() > 0, "No webdataset shards provided");
  DALI_ENFORCE(missing_component_behavior_ != MissingExt::Invalid,
               "Invalid value of missing_component_behavior");
  DALI_ENFORCE(IsIntegral(dtype_) || IsFloatingPoint(dtype_),
               "Only numeric output types are supported");

  std::vector<std::string> samples_exts = spec.GetRepeatedArgument<std::string>("ext");

  for (size_t exts_idx = 0; exts_idx < samples_exts.size(); exts_idx++) {
    std::stringstream exts_stream(samples_exts[exts_idx]);
    std::string ext;
    while (std::getline(exts_stream, ext, ';')) {
      if (ext_map_[ext].empty() || ext_map_[ext].back() != exts_idx) {
        ext_map_[ext].push_back(exts_idx);
      }
    }
  }

  ext_ = std::unordered_set<std::string>(samples_exts.begin(), samples_exts.end());
}

WebdatasetLoader::~WebdatasetLoader() {}

void WebdatasetLoader::PrepareEmpty(vector<Tensor<CPUBackend>>& empty) {
  empty = std::vector<Tensor<CPUBackend>>(ext_.size(), Tensor<CPUBackend>());
}

inline std::string GetExtension(std::string filepath) {
  const size_t base_name_pos = filepath.find_last_of('/') + 1;
  const size_t dot_pos = filepath.find_first_of('.', base_name_pos);
  return filepath.substr(dot_pos + 1);
}

inline Index WebdatasetLoader::GetCurrentSampleIndex() {
  return wds_shards_prefixsums_[current_wds_shard_index_] + current_sample_index_;
}

void WebdatasetLoader::ReadSample(
    vector<Tensor<CPUBackend>>& sample) {  // TODO(barci2): Implement sample reading
  MoveToNextShard(GetCurrentSampleIndex());
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
  while(std::getline(extensions_stream, extension)) {
    out.extensions.insert(extension);
  }
  DALI_ENFORCE(out.extensions.size() > 0_uz,
               "Malformed index file at '" + config_path "'");  // config validity check
  return out;
}

inline std::vector<WebdatasetLoader::SampleConfig> WebdatasetLoader::ParseConfig(
    std::string config_path) {
  std::ifstream config(config_path);
  size_t config_size;
  size_t sample_config_num = 0;  // for config validity check
  config >> config_size >> sample_config_num;
  DALI_ENFORCE(sample_config_num > 0_uz,
               "Empty index file at '" + config_path + "'");  // config validity check

  std::vector<SampleConfig> out;
  out.reserve(sample_config_num);

  for (size_t sample_index = 0; sample_index < sample_config_num; sample_index++) {
    SampleConfig new_sample = ParseSampleConfig(config, config_path);
    if (!out.empty()) {
      out.back().end_offset = std::min(out.back().end_offset, new_sample.start_offset);
    }

    // filtering out samples without the necessary extensions
    if (!std::includes(ext_.begin(), ext_.end(), new_sample.extensions.begin(),
                       new_sample.extensions.end())) {
      switch (missing_component_behavior_) {
        case MissingExt::Empty:
          break;
        case MissingExt::Skip:
          continue;
        case MissingExt::Raise:
          DALI_ERROR("Underfull sample detected in index file at '" + config_path + "'");
      };
    }
    out.push_back(std::move(new_sample));
  }
  if (out.size()) {
    out.back().end_offset = std::min(out.back().end_offset, config_size);
  }
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