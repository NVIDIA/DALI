#include "dali/operators/reader/loader/webdataset_loader.h"
#include <boost/algorithm/string.hpp>
#include <fstream>
#include "dali/util/file.h"

namespace dali {

WebdatasetLoader::WebdatasetLoader(const OpSpec& spec)
    : uris_(spec.GetRepeatedArgument<std::string>("uris")),
      configs_(spec.GetRepeatedArgument<std::string>("configs")),
      fail_on_missing_component_(spec.GetArgument<bool>("fail_on_missing_component")),
      dtype_(spec.GetArgument<DALIDataType>("dtype")),
      total_size_(0),
      wds_shards_{},
      first_wds_shard_index_(0),
      first_sample_offset_(0),
      current_wds_shard_index_(0) {
  DALI_ENFORCE(uris_.size() == configs_.size(),
               "Number of uris does not match the number of config files");
  DALI_ENFORCE(uris_.size() > 0, "No webdataset shards provided");
  DALI_ENFORCE(IsIntegral(dtype_) || IsFloatingPoint(dtype_),
               "Only numeric output types are supported");
  boost::split(ext_, spec.GetArgument<std::string>("ext"), kExtDelim);
}

WebdatasetLoader::~WebdatasetLoader() {}

void WebdatasetLoader::PrepareEmpty(vector<Tensor<CPUBackend>>& empty) {
  empty = std::vector<Tensor<CPUBackend>>(ext_.size(), Tensor<CPUBackend>());
}

inline std::tuple<std::string, std::string> SplitName(std::string filepath) {
  const size_t base_name_pos = filepath.find_last_of('/') + 1;
  const size_t dot_pos = filepath.find_last_of('.');
  if (dot_pos <= base_name_pos) {
    return std::make_tuple(filepath.substr(base_name_pos), "");
  }
  return std::make_tuple(filepath.substr(0, dot_pos), filepath.substr(dot_pos + 1));
}


void WebdatasetLoader::ReadSample(vector<Tensor<CPUBackend>>& sample) {
  detail::TarArchive& current_wds_shard = wds_shards_[current_wds_shard_index_];

  std::string sample_name;
  std::tie(sample_name, std::ignore) = SplitName(current_wds_shard.GetFileName());
}

Index WebdatasetLoader::SizeImpl() {
  return total_size_;
}

void WebdatasetLoader::PrepareMetadataImpl() {
  // collecting wds shard sizes from the config files
  vector<size_t> shard_sizes;
  for (auto& config_path : configs_) {
    std::ifstream config(config_path);
    size_t config_size;
    config >> config_size;
    shard_sizes.push_back(config_size);
    total_size_ += config_size;
  }

  // seeking the first wds shard to use
  const size_t start_offset =
      stick_to_shard_ ? start_index(shard_id_, num_shards_, total_size_) : 0;

  size_t current_offset = 0;
  for (; first_wds_shard_index_ < shard_sizes.size() &&
         current_offset + shard_sizes[first_wds_shard_index_] < start_offset;
       current_offset += shard_sizes[first_wds_shard_index_++]) {}

  current_wds_shard_index_ = first_wds_shard_index_;

  // collecting the offset to use for the first sample in the first wds shard
  std::ifstream start_config(configs_[first_wds_shard_index_]);
  for (size_t i = 0; i <= start_offset - current_offset; i++) {
    start_config.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  }
  start_config >> first_sample_offset_;

  // initializing all the readers
  for (auto& uri : uris_) {
    wds_shards_.emplace_back(FileStream::Open(uri, read_ahead_, !dont_use_mmap_));
  }

  // initializing the first reader
  if (stick_to_shard_) {
    wds_shards_[first_wds_shard_index_].Seek(first_sample_offset_);
  }
}

void WebdatasetLoader::Reset(bool wrap_to_shard) {
  for (size_t wds_shard_index = wrap_to_shard ? first_wds_shard_index_ : 0;
       wds_shard_index <= current_wds_shard_index_; wds_shard_index++) {
    wds_shards_[wds_shard_index].Seek(0);
  }
  if (wrap_to_shard) {
    current_wds_shard_index_ = first_wds_shard_index_;
    wds_shards_[first_wds_shard_index_].Seek(first_sample_offset_);
  } else {
    current_wds_shard_index_ = 0;
  }
}

}  // namespace dali