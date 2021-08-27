#ifndef DALI_OPERATORS_READER_LOADER_WEBDATASET_LOADER_H_
#define DALI_OPERATORS_READER_LOADER_WEBDATASET_LOADER_H_

#include <fstream>
#include <string>
#include <unordered_set>
#include <vector>
#include "dali/operators/reader/loader/loader.h"
#include "dali/operators/reader/loader/webdataset/tar_utils.h"
#include "dali/pipeline/data/tensor.h"

namespace dali {

// interesting arguments: uris, ext, dont_use_mmap, component_mode, dtype, read_ahead,
// tensor_init_bytes, num_shards, shard_id

class WebdatasetLoader : public Loader<CPUBackend, vector<Tensor<CPUBackend>>> {
 public:
  static const char kExtDelim = ';';
  explicit WebdatasetLoader(const OpSpec& spec);
  ~WebdatasetLoader() override;

  void PrepareEmpty(vector<Tensor<CPUBackend>>&) override;
  void ReadSample(vector<Tensor<CPUBackend>>&) override;

 protected:
  Index SizeImpl() override;
  void PrepareMetadataImpl() override;
  void Reset(bool wrap_to_shard) override;

  enum class MissingExt {
    FillEmpty,
    Skip,
    RaiseError,
    Invalid
  };
  static MissingExt Str2MissingExt(std::string);

  std::vector<std::string> uris_;
  std::vector<std::string> configs_;
  std::unordered_set<std::string> ext_;
  MissingExt missing_component_behavior_;
  DALIDataType dtype_;

 private:
  struct SampleConfig {
    size_t start_offset;
    size_t end_offset;
    std::unordered_set<std::string> extensions;
  };

  std::vector<std::vector<SampleConfig>> wds_shards_metadata_;  // data from the config files
  friend SampleConfig ParseSampleConfig(std::ifstream&, std::string);
  std::vector<SampleConfig> ParseConfig(std::string);
  

  size_t total_size_ = 0;                                       // total size of all config files
  std::vector<detail::TarArchive> wds_shards_;                  // archives for all wds shards
  std::vector<Index> wds_shards_prefixsums_;  // prefix sum of numbers of samples in wds shards
  
  size_t first_wds_shard_index_ = 0;          // the index of the first wds shard to use
  size_t current_wds_shard_index_ = 0;        // current archive that is being read
  size_t first_sample_index_ = 0;    // index of the first sample in the first wds shard to use
  size_t current_sample_index_ = 0;  // index of the current sample read from the current wds shard
  Index GetCurrentSampleIndex();

  FileStream::MappingReserver mmap_reserver_;
};

}  // namespace dali
#endif  // DALI_OPERATORS_READER_LOADER_WEBDATASET_LOADER_H_
