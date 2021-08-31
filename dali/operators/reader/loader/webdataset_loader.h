#ifndef DALI_OPERATORS_READER_LOADER_WEBDATASET_LOADER_H_
#define DALI_OPERATORS_READER_LOADER_WEBDATASET_LOADER_H_

#include <fstream>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>
#include "dali/operators/reader/loader/loader.h"
#include "dali/operators/reader/loader/webdataset/tar_utils.h"
#include "dali/pipeline/data/tensor.h"

namespace dali {
namespace {

const char kExtDelim = ';';

}  // namespace

class WebdatasetLoader : public Loader<CPUBackend, vector<Tensor<CPUBackend>>> {
 public:
  static const std::set<DALIDataType> kSupportedTypes;
  explicit WebdatasetLoader(const OpSpec& spec);
  ~WebdatasetLoader() override;

  void PrepareEmpty(std::vector<Tensor<CPUBackend>>&) override;
  void ReadSample(std::vector<Tensor<CPUBackend>>&) override;

 protected:
  Index SizeImpl() override;
  void PrepareMetadataImpl() override;
  void Reset(bool wrap_to_shard) override;

  enum class MissingExt
  {
    Empty,
    Skip,
    Raise,
    Invalid
  };
  static MissingExt Str2MissingExt(std::string);

  std::vector<std::string> uris_;
  std::vector<std::string> configs_;
  std::vector<std::set<std::string>> ext_;
  std::unordered_map<std::string, std::vector<size_t>>
      ext_map_;  // maps an extension to sample indicies
  MissingExt missing_component_behavior_;
  std::vector<DALIDataType> dtype_;

 private:
  size_t MaxCommonDtypeSize(const std::string& extension) const;
  void SetDataPointer(std::vector<Tensor<CPUBackend>>& sample, std::vector<bool>& sample_was_set,
                      const std::string& extension, const std::string& source_info,
                      std::shared_ptr<void> data, int64_t size) const;
  uint8_t* ShareDataPointer(std::vector<Tensor<CPUBackend>>& sample,
                            std::vector<bool>& sample_was_set, const std::string& extension,
                            const std::string& source_info, int64_t size,
                            size_t dtype_max_size) const;
  void MarkCached(std::vector<Tensor<CPUBackend>>& sample, std::vector<bool>& sample_was_set,
                  const std::string& extension, const std::string& source_info) const;

  struct SampleConfig {
    int64_t start_offset;
    int64_t end_offset;
    std::set<std::string> extensions;
  };

  std::vector<std::vector<SampleConfig>> wds_shards_metadata_;  // data from the config files
  friend SampleConfig ParseSampleConfig(std::ifstream&, std::string);
  std::vector<SampleConfig> ParseConfig(std::string);


  size_t total_size_ = 0;                       // total size of all config files
  std::vector<detail::TarArchive> wds_shards_;  // archives for all wds shards
  std::vector<Index> wds_shards_prefixsums_;    // prefix sum of numbers of samples in wds shards

  size_t first_wds_shard_index_ = 0;    // the index of the first wds shard to use
  size_t current_wds_shard_index_ = 0;  // current archive that is being read
  size_t first_sample_index_ = 0;       // index of the first sample in the first wds shard to use
  size_t current_sample_index_ = 0;  // index of the current sample read from the current wds shard
  Index GetCurrentSampleIndex() const;

  FileStream::MappingReserver mmap_reserver_;
};

}  // namespace dali
#endif  // DALI_OPERATORS_READER_LOADER_WEBDATASET_LOADER_H_
