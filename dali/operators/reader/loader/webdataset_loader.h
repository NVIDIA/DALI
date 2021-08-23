#ifndef DALI_OPERATORS_READER_LOADER_WEBDATASET_LOADER_H_
#define DALI_OPERATORS_READER_LOADER_WEBDATASET_LOADER_H_

#include <string>
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

 private:
  std::vector<std::string> uris_;
  std::vector<std::string> configs_;
  std::vector<std::string> ext_;
  bool fail_on_missing_component_;
  DALIDataType dtype_;

  size_t total_size_;                      // total size of all input archives
  vector<detail::TarArchive> wds_shards_;  // archives for all wds shards
  size_t first_wds_shard_index_;           // the index of the first wds shard to use
  size_t first_sample_offset_;  // the offset of the first sample in the wds_shards[0]; later used
                                // for seeking
  size_t current_wds_shard_index_;  // current archive that is being read
};

}  // namespace dali
#endif  // DALI_OPERATORS_READER_LOADER_WEBDATASET_LOADER_H_
