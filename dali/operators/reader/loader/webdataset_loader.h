#ifndef DALI_OPERATORS_READER_LOADER_WEBDATASET_LOADER_H_
#define DALI_OPERATORS_READER_LOADER_WEBDATASET_LOADER_H_

#include "dali/operators/reader/loader/loader.h"
#include "dali/operators/reader/loader/webdataset/tar_utils.h"
#include "dali/pipeline/data/tensor.h"

namespace dali {

// interesting arguments: urls, ext, dont_use_mmap, component_mode, dtype, read_ahead,
// tensor_init_bytes, num_shards, shard_id

class WebdatasetLoader : public Loader<CPUBackend, vector<Tensor<CPUBackend>>> {
 public:
  explicit WebdatasetLoader(const OpSpec& spec);
  ~WebdatasetLoader() override;

  void PrepareEmpty(vector<Tensor<CPUBackend>>&) override;
  void ReadSample(vector<Tensor<CPUBackend>>&) override;

 protected:
  Index SizeImpl() override;
  void PrepareMetadataImpl() override;
  void Reset(bool wrap_to_shard) override;

 private:
  vector<detail::TarArchive> wds_shards_;
  size_t first_wds_shard_offset_;
  size_t current_wds_shard_index_;
  
};

}  // namespace dali
#endif  // DALI_OPERATORS_READER_LOADER_WEBDATASET_LOADER_H_
