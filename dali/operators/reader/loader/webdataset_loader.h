#ifndef DALI_OPERATORS_READER_LOADER_WEBDATASET_LOADER_H_
#define DALI_OPERATORS_READER_LOADER_WEBDATASET_LOADER_H_

#include "dali/operators/reader/loader/loader.h"
#include "dali/pipeline/data/tensor.h"

namespace dali {

// interesting arguments: urls, ext, dont_use_mmap, component_mode, dtype, read_ahead
// possibly interesting: stick_to_shard, tensor_init_bytes, num_shards, shard_id

class WebdatasetLoader : public Loader<CPUBackend, vector<Tensor<CPUBackend>>> {
 public:
  explicit WebdatasetLoader(const OpSpec& spec);

  void PrepareEmpty(vector<Tensor<CPUBackend>>&) override;
  void ReadSample(vector<Tensor<CPUBackend>>&) override;
  Index SizeImpl() override;
  void Reset(bool wrap_to_shard) override;
};

}
#endif  // DALI_OPERATORS_READER_LOADER_WEBDATASET_LOADER_H_
