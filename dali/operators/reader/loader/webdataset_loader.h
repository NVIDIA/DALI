// Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_READER_LOADER_WEBDATASET_LOADER_H_
#define DALI_OPERATORS_READER_LOADER_WEBDATASET_LOADER_H_

#include <fstream>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "dali/operators/reader/loader/loader.h"
#include "dali/operators/reader/loader/webdataset/tar_utils.h"
#include "dali/pipeline/data/tensor.h"
#include "dali/core/bitmask.h"

namespace dali {
namespace detail {
namespace wds {

constexpr char kExtDelim = ';';
const std::set<DALIDataType> kSupportedTypes = {DALI_UINT8,   DALI_UINT16, DALI_UINT32, DALI_UINT64,
                                                DALI_INT8,    DALI_INT16,  DALI_INT32,  DALI_INT64,
                                                DALI_FLOAT16, DALI_FLOAT,  DALI_FLOAT64};

enum class MissingExtBehavior {
  Empty,
  Skip,
  Raise,
  Invalid
};
MissingExtBehavior ParseMissingExtBehavior(std::string);

struct ComponentDesc {
  std::string ext;
  int64_t size = 0;

  ComponentDesc(std::string new_ext, int64_t new_size) : ext(std::move(new_ext)), size(new_size) {}
};

struct SampleDesc {
  int64_t start_offset;
  int64_t end_offset;
  std::vector<ComponentDesc> index_file_metadata;
};

}  // namespace wds
}  // namespace detail

class DLL_PUBLIC WebdatasetLoader : public Loader<CPUBackend, vector<Tensor<CPUBackend>>> {
 public:
  explicit WebdatasetLoader(const OpSpec& spec);
  ~WebdatasetLoader() override;

  void PrepareEmpty(std::vector<Tensor<CPUBackend>>&) override;
  void ReadSample(std::vector<Tensor<CPUBackend>>&) override;

 protected:
  Index SizeImpl() override;
  void PrepareMetadataImpl() override;
  void Reset(bool wrap_to_shard) override;

  std::vector<std::string> uris_;
  std::vector<std::string> index_paths_;
  std::vector<std::set<std::string>> ext_;
  std::unordered_map<std::string, std::vector<size_t>>
      ext_map_;  // maps an extension to sample indicies
  detail::wds::MissingExtBehavior missing_component_behavior_;
  std::vector<DALIDataType> dtypes_;

 private:
  void SetDataPointer(std::vector<Tensor<CPUBackend>>& sample, bitmask& sample_was_set,
                      const std::string& extension, const std::string& source_info,
                      std::shared_ptr<void> data, int64_t size) const;
  uint8_t* ShareDataPointer(std::vector<Tensor<CPUBackend>>& sample,
                            bitmask& sample_was_set, const std::string& extension,
                            const std::string& source_info, int64_t size) const;
  void MarkCached(std::vector<Tensor<CPUBackend>>& sample, bitmask& sample_was_set,
                  const std::string& extension, const std::string& source_info) const;

  std::vector<std::vector<detail::wds::SampleDesc>>
      wds_shards_metadata_;  // data from the index files


  size_t total_size_ = 0;                       // total size of all index files
  std::vector<detail::TarArchive> wds_shards_;  // archives for all wds shards
  std::vector<Index> wds_shards_prefixsums_;    // prefix sum of numbers of samples in wds shards

  size_t first_wds_shard_index_ = 0;    // the index of the first wds shard to use
  size_t current_wds_shard_index_ = 0;  // current archive that is being read
  size_t first_sample_index_ = 0;       // index of the first sample in the first wds shard to use
  size_t current_sample_index_ = 0;  // index of the current sample read from the current wds shard
  Index GetCurrentSampleIndex() const;

  FileStream::MappingReserver mmap_reserver_;
};

}  // namespace dali
#endif  // DALI_OPERATORS_READER_LOADER_WEBDATASET_LOADER_H_
