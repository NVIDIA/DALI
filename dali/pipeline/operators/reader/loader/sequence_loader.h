// Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_PIPELINE_OPERATORS_READER_LOADER_SEQUENCE_LOADER_H_
#define DALI_PIPELINE_OPERATORS_READER_LOADER_SEQUENCE_LOADER_H_

#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "dali/common.h"
#include "dali/pipeline/operators/reader/loader/loader.h"

namespace dali {

struct TensorSequence {
  std::vector<Tensor<CPUBackend>> tensors;
};

// TODO(klecki) consider using FileLoader as base class
// TODO(klecki) ?? allow for more high level grouping of sequences:
//              we should probably make sure that other Loaders read
//              sequentially, and allow for SequenceLoader to wrap any other
//              loader, similar for parser and reader
// TODO(klecki) loader is responsible for handling shuffle_
class SequenceLoader : public Loader<CPUBackend, TensorSequence> {
 public:
  explicit SequenceLoader(const OpSpec &spec)
      : Loader(spec),
        file_root_(spec.GetArgument<string>("file_root")),
        sequence_length_(
            spec.GetArgument<int32_t>("sequence_length")),  // TODO(klecki) change to size_t
        streams_(ParseStreams(file_root_)),
        stream_sizes_(CalculateStreamSizes(streams_, sequence_length_)),
        total_size_(std::accumulate(stream_sizes_.begin(), stream_sizes_.end(), Index{})),
        current_stream_(0),
        current_frame_(0) {}

  void PrepareEmpty(TensorSequence *tensor) override;
  void ReadSample(TensorSequence *tensor) override;
  Index Size() override;

 private:
  // TODO(klecki) For now sequence is <directory, image list> pair, later it
  // will be a video file
  using Stream = std::pair<std::string, std::vector<std::string>>;

  string file_root_;
  int32_t sequence_length_;
  std::vector<Stream> streams_;
  std::vector<size_t> stream_sizes_;
  Index total_size_;
  size_t current_stream_, current_frame_;

  std::vector<Stream> ParseStreams(string file_root);
  std::vector<size_t> CalculateStreamSizes(const std::vector<Stream> &streams,
                                           size_t sample_lenght);

  void LoadFrame(const Stream &s, Index frame, Tensor<CPUBackend> *target);
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_READER_LOADER_SEQUENCE_LOADER_H_
