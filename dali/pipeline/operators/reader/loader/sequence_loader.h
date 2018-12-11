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

namespace filesystem {
using Stream = std::pair<std::string, std::vector<std::string>>;

/**
 * @brief Gather info about extracted streams
 *
 * Expects file_root to contain set of directories, each of them represents one extracted video
 * stream. Extracted video stream is represented by one file for each frame, sorting the paths to
 * frames lexicographically should give the original order of frames.
 *
 * Example:
 * > file_root
 *   > 0
 *     > 00001.png
 *     > 00002.png
 *     > 00003.png
 *     > 00004.png
 *     > 00005.png
 *     > 00006.png
 *     ....
 *   > 1
 *     > 00001.png
 *     > 00002.png
 *     > 00003.png
 *     > 00004.png
 *     > 00005.png
 *     > 00006.png
 *     ....
 *
 * @param file_root
 * @return std::vector<Stream> GatherExtractedStreams
 */
std::vector<Stream> DLL_PUBLIC GatherExtractedStreams(string file_root);

}  // namespace filesystem

namespace detail {
/**
 * @brief Calculate how many full sequences of sequence_lenght fit in each stream.
 *
 * For stream [0, 1, 2, 3, 4, 5], and sequence_lenght = 3 we will consider:
 * [0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5] -> 4 full sequences
 *
 * Behaviour for sequence of length 0 is undefined
 */
std::vector<size_t> DLL_PUBLIC
CalculateSequencesCounts(const std::vector<filesystem::Stream> &streams, size_t sequence_lenght);
}  // namespace detail

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
        streams_(filesystem::GatherExtractedStreams(file_root_)),
        sequences_counts_(detail::CalculateSequencesCounts(streams_, sequence_length_)),
        total_size_(std::accumulate(sequences_counts_.begin(), sequences_counts_.end(), Index{})),
        current_stream_(0),
        current_frame_(0) {}

  void PrepareEmpty(TensorSequence *tensor) override;
  void ReadSample(TensorSequence *tensor) override;
  Index Size() override;

 private:
  // TODO(klecki) For now sequence is <directory, image list> pair, later it
  // will be a video file

  string file_root_;
  int32_t sequence_length_;
  std::vector<filesystem::Stream> streams_;
  std::vector<size_t> sequences_counts_;
  Index total_size_;
  size_t current_stream_, current_frame_;

  void LoadFrame(const filesystem::Stream &s, Index frame, Tensor<CPUBackend> *target);
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_READER_LOADER_SEQUENCE_LOADER_H_
