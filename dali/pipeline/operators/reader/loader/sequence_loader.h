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
 * @brief Generate sets of paths for each sequence to be loaded.
 *
 * Only consider full sequences that do not cross stream boundary
 */
std::vector<std::vector<std::string>> DLL_PUBLIC
GenerateSequences(const std::vector<filesystem::Stream> &streams, size_t sequence_length,
                  size_t step, size_t stride);
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
        sequence_length_(spec.GetArgument<int32_t>("sequence_length")),
        step_(spec.GetArgument<int32_t>("step")),
        stride_(spec.GetArgument<int32_t>("stride")),
        streams_(filesystem::GatherExtractedStreams(file_root_)),
        sequences_(detail::GenerateSequences(streams_, sequence_length_, step_, stride_)),
        total_size_(sequences_.size()),
        current_sequence_(0) {
    DALI_ENFORCE(sequence_length_ > 0, "Sequence length must be positive");
    DALI_ENFORCE(step_ > 0, "Step must be positive");
    DALI_ENFORCE(stride_ > 0, "Stride must be positive");
    if (shuffle_) {
      // seeded with hardcoded value to get
      // the same sequence on every shard
      std::mt19937 g(524287);
      std::shuffle(sequences_.begin(), sequences_.end(), g);
    }
  }

  void PrepareEmpty(TensorSequence *tensor) override;
  void ReadSample(TensorSequence *tensor) override;
  Index Size() override;

 private:
  // TODO(klecki) For now sequence is <directory, image list> pair, later it
  // will be a video file

  string file_root_;
  int32_t sequence_length_;
  int32_t step_;
  int32_t stride_;
  std::vector<filesystem::Stream> streams_;
  std::vector<std::vector<std::string>> sequences_;
  Index total_size_;
  Index current_sequence_;

  void LoadFrame(const std::vector<std::string> &s, Index frame, Tensor<CPUBackend> *target);
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_READER_LOADER_SEQUENCE_LOADER_H_
