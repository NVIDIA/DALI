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

#include <glob.h>
#include <algorithm>
#include <map>
#include <memory>

#include "dali/pipeline/operators/reader/loader/sequence_loader.h"
#include "dali/util/file.h"

namespace dali {

namespace {

std::string parent_dir(std::string path) {
  size_t idx = path.rfind("/");
  DALI_ENFORCE(idx != std::string::npos, "NO PARENT DIRECTORY FOR GIVEN PATH");
  return path.substr(0, idx + 1);  // return with trailing "/"
}

}  // namespace

void SequenceLoader::PrepareEmpty(TensorSequence *sequence) {
  sequence->tensors.resize(sequence_length_);
  for (auto &t : sequence->tensors) {
    PrepareEmptyTensor(&t);
  }
}

void SequenceLoader::ReadSample(TensorSequence *sequence) {
  // TODO(klecki) this is written as a prototype for video handling
  const auto &stream = streams_[current_stream_];
  // TODO(klecki) we probably should buffer the "stream", or recently used
  // frames
  for (int i = 0; i < sequence_length_; i++) {
    LoadFrame(stream, current_frame_ + i, &sequence->tensors[i]);
  }
  current_frame_++;
  // wrap-around
  if (current_frame_ == stream_sizes_[current_stream_]) {
    current_stream_++;
    current_frame_ = 0;
  }
  if (current_stream_ == streams_.size()) {
    current_stream_ = 0;
  }
}

Index SequenceLoader::Size() {
  return total_size_;
}

std::vector<SequenceLoader::Stream> SequenceLoader::ParseStreams(string file_root) {
  glob_t glob_buff;
  std::string glob_pattern = file_root + "/*/*";
  const int glob_flags = 0;
  int glob_ret = glob(glob_pattern.c_str(), glob_flags, nullptr, &glob_buff);
  DALI_ENFORCE(glob_ret == 0,
               "Glob for pattern: \"" + glob_pattern + "\" failed. Verify the file_root argument");
  std::vector<std::string> files;
  for (size_t i = 0; i < glob_buff.gl_pathc; i++) {
    files.emplace_back(glob_buff.gl_pathv[i]);
  }
  std::sort(files.begin(), files.end());
  std::map<std::string, size_t> streamName_bucket_map;
  std::vector<Stream> streams;
  for (const auto &f : files) {
    auto parent = parent_dir(f);
    auto bucket = streamName_bucket_map.find(parent);
    if (bucket == streamName_bucket_map.end()) {
      streams.push_back({parent, {f}});
      streamName_bucket_map[parent] = streams.size() - 1;
    } else {
      streams[bucket->second].second.push_back(f);
    }
  }
  return streams;
}

std::vector<size_t> SequenceLoader::CalculateStreamSizes(const std::vector<Stream> &streams,
                                                         size_t sample_lenght) {
  std::vector<size_t> result;
  for (const auto &s : streams) {
    result.push_back(s.second.size() - (sample_lenght - 1));
  }
  return result;
}

void SequenceLoader::LoadFrame(const Stream &s, Index frame_idx, Tensor<CPUBackend> *target) {
  const auto frame_filename = s.second[frame_idx];
  std::unique_ptr<FileStream> frame(FileStream::Open(frame_filename));
  Index frame_size = frame->Size();
  target->Resize({frame_size});
  frame->Read(target->mutable_data<uint8_t>(), frame_size);
  frame->Close();
}

}  // namespace dali
