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

#include "dali/image/image.h"
#include "dali/pipeline/operators/reader/loader/sequence_loader.h"


namespace dali {

namespace {

std::string parent_dir(std::string path) {
  size_t idx = path.rfind("/");
  DALI_ENFORCE(idx != std::string::npos, "NO PARENT DIRECTORY FOR GIVEN PATH");
  return path.substr(0, idx + 1);  // return with trailing "/"
}

}  // namespace

namespace filesystem {

std::vector<Stream> GatherExtractedStreams(string file_root) {
  glob_t glob_buff;
  std::string glob_pattern = file_root + "/*/*";
  const int glob_flags = 0;
  int glob_ret = glob(glob_pattern.c_str(), glob_flags, nullptr, &glob_buff);
  DALI_ENFORCE(glob_ret == 0,
               "Glob for pattern: \"" + glob_pattern + "\" failed. Verify the file_root argument");
  std::vector<std::string> files;
  for (size_t i = 0; i < glob_buff.gl_pathc; i++) {
    std::string file = glob_buff.gl_pathv[i];
    if (HasKnownImageExtension(file)) {
      files.push_back(std::move(file));
    }
  }
  globfree(&glob_buff);
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

}  // namespace filesystem

namespace detail {

std::vector<std::vector<std::string>> GenerateSequences(
    const std::vector<filesystem::Stream> &streams, size_t sequence_length, size_t step,
    size_t stride) {
  std::vector<std::vector<std::string>> sequences;
  for (const auto &s : streams) {
    for (size_t i = 0; i < s.second.size(); i += step) {
      std::vector<std::string> sequence;
      sequence.reserve(sequence_length);
      // this sequence won't fit
      if (i + (sequence_length - 1) * stride >= s.second.size()) {
        break;
      }
      // fill the sequence
      for (size_t seq_elem = 0; seq_elem < sequence_length; seq_elem++) {
        sequence.push_back(s.second[i + seq_elem * stride]);
      }
      sequences.push_back((sequence));
    }
  }
  return sequences;
}

}  // namespace detail

void SequenceLoader::PrepareEmpty(TensorSequence &sequence) {
  sequence.tensors.resize(sequence_length_);
  for (auto &t : sequence.tensors) {
    PrepareEmptyTensor(t);
  }
}

void SequenceLoader::ReadSample(TensorSequence &sequence) {
  // TODO(klecki) this is written as a prototype for video handling
  const auto &sequence_paths = sequences_[current_sequence_];
  // TODO(klecki) we probably should buffer the "stream", or recently used
  // frames
  for (int i = 0; i < sequence_length_; i++) {
    LoadFrame(sequence_paths, i, &sequence.tensors[i]);
  }
  current_sequence_++;
  // wrap-around
  MoveToNextShard(current_sequence_);
}

Index SequenceLoader::SizeImpl() {
  return total_size_;
}

void SequenceLoader::LoadFrame(const std::vector<std::string> &s, Index frame_idx,
                               Tensor<CPUBackend> *target) {
  const auto frame_filename = s[frame_idx];
  target->SetSourceInfo(frame_filename);
  target->SetSkipSample(false);

  // if image is cached, skip loading
  if (ShouldSkipImage(frame_filename)) {
    target->set_type(TypeInfo::Create<uint8_t>());
    target->Resize({1});
    target->SetSkipSample(true);
    return;
  }

  auto frame = FileStream::Open(frame_filename, read_ahead_);
  Index frame_size = frame->Size();
  // Release and unmap memory previously obtained by Get call
  if (copy_read_data_) {
    target->Resize({frame_size});
    frame->Read(target->mutable_data<uint8_t>(), frame_size);
  } else {
    auto p = frame->Get(frame_size);
    // Wrap the raw data in the Tensor object.
    target->ShareData(p, frame_size, {frame_size});
    target->set_type(TypeInfo::Create<uint8_t>());
  }
  frame->Close();
}

}  // namespace dali
