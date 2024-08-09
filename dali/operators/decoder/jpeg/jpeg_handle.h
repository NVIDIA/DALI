/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Copyright 2019, NVIDIA CORPORATION. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// This file declares the functions and structures for memory I/O with libjpeg
// These functions are not meant to be used directly, see jpeg_mem.h instead.

#ifndef DALI_OPERATORS_DECODER_JPEG_JPEG_HANDLE_H_
#define DALI_OPERATORS_DECODER_JPEG_JPEG_HANDLE_H_

#include <string>
#include "dali/core/common.h"
#include "dali/operators/decoder/jpeg/jpeg_utils.h"

namespace dali {
namespace jpeg {

// Handler for fatal JPEG library errors: clean up & return
void CatchError(j_common_ptr cinfo);

// Handler for monitoring progress od jpeg decoding
// Can be used to limit the number of scans for progressive jpegs
// or arithmetic encoding, which could be exploited to exhaust CPU
// with a malicious file, see
// https://libjpeg-turbo.org/pmwiki/uploads/About/TwoIssueswiththeJPEGStandard.pdf
struct ProgressMgr{
  struct jpeg_progress_mgr pub;  // MUST be the first member to have the same address
  int max_scans;
  int scans_exceeded;
};
void ValidateProgress(j_common_ptr cinfo);
void SetupProgressMgr(j_decompress_ptr cinfo, ProgressMgr *progress);

typedef struct {
  struct jpeg_destination_mgr pub;
  JOCTET *buffer;
  int bufsize;
  int datacount;
  std::string *dest;
} MemDestMgr;

typedef struct {
  struct jpeg_source_mgr pub;
  const unsigned char *data;
  uint64_t datasize;
  bool try_recover_truncated_jpeg;
} MemSourceMgr;

void SetSrc(j_decompress_ptr cinfo, const void *data,
            uint64_t datasize, bool try_recover_truncated_jpeg);

// JPEG destination: we will store all the data in a buffer "buffer" of total
// size "bufsize", if the buffer overflows, we will be in trouble.
void SetDest(j_compress_ptr cinfo, void *buffer, int bufsize);
// Same as above, except that buffer is only used as a temporary structure and
// is emptied into "destination" as soon as it fills up.
void SetDest(j_compress_ptr cinfo, void *buffer, int bufsize,
             std::string *destination);

}  // namespace jpeg
}  // namespace dali

#endif  // DALI_OPERATORS_DECODER_JPEG_JPEG_HANDLE_H_
