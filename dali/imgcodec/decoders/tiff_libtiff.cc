// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <tiffio.h>
#include "dali/imgcodec/decoders/tiff_libtiff.h"
#include "dali/imgcodec/util/convert.h"

namespace dali {
namespace imgcodec {

class DecoderHelper {
 private:
  std::shared_ptr<InputStream> stream;
  const void *buffer;
  size_t buffer_size;

 public:
  DecoderHelper(ImageSource *in) : stream(in->Open()) {
    if (in->Kind() == InputKind::HostMemory) {
      buffer = in->RawData();
      buffer_size = in->Size();
    } else {
      buffer = nullptr;
    }
  }

  static tmsize_t read(thandle_t handle, void *buffer, tmsize_t n) {
    DecoderHelper *helper = reinterpret_cast<DecoderHelper *>(handle);
    return helper->stream->Read(buffer, n);
  }

  static tmsize_t write(thandle_t, void *, tmsize_t) {
    // Not used for decoding.
    return 0;
  }

  static toff_t seek(thandle_t handle, toff_t offset, int whence) {
    DecoderHelper *helper = reinterpret_cast<DecoderHelper *>(handle);
    helper->stream->SeekRead(offset, whence);
    return helper->stream->TellRead();
  }

  static int map(thandle_t handle, const void **base, toff_t *size) {
    DecoderHelper *helper = reinterpret_cast<DecoderHelper *>(handle);
    *base = helper->buffer;
    *size = helper->buffer_size;
    return 0;
  }

  static toff_t size(thandle_t handle) {
    DecoderHelper *helper = reinterpret_cast<DecoderHelper *>(handle);
    return helper->stream->Size();
  }

  static int close(thandle_t handle) {
    DecoderHelper *helper = reinterpret_cast<DecoderHelper *>(handle);
    delete helper;
    return 0;
  }
};

TIFF *openTiff(ImageSource *in) {
  if (in->Kind() == InputKind::Filename) {
    return TIFFOpen(in->Filename(), "r");
  } else {
    TIFFMapFileProc mapproc;
    DecoderHelper helper(in);
    if (in->Kind() == InputKind::HostMemory)
      mapproc = helper.map;

    return TIFFClientOpen("", "r", )
  }
}

DecodeResult LibTiffDecoderInstance::Decode(SampleView<CPUBackend> out,
                                           ImageSource *in,
                                           DecodeParams opts) {
  if (in->Kind() == InputKind::Filename)
}


}  // namespace imgcodec
}  // namespace dali
