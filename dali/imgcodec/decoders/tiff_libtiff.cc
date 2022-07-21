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

#define LIBTIFF_CALL_SUCCESS 1
#define LIBTIFF_CALL(call)                                \
  do {                                                    \
    int retcode = (call);                                 \
    DALI_ENFORCE(LIBTIFF_CALL_SUCCESS == retcode,         \
      "libtiff call failed with code "                    \
      + std::to_string(retcode) + ": " #call);            \
  } while (0)

namespace dali {
namespace imgcodec {

class DecoderHelper {
 private:
  std::shared_ptr<InputStream> stream;
  const void *buffer;
  size_t buffer_size;

 public:
  explicit DecoderHelper(ImageSource *in) : stream(in->Open()) {
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

  static int map(thandle_t handle, void **base, toff_t *size) {
    DecoderHelper *helper = reinterpret_cast<DecoderHelper *>(handle);
    *base = const_cast<void*>(helper->buffer);
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
    if (in->Kind() == InputKind::HostMemory)
      mapproc = &DecoderHelper::map;
    else
      mapproc = nullptr;
    return TIFFClientOpen("", "r", reinterpret_cast<thandle_t>(new DecoderHelper(in)),
                          &DecoderHelper::read,
                          &DecoderHelper::write,
                          &DecoderHelper::seek,
                          &DecoderHelper::close,
                          &DecoderHelper::size,
                          mapproc,
                          /* unmap */ 0);
  }
}

DecodeResult LibTiffDecoderInstance::Decode(SampleView<CPUBackend> out,
                                           ImageSource *in,
                                           DecodeParams opts) {
  TIFF *tiff = openTiff(in);
  DALI_ENFORCE(tiff != nullptr, make_string("Unable to open TIFF image: ", in->SourceInfo()));
  // TODO(skarpinski) Check if open was successful
  // TODO(skarpinski) Port CanDecode here

  using InType = uint8_t;
  using OutType = uint8_t;

  // TODO(skarpinski) other formats
  DALI_ENFORCE(opts.format == DALI_RGB, "Only RGB supported for now");
  unsigned out_channels = 3;

  unsigned in_channels = 3;  // TODO(skarpinski) Read this from TIFF

  // TODO(skarpinski) ROI
  uint32_t image_width, image_height;
  LIBTIFF_CALL(TIFFGetField(tiff, TIFFTAG_IMAGEWIDTH, &image_width));
  LIBTIFF_CALL(TIFFGetField(tiff, TIFFTAG_IMAGELENGTH, &image_height));
  uint64_t out_row_stride = image_width * out_channels;

  auto row_nbytes = TIFFScanlineSize(tiff);
  std::unique_ptr<InType, void(*)(void*)> row_buf{
    static_cast<InType *>(_TIFFmalloc(row_nbytes)), _TIFFfree};
  DALI_ENFORCE(row_buf.get() != nullptr, "Could not allocate memory");
  memset(row_buf.get(), 0, row_nbytes);  // TODO(skarpinski) Do we need to zero it out?

  InType * const row_in  = row_buf.get();
  OutType * const img_out = out.mutable_data<OutType>();

  for (uint64_t y = 0; y < image_height; y++) {
    LIBTIFF_CALL(TIFFReadScanline(tiff, row_in, y, 0));

    // TODO(skarpinski) Color space conversion
    memcpy(img_out + (y * out_row_stride), row_in, out_row_stride);
  }

  return {true, nullptr};
}


}  // namespace imgcodec
}  // namespace dali
