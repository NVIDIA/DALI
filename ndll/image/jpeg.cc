// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
#include "ndll/image/jpeg.h"

#include <turbojpeg.h>

namespace ndll {

namespace {
#define TJPG_CALL(code)                           \
  do {                                            \
    int error = code;                             \
    NDLL_ASSERT(!error, tjGetErrorStr());         \
  } while (0)

void PrintSubsampling(int sampling) {
  switch (sampling) {
  case TJSAMP_444:
    cout << "sampling ratio: 444" << endl;
    break;
  case TJSAMP_422:
    cout << "sampling ratio: 422" << endl;
    break;
  case TJSAMP_420:
    cout << "sampling ratio: 420" << endl;
    break;
  case TJSAMP_GRAY:
    cout << "sampling ratio: gray" << endl;
    break;
  case TJSAMP_440:
    cout << "sampling ratio: 440" << endl;
    break;
  case TJSAMP_411:
    cout << "sampling ratio: 411" << endl;
    break;
  default:
    cout << "unknown sampling ratio" << endl;
  }
}
}  // namespace

bool CheckIsJPEG(const uint8 *jpeg, int) {
  if ((jpeg[0] == 255) && (jpeg[1] == 216)) {
    return true;
  }
  return false;
}

NDLLError_t GetJPEGImageDims(const uint8 *jpeg, int size, int *h, int *w) {
  // Note: For now we use turbo-jpeg header decompression. This
  // may be more expensive than using the hacky method MXNet has.
  // Worth benchmarking this at a later point
  NDLL_ASSERT(CheckIsJPEG(jpeg, size));

  tjhandle handle = tjInitDecompress();
  int sampling, color;
  TJPG_CALL(tjDecompressHeader3(handle,
          jpeg, size, w, h, &sampling, &color));
#ifndef NDEBUG
  // PrintSubsampling(sampling);
#endif
  return NDLLSuccess;
}

NDLLError_t DecodeJPEGHost(const uint8 *jpeg, int size,
    NDLLImageType type, int h, int w, uint8 *image) {
#ifndef NDEBUG
  NDLL_ASSERT(jpeg != nullptr);
  NDLL_ASSERT(size > 0);
  NDLL_ASSERT(h > 0);
  NDLL_ASSERT(w > 0);
  NDLL_ASSERT(image != nullptr);
  NDLL_ASSERT(CheckIsJPEG(jpeg, size));
#endif
  tjhandle handle = tjInitDecompress();
  TJPF pixel_format;
  if (type == NDLL_RGB) {
    pixel_format = TJPF_RGB;
  } else if (type == NDLL_BGR) {
    pixel_format = TJPF_BGR;
  } else if (type == NDLL_GRAY) {
    pixel_format = TJPF_GRAY;
  } else {
    NDLL_RETURN_ERROR("Unsupported image type.");
  }
  TJPG_CALL(tjDecompress2(handle, jpeg, size, image,
          w, 0, h, pixel_format, 0));
  return NDLLSuccess;
}

}  // namespace ndll
