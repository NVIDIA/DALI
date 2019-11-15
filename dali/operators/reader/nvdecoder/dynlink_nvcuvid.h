/*
 * This copyright notice applies to this header file only:
 *
 * Copyright (c) 2010-2017 NVIDIA Corporation
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use,
 * copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the software, and to permit persons to whom the
 * software is furnished to do so, subject to the following
 * conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

#if !defined(DALI_OPERATORS_READER_NVDECODER_DYNLINK_NVCUVID_H_)
#define DALI_OPERATORS_READER_NVDECODER_DYNLINK_NVCUVID_H_

#include "dali/operators/reader/nvdecoder/cuviddec.h"
#include "dali/operators/reader/nvdecoder/nvcuvid.h"

#define NVCUVID_CALL(arg) CUDA_CALL(ptr_##arg)
#define NVCUVID_API_EXISTS(arg) (ptr_##arg != NULL)

typedef decltype(&cuvidCreateVideoSource) tcuvidCreateVideoSource;
typedef decltype(&cuvidCreateVideoSourceW) tcuvidCreateVideoSourceW;
typedef decltype(&cuvidDestroyVideoSource) tcuvidDestroyVideoSource;
typedef decltype(&cuvidSetVideoSourceState) tcuvidSetVideoSourceState;
typedef decltype(&cuvidGetVideoSourceState) tcuvidGetVideoSourceState;
typedef decltype(&cuvidGetSourceVideoFormat) tcuvidGetSourceVideoFormat;
typedef decltype(&cuvidGetSourceAudioFormat) tcuvidGetSourceAudioFormat;
typedef decltype(&cuvidCreateVideoParser) tcuvidCreateVideoParser;
typedef decltype(&cuvidParseVideoData) tcuvidParseVideoData;
typedef decltype(&cuvidDestroyVideoParser) tcuvidDestroyVideoParser;

typedef decltype(&cuvidGetDecoderCaps) tcuvidGetDecoderCaps;
typedef decltype(&cuvidCreateDecoder) tcuvidCreateDecoder;
typedef decltype(&cuvidDestroyDecoder) tcuvidDestroyDecoder;
typedef decltype(&cuvidDecodePicture) tcuvidDecodePicture;
typedef decltype(&cuvidGetDecodeStatus) tcuvidGetDecodeStatus;
typedef decltype(&cuvidReconfigureDecoder) tcuvidReconfigureDecoder;

#if !defined(__CUVID_DEVPTR64) || defined(__CUVID_INTERNAL)
  typedef decltype(&cuvidMapVideoFrame) tcuvidMapVideoFrame;
  typedef decltype(&cuvidUnmapVideoFrame) tcuvidUnmapVideoFrame;
#endif

#if defined(_WIN64) || defined(__LP64__) || defined(__x86_64) || defined(AMD64) || defined(_M_AMD64)
  typedef decltype(&cuvidMapVideoFrame64) tcuvidMapVideoFrame64;
  typedef decltype(&cuvidUnmapVideoFrame64) tcuvidUnmapVideoFrame64;
  #if defined(__CUVID_DEVPTR64) && !defined(__CUVID_INTERNAL)
    #define tcuvidMapVideoFrame      tcuvidMapVideoFrame64
    #define tcuvidUnmapVideoFrame    tcuvidUnmapVideoFrame64
  #endif
#endif
typedef decltype(&cuvidCtxLockCreate) tcuvidCtxLockCreate;
typedef decltype(&cuvidCtxLockDestroy) tcuvidCtxLockDestroy;
typedef decltype(&cuvidCtxLock) tcuvidCtxLock;
typedef decltype(&cuvidCtxUnlock) tcuvidCtxUnlock;

extern tcuvidCreateVideoSource               ptr_cuvidCreateVideoSource;
extern tcuvidCreateVideoSourceW              ptr_cuvidCreateVideoSourceW;
extern tcuvidDestroyVideoSource              ptr_cuvidDestroyVideoSource;
extern tcuvidSetVideoSourceState             ptr_cuvidSetVideoSourceState;
extern tcuvidGetVideoSourceState             ptr_cuvidGetVideoSourceState;
extern tcuvidGetSourceVideoFormat            ptr_cuvidGetSourceVideoFormat;
extern tcuvidGetSourceAudioFormat            ptr_cuvidGetSourceAudioFormat;

extern tcuvidCreateVideoParser               ptr_cuvidCreateVideoParser;
extern tcuvidParseVideoData                  ptr_cuvidParseVideoData;
extern tcuvidDestroyVideoParser              ptr_cuvidDestroyVideoParser;

extern tcuvidGetDecoderCaps                  ptr_cuvidGetDecoderCaps;
extern tcuvidCreateDecoder                   ptr_cuvidCreateDecoder;
extern tcuvidDestroyDecoder                  ptr_cuvidDestroyDecoder;
extern tcuvidDecodePicture                   ptr_cuvidDecodePicture;
extern tcuvidGetDecodeStatus                 ptr_cuvidGetDecodeStatus;
extern tcuvidReconfigureDecoder              ptr_cuvidReconfigureDecoder;

extern tcuvidMapVideoFrame                   ptr_cuvidMapVideoFrame;
extern tcuvidUnmapVideoFrame                 ptr_cuvidUnmapVideoFrame;

#if defined(_WIN64) || defined(__LP64__) || defined(__x86_64) || defined(AMD64) || defined(_M_AMD64)
  extern tcuvidMapVideoFrame64                 ptr_cuvidMapVideoFrame64;
  extern tcuvidUnmapVideoFrame64               ptr_cuvidUnmapVideoFrame64;
  #if defined(__CUVID_DEVPTR64) && !defined(__CUVID_INTERNAL)
    #define ptr_cuvidMapVideoFrame               ptr_cuvidMapVideoFrame64
    #define ptr_cuvidUnmapVideoFrame             ptr_cuvidUnmapVideoFrame64
  #endif
#endif
extern tcuvidCtxLockCreate                   ptr_cuvidCtxLockCreate;
extern tcuvidCtxLockDestroy                  ptr_cuvidCtxLockDestroy;
extern tcuvidCtxLock                         ptr_cuvidCtxLock;
extern tcuvidCtxUnlock                       ptr_cuvidCtxUnlock;

/**********************************************************************************************/

using DLLDRIVER = void *;

DLLDRIVER cuvidInitChecked(unsigned int Flags);
void cuvidDeinit(DLLDRIVER driver_lib);

#endif // DALI_OPERATORS_READER_NVDECODER_DYNLINK_NVCUVID_H_
