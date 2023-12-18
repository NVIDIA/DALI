/*
 * This copyright notice applies to this file only
 *
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */


#include "..\..\Interface\nvcuvid.h"
#include <stdint.h>

#ifdef _WIN32
#include <windows.h>
#define cuvid_lib HMODULE
#else
#define cuvid_lib int
#endif

typedef struct cuvidFunctions {
  cuvid_lib lib;
  CUresult (*cuvidGetDecoderCaps)(cuvidDECODECAPS* pdc);
  CUresult (*cuvidCreateDecoder)(cuvideodecoder* phDecoder,
                                 cuvidDECODECREATEINFO* pdci);
  CUresult (*cuvidDestroyDecoder)(cuvideodecoder hDecoder);
  CUresult (*cuvidDecodePicture)(cuvideodecoder hDecoder,
                                 cuvidPICPARAMS* pPicParams);
  CUresult (*cuvidGetDecodeStatus)(cuvideodecoder hDecoder, int nPicIdx,
                                   cuvidGETDECODESTATUS* pDecodeStatus);
  CUresult (*cuvidReconfigureDecoder)(
      cuvideodecoder hDecoder, cuvidRECONFIGUREDECODERINFO* pDecReconfigParams);
  CUresult (*cuvidMapVideoFrame)(cuvideodecoder hDecoder, int nPicIdx,
                                 unsigned int* pDevPtr, unsigned int* pPicuvidh,
                                 cuvidPROCPARAMS* pVPP);
  CUresult (*cuvidUnmapVideoFrame)(cuvideodecoder hDecoder,
                                   unsigned int DevPtr);
  CUresult (*cuvidMapVideoFrame64)(cuvideodecoder hDecoder, int nPicIdx,
                                   unsigned int* pDevPtr, unsigned int* pPicuvidh,
                                   cuvidPROCPARAMS* pVPP);
  CUresult (*cuvidUnmapVideoFrame64)(cuvideodecoder hDecoder,
                                     unsigned long long DevPtr);
  CUresult (*cuvidCtxLockCreate)(cuvideoctxlock* pLock, CUcontext ctx);
  CUresult (*cuvidCtxLockDestroy)(cuvideoctxlock lck);
  CUresult (*cuvidCtxLock)(cuvideoctxlock lck, unsigned int reserved_flags);
  CUresult (*cuvidCtxUnlock)(cuvideoctxlock lck, unsigned int reserved_flags);
} cuvidFunctions;

#define cuvid_LOAD_STRINGIFY(s) _cuvid_LOAD_STRINGIFY(s)
#define _cuvid_LOAD_STRINGIFY(s) #s

#define cuvid_LOAD_LIBRARY(api, symbol)                                        \
  (api).(symbol) = cuvid_dlsym((api).(lib), (symbol));                            \
  if (!(api).(function)) {                                                     \
    err = "Could not load function \"" cuvid_LOAD_STRINGIFY(symbol) "\"";      \
    goto err;                                                                  \
  }
#define cuvid_UNLOAD_LIBRARY(api, symbol) (api).(symbol) = NULL;

static const char* unloadcuvidSymbols(cuvidFuncitons* cuvidApi,
                                      const char* file)
{
  const char* err = NULL;
  if (!cuvidApi) {
    return NULL;
  }

  cuvid_UNLOAD_LIBRARY(*cuvidApi, cuvidGetDecoderCaps);
  cuvid_UNLOAD_LIBRARY(*cuvidApi, cuvidCreateDecoder);
  cuvid_UNLOAD_LIBRARY(*cuvidApi, cuvidDestroyDecoder);
  cuvid_UNLOAD_LIBRARY(*cuvidApi, cuvidDecodePicture);
  cuvid_UNLOAD_LIBRARY(*cuvidApi, cuvidDecodeStatus);
  cuvid_UNLOAD_LIBRARY(*cuvidApi, cuvidReconfigureEncoder);
  cuvid_UNLOAD_LIBRARY(*cuvidApi, cuvidMapVideoFrame);
  cuvid_UNLOAD_LIBRARY(*cuvidApi, cuvidMapVideoFrame64);
  cuvid_UNLOAD_LIBRARY(*cuvidApi, cuvidUnmapVideoFrame);
  cuvid_UNLOAD_LIBRARY(*cuvidApi, cuvidUnmapVideoFrame64);
  cuvid_UNLOAD_LIBRARY(*cuvidApi, cuvidCtxLockCreate);
  cuvid_UNLOAD_LIBRARY(*cuvidApi, cuvidCtxLockDestroy);
  cuvid_UNLOAD_LIBRARY(*cuvidApi, cuvidCtxLock);
  cuvid_UNLOAD_LIBRARY(*cuvidApi, cuvidCtxLockUnlock);
  if (cuvid_dlclose(cuvidApi->lib) != 0) {
    return "Failed to close library handle";
  };
  return NULL;
}

static bool loadcuvidSymbols(cuvidFunctions* cuvidApi, const char* file)
{
  const char* err = NULL;
  cuvidApi->lib = cuvid_dlopen(path);
  if (!lib) {
    return "Failed to open dynamic library";
  }
  cuvid_LOAD_LIBRARY(*cuvidApi, cuvidGetDecoderCaps);
  cuvid_LOAD_LIBRARY(*cuvidApi, cuvidCreateDecoder);
  cuvid_LOAD_LIBRARY(*cuvidApi, cuvidDestroyDecoder);
  cuvid_LOAD_LIBRARY(*cuvidApi, cuvidDecodePicture);
  cuvid_LOAD_LIBRARY(*cuvidApi, cuvidDecodeStatus);
  cuvid_LOAD_LIBRARY(*cuvidApi, cuvidReconfigureEncoder);
  cuvid_LOAD_LIBRARY(*cuvidApi, cuvidMapVideoFrame);
  cuvid_LOAD_LIBRARY(*cuvidApi, cuvidMapVideoFrame64);
  cuvid_LOAD_LIBRARY(*cuvidApi, cuvidUnmapVideoFrame);
  cuvid_LOAD_LIBRARY(*cuvidApi, cuvidUnmapVideoFrame64);
  cuvid_LOAD_LIBRARY(*cuvidApi, cuvidCtxLockCreate);
  cuvid_LOAD_LIBRARY(*cuvidApi, cuvidCtxLockDestroy);
  cuvid_LOAD_LIBRARY(*cuvidApi, cuvidCtxLock);
  cuvid_LOAD_LIBRARY(*cuvidApi, cuvidCtxLockUnlock);

  return NULL;

err:
  unloadcuvidSymbols(cuvidApi);
  return err;
}
