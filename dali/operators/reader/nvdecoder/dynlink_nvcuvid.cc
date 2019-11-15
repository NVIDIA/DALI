/*
 * Copyright 1993-2017 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */


#include <stdio.h>

#include "dali/operators/reader/nvdecoder/dynlink_nvcuvid.h"

tcuvidCreateVideoSource               ptr_cuvidCreateVideoSource;
tcuvidCreateVideoSourceW              ptr_cuvidCreateVideoSourceW;
tcuvidDestroyVideoSource              ptr_cuvidDestroyVideoSource;
tcuvidSetVideoSourceState             ptr_cuvidSetVideoSourceState;
tcuvidGetVideoSourceState             ptr_cuvidGetVideoSourceState;
tcuvidGetSourceVideoFormat            ptr_cuvidGetSourceVideoFormat;
tcuvidGetSourceAudioFormat            ptr_cuvidGetSourceAudioFormat;

tcuvidCreateVideoParser               ptr_cuvidCreateVideoParser;
tcuvidParseVideoData                  ptr_cuvidParseVideoData;
tcuvidDestroyVideoParser              ptr_cuvidDestroyVideoParser;


tcuvidGetDecoderCaps                  ptr_cuvidGetDecoderCaps;
tcuvidCreateDecoder                   ptr_cuvidCreateDecoder;
tcuvidDestroyDecoder                  ptr_cuvidDestroyDecoder;
tcuvidDecodePicture                   ptr_cuvidDecodePicture;
tcuvidGetDecodeStatus                 ptr_cuvidGetDecodeStatus;
tcuvidReconfigureDecoder              ptr_cuvidReconfigureDecoder;

#if !defined(__CUVID_DEVPTR64) || defined(__CUVID_INTERNAL)
tcuvidMapVideoFrame                   ptr_cuvidMapVideoFrame;
tcuvidUnmapVideoFrame                 ptr_cuvidUnmapVideoFrame;
#endif

#if defined(_WIN64) || defined(__LP64__) || defined(__x86_64) || defined(AMD64) || defined(_M_AMD64)
tcuvidMapVideoFrame64                 ptr_cuvidMapVideoFrame64;
tcuvidUnmapVideoFrame64               ptr_cuvidUnmapVideoFrame64;
#if defined(__CUVID_DEVPTR64) && !defined(__CUVID_INTERNAL)
#define ptr_cuvidMapVideoFrame        ptr_cuvidMapVideoFrame64
#define ptr_cuvidUnmapVideoFrame      ptr_cuvidUnmapVideoFrame64
#endif
#endif
tcuvidCtxLockCreate                   ptr_cuvidCtxLockCreate;
tcuvidCtxLockDestroy                  ptr_cuvidCtxLockDestroy;
tcuvidCtxLock                         ptr_cuvidCtxLock;
tcuvidCtxUnlock                       ptr_cuvidCtxUnlock;

#define STRINGIFY(X) #X

#include <dlfcn.h>

static char __DriverLibName[] = "libnvcuvid.so";
static char __DriverLibName1[] = "libnvcuvid.so.1";
static std::mutex m;

static CUresult LOAD_LIBRARY(DLLDRIVER *pInstance)
{
  *pInstance = dlopen(__DriverLibName, RTLD_NOW);

  if (*pInstance == NULL)
  {
    *pInstance = dlopen(__DriverLibName1, RTLD_NOW);

    if (*pInstance == NULL)
    {
      printf("dlopen \"%s\" failed!\n", __DriverLibName);
      return CUDA_ERROR_UNKNOWN;
    }
}

  return CUDA_SUCCESS;
}

#define GET_PROC_EX(name, alias, required)                          \
  ptr_##alias = (t##name )dlsym(driver_lib, #name);                  \
  if (ptr_##alias == NULL && required) {                            \
    printf("Failed to find required function \"%s\" in %s\n",       \
            #name, __DriverLibName);                                \
    return CUDA_ERROR_UNKNOWN;                                      \
  }

#define GET_PROC_EX_V2(name, alias, required)                       \
  alias = (t##name *)dlsym(driver_lib, STRINGIFY(name##_v2));        \
  if (alias == NULL && required) {                                  \
    printf("Failed to find required function \"%s\" in %s\n",       \
            STRINGIFY(name##_v2), __DriverLibName);                 \
    return CUDA_ERROR_UNKNOWN;                                      \
  }

#define CHECKED_CALL(call)          \
  do {                              \
    CUresult result = (call);       \
    if (CUDA_SUCCESS != result) {   \
      return result;                \
    }                               \
  } while(0)

#define GET_PROC_REQUIRED(name) GET_PROC_EX(name,name,1)
#define GET_PROC_OPTIONAL(name) GET_PROC_EX(name,name,0)
#define GET_PROC(name)          GET_PROC_REQUIRED(name)
#define GET_PROC_V2(name)       GET_PROC_EX_V2(name,name,1)

CUresult cuvidInit(unsigned int Flags, DLLDRIVER &driver_lib)
{
  driver_lib = NULL;
  CHECKED_CALL(LOAD_LIBRARY(&driver_lib));

  // fetch all function pointers
  GET_PROC(cuvidCreateVideoSource);
  GET_PROC(cuvidCreateVideoSourceW);
  GET_PROC(cuvidDestroyVideoSource);
  GET_PROC(cuvidSetVideoSourceState);
  GET_PROC(cuvidGetVideoSourceState);
  GET_PROC(cuvidGetSourceVideoFormat);
  GET_PROC(cuvidGetSourceAudioFormat);

  GET_PROC(cuvidCreateVideoParser);
  GET_PROC(cuvidParseVideoData);
  GET_PROC(cuvidDestroyVideoParser);


  GET_PROC(cuvidGetDecoderCaps);
  GET_PROC(cuvidCreateDecoder);
  GET_PROC(cuvidDestroyDecoder);
  GET_PROC(cuvidDecodePicture);
  GET_PROC_OPTIONAL(cuvidGetDecodeStatus);
  GET_PROC_OPTIONAL(cuvidReconfigureDecoder);

#if !defined(__CUVID_DEVPTR64) || defined(__CUVID_INTERNAL)
  GET_PROC(cuvidMapVideoFrame);
  GET_PROC(cuvidUnmapVideoFrame);
#endif

#if defined(_WIN64) || defined(__LP64__) || defined(__x86_64) || defined(AMD64) || defined(_M_AMD64)
  GET_PROC(cuvidMapVideoFrame64);
  GET_PROC(cuvidUnmapVideoFrame64);
#endif
  GET_PROC(cuvidCtxLockCreate);
  GET_PROC(cuvidCtxLockDestroy);
  GET_PROC(cuvidCtxLock);
  GET_PROC(cuvidCtxUnlock);

  return CUDA_SUCCESS;
}

DLLDRIVER cuvidInitChecked(unsigned int Flags) {
  DLLDRIVER driver_lib;
  std::lock_guard<std::mutex> lock(m);
  cuvidInit(Flags, driver_lib);
  return driver_lib;
}

void cuvidDeinit(DLLDRIVER driver_lib) {
  std::lock_guard<std::mutex> lock(m);
  dlclose(driver_lib);
}
