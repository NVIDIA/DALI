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

#include "dali/pipeline/operators/reader/nvdecoder/dynlink_nvcuvid.h"

tcuvidCreateVideoSource               *cuvidCreateVideoSource;
tcuvidCreateVideoSourceW              *cuvidCreateVideoSourceW;
tcuvidDestroyVideoSource              *cuvidDestroyVideoSource;
tcuvidSetVideoSourceState             *cuvidSetVideoSourceState;
tcuvidGetVideoSourceState             *cuvidGetVideoSourceState;
tcuvidGetSourceVideoFormat            *cuvidGetSourceVideoFormat;
tcuvidGetSourceAudioFormat            *cuvidGetSourceAudioFormat;

tcuvidCreateVideoParser               *cuvidCreateVideoParser;
tcuvidParseVideoData                  *cuvidParseVideoData;
tcuvidDestroyVideoParser              *cuvidDestroyVideoParser;

tcuvidGetDecoderCaps                  *cuvidGetDecoderCaps;
tcuvidCreateDecoder                   *cuvidCreateDecoder;
tcuvidDestroyDecoder                  *cuvidDestroyDecoder;
tcuvidDecodePicture                   *cuvidDecodePicture;

tcuvidMapVideoFrame                   *cuvidMapVideoFrame;
tcuvidUnmapVideoFrame                 *cuvidUnmapVideoFrame;

#if defined(WIN64) || defined(_WIN64) || defined(__x86_64) || defined(AMD64) || defined(_M_AMD64)
tcuvidMapVideoFrame64                 *cuvidMapVideoFrame64;
tcuvidUnmapVideoFrame64               *cuvidUnmapVideoFrame64;
#endif

//tcuvidGetVideoFrameSurface            *cuvidGetVideoFrameSurface;
tcuvidCtxLockCreate                   *cuvidCtxLockCreate;
tcuvidCtxLockDestroy                  *cuvidCtxLockDestroy;
tcuvidCtxLock                         *cuvidCtxLock;
tcuvidCtxUnlock                       *cuvidCtxUnlock;


// Auto-lock helper for C++ applications
CCtxAutoLock::CCtxAutoLock(CUvideoctxlock ctx) 
    : m_ctx(ctx) 
{
    cuvidCtxLock(m_ctx, 0); 
}
CCtxAutoLock::~CCtxAutoLock()
{ 
    cuvidCtxUnlock(m_ctx, 0); 
}



#define STRINGIFY(X) #X

#include <dlfcn.h>

static char __DriverLibName[] = "libnvcuvid.so";
static char __DriverLibName1[] = "libnvcuvid.so.1";

typedef void *DLLDRIVER;

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

#define GET_PROC_EX(name, alias, required)                              \
    alias = (t##name *)dlsym(DriverLib, #name);                        \
    if (alias == NULL && required) {                                    \
        printf("Failed to find required function \"%s\" in %s\n",       \
               #name, __DriverLibName);                                  \
        return CUDA_ERROR_UNKNOWN;                                      \
    }

#define GET_PROC_EX_V2(name, alias, required)                           \
    alias = (t##name *)dlsym(DriverLib, STRINGIFY(name##_v2));         \
    if (alias == NULL && required) {                                    \
        printf("Failed to find required function \"%s\" in %s\n",       \
               STRINGIFY(name##_v2), __DriverLibName);                    \
        return CUDA_ERROR_UNKNOWN;                                      \
    }

#define CHECKED_CALL(call)              \
    do {                                \
        CUresult result = (call);       \
        if (CUDA_SUCCESS != result) {   \
            return result;              \
        }                               \
    } while(0)

#define GET_PROC_REQUIRED(name) GET_PROC_EX(name,name,1)
#define GET_PROC_OPTIONAL(name) GET_PROC_EX(name,name,0)
#define GET_PROC(name)          GET_PROC_REQUIRED(name)
#define GET_PROC_V2(name)       GET_PROC_EX_V2(name,name,1)

CUresult cuvidInit(unsigned int Flags)
{
    DLLDRIVER DriverLib;

    CHECKED_CALL(LOAD_LIBRARY(&DriverLib));

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

#if defined(WIN64) || defined(_WIN64) || defined(__x86_64) || defined(AMD64) || defined(_M_AMD64)
    GET_PROC(cuvidMapVideoFrame64);
    GET_PROC(cuvidUnmapVideoFrame64);
    cuvidMapVideoFrame   = cuvidMapVideoFrame64;
    cuvidUnmapVideoFrame = cuvidUnmapVideoFrame64;
#else
    GET_PROC(cuvidMapVideoFrame);
    GET_PROC(cuvidUnmapVideoFrame);
#endif

//    GET_PROC(cuvidGetVideoFrameSurface);
    GET_PROC(cuvidCtxLockCreate);
    GET_PROC(cuvidCtxLockDestroy);
    GET_PROC(cuvidCtxLock);
    GET_PROC(cuvidCtxUnlock);

    return CUDA_SUCCESS;
}

bool cuvidInitChecked(unsigned int Flags) {
    static std::mutex m;
    static bool initialized = false;

    if (initialized)
        return true;

    std::lock_guard<std::mutex> lock(m);

    static CUresult res = cuvidInit(Flags);
    initialized = (res == CUDA_SUCCESS);
    return initialized;
}
