#include <cuda.h>
#include "VideoCodecSDKUtils/Interface/cuviddec.h"
#include "VideoCodecSDKUtils/Interface/nvcuvid.h"

void *NvcuvidLoadSymbol(const char *name);

#define LOAD_SYMBOL_FUNC Nvcuvid##LoadSymbol

#pragma GCC diagnostic ignored "-Wattributes"


CUresult CUDAAPI cuvidGetDecoderCapsNotFound(CUVIDDECODECAPS *) {
  return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuvidGetDecoderCaps(CUVIDDECODECAPS * pdc) {
  using FuncPtr = CUresult (CUDAAPI *)(CUVIDDECODECAPS *);
  static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuvidGetDecoderCaps")) ?
                           reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuvidGetDecoderCaps")) :
                           cuvidGetDecoderCapsNotFound;
  return func_ptr(pdc);
}

CUresult CUDAAPI cuvidCreateDecoderNotFound(CUvideodecoder *, CUVIDDECODECREATEINFO *) {
  return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuvidCreateDecoder(CUvideodecoder * phDecoder, CUVIDDECODECREATEINFO * pdci) {
  using FuncPtr = CUresult (CUDAAPI *)(CUvideodecoder *, CUVIDDECODECREATEINFO *);
  static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuvidCreateDecoder")) ?
                           reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuvidCreateDecoder")) :
                           cuvidCreateDecoderNotFound;
  return func_ptr(phDecoder, pdci);
}

CUresult CUDAAPI cuvidDestroyDecoderNotFound(CUvideodecoder) {
  return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuvidDestroyDecoder(CUvideodecoder hDecoder) {
  using FuncPtr = CUresult (CUDAAPI *)(CUvideodecoder);
  static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuvidDestroyDecoder")) ?
                           reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuvidDestroyDecoder")) :
                           cuvidDestroyDecoderNotFound;
  return func_ptr(hDecoder);
}

CUresult CUDAAPI cuvidDecodePictureNotFound(CUvideodecoder, CUVIDPICPARAMS *) {
  return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuvidDecodePicture(CUvideodecoder hDecoder, CUVIDPICPARAMS * pPicParams) {
  using FuncPtr = CUresult (CUDAAPI *)(CUvideodecoder, CUVIDPICPARAMS *);
  static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuvidDecodePicture")) ?
                           reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuvidDecodePicture")) :
                           cuvidDecodePictureNotFound;
  return func_ptr(hDecoder, pPicParams);
}

CUresult CUDAAPI cuvidGetDecodeStatusNotFound(CUvideodecoder, int, CUVIDGETDECODESTATUS *) {
  return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuvidGetDecodeStatus(CUvideodecoder hDecoder, int nPicIdx, CUVIDGETDECODESTATUS * pDecodeStatus) {
  using FuncPtr = CUresult (CUDAAPI *)(CUvideodecoder, int, CUVIDGETDECODESTATUS *);
  static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuvidGetDecodeStatus")) ?
                           reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuvidGetDecodeStatus")) :
                           cuvidGetDecodeStatusNotFound;
  return func_ptr(hDecoder, nPicIdx, pDecodeStatus);
}

CUresult CUDAAPI cuvidReconfigureDecoderNotFound(CUvideodecoder, CUVIDRECONFIGUREDECODERINFO *) {
  return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuvidReconfigureDecoder(CUvideodecoder hDecoder, CUVIDRECONFIGUREDECODERINFO * pDecReconfigParams) {
  using FuncPtr = CUresult (CUDAAPI *)(CUvideodecoder, CUVIDRECONFIGUREDECODERINFO *);
  static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuvidReconfigureDecoder")) ?
                           reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuvidReconfigureDecoder")) :
                           cuvidReconfigureDecoderNotFound;
  return func_ptr(hDecoder, pDecReconfigParams);
}

CUresult CUDAAPI cuvidMapVideoFrame64NotFound(CUvideodecoder, int, unsigned long long *, unsigned int *, CUVIDPROCPARAMS *) {
  return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuvidMapVideoFrame64(CUvideodecoder hDecoder, int nPicIdx, unsigned long long * pDevPtr, unsigned int * pPitch, CUVIDPROCPARAMS * pVPP) {
  using FuncPtr = CUresult (CUDAAPI *)(CUvideodecoder, int, unsigned long long *, unsigned int *, CUVIDPROCPARAMS *);
  static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuvidMapVideoFrame64")) ?
                           reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuvidMapVideoFrame64")) :
                           cuvidMapVideoFrame64NotFound;
  return func_ptr(hDecoder, nPicIdx, pDevPtr, pPitch, pVPP);
}

CUresult CUDAAPI cuvidUnmapVideoFrame64NotFound(CUvideodecoder, unsigned long long) {
  return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuvidUnmapVideoFrame64(CUvideodecoder hDecoder, unsigned long long DevPtr) {
  using FuncPtr = CUresult (CUDAAPI *)(CUvideodecoder, unsigned long long);
  static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuvidUnmapVideoFrame64")) ?
                           reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuvidUnmapVideoFrame64")) :
                           cuvidUnmapVideoFrame64NotFound;
  return func_ptr(hDecoder, DevPtr);
}

CUresult CUDAAPI cuvidCtxLockCreateNotFound(CUvideoctxlock *, CUcontext) {
  return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuvidCtxLockCreate(CUvideoctxlock * pLock, CUcontext ctx) {
  using FuncPtr = CUresult (CUDAAPI *)(CUvideoctxlock *, CUcontext);
  static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuvidCtxLockCreate")) ?
                           reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuvidCtxLockCreate")) :
                           cuvidCtxLockCreateNotFound;
  return func_ptr(pLock, ctx);
}

CUresult CUDAAPI cuvidCtxLockDestroyNotFound(CUvideoctxlock) {
  return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuvidCtxLockDestroy(CUvideoctxlock lck) {
  using FuncPtr = CUresult (CUDAAPI *)(CUvideoctxlock);
  static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuvidCtxLockDestroy")) ?
                           reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuvidCtxLockDestroy")) :
                           cuvidCtxLockDestroyNotFound;
  return func_ptr(lck);
}

CUresult CUDAAPI cuvidCtxLockNotFound(CUvideoctxlock, unsigned int) {
  return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuvidCtxLock(CUvideoctxlock lck, unsigned int reserved_flags) {
  using FuncPtr = CUresult (CUDAAPI *)(CUvideoctxlock, unsigned int);
  static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuvidCtxLock")) ?
                           reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuvidCtxLock")) :
                           cuvidCtxLockNotFound;
  return func_ptr(lck, reserved_flags);
}

CUresult CUDAAPI cuvidCtxUnlockNotFound(CUvideoctxlock, unsigned int) {
  return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuvidCtxUnlock(CUvideoctxlock lck, unsigned int reserved_flags) {
  using FuncPtr = CUresult (CUDAAPI *)(CUvideoctxlock, unsigned int);
  static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuvidCtxUnlock")) ?
                           reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuvidCtxUnlock")) :
                           cuvidCtxUnlockNotFound;
  return func_ptr(lck, reserved_flags);
}

CUresult CUDAAPI cuvidCreateVideoSourceNotFound(CUvideosource *, const char *, CUVIDSOURCEPARAMS *) {
  return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuvidCreateVideoSource(CUvideosource * pObj, const char * pszFileName, CUVIDSOURCEPARAMS * pParams) {
  using FuncPtr = CUresult (CUDAAPI *)(CUvideosource *, const char *, CUVIDSOURCEPARAMS *);
  static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuvidCreateVideoSource")) ?
                           reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuvidCreateVideoSource")) :
                           cuvidCreateVideoSourceNotFound;
  return func_ptr(pObj, pszFileName, pParams);
}

CUresult CUDAAPI cuvidCreateVideoSourceWNotFound(CUvideosource *, const wchar_t *, CUVIDSOURCEPARAMS *) {
  return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuvidCreateVideoSourceW(CUvideosource * pObj, const wchar_t * pwszFileName, CUVIDSOURCEPARAMS * pParams) {
  using FuncPtr = CUresult (CUDAAPI *)(CUvideosource *, const wchar_t *, CUVIDSOURCEPARAMS *);
  static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuvidCreateVideoSourceW")) ?
                           reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuvidCreateVideoSourceW")) :
                           cuvidCreateVideoSourceWNotFound;
  return func_ptr(pObj, pwszFileName, pParams);
}

CUresult CUDAAPI cuvidSetVideoSourceStateNotFound(CUvideosource, cudaVideoState) {
  return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuvidSetVideoSourceState(CUvideosource obj, cudaVideoState state) {
  using FuncPtr = CUresult (CUDAAPI *)(CUvideosource, cudaVideoState);
  static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuvidSetVideoSourceState")) ?
                           reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuvidSetVideoSourceState")) :
                           cuvidSetVideoSourceStateNotFound;
  return func_ptr(obj, state);
}

cudaVideoState CUDAAPI cuvidGetVideoSourceStateNotFound(CUvideosource) {
  return cudaVideoState_Error;
}

cudaVideoState cuvidGetVideoSourceState(CUvideosource obj) {
  using FuncPtr = cudaVideoState (CUDAAPI *)(CUvideosource);
  static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuvidGetVideoSourceState")) ?
                           reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuvidGetVideoSourceState")) :
                           cuvidGetVideoSourceStateNotFound;
  return func_ptr(obj);
}

CUresult CUDAAPI cuvidGetSourceVideoFormatNotFound(CUvideosource, CUVIDEOFORMAT *, unsigned int) {
  return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuvidGetSourceVideoFormat(CUvideosource obj, CUVIDEOFORMAT * pvidfmt, unsigned int flags) {
  using FuncPtr = CUresult (CUDAAPI *)(CUvideosource, CUVIDEOFORMAT *, unsigned int);
  static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuvidGetSourceVideoFormat")) ?
                           reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuvidGetSourceVideoFormat")) :
                           cuvidGetSourceVideoFormatNotFound;
  return func_ptr(obj, pvidfmt, flags);
}

CUresult CUDAAPI cuvidGetSourceAudioFormatNotFound(CUvideosource, CUAUDIOFORMAT *, unsigned int) {
  return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuvidGetSourceAudioFormat(CUvideosource obj, CUAUDIOFORMAT * paudfmt, unsigned int flags) {
  using FuncPtr = CUresult (CUDAAPI *)(CUvideosource, CUAUDIOFORMAT *, unsigned int);
  static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuvidGetSourceAudioFormat")) ?
                           reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuvidGetSourceAudioFormat")) :
                           cuvidGetSourceAudioFormatNotFound;
  return func_ptr(obj, paudfmt, flags);
}

CUresult CUDAAPI cuvidCreateVideoParserNotFound(CUvideoparser *, CUVIDPARSERPARAMS *) {
  return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuvidCreateVideoParser(CUvideoparser * pObj, CUVIDPARSERPARAMS * pParams) {
  using FuncPtr = CUresult (CUDAAPI *)(CUvideoparser *, CUVIDPARSERPARAMS *);
  static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuvidCreateVideoParser")) ?
                           reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuvidCreateVideoParser")) :
                           cuvidCreateVideoParserNotFound;
  return func_ptr(pObj, pParams);
}

CUresult CUDAAPI cuvidParseVideoDataNotFound(CUvideoparser, CUVIDSOURCEDATAPACKET *) {
  return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuvidParseVideoData(CUvideoparser obj, CUVIDSOURCEDATAPACKET * pPacket) {
  using FuncPtr = CUresult (CUDAAPI *)(CUvideoparser, CUVIDSOURCEDATAPACKET *);
  static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuvidParseVideoData")) ?
                           reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuvidParseVideoData")) :
                           cuvidParseVideoDataNotFound;
  return func_ptr(obj, pPacket);
}

CUresult CUDAAPI cuvidDestroyVideoParserNotFound(CUvideoparser) {
  return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
}

CUresult cuvidDestroyVideoParser(CUvideoparser obj) {
  using FuncPtr = CUresult (CUDAAPI *)(CUvideoparser);
  static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuvidDestroyVideoParser")) ?
                           reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("cuvidDestroyVideoParser")) :
                           cuvidDestroyVideoParserNotFound;
  return func_ptr(obj);
}
