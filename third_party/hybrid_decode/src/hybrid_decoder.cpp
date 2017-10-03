#include "hybrid_decoder.h"

#include "debug.h"
#include "device_buffer.h"
#include "host_buffer.h"
#include "y_to_rgb.h"

#include <cstring>
#include <stdexcept>

void parseRawJpegHost(const unsigned char *rawData, size_t len,
  JpegParserState *state, ParsedJpeg *jpeg) {
  jpeg->imageBuffer.resize(len + nppiJpegDecodeGetScanDeadzoneSize());
  memcpy(jpeg->imageBuffer.data(), rawData, len);
  InputStreamJPEG stream(jpeg->imageBuffer.data(), len);

  JpegParser parser(jpeg, state);
  parser.parse(&stream);
}

void huffmanDecodeHost(const ParsedJpeg &jpeg, HuffmanDecoderState *state,
  vector<HostBlocksDCT> *dctCoeffs) {
  HuffmanDecoder decoder(jpeg, state, dctCoeffs);
  decoder.decode();
}

void huffmanDecodeHost(const ParsedJpeg &jpeg, HuffmanDecoderState *state,
  vector<Npp16s*> *dctCoeffs) {
  // Be sneaky and wrap the pointers
  vector<HostBlocksDCT> tmpDctBlocks;
  for (int i = 0; i < jpeg.components; ++i) {
    HostBlocksDCT tmp(jpeg.yCbCrDims[i].width / 8,
        jpeg.yCbCrDims[i].height / 8,
        (*dctCoeffs)[i], jpeg.dctSize[i]);
    tmpDctBlocks.push_back(std::move(tmp));
  }
  
  HuffmanDecoder decoder(jpeg, state, &tmpDctBlocks);
  decoder.decode();
}

DctState::DctState() {
  // TODO(Trevor): Store this info somewhere globally
  // so we don't have to always redo it
  // int device;
  // cudaDeviceProp props;
  // CHECK_CUDA(cudaGetDevice(&device),
  //         "Failed to get device id");
  // CHECK_CUDA(cudaGetDeviceProperties(&props, device),
  //         "Failed to get device properties");
  // TODO(Trevor): We don't have global state for the lib, and repeatedly querying
  // the device props kills performance. For now we only support >=SM3
  sm3xOrMore = true; // props.major >= 3;
    
  NPP_CHECK_NPP(nppiDCTInitAlloc(&dctState),
    "Unable to allocate base DCT memory");
}

DctState::~DctState() {
  NPP_CHECK_NPP(nppiDCTFree(dctState),
    "Failed to free nppiDCTstate");
}

void dctQuantInv(const Npp16s *dctCoeff, unsigned dctStep, Npp8u *dst, unsigned dstStep,
  NppiSize dstSize, QuantizationTable::QuantizationTablePrecision prec,
  const void *quantTable, DctState *dctState) {
  TimeRange _tr("idct");

  if(dstSize.width % 8 !=0)
    throw std::runtime_error("Width not mod 8");
  if(dstSize.height % 8 !=0)
    throw std::runtime_error("Height not mod 8");
    
  if (dst == nullptr) {
    throw std::runtime_error("dst ptr is nullptr");
  }
  if (dctCoeff == nullptr) {
    throw std::runtime_error("dctCoeff is nullptr");
  }
    
  // TODO(Trevor): switch on >=SM3 to call the correct iDCT
  if (dctState->sm3xOrMore) {
    switch(prec) {
    case QuantizationTable::PRECISION_8_BIT:
      NPP_CHECK_NPP(nppiDCTQuantInv8x8LS_JPEG_16s8u_C1R_NEW(dctCoeff,
          dctStep, dst, dstStep, (Npp8u*)quantTable,
          dstSize, dctState->dctState),
        "Error performaing inverse DCT");
      break;
            
    case QuantizationTable::PRECISION_16_BIT:
      NPP_CHECK_NPP(nppiDCTQuant16Inv8x8LS_JPEG_16s8u_C1R_NEW(dctCoeff,
          dctStep, dst, dstStep, (Npp16u*)quantTable,
          dstSize, dctState->dctState),
        "Error performaing inverse DCT");
      break;
    }
  } else {
    throw std::runtime_error("sm < 3.0 not supported");
    // switch(prec) {
    // case QuantizationTable::PRECISION_8_BIT:
    //     NPP_CHECK_NPP(nppiDCTQuantInv8x8LS_JPEG_16s8u_C1R(dctCoeff,
    //                     dctStep, dst, dstStep, (Npp8u*)quantTable,
    //                     dstSize),
    //             "Error performaing inverse DCT");
    //     break;
            
    // case QuantizationTable::PRECISION_16_BIT:
    //     NPP_CHECK_NPP(nppiDCTQuant16Inv8x8LS_JPEG_16s8u_C1R(dctCoeff,
    //                     dctStep, dst, dstStep, (Npp16u*)quantTable,
    //                     dstSize),
    //             "Error performaing inverse DCT");
    //     break;
    // }
  }
}

BatchedDctParam::BatchedDctParam() : dParams(nullptr), dGridInfo(nullptr),
                                     dImgIdxs(nullptr), numBlocks(0) {}

BatchedDctParam::~BatchedDctParam() {}

void BatchedDctParam::loadToDevice(const vector<const Npp16s*> &dctCoeff,
  const vector<unsigned> &dctStep, const vector<Npp8u*> &dst,
  const vector<unsigned> &dstStep, const vector<NppiSize> &dstSize,
  const vector<const void*> &quantTable,
  QuantizationTable::QuantizationTablePrecision prec) {
  TimeRange _tr("load_to_device");
  int num = dctCoeff.size();
  ASSERT(dst.size() == num);
  ASSERT(dstStep.size() == num);
  ASSERT(dstSize.size() == num);
  ASSERT(quantTable.size() == num);
  p = prec;
  numBlocks = 0;

  // TODO(Trevor): This memory management can be done better. We should use the
  // tmp host buffer for hParams & hGridInfo and only extend it for hIdxs (we
  // don't know the dims of it before we complete the loop)
  //
  // Calculate dct params, grid info, img idxs, and num blocks
  vector<int> hIdxs;
  vector<DctParams> hParams(num);
  vector<int2> hGridInfo(num);
  for (int i = 0; i < num; ++i) {
    int newBlocks = 0;

    initParamsAndGridInfo(&hParams[i], &hGridInfo[i], &newBlocks, dctCoeff[i],
      dctStep[i], dst[i], dstStep[i], quantTable[i], dstSize[i]);

    // create vector of indices so each CTA can find its parameters
    for (int j = 0; j < newBlocks; ++j) {
      hIdxs.push_back(i);
    }

    // Store the block-offset for this image
    hGridInfo[i].y = numBlocks;
    numBlocks += newBlocks;
  }
    
  // // Pack into tmp buffer, copy to device, set pointers
  // tmp.resize(sizeof(int)*hIdxs.size() + sizeof(DctParams) * num + sizeof(int2) * num);
  // memcpy(tmp.data(), hIdxs.data(), sizeof(int) * hIdxs.size());
  // memcpy(tmp.data() + sizeof(int) * hIdxs.size(), hParams.data(), sizeof(DctParams) * num);
  // memcpy(tmp.data() + sizeof(int) * hIdxs.size() + sizeof(DctParams) * num,
  //         hGridInfo.data(), sizeof(int2) * num);

  // Copy to device
  {
    TimeRange _tr1("idxs");
    m1.resize(sizeof(int)*hIdxs.size());
    CHECK_CUDA(cudaMemcpyAsync(m1.data(), hIdxs.data(), m1.size(),
        cudaMemcpyHostToDevice, nppGetStream()));
  }

  {
    TimeRange _tr2("dctparam");
    m2.resize(sizeof(DctParams) * num);
    CHECK_CUDA(cudaMemcpyAsync(m2.data(), hParams.data(), m2.size(),
        cudaMemcpyHostToDevice, nppGetStream()));
  }

  {
    TimeRange _tr3("gridinfo");
    m3.resize(sizeof(int2) * num);
    CHECK_CUDA(cudaMemcpyAsync(m3.data(), hGridInfo.data(), m3.size(),
        cudaMemcpyHostToDevice, nppGetStream()));
  }
    
  // Set pointers
  dImgIdxs = (int*)m1.data();
  dParams = (DctParams*)m2.data();
  dGridInfo = (int2*)m3.data();
}

void BatchedDctParam::initParamsAndGridInfo(DctParams *params, int2 *gridInfo,
  int *numBlocks, const Npp16s *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
  const void *pQuantizationTable, NppiSize oSizeROI) {
  // Basic input validation
  ASSERT(pDst);
  ASSERT(pSrc);
  ASSERT(oSizeROI.width >= 0 && oSizeROI.height >= 0);
    
  // 8-byte aligned output pointer
  //
  // TODO(Trevor): Do we guarantee this with the
  // way we allocate the output memory?
  ASSERT(nDstStep % 8 == 0);
    
  // Output YCbCr channels are decoded into blocks of 8x8
  ASSERT(oSizeROI.width % 8 == 0);
  ASSERT(oSizeROI.height % 8 == 0);

  // Input dct coeffs are layed out in blocks of
  // 64x1, the stride must be a multiple of this
  ASSERT(nSrcStep % (64 * sizeof(Npp16s)) == 0);
    
  params->numBlocksPerRow = DivUp(oSizeROI.width, 8);
  params->img = (uint2*)((void*)(pDst));
  params->imgNumBlocksPerStride = DivUp(nDstStep, 8);
  params->dct = (int*)((void*)(pSrc));
  params->dctNumBlocksPerStride = DivUp(nSrcStep, (int)(64 * sizeof(Npp16s)));
  params->quantizationTable = pQuantizationTable;

  // X dim of the grid for this image
  gridInfo->x = DivUp(oSizeROI.width, 256);
    
  // Calculate the number of CTAs needed for this image
  *numBlocks += DivUp(oSizeROI.width, 256) * oSizeROI.height / 8;
}

void batchedDctQuantInv(BatchedDctParam *param) {
  TimeRange _tr("batched_idct");
  ASSERT(param);
    
  NPP_CHECK_NPP(batchedDCTQuantInv8x8xLS_JPEG_16s8u_C1R_NEW(param->dParams,
      param->dGridInfo, param->dImgIdxs, param->numBlocks),
    "Batched iDCT kernel failed");
}

bool validateBatchedDctQuantInvParams(const int *srcSteps, const NppiSize *dstDims, int num) {
  if (num <= 0) return false;
  for (int i = 0; i < num; ++i) {
    // Input DCT coefficients are layed out in blocks of
    // 64x1, the stride must must be a multiple of this
    if ((srcSteps[i] % (64 * sizeof(Npp16s))) != 0) return false;

    // The output YUV components are decoded into blocks of 8x8
    if (dstDims[i].width < 0 && dstDims[i].height < 0) return false;
    if (dstDims[i].width % 8 != 0) return false;
    if (dstDims[i].height % 8 != 0) return false;
  }
  return true;
}

// Note: The two following functions can take in dims of 0 with no affect. This is very
// useful for supporting grayscale images and color with the same code.
// 'getBatchedInvDctLaunchParams' only adds CTAs for components of non-zero size
// 'getBatchedInvDctImageIndices' only adds any image indices for components of non-zero size
void getBatchedInvDctLaunchParams(const NppiSize *dims, int num,
  int *numBlocks, int2 *gridInfo) {
  *numBlocks = 0;
  for (int i = 0; i < num; ++i) {
    gridInfo[i].x = DivUp(dims[i].width, 256);
    gridInfo[i].y = *numBlocks; // Offset in thet grid for this image

    (*numBlocks) += DivUp(dims[i].width, 256) * dims[i].height / 8;
  }
}

void getBatchedInvDctImageIndices(const NppiSize *dims, int num, int *imgIdxs) {
  int numBlocks = 0;
  for (int i = 0; i < num; ++i) {
    int newBlocks = DivUp(dims[i].width, 256) * dims[i].height / 8;
    for (int j = 0; j < newBlocks; ++j) {
      imgIdxs[numBlocks + j] = i;
    }
    numBlocks += newBlocks;
  }
}

void batchedDctQuantInv(Npp16s **srcs, int *srcSteps, void *quantTables,
  int2 *gridInfo, int *imgIdxs, Npp8u **dsts, NppiSize *dstDims, int numBlocks) {
  NPP_CHECK_NPP(batchedDCTQuantInv8x8xLS_JPEG_16s8u_C1R_NEW(srcs, srcSteps,
      quantTables, gridInfo, imgIdxs, dsts, dstDims, numBlocks),
    "Batched iDCT kernel failed");
}

void batchedDctQuantInv(DctQuantInvImageParam *params, void *quantTables,
  int *imgIdxs, int numBlocks) {
  NPP_CHECK_NPP(batchedDCTQuantInv8x8xLS_JPEG_16s8u_C1R_NEW(
      params, quantTables, imgIdxs, numBlocks),
    "Batched iDCT kernel failed");
}

void getImageSizeStepAndOffset(int w, int h, int c,
    ComponentSampling ratio, NppiSize *roi, int *step,
    int *offset) {
  roi->width = w;
  roi->height = h;
  if (c == 3) {
    // For RGB images, handle padding required by NPP
    // for different sampling ratios
    switch (ratio) {
    case YCbCr_422:
      if (w & 1) { // Must be multiple of 2
        roi->width = w + 1;
      }
      break;
    case YCbCr_420:
      if (h & 1) { // Must be multiple of 2
        roi->height = h + 1;
      }
      if (w & 1) {
        roi->width = w + 1;
      }
      break;
    case YCbCr_411:
      if (w & 3) { // Must be multiple of 4
        roi->width = (w & ~3) + 4;
      }
      break;
    }
  }
  *offset = roi->width * roi->height * c;
  *step = roi->width * c;
}

void yCbCrToRgb(const Npp8u *imgPlanes[3], int planeSteps[3], Npp8u *dst,
  int dstStep, NppiSize dims, ComponentSampling sRatio) {
  TimeRange _tr("to_rgb");
    
  switch (sRatio) {
  case YCbCr_444:
    // steps must all be the same
    if (planeSteps[0] != planeSteps[1] || planeSteps[0] != planeSteps[2]) {
      cout << planeSteps[0] << " " << planeSteps[1] << " " << planeSteps[2] << endl;
      throw std::runtime_error("Steps must all be the same for 444 sampling");
    }
    NPP_CHECK_NPP(nppiYCbCr444ToRGB_JPEG_8u_P3C3R(imgPlanes, planeSteps[0], dst, dstStep, dims),
      "Failed to launch YCbCr444->RGB kernel");
    break;
  case YCbCr_440:
    // we don't have this kernel in NPP
    throw std::runtime_error("YCbCr440->RGB not supported");
    break;
  case YCbCr_422:
    NPP_CHECK_NPP(nppiYCbCr422ToRGB_JPEG_8u_P3C3R(imgPlanes, planeSteps, dst, dstStep, dims),
      "Failed to launch YCbCr422->RGB kernel");
    break;
  case YCbCr_420:
    NPP_CHECK_NPP(nppiYCbCr420ToRGB_JPEG_8u_P3C3R(imgPlanes, planeSteps, dst, dstStep, dims),
      "Failed to launch YCbCr420->RGB kernel");
    break;
  case YCbCr_411:
    NPP_CHECK_NPP(nppiYCbCr411ToRGB_JPEG_8u_P3C3R(imgPlanes, planeSteps, dst, dstStep, dims),
      "Failed to launch YCbCr411->RGB kernel");
    break;
  case YCbCr_410:
    // we don't have this kernel in NPP
    throw std::runtime_error("YCbCr410->RGB not supported");
    break;
  case YCbCr_UNKNOWN:
    throw std::runtime_error("Unknown chrominance sampling ratio");
    break;
  }
}

void yCbCrToBgr(const Npp8u *imgPlanes[3], int planeSteps[3], Npp8u *dst,
  int dstStep, NppiSize dims, ComponentSampling sRatio) {
  TimeRange _tr("to_bgr");
    
  switch (sRatio) {
  case YCbCr_444:
    // steps must all be the same
    if (planeSteps[0] != planeSteps[1] || planeSteps[0] != planeSteps[2]) {
      cout << planeSteps[0] << " " << planeSteps[1] << " " << planeSteps[2] << endl;
      throw std::runtime_error("Steps must all be the same for 444 sampling");
    }
    NPP_CHECK_NPP(nppiYCbCr444ToBGR_JPEG_8u_P3C3R(imgPlanes, planeSteps[0], dst, dstStep, dims),
      "Failed to launch YCbCr444->BGR kernel");
    break;
  case YCbCr_440:
    // we don't have this kernel in NPP
    throw std::runtime_error("YCbCr440->BGR not supported");
    break;
  case YCbCr_422:
    NPP_CHECK_NPP(nppiYCbCr422ToBGR_JPEG_8u_P3C3R(imgPlanes, planeSteps, dst, dstStep, dims),
      "Failed to launch YCbCr422->BGR kernel");
    break;
  case YCbCr_420:
    NPP_CHECK_NPP(nppiYCbCr420ToBGR_JPEG_8u_P3C3R(imgPlanes, planeSteps, dst, dstStep, dims),
      "Failed to launch YCbCr420->BGR kernel");
    break;
  case YCbCr_411:
    NPP_CHECK_NPP(nppiYCbCr411ToBGR_JPEG_8u_P3C3R(imgPlanes, planeSteps, dst, dstStep, dims),
      "Failed to launch YCbCr411->BGR kernel");
    break;
  case YCbCr_410:
    // we don't have this kernel in NPP
    throw std::runtime_error("YCbCr410->BGR not supported");
    break;
  case YCbCr_UNKNOWN:
    throw std::runtime_error("Unknown chrominance sampling ratio");
    break;
  }
}

void yToRgb(const Npp8u *img, int step, Npp8u *dst, int dstStep,
    NppiSize dims, cudaStream_t stream) {
  grayToRgb(img, step, dst, dstStep, dims, stream);
}
