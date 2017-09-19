/**
 * Functions for decoding JPEG images using both CPU & GPU
 */

#ifndef HYBRID_DECODER_H_
#define HYBRID_DECODER_H_

#include "batched_idct.h"
#include "common.h"
#include "device_buffer.h"
#include "host_buffer.h"
#include "huffman_decoder.h"
#include "input_stream_jpeg.h"
#include "jpeg_parser.h"

struct DctParams;
struct HuffmanDecoderState;
struct JpegParserState;
struct ParsedJpeg;

void parseRawJpegHost(const unsigned char *rawData, size_t len,
    JpegParserState *state, ParsedJpeg *jpeg);

/**
 * Decodes jpeg stored in `jpeg` into DCT coefficients in `hostBlocksDct`
 *
 * NOTE: The way this API is set up currently is such that the memory allocation
 * on the host needed to store the DCT coefficients is handled internally with
 * the `HostBuffer` object, which avoids reallocation. This way the user can simply
 * pass the same HostBlocksDCT object in over and over and ignore the requirement to
 * resize it every time. If we want to let the user have more control over the memory
 * allocation, we can easily switch this function to take in a raw pointer that has
 * been pre-allocated to the correct size using the dims stored in `dctDims` in the
 * ParsedJpeg object.
 */
void huffmanDecodeHost(const ParsedJpeg &jpeg, HuffmanDecoderState *state,
    vector<HostBlocksDCT> *dctCoeffs);

void huffmanDecodeHost(const ParsedJpeg &jpeg, HuffmanDecoderState *state,
    vector<Npp16s*> *dctCoeffs);

struct DctState {
  DctState();
  ~DctState();    
  NppiDCTState *dctState;
  bool sm3xOrMore;
};

/**
 * De-quantizes and performs and inverse discrete cosine transform 
 * to convert quantized DCT coefficients into an image plane.
 *
 * @param pointer to the dct coefficients on device
 * @param step (stride) of dct coefficients in memory
 * @param pointer to store results on device
 * @param step of destination pointer
 * @param dims of the input DCT coefficients
 * @param precision of the quantization table
 * @param pointer to the quantization table on device
 * @param dct state object
 */
void dctQuantInv(const Npp16s *dctCoeff, unsigned dctStep, Npp8u* dst, unsigned dstStep,
    NppiSize dstSize, QuantizationTable::QuantizationTablePrecision prec,
    const void *quantTable, DctState *dctState);

/**
 * Contructed from data on all of the image planes that are to be idct-ed. Lets the 
 * user choose when to do the copy of parameters to device
 */
class BatchedDctParam {
public:
  BatchedDctParam();
  ~BatchedDctParam();
    
  // Calculates params & copies to device
  void loadToDevice(const vector<const Npp16s*> &dctCoeff,
      const vector<unsigned> &dctStep, const vector<Npp8u*> &dst,
      const vector<unsigned> &dstStep, const vector<NppiSize> &dstSize,
      const vector<const void*> &quantTable,
      QuantizationTable::QuantizationTablePrecision prec);

  int *dImgIdxs;
  DctParams *dParams;
  int2 *dGridInfo;
  int numBlocks;
  QuantizationTable::QuantizationTablePrecision p;

private:
  // Helper to build params
  void initParamsAndGridInfo(DctParams *params, int2 *gridInfo,
      int *numBlocks, const Npp16s *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
      const void *pQuantizationTable, NppiSize oSizeROI);

  // Underlying memory for device params
  DeviceBuffer m1;
  DeviceBuffer m2;
  DeviceBuffer m3;
    
  // tmp memory to work with
  HostBuffer tmp;
};

void batchedDctQuantInv(BatchedDctParam *param);

struct DctQuantInvImageParam {
  Npp16s *src = nullptr;
  int srcStep = 0;
  
  Npp8u *dst = nullptr;
  int dstWidth = 0;
  
  int2 gridInfo = {0, 0};
};
  
bool validateBatchedDctQuantInvParams(const int *srcSteps,
    const NppiSize *dstDims, int num);

void getBatchedInvDctLaunchParams(const NppiSize *dims, int num,
    int *numBlocks, int2 *gridInfo);

void getBatchedInvDctImageIndices(const NppiSize *dims, int num,
    int *imgIdxs);

// Note(tgale): We currently only support 8-bit quantization tables
void batchedDctQuantInv(Npp16s **srcs, int *srcSteps, void *quantTables,
    int2 *gridInfo, int *imgIdxs, Npp8u **dsts, NppiSize *dstDims, int numBlocks);

void batchedDctQuantInv(DctQuantInvImageParam *params, void *quantTables,
    int *imgIdxs, int numBlocks);

void getImageSizeStepAndOffset(int w, int h, int c,
    ComponentSampling ratio, NppiSize *roi, int *step,
    int *offset);

/**
 * Converts from YCbCr to RGB while also performing any 
 * needed upsampling of downsampled image components.
 *
 * @param array of pointers to each image component on device
 * @param step (stride) for each image component
 * @param dst pointer to store the output at on device
 * @param step of the dst pointer
 * @param dimensions of the (full scale) input image planes
 * @param the sampling ratio 
 *
 * NOTE: We could calculate the sampling ratio from the dims, but 
 * for the batched version it might actually save some time to 
 * have all of the sampling ratios calculated prior. Calculating 
 * the sampling ratio of each image during prefetching would mask 
 * this potential cost.
 */
void yCbCrToRgb(const Npp8u *imgPlanes[3], int planeSteps[3], Npp8u *dst,
    int dstStep, NppiSize dims, ComponentSampling sRatio);

// Not tested
void yCbCrToBgr(const Npp8u *imgPlanes[3], int planeSteps[3], Npp8u *dst,
    int dstStep, NppiSize dims, ComponentSampling sRatio);

void yToRgb(const Npp8u *img, int step, Npp8u *dst, int dstStep,
    NppiSize dims, cudaStream_t stream);

#endif // HYBRID_DECODER_H_
