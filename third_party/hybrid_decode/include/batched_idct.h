#ifndef BATCHED_IDCT_H_
#define BATCHED_IDCT_H_

#include <npp.h>

struct DctQuantInvImageParam;

struct DctParams {
  int     numBlocksPerRow;
  uint2  *img;
  int     imgNumBlocksPerStride;
  int    *dct;
  int     dctNumBlocksPerStride;
  const void *quantizationTable;
};

NppStatus batchedDCTQuantInv8x8xLS_JPEG_16s8u_C1R_NEW(DctParams *dParams,
        int2 *dGridInfo, int *dImgIdxs, int numBlocks);

NppStatus batchedDCTQuantInv8x8xLS_JPEG_16s8u_C1R_NEW(Npp16s **srcs, int *srcSteps, void *quantTables, int2 *gridInfo, int *imgIdxs, Npp8u **dsts, NppiSize *dstDims, int numBlocks);

NppStatus batchedDCTQuantInv8x8xLS_JPEG_16s8u_C1R_NEW(DctQuantInvImageParam *params,
  void *quantTables, int *imgIdxs, int numBlocks);

#endif // BATCHED_IDCT_H_
