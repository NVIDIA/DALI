#include "batched_idct.h"

#include "hybrid_decoder.h"

#define C_a 1.387039845322148f //!< a = (2^0.5) * cos(    pi / 16);  Used in inverse DCT.  
#define C_b 1.306562964876377f //!< b = (2^0.5) * cos(    pi /  8);  Used in inverse DCT.  
#define C_c 1.175875602419359f //!< c = (2^0.5) * cos(3 * pi / 16);  Used in inverse DCT.  
#define C_d 0.785694958387102f //!< d = (2^0.5) * cos(5 * pi / 16);  Used in inverse DCT.  
#define C_e 0.541196100146197f //!< e = (2^0.5) * cos(3 * pi /  8);  Used in inverse DCT.  
#define C_f 0.275899379282943f //!< f = (2^0.5) * cos(7 * pi / 16);  Used in inverse DCT.  

// Normalization constant that is used in forward and inverse DCT
#define C_norm 0.3535533905932737f // 1 / (8^0.5)

static __constant__ int4 constZigzag[] = { 
  {  0,  2,  3,  9 }, 
  {  1,  4,  8, 11 },
  {  5,  7, 12, 18 },
  {  6, 13, 17, 24 },
  { 14, 16, 25, 31 },
  { 15, 26, 30, 40 },
  { 27, 29, 41, 44 },
  { 28, 42, 43, 53 },
  { 10, 20, 21, 35 },
  { 19, 22, 34, 36 },
  { 23, 33, 37, 48 },
  { 32, 38, 47, 49 },
  { 39, 46, 50, 57 },
  { 45, 51, 56, 58 },
  { 52, 55, 59, 62 },
  { 54, 60, 61, 63 }
};

static __device__ __forceinline__ Npp8u clampToUint8(int x) {
  if( x & ~0xFF )
    x = (-x) >> 31;
  return (Npp8u) x;
}

static __device__ __forceinline__ int divUp(int x, int d) {
    return (x + d - 1) / d;
}

template< typename QuantType, int READ_ZIGZAG >
__global__ __launch_bounds__(256, 6) void batchedInverseDct32x8Kernel(DctParams *allParams,
    int2 *gridInfo, int *imgIdxs)
{
  // find my image and my block ids in the sub-grid for my image
  int imgIdx = imgIdxs[blockIdx.x];
  DctParams params = allParams[imgIdx];
  int2 gInfo = gridInfo[imgIdx];
  int blockOffset = gInfo.y;
  
  int xBlockId = (blockIdx.x - blockOffset) % gInfo.x;
  int yBlockId = (blockIdx.x - blockOffset) / gInfo.x;


  // Shared memory to transpose blocks.
  __shared__ float smemBlock[8][256];

  // The number of blocks computed by this CTA.
  const int numBlocks = min(params.numBlocksPerRow - 32*xBlockId, 32);

  // We work on 32 blocks per CTA. Each block produces 64 values (64x16-bit i.e. 32x32-bit).
  int dctOffset = 32*yBlockId*params.dctNumBlocksPerStride + 1024*xBlockId;

  // The global offset for the block.
  for( int k = 0, i = threadIdx.y ; k < 4 ; ++k, i += 8 )
    if( i < numBlocks ) {
      reinterpret_cast<int*>(smemBlock)[33*i + threadIdx.x] = params.dct[dctOffset + 32*i + threadIdx.x];
    }
  __syncthreads();

  // Read the values from SMEM. We use a stride of 33 (66 because it's on 16-bit numbers) to avoid bank conflicts.
  float x0, x1, x2, x3, x4, x5, x6, x7;
  if( READ_ZIGZAG )
  {
    // Zigzag offsets.
    int4 zigzag0 = constZigzag[threadIdx.y + 0];
    int4 zigzag8 = constZigzag[threadIdx.y + 8];

    x0 = (float) reinterpret_cast<Npp16s*>(smemBlock)[66*threadIdx.x + zigzag0.x];
    x1 = (float) reinterpret_cast<Npp16s*>(smemBlock)[66*threadIdx.x + zigzag0.y];
    x2 = (float) reinterpret_cast<Npp16s*>(smemBlock)[66*threadIdx.x + zigzag0.z];
    x3 = (float) reinterpret_cast<Npp16s*>(smemBlock)[66*threadIdx.x + zigzag0.w];
    x4 = (float) reinterpret_cast<Npp16s*>(smemBlock)[66*threadIdx.x + zigzag8.x];
    x5 = (float) reinterpret_cast<Npp16s*>(smemBlock)[66*threadIdx.x + zigzag8.y];
    x6 = (float) reinterpret_cast<Npp16s*>(smemBlock)[66*threadIdx.x + zigzag8.z];
    x7 = (float) reinterpret_cast<Npp16s*>(smemBlock)[66*threadIdx.x + zigzag8.w];
  }
  else
  {
    x0 = (float) reinterpret_cast<Npp16s*>(smemBlock)[66*threadIdx.x + threadIdx.y +  0];
    x1 = (float) reinterpret_cast<Npp16s*>(smemBlock)[66*threadIdx.x + threadIdx.y +  8];
    x2 = (float) reinterpret_cast<Npp16s*>(smemBlock)[66*threadIdx.x + threadIdx.y + 16];
    x3 = (float) reinterpret_cast<Npp16s*>(smemBlock)[66*threadIdx.x + threadIdx.y + 24];
    x4 = (float) reinterpret_cast<Npp16s*>(smemBlock)[66*threadIdx.x + threadIdx.y + 32];
    x5 = (float) reinterpret_cast<Npp16s*>(smemBlock)[66*threadIdx.x + threadIdx.y + 40];
    x6 = (float) reinterpret_cast<Npp16s*>(smemBlock)[66*threadIdx.x + threadIdx.y + 48];
    x7 = (float) reinterpret_cast<Npp16s*>(smemBlock)[66*threadIdx.x + threadIdx.y + 56];
  }

  // Each warp loads the quantization values from the table.
  QuantType *qTable = (QuantType *) params.quantizationTable;
  QuantType q = qTable[threadIdx.x];

  // Convert to float.
  float q0 = (float) q.x;
  float q1 = (float) q.y;
    
  // Each thread quantize its values. 
  // Use sync version of shfl for Volta.
  x0 = x0 * __shfl_sync(0xffffffff, q0, threadIdx.y +  0);
  x1 = x1 * __shfl_sync(0xffffffff, q0, threadIdx.y +  8);
  x2 = x2 * __shfl_sync(0xffffffff, q0, threadIdx.y + 16);
  x3 = x3 * __shfl_sync(0xffffffff, q0, threadIdx.y + 24);
  x4 = x4 * __shfl_sync(0xffffffff, q1, threadIdx.y +  0);
  x5 = x5 * __shfl_sync(0xffffffff, q1, threadIdx.y +  8);
  x6 = x6 * __shfl_sync(0xffffffff, q1, threadIdx.y + 16);
  x7 = x7 * __shfl_sync(0xffffffff, q1, threadIdx.y + 24);

  // Run the Inverse DCT. Start with columns.
  float Y04P = x0 + x4;
  float Y04M = x0 - x4;

  float Y2b6eP = C_b * x2 + C_e * x6;
  float Y2e6bM = C_e * x2 - C_b * x6;

  float Y04P2b6ePP = Y04P + Y2b6eP;
  float Y04P2b6ePM = Y04P - Y2b6eP;
  float Y04M2e6bMP = Y04M + Y2e6bM;
  float Y04M2e6bMM = Y04M - Y2e6bM;

  float Y7f1aP3c5dPP = C_f * x7 + C_a * x1 + C_c * x3 + C_d * x5;
  float Y7a1fM3d5cMP = C_a * x7 - C_f * x1 + C_d * x3 - C_c * x5;
  float Y1c7dM3f5aPM = C_c * x1 - C_d * x7 - C_f * x3 - C_a * x5;
  float Y1d7cP3a5fMM = C_d * x1 + C_c * x7 - C_a * x3 + C_f * x5;

  x0 = C_norm * (Y04P2b6ePP + Y7f1aP3c5dPP);
  x7 = C_norm * (Y04P2b6ePP - Y7f1aP3c5dPP);
  x4 = C_norm * (Y04P2b6ePM + Y7a1fM3d5cMP);
  x3 = C_norm * (Y04P2b6ePM - Y7a1fM3d5cMP);

  x1 = C_norm * (Y04M2e6bMP + Y1c7dM3f5aPM);
  x5 = C_norm * (Y04M2e6bMM - Y1d7cP3a5fMM);
  x2 = C_norm * (Y04M2e6bMM + Y1d7cP3a5fMM);
  x6 = C_norm * (Y04M2e6bMP - Y1c7dM3f5aPM);

  // Make sure the threads have read their values from SMEM.
  __syncthreads();

  // Store to SMEM.
  smemBlock[threadIdx.y][threadIdx.x +   0] = x0;
  smemBlock[threadIdx.y][threadIdx.x +  32] = x1;
  smemBlock[threadIdx.y][threadIdx.x +  64] = x2;
  smemBlock[threadIdx.y][threadIdx.x +  96] = x3;
  smemBlock[threadIdx.y][threadIdx.x + 128] = x4;
  smemBlock[threadIdx.y][threadIdx.x + 160] = x5;
  smemBlock[threadIdx.y][threadIdx.x + 192] = x6;
  smemBlock[threadIdx.y][threadIdx.x + 224] = x7;

  // Make sure the threads have stored their values in SMEM.
  __syncthreads();

  // Extract the columns. Threads in a warp work on different image blocks.
  x0 = smemBlock[0][32*threadIdx.y + threadIdx.x];
  x1 = smemBlock[1][32*threadIdx.y + threadIdx.x];
  x2 = smemBlock[2][32*threadIdx.y + threadIdx.x];
  x3 = smemBlock[3][32*threadIdx.y + threadIdx.x];
  x4 = smemBlock[4][32*threadIdx.y + threadIdx.x];
  x5 = smemBlock[5][32*threadIdx.y + threadIdx.x];
  x6 = smemBlock[6][32*threadIdx.y + threadIdx.x];
  x7 = smemBlock[7][32*threadIdx.y + threadIdx.x];

  // Run the Inverse DCT. Start with columns.
  Y04P = x0 + x4;
  Y04M = x0 - x4;

  Y2b6eP = C_b * x2 + C_e * x6;
  Y2e6bM = C_e * x2 - C_b * x6;

  Y04P2b6ePP = Y04P + Y2b6eP;
  Y04P2b6ePM = Y04P - Y2b6eP;
  Y04M2e6bMP = Y04M + Y2e6bM;
  Y04M2e6bMM = Y04M - Y2e6bM;

  Y7f1aP3c5dPP = C_f * x7 + C_a * x1 + C_c * x3 + C_d * x5;
  Y7a1fM3d5cMP = C_a * x7 - C_f * x1 + C_d * x3 - C_c * x5;
  Y1c7dM3f5aPM = C_c * x1 - C_d * x7 - C_f * x3 - C_a * x5;
  Y1d7cP3a5fMM = C_d * x1 + C_c * x7 - C_a * x3 + C_f * x5;

  x0 = C_norm * (Y04P2b6ePP + Y7f1aP3c5dPP);
  x7 = C_norm * (Y04P2b6ePP - Y7f1aP3c5dPP);
  x4 = C_norm * (Y04P2b6ePM + Y7a1fM3d5cMP);
  x3 = C_norm * (Y04P2b6ePM - Y7a1fM3d5cMP);

  x1 = C_norm * (Y04M2e6bMP + Y1c7dM3f5aPM);
  x5 = C_norm * (Y04M2e6bMM - Y1d7cP3a5fMM);
  x2 = C_norm * (Y04M2e6bMM + Y1d7cP3a5fMM);
  x6 = C_norm * (Y04M2e6bMP - Y1c7dM3f5aPM);

  // Clamp the numbers to 0..255.
  Npp8u c0 = clampToUint8((int) rintf(x0+128.f)); 
  Npp8u c1 = clampToUint8((int) rintf(x1+128.f)); 
  Npp8u c2 = clampToUint8((int) rintf(x2+128.f)); 
  Npp8u c3 = clampToUint8((int) rintf(x3+128.f)); 
  Npp8u c4 = clampToUint8((int) rintf(x4+128.f)); 
  Npp8u c5 = clampToUint8((int) rintf(x5+128.f)); 
  Npp8u c6 = clampToUint8((int) rintf(x6+128.f)); 
  Npp8u c7 = clampToUint8((int) rintf(x7+128.f)); 

  // Pack into two unsigned int.
  uchar4 u0 = make_uchar4(c0, c1, c2, c3);
  uchar4 u1 = make_uchar4(c4, c5, c6, c7);

  // Each CTA works on 1 row (yBlockId) and 32 columns.
  const int row =  8*yBlockId + threadIdx.y;
  const int col = 32*xBlockId + threadIdx.x;

  // Each thread stores 8 pixels.
  uint2 packedBlock = make_uint2(reinterpret_cast<unsigned&>(u0), reinterpret_cast<unsigned&>(u1));
  if( col < params.numBlocksPerRow ) {
    params.img[row*params.imgNumBlocksPerStride + col] = packedBlock;
  }
}

template< typename QuantType, int READ_ZIGZAG >
__global__ __launch_bounds__(256, 6) void batchedInverseDct32x8Kernel(
  Npp16s **srcs, int *srcSteps, QuantType *quantTables, int2 *gridInfo,
  int *imgIdxs, Npp8u **dsts, NppiSize *dstDims) {
  // find my image and my block ids in the sub-grid for my image
  int imgIdx = imgIdxs[blockIdx.x];

  // Get input meta-data
  NppiSize dstDim = dstDims[imgIdx];
  int srcStep = srcSteps[imgIdx];
  
  // setup params
  int numBlocksPerRow = divUp(dstDim.width, 8);
  uint2 *img = (uint2*)dsts[imgIdx];
  int imgNumBlocksPerStride = numBlocksPerRow;
  int *dct = (int*)srcs[imgIdx];
  int dctNumBlocksPerStride = divUp(srcStep, (int)(64*sizeof(Npp16s)));
  QuantType *quantizationTable = quantTables + imgIdx * 64;
  
  int2 gInfo = gridInfo[imgIdx];
  int blockOffset = gInfo.y;
  
  int xBlockId = (blockIdx.x - blockOffset) % gInfo.x;
  int yBlockId = (blockIdx.x - blockOffset) / gInfo.x;


  // Shared memory to transpose blocks.
  __shared__ float smemBlock[8][256];

  // The number of blocks computed by this CTA.
  const int numBlocks = min(numBlocksPerRow - 32*xBlockId, 32);

  // We work on 32 blocks per CTA. Each block produces 64 values (64x16-bit i.e. 32x32-bit).
  int dctOffset = 32*yBlockId*dctNumBlocksPerStride + 1024*xBlockId;

  // The global offset for the block.
  for( int k = 0, i = threadIdx.y ; k < 4 ; ++k, i += 8 )
    if( i < numBlocks ) {
      reinterpret_cast<int*>(smemBlock)[33*i + threadIdx.x] = dct[dctOffset + 32*i + threadIdx.x];
    }
  __syncthreads();

  // Read the values from SMEM. We use a stride of 33 (66 because it's on 16-bit numbers) to avoid bank conflicts.
  float x0, x1, x2, x3, x4, x5, x6, x7;
  if( READ_ZIGZAG )
  {
    // Zigzag offsets.
    int4 zigzag0 = constZigzag[threadIdx.y + 0];
    int4 zigzag8 = constZigzag[threadIdx.y + 8];

    x0 = (float) reinterpret_cast<Npp16s*>(smemBlock)[66*threadIdx.x + zigzag0.x];
    x1 = (float) reinterpret_cast<Npp16s*>(smemBlock)[66*threadIdx.x + zigzag0.y];
    x2 = (float) reinterpret_cast<Npp16s*>(smemBlock)[66*threadIdx.x + zigzag0.z];
    x3 = (float) reinterpret_cast<Npp16s*>(smemBlock)[66*threadIdx.x + zigzag0.w];
    x4 = (float) reinterpret_cast<Npp16s*>(smemBlock)[66*threadIdx.x + zigzag8.x];
    x5 = (float) reinterpret_cast<Npp16s*>(smemBlock)[66*threadIdx.x + zigzag8.y];
    x6 = (float) reinterpret_cast<Npp16s*>(smemBlock)[66*threadIdx.x + zigzag8.z];
    x7 = (float) reinterpret_cast<Npp16s*>(smemBlock)[66*threadIdx.x + zigzag8.w];
  }
  else
  {
    x0 = (float) reinterpret_cast<Npp16s*>(smemBlock)[66*threadIdx.x + threadIdx.y +  0];
    x1 = (float) reinterpret_cast<Npp16s*>(smemBlock)[66*threadIdx.x + threadIdx.y +  8];
    x2 = (float) reinterpret_cast<Npp16s*>(smemBlock)[66*threadIdx.x + threadIdx.y + 16];
    x3 = (float) reinterpret_cast<Npp16s*>(smemBlock)[66*threadIdx.x + threadIdx.y + 24];
    x4 = (float) reinterpret_cast<Npp16s*>(smemBlock)[66*threadIdx.x + threadIdx.y + 32];
    x5 = (float) reinterpret_cast<Npp16s*>(smemBlock)[66*threadIdx.x + threadIdx.y + 40];
    x6 = (float) reinterpret_cast<Npp16s*>(smemBlock)[66*threadIdx.x + threadIdx.y + 48];
    x7 = (float) reinterpret_cast<Npp16s*>(smemBlock)[66*threadIdx.x + threadIdx.y + 56];
  }

  // Each warp loads the quantization values from the table.
  QuantType *qTable = (QuantType *) quantizationTable;
  QuantType q = qTable[threadIdx.x];

  // Convert to float.
  float q0 = (float) q.x;
  float q1 = (float) q.y;
    
  // Each thread quantize its values. 
  // Use sync version of shfl for Volta.
  x0 = x0 * __shfl_sync(0xffffffff, q0, threadIdx.y +  0);
  x1 = x1 * __shfl_sync(0xffffffff, q0, threadIdx.y +  8);
  x2 = x2 * __shfl_sync(0xffffffff, q0, threadIdx.y + 16);
  x3 = x3 * __shfl_sync(0xffffffff, q0, threadIdx.y + 24);
  x4 = x4 * __shfl_sync(0xffffffff, q1, threadIdx.y +  0);
  x5 = x5 * __shfl_sync(0xffffffff, q1, threadIdx.y +  8);
  x6 = x6 * __shfl_sync(0xffffffff, q1, threadIdx.y + 16);
  x7 = x7 * __shfl_sync(0xffffffff, q1, threadIdx.y + 24);

  // Run the Inverse DCT. Start with columns.
  float Y04P = x0 + x4;
  float Y04M = x0 - x4;

  float Y2b6eP = C_b * x2 + C_e * x6;
  float Y2e6bM = C_e * x2 - C_b * x6;

  float Y04P2b6ePP = Y04P + Y2b6eP;
  float Y04P2b6ePM = Y04P - Y2b6eP;
  float Y04M2e6bMP = Y04M + Y2e6bM;
  float Y04M2e6bMM = Y04M - Y2e6bM;

  float Y7f1aP3c5dPP = C_f * x7 + C_a * x1 + C_c * x3 + C_d * x5;
  float Y7a1fM3d5cMP = C_a * x7 - C_f * x1 + C_d * x3 - C_c * x5;
  float Y1c7dM3f5aPM = C_c * x1 - C_d * x7 - C_f * x3 - C_a * x5;
  float Y1d7cP3a5fMM = C_d * x1 + C_c * x7 - C_a * x3 + C_f * x5;

  x0 = C_norm * (Y04P2b6ePP + Y7f1aP3c5dPP);
  x7 = C_norm * (Y04P2b6ePP - Y7f1aP3c5dPP);
  x4 = C_norm * (Y04P2b6ePM + Y7a1fM3d5cMP);
  x3 = C_norm * (Y04P2b6ePM - Y7a1fM3d5cMP);

  x1 = C_norm * (Y04M2e6bMP + Y1c7dM3f5aPM);
  x5 = C_norm * (Y04M2e6bMM - Y1d7cP3a5fMM);
  x2 = C_norm * (Y04M2e6bMM + Y1d7cP3a5fMM);
  x6 = C_norm * (Y04M2e6bMP - Y1c7dM3f5aPM);

  // Make sure the threads have read their values from SMEM.
  __syncthreads();

  // Store to SMEM.
  smemBlock[threadIdx.y][threadIdx.x +   0] = x0;
  smemBlock[threadIdx.y][threadIdx.x +  32] = x1;
  smemBlock[threadIdx.y][threadIdx.x +  64] = x2;
  smemBlock[threadIdx.y][threadIdx.x +  96] = x3;
  smemBlock[threadIdx.y][threadIdx.x + 128] = x4;
  smemBlock[threadIdx.y][threadIdx.x + 160] = x5;
  smemBlock[threadIdx.y][threadIdx.x + 192] = x6;
  smemBlock[threadIdx.y][threadIdx.x + 224] = x7;

  // Make sure the threads have stored their values in SMEM.
  __syncthreads();

  // Extract the columns. Threads in a warp work on different image blocks.
  x0 = smemBlock[0][32*threadIdx.y + threadIdx.x];
  x1 = smemBlock[1][32*threadIdx.y + threadIdx.x];
  x2 = smemBlock[2][32*threadIdx.y + threadIdx.x];
  x3 = smemBlock[3][32*threadIdx.y + threadIdx.x];
  x4 = smemBlock[4][32*threadIdx.y + threadIdx.x];
  x5 = smemBlock[5][32*threadIdx.y + threadIdx.x];
  x6 = smemBlock[6][32*threadIdx.y + threadIdx.x];
  x7 = smemBlock[7][32*threadIdx.y + threadIdx.x];

  // Run the Inverse DCT. Start with columns.
  Y04P = x0 + x4;
  Y04M = x0 - x4;

  Y2b6eP = C_b * x2 + C_e * x6;
  Y2e6bM = C_e * x2 - C_b * x6;

  Y04P2b6ePP = Y04P + Y2b6eP;
  Y04P2b6ePM = Y04P - Y2b6eP;
  Y04M2e6bMP = Y04M + Y2e6bM;
  Y04M2e6bMM = Y04M - Y2e6bM;

  Y7f1aP3c5dPP = C_f * x7 + C_a * x1 + C_c * x3 + C_d * x5;
  Y7a1fM3d5cMP = C_a * x7 - C_f * x1 + C_d * x3 - C_c * x5;
  Y1c7dM3f5aPM = C_c * x1 - C_d * x7 - C_f * x3 - C_a * x5;
  Y1d7cP3a5fMM = C_d * x1 + C_c * x7 - C_a * x3 + C_f * x5;

  x0 = C_norm * (Y04P2b6ePP + Y7f1aP3c5dPP);
  x7 = C_norm * (Y04P2b6ePP - Y7f1aP3c5dPP);
  x4 = C_norm * (Y04P2b6ePM + Y7a1fM3d5cMP);
  x3 = C_norm * (Y04P2b6ePM - Y7a1fM3d5cMP);

  x1 = C_norm * (Y04M2e6bMP + Y1c7dM3f5aPM);
  x5 = C_norm * (Y04M2e6bMM - Y1d7cP3a5fMM);
  x2 = C_norm * (Y04M2e6bMM + Y1d7cP3a5fMM);
  x6 = C_norm * (Y04M2e6bMP - Y1c7dM3f5aPM);

  // Clamp the numbers to 0..255.
  Npp8u c0 = clampToUint8((int) rintf(x0+128.f)); 
  Npp8u c1 = clampToUint8((int) rintf(x1+128.f)); 
  Npp8u c2 = clampToUint8((int) rintf(x2+128.f)); 
  Npp8u c3 = clampToUint8((int) rintf(x3+128.f)); 
  Npp8u c4 = clampToUint8((int) rintf(x4+128.f)); 
  Npp8u c5 = clampToUint8((int) rintf(x5+128.f)); 
  Npp8u c6 = clampToUint8((int) rintf(x6+128.f)); 
  Npp8u c7 = clampToUint8((int) rintf(x7+128.f)); 

  // Pack into two unsigned int.
  uchar4 u0 = make_uchar4(c0, c1, c2, c3);
  uchar4 u1 = make_uchar4(c4, c5, c6, c7);

  // Each CTA works on 1 row (yBlockId) and 32 columns.
  const int row =  8*yBlockId + threadIdx.y;
  const int col = 32*xBlockId + threadIdx.x;

  // Each thread stores 8 pixels.
  uint2 packedBlock = make_uint2(reinterpret_cast<unsigned&>(u0), reinterpret_cast<unsigned&>(u1));
  if( col < numBlocksPerRow ) {
    img[row*imgNumBlocksPerStride + col] = packedBlock;
  }
}

template< typename QuantType, int READ_ZIGZAG >
__global__ __launch_bounds__(256, 6) void batchedInverseDct32x8Kernel(
  DctQuantInvImageParam *params, QuantType *quantTables, int *imgIdxs) {
  // find my image and my block ids in the sub-grid for my image
  int imgIdx = imgIdxs[blockIdx.x];

  // Get input meta-data
  DctQuantInvImageParam param = params[imgIdx];
  
  // setup params
  int numBlocksPerRow = divUp(param.dstWidth, 8);
  uint2 *img = (uint2*)param.dst;
  int imgNumBlocksPerStride = numBlocksPerRow;
  
  int *dct = (int*)param.src;
  int dctNumBlocksPerStride = divUp(param.srcStep, (int)(64*sizeof(Npp16s)));
  QuantType *quantizationTable = (QuantType*)((Npp8u*)quantTables + imgIdx * 64);
  
  int blockOffset = param.gridInfo.y;
  int xBlockId = (blockIdx.x - blockOffset) % param.gridInfo.x;
  int yBlockId = (blockIdx.x - blockOffset) / param.gridInfo.x;

  // if (xBlockId == 0 && yBlockId == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
  //   printf("comp/outptr: %d / %lld\n", imgIdx, (long long)img);
  // }
  
  // Shared memory to transpose blocks.
  __shared__ float smemBlock[8][256];

  // The number of blocks computed by this CTA.
  const int numBlocks = min(numBlocksPerRow - 32*xBlockId, 32);

  // We work on 32 blocks per CTA. Each block produces 64 values (64x16-bit i.e. 32x32-bit).
  int dctOffset = 32*yBlockId*dctNumBlocksPerStride + 1024*xBlockId;

  // The global offset for the block.
  for( int k = 0, i = threadIdx.y ; k < 4 ; ++k, i += 8 )
    if( i < numBlocks ) {
      // DEBUG
      if (threadIdx.x == 0 && threadIdx.y == 4 && blockIdx.x == 409) {
        printf("accessing dct ptr (%lld) at offset %d", dct, dctOffset+32*i+threadIdx.x*sizeof(int));
      }
      
      reinterpret_cast<int*>(smemBlock)[33*i + threadIdx.x] = dct[dctOffset + 32*i + threadIdx.x];
    }
  __syncthreads();

  // Read the values from SMEM. We use a stride of 33 (66 because it's on 16-bit numbers) to avoid bank conflicts.
  float x0, x1, x2, x3, x4, x5, x6, x7;
  if( READ_ZIGZAG )
  {
    // Zigzag offsets.
    int4 zigzag0 = constZigzag[threadIdx.y + 0];
    int4 zigzag8 = constZigzag[threadIdx.y + 8];

    x0 = (float) reinterpret_cast<Npp16s*>(smemBlock)[66*threadIdx.x + zigzag0.x];
    x1 = (float) reinterpret_cast<Npp16s*>(smemBlock)[66*threadIdx.x + zigzag0.y];
    x2 = (float) reinterpret_cast<Npp16s*>(smemBlock)[66*threadIdx.x + zigzag0.z];
    x3 = (float) reinterpret_cast<Npp16s*>(smemBlock)[66*threadIdx.x + zigzag0.w];
    x4 = (float) reinterpret_cast<Npp16s*>(smemBlock)[66*threadIdx.x + zigzag8.x];
    x5 = (float) reinterpret_cast<Npp16s*>(smemBlock)[66*threadIdx.x + zigzag8.y];
    x6 = (float) reinterpret_cast<Npp16s*>(smemBlock)[66*threadIdx.x + zigzag8.z];
    x7 = (float) reinterpret_cast<Npp16s*>(smemBlock)[66*threadIdx.x + zigzag8.w];
  }
  else
  {
    x0 = (float) reinterpret_cast<Npp16s*>(smemBlock)[66*threadIdx.x + threadIdx.y +  0];
    x1 = (float) reinterpret_cast<Npp16s*>(smemBlock)[66*threadIdx.x + threadIdx.y +  8];
    x2 = (float) reinterpret_cast<Npp16s*>(smemBlock)[66*threadIdx.x + threadIdx.y + 16];
    x3 = (float) reinterpret_cast<Npp16s*>(smemBlock)[66*threadIdx.x + threadIdx.y + 24];
    x4 = (float) reinterpret_cast<Npp16s*>(smemBlock)[66*threadIdx.x + threadIdx.y + 32];
    x5 = (float) reinterpret_cast<Npp16s*>(smemBlock)[66*threadIdx.x + threadIdx.y + 40];
    x6 = (float) reinterpret_cast<Npp16s*>(smemBlock)[66*threadIdx.x + threadIdx.y + 48];
    x7 = (float) reinterpret_cast<Npp16s*>(smemBlock)[66*threadIdx.x + threadIdx.y + 56];
  }

  // Each warp loads the quantization values from the table.
  QuantType *qTable = (QuantType *) quantizationTable;
  QuantType q = qTable[threadIdx.x];

  // Convert to float.
  float q0 = (float) q.x;
  float q1 = (float) q.y;
    
  // Each thread quantize its values. 
  // Use sync version of shfl for Volta.
  x0 = x0 * __shfl_sync(0xffffffff, q0, threadIdx.y +  0);
  x1 = x1 * __shfl_sync(0xffffffff, q0, threadIdx.y +  8);
  x2 = x2 * __shfl_sync(0xffffffff, q0, threadIdx.y + 16);
  x3 = x3 * __shfl_sync(0xffffffff, q0, threadIdx.y + 24);
  x4 = x4 * __shfl_sync(0xffffffff, q1, threadIdx.y +  0);
  x5 = x5 * __shfl_sync(0xffffffff, q1, threadIdx.y +  8);
  x6 = x6 * __shfl_sync(0xffffffff, q1, threadIdx.y + 16);
  x7 = x7 * __shfl_sync(0xffffffff, q1, threadIdx.y + 24);

  // Run the Inverse DCT. Start with columns.
  float Y04P = x0 + x4;
  float Y04M = x0 - x4;

  float Y2b6eP = C_b * x2 + C_e * x6;
  float Y2e6bM = C_e * x2 - C_b * x6;

  float Y04P2b6ePP = Y04P + Y2b6eP;
  float Y04P2b6ePM = Y04P - Y2b6eP;
  float Y04M2e6bMP = Y04M + Y2e6bM;
  float Y04M2e6bMM = Y04M - Y2e6bM;

  float Y7f1aP3c5dPP = C_f * x7 + C_a * x1 + C_c * x3 + C_d * x5;
  float Y7a1fM3d5cMP = C_a * x7 - C_f * x1 + C_d * x3 - C_c * x5;
  float Y1c7dM3f5aPM = C_c * x1 - C_d * x7 - C_f * x3 - C_a * x5;
  float Y1d7cP3a5fMM = C_d * x1 + C_c * x7 - C_a * x3 + C_f * x5;

  x0 = C_norm * (Y04P2b6ePP + Y7f1aP3c5dPP);
  x7 = C_norm * (Y04P2b6ePP - Y7f1aP3c5dPP);
  x4 = C_norm * (Y04P2b6ePM + Y7a1fM3d5cMP);
  x3 = C_norm * (Y04P2b6ePM - Y7a1fM3d5cMP);

  x1 = C_norm * (Y04M2e6bMP + Y1c7dM3f5aPM);
  x5 = C_norm * (Y04M2e6bMM - Y1d7cP3a5fMM);
  x2 = C_norm * (Y04M2e6bMM + Y1d7cP3a5fMM);
  x6 = C_norm * (Y04M2e6bMP - Y1c7dM3f5aPM);

  // Make sure the threads have read their values from SMEM.
  __syncthreads();

  // Store to SMEM.
  smemBlock[threadIdx.y][threadIdx.x +   0] = x0;
  smemBlock[threadIdx.y][threadIdx.x +  32] = x1;
  smemBlock[threadIdx.y][threadIdx.x +  64] = x2;
  smemBlock[threadIdx.y][threadIdx.x +  96] = x3;
  smemBlock[threadIdx.y][threadIdx.x + 128] = x4;
  smemBlock[threadIdx.y][threadIdx.x + 160] = x5;
  smemBlock[threadIdx.y][threadIdx.x + 192] = x6;
  smemBlock[threadIdx.y][threadIdx.x + 224] = x7;

  // Make sure the threads have stored their values in SMEM.
  __syncthreads();

  // Extract the columns. Threads in a warp work on different image blocks.
  x0 = smemBlock[0][32*threadIdx.y + threadIdx.x];
  x1 = smemBlock[1][32*threadIdx.y + threadIdx.x];
  x2 = smemBlock[2][32*threadIdx.y + threadIdx.x];
  x3 = smemBlock[3][32*threadIdx.y + threadIdx.x];
  x4 = smemBlock[4][32*threadIdx.y + threadIdx.x];
  x5 = smemBlock[5][32*threadIdx.y + threadIdx.x];
  x6 = smemBlock[6][32*threadIdx.y + threadIdx.x];
  x7 = smemBlock[7][32*threadIdx.y + threadIdx.x];

  // Run the Inverse DCT. Start with columns.
  Y04P = x0 + x4;
  Y04M = x0 - x4;

  Y2b6eP = C_b * x2 + C_e * x6;
  Y2e6bM = C_e * x2 - C_b * x6;

  Y04P2b6ePP = Y04P + Y2b6eP;
  Y04P2b6ePM = Y04P - Y2b6eP;
  Y04M2e6bMP = Y04M + Y2e6bM;
  Y04M2e6bMM = Y04M - Y2e6bM;

  Y7f1aP3c5dPP = C_f * x7 + C_a * x1 + C_c * x3 + C_d * x5;
  Y7a1fM3d5cMP = C_a * x7 - C_f * x1 + C_d * x3 - C_c * x5;
  Y1c7dM3f5aPM = C_c * x1 - C_d * x7 - C_f * x3 - C_a * x5;
  Y1d7cP3a5fMM = C_d * x1 + C_c * x7 - C_a * x3 + C_f * x5;

  x0 = C_norm * (Y04P2b6ePP + Y7f1aP3c5dPP);
  x7 = C_norm * (Y04P2b6ePP - Y7f1aP3c5dPP);
  x4 = C_norm * (Y04P2b6ePM + Y7a1fM3d5cMP);
  x3 = C_norm * (Y04P2b6ePM - Y7a1fM3d5cMP);

  x1 = C_norm * (Y04M2e6bMP + Y1c7dM3f5aPM);
  x5 = C_norm * (Y04M2e6bMM - Y1d7cP3a5fMM);
  x2 = C_norm * (Y04M2e6bMM + Y1d7cP3a5fMM);
  x6 = C_norm * (Y04M2e6bMP - Y1c7dM3f5aPM);

  // Clamp the numbers to 0..255.
  Npp8u c0 = clampToUint8((int) rintf(x0+128.f)); 
  Npp8u c1 = clampToUint8((int) rintf(x1+128.f)); 
  Npp8u c2 = clampToUint8((int) rintf(x2+128.f)); 
  Npp8u c3 = clampToUint8((int) rintf(x3+128.f)); 
  Npp8u c4 = clampToUint8((int) rintf(x4+128.f)); 
  Npp8u c5 = clampToUint8((int) rintf(x5+128.f)); 
  Npp8u c6 = clampToUint8((int) rintf(x6+128.f)); 
  Npp8u c7 = clampToUint8((int) rintf(x7+128.f)); 

  // Pack into two unsigned int.
  uchar4 u0 = make_uchar4(c0, c1, c2, c3);
  uchar4 u1 = make_uchar4(c4, c5, c6, c7);

  // Each CTA works on 1 row (yBlockId) and 32 columns.
  const int row =  8*yBlockId + threadIdx.y;
  const int col = 32*xBlockId + threadIdx.x;

  // Each thread stores 8 pixels.
  uint2 packedBlock = make_uint2(reinterpret_cast<unsigned&>(u0), reinterpret_cast<unsigned&>(u1));
  if( col < numBlocksPerRow ) {
    img[row*imgNumBlocksPerStride + col] = packedBlock;
  }
}

NppStatus batchedDCTQuantInv8x8xLS_JPEG_16s8u_C1R_NEW(DctParams *dParams,
        int2 *dGridInfo, int *dImgIdxs, int numBlocks) {
    batchedInverseDct32x8Kernel<uchar2, 1> <<<numBlocks, dim3(32, 8), 0, nppGetStream()>>>(dParams,
            dGridInfo, dImgIdxs);
    return NPP_SUCCESS;
}

NppStatus batchedDCTQuantInv8x8xLS_JPEG_16s8u_C1R_NEW(Npp16s **srcs, int *srcSteps, void *quantTables, int2 *gridInfo, int *imgIdxs, Npp8u **dsts, NppiSize *dstDims, int numBlocks) {
  batchedInverseDct32x8Kernel<uchar2, 1><<<numBlocks, dim3(32, 8), 0, nppGetStream()>>>(
    srcs, srcSteps, (uchar2*)quantTables, gridInfo, imgIdxs, dsts, dstDims);
  return NPP_SUCCESS;
}

NppStatus batchedDCTQuantInv8x8xLS_JPEG_16s8u_C1R_NEW(DctQuantInvImageParam *params,
  void *quantTables, int *imgIdxs, int numBlocks) {
  batchedInverseDct32x8Kernel<uchar2, 1><<<numBlocks, dim3(32, 8), 0, nppGetStream()>>>(
    params, (uchar2*)quantTables, imgIdxs);
  return NPP_SUCCESS;
}
