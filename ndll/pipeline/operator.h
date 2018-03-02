// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#ifndef NDLL_PIPELINE_OPERATOR_H_
#define NDLL_PIPELINE_OPERATOR_H_

#include <string>

#include "ndll/common.h"
#include "ndll/error_handling.h"
#include "ndll/pipeline/workspace/device_workspace.h"
#include "ndll/pipeline/ndll.pb.h"
#include "ndll/pipeline/data/backend.h"
#include "ndll/pipeline/operator_factory.h"
#include "ndll/pipeline/op_schema.h"
#include "ndll/pipeline/op_spec.h"
#include "ndll/pipeline/workspace/sample_workspace.h"

namespace ndll {

enum NDLLOpType {
  NDLL_GPU = 0,
  NDLL_CPU = 1,
  NDLL_MIXED = 2
};

/**
 * @brief Baseclass for the basic unit of computation in the pipeline.
 *
 * Operator defines the API used by the pipeline to execute operations.
 * To create a custom operator, derive from this class, implement the
 * RunPerSampleCPU / RunBatchedGPU methods as desired, and register
 * the operator using the macros NDLL_REGISTER_{CPU,GPU}_OPERATOR.
 * To define meta-data about the op like the min/max number of inputs
 * it takes, a doctstring (for python), etc., use the NDLL_OPERATOR_SCHEMA,
 * macro. The op can then be added to a pipeline through its registered
 * name (the first arg to the registration macros).
 */
class Operator {
 public:
  inline explicit Operator(const OpSpec &spec) :
    spec_(spec), num_threads_(spec.GetArgument<int>("num_threads")),
    batch_size_(spec.GetArgument<int>("batch_size")),
    input_sets_(spec.GetArgument<int>("num_input_sets")) {
    NDLL_ENFORCE(num_threads_ > 0, "Invalid value for argument num_threads.");
    NDLL_ENFORCE(batch_size_ > 0, "Invalid value for argument batch_size.");
  }

  virtual inline ~Operator() = default;

  /**
   * @brief Executes the operator on a single sample on the CPU.
   */
  inline virtual void Run(SampleWorkspace *ws) {
#ifndef NDEBUG
    NDLL_ENFORCE_VALID_INDEX(ws->thread_idx(), num_threads_);
    NDLL_ENFORCE_VALID_INDEX(ws->data_idx(), batch_size_);
#endif

    SetupSharedSampleParams(ws);

    for (int i = 0; i < input_sets_; ++i) {
      RunPerSampleCPU(ws, i);
    }
  }

  /**
   * @brief Executes the operator on a batch of samples on the GPU.
   */
  inline virtual void Run(DeviceWorkspace *ws) {
    SetupSharedSampleParams(ws);

    for (int i = 0; i < input_sets_; ++i) {
      RunBatchedGPU(ws, i);
    }
  }

  /**
   * @brief Used by operators interfacing with both CPU and GPU.
   */
  virtual void Run(MixedWorkspace *ws) {
    NDLL_FAIL("Run using mixed workspace is not implemented for this operator!");
  }

  /**
   * @brief returns the name of the operator. By default returns
   * the name of the op as specified by the OpSpec it was constructed
   * from.
   */
  virtual string name() const {
    return spec_.name();
  }

  /**
   * @brief For reader Ops, returns the size of the dataset
   * For all other Ops, returns -1
   */
  virtual Index epoch_size() const {
    return -1;
  }

  int GetNumInputSets() const {
    return input_sets_;
  }

  DISABLE_COPY_MOVE_ASSIGN(Operator);

 protected:
  /**
   * @brief Per image CPU computation of the operator to be
   * implemented by derived ops.
   */
  virtual inline void RunPerSampleCPU(SampleWorkspace *ws, int idx = 0) {
    NDLL_FAIL("RunPerSampleCPU not implemented");
  }

  /**
   * @brief Batched GPU computation of the operator to be
   * implemented by derived ops.
   */
  virtual inline void RunBatchedGPU(DeviceWorkspace *ws, int idx = 0) {
    NDLL_FAIL("RunBatchedGPU not implemented");
  }

  /**
   * @brief Shared param setup for CPU computation
   */
  virtual inline void SetupSharedSampleParams(SampleWorkspace *ws) {}

  /**
   * @brief Shared param setup for GPU computation
   */
  virtual inline void SetupSharedSampleParams(DeviceWorkspace *ws) {}

  OpSpec spec_;
  int num_threads_;
  int batch_size_;
  int input_sets_;
};

#define USE_OPERATOR_MEMBERS()                  \
  using Operator::spec_;               \
  using Operator::num_threads_;        \
  using Operator::batch_size_

// Create registries for CPU & GPU Operators
NDLL_DECLARE_OPTYPE_REGISTRY(CPUOperator, Operator);
NDLL_DECLARE_OPTYPE_REGISTRY(GPUOperator, Operator);
NDLL_DECLARE_OPTYPE_REGISTRY(MixedOperator, Operator);

// Must be called from .cc or .cu file
#define NDLL_REGISTER_OPERATOR(OpName, OpType, device)        \
  int NDLL_OPERATOR_SCHEMA_REQUIRED_FOR_##OpName();            \
  static int ANONYMIZE_VARIABLE(OpName) =                 \
    NDLL_OPERATOR_SCHEMA_REQUIRED_FOR_##OpName();              \
  NDLL_DEFINE_OPTYPE_REGISTERER(OpName, OpType,           \
      device##Operator, ndll::Operator)

  class ResizeAttr;
  void DataDependentSetupCPU(const Tensor<CPUBackend> &input, Tensor<CPUBackend> *output,
              const char *pOpName = NULL,
              vector<const uint8 *> *inPtrs = NULL, vector<uint8 *> *outPtrs = NULL,
              vector<NDLLSize> *pSizes = NULL, const NDLLSize *out_size = NULL);
  void DataDependentSetupGPU(const TensorList<GPUBackend> &input, TensorList<GPUBackend> *output,
              size_t batch_size, bool reshapeBatch = false,
              vector<const uint8 *> *iPtrs = NULL, vector<uint8 *> *oPtrs = NULL,
              vector<NDLLSize> *pSizes = NULL, ResizeAttr *pntr = NULL, NDLLSize *pResize = NULL);
  void CollectPointersForExecution(size_t batch_size,
              const TensorList<GPUBackend> &input, vector<const uint8 *> *inPtrs,
              TensorList<GPUBackend> *output, vector<uint8 *> *outPtrs);
}  // namespace ndll


// Macros for  creation of the CPU/GPU augmentation methods:

#define AUGMENT_TRANSFORM(H, W, C, img_in, img_out,         \
            AUGMENT_PREAMBLE, AUGMENT_CORE, AUGMENT_DEF,    \
            stepW, stepH, startW, startH, imgIdx, ...)      \
    AUGMENT_PREAMBLE(H, W, C, __VA_ARGS__);                 \
    const uint32_t offset = nYoffset(W, C);                 \
    const uint32_t stride = H * offset * imgIdx;            \
    const uint32_t shift = stepH * offset;                  \
    const uint8 *in = img_in + stride;                      \
    uint8 *out = img_out + stride + startH * offset - shift;\
    int newX, newY, useNew, from;                           \
    for (int y = startH; y < H; y += stepH) {               \
        out += shift;                                       \
        for (int x = startW; x < W; x += stepW) {           \
            AUGMENT_CORE(H, W, C, __VA_ARGS__);             \
            from = useNew?  newX * C + newY * offset :      \
                            AUGMENT_DEF(x, y, W, C);        \
            const int to = x * C;                           \
            out[to] = in[from];                             \
            if (C > 1) {                                    \
                out[to + 1] = in[from + 1];                 \
                out[to + 2] = in[from + 2];                 \
            }                                               \
        }                                                   \
    }

#define AUGMENT_RESIZE(H, W, C, img_in, img_out,                    \
            AUGMENT_PREAMBLE, AUGMENT_CORE,                         \
            stepW, stepH, startW, startH, imgIdx, ...)              \
    AUGMENT_PREAMBLE(H, W, C);                                      \
    const uint32_t offset = nYoffset(W, C);                         \
    const uint32_t shift = stepH * offset;                          \
    const uint8 *in = img_in + H0 *nYoffset(W0, C) * imgIdx;        \
    uint8 *out = img_out + (H * imgIdx + startH) * offset - shift;  \
    for (int y = startH; y < H; y += stepH) {                       \
        out += shift;                                               \
        for (int x = startW; x < W; x += stepW) {                   \
            AUGMENT_CORE(C);                                        \
        }                                                           \
    }

#define AUGMENT_RESIZE_CPU(H, W, C, img_in, img_out, KIND)          \
        AUGMENT_RESIZE(H, W, C, img_in, img_out, KIND ## _PREAMBLE, \
        KIND ## _CORE, 1, 1, 0, 0, 0)

#define AUGMENT_RESIZE_GPU(H, W, C, img_in, img_out, KIND)          \
        AUGMENT_RESIZE(H, W, C, img_in, img_out, KIND ## _PREAMBLE, \
        KIND ## _CORE, blockDim.x, blockDim.y, threadIdx.x, threadIdx.y, blockIdx.x)


#define AUGMENT_RESIZE_CROP(H, W, C, img_in, img_out,               \
            AUGMENT_PREAMBLE, AUGMENT_CORE,                         \
            stepW, stepH, startW, startH, imgIdx, ...)              \
    AUGMENT_PREAMBLE(H, W, C);                                      \
    const uint32_t offset = nYoffset(W, C);                         \
    const uint32_t shift = stepH * offset;                          \
    const uint8 *in = img_in + H0 *nYoffset(W0, C) * imgIdx;        \
    uint8 *out = img_out + (H * imgIdx + startH) * offset - shift;  \
    for (int y = startH; y < H; y += stepH) {                       \
        out += shift;                                               \
        for (int x = startW; x < W; x += stepW) {                   \
            AUGMENT_CORE(C);                                        \
        }                                                           \
    }

#define AUGMENT_RESIZE__CROP_CPU(H, W, C, img_in, img_out, KIND)    \
        AUGMENT_RESIZE(H, W, C, img_in, img_out, KIND ## _PREAMBLE, \
        KIND ## _CORE, 1, 1, 0, 0, 0)

#define AUGMENT_RESIZE_CROP_GPU(H, W, C, img_in, img_out, KIND)     \
        AUGMENT_RESIZE(H, W, C, img_in, img_out, KIND ## _PREAMBLE, \
        KIND ## _CORE, blockDim.x, blockDim.y, threadIdx.x, threadIdx.y, blockIdx.x)


#define AUGMENT_TRANSFORM_N(H, W, C, img_in, img_out,       \
            AUGMENT_PREAMBLE, AUGMENT_CORE, AUGMENT_DEF,    \
            stepW, stepH, startW, startH, imgIdx, ...)      \
    AUGMENT_PREAMBLE(H, W, C, __VA_ARGS__);                 \
    const uint32_t offset = nYoffset(W, C);                 \
    const uint32_t stride = H * offset * imgIdx;            \
    const uint32_t shift = stepH * offset;                  \
    const uint8 *in = img_in + stride;                      \
    const int startY = START_COORD(startH, H >> 1, stepH);  \
    const int startX = START_COORD(startW, W >> 1, stepW);  \
    bool useVertSymmetry = !(H & 1) || 2 * startY > H;      \
    const bool useHor = !(W & 1) || 2 * startX > W;         \
    const int adjY = H * offset;                            \
    const int adjXto = offset - C;                          \
    const int stepAdj = 2 * stepH * offset;                 \
    int adjYto = adjY - (2 * startY + 1) * offset + stepAdj;\
    uint8 *out = img_out + stride + startY * offset - shift;\
    int newX, newY, useNew, from1, from2, from3, from4;     \
    for (int y = startY; y < H; y += stepH) {               \
        out += shift;                                       \
        /* VertSymmetry could be used only when */          \
        /* y is not in the middle               */          \
        adjYto -= stepAdj;                                  \
        bool useHorSymmetry = useHor;                       \
        for (int x = startX; x < W; x += stepW) {           \
            AUGMENT_CORE(H, W, C, __VA_ARGS__);             \
            /* The order of points 1,2,3,4 is clockwise  */ \
            /* starting from point 1, which is (x,y)     */ \
            if (useNew) {                                   \
                newX *= C;                                  \
                newY *= offset;                             \
                from1 = newX + newY;                        \
                from2 = offset - newX + newY;               \
                from3 = offset - newX + adjY - newY;        \
                from4 = newX + adjY - newY;                 \
            } else {                                        \
                from1 = DEF_ZERO(x, y, W, C);               \
                from2 = DEF_ZERO(W - x, y, W, C);           \
                from3 = DEF_ZERO(W - x, H - y, W, C);       \
                from4 = DEF_ZERO(x, H - y, W, C);           \
            }                                               \
                                                            \
            int to1 = x * C;                                \
            out[to1] = in[from1];                           \
            if (C > 1) {                                    \
                out[to1 + 1] = in[from1 + 1];               \
                out[to1 + 2] = in[from1 + 2];               \
            }                                               \
                                                            \
            if (!useHorSymmetry) {                          \
                if (useVertSymmetry) {                      \
                    /* Only point #4 should be set. */      \
                    /* Because to4 = to1 + adjYto;  */      \
                    /* we will use t1 instead of t4 */      \
                    out[to1 += adjYto] = in[from4];         \
                    if (C > 1) {                            \
                        out[to1 + 1] = in[from4 + 1];       \
                        out[to1 + 2] = in[from4 + 2];       \
                    }                                       \
                }                                           \
                useHorSymmetry = true;                      \
                continue;                                   \
            }                                               \
                                                            \
            const int to2 = adjXto - to1;                   \
            if (!useVertSymmetry) {                         \
                /* Only point #2 should be set */           \
                out[to2] = in[from2];                       \
                if (C > 1) {                                \
                    out[to2 + 1] = in[from2 + 1];           \
                    out[to2 + 2] = in[from2 + 2];           \
                }                                           \
                continue;                                   \
            }                                               \
                                                            \
            /* Points # 2, 3 and 4 should be set */         \
            const int to3 = to2 + adjYto;                   \
            out[to2] = in[from2];                           \
            out[to3] = in[from3];                           \
            out[to1 += to3 - to2] = in[from4];              \
            if (C > 1) {                                    \
                out[to2 + 1] = in[from2 + 1];               \
                out[to2 + 2] = in[from2 + 2];               \
                out[to3 + 1] = in[from3 + 1];               \
                out[to3 + 2] = in[from3 + 2];               \
                out[to1 + 1] = in[from4 + 1];               \
                out[to1 + 2] = in[from4 + 2];               \
            }                                               \
        }                                                   \
        useVertSymmetry = true;                             \
    }

#define DEF_CORE(x, y, W, C)       (y * W + x) * C      // identical
#define DEF_ZERO(x, y, W, C)        0                   // zero idx

#define AUGMENT_TRANSFORM_CPU(H, W, C, img_in, img_out, KIND, def, ...) \
        AUGMENT_TRANSFORM(H, W, C, img_in, img_out, KIND ## _PREAMBLE,  \
        KIND ## _CORE, def, 1, 1, 0, 0, 0, __VA_ARGS__)

#define AUGMENT_TRANSFORM_GPU(H, W, C, img_in, img_out, KIND, def, ...) \
        AUGMENT_TRANSFORM(H, W, C, img_in, img_out, KIND ## _PREAMBLE,  \
        KIND ## _CORE, def, blockDim.x, blockDim.y,                     \
        threadIdx.x, threadIdx.y, blockIdx.x, __VA_ARGS__)

#define AUGMENT_TRANSFORM_CPU_N(H, W, C, img_in, img_out, KIND, def, ...) \
        AUGMENT_TRANSFORM_N(H, W, C, img_in, img_out, KIND ## _PREAMBLE,  \
        KIND ## _CORE, def, 1, 1, 0, 0, 0, __VA_ARGS__)

#define AUGMENT_TRANSFORM_GPU_N(H, W, C, img_in, img_out, KIND, def, ...) \
        AUGMENT_TRANSFORM_N(H, W, C, img_in, img_out, KIND ## _PREAMBLE,  \
        KIND ## _CORE, def, blockDim.x, blockDim.y,                       \
        threadIdx.x, threadIdx.y, blockIdx.x, __VA_ARGS__)

#define DEF_PREAMBLE(H, W, C)                           // empty macro

#define START_COORD(a, b, s)    (s == 1)? (b) : (a >= (b)? a : a + ((b) - a + s - 1) / (s) * (s))
#define nYoffset(W, C)          ((W) * (C))
#define LRINT(x)                lrint(x)
#define PI                      3.14159265359

typedef enum {
    coord_X,
    coord_Y
} coordID;

#endif  // NDLL_PIPELINE_OPERATOR_H_
