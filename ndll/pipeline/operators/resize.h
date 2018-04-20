// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#ifndef NDLL_PIPELINE_OPERATORS_RESIZE_H_
#define NDLL_PIPELINE_OPERATORS_RESIZE_H_

#include <random>
#include <utility>
#include <vector>

#include "ndll/common.h"
#include "ndll/error_handling.h"
#include "ndll/image/transform.h"
#include "ndll/pipeline/operator.h"

namespace ndll {

typedef enum {
    input_t,
    output_t
} io_type;

typedef std::pair<int, int> resize_t;

class ResizeAttr;
typedef NppiPoint MirroringInfo;

class ResizeParamDescr {
public:
    ResizeParamDescr(ResizeAttr *pntr, NppiPoint *pOutResize = NULL, MirroringInfo *pMirror = NULL,
                        size_t pTotalSize[] = NULL, size_t batchSliceNumb = 0) :
                        pResize_(pntr), pResizeParam_(pOutResize), pMirroring_(pMirror),
                        pTotalSize_(pTotalSize), nBatchSlice_(batchSliceNumb) {}
    ResizeAttr *pResize_;
    NppiPoint *pResizeParam_;
    MirroringInfo *pMirroring_;
    size_t *pTotalSize_;
    size_t nBatchSlice_;
};

void DataDependentSetupCPU(const Tensor<CPUBackend> &input, Tensor<CPUBackend> *output,
                           const char *pOpName = NULL,
                           const uint8 **pInRaster = NULL, uint8 **ppOutRaster = NULL,
                           vector<NDLLSize> *pSizes = NULL, const NDLLSize *out_size = NULL);
bool DataDependentSetupGPU(const TensorList<GPUBackend> &input, TensorList<GPUBackend> *output,
          size_t batch_size, bool reshapeBatch = false,
          vector<const uint8 *> *iPtrs = NULL, vector<uint8 *> *oPtrs = NULL,
          vector<NDLLSize> *pSizes = NULL, ResizeParamDescr *pResizeParam = NULL);
void CollectPointersForExecution(size_t batch_size,
          const TensorList<GPUBackend> &input, vector<const uint8 *> *inPtrs,
          TensorList<GPUBackend> *output, vector<uint8 *> *outPtrs);

template <typename T>
void GetSingleOrDoubleArg(const OpSpec &spec, vector<T> *arg, const char *argName, T defVal, bool doubleArg = true) {
    try {
        *arg = spec.GetRepeatedArgument<T>(argName);
    } catch (std::runtime_error e) {
        try {
            *arg = {spec.GetArgument<T>(argName, defVal)};
        } catch (std::runtime_error e) {
            NDLL_FAIL("Invalid type of argument \"" + argName + "\"");
        }
    }

    if (doubleArg && arg->size() == 1)
        arg->push_back(arg->back());
}

class ResizeAttr {
 public:
    explicit inline ResizeAttr(const OpSpec &spec) :
            rand_gen_(time(nullptr)),
            random_resize_(spec.GetArgument<bool>("random_resize", false)),
            warp_resize_(spec.GetArgument<bool>("warp_resize", false)),
            image_type_(spec.GetArgument<NDLLImageType>("image_type", NDLL_RGB)),
            color_(IsColor(image_type_)), C_(color_ ? 3 : 1),
            random_crop_(spec.GetArgument<bool>("random_crop", false)),
            crop_h_(spec.GetArgument<int>("crop_h", -1)),
            crop_w_(spec.GetArgument<int>("crop_w", -1)),
            type_(spec.GetArgument<NDLLInterpType>("interp_type", NDLL_INTERP_LINEAR)) {
        resize_.first = spec.GetArgument<int>("resize_a", -1);
        resize_.second = spec.GetArgument<int>("resize_b", -1);

        GetSingleOrDoubleArg(spec, &mirror_prob_, "mirror_prob", 0.f, false);

        // Validate input parameters
        NDLL_ENFORCE(resize_.first > 0 && resize_.second > 0);
        NDLL_ENFORCE(resize_.first <= resize_.second);

        size_t i = mirror_prob_.size();
        NDLL_ENFORCE(i <= 2, "Argument \"mirror_prob\" expects a list of at most 2 elements, "
                     + to_string(i) + " given.");
        while (i--)
            NDLL_ENFORCE(mirror_prob_[i] <= 1.f && mirror_prob_[i] >= 0.f);
    }

    void SetSize(NDLLSize *in_size, const vector<Index> &shape,
                 const resize_t &rand, NDLLSize *out_size) const;

    inline vector<NDLLSize> &sizes(io_type type)            { return sizes_[type]; }
    inline NDLLSize *size(io_type type, size_t idx)         { return sizes(type).data() + idx; }
    inline const resize_t &newSizes(size_t idx) const       { return per_sample_rand_[idx]; }
    inline int randomUniform(int max, int min = 0) const    {
                return std::uniform_int_distribution<>(min, max)(rand_gen_);
            }

    void DefineCrop(NDLLSize *out_size, int *pCropX, int *pCropY) const;

    bool CropNeeded(const NDLLSize &out_size) const {
        return 0 < crop_h_ && crop_h_ <= out_size.height &&
               0 < crop_w_ && crop_w_ <= out_size.width;
    }

    void MirrorNeeded(NppiPoint *pntr) const {
        MirrorNeeded(reinterpret_cast<bool *>(&pntr->x), reinterpret_cast<bool *>(&pntr->y));
    }

protected:
    void MirrorNeeded(bool *pHorMirror, bool *pVertMirror = NULL) const {
        if (pHorMirror) {
            *pHorMirror = mirror_prob_.empty()? false :
                          std::bernoulli_distribution(mirror_prob_[0])(rand_gen_);
        }

        if (pVertMirror) {
            *pVertMirror = mirror_prob_.size() <= 1? false :
                           std::bernoulli_distribution(mirror_prob_[1])(rand_gen_);
        }
    }

    inline vector<const uint8*> *inputImages()              { return &input_ptrs_; }
    inline vector<uint8 *> *outputImages()                  { return &output_ptrs_; }
    inline const resize_t &resize() const                   { return resize_; }

    mutable std::mt19937 rand_gen_;

    // Resize meta-data
    bool random_resize_;
    bool warp_resize_;
    resize_t resize_;

    // Input/output channels meta-data
    NDLLImageType image_type_;
    bool color_;
    int C_;

    bool random_crop_;
    int crop_h_, crop_w_;
    vector<float> mirror_prob_;

    // Interpolation type
    NDLLInterpType type_;

    // store per-thread data for same resize on multiple data
    std::vector<resize_t> per_sample_rand_;

    vector<const uint8*> input_ptrs_;
    vector<uint8*> output_ptrs_;

    vector<NDLLSize> sizes_[2];
};

template <typename Backend>
class Resize : public Operator, public ResizeAttr {
 public:
  explicit inline Resize(const OpSpec &spec) :
    Operator(spec), ResizeAttr(spec) {
      resizeParam_.resize(batch_size_);
      // Resize per-image data
      input_ptrs_.resize(batch_size_);
      output_ptrs_.resize(batch_size_);
      sizes_[0].resize(batch_size_);
      sizes_[1].resize(batch_size_);

      // Per set-of-samples random numbers
      per_sample_rand_.resize(batch_size_);
  }

  virtual inline ~Resize() = default;

 protected:
  inline void SetupSharedSampleParams(DeviceWorkspace* ws) override {
    const int resize_a = resize_.first;
    const int resize_b = resize_.second;
    for (int i = 0; i < batch_size_; ++i) {
      auto rand_a = randomUniform(resize_b, resize_a);
      auto rand_b = randomUniform(resize_b, resize_a);

      per_sample_rand_[i] = std::make_pair(rand_a, rand_b);
    }
  }

  void RunPerSampleCPU(SampleWorkspace *ws, const int idx) override {
        const auto &input = ws->Input<CPUBackend>(idx);
        auto output = ws->Output<CPUBackend>(idx);

        const vector <Index> &input_shape = input.shape();
        NDLLSize input_size, out_size;
        SetSize(&input_size, input_shape, resize(), &out_size);

        const uint8 *pInRaster;
        uint8 *pOutRaster;
        DataDependentSetupCPU(input, output, "Resize", &pInRaster, &pOutRaster, NULL, &out_size);
        NDLL_CALL(BatchedResize(&pInRaster, 1, C_, &input_size,
                                    &pOutRaster, &out_size, type_));
  }

  inline void RunBatchedGPU(DeviceWorkspace *ws, const int idx) override {
    const auto &input = ws->Input<GPUBackend>(idx);
    auto output = ws->Output<GPUBackend>(idx);

    ResizeParamDescr resizeDescr(this, resizeParam_.data());
    DataDependentSetupGPU(input, output, batch_size_, false,
                            inputImages(), outputImages(), NULL, &resizeDescr);

    // Run the kernel
    cudaStream_t old_stream = nppGetStream();
    nppSetStream(ws->stream());
    BatchedResize(
        (const uint8**)input_ptrs_.data(),
        batch_size_, C_, sizes(input_t).data(),
        output_ptrs_.data(), sizes(output_t).data(),
        type_, resizeParam_.data());
    nppSetStream(old_stream);
  }

    vector<NppiPoint>resizeParam_;
    USE_OPERATOR_MEMBERS();
};

}  // namespace ndll

#endif  // NDLL_PIPELINE_OPERATORS_RESIZE_H_
