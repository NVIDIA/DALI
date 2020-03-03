#ifndef DALI_DALI_KERNELS_AUDIO_MEL_SCALE_MEL_FILTER_BANK_TEST_H_
#define DALI_DALI_KERNELS_AUDIO_MEL_SCALE_MEL_FILTER_BANK_TEST_H_

#include <vector>
#include <cassert>
#include "dali/kernels/audio/mel_scale/mel_scale.h"
#include "dali/core/common.h"
#include "dali/kernels/kernel_params.h"

namespace dali {
namespace kernels {
namespace audio {
namespace test {

template <typename T, int64_t Dims>
void print_data(const OutTensorCPU<T, Dims> &data_view) {
  static_assert(Dims > 2, "Print works for Dims > 1");

}

template <typename T>
void print_data(const OutTensorCPU<T, 2>& data_view) {
  auto sh = data_view.shape;
  for (int i0 = 0; i0 < sh[0]; i0++) {
    for (int i1 = 0; i1 < sh[1]; i1++) {
      int k = i0 * sh[1] + i1;
      LOG_LINE << " " << data_view.data[k];
    }
    LOG_LINE << "\n";
  }
}

std::vector<std::vector<float>> ReferenceFilterBanks(int nfilter, int nfft, float sample_rate,
                                                     float low_freq, float high_freq);

}
}
}
}

#endif //DALI_DALI_KERNELS_AUDIO_MEL_SCALE_MEL_FILTER_BANK_TEST_H_
