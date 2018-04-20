// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#ifndef NDLL_PIPELINE_OPERATORS_READER_FILE_READER_OP_H_
#define NDLL_PIPELINE_OPERATORS_READER_FILE_READER_OP_H_

#include "ndll/pipeline/operators/reader/reader_op.h"
#include "ndll/pipeline/operators/reader/loader/file_loader.h"

namespace ndll {

class FileReader : public DataReader<CPUBackend> {
 public:
  explicit FileReader(const OpSpec& spec)
    : DataReader<CPUBackend>(spec) {
    loader_.reset(new FileLoader(spec));
  }

  DEFAULT_READER_DESTRUCTOR(FileReader, CPUBackend);

  void RunImpl(SampleWorkspace *ws, const int i) override {
    const int idx = ws->data_idx();

    auto* raw_data = prefetched_batch_[idx];

    // copy from raw_data -> outputs directly
    auto *image_output = ws->Output<CPUBackend>(0);
    auto *label_output = ws->Output<CPUBackend>(1);

    Index raw_size = raw_data->size();
    Index image_size = raw_size - sizeof(int);

    image_output->Resize({image_size});
    image_output->mutable_data<uint8_t>();
    label_output->Resize({1});

    std::memcpy(image_output->raw_mutable_data(),
                raw_data->raw_data(),
                image_size);

    label_output->mutable_data<int>()[0] =
       *reinterpret_cast<const int*>(raw_data->data<uint8_t>() + image_size);
    return;
  }

 protected:
  USE_READER_OPERATOR_MEMBERS(CPUBackend);
};

}  // namespace ndll

#endif  // NDLL_PIPELINE_OPERATORS_READER_FILE_READER_OP_H_
