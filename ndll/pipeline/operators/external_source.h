#ifndef NDLL_PIPELINE_OPERATORS_EXTERNAL_SOURCE_H_
#define NDLL_PIPELINE_OPERATORS_EXTERNAL_SOURCE_H_

#include "ndll/pipeline/operator.h"

namespace ndll {

/**
 * @brief Provides in-graph access to data fed in from outside of ndll.
 * For now, we do a copy from the passed in data into our data to avoid
 * potential scoping and data corruption issues.
 */
template <typename Backend>
class ExternalSource : public Operator<Backend> {
public:
  inline explicit ExternalSource(const OpSpec &spec) :
    Operator<Backend>(spec) {
    output_name_ = spec.Output(0);
  }
  
  virtual inline ~ExternalSource() = default;

  inline bool SupportsInPlace() const override { return true; }
  
  inline int MaxNumInput() const override { return 0; }
  inline int MinNumInput() const override { return 0; }
  inline int MaxNumOutput() const override { return 1; }
  inline int MinNumOutput() const override { return 1; }

  inline string name() const override {
    return "ExternalSource (" + output_name_ + ")";
  }
  
  /**
   * @brief Sets the data that should be passed out of the op
   * on the next iteration.
   */
  inline void SetDataSource(const TensorList<Backend> &tl) {
    // Note: If we create a GPU source, we will need to figure
    // out what stream we want to do this copy in. CPU we can
    // pass anything as it is ignored.
    data_.Copy(tl, 0);
  }
  
  DISABLE_COPY_MOVE_ASSIGN(ExternalSource);
protected:
  inline void RunPerSampleCPU(SampleWorkspace *ws) override {
    // Wrap the output tensor around our data
    auto output = ws->Output<CPUBackend>(0);
    output->ShareData(&data_, ws->data_idx());
    output->set_type(data_.type());
    output->Resize(data_.tensor_shape(ws->data_idx()));
  }
  
  string output_name_;
  TensorList<Backend> data_;
};

} // namespace ndll

#endif // NDLL_PIPELINE_OPERATORS_EXTERNAL_SOURCE_H_
