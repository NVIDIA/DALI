#ifndef DALI_OPERATOR_ADAPTER_H
#define DALI_OPERATOR_ADAPTER_H

namespace dali {

namespace testing {

class OperatorAdapter {
public:
  enum Backend {
    CPU, GPU,
  };

private:
  std::string operator_name_;
  Backend input_backend_, output_backend_;
};

}  // namespace testing

}  // namespace dali

#endif //DALI_OPERATOR_ADAPTER_H
