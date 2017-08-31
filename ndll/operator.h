#ifndef NDLL_OPERATOR_H_
#define NDLL_OPERATOR_H_

#include "ndll/common.h"
#include "ndll/stage.h"

namespace ndll {

class Operator {
public:
  Operator() {}
  virtual ~Operator() = default;

  // Indicates the number of computation stages in this Op
  virtual int NumStages() const = 0;

  // Returns the stage at the input index
  virtual shared_ptr<Stage> stage(int idx) = 0;
  
  DISABLE_COPY_ASSIGN(Operator);
private:
  
};

class Transformer : public Operator {
public:
  Transformer() {
    
  }

  virtual ~Transformer() = default;
  
  /**
   * @brief Tranformers have a single stage of computation
   */
  virtual int NumStages() const override final {
    return 1;
  }

  virtual shared_ptr<Stage> stage(int idx) {
    NDLL_ASSERT(idx == 0);
    return stage_;
  }

  DISABLE_COPY_ASSIGN(Transformer);
private:
  shared_ptr<Stage> stage_;
};

class Decoder : public Operator {
public:
  Decoder() {

  }

  virtual ~Decoder() = default;
  
  /**
   * @brief Decoders can be CPU, GPU, or Hybrid. Thus, 
   * they are broken into two stages of computation
   */
  virtual int NumStages() const override final {
    return 2;
  }

  virtual shared_ptr<Stage> stage(int idx) {
    NDLL_ASSERT(idx == 0 || idx == 1);
    return stages_[idx];
  }
  
  DISABLE_COPY_ASSIGN(Decoder);
private:
  array<shared_ptr<Stage>, 2> stages_;
};

} // namespace ndll

#endif // NDLL_OPERATOR_H_
