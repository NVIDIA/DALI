#ifndef NDLL_PIPELINE_EXECUTOR_H_
#define NDLL_PIPELINE_EXECUTOR_H_

#include "ndll/common.h"
#include "ndll/error_handling.h"
#include "ndll/pipeline/op_graph.h"

namespace ndll {

class Executor {
public:
  inline explicit Executor(OpGraph *graph);
  inline Executor();
  virtual ~Executor() = default;

  void Build(OpGraph *graph);

  void RunCPU();

  void RunInternal();

  void RunGPU();
  
  DISABLE_COPY_MOVE_ASSIGN(Executor);
protected:
  
};

} // namespace ndll

#endif // NDLL_PIPELINE_EXECUTOR_H_
