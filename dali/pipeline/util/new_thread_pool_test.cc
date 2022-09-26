#include <gtest/gtest.h>
#include "new_thread_pool.h"

namespace dali {
namespace experimental {

struct SerialExecutor {
  template <typename Runnable>
  std::enable_if_t<std::is_convertible_v<Runnable, std::function<void(int)>>>
  AddTask(Runnable &&runnable) {
    runnable(0);
  }
};

TEST(NewThreadPool, RunJobInSeries) {
  Job job;
  SerialExecutor tp;
  int a = 0, b = 0, c = 0;
  job.AddTask([&](int) {
    a = 1;
  });
  job.AddTask([&](int) {
    b = 2;
  });
  job.AddTask([&](int) {
    c = 3;
  });
  job.Run(tp, true);
  EXPECT_EQ(a, 1);
  EXPECT_EQ(b, 2);
  EXPECT_EQ(c, 3);
}

TEST(NewThreadPool, RunJobInThreadPool) {
  Job job;
  ThreadPool tp(4);
  int a = 0, b = 0, c = 0;
  job.AddTask([&](int) {
    a = 1;
  });
  job.AddTask([&](int) {
    b = 2;
  });
  job.AddTask([&](int) {
    c = 3;
  });
  job.Run(tp, true);
  EXPECT_EQ(a, 1);
  EXPECT_EQ(b, 2);
  EXPECT_EQ(c, 3);
}


TEST(NewThreadPool, RethrowMultipleErrors) {
  Job job;
  ThreadPool tp(4);
  job.AddTask([&](int) {
    throw std::runtime_error("Runtime");
  });
  job.AddTask([&](int) {
    // do nothing
  });
  job.AddTask([&](int) {
    throw std::logic_error("Logic");
  });
  EXPECT_THROW(job.Run(tp, true), MultipleErrors);
}


}  // namespace experimental
}  // namespace dali
