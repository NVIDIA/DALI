#include <mutex>

#include "ndll/util/user_stream.h"

namespace ndll {

  UserStream * UserStream::us_ = nullptr;
  std::mutex UserStream::m_;

}  // namespace ndll
