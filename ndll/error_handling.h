#ifndef NDLL_ERROR_HANDLING_H_
#define NDLL_ERROR_HANDLING_H_

#include <sstream>
#include <string>

namespace ndll {

// Note: We won't control all of the code that is executing in this
// library because users can define their own ops. Because of this,
// we can't use fixed error types like other Nvidia libraries. To
// deal with this, we'll define an exception type and allow users
// to pass in error strings so that they can be propagated out to
// the caller when an error occurs.

/**
 * @brief Basic exception class used by ndll error checking.
 */
class NdllException {
public:
  NdllException() {}
  
  explicit NdllException(std::string str) {
    str_.append(str);
  }

  const char* what() {
    return str_.c_str();
  }

  // TODO(tgale): This currently only supports basic types. Things like
  // std::endl that are functions don't work with this yet. Need to add
  // support. See https://stackoverflow.com/questions/1134388/.
  template <typename T>
  NdllException& operator<<(const T &info) {
    std::stringstream ss;
    ss << info;
    this->str_.append(ss.str());
    return *this;
  }
  
private:
  std::string str_;
};

inline NdllException failed_assert(std::string statement,
    std::string file, int line) {
  std::string error = "[" + file + ":" + std::to_string(line) +
    "]: Assert on \"" + statement + "\" failed: ";
  return NdllException(error);
}

// Note: These can only be called from inside the ndll namespace
#define NDLL_ASSERT(val)                                  \
  if (!val) throw failed_assert(#val, __FILE__, __LINE__)

// CUDA error checking utility
#define CUDA_CALL(code)                             \
  do {                                              \
    cudaError_t status = code;                      \
    if (code != cudaSuccess) {                      \
      std::string file = __FILE__;                  \
      std::string line = std::to_string(__LINE__);  \
      std::string error = "[" + file + ":" + line + \
        "]: CUDA error \"" +                        \
        cudaGetErrorString(code) + "\"";            \
      throw NdllException(error);                   \
    }                                               \
  } while (0)

// Note: We can define error checking for other libraries
// here. E.g. CUBLAS_CALL, NPP_CALL, etc.

} // namespace ndll

#endif // NDLL_ERROR_HANDLING_H_
