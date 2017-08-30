#ifndef NDLL_BATCH_H_
#define NDLL_BATCH_H_

#include <vector>

#include "ndll/common.h"

namespace ndll {

// Used for sample-wise dimensions
typedef vector<int> Dims;

/**
 * @brief Basic data storage unit for ndll. Stores a batch of data.
 *
 * Note: For now this is templated. This shouldn't be too big of an issue, 
 * most of the image processing pipeline should be done in uint8 unless
 * absoultely necessary to avoid memory transfers. If the template turns
 * out to be a major pain, we can switch to a simple type system.
 *
 * The pipeline won't need to be templated because it doesn't touch data
 * and the ops will all be referenced through their base classes.
 */
template <typename Backend, typename T>
class Batch {
public:
  Batch() {}
  ~Batch() {
    Backend::Delete(data_, bytes_);
  }

  /**
   * @brief Get const pointer to underlying data
   */
  const T* data() const {
    return data_;
  }

  /**
   * @brief Get pointer to underlying data
   */
  T* data() {
    return data_;
  }

  /**
   * @brief resize the buffer to store `num` elements. This
   * method does not guarantee that that the data in the 
   * underlying storage is preserved
   */
  void Resize(int num) {
    size_t new_bytes = num * sizeof(T);
    if (new_bytes > bytes_) {
      Backend::Delete(data_, bytes_);
      data_ = Backend::New(new_bytes);
      bytes_ = new_bytes;
    }
    num_ = num;
  }
  
  DISABLE_COPY_ASSIGN(Batch);
private:
  T *data_ = nullptr;
  int num_ = 0;

  // Batches are only resized if the input size is
  // larger than the underlying allocation size.
  // This variable stores the true size of the
  // buffer in bytes
  size_t bytes_ = 0;
};

} // namespace ndll

#endif // NDLL_BATCH_H_
