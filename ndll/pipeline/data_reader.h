#ifndef NDLL_PIPELINE_DATA_READER_H_
#define NDLL_PIPELINE_DATA_READER_H_

#include <type_traits>

#include "ndll/pipeline/data/backend.h"
#include "ndll/pipeline/data/batch.h"
#include "ndll/pipeline/data/datum.h"

namespace ndll {

/**
 * @brief Defines the API for a DataReader. DataReaders provide access to some
 * form of underlying data storage. The API is currently quite simplicitic, but
 * may evolve over time.
 *
 * Note: 'Read' will be called in the main thread, it does not need to be thread
 * safe.
 */
template <typename Backend,
          typename std::enable_if<std::is_base_of<CPUBackend, Backend>::value, int>::type = 0>
class DataReader {
public:
  DataReader(){}
  virtual ~DataReader() = default;

  /**
   * @brief fills the input Datum object with a single data sample
   */
  virtual void Read(Datum<Backend> *datum) = 0;

  /**
   * @brief Resets the reader to the intial state, e.g. 
   * the beginning of the database.
   */
  virtual void Reset() = 0;
  
  virtual DataReader* Clone() const = 0;
  
  DISABLE_COPY_MOVE_ASSIGN(DataReader);
protected:
};

/**
 * @brief Basic data reader that provides access to data loaded into a batch
 * externally. This may go away in the future, it currently exists to provide
 * compatibility with the way we have been doing things before-data-reader.
 */
template <typename Backend>
class BatchDataReader final : public DataReader<Backend> {
public:
  BatchDataReader(shared_ptr<Batch<Backend>> data_store)
    : data_store_(data_store), cursor_(0), batch_size_(data_store->ndatum()) {
    NDLL_ENFORCE(data_store_ != nullptr);
    NDLL_ENFORCE(batch_size_ > 0);
  }
  
  /**
   * @brief Wraps the input Datum object around a single data sample
   */
  void Read(Datum<Backend> *datum) override;

  /**
   * Resets the cursor to 0
   */
  void Reset() override {
    cursor_ = 0;
  }
  
  BatchDataReader<Backend>* Clone() const override {
    return new BatchDataReader(data_store_);
  }
  
  DISABLE_COPY_MOVE_ASSIGN(BatchDataReader);
private:
  // On construction, the BatchDataReader wraps a 'Batch' object.
  // Over the course of training, it then simply provides access
  // to this batch.
  shared_ptr<Batch<Backend>> data_store_;

  // The current element to return from the batch
  int cursor_;
  int batch_size_;
};

template <typename Backend>
void BatchDataReader<Backend>::Read(Datum<Backend> *datum) {
  datum->WrapSample(data_store_.get(), cursor_);
  cursor_ = (cursor_ + 1) % batch_size_;
}

} // namespace ndll

#endif // NDLL_PIPELINE_DATA_READER_H_
