#ifndef NDLL_PIPELINE_DATA_READER_H_
#define NDLL_PIPELINE_DATA_READER_H_

#include <type_traits>

#include "ndll/pipeline/data/backend.h"
#include "ndll/pipeline/data/batch.h"
#include "ndll/pipeline/data/datum.h"
#include "ndll/pipeline/operator_factory.h"
#include "ndll/pipeline/op_spec.h"
#include "ndll/util/image.h"

namespace ndll {

/**
 * @brief Defines the API for a DataReader. DataReaders provide access to some
 * form of underlying data storage. The API is currently quite simplicitic, but
 * may evolve over time.
 */
class DataReader {
public:
  inline DataReader(const OpSpec &spec) :
    batch_size_(spec.GetSingleArgument<int>("batch_size", -1)) {
    NDLL_ENFORCE(batch_size_ > 0, "Invalid value for argument batch_size.");
  }
  
  virtual ~DataReader() = default;

  /**
   * @brief fills the input Datum object with a single data sample
   */
  virtual void Read(Datum<CPUBackend> *datum) = 0;

  /**
   * @brief Resets the reader to the intial state, e.g. 
   * the beginning of the database.
   */
  virtual void Reset() = 0;
  
  DISABLE_COPY_MOVE_ASSIGN(DataReader);
protected:
  int batch_size_;
};

// Create registries for DataReaders
NDLL_DECLARE_OPTYPE_REGISTRY(DataReader, DataReader);

#define NDLL_REGISTER_DATA_READER(OpName, OpType) \
  NDLL_DEFINE_OPTYPE_REGISTERER(OpName, OpType,   \
      ndll::DataReader, ndll::DataReader)

/**
 * @brief Basic data reader that provides access to data loaded into a batch
 * externally. This may go away in the future, it currently exists to provide
 * compatibility with the way we have been testing things pre-data-reader.
 */
class BatchDataReader final : public DataReader {
public:
  inline BatchDataReader(const OpSpec &spec) :
    DataReader(spec), cursor_(0) {
    // Load the images from the specified folder and create a
    // Batch object that contains them.
    string jpeg_folder = spec.GetSingleArgument<string>("jpeg_folder", "");
    vector<uint8*> jpegs;
    vector<int> jpeg_sizes;
    if (jpeg_folder.empty()) {
      // Load all the specified images and copy them into a Batch object
      vector<string> jpeg_names = spec.GetRepeatedArgument<string>("jpeg_images");
      NDLL_ENFORCE(jpeg_names.size() > 0, "No image files specified.");
      LoadJPEGS(jpeg_names, &jpegs, &jpeg_sizes);
    } else {
      vector<string> jpeg_names;
      LoadJPEGS(jpeg_folder, &jpeg_names, &jpegs, &jpeg_sizes);
    }
    data_store_.reset(CreateJPEGBatch<CPUBackend>(jpegs, jpeg_sizes, batch_size_));
  }
  
  /**
   * @brief Wraps the input Datum object around a single data sample.
   */
  inline void Read(Datum<CPUBackend> *datum) override {
    datum->WrapSample(data_store_.get(), cursor_);
    cursor_ = (cursor_ + 1) % batch_size_;
  }

  /**
   * @brief Resets the cursor to 0.
   */
  void Reset() override {
    cursor_ = 0;
  }

  DISABLE_COPY_MOVE_ASSIGN(BatchDataReader);
private:
  // Over the course of training, the BatchDataReader
  // simply provides access to the loaded batch of images
  std::unique_ptr<Batch<CPUBackend>> data_store_;

  // The current element to return from the batch
  int cursor_;
};

} // namespace ndll

#endif // NDLL_PIPELINE_DATA_READER_H_
