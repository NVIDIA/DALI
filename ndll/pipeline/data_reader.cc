#include "ndll/pipeline/data_reader.h"

namespace ndll {

NDLL_DEFINE_OPTYPE_REGISTRY(DataReader, DataReader);

NDLL_REGISTER_DATA_READER(BatchDataReader, BatchDataReader);

} // namespace ndll
