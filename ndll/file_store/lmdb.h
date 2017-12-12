#ifndef NDLL_FILE_STORE_LMDB_H_
#define NDLL_FILE_STORE_LMDB_H_

#ifdef USE_LMDB

#include <lmdb.h>
#include "ndll/file_store/file_store_reader.h"

namespace ndll {

namespace {
  inline void CHECK_LMDB(int status) {
    NDLL_ENFORCE(status == MDB_SUCCESS,
        "LMDB Error: " + string(mdb_strerror(status)));
  }

  bool SeekLMDB(MDB_cursor* cursor, MDB_cursor_op op, MDB_val& key, MDB_val& value) {
    int status = mdb_cursor_get(cursor, &key, &value, op);

    if (status == MDB_NOTFOUND) {
      // reached the end of the db
      return false;
    } else {
      CHECK_LMDB(status);
      return true;
    }
  }

  void PrintLMDBStats(MDB_txn* txn, MDB_dbi dbi) {
    MDB_stat* stat;

    LMDB_CHECK(txn, dbi, stat);

    printf("DB has %d entries\n", stat.ms_entries);
  }
}

class LMDBReader : public FileStoreReader {
 public:
  LMDBReader(std::string uri, Options& options)
    : FileStoreReader(uri, options) {
    CHECK_LMDB(mdb_env_create(&mdb_env_));
    auto mdb_flags = MDB_RDONLY | MDB_NOTLS | MDB_NOLOCK;
    CHECK_LMDB(mdb_env_open(mdb_env_, uri.c_str(), mdb_flags, 0664));

    CHECK_LMDB(mdb_txn_begin(mdb_env_, NULL, MDB_RDONLY, &mdb_transaction_));
    CHECK_LMDB(mdb_dbi_open(mdb_transaction_, NULL, 0, &mdb_dbi_));
    CHECK_LMDB(mdb_cursor_open(mdb_transaction_, mdb_dbi_, &mdb_cursor_));

    PrintLMDBStats(mdb_transaction, mdb_dbi_);
  }
  ~LMDBReader() {
    mdb_cursor_close(mdb_cursor_);
    mdb_dbi_close(mdb_env_, mdb_dbi_);
    mdb_txn_abort(mdb_transaction_);
    mdb_env_close(mdb_env_);
    mdb_env_ = nullptr;
  }

  void ReadSample(Tensor<CPUBackend>* tensor) {
    // assume cursor is valid, read next, loop to start if necessary
    Sample sample;

    bool ok = SeekLMDB(mdb_cursor_, MDB_NEXT, key_, value_);

    if (!ok) {
      SeekLMDB(mdb_cursor_, MDB_FIRST, key_, value_);
    }

    tensor->Resize({value_.mv_size});
    data_ptr = tensor->mutable_data<uint8_t>();
    std::memcpy(value_.mv_data, data_ptr, value_.mv_size);

    return;
  }
 private:
  MDB_env* mdb_env_;
  MDB_cursor* mdb_cursor_;
  MDB_dbi mdb_dbi_;
  MDB_txn* mdb_transaction_;

  // values
  MDB_val key_, value_;

};

}; // namespace ndll

#endif // USE_LMDB

#endif // NDLL_FILE_STORE_LMDB_H_
