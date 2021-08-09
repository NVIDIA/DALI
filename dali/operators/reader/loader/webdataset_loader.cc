#include "dali/operators/reader/loader/webdataset_loader.h"
#include <vector>

namespace dali {

WebdatasetLoader::WebdatasetLoader(const OpSpec& spec) {
  // TODO(barci2)
}

WebdatasetLoader::~WebdatasetLoader() {
  // TODO(barci2)
}

void WebdatasetLoader::PrepareEmpty(vector<Tensor<CPUBackend>>& empty) {
  empty = std::vector<Tensor<CPUBackend>>();
}

void WebdatasetLoader::ReadSample(vector<Tensor<CPUBackend>>& sample) {
  // TODO(barci2)
}

Index WebdatasetLoader::SizeImpl() {
  // TODO(barci2)
}

void WebdatasetLoader::PrepareMetadataImpl() {
  // TODO(barci2)
}

void WebdatasetLoader::Reset(bool wrap_to_shard) {
  // TODO(barci2)
}

}  // namespace dali