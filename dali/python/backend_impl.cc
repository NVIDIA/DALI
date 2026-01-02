// Copyright (c) 2017-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <sstream>
#include <cstring>
#include "dali/core/common.h"
#include "dali/core/cuda_utils.h"
#include "dali/core/device_guard.h"
#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "pyerrors.h"  // NOLINT(build/include)
#if SHM_WRAPPER_ENABLED
#include "dali/core/os/shared_mem.h"
#endif
#include "dali/core/python_util.h"
#include "dali/core/mm/default_resources.h"
#include "dali/operators.h"
#include "dali/kernels/kernel.h"
#include "dali/operators/reader/parser/tf_feature.h"
#include "dali/pipeline/data/copy_to_external.h"
#include "dali/pipeline/data/dltensor.h"
#include "dali/pipeline/data/tensor.h"
#include "dali/pipeline/data/tensor_list.h"
#include "dali/pipeline/init.h"
#include "dali/pipeline/operator/error_reporting.h"
#include "dali/pipeline/operator/op_schema.h"
#include "dali/pipeline/operator/op_spec.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/pipeline.h"
#include "dali/pipeline/pipeline_debug.h"
#include "dali/plugin/plugin_manager.h"
#include "dali/python/python3_compat.h"
#include "dali/util/pybind.h"
#include "dali/util/user_stream.h"

namespace dali {
namespace python {


#if (CUDART_VERSION >= 10200 && CUDART_VERSION < 11100)
// add this alignment to work around a patchelf bug/feature which
// changes TLS alignment and break DALI interoperability with CUDA RT
alignas(0x1000) thread_local volatile bool __backend_impl_force_tls_align;

void __backend_impl_force_tls_align_fun(void) {
  __backend_impl_force_tls_align = 0;
}
#else
void __backend_impl_force_tls_align_fun(void) {}
#endif


using namespace pybind11::literals; // NOLINT

/**
 * @brief Override the default __module__ of Tensor classes from nvidia.dali.backend_impl
 * with the user-friendly Python module.
 * This definition must be provided as a first one for the Tensor classes, so all following
 * definitions can look it up when pybind is generating the signatures, otherwise the annotations
 * will contain the backend_impl module path.
 */
static std::string tensor_module_impl(const py::object &object) {
  (void)object;
  return "nvidia.dali.tensors";
}

static void* ctypes_void_ptr(const py::object& object) {
  auto ptr_as_int = getattr(object, "value", py::none());
  if (ptr_as_int.is_none()) {
    return nullptr;
  }
  void *ptr = PyLong_AsVoidPtr(ptr_as_int.ptr());
  return ptr;
}

TensorShape<> shape_from_py(py::tuple tup) {
  TensorShape<> shape;
  shape.resize(tup.size());
  for (size_t i = 0; i < tup.size(); ++i) {
    shape[i] = tup[i].cast<int64_t>();
  }
  return shape;
}

template <int ndim>
py::list as_py_list(const TensorShape<ndim> &shape) {
  py::list ret(shape.size());
  for (int i = 0; i < shape.size(); i++) {
    ret[i] = shape[i];
  }
  return ret;
}

template <typename Backend>
py::list py_shape(const Tensor<Backend> &t) {
  return as_py_list(t.shape());
}

template <typename Backend>
std::vector<py::tuple> py_shape_list(const TensorList<Backend> &tl) {
  std::vector<py::tuple> ret(tl.shape().size());
  for (int i = 0; i < tl.shape().size(); ++i) {
    ret[i] = py::tuple(as_py_list(tl.tensor_shape(i)));
  }
  return ret;
}

static string TensorLayoutRepr(const TensorLayout &tl) {
  std::stringstream ss;
  ss << "nvidia.dali.types.TensorLayout('";
  escape_string(ss, tl.c_str());
  ss << "')";
  return ss.str();
}

template<typename Backend>
py::dict ArrayInterfaceRepr(Tensor<Backend> &t) {
  py::dict d;
  py::tuple tup(2);
  d["typestr"] = FormatStrFromType(t.type());
  // __array_interface__ expects shape to be a tuple
  d["shape"] = py::tuple(py_shape<Backend>(t));
  // tuple of (raw_data_pointer, if_data_is_read_only)
  tup[0] = py::reinterpret_borrow<py::object>(PyLong_FromVoidPtr(t.raw_mutable_data()));
  // if we make it readonly, it prevents us from sharing memory with PyTorch tensor
  tup[1] = false;
  d["data"] = tup;
  if constexpr (std::is_same<Backend, GPUBackend>::value) {
    // see https://numba.pydata.org/numba-doc/dev/cuda/cuda_array_interface.html
    // this set of atributes is tagged as version 2
    d["version"] = 2;
  } else {
    // see https://docs.scipy.org/doc/numpy/reference/arrays.interface.html
    // this set of atributes is tagged as version 3
    d["version"] = 3;
    if (t.is_pinned()) {
      if (auto &event = t.ready_event())
        AccessOrder::host().wait(event);  // more fine-grained synchronization
      else
        AccessOrder::host().wait(t.order());
    }
  }
  d["strides"] = py::none();
  return d;
}

namespace {
  const uint32_t kCPUTensorColor = DomainTimeRange::kBlue1;
  const uint32_t kGPUTensorColor = DomainTimeRange::knvGreen;
}  // namespace

template<typename SrcBackend>
const TensorListShape<> ConvertShape(const TensorShape<> &shape,
                                      TensorList<SrcBackend> *shape_type_placeholder) {
  return uniform_list_shape(shape[0], shape.last(shape.size()-1));
}

template<typename SrcBackend>
const TensorShape<> &ConvertShape(const TensorShape<> &shape,
                                  Tensor<SrcBackend> *shape_type_placeholder) {
  return shape;
}

template<typename TStrides, typename TShape>
void CheckContiguousTensor(const TStrides &strides, int num_strides,
                           const TShape &shape, int num_extents, size_t element_size) {
  DALI_ENFORCE(num_strides == num_extents,
    "There should be exactly as many strides as there are extents in array shape.");
  int64_t stride_from_shape = element_size;
  int64_t stride_from_shape_collapsed = 1;
  int64_t last_non_one_dim = 1;
  for (int i = num_strides - 1; i >= 0; i--) {
    DALI_ENFORCE(strides[i] == stride_from_shape || strides[i] == stride_from_shape_collapsed,
        make_string("Strided data not supported. Dimension ", i, " has stride ", strides[i],
        " whereas densely packed data of this shape would have a stride ", stride_from_shape));
    stride_from_shape *= shape[i];
    // for shapes [1, 1, 5] leading dimensions may not contribute to stride
    if (shape[i] != 1) {
      stride_from_shape_collapsed *= last_non_one_dim;
      last_non_one_dim = shape[i];
    }
  }
}

template<typename TStrides, typename TShape>
void CheckContiguousTensor(const TStrides &strides, const TShape &shape, size_t element_size) {
  CheckContiguousTensor(strides, dali::size(strides), shape, dali::size(shape), element_size);
}

template <typename Backend>
void SetLayout(
      Tensor<Backend> &t,
      const std::optional<std::string> &layout_str,
      bool clear_if_none = true) {
  if (layout_str && !layout_str->empty()) {
    TensorLayout layout = *layout_str;
    if (t.ndim() != layout.ndim()) {
      throw py::value_error(make_string(
        "A non-empty layout must have the same number of dimensions as the "
        "number of dimensions of the Tensor.\n"
        "Got: ", layout.ndim(), " (", layout, ")\n",
        "Expected: ", t.ndim(), "."));
    }
    t.SetLayout(layout);
  } else if (clear_if_none) {
    t.SetLayout({});
  }
}

template <typename Backend>
void SetLayout(
      TensorList<Backend> &t,
      const std::optional<std::string> &layout_str,
      bool clear_if_none = true) {
  if (layout_str && !layout_str->empty()) {
    TensorLayout layout = *layout_str;
    if (t.sample_dim() != layout.ndim()) {
      throw py::value_error(make_string(
        "A non-empty layout must have the same number of dimensions as the "
        "number of dimensions of the TensorList.\n"
        "Got: ", layout.ndim(), " (", layout, ")\n",
        "Expected: ", t.sample_dim(), "."));
    }
    t.SetLayout(layout);
  } else if (clear_if_none) {
    t.SetLayout({});
  }
}

template<typename SrcBackend, template<typename> class SourceDataType>
void FillTensorFromDlPack(
      py::capsule capsule,
      SourceDataType<SrcBackend> *batch,
      const std::optional<std::string> &layout) {
  auto dlm_tensor_ptr = DLMTensorPtrFromCapsule(capsule);
  const auto &dl_tensor = dlm_tensor_ptr->dl_tensor;
  DALI_ENFORCE((std::is_same<SrcBackend, GPUBackend>::value &&
                  dl_tensor.device.device_type == kDLCUDA) ||
               (std::is_same<SrcBackend, CPUBackend>::value &&
                  dl_tensor.device.device_type == kDLCPU),
               "DLPack device type doesn't match Tensor type");

  const TypeInfo &dali_type = TypeTable::GetTypeInfo(ToDALIType(dl_tensor.dtype));
  TensorShape<> shape;
  shape.resize(dl_tensor.ndim);
  for (ssize_t i = 0; i < dl_tensor.ndim; ++i) {
    shape[i] = dl_tensor.shape[i];
  }

  if (dl_tensor.strides)
    CheckContiguousTensor(dl_tensor.strides, dl_tensor.ndim, dl_tensor.shape, dl_tensor.ndim, 1);

  size_t bytes = volume(shape) * dali_type.size();

  const auto &typed_shape = ConvertShape(shape, batch);
  bool is_pinned = dl_tensor.device.device_type == kDLCUDAHost;
  int device_id = CPU_ONLY_DEVICE_ID;
  // according to the docs kDLCUDAHost = kDLCPU | kDLCUDA so test it as a the first option
  if (dl_tensor.device.device_type == kDLCUDAHost) {
    device_id = CPU_ONLY_DEVICE_ID;
  } else if (dl_tensor.device.device_type == kDLCPU) {
    device_id = CPU_ONLY_DEVICE_ID;
  } else if (dl_tensor.device.device_type == kDLCUDA) {
    device_id = dl_tensor.device.device_id;
  } else {
    DALI_FAIL(make_string("Not supported DLPack device type: ", dl_tensor.device.device_type, "."));
  }

  // empty lambda that just captures dlm_tensor_ptr unique ptr that would be destructed when
  // shared ptr is destroyed
  batch->ShareData(shared_ptr<void>(dl_tensor.data,
                                    [dlm_tensor_ptr = std::move(dlm_tensor_ptr)](void*) {}),
                                    bytes, is_pinned, typed_shape, dali_type.id(), device_id);


  SetLayout(*batch, layout);
}

template <typename TensorType>
void FillTensorFromCudaArray(const py::object &object,
                             TensorType *batch,
                             int device_id,
                             const std::optional<std::string> &layout) {
  auto cu_a_interface_val = getattr(object, "__cuda_array_interface__", py::none());
  if (cu_a_interface_val.is_none()) {
    DALI_FAIL("Provided object doesn't support cuda array interface protocol.")
  }
  py::dict cu_a_interface = py::cast<py::dict>(cu_a_interface_val);

  DALI_ENFORCE(cu_a_interface.contains("typestr") &&
                // see detail::PyUnicode_Check_Permissive implementation
                (PyUnicode_Check(cu_a_interface["typestr"].ptr()) ||
                PYBIND11_BYTES_CHECK(cu_a_interface["typestr"].ptr())) &&
                cu_a_interface.contains("shape") &&
                PyTuple_Check(cu_a_interface["shape"].ptr()) &&
                cu_a_interface.contains("data") &&
                PyTuple_Check(cu_a_interface["data"].ptr()) &&
                cu_a_interface["data"].cast<py::tuple>().size() >= 2 &&
                cu_a_interface.contains("version"),
                "Provided object doesn't have required cuda array interface "
                "protocol fields of necessary type.");
  DALI_ENFORCE(!cu_a_interface.contains("mask") || cu_a_interface["mask"].is_none(),
                "Masked tensors are not supported");

  // Create the Tensor and wrap the data
  TensorShape<> shape = shape_from_py(cu_a_interface["shape"].cast<py::tuple>());

  std::string typestr = cu_a_interface["typestr"].cast<py::str>();
  const TypeInfo &type = TypeFromFormatStr(typestr);
  size_t bytes = volume(shape) * type.size();

  if (cu_a_interface.contains("strides") && !cu_a_interface["strides"].is_none()) {
    TensorShape<> strides = shape_from_py(cu_a_interface["strides"].cast<py::tuple>());
    CheckContiguousTensor(strides, shape, type.size());
  }

  const auto &typed_shape = ConvertShape(shape, batch);
  auto *ptr = PyLong_AsVoidPtr(cu_a_interface["data"].cast<py::tuple>()[0].ptr());

  // it is for __cuda_array_interface__ so device_id < 0 is not a valid value
  if (device_id < 0) {
    CUDA_CALL(cudaGetDevice(&device_id));
  }

  batch->Reset();

  if (cu_a_interface.contains("stream")) {
    const auto &stream_obj = cu_a_interface["stream"];
    if (!stream_obj.is_none()) {
      auto stream_long_value = cu_a_interface["stream"].cast<int64_t>();
      auto stream_value = PyLong_AsVoidPtr(cu_a_interface["stream"].ptr());
      DALI_ENFORCE(stream_value != 0, make_string("Provided stream is not a valid CUDA stream ",
                   "based on CUDA Array Interface v3. `0` value is ambiguous and disallowed"));
      if (stream_long_value == 1) stream_value = 0;
      if (stream_long_value == 2) stream_value = CU_STREAM_PER_THREAD;
      auto order = AccessOrder(cudaStream_t(stream_value));
      batch->set_order(order);
    }
  }

  // Keep a copy of the input object ref in the deleter, so its refcount is increased
  // while this shared_ptr is alive (and the data should be kept alive)
  batch->ShareData(shared_ptr<void>(ptr, [obj_ref = object](void *) mutable {  // NOLINT
    // acquire GIL ...
    py::gil_scoped_acquire aqr;
    {
      // now move out the object stored in the closure to a local variable...
      auto tmp = std::move(obj_ref);
      (void)tmp;
      /// ...and let it go out of scope while GIL is held
    }
  }),
      bytes, false, typed_shape, type.id(), device_id);

  SetLayout(*batch, layout);
}

template <typename Backend>
void ReinterpretTensor(Tensor<Backend> &t, DALIDataType new_type) {
  t.Reinterpret(new_type);
}

template <typename Backend>
void ReinterpretTensorList(TensorList<Backend> &tl, DALIDataType new_type) {
  tl.Reinterpret(new_type);
}

void ExposeTensorLayout(py::module &m) {
  py::class_<TensorLayout> tl(m, "TensorLayout");
  tl.def(py::init([](const std::string &s) {
    return new TensorLayout(s);
  }))
  .def("__str__", &TensorLayout::str)
  .def("__repr__", TensorLayoutRepr)
  .def("__len__", &TensorLayout::ndim);
#define DEFINE_LAYOUT_CMP(name, expr)\
    tl.def("__" #name "__", [](const TensorLayout &self, const TensorLayout *other) {\
      return expr;\
    })\
    .def("__" #name "__", [](const TensorLayout &self, const string *other) {\
      return expr;\
    })
  DEFINE_LAYOUT_CMP(eq, other  && self == *other);
  DEFINE_LAYOUT_CMP(ne, !other || self != *other);  // null is not equal to non-null
  DEFINE_LAYOUT_CMP(lt, !other || self <  *other);  // null precedes non-null
  DEFINE_LAYOUT_CMP(gt, other  && self >  *other);
  DEFINE_LAYOUT_CMP(le, !other || self <= *other);
  DEFINE_LAYOUT_CMP(ge, other  && self >= *other);
#undef DEFINE_LAYOUT_CMP
}

// Placeholder enum for defining __call__ on dtype member of Tensor (to be deprecated).
enum DALIDataTypePlaceholder {};

/**
 * @brief Extracts attribute named `attr_name` from the python object.
 *
 * @param object python object.
 * @param attr_name name of the requested attribute.
 */
auto ExtractPythonAttr(py::object &&object, const char *attr_name) {
  return object.attr(attr_name);
}

/**
 * @brief Extracts nested attribute from the python object.
 *
 * @param object python object.
 * @param attr_name name of the next requested attribute.
 * @param rest rest of the requested attributes names.
 */
template <typename... Args>
auto ExtractPythonAttr(py::object &&object, const char *attr_name, Args... rest) {
  return ExtractPythonAttr(object.attr(attr_name), rest...);
}

/**
 * @brief Extracts nested attribute from imported python module.
 *
 * @param python_module name of the python module.
 * @param attr_name outer most attribute name.
 * @param attr_names rest of the attribute names.
 */
template <typename... Args>
auto FromPythonTrampoline(const char *python_module, const char *attr_name, Args... attr_names) {
  return ExtractPythonAttr(py::module::import(python_module).attr(attr_name), attr_names...);
}

/**
 * @brief Extracts attribute from imported python module.
 *
 * @param python_module name of the python module.
 * @param attr_name name of the attribute.
 */
auto FromPythonTrampoline(const char *python_module, const char *attr_name) {
  return py::module::import(python_module).attr(attr_name);
}

/**
 * @brief Copies the contents of the source DALI batch to an external buffer
 *
 * The function schedules a copy of the contents of src to the target destination buffer.
 * The copy will be scheduled on the provided `cuda_stream` or, if left out, on an internal DALI
 * stream.
 * If a non-blocking copy is requested, the function will synchronize the source buffer's
 * associated access order with the provided stream; otherwise, the function will wait until the
 * copy completes.
 *
 * @tparam SourceObject  a data store on GPUBackend (Tensor, TensorList, TensorList)
 * @param src             Source batch
 * @param dst_ptr         Destination pointer, wrapped in a C void_ptr Python type
 * @param cuda_stream     CUDA stream, wrapped in a C void_ptr type
 * @param non_blocking    whether the function should wait on host for the copy to complete
 * @param use_copy_kernel if true, the copy will be done using a kernel instead of cudaMemcpyAsync
 */
template <typename SourceObject>
void CopyToExternalImplGPU(SourceObject &src,
                           py::object dst_ptr, py::object cuda_stream,
                           bool non_blocking, bool use_copy_kernel) {
  CUDAStreamLease lease;
  AccessOrder copy_order;
  AccessOrder wait_order = non_blocking ? src.order() : AccessOrder::host();
  int device = src.device_id();
  if (!cuda_stream.is_none()) {
    cudaStream_t stream = static_cast<cudaStream_t>(ctypes_void_ptr(cuda_stream));
    copy_order = AccessOrder(stream, device);
  } else {
    lease = CUDAStreamPool::instance().Get(device);
    copy_order = AccessOrder(lease, device);
  }

  void *ptr = ctypes_void_ptr(dst_ptr);
  CopyToExternal<mm::memory_kind::device>(ptr, std::nullopt, src, copy_order, use_copy_kernel);

  wait_order.wait(copy_order);
}

template <typename Backend>
py::object GetTensorProperty(const Tensor<Backend> &tensor, std::string name) {
  if (name == "layout") {
    TensorLayout layout = tensor.GetLayout();
    if (layout.empty())
      return py::none();
    else
      return py::str(layout.c_str());
  } else if (name == "source_info") {
    auto &&srcinfo = tensor.GetSourceInfo();
    if (srcinfo.empty())
      return py::none();
    else
      return py::str(srcinfo);
  } else {
    // TODO(michalz): Make TensorMeta more flexible and have some dictionary
    return py::none();
  }
}

template <typename Backend>
DLDevice GetDLDevice(const Tensor<Backend> &tensor) {
  if constexpr (std::is_same_v<Backend, GPUBackend>)
    return { kDLCUDA, tensor.device_id() };
  else
    return { tensor.is_pinned() ? kDLCUDAHost : kDLCPU };
}

template <typename Backend>
DLMTensorPtr ToDLMTensor(Tensor<Backend> &tensor,
                         std::optional<intptr_t> stream_handle_value,
                         std::optional<std::pair<DLDeviceType, int>> dl_device) {
  DLDevice dev;

  if (dl_device.has_value())
    dev = { dl_device->first, dl_device->second };
  else
    dev = GetDLDevice(tensor);

  if (dev.device_type == kDLCUDA) {
    AccessOrder target_order = cudaStreamLegacy;
    if (stream_handle_value.has_value()) {
      if (*stream_handle_value == -1) {
        target_order = AccessOrder{};
      } else {
        cudaStream_t stream = cudaStream_t(*stream_handle_value);
        if (stream == 0)
          throw std::invalid_argument("Stream 0 is explicitly forbidden by DLPack protocol");
        target_order = AccessOrder(stream, dev.device_id);
      }
    }

    if constexpr (std::is_same_v<Backend, CPUBackend>) {
      throw std::runtime_error(
          "The tensor is in CPU memory and a CUDA DLPack tensor was requested");
    }
    if (dev.device_id != tensor.device_id())
      throw std::runtime_error(make_string("Requested a DLPack tensor for GPU_", dev.device_id,
        "while the tensor resides in GPU_", tensor.device_id(), " memory."));

    if (target_order) {
      if (tensor.ready_event()) {
        target_order.wait(tensor.ready_event());
      } else {
        target_order.wait(tensor.order());
      }
    }
    return GetSharedDLTensor(tensor);
  } else if (dev.device_type == kDLCPU || dev.device_type == kDLCUDAHost) {
    if constexpr (std::is_same_v<Backend, GPUBackend>) {
      throw std::runtime_error(
          "The tensor is in CUDA GPU memory and a CPU DLPack tensor was requested");
    }

    if (dev.device_type == kDLCUDAHost && !tensor.is_pinned())
      throw std::runtime_error(
          "A CUDA host (pinned) DLPack was requested, but the tensor buffer is not pinned.");

    if (tensor.is_pinned() && tensor.ready_event()) {
      // DLPack doesn't support stream-ordered CUDA host tensors
      AccessOrder::host().wait(tensor.ready_event());
    }

    DLMTensorPtr dlm_tensor = GetSharedDLTensor(tensor);
    // Set the device type to the desired one - if the original tensor was pinned, we can
    // downgrade it to regular host memory.
    dlm_tensor->dl_tensor.device = dev;
    return dlm_tensor;
  } else {
    throw std::runtime_error(make_string(
      "An unsupported DLPack device was requested: ", static_cast<int>(dev.device_type)));
  }
}

template <typename Backend>
py::capsule ToDLPack(Tensor<Backend> &tensor,
                     std::optional<intptr_t> stream,
                     std::optional<std::pair<DLDeviceType, int>> dl_device) {
  return DLTensorToCapsule([&]() {
    py::gil_scoped_release interpreter_unlock{};
    return ToDLMTensor(tensor, stream, dl_device);
  }());
}

AccessOrder AccessOrderFromPythonStreamObj(const py::object &cuda_stream) {
  AccessOrder order;
  if (!cuda_stream.is_none()) {
    auto cuda_stream_interface = getattr(cuda_stream, "__cuda_stream__", py::none());
    if (!cuda_stream_interface.is_none()) {
      auto [version, stream_ptr] = cuda_stream_interface().cast<std::tuple<int, uintptr_t>>();
      cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
      order = AccessOrder(stream);
    } else if (py::hasattr(cuda_stream, "value")) {
      cudaStream_t stream = static_cast<cudaStream_t>(ctypes_void_ptr(cuda_stream));
    } else if (py::isinstance<py::int_>(cuda_stream)) {
      cudaStream_t stream = reinterpret_cast<cudaStream_t>(py::cast<uintptr_t>(cuda_stream));
      order = AccessOrder(stream);
    }
  } else {
    order = AccessOrder::host();
  }
  return order;
}

/**
 * Pipeline output descriptor.
 */
using OutputDesc = std::tuple<std::string  /* name */,
                              std::string  /* device */,
                              DALIDataType /* dtype */,
                              int          /* ndim */,
                              std::string  /* layout */>;

void ExposeTensor(py::module &m) {
  m.def("CheckDLPackCapsule",
        [](py::object &p) {
          py::list list;
          if (PyCapsule_CheckExact(p.ptr())) {
            py::capsule capsule = py::reinterpret_borrow<py::capsule>(p);
            // do not consume capsule
            auto dlm_tensor_ptr = DLMTensorRawPtrFromCapsule(capsule, false);
            const auto &dl_tensor = dlm_tensor_ptr->dl_tensor;
            list.append(dl_tensor.device.device_type == kDLCUDA ||
                        dl_tensor.device.device_type == kDLCPU);
            list.append(dl_tensor.device.device_type == kDLCUDA);
          } else {
            list.append(false);
            list.append(false);
          }
          return py::cast<py::tuple>(list);
        },
      "ptr"_a,
      R"code(
      Check if provided python object represent a valid DLPack capsule.
      It returns a tuple of two boolean values: one indicating if this is a valid DLPack object, and the other if the data

      p : python object
          Python object to be checked
      )code");

  auto tensor_cpu_binding = py::class_<Tensor<CPUBackend>>(m, "TensorCPU", py::buffer_protocol())
    .def_property_readonly_static("__module__", tensor_module_impl)
    .def(py::init([](py::capsule &capsule, std::optional<std::string> layout = {}) {
          DomainTimeRange range("TensorCPU::init", kCPUTensorColor);
          auto t = std::make_unique<Tensor<CPUBackend>>();
          FillTensorFromDlPack(capsule, t.get(), layout);
          return t.release();
        }),
      "object"_a,
      "layout"_a = py::none(),
      R"code(
      Wrap a DLPack Tensor residing in the CPU memory.

      object : DLPack object
            Python DLPack object
      layout : str
            Layout of the data
      )code")
    .def(
      "__dlpack_device__", [](const Tensor<CPUBackend> &tensor) {
        auto dev = GetDLDevice(tensor);
        return std::make_tuple(dev.device_type, dev.device_id);
      },
      R"code(
      Returns device type and device ID in DLPack format.
      )code")
    .def(
      "__dlpack__", ToDLPack<CPUBackend>,
      "stream"_a = py::none(),
      "dl_device"_a = py::none(),
      R"code(
      Exposes the tensor as a DLPack capsule.

      Note:
        When NOT using the default execution model (i.e., when ``exec_dynamic=False`` or other
        parameters are incompatible with this execution mode), the pipeline outputs may be reused
        and overwritten by DALI after ``release_outputs`` has been called. Make sure that the
        default execution model is enabled if you want to keep the outputs indefinitely.

      stream : int, None
          The CUDA stream the the caller is going to use to access the buffer.
          A synchronization event might be inserted, if necessary, into that stream.
          Special values:

          * ``None`` - any stream; wait on host
          * ``-1``   - do not synchronize at all
          * ``1``    - legacy default stream
          * ``2``    - legacy per-thread stream
          * ``>2``   - a CUDA stream handle converted to an integer
          * ``0``    - forbidden value
      )code")
    .def_buffer([](Tensor<CPUBackend> &t) -> py::buffer_info {
          DALI_ENFORCE(IsValidType(t.type()), "Cannot produce "
            "buffer info for tensor w/ invalid type.");

          std::vector<ssize_t> shape(t.ndim()), stride(t.ndim());
          size_t dim_prod = 1;
          for (int i = 0; i < t.ndim(); ++i) {
            shape[i] = t.shape()[i];

            // We iterate over stride backwards
            stride[(t.ndim()-1) - i] = t.type_info().size()*dim_prod;
            dim_prod *= t.shape()[(t.ndim()-1) - i];
          }

          if (t.is_pinned()) {
            if (auto &event = t.ready_event())
              AccessOrder::host().wait(event);  // more fine-grained synchronization
            else
              AccessOrder::host().wait(t.order());
          }

          return py::buffer_info(
              t.raw_mutable_data(),
              t.type_info().size(),
              FormatStrFromType(t.type()),
              t.ndim(), shape, stride);
        })
    .def(py::init([](py::buffer b, std::optional<std::string> layout = {}, bool is_pinned = false) {
          // We need to verify that the input data is c contiguous
          // and of a type that we can work with in the backend
          __backend_impl_force_tls_align_fun();
          py::buffer_info info = b.request();

          std::vector<Index> i_shape;
          for (auto &dim : info.shape) {
            i_shape.push_back(dim);
          }
          size_t bytes = volume(i_shape) * info.itemsize;

          // Validate the stride
          CheckContiguousTensor(info.strides, info.shape, info.itemsize);

          // TODO(klecki): Extend the constructor with stream and device_id
          // Assume that we cannot use pinned memory in CPU_ONLY mode
          int device_id = CPU_ONLY_DEVICE_ID;
          if (is_pinned) {
            CUDA_CALL(cudaGetDevice(&device_id));
          }

          // Create the Tensor and wrap the data
          auto t = std::make_unique<Tensor<CPUBackend>>();
          const TypeInfo &type = TypeFromFormatStr(info.format);
          // Keep a copy of the input buffer ref in the deleter, so its refcount is increased
          // while this shared_ptr is alive (and the data should be kept alive)
          // Use dynamically allocated memory so we can call deleter inside py::gil_scoped_acquire
          // scope
          py::buffer *buf_tmp = new py::buffer(b);
          t->ShareData(shared_ptr<void>(info.ptr, [buf_ref = buf_tmp](void *) {
             py::gil_scoped_acquire aqr;
             delete buf_ref;
          }),
                       bytes, is_pinned, i_shape, type.id(), device_id);
          SetLayout(*t, layout);
          return t.release();
        }),
      "b"_a,
      "layout"_a = py::none(),
      "is_pinned"_a = false,
      R"code(
      Wrap a Tensor residing in the CPU memory.

      b : object
            the buffer to wrap into the TensorListCPU object
      layout : str
            Layout of the data
      is_pinned : bool
            If provided memory is page-locked (pinned)
      )code")
    .def("shape", &py_shape<CPUBackend>,
         R"code(
         Shape of the tensor.
         )code")
    .def("ndim", &Tensor<CPUBackend>::ndim,
         R"code(
         Number of dimensions of the tensor.
         )code")
    .def("squeeze",
      [](Tensor<CPUBackend> &t, py::object dim_arg) -> bool {
        if (!dim_arg.is_none()) {
          int dim = dim_arg.cast<int>();
          return t.Squeeze(dim);
        }
        return t.Squeeze();
      },
      "dim"_a = py::none(),
      R"code(
      Remove single-dimensional entries from the shape of the Tensor and it returns true
      if the shape changed or false if it remained unchanged.

      dim : int
            If specified, it represents the axis of a single dimension to be squeezed.
      )code")
    .def("layout", [](Tensor<CPUBackend> &t) {
      return t.GetLayout().str();
    })
    .def("set_layout", [](Tensor<CPUBackend> &t, const std::optional<std::string> &layout) {
      SetLayout(t, layout);
    })
    .def("source_info", &Tensor<CPUBackend>::GetSourceInfo,
        R"(Gets a string descrbing the source of the data in the tensor, e.g. a name of the file
        from which the data was loaded.)")
    .def("get_property", GetTensorProperty<CPUBackend>)
    .def("_as_gpu", [](Tensor<CPUBackend> &t) -> Tensor<GPUBackend>* {
          auto ret = std::make_unique<Tensor<GPUBackend>>();
          int dev = -1;
          CUDA_CALL(cudaGetDevice(&dev));
          ret->set_device_id(dev);
          UserStream *us = UserStream::Get();
          cudaStream_t s = us->GetStream(*ret);
          ret->Copy(t, s);
          us->Wait(*ret);
          return ret.release();
        },
      R"code(
      Returns a `TensorGPU` object being a copy of this `TensorCPU`.
      )code",
      py::return_value_policy::take_ownership)
    .def("as_cpu", [](Tensor<CPUBackend> &t) -> Tensor<CPUBackend>& {
          return t;
        },
      R"code(Passthrough, since the object is already an instance of `TensorCPU`.)code",
      py::return_value_policy::reference_internal)
    .def("_set_stream", [](Tensor<CPUBackend> &t, py::object stream) {
      t.set_order(AccessOrderFromPythonStreamObj(stream));
    })
    .def("_make_copy", [](const Tensor<CPUBackend> &t) {
        auto dst = std::make_unique<Tensor<CPUBackend>>();
        dst->set_device_id(t.device_id());
        dst->set_order(t.order());
        dst->set_pinned(t.is_pinned());
        dst->Copy(t);
        return dst;
      },
      py::return_value_policy::take_ownership)
    .def("copy_to_external",
        [](Tensor<CPUBackend> &t, py::object p) {
          CopyToExternal<mm::memory_kind::host>(
              ctypes_void_ptr(p), std::nullopt, t, AccessOrder::host(), false);
        },
      "ptr"_a,
      R"code(
      Copy to external pointer in the CPU memory.

      ptr : ctypes.c_void_p
            Destination of the copy.
      )code")
    .def("data_ptr", [](Tensor<CPUBackend> &t) {
          return py::reinterpret_borrow<py::object>(PyLong_FromVoidPtr(t.raw_mutable_data()));
        },
      R"code(
      Returns the address of the first element of tensor.
      )code")
    .def("reinterpret", ReinterpretTensor<CPUBackend>,
      "new_type"_a,
      R"code(
      Reinterpret the contents of the tensor as a new type. The element size must not change.
      )code")
    .def("__str__", [](Tensor<CPUBackend> &t) {
      return FromPythonTrampoline("nvidia.dali.tensors", "_tensor_to_string")(t);
    })
    .def("__repr__", [](Tensor<CPUBackend> &t) {
      return FromPythonTrampoline("nvidia.dali.tensors", "_tensor_to_string")(t, false);
    })
    .def_property("__array_interface__", &ArrayInterfaceRepr<CPUBackend>, nullptr,
      R"code(
      Returns Array Interface representation of TensorCPU.
      )code")
    .def_property_readonly("dtype", [](Tensor<CPUBackend> &t) {
          return static_cast<DALIDataTypePlaceholder>(t.type());
        },
      R"code(
      Data type of the TensorCPU's elements.

      :type: DALIDataType
      )code");
  tensor_cpu_binding.doc() = R"code(
      Class representing a Tensor residing in host memory. It can be used to access individual
      samples of a :class:`TensorListCPU` or used to wrap CPU memory that is intended
      to be passed as an input to DALI.

      It is compatible with `Python Buffer Protocol <https://docs.python.org/3/c-api/buffer.html>`_,
      `NumPy Array Interface <https://numpy.org/doc/stable/reference/arrays.interface.html>`_
      and `DLPack <https://github.com/dmlc/dlpack>`_.)code";

  py::implicitly_convertible<py::buffer, Tensor<CPUBackend>>();
  py::implicitly_convertible<py::capsule&, Tensor<CPUBackend>>();

  auto tensor_gpu_binding = py::class_<Tensor<GPUBackend>>(m, "TensorGPU")
    .def_property_readonly_static("__module__", tensor_module_impl)
    .def(py::init([](
              py::capsule &capsule,
              std::optional<std::string> layout = {},
              py::object stream = py::none()) {
          DomainTimeRange range("TensorGPU::init from capsule", kGPUTensorColor);
          auto t = std::make_unique<Tensor<GPUBackend>>();
          FillTensorFromDlPack(capsule, t.get(), layout);
          if (!stream.is_none())  // use a separately provided stream - there's none in the capsule
            t->set_order(AccessOrderFromPythonStreamObj(stream));
          return t.release();
        }),
      "object"_a,
      "layout"_a = py::none(),
      "stream"_a = py::none(),
      R"code(
      Wrap a DLPack Tensor residing in the GPU memory.

      object : DLPack object
            Python DLPack object
      layout : str
            Layout of the data
      stream : dali.Stream, int, ctypes_void_ptr, None
            Stream to accociate the tensor with
      )code")
    .def(
      "device_id", &Tensor<GPUBackend>::device_id)
    .def(
      "__dlpack_device__", [](const Tensor<GPUBackend> &tensor) {
        auto dev = GetDLDevice(tensor);
        return std::make_tuple(dev.device_type, dev.device_id);
      },
      R"code(
      Returns device type and device ID in DLPack format.
      )code")
    .def(
      "__dlpack__", ToDLPack<GPUBackend>,
      "stream"_a = py::none(),
      "dl_device"_a = py::none(),
      R"code(
      Exposes the tensor as a DLPack capsule.

      Note:
        When NOT using the default execution model (i.e., when ``exec_dynamic=False`` or other
        parameters are incompatible with this execution mode), the pipeline outputs may be reused
        and overwritten by DALI after ``release_outputs`` has been called. Make sure that the
        default execution model is enabled if you want to keep the outputs indefinitely.

      stream : int, None
          The CUDA stream the the caller is going to use to access the buffer.
          A synchronization event might be inserted, if necessary, into that stream.
          Special values:

          * ``None`` - any stream; wait on host
          * ``-1``   - do not synchronize at all
          * ``1``    - legacy default stream
          * ``2``    - legacy per-thread stream
          * ``>2``   - a CUDA stream handle converted to an integer
          * ``0``    - forbidden value
      )code")
    .def(py::init([](const py::object &object,
                     const std::optional<std::string> &layout = {},
                     int device_id = -1) {
          DomainTimeRange range("TensorGPU::init from CUDA array", kGPUTensorColor);
          auto t = std::make_unique<Tensor<GPUBackend>>();
          FillTensorFromCudaArray(object, t.get(), device_id, layout);
          return t.release();
        }),
      "object"_a,
      "layout"_a = py::none(),
      "device_id"_a = -1,
      R"code(
      Wrap a Tensor residing in the GPU memory that implements CUDA Array Interface.

      object : object
            Python object that implements CUDA Array Interface
      layout : str
            Layout of the data
      device_id: int
            Device of where this tensor resides. If not provided, the current device is used.
      )code")
    .def("shape", &py_shape<GPUBackend>,
         R"code(
         Shape of the tensor.
         )code")
    .def("ndim", &Tensor<GPUBackend>::ndim,
         R"code(
         Number of dimensions of the tensor.
         )code")
    .def("layout", [](Tensor<GPUBackend> &t) {
      return t.GetLayout().str();
    })
    .def("set_layout", [](Tensor<GPUBackend> &t, const std::optional<std::string> &layout) {
      SetLayout(t, layout);
    })
    .def("source_info", &Tensor<GPUBackend>::GetSourceInfo,
        R"(Gets a string descrbing the source of the data in the tensor, e.g. a name of the file
        from which the data was loaded.)")
    .def("get_property", GetTensorProperty<GPUBackend>)
    .def("as_cpu", [](Tensor<GPUBackend> &t) -> Tensor<CPUBackend>* {
          DeviceGuard g(t.device_id());
          auto ret = std::make_unique<Tensor<CPUBackend>>();
          ret->set_pinned(false);
          ret->set_order(AccessOrder::host());
          UserStream * us = UserStream::Get();
          cudaStream_t s = us->GetStream(t);
          ret->Copy(t, s);
          us->Wait(t);
          return ret.release();
        },
      R"code(
      Returns a `TensorCPU` object being a copy of this `TensorGPU`.
      )code",
      py::return_value_policy::take_ownership)
    .def("_set_stream", [](Tensor<GPUBackend> &t, py::object stream) {
      t.set_order(AccessOrderFromPythonStreamObj(stream));
    })
    .def("_make_copy", [](const Tensor<GPUBackend> &t) {
        DeviceGuard dg(t.device_id());
        auto dst = std::make_unique<Tensor<GPUBackend>>();
        dst->set_device_id(t.device_id());
        dst->set_order(t.order());
        dst->Copy(t);
        return dst;
      },
      py::return_value_policy::take_ownership)
    .def("squeeze",
      [](Tensor<GPUBackend> &t, py::object dim_arg) -> bool {
        if (!dim_arg.is_none()) {
          int dim = dim_arg.cast<int>();
          return t.Squeeze(dim);
        }
        return t.Squeeze();
      },
      "dim"_a = py::none(),
      R"code(
      Remove single-dimensional entries from the shape of the Tensor and it returns true
      if the shape changed or false if it remained unchanged.

      dim : int
            If specified, it represents the axis of a single dimension to be squeezed.
      )code")
    .def("reinterpret", ReinterpretTensor<GPUBackend>,
      "new_type"_a,
      R"code(
      Reinterpret the contents of the tensor as a new type. The element size must not change.
      )code")
    .def("copy_to_external",
        [](Tensor<GPUBackend> &t, py::object p, py::object cuda_stream,
           bool non_blocking, bool use_copy_kernel) {
          CopyToExternalImplGPU(t, p, cuda_stream, non_blocking, use_copy_kernel);
        },
      "ptr"_a,
      "cuda_stream"_a = py::none(),
      "non_blocking"_a = false,
      "use_copy_kernel"_a = false,
      R"code(
      Copy to external pointer in the GPU memory.

      ptr : ctypes.c_void_p
            Destination of the copy.
      cuda_stream : ctypes.c_void_p
            CUDA stream to schedule the copy on (default stream if not provided).
      non_blocking : bool
            Asynchronous copy.
      )code")
    .def_property_readonly("stream", [](const Tensor<GPUBackend> &t)->py::object {
      if (t.order().is_device())
        return py::reinterpret_borrow<py::object>(PyLong_FromVoidPtr(t.order().stream()));
      else
        return py::none();
    })
    .def("data_ptr",
        [](Tensor<GPUBackend> &t) {
          return py::reinterpret_borrow<py::object>(PyLong_FromVoidPtr(t.raw_mutable_data()));
        },
      R"code(
      Returns the address of the first element of tensor.
      )code")
    .def("__str__", [](Tensor<GPUBackend> &t) {
      return FromPythonTrampoline("nvidia.dali.tensors", "_tensor_to_string")(t);
    })
    .def("__repr__", [](Tensor<GPUBackend> &t) {
      return FromPythonTrampoline("nvidia.dali.tensors", "_tensor_to_string")(t, false);
    })
    .def_property("__cuda_array_interface__",  &ArrayInterfaceRepr<GPUBackend>, nullptr,
      R"code(
      Returns CUDA Array Interface (Version 2) representation of TensorGPU.
      )code")
    .def_property_readonly("dtype", [](Tensor<GPUBackend> &t) {
          return static_cast<DALIDataTypePlaceholder>(t.type());
        },
      R"code(
      Data type of the TensorGPU's elements.

      :type: DALIDataType
      )code");

  py::implicitly_convertible<py::object, Tensor<GPUBackend>>();
  py::implicitly_convertible<py::capsule&, Tensor<GPUBackend>>();

  tensor_gpu_binding.doc() = R"code(
      Class representing a Tensor residing in GPU memory. It can be used to access individual
      samples of a :class:`TensorListGPU` or used to wrap GPU memory that is intended
      to be passed as an input to DALI.

      It is compatible with `CUDA Array Interface <https://numba.pydata.org/numba-doc/dev/cuda/cuda_array_interface.html>`_
      and `DLPack <https://github.com/dmlc/dlpack>`_.)code";
}

template <typename Backend>
std::unique_ptr<Tensor<Backend> > TensorListGetItemImpl(TensorList<Backend> &t, Index id) {
  int num_tensors = static_cast<int>(t.num_samples());
  if (id < 0) {
    id = num_tensors + id;
  }
  if (id >= num_tensors || id < 0) {
      throw py::index_error("TensorListCPU index out of range");
  }
  auto ptr = std::make_unique<Tensor<Backend>>();
  // TODO(klecki): Rework this with proper sample-based tensor batch data structure
  auto &sample_shared_ptr = unsafe_sample_owner(t, id);
  auto &tshape = t.tensor_shape(id);
  size_t num_bytes = tshape.num_elements() * t.type_info().size();
  ptr->ShareData(sample_shared_ptr, num_bytes, t.is_pinned(), tshape, t.type(),
                 t.device_id(), t.order(), t.ready_event());
  ptr->SetMeta(t.GetMeta(id));
  return ptr;
}

template <typename Backend>
std::shared_ptr<TensorList<Backend>> TensorListFromListOfTensors(
      py::list &list_of_tensors,
      const std::optional<std::string> &layout = {},
      bool contiguous = true) {
  DomainTimeRange range("TensorListFromListOfTensors");
  if (list_of_tensors.empty()) {
    auto ptr = std::make_shared<TensorList<Backend>>();
    if (layout.has_value()) {
      ptr->set_sample_dim(layout->length());
      ptr->SetLayout(*layout);
    }
    return ptr;
  }

  std::optional<TensorList<Backend>> non_contiguous_tmp;
  std::shared_ptr<TensorList<Backend>> non_contiguous_out;

  if (contiguous)
    non_contiguous_tmp = TensorList<Backend>(list_of_tensors.size());
  else
    non_contiguous_out = std::make_shared<TensorList<Backend>>(list_of_tensors.size());

  TensorList<Backend> &non_contiguous = contiguous
    ? non_contiguous_tmp.value()
    : *non_contiguous_out;

  int expected_type = -2;

  AccessOrder wait_order = AccessOrder::host();
  AccessOrder copy_order = AccessOrder::host();

  {
    DomainTimeRange range("Build initial list");

    for (size_t i = 0; i < list_of_tensors.size(); ++i) {
      try {
        auto &t = list_of_tensors[i].cast<Tensor<Backend> &>();
        if (i == 0) {
          non_contiguous.SetupLike(t);
          if constexpr (std::is_same_v<Backend, GPUBackend>) {
            copy_order = AccessOrder(UserStream::Get()->GetStream(t));
          }
        }
        DALIDataType cur_type = t.type();

        if (expected_type == -2) {
          expected_type = t.type();
        } else if (expected_type != cur_type) {
          throw py::type_error(make_string(
              "Tensors cannot have different data types. Tensor at position ", i, " has type '",
              cur_type, "' expected to have type '", DALIDataType(expected_type), "'."));
        }
        non_contiguous.SetSample(i, t);
      } catch (const py::type_error &) {
        throw;
      } catch (const std::runtime_error &) {
        throw py::type_error(make_string("Object at position ", i, " cannot be converted to Tensor",
                                         std::is_same_v<Backend, GPUBackend> ? "GPU." : "CPU."));
      }
    }
  }

  if (!contiguous) {
    SetLayout(non_contiguous, layout, false);
    copy_order.wait(wait_order);
    return non_contiguous_out;
  }

  {
    DomainTimeRange range("Copy to contiguous");
    auto contiguous_out = std::make_shared<TensorList<Backend>>();
    contiguous_out->SetContiguity(BatchContiguity::Contiguous);
    contiguous_out->set_pinned(non_contiguous.is_pinned());
    contiguous_out->Copy(non_contiguous, copy_order);
    SetLayout(*contiguous_out, layout, false);
    copy_order.wait(wait_order);
    return contiguous_out;
  }
}

template <typename Backend>
using tensor_list_py_class_t =
    py::class_<TensorList<Backend>, std::shared_ptr<TensorList<Backend>>>;

void ExposeTensorListCPU(py::module &m) {
    auto tensor_list_cpu_class =
      py::class_<TensorList<CPUBackend>, std::shared_ptr<TensorList<CPUBackend>>>(
          m, "TensorListCPU", py::buffer_protocol())
    .def_property_readonly_static("__module__", tensor_module_impl)
    .def(py::init([](py::capsule &capsule, std::optional<std::string> layout = {}) {
            DomainTimeRange range("TensorListCPU::init from capsule", kCPUTensorColor);
            auto t = std::make_shared<TensorList<CPUBackend>>();
            FillTensorFromDlPack(capsule, t.get(), layout);
            return t;
          }),
        "object"_a,
        "layout"_a = py::none(),
        R"code(
        List of tensors residing in the CPU memory.

        object : DLPack object
              Python DLPack object representing TensorList
        layout : str
              Layout of the data
        )code")
    .def(py::init([](TensorList<CPUBackend> *tl, std::optional<std::string> layout = {}) {
          DomainTimeRange range("TensorListCPU::init from a list of tensors", kCPUTensorColor);
          if (!tl)
            throw py::value_error("The source object must not be null");
          auto t = std::make_shared<TensorList<CPUBackend>>();
          t->ShareData(*tl);
          // If layout is not given, use the one from tl
          SetLayout(*t, layout, false);
          return t;
        }),
      "tl"_a,
      "layout"_a = py::none())
    .def(py::init([](py::buffer b, std::optional<std::string> layout = {}, bool is_pinned = false) {
         DomainTimeRange range("TensorListCPU::init from a buffer", kCPUTensorColor);
        // We need to verify that the input data is C_CONTIGUOUS
        // and of a type that we can work with in the backend
        py::buffer_info info = b.request();

        DALI_ENFORCE(info.shape.size() > 0, "Cannot create TensorList from 0-dim array.");

        // Create a list of shapes
        std::vector<Index> tensor_shape(info.shape.size()-1);
        for (size_t i = 1; i < info.shape.size(); ++i) {
          tensor_shape[i-1] = info.shape[i];
        }
        auto i_shape = uniform_list_shape(info.shape[0], tensor_shape);
        size_t bytes = volume(tensor_shape)*i_shape.size()*info.itemsize;

        // Validate the stride
        CheckContiguousTensor(info.strides, info.shape, info.itemsize);

        // TODO(klecki): Extend the constructor with stream and device_id
        // Assume that we cannot use pinned memory in CPU_ONLY mode
        int device_id = CPU_ONLY_DEVICE_ID;
        if (is_pinned) {
          CUDA_CALL(cudaGetDevice(&device_id));
        }

        // Create the Tensor and wrap the data
        auto t = std::make_shared<TensorList<CPUBackend>>();
        const TypeInfo &type = TypeFromFormatStr(info.format);
        // Keep a copy of the input buffer ref in the deleter, so its refcount is increased
        // while this shared_ptr is alive (and the data should be kept alive)
        t->ShareData(shared_ptr<void>(info.ptr, [buf_ref = b](void *) mutable {  // NOLINT
              py::gil_scoped_acquire aqr;
              {
                auto tmp = std::move(buf_ref);
                (void)tmp;
              }
            }), bytes, is_pinned, i_shape, type.id(), device_id);
        SetLayout(*t, layout);
        return t;
      }),
      "b"_a,
      "layout"_a = py::none(),
      "is_pinned"_a = false,
      R"code(
      List of tensors residing in the CPU memory.

      b : object
            the buffer to wrap into the TensorListCPU object
      layout : str
            Layout of the data
      is_pinned : bool
            If provided memory is page-locked (pinned)
      )code")
    .def(py::init([](
          py::list &list_of_tensors,
          std::optional<std::string> layout = {},
          bool contiguous = true) {
        DomainTimeRange range("TensorListCPU::init from a Python list of tensors", kCPUTensorColor);
        return TensorListFromListOfTensors<CPUBackend>(list_of_tensors, layout, contiguous);
      }),
      "list_of_tensors"_a,
      "layout"_a = py::none(),
      "contiguous"_a = true,
      R"code(
      List of tensors residing in the CPU memory.

      list_of_tensors : [TensorCPU]
            Python list of TensorCPU objects
      layout : str or None
            Layout of the data
      contiguous : bool = True
            If True, the list of tensors is converted to a contiguous TensorListCPU, necessarily
            creating a copy. Otherwise, the copy may be avoided.
      )code")
    .def_static("broadcast", [](const Tensor<CPUBackend> &t, int num_samples) {
        return std::make_shared<TensorList<CPUBackend>>(t, num_samples);
      })
    .def("_as_gpu", [](TensorList<CPUBackend> &t) {
          auto ret = std::make_shared<TensorList<GPUBackend>>();
          int dev = -1;
          CUDA_CALL(cudaGetDevice(&dev));
          ret->set_device_id(dev);
          UserStream *us = UserStream::Get();
          cudaStream_t s = us->GetStream(*ret);
          ret->Copy(t, s);
          us->Wait(*ret);
          return ret;
        },
      R"code(
      Returns a `TensorListGPU` object being a copy of this `TensorListCPU`.
      )code",
      py::return_value_policy::take_ownership)
    .def("as_cpu", [](TensorList<CPUBackend> &t) -> TensorList<CPUBackend> & {
        return t;
      }, R"code(Passthrough, as it is already an instance of `TensorListCPU`.)code",
      py::return_value_policy::reference_internal)
    .def("_set_stream", [](TensorList<CPUBackend> &t, py::object stream) {
      t.set_order(AccessOrderFromPythonStreamObj(stream));
    })
    .def("_make_copy", [](const TensorList<CPUBackend> &t) {
        auto dst = std::make_shared<TensorList<CPUBackend>>();
        dst->set_device_id(t.device_id());
        dst->set_order(t.order());
        dst->set_pinned(t.is_pinned());
        dst->Copy(t);
        return dst;
      })
    .def("layout", [](TensorList<CPUBackend> &t) {
      return t.GetLayout().str();
    })
    .def("set_layout", [](TensorList<CPUBackend> &t, const std::optional<std::string> &layout) {
      SetLayout(t, layout);
    })
    .def("shape", &py_shape_list<CPUBackend>,
      R"code(
      Shape of the tensor list.
      )code")
    .def("ndim", [](TensorList<CPUBackend> &tl) { return tl.shape().sample_dim(); },
      R"code(
      Number of dimensions of the tensors in the list.
      )code")
    .def("at", [](TensorList<CPUBackend> &tl, Index id) -> py::array {
          DALI_ENFORCE(IsValidType(tl.type()), "Cannot produce "
              "buffer info for tensor w/ invalid type.");
          DALI_ENFORCE(id < tl.num_samples(), "Index is out-of-range.");
          DALI_ENFORCE(id >= 0, "Index is out-of-range.");

          std::vector<ssize_t> shape(tl.tensor_shape(id).size()),
                                     stride(tl.tensor_shape(id).size());
          size_t dim_prod = 1;
          for (size_t i = 0; i < shape.size(); ++i) {
            shape[i] = tl.tensor_shape(id)[i];

            // We iterate over stride backwards
            stride[(stride.size()-1) - i] = tl.type_info().size()*dim_prod;
            dim_prod *= tl.tensor_shape(id)[(shape.size()-1) - i];
          }

          return py::array(py::buffer_info(
              tl.raw_mutable_tensor(id),
              tl.type_info().size(),
              FormatStrFromType(tl.type()),
              shape.size(), shape, stride));
        },
      R"code(
      Returns tensor at given position in the list.

      )code")
      .def("__getitem__",
        [](TensorList<CPUBackend> &tl, Index i) -> std::unique_ptr<Tensor<CPUBackend>> {
          return TensorListGetItemImpl(tl, i);
        },
      "i"_a,
      R"code(
      Returns a tensor at given position `i` in the list.

      )code",
      py::keep_alive<0, 1>())
    .def("as_array", [](TensorList<CPUBackend> &tl) -> py::array {
          void* raw_mutable_data = nullptr;
          std::string format;
          size_t type_size;

          if (tl.shape().num_elements() > 0) {
            DALI_ENFORCE(IsValidType(tl.type()), "Cannot produce "
                "buffer info for tensor w/ invalid type.");
            DALI_ENFORCE(tl.IsDenseTensor(),
                        "Tensors in the list must have the same shape");
            raw_mutable_data = contiguous_raw_mutable_data(tl);
          }

          if (IsValidType(tl.type())) {
            format = FormatStrFromType(tl.type());
            type_size = tl.type_info().size();
          } else {
            // Default is float
            format = py::format_descriptor<float>::format();
            type_size = sizeof(float);
          }

          auto shape_size = tl.shape().size() > 0 ? tl.tensor_shape(0).size() : 0;
          std::vector<ssize_t> shape(shape_size + 1);
          std::vector<ssize_t> strides(shape_size + 1);
          size_t dim_prod = 1;
          for (size_t i = 0; i < shape.size(); ++i) {
            if (i == 0) {
              shape[i] = tl.shape().size();
            } else {
              shape[i] = tl.tensor_shape(0)[i - 1];
            }

            // We iterate over stride backwards
            strides[(strides.size()-1) - i] = type_size*dim_prod;
            if (i == shape.size() - 1) {
              dim_prod *= tl.shape().size();
            } else {
              dim_prod *= tl.tensor_shape(0)[(shape.size()-2) - i];
            }
          }

          return py::array(py::dtype(format), shape, strides, raw_mutable_data);
        },
      R"code(
      Returns TensorList as a numpy array. TensorList must be dense.

      )code")
    .def("__len__", [](TensorList<CPUBackend> &tl) {
          return tl.num_samples();
        })
    .def("is_dense_tensor", &TensorList<CPUBackend>::IsDenseTensor,
      R"code(
      Checks whether all tensors in this `TensorList` have the same shape
      (and so the list itself can be viewed as a tensor).

      For example, if `TensorList` contains `N` tensors, each with shape
      `(H,W,C)` (with the same values of `H`, `W` and `C`), then the list
      may be viewed as a tensor of shape `(N, H, W, C)`.
      )code")
    .def("copy_to_external",
        [](TensorList<CPUBackend> &tl, py::object p) {
          CopyToExternal<mm::memory_kind::host>(
              ctypes_void_ptr(p), std::nullopt, tl, AccessOrder::host(), false);
        },
      R"code(
      Copy the contents of this `TensorList` to an external pointer
      (of type `ctypes.c_void_p`) residing in CPU memory.

      This function is used internally by plugins to interface with
      tensors from supported Deep Learning frameworks.

      )code")
    .def("as_reshaped_tensor",
        [](TensorList<CPUBackend> &tl, const vector<Index> &new_shape) {
          return tl.AsReshapedTensor(new_shape);
        },
      R"code(
      Returns a tensor that is a view of this `TensorList` cast to the given shape.

      This function can only be called if `TensorList` is contiguous in memory and
      the volumes of requested `Tensor` and `TensorList` matches.
      )code")
    .def("as_tensor", &TensorList<CPUBackend>::AsTensor,
      R"code(
      Returns a tensor that is a view of this `TensorList`.

      This function can only be called if `is_dense_tensor` returns `True`.
      )code")
    .def("data_ptr",
        [](TensorList<CPUBackend> &tl) {
          return py::reinterpret_borrow<py::object>(
              PyLong_FromVoidPtr(contiguous_raw_mutable_data(tl)));
        },
      R"code(
      Returns the address of the first element of TensorList.
      )code")
    .def("reinterpret", ReinterpretTensorList<CPUBackend>,
      "new_type"_a,
      R"code(
      Reinterpret the contents of the tensor list as a new type. The element size must not change.
      )code")
    .def("reset", &TensorList<CPUBackend>::Reset)
    .def("__str__", [](TensorList<CPUBackend> &t) {
      return FromPythonTrampoline("nvidia.dali.tensors", "_tensorlist_to_string")(t);
    })
    .def("__repr__", [](TensorList<CPUBackend> &t) {
      // Repr might be used in exceptions and the data might not be possible to be represented
      // (DALI enums do not support buffer protocol due to difference between C++ numeric
      // representation and Python "O" - object/pointer-based representation).
      // That why we skip the data part.
      return FromPythonTrampoline("nvidia.dali.tensors", "_tensorlist_to_string")(t, false);
    })
    .def_property_readonly("dtype", [](TensorList<CPUBackend> &tl) {
          return tl.type();
        },
      R"code(
      Data type of the TensorListCPU's elements.

      :type: DALIDataType
      )code");
}

void ExposeTesorListGPU(py::module &m) {
  auto tensor_list_gpu_class =
      py::class_<TensorList<GPUBackend>, std::shared_ptr<TensorList<GPUBackend>>>(
          m, "TensorListGPU", py::buffer_protocol())
    .def_property_readonly_static("__module__", tensor_module_impl)
    .def(py::init([](py::capsule &capsule, std::optional<std::string> layout = {}) {
            DomainTimeRange range("TensorListGPU::init from a DLPack capsule", kGPUTensorColor);
            auto t = std::make_shared<TensorList<GPUBackend>>();
            FillTensorFromDlPack(capsule, t.get(), layout);
            return t;
          }),
        "object"_a,
        "layout"_a = py::none(),
        R"code(
        List of tensors residing in the GPU memory.

        object : DLPack object
              Python DLPack object representing TensorList
        layout : str
              Layout of the data
        )code")
    .def(py::init([](TensorList<GPUBackend> *tl, std::optional<std::string> layout) {
          DomainTimeRange range("TensorListGPU::init from a list of tensors", kGPUTensorColor);
          if (!tl)
            throw py::value_error("The source object must not be null");
          auto t = std::make_shared<TensorList<GPUBackend>>();
          t->ShareData(*tl);
          // If layout is not given, use the one from `t`
          SetLayout(*t, layout, false);
          return t;
        }),
      "tl"_a,
      "layout"_a = py::none())
    .def(py::init([](
          py::list &list_of_tensors,
          std::optional<std::string> layout = {},
          bool contiguous = true) {
        DomainTimeRange range("TensorListGPU::init from a Python list of tensors", kGPUTensorColor);
        return TensorListFromListOfTensors<GPUBackend>(list_of_tensors, layout, contiguous);
      }),
      "list_of_tensors"_a,
      "layout"_a = py::none(),
      "contiguous"_a = true,
      R"code(
      List of tensors residing in the GPU memory.

      list_of_tensors : [TensorGPU]
            Python list of TensorGPU objects
      layout : str
            Layout of the data
      contiguous : bool = True
            If True, the list of tensors is converted to a contiguous TensorListGPU, necessarily
            creating a copy. Otherwise, the copy may be avoided.
      )code")
    .def(py::init([](const py::object &object,
                     const std::optional<std::string> &layout = {},
                     int device_id = -1) {
          DomainTimeRange range("TensorListGPU::init from a CUDA array", kGPUTensorColor);
          auto t = std::make_shared<TensorList<GPUBackend>>();
          FillTensorFromCudaArray(object, t.get(), device_id, layout);
          return t;
        }),
      "object"_a,
      "layout"_a = py::none(),
      "device_id"_a = -1,
      R"code(
      List of tensors residing in the GPU memory.

      object : object
            Python object that implement CUDA Array Interface
      layout : str
            Layout of the data
      device_id : int
            Device of where this tensor resides. If not provided, the current device is used.
      )code")
    .def(py::init([]() {
          // Construct a default TensorList on GPU
          return new TensorList<GPUBackend>;
        }),
      R"code(
      List of tensors residing in the GPU memory.
      )code")
    .def_static("broadcast", [](const Tensor<GPUBackend> &t, int num_samples) {
        return std::make_shared<TensorList<GPUBackend>>(t, num_samples);
      })
    .def("as_cpu", [](TensorList<GPUBackend> &t) {
          DeviceGuard g(t.device_id());
          auto ret = std::make_shared<TensorList<CPUBackend>>();
          ret->set_pinned(false);
          ret->set_order(AccessOrder::host());
          ret->SetContiguity(BatchContiguity::Contiguous);
          UserStream * us = UserStream::Get();
          cudaStream_t s = us->GetStream(t);
          ret->Copy(t, s);
          us->Wait(t);
          return ret;
        },
      R"code(
      Returns a `TensorListCPU` object being a copy of this `TensorListGPU`.
      )code",
      py::return_value_policy::take_ownership)
    .def("_set_stream", [](TensorList<GPUBackend> &t, py::object stream) {
      t.set_order(AccessOrderFromPythonStreamObj(stream));
    })
    .def("_make_copy", [](const TensorList<GPUBackend> &tl) {
        DeviceGuard dg(tl.device_id());
        auto dst = std::make_shared<TensorList<GPUBackend>>();
        dst->set_device_id(tl.device_id());
        dst->set_order(tl.order());
        dst->set_pinned(tl.is_pinned());
        dst->Copy(tl);
        return dst;
      })
    .def(
      "device_id", &TensorList<GPUBackend>::device_id)
    .def("shape", &py_shape_list<GPUBackend>,
      R"code(
      Shape of the tensor list.
      )code")
    .def("ndim", [](TensorList<GPUBackend> &tl) { return tl.shape().sample_dim(); },
      R"code(
      Number of dimensions of the tensors in the list.
      )code")
    .def("__len__", [](TensorList<GPUBackend> &t) {
          return t.num_samples();
        })
    .def("is_dense_tensor", &TensorList<GPUBackend>::IsDenseTensor,
      R"code(
      Checks whether all tensors in this `TensorList` have the same shape
      (and so the list itself can be viewed as a tensor).

      For example, if `TensorList` contains `N` tensors, each with shape
      `(H,W,C)` (with the same values of `H`, `W` and `C`), then the list
      may be viewed as a tensor of shape `(N, H, W, C)`.
      )code")
    .def("copy_to_external",
        [](TensorList<GPUBackend> &t, py::object p, py::object cuda_stream,
           bool non_blocking, bool use_copy_kernel) {
          CopyToExternalImplGPU(t, p, cuda_stream, non_blocking, use_copy_kernel);
        },
      "ptr"_a,
      "cuda_stream"_a = py::none(),
      "non_blocking"_a = false,
      "use_copy_kernel"_a = false,
      R"code(
      Copy the contents of this `TensorList` to an external pointer
      residing in GPU memory.

      This function is used internally by plugins to interface with
      tensors from supported Deep Learning frameworks.

      ptr : ctypes.c_void_p
            Destination of the copy.
      cuda_stream : ctypes.c_void_p
            CUDA stream to schedule the copy on (default stream if not provided).
      non_blocking : bool
            Asynchronous copy.
      )code")
    .def("reset", &TensorList<GPUBackend>::Reset)
    .def("__getitem__",
        [](TensorList<GPUBackend> &t, Index i) -> std::unique_ptr<Tensor<GPUBackend>> {
          return TensorListGetItemImpl(t, i);
        },
      "i"_a,
      R"code(
      Returns a tensor at given position `i` in the list.
      )code",
      py::keep_alive<0, 1>())
    .def("at",
        [&](TensorList<GPUBackend> &t, Index id) -> std::unique_ptr<Tensor<GPUBackend>> {
          std::cout << "Warning: `TensorListGPU.at` is deprecated for `TensorListGPU.__getitem__`. "
                       "It will be removed in future version of DALI."
                       "Please make sure to update your projects with the item access operator []"
                    << std::endl;
          return TensorListGetItemImpl(t, id);
        },
      R"code(
      Returns a tensor at given position in the list.
      Deprecated for __getitem__().
      )code",
      py::keep_alive<0, 1>())
    .def("layout", [](TensorList<GPUBackend> &t) {
      return t.GetLayout().str();
    })
    .def("set_layout", [](TensorList<GPUBackend> &t, const std::optional<std::string> &layout) {
      SetLayout(t, layout);
    })
    .def("as_reshaped_tensor",
        [](TensorList<GPUBackend> &tl, const vector<Index> &new_shape) {
          return tl.AsReshapedTensor(new_shape);
        },
      R"code(
      Returns a tensor that is a view of this `TensorList` cast to the given shape.

      This function can only be called if `TensorList` is contiguous in memory and
      the volumes of requested `Tensor` and `TensorList` matches.
      )code")
    .def("as_tensor", &TensorList<GPUBackend>::AsTensor,
      R"code(
      Returns a tensor that is a view of this `TensorList`.

      This function can only be called if `is_dense_tensor` returns `True`.
      )code")
    .def("data_ptr",
        [](TensorList<GPUBackend> &tl) {
          return py::reinterpret_borrow<py::object>(
              PyLong_FromVoidPtr(contiguous_raw_mutable_data(tl)));
        },
      R"code(
      Returns the address of the first element of TensorList.
      )code")
    .def("reinterpret", ReinterpretTensorList<GPUBackend>,
      "new_type"_a,
      R"code(
      Reinterpret the contents of the tensor list as a new type. The element size must not change.
      )code")
    .def("__str__", [](TensorList<GPUBackend> &t) {
      return FromPythonTrampoline("nvidia.dali.tensors", "_tensorlist_to_string")(t);
    })
    .def("__repr__", [](TensorList<GPUBackend> &t) {
      return FromPythonTrampoline("nvidia.dali.tensors", "_tensorlist_to_string")(t, false);
    })
    .def_property_readonly("stream", [](const TensorList<GPUBackend> &t)->py::object {
      if (t.order().is_device())
        return py::reinterpret_borrow<py::object>(PyLong_FromVoidPtr(t.order().stream()));
      else
        return py::none();
    })
    .def_property_readonly("dtype", [](TensorList<GPUBackend> &tl) {
          return tl.type();
        },
      R"code(
      Data type of the TensorListGPU's elements.

      :type: DALIDataType
      )code");
}

void ExposeTensorList(py::module &m) {
  ExposeTensorListCPU(m);
  ExposeTesorListGPU(m);
}

#define GetRegisteredOpsFor(OPTYPE)                                           \
static vector<string> GetRegistered##OPTYPE##Ops(bool internal_ops = false) { \
  return OPTYPE##OperatorRegistry::Registry().RegisteredNames(internal_ops);  \
}
GetRegisteredOpsFor(CPU)
GetRegisteredOpsFor(GPU)
GetRegisteredOpsFor(Mixed)
#undef GetRegisteredOpsFor

static const OpSchema &GetSchema(const string &name) {
  return SchemaRegistry::GetSchema(name);
}

static const OpSchema *TryGetSchema(const string &name) {
  return SchemaRegistry::TryGetSchema(name);
}

static constexpr int GetCxx11AbiFlag() {
#ifdef _GLIBCXX_USE_CXX11_ABI
  return _GLIBCXX_USE_CXX11_ABI;
#else
  return 0;
#endif
}

#ifdef DALI_BUILD_PROTO3
using TFFeatureType = TFUtil::FeatureType;
using TFFeature = TFUtil::Feature;
using TFValue = TFFeature::Value;

TFValue ConvertTFRecordDefaultValue(TFFeatureType type, py::object val) {
  PyObject *ptr = val.ptr();
  TFValue ret = {};
  switch (type) {
    case TFFeatureType::int64:
      DALI_ENFORCE(PyInt_Check(ptr) || PyLong_Check(ptr),
          "Invalid type for default value, expected int.");
      ret.int64 = PyInt_AsLong(ptr);
      break;
    case TFFeatureType::string:
      DALI_ENFORCE(PyStr_Check(ptr),
          "Invalid type for default value, expected string.");
      ret.str = PyStr_AsString(ptr);
      break;
    case TFFeatureType::float32:
      DALI_ENFORCE(PyFloat_Check(ptr),
          "Invalid type for default value, expected float.");
      ret.float32 = PyFloat_AsDouble(ptr);
      break;
    default:
      DALI_FAIL("Invalid type for default value, expected string, int or float.");
  }
  return ret;
}
#endif  // DALI_BUILD_PROTO3

void ExposeBufferPolicyFunctions(py::module &m) {
  m.def("SetHostBufferShrinkThreshold", [](double ratio) {
    if (ratio < 0 || ratio > 1)
      throw py::value_error("Shrink threshold must be between 0 (never shrink) "
                            "and 1 (always shrink).");

    Buffer<CPUBackend>::SetShrinkThreshold(ratio);
  });

  m.def("SetHostBufferGrowthFactor", [](double factor) {
    const double max_factor = std::min(
        Buffer<CPUBackend>::kMaxGrowthFactor,
        Buffer<GPUBackend>::kMaxGrowthFactor);

    if (factor < 1 || factor > max_factor)
      throw py::value_error(make_string("Growth factor must be between 1 and ", max_factor, "."));

    Buffer<CPUBackend>::SetGrowthFactor(factor);
  });

  m.def("SetDeviceBufferGrowthFactor", [](double factor) {
    const double max_factor = Buffer<CPUBackend>::kMaxGrowthFactor;
    if (factor < 1 || factor > max_factor)
      throw py::value_error(make_string("Growth factor must be between 1 and ", max_factor, "."));

    Buffer<GPUBackend>::SetGrowthFactor(factor);
  });

  m.def("SetBufferGrowthFactor", [](double factor) {
    const double max_factor = Buffer<GPUBackend>::kMaxGrowthFactor;
    if (factor < 1 || factor > max_factor)
      throw py::value_error(make_string("Growth factor must be between 1 and ", max_factor, "."));

    Buffer<CPUBackend>::SetGrowthFactor(factor);
    Buffer<GPUBackend>::SetGrowthFactor(factor);
  });

  m.def("GetHostBufferShrinkThreshold", Buffer<CPUBackend>::GetShrinkThreshold);
  m.def("GetHostBufferGrowthFactor", Buffer<CPUBackend>::GetGrowthFactor);
  m.def("GetDeviceBufferGrowthFactor", Buffer<GPUBackend>::GetGrowthFactor);
  m.def("RestrictPinnedMemUsage", RestrictPinnedMemUsage);
  m.def("GetCUDADeviceCount", []() {
    int count = 0;
    CUDA_CALL(cudaGetDeviceCount(&count));
    return count;
  });
  m.def("SetCUDACurrentDevice", [](int device_id) {
    CUDA_CALL(cudaSetDevice(device_id));
  });
  m.def("GetCUDACurrentDevice", []() {
    int device_id = 0;
    CUDA_CALL(cudaGetDevice(&device_id));
    return device_id;
  });
  m.def("PreallocateDeviceMemory", mm::PreallocateDeviceMemory,
R"(Preallocate memory on given device

The function ensures that after the call, the amount of memory given in `bytes` can be
allocated from the pool (without further requests to the OS).

Calling this function while DALI pipelines are running is generally safe, but it should not be used
to preallocate memory for a pipeline that's already running - this may result in a race
for memory and possibly trigger out-of-memory error in the pipeline.
)", "bytes"_a, "device_id"_a);
  m.def("PreallocatePinnedMemory", mm::PreallocatePinnedMemory,
R"(Preallocate non-pageable (pinned) host memory

The function ensures that after the call, the amount of memory given in `bytes` can be
allocated from the pool (without further requests to the OS).

Calling this function while DALI pipelines are running is generally safe, but it should not be used
to preallocate memory for a pipeline that's already running - this may result in a race
for memory and possibly trigger out-of-memory error in the pipeline.
)", "bytes"_a);

  m.def("ReleaseUnusedMemory", mm::ReleaseUnusedMemory,
R"(Frees unused blocks from memory pools.

Only blocks that are completely free are released. The function frees the memory from all device
pools as well as from the host pinned memory pool.

This function is safe to use while DALI pipelines are running.)");
}

py::dict ArgumentDeprecationInfoToDict(const ArgumentDeprecation & meta) {
  py::dict d;
  d["msg"] = meta.msg;
  d["removed"] = meta.removed;
  d["renamed_to"] = meta.renamed_to;
  d["deprecated_in_version"] = meta.version;
  return d;
}

py::dict ReaderMetaToDict(const ReaderMeta &meta) {
  py::dict d;
  d["epoch_size"] = meta.epoch_size;
  d["epoch_size_padded"] = meta.epoch_size_padded;
  d["number_of_shards"] = meta.number_of_shards;
  d["shard_id"] = meta.shard_id;
  d["pad_last_batch"] = meta.pad_last_batch;
  d["stick_to_shard"] = meta.stick_to_shard;
  return d;
}

py::dict ExecutorMetaToDict(const ExecutorMetaMap &meta) {
  py::dict d;
  for (const auto &stat : meta) {
    py::dict op_dict;
    py::list real_memory_size;
    py::list reserved_memory_size;
    py::list max_real_memory_size;
    py::list max_reserved_memory_size;
    for (const auto &entry : stat.second) {
      real_memory_size.append(entry.real_size);
      max_real_memory_size.append(entry.max_real_size);
      reserved_memory_size.append(entry.reserved);
      max_reserved_memory_size.append(entry.max_reserved);
    }
    op_dict["real_memory_size"] = real_memory_size;
    op_dict["max_real_memory_size"] = max_real_memory_size;
    op_dict["reserved_memory_size"] = reserved_memory_size;
    op_dict["max_reserved_memory_size"] = max_reserved_memory_size;
    d[stat.first.c_str()] = op_dict;
  }
  return d;
}

void ExposePipelineDebug(py::module &m) {
  py::class_<PipelineDebug>(m, "PipelineDebug")
      .def(py::init([](int batch_size, int num_threads, int device_id, bool set_affinity = false) {
        return std::make_unique<PipelineDebug>(batch_size, num_threads, device_id, set_affinity);
      }),
      "batch_size"_a,
      "num_threads"_a,
      "device_id"_a,
      "set_affinity"_a = false)
      .def("AddOperator", &PipelineDebug::AddOperator)
      .def("AddMultipleOperators", &PipelineDebug::AddMultipleOperators)
      .def("RunOperatorCPU", &PipelineDebug::RunOperator<CPUBackend>)
      .def("RunOperatorGPU", &PipelineDebug::RunOperator<GPUBackend>)
      .def("RunOperatorMixed", &PipelineDebug::RunOperator<MixedBackend>)
      .def("Shutdown", [](PipelineDebug *){});
}

template <typename Backend>
void FeedPipeline(Pipeline *p, const string &name, py::list list, AccessOrder order,
                  bool sync = false, bool use_copy_kernel = false) {
  TensorList<Backend> tv(list.size());
  for (size_t i = 0; i < list.size(); ++i) {
    auto &t = list[i].cast<Tensor<Backend>&>();
    // TODO(klecki): evaluate if we want to keep such code - we need to be able to set
    // order, pinned, type, dimensionality and layout every time if we don't want
    // SetSample to do that.
    if (i == 0) {
      tv.SetupLike(t);
    }
    tv.SetSample(i, t);
    // TODO(klecki): tv[i] = std::move(t);
  }
  p->SetExternalInput(name, tv, order, sync, use_copy_kernel);
}

struct PyPipeline: public Pipeline {
  using Pipeline::Pipeline;

  ~PyPipeline() override {
    py::gil_scoped_release interpreter_unlock{};
    Shutdown();
  }
};

void ExposePipelineParams(py::module &m) {
  py::enum_<ExecutorType>(m, "_ExecutorType")
    .value("Simple", ExecutorType::Simple)
    .value("PipelinedFlag", ExecutorType::PipelinedFlag)
    .value("SeparatedFlag", ExecutorType::SeparatedFlag)
    .value("AsyncFlag", ExecutorType::AsyncFlag)
    .value("DynamicFlag", ExecutorType::DynamicFlag)
    .value("AsyncPipelined", ExecutorType::AsyncPipelined)
    .value("SeparatedPipelined", ExecutorType::SeparatedPipelined)
    .value("AsyncSeparatedPipelined", ExecutorType::AsyncSeparatedPipelined)
    .value("Dynamic", ExecutorType::Dynamic);

  py::enum_<ExecutorFlags>(m, "_ExecutorFlags")
    .value("NoFlags", ExecutorFlags::None)
    .value("SetAffinity", ExecutorFlags::SetAffinity)
    .value("StreamPolicyMask", ExecutorFlags::StreamPolicyMask)
    .value("StreamPolicyPerOperator", ExecutorFlags::StreamPolicyPerOperator)
    .value("StreamPolicyPerBackend", ExecutorFlags::StreamPolicyPerBackend)
    .value("StreamPolicySingle", ExecutorFlags::StreamPolicySingle)
    .value("ConcurrencyMask", ExecutorFlags::ConcurrencyMask)
    .value("ConcurrencyNone", ExecutorFlags::ConcurrencyNone)
    .value("ConcurrencyFull", ExecutorFlags::ConcurrencyFull)
    .value("ConcurrencyBackend", ExecutorFlags::ConcurrencyBackend);

  m.def("_MakeExecutorType", MakeExecutorType);

  py::class_<PipelineParams>(m, "_PipelineParams")
    .def(py::init([](
        std::optional<int> max_batch_size,
        std::optional<int> num_threads,
        std::optional<int> device_id,
        std::optional<int64_t> seed,
        std::optional<ExecutorType> executor_type,
        std::optional<ExecutorFlags> executor_flags,
        std::optional<std::pair<int, int>> prefetch_queue_depths,
        std::optional<bool> enable_checkpointing,
        std::optional<bool> enable_memory_stats,
        std::optional<size_t> bytes_per_sample_hint) {
      std::optional<QueueSizes> queue_sizes;
      if (prefetch_queue_depths)
        queue_sizes = QueueSizes{prefetch_queue_depths->first, prefetch_queue_depths->second};

      return std::unique_ptr<PipelineParams>(new PipelineParams{
        max_batch_size,
        num_threads,
        device_id,
        seed,
        executor_type,
        executor_flags,
        queue_sizes,
        enable_checkpointing,
        enable_memory_stats,
        bytes_per_sample_hint
      });
    }),
    "max_batch_size"_a = py::none(),
    "num_threads"_a = py::none(),
    "device_id"_a = py::none(),
    "seed"_a = py::none(),
    "executor_type"_a = py::none(),
    "executor_flags"_a = py::none(),
    "prefetch_queue_depths"_a = py::none(),
    "enable_checkpointing"_a = py::none(),
    "enable_memory_stats"_a = py::none(),
    "bytes_per_sample_hint"_a = py::none()
    )
  .def_readwrite("max_batch_size", &PipelineParams::max_batch_size)
  .def_readwrite("num_threads", &PipelineParams::num_threads)
  .def_readwrite("device_id", &PipelineParams::device_id)
  .def_readwrite("seed", &PipelineParams::seed)
  .def_readwrite("executor_type", &PipelineParams::executor_type)
  .def_readwrite("executor_flags", &PipelineParams::executor_flags)
  .def_property("prefetch_queue_depths",
    [](const PipelineParams &self)->std::optional<std::pair<int, int>> {
      if (self.prefetch_queue_depths.has_value()) {
        return std::make_pair(self.prefetch_queue_depths->cpu_size,
                              self.prefetch_queue_depths->gpu_size);
      } else {
        return std::nullopt;
      }
    },
    [](PipelineParams &self, std::optional<std::pair<int, int>> &queue_depths) {
      if (queue_depths.has_value())
        self.prefetch_queue_depths = QueueSizes{queue_depths->first, queue_depths->second};
    })
  .def_readwrite("enable_checkpointing", &PipelineParams::enable_checkpointing)
  .def_readwrite("enable_memory_stats", &PipelineParams::enable_memory_stats)
  .def_readwrite("bytes_per_sample_hint", &PipelineParams::bytes_per_sample_hint);
}

void ExposePipeline(py::module &m) {
  py::class_<Pipeline, PyPipeline>(m, "Pipeline")
    .def(py::init(
            [](const PipelineParams &params) {
              return std::make_unique<PyPipeline>(params);
            }),
        "params"_a
        )
    // initialize from serialized pipeline
    .def(py::init(
          [](const string &serialized_pipe,
             const PipelineParams &params) {
              return std::make_unique<PyPipeline>(serialized_pipe, params);
            }),
        "serialized_pipe"_a,
        "params"_a
        )
    .def("AddOperator",
         static_cast<int (Pipeline::*)(const OpSpec &, std::string_view)>
                                      (&Pipeline::AddOperator))
    .def("AddOperator",
         static_cast<int (Pipeline::*)(const OpSpec &, std::string_view, int)>
                                      (&Pipeline::AddOperator))
    .def("GetOperatorNode", &Pipeline::GetOperatorNode)
    .def("Build",
         [](Pipeline *p, const std::vector<OutputDesc> &outputs) {
             std::vector<PipelineOutputDesc> build_args;
             for (auto& out : outputs) {
               build_args.emplace_back(to_struct<PipelineOutputDesc>(out));
             }
             p->Build(build_args);
         })
    .def("Build", [](Pipeline *p) { p->Build(); })
    .def("Shutdown", &Pipeline::Shutdown, py::call_guard<py::gil_scoped_release>())
    .def("EnableCheckpointing",
        [](Pipeline *p, bool checkpointing) {
          p->EnableCheckpointing(checkpointing);
        },
        "checkpointing"_a = true)
    .def("GetSerializedCheckpoint",
        [](Pipeline *p, const ExternalContextCheckpoint &external_ctx_cpt) -> py::bytes {
          return p->GetSerializedCheckpoint(external_ctx_cpt);
          })
    .def("RestoreFromSerializedCheckpoint",
        [](Pipeline *p, const std::string &serialized_checkpoint) {
          return p->RestoreFromSerializedCheckpoint(serialized_checkpoint);
        })
    .def("executor_statistics",
        [](Pipeline *p) {
          auto ret = p->GetExecutorMeta();
          return ExecutorMetaToDict(ret);
        })
    .def("SetOutputDescs",
        [](Pipeline *p, const std::vector<OutputDesc>& outputs) {
          std::vector<PipelineOutputDesc> out_desc;
          for (auto& out : outputs) {
            out_desc.emplace_back(to_struct<PipelineOutputDesc>(out));
          }
          p->SetOutputDescs(out_desc);
        })
    .def("Run", &Pipeline::Run, py::call_guard<py::gil_scoped_release>())
    .def("Prefetch", &Pipeline::Prefetch, py::call_guard<py::gil_scoped_release>())
    .def("Outputs",
        [](Pipeline *p, py::object cuda_stream) {
          Workspace ws;

          if (!cuda_stream.is_none())
            ws.set_output_order(static_cast<cudaStream_t>(ctypes_void_ptr(cuda_stream)));

          {
            py::gil_scoped_release interpreter_unlock{};
            p->Outputs(&ws);
          }
          py::tuple outs(ws.NumOutput());
          for (int i = 0; i < ws.NumOutput(); ++i) {
            if (ws.OutputIsType<CPUBackend>(i)) {
              outs[i] = ws.OutputPtr<CPUBackend>(i);
            } else {
              outs[i] = ws.OutputPtr<GPUBackend>(i);
            }
          }
          return outs;
        },
        "cuda_stream"_a = py::none(),
        py::return_value_policy::take_ownership)
    .def("ShareOutputs",
        [](Pipeline *p, py::object cuda_stream) {
          Workspace ws;

          if (!cuda_stream.is_none())
            ws.set_output_order(static_cast<cudaStream_t>(ctypes_void_ptr(cuda_stream)));

          {
            py::gil_scoped_release interpreter_unlock{};
            p->ShareOutputs(&ws);
          }

          py::tuple outs(ws.NumOutput());
          for (int i = 0; i < ws.NumOutput(); ++i) {
            if (ws.OutputIsType<CPUBackend>(i)) {
              outs[i] = ws.OutputPtr<CPUBackend>(i);
            } else {
              outs[i] = ws.OutputPtr<GPUBackend>(i);
            }
          }
          return outs;
        },
        "cuda_stream"_a = py::none(),
        py::return_value_policy::take_ownership)
    .def("ReleaseOutputs", &Pipeline::ReleaseOutputs, py::call_guard<py::gil_scoped_release>())
    .def("batch_size", &Pipeline::batch_size)
    .def("num_threads", &Pipeline::num_threads)
    .def("device_id", &Pipeline::device_id)
    .def("params", &Pipeline::GetParams)
    .def("requires_gpu", &Pipeline::requires_gpu)
    .def("output_dtype",
         [](Pipeline *p) {
             auto &descs = p->output_descs();
             std::vector<DALIDataType> ret(descs.size());
             for (size_t i = 0; i < descs.size(); i++) {
               ret[i] = descs[i].dtype;
             }
             return ret;
         })
    .def("output_ndim",
         [](Pipeline *p) {
             auto &descs = p->output_descs();
             std::vector<int> ret(descs.size());
             for (size_t i = 0; i < descs.size(); i++) {
               ret[i] = descs[i].ndim;
             }
             return ret;
         })
    .def("InputFeedCount", &Pipeline::InputFeedCount, "input_name"_a)
    .def("SetExternalTLInput",
        [](Pipeline *p, const string &name, const TensorList<CPUBackend> &tl,
           py::object /*cuda_stream*/, bool /*use_copy_kernel*/) {
          p->SetExternalInput(name, tl, {}, true);
        },
        "name"_a,
        "list"_a,
        "cuda_stream"_a = py::none(),
        "use_copy_kernel"_a = false)
    .def("SetExternalTLInput",
        [](Pipeline *p, const string &name, const TensorList<GPUBackend> &tl,
           py::object cuda_stream, bool use_copy_kernel) {
           cudaStream_t stream = cuda_stream.is_none()
                                 ? UserStream::Get()->GetStream(tl)
                                 : static_cast<cudaStream_t>(ctypes_void_ptr(cuda_stream));
          p->SetExternalInput(name, tl, stream, cuda_stream.is_none(), use_copy_kernel);
        },
        "name"_a,
        "list"_a,
        "cuda_stream"_a = py::none(),
        "use_copy_kernel"_a = false)
    .def("SetExternalTensorInput",
        [](Pipeline *p, const string &name, py::list list, py::object cuda_stream,
           bool use_copy_kernel) {
          // Note: This is a hack to get around weird casting
          // issues w/ pybind and a non-copyable type (dali::Tensor).
          // We cannot use pybind::cast<Tensor<CPUBackend>>
          // because somewhere through the chain of templates
          // pybind returns the calling template type, which
          // tries to call the deleted copy constructor for Tensor.
          // instead, we cast to a reference type and manually
          // move into the vector.
          DALI_ENFORCE(static_cast<int>(list.size()) <= p->max_batch_size(),
             "Data list provided to feed_input exceeds maximum batch_size for this pipeline.");

          // not the most beautiful but at least it doesn't throw as plain cast<T>()
          py::detail::make_caster<Tensor<CPUBackend>&> conv;
          bool is_cpu_data = list.empty() || conv.load(static_cast<py::object>(list[0]), true);
          if (is_cpu_data) {
            FeedPipeline<CPUBackend>(p, name, list, AccessOrder::host(), true);
          } else {
            int device_id = p->device_id();
            cudaStream_t stream = 0;
            if (!cuda_stream.is_none())
              stream = static_cast<cudaStream_t>(ctypes_void_ptr(cuda_stream));

            if (!list.empty()) {
              auto &sample0 = list[0].cast<Tensor<GPUBackend>&>();
              if (cuda_stream.is_none())
                stream = UserStream::Get()->GetStream(sample0);
              device_id = sample0.device_id();
            }
            AccessOrder order(stream, device_id);
            if (order.is_device()) {
              CUcontext ctx = nullptr;
              CUDA_CALL(cuStreamGetCtx(order.stream(), &ctx));
              CUDA_CALL(cuCtxPushCurrent(ctx));
              CUdevice device;
              CUDA_CALL(cuCtxGetDevice(&device));
              CUDA_CALL(cuCtxPopCurrent(&ctx));
            }
            FeedPipeline<GPUBackend>(p, name, list, order, cuda_stream.is_none(), use_copy_kernel);
          }
        },
        "name"_a,
        "list"_a,
        "cuda_stream"_a = py::none(),
        "use_copy_kernel"_a = false)
    .def("SetPyObjDependency",
      [](Pipeline *p, py::object obj) {}, "obj"_a, py::keep_alive<1, 2>())
    .def("SerializeToProtobuf",
        [](Pipeline *p) -> py::bytes {
          string s = p->SerializeToProtobuf();
          return s;
          }, py::return_value_policy::take_ownership)
    .def("SaveGraphToDotFile", &Pipeline::SaveGraphToDotFile,
        "path"_a,
        "show_tensors"_a = false,
        "use_colors"_a = false)
    .def("reader_meta", [](Pipeline* p) {
          py::dict d;
          for (auto const&[name, meta] : p->GetReaderMeta()) {
            std::string name_str(name);  // pybind11 doesn't have operator[](string_view)
            d[name_str.c_str()] = ReaderMetaToDict(meta);
          }
          return d;
        })
    .def("reader_meta",
        [](Pipeline* p, const std::string& op_name) {
          ReaderMeta meta = p->GetReaderMeta(op_name);
          DALI_ENFORCE(meta,
              "Operator " + op_name + "  not found or does not expose valid metadata.");
          return ReaderMetaToDict(meta);
        });
}

template <typename Backend>
std::shared_ptr<TensorList<Backend>> CloneTL(const TensorList<Backend> &tl) {
  auto tl_clone = std::make_shared<TensorList<Backend>>();
  tl_clone->ShareData(tl);
  return tl_clone;
}

void ExposeStream(py::module &m) {
  py::class_<dali::CUDAStreamLease>(m, "Stream")
    .def(py::init([](std::optional<int> device_id) {
      return std::make_unique<dali::CUDAStreamLease>(
        dali::CUDAStreamPool::instance().Get(device_id.value_or(-1)));
    }))
    // __cuda_stream__ protocol, as per cuda.core 0.3.0
    .def("__cuda_stream__", [](dali::CUDAStreamLease &self) {
      // version, stream - version is 0
      return std::make_tuple(0, reinterpret_cast<uintptr_t>(self.get()));
    })
    .def_property_readonly("device_id", [](dali::CUDAStreamLease &self) {
      return self.device_id();
    })
    .def_property_readonly("handle", [](dali::CUDAStreamLease &self) {
      return reinterpret_cast<uintptr_t>(self.get());
    });
}

/** Python wrapper for the Workspace */
class PyWorkspace : public Workspace {
 public:
  PyWorkspace() = default;
  PyWorkspace(const PyWorkspace &other) = default;
  PyWorkspace(PyWorkspace &&other) = default;
  PyWorkspace(const Workspace &other) :  Workspace(other) {}  // NOLINT
  PyWorkspace(Workspace &&other) :  Workspace(std::move(other)) {}  // NOLINT

  /** Set the thread pool and store the shared pointer to keep it alive */
  void SetThreadPool(std::shared_ptr<ThreadPool> thread_pool) {
    shared_thread_pool_ = thread_pool;;
    Workspace::SetThreadPool(thread_pool.get());
  }

  void SetStream(py::object cuda_stream) {
    set_output_order(AccessOrderFromPythonStreamObj(cuda_stream));
    py_stream_ = cuda_stream;  // keep the stream python object alive
  }

 private:
  std::shared_ptr<ThreadPool> shared_thread_pool_;
  py::object py_stream_;
};

void ExposeThreadPool(py::module &m) {
  py::class_<ThreadPool, std::shared_ptr<ThreadPool>>(m, "_ThreadPool")
    .def(py::init([](
          int num_threads,
          std::optional<int> device_id,
          bool set_affinity,
          std::string_view name) {
      if (!device_id.has_value()) {
        int dev = 0;
        CUDA_CALL(cudaGetDevice(&dev));
        device_id = dev;
      }
      return std::make_shared<ThreadPool>(num_threads, *device_id, set_affinity, name.data());
    }),
    "num_threads"_a,
    "device_id"_a = py::none(),
    "set_affinity"_a = false,
    "name"_a = "")
    .def_property_readonly("num_threads", &ThreadPool::NumThreads);
}

void ExposeWorkspace(py::module &m) {
  // Expose PyWorkspace as a private class within the backend_impl module
  py::class_<PyWorkspace>(m, "_Workspace")
    .def(py::init([](std::shared_ptr<ThreadPool> thread_pool, py::object stream) {
      auto ws = std::make_unique<PyWorkspace>();
      ws->SetThreadPool(std::move(thread_pool));
      if (!stream.is_none()) {
        cudaStream_t s = static_cast<cudaStream_t>(ctypes_void_ptr(stream));
        ws->set_output_order(s);
      }
      return ws;
    }), "thread_pool"_a = nullptr, "stream"_a = py::none())
    .def("SetStream", &PyWorkspace::SetStream, "cuda_stream"_a = py::none())
    .def("AddInput", [](PyWorkspace &self, const TensorList<CPUBackend> &tl) {
      self.AddInput(CloneTL(tl));
    })
    .def("AddInput", [](PyWorkspace &self, const TensorList<GPUBackend> &tl) {
      self.AddInput(CloneTL(tl));
    })
    .def("AddArgumentInput", [](
          PyWorkspace &self,
          std::string name,
          const TensorList<CPUBackend> &tl) {
      self.AddArgumentInput(std::move(name), CloneTL(tl));
    })
    .def("AddOutput", [](PyWorkspace &self, const TensorList<CPUBackend> &tl) {
      self.AddOutput(CloneTL(tl));
    })
    .def("AddOutput", [](PyWorkspace &self, const TensorList<GPUBackend> &tl) {
      self.AddOutput(CloneTL(tl));
    })
    .def("AddEmptyOutput", [](PyWorkspace &self, std::string_view device) {
      auto storage_device = ParseStorageDevice(device);
      if (storage_device == StorageDevice::CPU) {
        self.AddOutput(std::make_shared<TensorList<CPUBackend>>());
      } else {
        assert(storage_device == StorageDevice::GPU);
        self.AddOutput(std::make_shared<TensorList<GPUBackend>>());
      }
    }, "device"_a)
    .def("SetThreadPool", [](PyWorkspace &self, std::shared_ptr<ThreadPool> thread_pool) {
      self.SetThreadPool(std::move(thread_pool));
    }, "thread_pool"_a)
    .def("GetOutputs", [](PyWorkspace &self) {
      py::tuple ret(self.NumOutput());
      for (int i = 0; i < self.NumOutput(); i++) {
        if (self.OutputIsType<CPUBackend>(i)) {
          ret[i] = self.OutputPtr<CPUBackend>(i);
        } else {
          ret[i] = self.OutputPtr<GPUBackend>(i);
        }
      }
      return ret;
    }, py::return_value_policy::take_ownership);
}

void SetupAndRun(OperatorBase &self, Workspace &ws, std::optional<int> batch_size) {
  DomainTimeRange setup_and_run_tr("SetupAndRun " + GetOpDisplayName(self.GetSpec(), true));
  std::vector<dali::OutputDesc> out_descs;
  const auto &spec = self.GetSpec();
  if (ws.NumOutput() != 0)
    throw std::runtime_error("Workspace already has outputs defined");

  for (int i = 0; i < spec.NumOutput(); i++) {
    if (spec.OutputDevice(i) == StorageDevice::CPU) {
      auto out = std::make_shared<TensorList<CPUBackend>>();
      out->set_order(ws.output_order(), false);
      out->set_pinned(true);
      ws.AddOutput(std::move(out));
    } else {
      auto out = std::make_shared<TensorList<GPUBackend>>();
      out->set_order(ws.output_order(), false);
      ws.AddOutput(std::move(out));
    }
  }

  if (batch_size.has_value()) {
    ws.SetBatchSizes(batch_size.value());
  } else {
    if (ws.NumOutput() > 0 && ws.GetRequestedBatchSize(0) == 0) {
      if (ws.NumInput() > 0) {
        ws.SetBatchSizes(ws.GetInputBatchSize(0));
      } else if (ws.NumArgumentInput() > 0) {
        ws.SetBatchSizes(ws.ArgumentInput(0).num_samples());
      } else {
        int max_bs = self.GetSpec().GetArgument<int>("max_batch_size");
        ws.SetBatchSizes(max_bs);
      }
    }
  }

  {
    DomainTimeRange setup_tr("Setup " + GetOpDisplayName(self.GetSpec(), true));
    if (self.Setup(out_descs, ws)) {
      for (int i = 0; i < ws.NumOutput(); i++) {
        if (ws.OutputIsType<CPUBackend>(i))
          ws.Output<CPUBackend>(i).Resize(out_descs[i].shape, out_descs[i].type);
        else
          ws.Output<GPUBackend>(i).Resize(out_descs[i].shape, out_descs[i].type);
      }
    }
  }
  {
    DomainTimeRange run_tr("Run " + GetOpDisplayName(self.GetSpec(), true));
    self.Run(ws);
  }
}

void ExposeOperator(py::module &m) {
  py::class_<OperatorBase, std::unique_ptr<OperatorBase>>(m, "_Operator")
    .def(py::init([](const OpSpec &spec) {
      DomainTimeRange tr("Instantiate " + GetOpDisplayName(spec, true));
      return dali::InstantiateOperator(spec);
    }))
    .def("Setup", [](OperatorBase &self, std::vector<dali::OutputDesc> &out_descs, Workspace &ws) {
      py::gil_scoped_release interpreter_unlock{};
      DomainTimeRange tr("Setup " + GetOpDisplayName(self.GetSpec(), true));
      return self.Setup(out_descs, ws);
    })
    .def("Run", [](OperatorBase &self, Workspace &ws) {
      py::gil_scoped_release interpreter_unlock{};
      DomainTimeRange tr("Run " + GetOpDisplayName(self.GetSpec(), true));
      self.Run(ws);
    } )
    .def("SetupAndRun", [](OperatorBase &self, PyWorkspace &ws, std::optional<int> batch_size) {
      py::gil_scoped_release interpreter_unlock{};
      SetupAndRun(self, ws, batch_size);
    }, "ws"_a, "batch_size"_a = py::none())
    .def("GetReaderMeta", [](OperatorBase &self) {
      return ReaderMetaToDict(self.GetReaderMeta());
    });
}

auto GetSupportedBackends(OpSchema &schema) {
  std::vector<std::string_view> ret;
  ret.reserve(2);  // the vast majority of operators will have only one or two supported backends
  auto &name = schema.name();
  if (CPUOperatorRegistry::Registry().IsRegistered(name))
    ret.push_back("cpu");
  if (GPUOperatorRegistry::Registry().IsRegistered(name))
    ret.push_back("gpu");
  if (MixedOperatorRegistry::Registry().IsRegistered(name))
    ret.push_back("mixed");
  return ret;
}

PYBIND11_MODULE(backend_impl, m, py::mod_gil_not_used()) {
  dali::InitOperatorsLib();
  m.doc() = "Python bindings for the C++ portions of DALI";

  // DALI Init function
  m.def("Init", &DALIInit);

  ExposeBufferPolicyFunctions(m);

  m.def("LoadLibrary", &PluginManager::LoadLibrary,
    py::arg("lib_path"),
    py::arg("global_symbols") = false,
    py::arg("allow_fail") = false);

  m.def("LoadDirectory", &PluginManager::LoadDirectory,
    py::arg("dir_path"),
    py::arg("global_symbols") = false,
    py::arg("allow_fail") = false);

  m.def("LoadDefaultPlugins", &PluginManager::LoadDefaultPlugins);

  m.def("GetCxx11AbiFlag", &GetCxx11AbiFlag);

  m.def("IsDriverInitialized", [] {
    // we just want to check if cuda has been loaded already
    if (dlopen("libcuda.so", RTLD_NOLOAD | RTLD_NOW) ||
        dlopen("libcuda.so.1", RTLD_NOLOAD | RTLD_NOW)) {
      int place_holder = -1;
      // call cuDeviceGetCount only if cuda is loaded, if not there is not point in calling
      // a check that would load it
      return CUDA_SUCCESS == cuDeviceGetCount(&place_holder);
    }
    return false;
  });

  m.def("GetCudaVersion", [] {
    int version = -1;
    auto ret = cudaDriverGetVersion(&version);
    if (ret != cudaSuccess) {
      return -1;
    } else {
      return version;
    }
  });

  m.def("GetCufftVersion", [] {
    int ret = -1;
    try {
      // we don't want to throw when it is not available, just return -1
      ret = GetCufftVersion();
    } catch (const std::runtime_error &) {}
    return ret;
  });

  m.def("GetNppVersion", [] {
    int ret = -1;
    try {
      // we don't want to throw when it is not available, just return -1
      ret = GetNppVersion();
    } catch (const std::runtime_error &) {}
    return ret;
  });

  m.def("GetNvjpegVersion", [] {
    int ret = -1;
    try {
      // we don't want to throw when it is not available, just return -1
      ret = GetNvjpegVersion();
    } catch (const std::runtime_error &) {}
    return ret;
  });

  m.def("GetNvimgcodecVersion", [] {
    int ret = -1;
    try {
      // we don't want to throw when it is not available, just return -1
      ret = GetNvimgcodecVersion();
    } catch (const std::runtime_error &) {}
    return ret;
  });

#if SHM_WRAPPER_ENABLED

  py::class_<SharedMem>(m, "SharedMem")
      .def(py::init<int, uint64_t>())
      .def_property_readonly("size", &SharedMem::size)
      .def_property_readonly("handle", &SharedMem::handle)
      .def("buf",
           [](SharedMem *shm) {
             if (shm == nullptr) {
               throw py::value_error("Cannot create buffer - no shared memory object provided");
             }
             auto *ptr = shm->get_raw_ptr();
             if (ptr == nullptr) {
               throw py::value_error("Cannot create buffer - no memory has been mapped");
             }
             return py::memoryview::from_buffer(ptr, {shm->size()}, {sizeof(uint8_t)});
           })
      .def("resize", &SharedMem::resize)
      .def("close_handle", &SharedMem::close_handle)
      .def("close", &SharedMem::close);

#endif

  // Types
  py::module types_m = m.def_submodule("types");
  types_m.doc() = "Datatypes and options used by DALI";
  types_m.add_object("CPU_ONLY_DEVICE_ID", PyLong_FromLong(CPU_ONLY_DEVICE_ID));

  py::enum_<DLDeviceType> dl_device_type(
    m, "DLDeviceType", "DLPack device type");

  #define DL_DEVICE_TYPE(x) dl_device_type.value(#x, x)

  DL_DEVICE_TYPE(kDLCPU);
  DL_DEVICE_TYPE(kDLCUDA);
  DL_DEVICE_TYPE(kDLCUDAHost);
  DL_DEVICE_TYPE(kDLOpenCL);
  DL_DEVICE_TYPE(kDLVulkan);
  DL_DEVICE_TYPE(kDLMetal);
  DL_DEVICE_TYPE(kDLVPI);
  DL_DEVICE_TYPE(kDLROCM);
  DL_DEVICE_TYPE(kDLROCMHost);
  DL_DEVICE_TYPE(kDLExtDev);
  DL_DEVICE_TYPE(kDLCUDAManaged);
  DL_DEVICE_TYPE(kDLOneAPI);
  DL_DEVICE_TYPE(kDLWebGPU);
  DL_DEVICE_TYPE(kDLHexagon);
  DL_DEVICE_TYPE(kDLMAIA);

  // DALIDataType
  py::enum_<DALIDataType> dali_data_type(
      types_m, "DALIDataType", "Object representing the data type of a Tensor.\n<SPHINX_IGNORE>");
  dali_data_type
    .value("NO_TYPE",       DALI_NO_TYPE)
    .value("UINT8",         DALI_UINT8)
    .value("UINT16",        DALI_UINT16)
    .value("UINT32",        DALI_UINT32)
    .value("UINT64",        DALI_UINT64)
    .value("INT8",          DALI_INT8)
    .value("INT16",         DALI_INT16)
    .value("INT32",         DALI_INT32)
    .value("INT64",         DALI_INT64)
    .value("FLOAT16",       DALI_FLOAT16)
    .value("FLOAT",         DALI_FLOAT)
    .value("FLOAT64",       DALI_FLOAT64)
    .value("BOOL",          DALI_BOOL)
    .value("STRING",        DALI_STRING)
    .value("_BOOL_VEC",     DALI_BOOL_VEC)
    .value("_INT32_VEC",    DALI_INT_VEC)
    .value("_STRING_VEC",   DALI_STRING_VEC)
    .value("_FLOAT_VEC",    DALI_FLOAT_VEC)
#ifdef DALI_BUILD_PROTO3
    .value("FEATURE",       DALI_TF_FEATURE)
    .value("_FEATURE_VEC",  DALI_TF_FEATURE_VEC)
    .value("_FEATURE_DICT", DALI_TF_FEATURE_DICT)
#endif  // DALI_BUILD_PROTO3
    .value("IMAGE_TYPE",    DALI_IMAGE_TYPE)
    .value("DATA_TYPE",     DALI_DATA_TYPE)
    .value("INTERP_TYPE",   DALI_INTERP_TYPE)
    .value("TENSOR_LAYOUT", DALI_TENSOR_LAYOUT)
    .value("PYTHON_OBJECT", DALI_PYTHON_OBJECT)
    .value("_TENSOR_LAYOUT_VEC", DALI_TENSOR_LAYOUT_VEC)
    .value("_DATA_TYPE_VEC", DALI_DATA_TYPE_VEC)
    .export_values();

  // Placeholder data type allowing to use legacy __call__ method on dtype (to be deprecated).
  py::class_<DALIDataTypePlaceholder>(types_m, "_DALIDataType", dali_data_type)
      .def("__call__",
           [](DALIDataTypePlaceholder self) {
             auto deprecation_func =
                 py::module::import("nvidia.dali.backend").attr("deprecation_warning");
             deprecation_func("Calling '.dtype()' is deprecated, please use '.dtype' instead");
             return FormatStrFromType(static_cast<DALIDataType>(self));
           })
      .def("__repr__",
           [](DALIDataTypePlaceholder self) {
             return FromPythonTrampoline("nvidia.dali.types", "DALIDataType",
                                         "__repr__")(static_cast<DALIDataType>(self));
           })
      .def("__str__", [](DALIDataTypePlaceholder self) {
        return FromPythonTrampoline("nvidia.dali.types", "DALIDataType",
                                    "__str__")(static_cast<DALIDataType>(self));
      });

  // DALIImageType
  py::enum_<DALIImageType>(types_m, "DALIImageType", "Image type\n<SPHINX_IGNORE>")
    .value("RGB", DALI_RGB)
    .value("BGR", DALI_BGR)
    .value("GRAY", DALI_GRAY)
    .value("YCbCr", DALI_YCbCr)
    .value("ANY_DATA", DALI_ANY_DATA)
    .export_values();

  // DALIInterpType
  py::enum_<DALIInterpType>(types_m, "DALIInterpType",
                           "Interpolation mode.\n Note: for 2D inputs, linear and cubic are "
                           "synonymous with bilinear and bicubic, respectively."
                           "\n<SPHINX_IGNORE>")
    .value("INTERP_NN", DALI_INTERP_NN,
          "Nearest neighbour.")
    .value("INTERP_LINEAR", DALI_INTERP_LINEAR,
          "Linear interpolation. Synonymous with bilinear for 2D inputs.")
    .value("INTERP_CUBIC", DALI_INTERP_CUBIC,
          "Cubic interpolation. Synonymous with bicubic for 2D inputs.")
    .value("INTERP_LANCZOS3", DALI_INTERP_LANCZOS3,
          "Resampling with a Lanczos window with 3 lobes.")
    .value("INTERP_TRIANGULAR", DALI_INTERP_TRIANGULAR,
          "Resampling with a triangular window.")
    .value("INTERP_GAUSSIAN", DALI_INTERP_GAUSSIAN,
           "Resampling with a Gaussian window.")
    .export_values();

  py::class_<ExternalContextCheckpoint>(m, "ExternalContextCheckpoint")
    .def(py::init<>())
    .def_property("pipeline_data",
        [](const ExternalContextCheckpoint &self) {
          return py::bytes(self.pipeline_data);
        },
        [](ExternalContextCheckpoint &self, const std::string &new_data) {
          self.pipeline_data = new_data;
        })
    .def_property("iterator_data",
        [](const ExternalContextCheckpoint &self) {
          return py::bytes(self.iterator_data);
        },
        [](ExternalContextCheckpoint &self, const std::string &new_data) {
          self.iterator_data = new_data;
        });

  // Pipeline class and parameters
  ExposeStream(m);
  ExposePipelineParams(m);
  ExposePipeline(m);
  ExposeThreadPool(m);
  ExposeWorkspace(m);
  ExposeOperator(m);

#define DALI_OPSPEC_ADDARG(T) \
    .def("AddArg", \
        [](OpSpec *spec, const string& name, T v) -> OpSpec& { \
        spec->AddArg(name, v); \
        return *spec; \
      }, py::return_value_policy::reference_internal) \
    .def("AddArg", \
        [](OpSpec *spec, const string& name, std::vector<T> v) -> OpSpec& { \
        spec->AddArg(name, v); \
        return *spec; \
      }, py::return_value_policy::reference_internal)

  py::class_<OpSpec>(m, "OpSpec")
    .def(py::init<std::string>(), "name"_a)
    .def("AddInput", [](OpSpec *spec, const string &name, const string &device, bool regular) {
          return spec->AddInput(name, ParseStorageDevice(device), regular);
        },
        "name"_a,
        "device"_a,
        "regular_input"_a = true,
        py::return_value_policy::reference_internal)
    .def("AddArgumentInput", &OpSpec::AddArgumentInput,
        py::return_value_policy::reference_internal)
    .def("AddOutput", [](OpSpec *spec, const string &name, const string &device) {
          return spec->AddOutput(name, ParseStorageDevice(device));
        },
        py::return_value_policy::reference_internal)
    .def("RenameInput", &OpSpec::RenameInput, "idx"_a, "name"_a)
    .def("RenameOutput", &OpSpec::RenameOutput, "idx"_a, "name"_a)
    .def("InputName", &OpSpec::InputName, "idx"_a)
    .def("InputDevice", [](const OpSpec *spec, int idx) {
        return spec->InputDevice(idx) == StorageDevice::GPU ? "gpu" : "cpu";
      }, "idx"_a)
    .def("OutputName", &OpSpec::OutputName, "idx"_a)
    .def("OutputDevice", [](const OpSpec *spec, int idx) {
        return spec->OutputDevice(idx) == StorageDevice::GPU ? "gpu" : "cpu";
      }, "idx"_a)
    .def("NumInput", &OpSpec::NumInput)
    .def("NumRegularInput", &OpSpec::NumRegularInput)
    .def("NumOutput", &OpSpec::NumOutput)
    DALI_OPSPEC_ADDARG(std::string)
    DALI_OPSPEC_ADDARG(bool)
    DALI_OPSPEC_ADDARG(int64_t)
    DALI_OPSPEC_ADDARG(float)
    DALI_OPSPEC_ADDARG(DALIDataType)
    DALI_OPSPEC_ADDARG(DALIImageType)
    DALI_OPSPEC_ADDARG(DALIInterpType)
#ifdef DALI_BUILD_PROTO3
    DALI_OPSPEC_ADDARG(TFFeature)
#endif
    .def("AddArg",
        [](OpSpec *spec, const string &name, py::object obj) -> OpSpec& {
          DALI_FAIL("Unsupported argument type with name " + name);
          return *spec;
        }, py::return_value_policy::reference_internal)
    .def("__repr__", &OpSpec::ToString)
    .def("copy", [](OpSpec &o) -> OpSpec * {
        OpSpec * ret = new OpSpec(o);
        return ret;
        }, py::return_value_policy::take_ownership)
    .def("AddArgEmptyList",
         [](OpSpec *spec, const string &name, DALIDataType data_type) -> OpSpec & {
           TYPE_SWITCH(data_type, type2id, T, (std::string, bool, int32_t, int64_t, float),
             (spec->AddArg(name, std::vector<T>());),
             (DALI_FAIL(make_string("Unsupported data type: ", data_type))));
           return *spec;
         }, py::return_value_policy::reference_internal);

  // Registries for cpu, gpu & mixed operators
  m.def("RegisteredCPUOps", &GetRegisteredCPUOps, py::arg("internal_ops") = false);
  m.def("RegisteredGPUOps", &GetRegisteredGPUOps, py::arg("internal_ops") = false);
  m.def("RegisteredMixedOps", &GetRegisteredMixedOps, py::arg("internal_ops") = false);

  // Registry for OpSchema
  m.def("GetSchema", &GetSchema, py::return_value_policy::reference);
  m.def("TryGetSchema", &TryGetSchema, py::return_value_policy::reference);

  py::class_<OpSchema>(m, "OpSchema")
    .def("Name", &OpSchema::name)
    .def("OperatorName", &OpSchema::OperatorName)
    .def("ModulePath", &OpSchema::ModulePath)
    .def("Dox", &OpSchema::Dox)
    .def("CanUseAutoInputDox", &OpSchema::CanUseAutoInputDox)
    .def("AppendKwargsSection", &OpSchema::AppendKwargsSection)
    .def("HasCallDox", &OpSchema::HasCallDox)
    .def("GetCallDox", &OpSchema::GetCallDox)
    .def("HasInputDox", &OpSchema::HasInputDox)
    .def("GetCallSignatureInputs", &OpSchema::GetCallSignatureInputs)
    .def("GetInputName", &OpSchema::GetInputName)
    .def("GetInputType", &OpSchema::GetInputType)
    .def("GetInputDevice", [](OpSchema *schema,
                              int index,
                              std::optional<std::string_view> actual_device,
                              std::optional<std::string_view> operator_device)->py::object {
        switch (schema->GetInputDevice(index)) {
          case InputDevice::CPU:
            return py::str("cpu");
          case InputDevice::GPU:
            return py::str("gpu");
          case InputDevice::MatchBackend:
            if (operator_device)
              return py::str(*operator_device);
            else
              return py::none();
          case InputDevice::MatchBackendOrCPU:
            if (actual_device) {
              // If the operator is not GPU, the input must be CPU
              if (operator_device && *operator_device != "gpu")
                return py::str("cpu");
              // Otherwise we can just take anything.
              return py::str(*actual_device);
            } else {
              if (operator_device)
                return py::str(*operator_device);
              else
                return py::none();
            }
          case InputDevice::Metadata:
          case InputDevice::Any:
          default:
            if (actual_device)
              return py::str(*actual_device);
            else if (operator_device)
              return py::str(*operator_device);
            else
              return py::none();
        }
      },
      "index"_a,
      "actual_device"_a = py::none(),
      "operator_device"_a = py::none())
    .def("GetInputDox", &OpSchema::GetInputDox)
    .def("MaxNumInput", &OpSchema::MaxNumInput)
    .def("MinNumInput", &OpSchema::MinNumInput)
    .def("HasOutputFn", &OpSchema::HasOutputFn)
    .def("CalculateOutputs", &OpSchema::CalculateOutputs)
    .def("CalculateAdditionalOutputs", &OpSchema::CalculateAdditionalOutputs)
    .def("SupportsInPlace", &OpSchema::SupportsInPlace)
    .def("CheckArgs", &OpSchema::CheckArgs)
    .def("GetArgumentDox", &OpSchema::GetArgumentDox)
    .def("GetArgumentType", &OpSchema::GetArgumentType)
    .def("HasArgumentDefaultValue", &OpSchema::HasArgumentDefaultValue)
    .def("GetArgumentDefaultValueString", &OpSchema::GetArgumentDefaultValueString)
    .def("GetArgumentNames", &OpSchema::GetArgumentNames, "include_hidden"_a = false)
    .def("IsArgumentOptional", &OpSchema::HasOptionalArgument,
        "arg_name"_a)
    .def("IsTensorArgument", &OpSchema::IsTensorArgument)
    .def("ArgSupportsPerFrameInput", &OpSchema::ArgSupportsPerFrameInput)
    .def("IsSequenceOperator", &OpSchema::IsSequenceOperator)
    .def("AllowsSequences", &OpSchema::AllowsSequences)
    .def("SupportsVolumetric", &OpSchema::SupportsVolumetric)
    .def("IsStateful", &OpSchema::IsStateful)
    .def("IsInternal", &OpSchema::IsInternal)
    .def("IsDocHidden", &OpSchema::IsDocHidden)
    .def("IsDocPartiallyHidden", &OpSchema::IsDocPartiallyHidden)
    .def("IsNoPrune", &OpSchema::IsNoPrune)
    .def("IsDeprecated", &OpSchema::IsDeprecated)
    .def("DeprecatedInVersion", &OpSchema::DeprecatedInVersion)
    .def("DeprecatedInFavorOf", &OpSchema::DeprecatedInFavorOf)
    .def("DeprecationMessage", &OpSchema::DeprecationMessage)
    .def("IsDeprecatedArg", &OpSchema::IsDeprecatedArg)
    .def("DeprecatedArgInfo",
        [](OpSchema *schema, const std::string &arg_name) {
          auto meta = schema->DeprecatedArgInfo(arg_name);
          return ArgumentDeprecationInfoToDict(meta);
        })
    .def("GetSupportedLayouts", &OpSchema::GetSupportedLayouts)
    .def("HasArgument",
        [](OpSchema *schema, const std::string &arg_name) {
          return schema->HasArgument(arg_name);
        })
    .def("GetSupportedBackends", &GetSupportedBackends)
    .def("HasRandomSeedArg", &OpSchema::HasRandomSeedArg)
    .def("HasRandomStateArg", &OpSchema::HasRandomStateArg);

  ExposeTensorLayout(types_m);
  ExposeTensor(m);
  ExposeTensorList(m);
  ExposePipelineDebug(m);

  types_m.attr("NHWC") = "HWC";
  types_m.attr("NCHW") = "CHW";
  types_m.attr("NFHWC") = "FHWC";
  types_m.attr("NFCHW") = "FCHW";
  types_m.attr("SAME") = "";

  // We can register exception translator and translate directly into Python error, without
  // tying DALI internals into using the py::type_error
  py::register_exception_translator([](std::exception_ptr p) {
    try {
      if (p)
        std::rethrow_exception(p);
    } catch (const invalid_key &e) {
      PyErr_SetString(PyExc_KeyError, e.what());
    } catch (const DaliRuntimeError &e) {
      PyErr_SetString(PyExc_RuntimeError, e.what());
    } catch (const DaliIndexError &e) {
      PyErr_SetString(PyExc_IndexError, e.what());
    } catch (const DaliTypeError &e) {
      PyErr_SetString(PyExc_TypeError, e.what());
    } catch (const DaliValueError &e) {
      PyErr_SetString(PyExc_ValueError, e.what());
    } catch (const DaliStopIteration &e) {
      PyErr_SetString(PyExc_StopIteration, e.what());
    } catch (const DaliError &e) {
      // Translate top-level errors to RuntimeError.
      PyErr_SetString(PyExc_RuntimeError, e.what());
    }
  });

#ifdef DALI_BUILD_PROTO3
  // TFRecord
  py::module tfrecord_m = m.def_submodule("tfrecord");
  tfrecord_m.doc() = "Additional data structures and constants for TFRecord file format support";
  tfrecord_m.attr("int64") = static_cast<int>(TFFeatureType::int64);
  tfrecord_m.attr("string") = static_cast<int>(TFFeatureType::string);
  tfrecord_m.attr("float32") = static_cast<int>(TFFeatureType::float32);

  py::class_<TFFeature>(tfrecord_m, "Feature")
    .def(py::init<const TFFeature&>());

  tfrecord_m.def("FixedLenFeature",
      [](vector<Index> converted_shape, int type, py::object default_value) {
        TFFeatureType converted_type = static_cast<TFFeatureType>(type);
        TFValue converted_default_value =
          ConvertTFRecordDefaultValue(converted_type, default_value);
        return new TFFeature(converted_shape, converted_type, converted_default_value);
      });
  tfrecord_m.def("VarLenFeature",
      [](int type, py::object default_value) {
        TFFeatureType converted_type = static_cast<TFFeatureType>(type);
        TFValue converted_default_value =
          ConvertTFRecordDefaultValue(converted_type, default_value);
        return new TFFeature(converted_type, converted_default_value);
      });
  tfrecord_m.def("VarLenFeature",
      [](vector<Index> partial_shape, int type, py::object default_value) {
        TFFeatureType converted_type = static_cast<TFFeatureType>(type);
        TFValue converted_default_value =
          ConvertTFRecordDefaultValue(converted_type, default_value);
        return new TFFeature(converted_type, converted_default_value, partial_shape);
      });
#endif  // DALI_BUILD_PROTO3
}  // NOLINT(readability/fn_size)

}  // namespace python
}  // namespace dali
