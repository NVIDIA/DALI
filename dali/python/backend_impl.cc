// Copyright (c) 2017-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "dali/core/cuda_utils.h"
#include "dali/core/device_guard.h"
#if SHM_WRAPPER_ENABLED
#include "dali/core/os/shared_mem.h"
#endif
#include "dali/core/python_util.h"
#include "dali/operators.h"
#include "dali/operators/reader/parser/tfrecord_parser.h"
#include "dali/pipeline/data/copy_to_external.h"
#include "dali/pipeline/data/dltensor.h"
#include "dali/pipeline/data/tensor.h"
#include "dali/pipeline/data/tensor_list.h"
#include "dali/pipeline/init.h"
#include "dali/pipeline/operator/op_schema.h"
#include "dali/pipeline/operator/op_spec.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/pipeline.h"
#include "dali/plugin/plugin_manager.h"
#include "dali/python/python3_compat.h"
#include "dali/util/half.hpp"
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
  tup[1] = true;
  d["data"] = tup;
  if (std::is_same<Backend, GPUBackend>::value) {
    // see https://numba.pydata.org/numba-doc/dev/cuda/cuda_array_interface.html
    // this set of atributes is tagged as version 2
    d["version"] = 2;
  } else {
    // see https://docs.scipy.org/doc/numpy/reference/arrays.interface.html
    // this set of atributes is tagged as version 3
    d["version"] = 3;
  }
  d["strides"] = py::none();
  return d;
}

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
  for (int i = num_strides - 1; i >= 0; i--) {
    DALI_ENFORCE(strides[i] == stride_from_shape,
        make_string("Strided data not supported. Dimension ", i, " has stride ", strides[i],
        " whereas densely packed data of this shape would have a stride ", stride_from_shape));
    stride_from_shape *= shape[i];
  }
}

template<typename TStrides, typename TShape>
void CheckContiguousTensor(const TStrides &strides, const TShape &shape, size_t element_size) {
  CheckContiguousTensor(strides, dali::size(strides), shape, dali::size(shape), element_size);
}

template<typename SrcBackend, template<typename> class SourceDataType>
void FillTensorFromDlPack(py::capsule capsule, SourceDataType<SrcBackend> *batch, string layout) {
  auto dlm_tensor_ptr = DLMTensorPtrFromCapsule(capsule);
  const auto &dl_tensor = dlm_tensor_ptr->dl_tensor;
  DALI_ENFORCE((std::is_same<SrcBackend, GPUBackend>::value &&
                  dl_tensor.device.device_type == kDLCUDA) ||
               (std::is_same<SrcBackend, CPUBackend>::value &&
                  dl_tensor.device.device_type == kDLCPU),
               "DLPack device type doesn't match Tensor type");

  auto dali_type = TypeTable::GetTypeInfo(DLToDALIType(dl_tensor.dtype));
  TensorShape<> shape;
  shape.resize(dl_tensor.ndim);
  for (ssize_t i = 0; i < dl_tensor.ndim; ++i) {
    shape[i] = dl_tensor.shape[i];
  }

  CheckContiguousTensor(dl_tensor.strides, dl_tensor.ndim, dl_tensor.shape, dl_tensor.ndim, 1);
  size_t bytes = volume(shape) * dali_type.size();

  // empty lambda that just captures dlm_tensor_ptr unique ptr that would be destructed when
  // shared ptr is destroyed
  auto typed_shape = ConvertShape(shape, batch);
  batch->ShareData(shared_ptr<void>(dl_tensor.data,
                                    [dlm_tensor_ptr = move(dlm_tensor_ptr)](void*) {}),
                                    bytes, typed_shape, dali_type);

  // according to the docs kDLCUDAHost = kDLCPU | kDLCUDA so test it as a the first option
  if (dl_tensor.device.device_type == kDLCUDAHost) {
    batch->set_device_id(-1);
    batch->set_pinned(true);
  } else if (dl_tensor.device.device_type == kDLCPU) {
    batch->set_device_id(-1);
  } else if (dl_tensor.device.device_type == kDLCUDA) {
    batch->set_device_id(dl_tensor.device.device_id);
  } else {
    DALI_FAIL(make_string("Not supported DLPack device type: ", dl_tensor.device.device_type, "."));
  }

  batch->SetLayout(layout);
}

template <typename TensorType>
void FillTensorFromCudaArray(const py::object object, TensorType *batch, int device_id,
                             string layout) {
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

  TypeInfo type = TypeFromFormatStr(cu_a_interface["typestr"].cast<py::str>());
  size_t bytes = volume(shape) * type.size();

  if (cu_a_interface.contains("strides") && !cu_a_interface["strides"].is_none()) {
    TensorShape<> strides = shape_from_py(cu_a_interface["strides"].cast<py::tuple>());
    CheckContiguousTensor(strides, shape, type.size());
  }

  auto typed_shape = ConvertShape(shape, batch);
  auto *ptr = PyLong_AsVoidPtr(cu_a_interface["data"].cast<py::tuple>()[0].ptr());

  // Keep a copy of the input object ref in the deleter, so its refcount is increased
  // while this shared_ptr is alive (and the data should be kept alive)
  // We set the type and shape even before the set_device_id as we only wrap the allocation
  batch->ShareData(shared_ptr<void>(ptr, [obj_ref = object](void *) {}), bytes, typed_shape, type);
  batch->SetLayout(layout);
  // it is for __cuda_array_interface__ so device_id < 0 is not a valid value
  if (device_id < 0) {
    CUDA_CALL(cudaGetDevice(&device_id));
  }
  batch->set_device_id(device_id);
}

void ExposeTensorLayout(py::module &m) {
  py::class_<TensorLayout> tl(m, "TensorLayout");
  tl.def(py::init([](string s) {
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
    .def(py::init([](py::capsule &capsule, string layout = "") {
          auto t = std::make_unique<Tensor<CPUBackend>>();
          FillTensorFromDlPack(capsule, t.get(), layout);
          return t.release();
        }),
      "object"_a,
      "layout"_a = "",
      R"code(
      Wrap a DLPack Tensor residing in the CPU memory.

      object : DLPack object
            Python DLPack object
      layout : str
            Layout of the data
      )code")
    .def_buffer([](Tensor<CPUBackend> &t) -> py::buffer_info {
          DALI_ENFORCE(IsValidType(t.type()), "Cannot produce "
            "buffer info for tensor w/ invalid type.");

          std::vector<ssize_t> shape(t.ndim()), stride(t.ndim());
          size_t dim_prod = 1;
          for (int i = 0; i < t.ndim(); ++i) {
            shape[i] = t.shape()[i];

            // We iterate over stride backwards
            stride[(t.ndim()-1) - i] = t.type().size()*dim_prod;
            dim_prod *= t.shape()[(t.ndim()-1) - i];
          }

          return py::buffer_info(
              t.raw_mutable_data(),
              t.type().size(),
              FormatStrFromType(t.type()),
              t.ndim(), shape, stride);
        })
    .def(py::init([](py::buffer b, string layout = "", bool is_pinned = false) {
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

          // Create the Tensor and wrap the data
          auto t = std::make_unique<Tensor<CPUBackend>>();
          t->set_pinned(is_pinned);
          TypeInfo type = TypeFromFormatStr(info.format);
          // Keep a copy of the input buffer ref in the deleter, so its refcount is increased
          // while this shared_ptr is alive (and the data should be kept alive)
          t->ShareData(shared_ptr<void>(info.ptr, [buf_ref = b](void *) {}), bytes, i_shape, type);
          t->SetLayout(layout);
          return t.release();
        }),
      "b"_a,
      "layout"_a = "",
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
    .def("copy_to_external",
        [](Tensor<CPUBackend> &t, py::object p) {
          CopyToExternal<mm::memory_kind::host>(ctypes_void_ptr(p), t, 0, false);
        },
      "ptr"_a,
      R"code(
      Copy to external pointer in the CPU memory.

      ptr : ctypes.c_void_p
            Destination of the copy.
      )code")
    .def("dtype",
        [](Tensor<CPUBackend> &t) {
          return FormatStrFromType(t.type());
        },
      R"code(
      String representing NumPy type of the Tensor.
      )code")
    .def("data_ptr", [](Tensor<CPUBackend> &t) {
          return py::reinterpret_borrow<py::object>(PyLong_FromVoidPtr(t.raw_mutable_data()));
        },
      R"code(
      Returns the address of the first element of tensor.
      )code")
    .def_property("__array_interface__", &ArrayInterfaceRepr<CPUBackend>, nullptr,
      R"code(
      Returns Array Interface representation of TensorCPU.
      )code");
  tensor_cpu_binding.doc() = R"code(
      Class representing a Tensor residing in host memory. It can be used to access individual
      samples of a :class:`TensorListCPU` or used to wrap CPU memory that is intended
      to be passed as an input to DALI.

      It is compatible with `Python Buffer Protocol <https://docs.python.org/3/c-api/buffer.html>`_
      and `NumPy Array Interface <https://numpy.org/doc/stable/reference/arrays.interface.html>`_.)code";

  auto tensor_gpu_binding = py::class_<Tensor<GPUBackend>>(m, "TensorGPU")
    .def(py::init([](py::capsule &capsule, string layout = "") {
          auto t = std::make_unique<Tensor<GPUBackend>>();
          FillTensorFromDlPack(capsule, t.get(), layout);
          return t.release();
        }),
      "object"_a,
      "layout"_a = "",
      R"code(
      Wrap a DLPack Tensor residing in the GPU memory.

      object : DLPack object
            Python DLPack object
      layout : str
            Layout of the data
      )code")
    .def(py::init([](const py::object object, string layout = "", int device_id = -1) {
          auto t = std::make_unique<Tensor<GPUBackend>>();
          FillTensorFromCudaArray(object, t.get(), device_id, layout);
          return t.release();
        }),
      "object"_a,
      "layout"_a = "",
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
    .def("layout", [](Tensor<GPUBackend> &t) {
      return t.GetLayout().str();
    })
    .def("as_cpu", [](Tensor<GPUBackend> &t) -> Tensor<CPUBackend>* {
          auto ret = std::make_unique<Tensor<CPUBackend>>();
          ret->set_pinned(false);
          UserStream * us = UserStream::Get();
          cudaStream_t s = us->GetStream(t);
          DeviceGuard g(t.device_id());
          ret->Copy(t, s);
          us->Wait(t);
          return ret.release();
        },
      R"code(
      Returns a `TensorCPU` object being a copy of this `TensorGPU`.
      )code",
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
    .def("copy_to_external",
        [](Tensor<GPUBackend> &t, py::object p, py::object cuda_stream,
           bool non_blocking, bool use_copy_kernel) {
          void *ptr = ctypes_void_ptr(p);
          cudaStream_t stream = cuda_stream.is_none()
                ? UserStream::Get()->GetStream(t)
                : static_cast<cudaStream_t>(ctypes_void_ptr(cuda_stream));
          CopyToExternal<mm::memory_kind::device>(ptr, t, stream, use_copy_kernel);
          if (!non_blocking) {
            CUDA_CALL(cudaStreamSynchronize(stream));
          }
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
    .def("dtype",
        [](Tensor<GPUBackend> &t) {
          return FormatStrFromType(t.type());
        },
      R"code(
      String representing NumPy type of the Tensor.
      )code")
    .def("data_ptr",
        [](Tensor<GPUBackend> &t) {
          return py::reinterpret_borrow<py::object>(PyLong_FromVoidPtr(t.raw_mutable_data()));
        },
      R"code(
      Returns the address of the first element of tensor.
      )code")
    .def_property("__cuda_array_interface__",  &ArrayInterfaceRepr<GPUBackend>, nullptr,
      R"code(
      Returns CUDA Array Interface (Version 2) representation of TensorGPU.
      )code");
  tensor_gpu_binding.doc() = R"code(
      Class representing a Tensor residing in GPU memory. It can be used to access individual
      samples of a :class:`TensorListGPU` or used to wrap GPU memory that is intended
      to be passed as an input to DALI.

      It is compatible with `CUDA Array Interface <https://numba.pydata.org/numba-doc/dev/cuda/cuda_array_interface.html>`_.)code";
}

template <typename Backend>
std::unique_ptr<Tensor<Backend> > TensorListGetItemImpl(TensorList<Backend> &t, Index id) {
  int num_tensors = static_cast<int>(t.ntensor());
  if (id < 0) {
    id = num_tensors + id;
  }
  if (id >= num_tensors || id < 0) {
      throw py::index_error("TensorListCPU index out of range");
  }
  auto ptr = std::make_unique<Tensor<Backend>>();
  ptr->ShareData(&t, id);
  return ptr;
}

#if 0  // TODO(spanev): figure out which return_value_policy to choose
template <typename Backend>
py::tuple TensorListGetItemSliceImpl(TensorList<Backend> &t, py::slice slice) {
  size_t start, stop, step, slicelength;
  if (!slice.compute(t.ntensor(), &start, &stop, &step, &slicelength))
      throw py::error_already_set();
  py::list list;
  for (; start < stop; start += step) {
      auto ptr = make_uqnieu<Tensor<Backend>>();
      ptr->ShareData(&t, static_cast<int>(start));
      list.append(ptr.release());
  }
  return list;
}
#endif

void ExposeTensorList(py::module &m) {
  py::class_<TensorList<CPUBackend>, std::shared_ptr<TensorList<CPUBackend>>>(
      m, "TensorListCPU", py::buffer_protocol())
    .def(py::init([](py::capsule &capsule, string layout = "") {
            auto t = std::make_shared<TensorList<CPUBackend>>();
            FillTensorFromDlPack(capsule, t.get(), layout);
            return t;
          }),
        "object"_a,
        "layout"_a = "",
        R"code(
        List of tensors residing in the CPU memory.

        object : DLPack object
              Python DLPack object representing TensorList
        layout : str
              Layout of the data
        )code")
    .def(py::init([](TensorList<CPUBackend> *tl, py::object layout) {
          if (!tl)
            throw py::value_error("The source object must not be null");
          auto t = std::make_shared<TensorList<CPUBackend>>();
          t->ShareData(tl);
          if (!layout.is_none()) {
            if (!py::isinstance<py::str>(layout))
              throw py::type_error("`layout` must be a string or None");
            t->SetLayout(std::string(layout.cast<py::str>()));
          }
          return t;
        }),
      "tl"_a,
      "layout"_a = py::none())
    .def(py::init([](py::buffer b, string layout = "", bool is_pinned = false) {
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

        // Create the Tensor and wrap the data
        auto t = std::make_shared<TensorList<CPUBackend>>();
        t->set_pinned(false);
        TypeInfo type = TypeFromFormatStr(info.format);
        // Keep a copy of the input buffer ref in the deleter, so its refcount is increased
        // while this shared_ptr is alive (and the data should be kept alive)
        t->ShareData(shared_ptr<void>(info.ptr, [buf_ref = b](void *){}), bytes, i_shape, type);
        t->SetLayout(layout);
        return t;
      }),
      "b"_a,
      "layout"_a = "",
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
    .def("layout", [](TensorList<CPUBackend> &t) {
      return t.GetLayout().str();
    })
    .def("at", [](TensorList<CPUBackend> &tl, Index id) -> py::array {
          DALI_ENFORCE(IsValidType(tl.type()), "Cannot produce "
              "buffer info for tensor w/ invalid type.");
          DALI_ENFORCE(static_cast<size_t>(id) < tl.ntensor(), "Index is out-of-range.");
          DALI_ENFORCE(id >= 0, "Index is out-of-range.");

          std::vector<ssize_t> shape(tl.tensor_shape(id).size()),
                                     stride(tl.tensor_shape(id).size());
          size_t dim_prod = 1;
          for (size_t i = 0; i < shape.size(); ++i) {
            shape[i] = tl.tensor_shape(id)[i];

            // We iterate over stride backwards
            stride[(stride.size()-1) - i] = tl.type().size()*dim_prod;
            dim_prod *= tl.tensor_shape(id)[(shape.size()-1) - i];
          }

          return py::array(py::buffer_info(
              tl.raw_mutable_tensor(id),
              tl.type().size(),
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
#if 0  // TODO(spanev): figure out which return_value_policy to choose
      .def("__getitem__",
        [](TensorList<CPUBackend> &tl, py::slice slice) -> py::tuple {
          return TensorListGetItemSliceImpl(tl, slice);
        },
      R"code(
      Returns a tensor at given position in the list.

      )code")
#endif
    .def("as_array", [](TensorList<CPUBackend> &tl) -> py::array {
          void* raw_mutable_data = nullptr;
          std::string format;
          size_t type_size;

          if (tl.size() > 0) {
            DALI_ENFORCE(IsValidType(tl.type()), "Cannot produce "
                "buffer info for tensor w/ invalid type.");
            DALI_ENFORCE(tl.IsDenseTensor(),
                        "Tensors in the list must have the same shape");
            raw_mutable_data = tl.raw_mutable_data();
          }

          if (IsValidType(tl.type())) {
            format = FormatStrFromType(tl.type());
            type_size = tl.type().size();
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
          return tl.ntensor();
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
          CopyToExternal<mm::memory_kind::host>(ctypes_void_ptr(p), tl, 0, false);
        },
      R"code(
      Copy the contents of this `TensorList` to an external pointer
      (of type `ctypes.c_void_p`) residing in CPU memory.

      This function is used internally by plugins to interface with
      tensors from supported Deep Learning frameworks.

      )code")
    .def("as_reshaped_tensor",
        [](TensorList<CPUBackend> &tl, const vector<Index> &new_shape) -> Tensor<CPUBackend>* {
          return tl.AsReshapedTensor(new_shape);
        },
      R"code(
      Returns a tensor that is a view of this `TensorList` cast to the given shape.

      This function can only be called if `TensorList` is contiguous in memory and
      the volumes of requested `Tensor` and `TensorList` matches.
      )code",
      py::return_value_policy::reference_internal)
    .def("as_tensor", &TensorList<CPUBackend>::AsTensor,
      R"code(
      Returns a tensor that is a view of this `TensorList`.

      This function can only be called if `is_dense_tensor` returns `True`.
      )code",
      py::return_value_policy::reference_internal)
    .def("data_ptr",
        [](TensorList<CPUBackend> &tl) {
          return py::reinterpret_borrow<py::object>(PyLong_FromVoidPtr(tl.raw_mutable_data()));
        },
      R"code(
      Returns the address of the first element of TensorList.
      )code");

  py::class_<TensorList<GPUBackend>, std::shared_ptr<TensorList<GPUBackend>>>(
      m, "TensorListGPU", py::buffer_protocol())
    .def(py::init([](py::capsule &capsule, string layout = "") {
            auto t = std::make_shared<TensorList<GPUBackend>>();
            FillTensorFromDlPack(capsule, t.get(), layout);
            return t;
          }),
        "object"_a,
        "layout"_a = "",
        R"code(
        List of tensors residing in the GPU memory.

        object : DLPack object
              Python DLPack object representing TensorList
        layout : str
              Layout of the data
        )code")
    .def(py::init([](TensorList<GPUBackend> *tl, py::object layout) {
          if (!tl)
            throw py::value_error("The source object must not be null");
          auto t = std::make_shared<TensorList<GPUBackend>>();
          t->ShareData(tl);
          if (!layout.is_none()) {
            if (!py::isinstance<py::str>(layout))
              throw py::type_error("`layout` must be a string or None");
            t->SetLayout(std::string(layout.cast<py::str>()));
          }
          return t;
        }),
      "tl"_a,
      "layout"_a = py::none())
    .def(py::init([](const py::object object, string layout = "", int device_id = -1) {
          auto t = std::make_shared<TensorList<GPUBackend>>();
          FillTensorFromCudaArray(object, t.get(), device_id, layout);
          return t;
        }),
      "object"_a,
      "layout"_a = "",
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
    .def("as_cpu", [](TensorList<GPUBackend> &t) {
          auto ret = std::make_shared<TensorList<CPUBackend>>();
          ret->set_pinned(false);
          UserStream * us = UserStream::Get();
          cudaStream_t s = us->GetStream(t);
          DeviceGuard g(t.device_id());
          ret->Copy(t, s);
          us->Wait(t);
          return ret;
        },
      R"code(
      Returns a `TensorListCPU` object being a copy of this `TensorListGPU`.
      )code",
      py::return_value_policy::take_ownership)
    .def("__len__", [](TensorList<GPUBackend> &t) {
          return t.ntensor();
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
          void *ptr = ctypes_void_ptr(p);
          cudaStream_t stream = cuda_stream.is_none()
                ? UserStream::Get()->GetStream(t)
                : static_cast<cudaStream_t>(ctypes_void_ptr(cuda_stream));
          CopyToExternal<mm::memory_kind::device>(ptr, t, stream, use_copy_kernel);
          if (!non_blocking) {
            CUDA_CALL(cudaStreamSynchronize(stream));
          }
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
    .def("__getitem__",
        [](TensorList<GPUBackend> &t, Index i) -> std::unique_ptr<Tensor<GPUBackend>> {
          return TensorListGetItemImpl(t, i);
        },
      "i"_a,
      R"code(
      Returns a tensor at given position `i` in the list.
      )code",
      py::keep_alive<0, 1>())
#if 0  // TODO(spanev): figure out which return_value_policy to choose
      .def("__getitem__",
        [](TensorList<GPUBackend> &t, py::slice slice) -> py::tuple {
          return TensorListGetItemSliceImpl(t, slice);
        },
      R"code(
      Returns a tensor at given position in the list.
      )code")
#endif
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
    .def("as_reshaped_tensor",
        [](TensorList<GPUBackend> &tl, const vector<Index> &new_shape) -> Tensor<GPUBackend>* {
          return tl.AsReshapedTensor(new_shape);
        },
      R"code(
      Returns a tensor that is a view of this `TensorList` cast to the given shape.

      This function can only be called if `TensorList` is contiguous in memory and
      the volumes of requested `Tensor` and `TensorList` matches.
      )code",
      py::return_value_policy::reference_internal)
    .def("as_tensor", &TensorList<GPUBackend>::AsTensor,
      R"code(
      Returns a tensor that is a view of this `TensorList`.

      This function can only be called if `is_dense_tensor` returns `True`.
      )code",
      py::return_value_policy::reference_internal)
    .def("data_ptr",
        [](TensorList<GPUBackend> &tl) {
          return py::reinterpret_borrow<py::object>(PyLong_FromVoidPtr(tl.raw_mutable_data()));
        },
      R"code(
      Returns the address of the first element of TensorList.
      )code");
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
typedef dali::TFRecordParser::FeatureType TFFeatureType;
typedef dali::TFRecordParser::Feature TFFeature;
typedef TFFeature::Value TFValue;

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
}

py::dict DeprecatedArgMetaToDict(const DeprecatedArgDef & meta) {
  py::dict d;
  d["msg"] = meta.msg;
  d["removed"] = meta.removed;
  d["renamed_to"] = meta.renamed_to;
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

template <typename Backend>
void FeedPipeline(Pipeline *p, const string &name, py::list list, cudaStream_t stream,
                  bool sync = false, bool use_copy_kernel = false) {
  TensorVector<Backend> tv(list.size());
  for (size_t i = 0; i < list.size(); ++i) {
    auto &t = list[i].cast<Tensor<Backend>&>();
    tv[i] = std::move(t);
  }
  p->SetExternalInput(name, tv, stream, sync, use_copy_kernel);
}

PYBIND11_MODULE(backend_impl, m) {
  dali::InitOperatorsLib();
  m.doc() = "Python bindings for the C++ portions of DALI";

  // DALI Init function
  m.def("Init", &DALIInit);

  ExposeBufferPolicyFunctions(m);

  m.def("LoadLibrary", &PluginManager::LoadLibrary);

  m.def("GetCxx11AbiFlag", &GetCxx11AbiFlag);

  m.def("HasCudaContext", []{
    if (!cuInitChecked()) {
      return false;
    }
    CUcontext context;
    CUDA_CALL(cuCtxGetCurrent(&context));
    return context != nullptr;
  });

#if SHM_WRAPPER_ENABLED

  py::class_<SharedMem>(m, "SharedMem")
      .def(py::init<int, int>())
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
      .def("close", &SharedMem::close);

#endif

  // Types
  py::module types_m = m.def_submodule("types");
  types_m.doc() = "Datatypes and options used by DALI";
  types_m.add_object("CPU_ONLY_DEVICE_ID", PyLong_FromLong(CPU_ONLY_DEVICE_ID));

  // DALIDataType
  py::enum_<DALIDataType>(types_m, "DALIDataType", "Data type of image.\n<SPHINX_IGNORE>")
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

  // DALIImageType
  py::enum_<DALIImageType>(types_m, "DALIImageType", "Image type\n<SPHINX_IGNORE>")
    .value("RGB", DALI_RGB)
    .value("BGR", DALI_BGR)
    .value("GRAY", DALI_GRAY)
    .value("YCbCr", DALI_YCbCr)
    .value("ANY_DATA", DALI_ANY_DATA)
    .export_values();

  // DALIInterpType
  py::enum_<DALIInterpType>(types_m, "DALIInterpType", "Interpolation mode\n<SPHINX_IGNORE>")
    .value("INTERP_NN", DALI_INTERP_NN)
    .value("INTERP_LINEAR", DALI_INTERP_LINEAR)
    .value("INTERP_CUBIC", DALI_INTERP_CUBIC)
    .value("INTERP_LANCZOS3", DALI_INTERP_LANCZOS3)
    .value("INTERP_TRIANGULAR", DALI_INTERP_TRIANGULAR)
    .value("INTERP_GAUSSIAN", DALI_INTERP_GAUSSIAN)
    .export_values();

  // Operator node
  py::class_<OpNode>(m, "OpNode")
    .def("instance_name",
        [](OpNode* node) {
          return node->instance_name;
        })
    .def("name",
        [](OpNode* node) {
          return node->spec.name();
        });

  // Pipeline class
  py::class_<Pipeline>(m, "Pipeline")
    .def(py::init(
            [](int batch_size, int num_threads, int device_id, int64_t seed = -1,
                bool pipelined_execution = true, int prefetch_queue_depth = 2,
                bool async_execution = true, size_t bytes_per_sample_hint = 0,
                bool set_affinity = false, int max_num_stream = -1,
                int default_cuda_stream_priority = 0) {
              return std::make_unique<Pipeline>(
                      batch_size, num_threads, device_id, seed, pipelined_execution,
                      prefetch_queue_depth, async_execution, bytes_per_sample_hint, set_affinity,
                      max_num_stream, default_cuda_stream_priority);
            }),
        "batch_size"_a,
        "num_threads"_a,
        "device_id"_a,
        "seed"_a = -1,
        "exec_pipelined"_a = true,
        "prefetch_queue_depth"_a = 2,
        "exec_async"_a = true,
        "bytes_per_sample_hint"_a = 0,
        "set_affinity"_a = false,
        "max_num_stream"_a = -1,
        "default_cuda_stream_priority"_a = 0
        )
    // initialize from serialized pipeline
    .def(py::init(
          [](string serialized_pipe,
             int batch_size = -1, int num_threads = -1, int device_id = -1,
             bool pipelined_execution = true,  int prefetch_queue_depth = 2,
             bool async_execution = true, size_t bytes_per_sample_hint = 0,
             bool set_affinity = false, int max_num_stream = -1,
             int default_cuda_stream_priority = 0) {
              return std::make_unique<Pipeline>(
                               serialized_pipe,
                               batch_size, num_threads, device_id, pipelined_execution,
                               prefetch_queue_depth, async_execution, bytes_per_sample_hint,
                               set_affinity, max_num_stream, default_cuda_stream_priority);
            }),
        "serialized_pipe"_a,
        "batch_size"_a = -1,
        "num_threads"_a = -1,
        "device_id"_a = -1,
        "exec_pipelined"_a = true,
        "prefetch_queue_depth"_a = 2,
        "exec_async"_a = true,
        "bytes_per_sample_hint"_a = 0,
        "set_affinity"_a = false,
        "max_num_stream"_a = -1,
        "default_cuda_stream_priority"_a = 0
        )
    .def("AddOperator",
         static_cast<int (Pipeline::*)(const OpSpec &, const std::string &)>
                                      (&Pipeline::AddOperator))
    .def("AddOperator",
         static_cast<int (Pipeline::*)(const OpSpec &, const std::string &, int)>
                                      (&Pipeline::AddOperator))
    .def("GetOperatorNode", &Pipeline::GetOperatorNode)
    .def("Build",
        [](Pipeline *p, const std::vector<std::pair<string, string>>& outputs) {
          p->Build(outputs);
          })
    .def("Build",
        [](Pipeline *p) {
          p->Build();
          })
    .def("SetExecutionTypes",
        [](Pipeline *p, bool exec_pipelined, bool exec_separated, bool exec_async) {
          p->SetExecutionTypes(exec_pipelined, exec_separated, exec_async);
        },
        "exec_pipelined"_a = true,
        "exec_separated"_a = false,
        "exec_async"_a = true)
    .def("EnableExecutorMemoryStats",
        [](Pipeline *p, bool enable_memory_stats) {
          p->EnableExecutorMemoryStats(enable_memory_stats);
        },
        "enable_memory_stats"_a = true)
    .def("executor_statistics",
        [](Pipeline *p) {
          auto ret = p->GetExecutorMeta();
          return ExecutorMetaToDict(ret);
        })
    .def("SetQueueSizes",
        [](Pipeline *p, int cpu_size, int gpu_size) {
          p->SetQueueSizes(cpu_size, gpu_size);
        })
    .def("SetOutputNames",
        [](Pipeline *p, const std::vector<std::pair<string, string>>& outputs) {
          p->SetOutputNames(outputs);
          })
    .def("RunCPU", &Pipeline::RunCPU, py::call_guard<py::gil_scoped_release>())
    .def("RunGPU", &Pipeline::RunGPU)
    .def("Outputs",
        [](Pipeline *p) {
          DeviceWorkspace ws;
          p->Outputs(&ws);

          py::tuple outs(ws.NumOutput());
          for (int i = 0; i < ws.NumOutput(); ++i) {
            if (ws.OutputIsType<CPUBackend>(i)) {
              outs[i] = ws.OutputPtr<CPUBackend>(i);
            } else {
              outs[i] = ws.OutputPtr<GPUBackend>(i);
            }
          }
          return outs;
        }, py::return_value_policy::take_ownership)
    .def("ShareOutputs",
        [](Pipeline *p) {
          DeviceWorkspace ws;
          p->ShareOutputs(&ws);

          py::tuple outs(ws.NumOutput());
          for (int i = 0; i < ws.NumOutput(); ++i) {
            if (ws.OutputIsType<CPUBackend>(i)) {
              outs[i] = ws.OutputPtr<CPUBackend>(i);
            } else {
              outs[i] = ws.OutputPtr<GPUBackend>(i);
            }
          }
          return outs;
        }, py::return_value_policy::take_ownership)
    .def("ReleaseOutputs",
        [](Pipeline *p) {
          p->ReleaseOutputs();
        })
    .def("batch_size", &Pipeline::batch_size)
    .def("num_threads", &Pipeline::num_threads)
    .def("device_id", &Pipeline::device_id)
    .def("SetExternalTLInput",
        [](Pipeline *p, const string &name, const TensorList<CPUBackend> &tl,
           py::object /*cuda_stream*/, bool /*use_copy_kernel*/) {
          p->SetExternalInput(name, tl, 0, true);
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
          bool is_cpu_data = conv.load(static_cast<py::object>(list[0]), true);
          if (is_cpu_data) {
            FeedPipeline<CPUBackend>(p, name, list, 0, true);
          } else {
            cudaStream_t stream = cuda_stream.is_none()
                                ? UserStream::Get()->GetStream(list[0].cast<Tensor<GPUBackend>&>())
                                : static_cast<cudaStream_t>(ctypes_void_ptr(cuda_stream));
            FeedPipeline<GPUBackend>(p, name, list, stream, cuda_stream.is_none(), use_copy_kernel);
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
        "show_ids"_a = false,
        "use_colors"_a = false)
    .def("reader_meta", [](Pipeline* p) {
          std::map<std::string, ReaderMeta> meta_map = p->GetReaderMeta();
          py::dict d;
          for (auto const& value : meta_map) {
            d[value.first.c_str()] = ReaderMetaToDict(value.second);
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
    .def("AddInput", &OpSpec::AddInput,
        "name"_a,
        "device"_a,
        "regular_input"_a = true,
        py::return_value_policy::reference_internal)
    .def("AddArgumentInput", &OpSpec::AddArgumentInput,
        py::return_value_policy::reference_internal)
    .def("AddOutput", &OpSpec::AddOutput,
        py::return_value_policy::reference_internal)
    DALI_OPSPEC_ADDARG(std::string)
    DALI_OPSPEC_ADDARG(bool)
    DALI_OPSPEC_ADDARG(int64)
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
    .def("Dox", &OpSchema::Dox)
    .def("CanUseAutoInputDox", &OpSchema::CanUseAutoInputDox)
    .def("AppendKwargsSection", &OpSchema::AppendKwargsSection)
    .def("HasCallDox", &OpSchema::HasCallDox)
    .def("GetCallDox", &OpSchema::GetCallDox)
    .def("HasInputDox", &OpSchema::HasInputDox)
    .def("GetCallSignatureInputs", &OpSchema::GetCallSignatureInputs)
    .def("GetInputName", &OpSchema::GetInputName)
    .def("GetInputType", &OpSchema::GetInputType)
    .def("GetInputDevice", [](OpSchema *schema, int index)->py::object {
      switch (schema->GetInputDevice(index)) {
        case InputDevice::CPU:
          return py::str("cpu");
        case InputDevice::GPU:
          return py::str("gpu");
        default:
          return py::none();
      }
    })
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
    .def("GetArgumentNames", &OpSchema::GetArgumentNames)
    .def("IsArgumentOptional", &OpSchema::HasOptionalArgument,
        "arg_name"_a,
        "local_only"_a = false)
    .def("IsTensorArgument", &OpSchema::IsTensorArgument)
    .def("IsSequenceOperator", &OpSchema::IsSequenceOperator)
    .def("AllowsSequences", &OpSchema::AllowsSequences)
    .def("SupportsVolumetric", &OpSchema::SupportsVolumetric)
    .def("IsInternal", &OpSchema::IsInternal)
    .def("IsDocHidden", &OpSchema::IsDocHidden)
    .def("IsDocPartiallyHidden", &OpSchema::IsDocPartiallyHidden)
    .def("IsNoPrune", &OpSchema::IsNoPrune)
    .def("IsDeprecated", &OpSchema::IsDeprecated)
    .def("DeprecatedInFavorOf", &OpSchema::DeprecatedInFavorOf)
    .def("DeprecationMessage", &OpSchema::DeprecationMessage)
    .def("IsDeprecatedArg", &OpSchema::IsDeprecatedArg)
    .def("DeprecatedArgMeta",
        [](OpSchema *schema, const std::string &arg_name) {
          auto meta = schema->DeprecatedArgMeta(arg_name);
          return DeprecatedArgMetaToDict(meta);
        })
    .def("GetSupportedLayouts", &OpSchema::GetSupportedLayouts);

  ExposeTensorLayout(types_m);
  ExposeTensor(m);
  ExposeTensorList(m);

  types_m.attr("NHWC") = "HWC";
  types_m.attr("NCHW") = "CHW";
  types_m.attr("NFHWC") = "FHWC";
  types_m.attr("NFCHW") = "FCHW";
  types_m.attr("SAME") = "";

#ifdef DALI_BUILD_PROTO3
  // TFRecord
  py::module tfrecord_m = m.def_submodule("tfrecord");
  tfrecord_m.doc() = "Additional data structures and constants for TFRecord file format support";
  tfrecord_m.attr("int64") = static_cast<int>(TFFeatureType::int64);
  tfrecord_m.attr("string") = static_cast<int>(TFFeatureType::string);
  tfrecord_m.attr("float32") = static_cast<int>(TFFeatureType::float32);

  py::class_<TFFeature>(tfrecord_m, "Feature");

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
}

}  // namespace python
}  // namespace dali
