Create a custom operator
==========================

DALI allows you to create a custom operator in C++ and load it at runtime.
There are several reasons you might need to write your custom operator, for instance

- DALI doesn't support the operation you want to perform and it cannot be expressed by a composition of other operators
- You want to write an operator that depends on a third party library
- You want to optimize your pipeline by providing a manually fused operation in C++

In this tutorial we will walk you through the process of writing, compiling and loading a plugin with a DALI custom operator. For demonstration purposes we will provide a CPU and a GPU implementation for the 'CustomDummy' operator. The implementation will just copy the input data to the output without any modifications.


PREREQUISITES:
    - Knowledge of C++
    - DALI installed from the binary distribution or compiled from source
    - Basic knowledge of CMake

Defining the operator (dummy.h)
+++++++++++++++++++++++++++++++
First we define the operator in a header file 

.. literalinclude:: customdummy/dummy.h
   :language: c++

CPU implementation (dummy.cc)
+++++++++++++++++++++++++++++++
Next, we provide the CPU implementation in a C++ implementation file. We register the schema for the custom operator with DALI_REGISTER_SCHEMA macro and define the CPU version of the operator with DALI_REGISTER_OPERATOR.

.. literalinclude:: customdummy/dummy.cc
   :language: c++

GPU implementation (dummy.cu)
+++++++++++++++++++++++++++++++
Similarly, we provide a GPU implementation in a CUDA implementation file and register it with DALI_REGISTER_OPERATOR

.. literalinclude:: customdummy/dummy.cu
   :language: c++

Compiling the plugin (CMakeLists.txt)
+++++++++++++++++++++++++++++++++++++++
The last step is to specify the build configuration.

We can use nvidia.dali.sysconfig to retrieve the build configuration parameters

.. code-block:: none

    >>> import nvidia.dali.sysconfig as sysconfig
    >>> sysconfig.get_include_dir()
    '/usr/local/lib/python3.5/dist-packages/nvidia/dali/include'
    >>> sysconfig.get_lib_dir()
    '/usr/local/lib/python3.5/dist-packages/nvidia/dali'
    >>> sysconfig.get_compile_flags()
    ['-I/usr/local/lib/python3.5/dist-packages/nvidia/dali/include', '-D_GLIBCXX_USE_CXX11_ABI=0']
    >>> sysconfig.get_link_flags()
    ['-L/usr/local/lib/python3.5/dist-packages/nvidia/dali', '-ldali']

In this example we used CMake to build the plugin

.. literalinclude:: customdummy/CMakeLists.txt
   :language: cmake

Now we are ready to compile our plugin containing the custom operator 'CustomDummy'

.. code-block:: bash

    mkdir build
    cd build
    cmake ..
    make

After the build steps we should have a dynamic library file 'libcustomdummy.so' created and ready to use.

Loading the plugin
+++++++++++++++++++
First we can see that there is no such plugin named 'CustomDummy'

.. code-block:: none

    >>> import nvidia.dali.ops as ops
    >>> help(ops.CustomDummy)
    Traceback (most recent call last):
    File "<stdin>", line 1, in <module>
    AttributeError: 'module' object has no attribute 'CustomDummy'

We can now load the plugin and verify that the new operator is available

.. code-block:: none

    >>> import nvidia.dali.plugin_manager as plugin_manager
    >>> plugin_manager.load_library('./libcustomdummy.so')
    >>> help(ops.CustomDummy)

    Help on class CustomDummy in module nvidia.dali.ops:

    class CustomDummy(__builtin__.object)
    |  This is 'CPU', 'GPU' operator
    |  
    |  Make a copy of the input tensor
    |  
    |  Parameters
    |  ----------
    |  
    |  Methods defined here:
    |  
    |  __call__(self, *inputs, **kwargs)
    |  
    |  __init__(self, **kwargs)
    |  
    |  ----------------------------------------------------------------------
    |  Data descriptors defined here:
    |  
    |  __dict__
    |      dictionary for instance variables (if defined)
    |  
    |  __weakref__
    |      list of weak references to the object (if defined)
    |  
    |  device
    |  
    |  schema
    |  
    |  spec

