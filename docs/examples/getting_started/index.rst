Getting Started
===============

Overview
--------

NVIDIA Data Loading Library (DALI) is a collection of highly optimized building blocks and an execution engine that accelerates the data pipeline for computer vision and audio deep learning applications.

Input and augmentation pipelines provided by Deep Learning frameworks fit typically into one of two categories:

* **fast, but inflexible** - written in C++, they are exposed as a single monolithic Python object with very specific set and ordering of operations it provides
* **slow, but flexible** - set of building blocks written in either C++ or Python, that can be used to compose arbitrary data pipelines that end up being slow. One of the biggest overheads for this type of data pipelines is Global Interpreter Lock (GIL) in Python. This forces developers to use multiprocessing, complicating the design of efficient input pipelines.

DALI stands out by providing both performance and flexibility of accelerating different data pipelines. It achieves that by exposing optimized building blocks which are executed using simple and efficient engine, and enabling offloading of operations to GPU (thus enabling scaling to multi-GPU systems).

It is a single library, that can be easily integrated into different deep learning training and inference applications.

DALI offers ease-of-use and flexibility across GPU enabled systems with direct framework plugins, multiple input data formats, and configurable graphs. DALI can help achieve overall speedup on deep learning workflows that are bottlenecked on I/O pipelines due to the limitations of CPU cycles. Typically, systems with high GPU to CPU ratio (such as Amazon EC2 P3.16xlarge, NVIDIA DGX1-V or NVIDIA DGX-2) are constrained on the host CPU, thereby under-utilizing the available GPU compute capabilities. DALI significantly accelerates input processing on such dense GPU configurations to achieve the overall throughput.


Choosing a Mode
---------------

DALI offers two ways to use its operators:

**Pipeline Mode**
    Define a computation graph upfront using the ``@pipeline_def`` decorator, then execute it repeatedly. The graph is optimized and executed efficiently. This is the traditional way to use DALI and is optimized for maximum throughput in production training loops.

**Dynamic Mode**
    Call operators directly as functions without defining a graph upfront. This offers more flexibility than pipeline mode, making it easier to integrate DALI into existing code and to experiment with different processing steps.

Both modes provide the same high-performance operators and GPU acceleration. Choose the mode that best fits your workflow:

* Use **pipeline mode** when you have a fixed processing pipeline and want maximum throughput
* Use **dynamic mode** when you need flexibility, are experimenting, or want to integrate DALI into existing code


Tutorials
---------

.. toctree::
   :maxdepth: 1

   Pipeline Mode <pipeline_mode>
   Dynamic Mode <dynamic_mode>
