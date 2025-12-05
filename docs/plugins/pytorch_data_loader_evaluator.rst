Loader Evaluator Tool
================================

A Lightweight, Self-Service Diagnostic for Data Loading Bottlenecks
---------------------------------------------------------------------

.. toctree::
   :maxdepth: 2
   :glob:

   ../examples/frameworks/pytorch/loader_evaluator/index


Overview
~~~~~~~~

The Loader Evaluator Tool is a lightweight, user-friendly utility designed to help users identify
bottlenecks in data loading pipelines. It simulates ideal data loading conditions by caching
a few batches and replaying them, allowing users to compare real vs. ideal performance and pinpoint
inefficiencies without needing deep profiling expertise.

Technical Analysis
~~~~~~~~~~~~~~~~~~

The Loader Evaluator Tool is built around a simple idea: wrap a PyTorch DataLoader to simulate
ideal data loading conditions that don't impact training speed (its overhead is close to 0).
This is achieved by:

- **Logging** performance metrics from real data loading to establish baseline performance
- **Caching** a small number of batches in memory from the original DataLoader (in replay mode)
- **Replaying** those cached batches in a loop to simulate a perfect, bottleneck-free pipeline
- **Collecting** performance metrics such as:

  - Batch load time
  - GPU utilization
  - Throughput (images/sec or videos/sec)

- **Visualizing** the difference between real and ideal runs to highlight bottlenecks

This approach is conceptually similar to how the DALI PyTorch iterator works: it produces an object
that is fully compatible with the PyTorch DataLoader API. This means users can drop it into
their training loop with minimal modifications to their training logic and no modifications
to other parts of the code. The wrapper behaves like a standard DataLoader but adds
caching and performance metrics collection under the hood.

This compatibility ensures:

- Seamless integration with existing PyTorch training pipelines
- Minimal learning curve for users already familiar with PyTorch
- Flexibility to switch between real and ideal (cached) modes with a single line of code

Example Workflow
^^^^^^^^^^^^^^^^

Here's a mockup of how the tool might be used in practice to identify if data loading
is the bottleneck:

.. code-block:: python

   # 1. Logging real performance
   wrapped_loader = LoaderEvaluator(dataloader, mode="log")
   train_one_epoch(model, wrapped_loader, log_file="logged_real_run.json")

   # 2. Simulating ideal performance (automatically caches batches during construction)
   replay_loader = LoaderEvaluator(dataloader, mode="replay", num_cached_batches=20)
   train_one_epoch(model, replay_loader, log_file="logged_ideal_run.json")

   # 3. Visualizing results
   plot_comparison(
       real_log="logged_real_run.json",
       ideal_log="logged_ideal_run.json"
   )

Why not use other tools?
~~~~~~~~~~~~~~~~~~~~~~~~~

There are other tools that can help you identify data loading bottlenecks, but they have their
own limitations.

NSYS Profiler: The Observational Tool
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

NVIDIA NSight Systems (nsys) is a powerful, low-level profiler that provides detailed insights
into system-wide performance, including GPU kernel execution timelines, CPU thread activity,
memory transfers, and synchronization events.

NSYS offers extremely detailed, accurate data and is ideal for advanced users who need deep
diagnostics, but it requires expertise, has a steep learning curve, introduces overhead, and can
be complex and slow for quick debugging. It is not beginner-friendly for quick, iterative debugging.

When analyzing and optimizing deep learning data pipelines, tools like NVIDIA NSight Systems
(nsys) and the Loader Evaluator Tool are not mutually exclusive but rather complementary,
offering different yet valuable approaches to performance diagnostics.

PyTorch Profiler: The In-Framework Observational Tool
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The PyTorch Profiler is an API within PyTorch designed to help users determine the most expensive
operators in their models by measuring time and memory consumption. It allows you to profile
various activities, including PyTorch operators, TorchScript functions, user-defined code labels,
and on-device CUDA or XPU kernels.

The PyTorch Profiler helps you see where time and memory are spent in your model, export
profiling traces (e.g., as .json files for Chrome trace viewer), examine stack traces, and
schedule profiling for specific segments in long training loops.

While simpler to integrate within a PyTorch workflow compared to nsys, PyTorch Profiler still
requires users to interpret detailed tables and potentially complex trace files. It acts as
an observational tool, providing insights into the current workflow's performance without altering it.
