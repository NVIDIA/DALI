Supported operations
====================
- **CPU** operator means that the operator can be scheduled on the CPU.
- **GPU** operator means that the operator can be scheduled on the GPU.
- **Mixed** operator means that the operator accepts input on the CPU while producing the output on the GPU.
- **Support** is a special type of operator that provides data driving other operators (like a random generator). Its output cannot be used as a DALI output.
- **Sequences** are an operator that can work (produce or accept as an input) sequence (video like) kind of input.

.. include:: op_inclusion

.. automodule:: nvidia.dali.ops
   :members:

.. autoclass:: nvidia.dali.plugin.pytorch.TorchPythonFunction
   :members:
