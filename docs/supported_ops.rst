Supported operations
====================
- **CPU** operator means that the operator can be scheduled on the CPU. Outputs of CPU operators may be used as regular inputs and to provide per-sample parameters for other operators through tensor arguments.
- **GPU** operator means that the operator can be scheduled on the GPU. Their outputs may only be used as regular inputs for other GPU operators and pipeline outputs.
- **Mixed** operator means that the operator accepts input on the CPU while producing the output on the GPU.
- **Sequences** means that the operator can work (produce or accept as an input) sequence (video like) kind of input.
- **Volumetric** means that the operator supports 3D data processing.

.. include:: op_inclusion

.. automodule:: nvidia.dali.ops
   :members:

.. autoclass:: nvidia.dali.plugin.pytorch.TorchPythonFunction
   :members:
