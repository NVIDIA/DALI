Torchvision API
===============

Overview
--------
DALI Torchvision API uses well established interface to build efficient CPU and GPU pipelines or to execute singe operators. It is based on the v2 API version of Torchvision documented here: [TODO: instert torvhsion API]. The implementation covers object oriented and functional API.

The main goal of the API is to bring well known interface to DALI and to replace Torchvision  with minimal (possibly one line) change in an existing code, in cases where performance mattes.


Supported operators
---------


Limitations
---------
1. Input types are currently supported by DALI Torchvision API:
- PILImages 
- torch.Tensors
2. The object oriented operators need to be encompassed with Compose operator and cannot be used as standalone.
3. There is no guarantee that the operator will be bit compatible with Torchvision implementation, off by 1 is the most common difference.

.. toctree::
   :maxdepth: 1

   How to use DALI Torchvision API <torchvision_api>
