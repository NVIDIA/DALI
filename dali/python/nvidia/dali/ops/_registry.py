# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from nvidia.dali import backend as _b

# Registry of operators names for given backends
_cpu_ops = set({})
_gpu_ops = set({})
_mixed_ops = set({})


def cpu_ops():
    """Get the set of the names of all registered CPU operators"""
    return _cpu_ops


def gpu_ops():
    """Get the set of the names of all registered GPU operators"""
    return _gpu_ops


def mixed_ops():
    """Get the set of the names of all registered Mixed operators"""
    return _mixed_ops


def _all_registered_ops():
    """Return the set of the names of all registered operators"""
    return _cpu_ops.union(_gpu_ops).union(_mixed_ops)


def register_cpu_op(name):
    """Add new CPU op name to the registry."""
    global _cpu_ops
    _cpu_ops = _cpu_ops.union({name})


def register_gpu_op(name):
    """Add new GPU op name to the registry"""
    global _gpu_ops
    _gpu_ops = _gpu_ops.union({name})


def _discover_ops():
    """Query the backend for all registered operator names, update the Python-side registry of
    operator names."""
    global _cpu_ops
    global _gpu_ops
    global _mixed_ops
    _cpu_ops = _cpu_ops.union(set(_b.RegisteredCPUOps()))
    _gpu_ops = _gpu_ops.union(set(_b.RegisteredGPUOps()))
    _mixed_ops = _mixed_ops.union(set(_b.RegisteredMixedOps()))
