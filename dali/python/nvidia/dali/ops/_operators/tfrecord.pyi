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

from typing import Union, Optional
from typing import Sequence, Any, Callable, Mapping

from nvidia.dali.data_node import DataNode
from nvidia.dali.types import DALIDataType, DALIImageType, DALIInterpType

from nvidia.dali.tfrecord import Feature as Feature

class TFRecordReader:
    """
    .. warning::

       This operator is now deprecated. Use :meth:`readers.TFRecord` instead.

       In DALI 1.0 all readers were moved into a dedicated :mod:`~nvidia.dali.fn.readers`
       submodule and renamed to follow a common pattern. This is a placeholder operator with identical
       functionality to allow for backward compatibility.

    Legacy alias for :meth:`readers.tfrecord`.
    """

    def __init__(
        self,
        /,
        *,
        features: Optional[Mapping[str, Feature]] = None,
        index_path: Union[Sequence[str], str, None] = None,
        path: Union[Sequence[str], str, None] = None,
        bytes_per_sample_hint: Union[Sequence[int], int, None] = [0],
        dont_use_mmap: Optional[bool] = False,
        initial_fill: Optional[int] = 1024,
        lazy_init: Optional[bool] = False,
        num_shards: Optional[int] = 1,
        pad_last_batch: Optional[bool] = False,
        prefetch_queue_depth: Optional[int] = 1,
        preserve: Optional[bool] = False,
        random_shuffle: Optional[bool] = False,
        read_ahead: Optional[bool] = False,
        seed: Optional[int] = -1,
        shard_id: Optional[int] = 0,
        skip_cached_images: Optional[bool] = False,
        stick_to_shard: Optional[bool] = False,
        tensor_init_bytes: Optional[int] = 1048576,
        use_o_direct: Optional[bool] = False,
        device: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None: ...
    def __call__(
        self,
        /,
        *,
        features: Optional[Mapping[str, Feature]] = None,
        index_path: Union[Sequence[str], str, None] = None,
        path: Union[Sequence[str], str, None] = None,
        bytes_per_sample_hint: Union[Sequence[int], int, None] = [0],
        dont_use_mmap: Optional[bool] = False,
        initial_fill: Optional[int] = 1024,
        lazy_init: Optional[bool] = False,
        num_shards: Optional[int] = 1,
        pad_last_batch: Optional[bool] = False,
        prefetch_queue_depth: Optional[int] = 1,
        preserve: Optional[bool] = False,
        random_shuffle: Optional[bool] = False,
        read_ahead: Optional[bool] = False,
        seed: Optional[int] = -1,
        shard_id: Optional[int] = 0,
        skip_cached_images: Optional[bool] = False,
        stick_to_shard: Optional[bool] = False,
        tensor_init_bytes: Optional[int] = 1048576,
        use_o_direct: Optional[bool] = False,
        device: Optional[str] = None,
        name: Optional[str] = None,
    ) -> Union[DataNode, Sequence[DataNode], None]:
        """__call__(**kwargs)

        Operator call to be used in graph definition. This operator doesn't have any inputs.

        """
        ...

class TFRecord:
    """Reads samples from a TensorFlow TFRecord file."""

    def __init__(
        self,
        /,
        *,
        features: Optional[Mapping[str, Feature]] = None,
        index_path: Union[Sequence[str], str, None] = None,
        path: Union[Sequence[str], str, None] = None,
        bytes_per_sample_hint: Union[Sequence[int], int, None] = [0],
        dont_use_mmap: Optional[bool] = False,
        initial_fill: Optional[int] = 1024,
        lazy_init: Optional[bool] = False,
        num_shards: Optional[int] = 1,
        pad_last_batch: Optional[bool] = False,
        prefetch_queue_depth: Optional[int] = 1,
        preserve: Optional[bool] = False,
        random_shuffle: Optional[bool] = False,
        read_ahead: Optional[bool] = False,
        seed: Optional[int] = -1,
        shard_id: Optional[int] = 0,
        skip_cached_images: Optional[bool] = False,
        stick_to_shard: Optional[bool] = False,
        tensor_init_bytes: Optional[int] = 1048576,
        use_o_direct: Optional[bool] = False,
        device: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None: ...
    def __call__(
        self,
        /,
        *,
        features: Optional[Mapping[str, Feature]] = None,
        index_path: Union[Sequence[str], str, None] = None,
        path: Union[Sequence[str], str, None] = None,
        bytes_per_sample_hint: Union[Sequence[int], int, None] = [0],
        dont_use_mmap: Optional[bool] = False,
        initial_fill: Optional[int] = 1024,
        lazy_init: Optional[bool] = False,
        num_shards: Optional[int] = 1,
        pad_last_batch: Optional[bool] = False,
        prefetch_queue_depth: Optional[int] = 1,
        preserve: Optional[bool] = False,
        random_shuffle: Optional[bool] = False,
        read_ahead: Optional[bool] = False,
        seed: Optional[int] = -1,
        shard_id: Optional[int] = 0,
        skip_cached_images: Optional[bool] = False,
        stick_to_shard: Optional[bool] = False,
        tensor_init_bytes: Optional[int] = 1048576,
        use_o_direct: Optional[bool] = False,
        device: Optional[str] = None,
        name: Optional[str] = None,
    ) -> Union[DataNode, Sequence[DataNode], None]:
        """__call__(**kwargs)

        Operator call to be used in graph definition. This operator doesn't have any inputs.

        """
        ...

def tfrecord_reader(
    *,
    features: Mapping[str, Feature] = None,
    index_path: Union[Sequence[str], str],
    path: Union[Sequence[str], str],
    bytes_per_sample_hint: Union[Sequence[int], int, None] = [0],
    dont_use_mmap: Optional[bool] = False,
    initial_fill: Optional[int] = 1024,
    lazy_init: Optional[bool] = False,
    num_shards: Optional[int] = 1,
    pad_last_batch: Optional[bool] = False,
    prefetch_queue_depth: Optional[int] = 1,
    preserve: Optional[bool] = False,
    random_shuffle: Optional[bool] = False,
    read_ahead: Optional[bool] = False,
    seed: Optional[int] = -1,
    shard_id: Optional[int] = 0,
    skip_cached_images: Optional[bool] = False,
    stick_to_shard: Optional[bool] = False,
    tensor_init_bytes: Optional[int] = 1048576,
    use_o_direct: Optional[bool] = False,
    device: Optional[str] = None,
    name: Optional[str] = None,
) -> Union[DataNode, Sequence[DataNode], None]:
    """
    .. warning::

       This operator is now deprecated. Use :meth:`readers.TFRecord` instead.

       In DALI 1.0 all readers were moved into a dedicated :mod:`~nvidia.dali.fn.readers`
       submodule and renamed to follow a common pattern. This is a placeholder operator with identical
       functionality to allow for backward compatibility.

    Legacy alias for :meth:`readers.tfrecord`.
    """
    ...

def tfrecord(
    *,
    features: Mapping[str, Feature] = None,
    index_path: Union[Sequence[str], str],
    path: Union[Sequence[str], str],
    bytes_per_sample_hint: Union[Sequence[int], int, None] = [0],
    dont_use_mmap: Optional[bool] = False,
    initial_fill: Optional[int] = 1024,
    lazy_init: Optional[bool] = False,
    num_shards: Optional[int] = 1,
    pad_last_batch: Optional[bool] = False,
    prefetch_queue_depth: Optional[int] = 1,
    preserve: Optional[bool] = False,
    random_shuffle: Optional[bool] = False,
    read_ahead: Optional[bool] = False,
    seed: Optional[int] = -1,
    shard_id: Optional[int] = 0,
    skip_cached_images: Optional[bool] = False,
    stick_to_shard: Optional[bool] = False,
    tensor_init_bytes: Optional[int] = 1048576,
    use_o_direct: Optional[bool] = False,
    device: Optional[str] = None,
    name: Optional[str] = None,
) -> Union[DataNode, Sequence[DataNode], None]:
    """Reads samples from a TensorFlow TFRecord file."""
    ...
