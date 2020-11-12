# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

class SharedMem:
    """SharedMem allows you to allocate and access shared memory.
Provides memory view of the allocated memory via buf property.
You can transfer access to the same shared memory chunk by sending related
file descriptor available as fd property. Use SharedMem.allocate
to allocate new chunk of shared memory and SharedMem.open if
you received file descriptor to already existing memory chunk.

There is out of the box support for shared memory starting from Python3.8, though
the only way there to transfer the memory to other processes is via filename,
which might 'leak' if process was closed abruptly.

Parameters
----------
`fd` : int
    File descriptor identifying related shared memory object. Pass -1 to allocate new memory chunk.
`size` : int
    When fd=-1 it is the size of shared memory to allocate in bytes, otherwise it must be
    the size of shared memory objects that provided fd represents.
"""

    def __init__(self, fd, size):
        self.shm = _b.SharedMem(fd, size)

    def __getattr__(self, key):
        # lazily evaluate and cache 'buf' property, so that it is created only once and only when requested
        if key == 'buf':
            buf = self.shm.buf()
            self.__dict__['buf'] = buf
            return buf
        raise AttributeError

    @classmethod
    def allocate(cls, size):
        """Creates new SharedMem instance representing freshly allocated
        shared memory of ``size`` bytes.

        Parameters
        ----------
        `size` : int
            Number of bytes to allocate.
        """
        return cls(-1, size)

    @classmethod
    def open(cls, fd, size):
        """Creates new SharedMem instance that points to already allocated shared
        memory chunk accessible via provided file descriptor ``fd``.

        Parameters
        ----------
        `fd`: int
            File descriptor pointing to already existing shared memory chunk.
        `size` : int
            Size of the existing shared memory chunk.
        """
        return cls(fd, size)

    @property
    def fd(self):
        """File descriptor, use it to transfer access to the shared memory object to another process.
        You can transfer it between processes via socket using multiprocessing.reduction.sendfds
        """
        return self.shm.fd

    def resize(self, size, trunc=False):
        """Resize already allocated shared memory chunk. If you want to resize the underlying
        shared memory chunk pass trunc=True, if the memory chunk has already been resized
        via another SharedMem instance (possibly in another process), pass new size and
        trunc=False to simply adjust mmaping of the memory into the current process address space.
        """
        if 'buf' in self.__dict__:
            del self.__dict__['buf']
        self.shm.resize(size, trunc)

    def close_fd(self):
        """Close file descriptor identifying memory chunk. You can access the memory
        via buf property after closing fd, but you neither can transfer the access via fd
        to another process nor you can resize the chunk anymore.
        """
        self.shm.close_fd()

    def close(self):
        """Removes maping of the memory into process address space and closes related file descriptor.
        If all processes sharing given chunk close it, it will be automatically released by the OS.
        You don't have to call this method, as corresponding clean up is performed when instance
        gets garbage collected but you can call it as soon as you no longer need it for more
        effective resources handling.
        """
        self.buf = None
        self.shm.close_map()
        self.shm.close_fd()
