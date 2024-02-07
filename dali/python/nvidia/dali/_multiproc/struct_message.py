# Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


import struct


class Structure:
    """
    Utility around Python `struct` module (https://docs.python.org/3.6/library/struct.html)
     that allows to access and modify `_fields` like an ordinary object attributes
     and read/write their values from/into the buffer in C struct like format.
    Similar approach of declaring _fields_ with corresponding C types can be found in
    Python `ctypes` module (https://docs.python.org/3/library/ctypes.html).
    """

    # A tuple of (name, type) pairs, where type is a string encoding of a simple type,
    # as used in struct
    _fields = tuple()

    def __init__(self, *values):
        self.setup_struct()
        self.set_values(*values)

    @classmethod
    def setup_struct(cls):
        if "_struct_desc" not in cls.__dict__:
            cls._struct_desc = "@" + "".join(field_type for _, field_type in cls._fields)
            cls._struct = struct.Struct(cls._struct_desc)

    def __getstate__(self):
        return self.__dict__.copy()

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.setup_struct()

    def set_values(self, *values):
        for (field_name, _), value in zip(self._fields, values):
            setattr(self, field_name, value)

    def get_values(self):
        return tuple(getattr(self, field_name) for field_name, _ in self._fields)

    def pack_into(self, buf, offset):
        try:
            values = self.get_values()
            return self._struct.pack_into(buf, offset, *values)
        except struct.error as e:
            raise RuntimeError(
                "Failed to serialize object as C-like structure. "
                "Tried to populate following fields: `{}` with respective values: `{}` ".format(
                    self._fields, self.get_values()
                )
            ) from e

    def unpack_from(self, buf, offset):
        values = self._struct.unpack_from(buf, offset)
        self.set_values(*values)
        return self

    def get_size(self):
        return self._struct.size
