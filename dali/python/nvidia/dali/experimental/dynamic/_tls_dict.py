# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import threading
from collections import UserDict


class TLSDict(UserDict):
    """Thread-local dictionary used for the instance cache"""

    def __init__(self, *args, **kwargs):
        self._local = threading.local()
        super().__init__(*args, **kwargs)

    @property
    def data(self):
        if not hasattr(self._local, "store"):
            self._local.store = {}
        return self._local.store

    @data.setter
    def data(self, value):
        self._local.store = value
