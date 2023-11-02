# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
# Copyright (c) 2017-2019, NVIDIA CORPORATION. All rights reserved.
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

import errno
import os
import shutil
import sys
import tarfile
import tempfile

try:
    from urllib.request import urlopen
    from urllib.parse import urlparse
except Exception:  # python 2
    from urllib2 import urlopen
    from urlparse import urlparse

import paddle

def _extract_tar(filename, dest):
    print("extracting to {}".format(dest))
    if not os.path.exists(dest):
        try:
            os.makedirs(dest)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
    f = tarfile.open(filename)
    f.extractall(dest)


def _download_weight(url):
    weight_dir = os.path.expanduser("~/.cache/paddle/weights")
    os.makedirs(weight_dir, exist_ok=True)
    filename = os.path.basename(urlparse(url).path)
    base, ext = os.path.splitext(filename)
    assert ext in ['.tar', '.pdparams'], "Unsupported weight format"
    if ext == '.tar':
        dest = os.path.join(weight_dir, base)
        if os.path.exists(dest):
            assert os.path.isdir(dest), "weight path is not a directory"
            return dest
    else:
        dest = os.path.join(weight_dir, filename)
        if os.path.isfile(dest):
            return dest

    print("downloading {}".format(url))

    req = urlopen(url)
    total = float(req.headers['content-length'])
    tmp = tempfile.NamedTemporaryFile(delete=False)
    downloaded = 0

    try:
        while True:
            buffer = req.read(8192)
            if len(buffer) == 0:
                break
            tmp.write(buffer)
            downloaded += len(buffer)
            sys.stdout.write("\r{0:.1f}%".format(100 * downloaded / total))
            sys.stdout.flush()

        sys.stdout.write('\n')
        tmp.close()
        if ext == '.tar':
            _extract_tar(tmp.name, weight_dir)
        else:
            shutil.move(tmp.name, dest)
    finally:
        tmp.close()
        if os.path.exists(tmp.name):
            os.remove(tmp.name)

    return dest


def load_weights(exe, prog, url):
    weight_path = _download_weight(url)

    if os.path.isdir(weight_path):
        paddle.static.io.load_vars(
            exe, weight_path, prog,
            predicate=lambda v: os.path.exists(
                os.path.join(weight_path, v.name)))
    else:
        paddle.distributed.io.load_persistables(exe, '', prog, filename=weight_path)
