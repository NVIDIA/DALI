# Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import glob
import numpy as np
import nvidia.dali.fn as fn
import os
import random
import tempfile
from nvidia.dali import Pipeline, pipeline_def

from nose_utils import assert_raises
from test_utils import compare_pipelines


def ref_contents(path):
    fname = path[path.rfind("/") + 1 :]
    return "Contents of " + fname + ".\n"


def populate(root, files):
    for fname in files:
        with open(os.path.join(root, fname), "w") as f:
            f.write(ref_contents(fname))


g_root = None
g_tmpdir = None
g_files = None


def setUpModule():
    global g_root
    global g_files
    global g_tmpdir

    g_tmpdir = tempfile.TemporaryDirectory()
    g_root = g_tmpdir.__enter__()
    g_files = [str(i) + " x.dat" for i in range(10)]  # name with a space in the middle!
    populate(g_root, g_files)


def tearDownModule():
    global g_root
    global g_files
    global g_tmpdir

    g_tmpdir.__exit__(None, None, None)
    g_tmpdir = None
    g_root = None
    g_files = None


def _test_reader_files_arg(use_root, use_labels, shuffle):
    root = g_root
    fnames = g_files
    if not use_root:
        fnames = [os.path.join(root, f) for f in fnames]
        root = None

    lbl = None
    if use_labels:
        lbl = [10000 + i for i in range(len(fnames))]

    batch_size = 3
    pipe = Pipeline(batch_size, 1, 0)
    files, labels = fn.readers.file(
        file_root=root, files=fnames, labels=lbl, random_shuffle=shuffle
    )
    pipe.set_outputs(files, labels)

    num_iters = (len(fnames) + 2 * batch_size) // batch_size
    for i in range(num_iters):
        out_f, out_l = pipe.run()
        for j in range(batch_size):
            contents = bytes(out_f.at(j)).decode("utf-8")
            label = out_l.at(j)[0]
            index = label - 10000 if use_labels else label
            assert contents == ref_contents(fnames[index])


def test_file_reader():
    for use_root in [False, True]:
        for use_labels in [False, True]:
            for shuffle in [False, True]:
                yield _test_reader_files_arg, use_root, use_labels, shuffle


def test_file_reader_relpath():
    batch_size = 3
    rel_root = os.path.relpath(g_root, os.getcwd())
    fnames = [os.path.join(rel_root, f) for f in g_files]

    pipe = Pipeline(batch_size, 1, 0)
    files, labels = fn.readers.file(files=fnames, random_shuffle=True)
    pipe.set_outputs(files, labels)

    num_iters = (len(fnames) + 2 * batch_size) // batch_size
    for i in range(num_iters):
        out_f, out_l = pipe.run()
        for j in range(batch_size):
            contents = bytes(out_f.at(j)).decode("utf-8")
            index = out_l.at(j)[0]
            assert contents == ref_contents(fnames[index])


def test_file_reader_relpath_file_list():
    batch_size = 3
    fnames = g_files

    list_file = os.path.join(g_root, "list.txt")
    with open(list_file, "w") as f:
        for i, name in enumerate(fnames):
            f.write("{0} {1}\n".format(name, 10000 - i))

    pipe = Pipeline(batch_size, 1, 0)
    files, labels = fn.readers.file(file_list=list_file, random_shuffle=True)
    pipe.set_outputs(files, labels)

    num_iters = (len(fnames) + 2 * batch_size) // batch_size
    for i in range(num_iters):
        out_f, out_l = pipe.run()
        for j in range(batch_size):
            contents = bytes(out_f.at(j)).decode("utf-8")
            label = out_l.at(j)[0]
            index = 10000 - label
            assert contents == ref_contents(fnames[index])


def _test_file_reader_filter(
    filters, glob_filters, batch_size, num_threads, subpath, case_sensitive_filter
):
    pipe = Pipeline(batch_size, num_threads, 0)
    root = os.path.join(os.environ["DALI_EXTRA_PATH"], subpath)
    files, labels = fn.readers.file(
        file_root=root, file_filters=filters, case_sensitive_filter=case_sensitive_filter
    )
    pipe.set_outputs(files, labels)

    fnames = set()
    for label, dir in enumerate(sorted(next(os.walk(root))[1])):
        for filter in glob_filters:
            for file in glob.glob(os.path.join(root, dir, filter)):
                fnames.add((label, file.split("/")[-1], file))

    fnames = sorted(fnames)

    for i in range(len(fnames) // batch_size):
        out_f, _ = pipe.run()
        for j in range(batch_size):
            with open(fnames[i * batch_size + j][2], "rb") as file:
                contents = np.array(list(file.read()))
                assert all(contents == out_f.at(j))


def test_file_reader_filters():
    for filters in [["*.jpg"], ["*.jpg", "*.png", "*.jpeg"], ["dog*.jpg", "cat*.png", "*.jpg"]]:
        num_threads = random.choice([1, 2, 4, 8])
        batch_size = random.choice([1, 3, 10])
        yield (
            _test_file_reader_filter,
            filters,
            filters,
            batch_size,
            num_threads,
            "db/single/mixed",
            False,
        )

    yield _test_file_reader_filter, ["*.jPg", "*.JPg"], [
        "*.jPg",
        "*.JPg",
    ], 3, 1, "db/single/case_sensitive", True
    yield _test_file_reader_filter, ["*.JPG"], [
        "*.jpg",
        "*.jpG",
        "*.jPg",
        "*.jPG",
        "*.Jpg",
        "*.JpG",
        "*.JPg",
        "*.JPG",
    ], 3, 1, "db/single/case_sensitive", False


batch_size_alias_test = 64


@pipeline_def(batch_size=batch_size_alias_test, device_id=0, num_threads=4)
def file_pipe(file_op, file_list):
    files, labels = file_op(file_list=file_list)
    return files, labels


def test_file_reader_alias():
    fnames = g_files

    file_list = os.path.join(g_root, "list.txt")
    with open(file_list, "w") as f:
        for i, name in enumerate(fnames):
            f.write("{0} {1}\n".format(name, 10000 - i))
    new_pipe = file_pipe(fn.readers.file, file_list)
    legacy_pipe = file_pipe(fn.file_reader, file_list)
    compare_pipelines(new_pipe, legacy_pipe, batch_size_alias_test, 50)


def test_invalid_number_of_shards():
    @pipeline_def(batch_size=1, device_id=0, num_threads=4)
    def get_test_pipe():
        root = os.path.join(os.environ["DALI_EXTRA_PATH"], "db/single/mixed")
        files, labels = fn.readers.file(file_root=root, shard_id=0, num_shards=9999)
        return files, labels

    pipe = get_test_pipe()
    assert_raises(
        RuntimeError,
        pipe.build,
        glob=(
            "The number of input samples: *,"
            " needs to be at least equal to the requested number of shards:*."
        ),
    )
