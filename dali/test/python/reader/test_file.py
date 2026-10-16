# Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import nvidia.dali.types as types
from nvidia.dali import Pipeline, pipeline_def

from nose2.tools import params
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


@pipeline_def(num_threads=1, device_id=0)
def _file_reader_pipe(reader_args):
    files, labels = fn.readers.file(**reader_args)
    return files, labels


def _label_pipe(batch_size=3, **reader_args):
    return _file_reader_pipe(reader_args, batch_size=batch_size)


def _check_file_labels(pipe, fnames, expected_labels, dtype):
    batch_size = pipe.max_batch_size
    num_iters = (len(fnames) + 2 * batch_size) // batch_size
    label_of = {ref_contents(f): lbl for f, lbl in zip(fnames, expected_labels, strict=True)}
    for _ in range(num_iters):
        out_f, out_l = pipe.run()
        for j in range(batch_size):
            contents = bytes(out_f.at(j)).decode("utf-8")
            label = out_l.at(j)
            assert label.dtype == dtype, f"{label.dtype} != {dtype}"
            assert label.shape == (1,)
            assert label[0] == dtype(label_of[contents]), f"{label[0]} != {label_of[contents]}"


def _write_file_list(fnames, labels):
    list_file = os.path.join(g_root, "list_label_dtype.txt")
    with open(list_file, "w") as f:
        for name, label in zip(fnames, labels, strict=True):
            f.write(f"{name} {label}\n")
    return list_file


@params(1, 3)
def test_file_list_float_labels(batch_size):
    fnames = g_files
    labels = [0.375 * i - 1 for i in range(len(fnames))]
    list_file = _write_file_list(fnames, labels)
    pipe = _label_pipe(
        batch_size=batch_size, file_list=list_file, label_dtype=types.FLOAT, random_shuffle=True
    )
    _check_file_labels(pipe, fnames, labels, np.float32)


def test_file_list_float_labels_exponent():
    fnames = g_files
    labels = [f"{i}e-3" for i in range(len(fnames))]
    list_file = _write_file_list(fnames, labels)
    pipe = _label_pipe(file_list=list_file, label_dtype=types.FLOAT)
    _check_file_labels(pipe, fnames, [float(x) for x in labels], np.float32)


@params(
    (types.INT32, np.int32),
    (types.INT64, np.int64),
    (types.FLOAT, np.float32),
)
def test_file_list_boundary_labels(label_dtype, np_dtype):
    fnames = g_files[:4]
    if np_dtype == np.float32:
        info = np.finfo(np_dtype)
        labels = [repr(float(x)) for x in (info.max, -info.max, info.tiny)] + ["-0.0"]
    else:
        info = np.iinfo(np_dtype)
        labels = [info.max, info.min, 0, -1]
    list_file = _write_file_list(fnames, labels)
    pipe = _label_pipe(batch_size=1, file_list=list_file, label_dtype=label_dtype)
    _check_file_labels(pipe, fnames, [np_dtype(x) for x in labels], np_dtype)


def test_file_list_int64_labels():
    fnames = g_files
    labels = [(1 << 40) + i for i in range(len(fnames))]
    list_file = _write_file_list(fnames, labels)
    pipe = _label_pipe(file_list=list_file, label_dtype=types.INT64, random_shuffle=True)
    _check_file_labels(pipe, fnames, labels, np.int64)


def test_file_list_default_label_dtype():
    fnames = g_files
    labels = [10000 - i for i in range(len(fnames))]
    list_file = _write_file_list(fnames, labels)
    pipe = _label_pipe(file_list=list_file)
    _check_file_labels(pipe, fnames, labels, np.int32)


@params(
    (1, None),
    (3, None),
    (3, types.FLOAT),
)
def test_files_float_labels(batch_size, label_dtype):
    fnames = g_files
    labels = [0.5 + i for i in range(len(fnames))]
    pipe = _label_pipe(
        batch_size=batch_size,
        file_root=g_root,
        files=fnames,
        float_labels=labels,
        label_dtype=label_dtype,
        random_shuffle=True,
    )
    _check_file_labels(pipe, fnames, labels, np.float32)


@params(
    (types.INT32, np.int32, [(1 << 31) - 1, -(1 << 31)]),
    (types.INT64, np.int64, [(1 << 63) - 1, -(1 << 63), 1 << 40]),
    (types.FLOAT, np.float32, [10000, -1]),
)
def test_files_labels_label_dtype(label_dtype, np_dtype, boundary):
    fnames = g_files
    labels = boundary + list(range(len(fnames) - len(boundary)))
    pipe = _label_pipe(
        batch_size=1, file_root=g_root, files=fnames, labels=labels, label_dtype=label_dtype
    )
    _check_file_labels(pipe, fnames, labels, np_dtype)


def test_files_index_labels_float():
    fnames = g_files
    pipe = _label_pipe(file_root=g_root, files=fnames, label_dtype=types.FLOAT)
    _check_file_labels(pipe, fnames, list(range(len(fnames))), np.float32)


def test_file_root_label_dtype():
    batch_size = 4
    root = os.path.join(os.environ["DALI_EXTRA_PATH"], "db/single/mixed")
    ref_pipe = _label_pipe(batch_size=batch_size, file_root=root)
    pipe = _label_pipe(batch_size=batch_size, file_root=root, label_dtype=types.FLOAT)
    for _ in range(3):
        (ref_f, ref_l), (out_f, out_l) = ref_pipe.run(), pipe.run()
        for j in range(batch_size):
            assert np.array_equal(ref_f.at(j), out_f.at(j))
            assert out_l.at(j).dtype == np.float32
            assert out_l.at(j)[0] == ref_l.at(j)[0]


@params(
    (0.375, None),
    (0.375, types.INT32),
    (0.375, types.INT64),
    ("abc", types.FLOAT),
    ("1.5x", types.FLOAT),
    ("1e-100", types.FLOAT),
    ("1e100", types.FLOAT),
    (1 << 31, types.INT32),
    (-(1 << 31) - 1, types.INT32),
    (1 << 63, types.INT64),
)
def test_file_list_malformed_label(label, label_dtype):
    fnames = g_files[:3]
    list_file = _write_file_list(fnames, [0, label, 1])
    pipe = _label_pipe(file_list=list_file, label_dtype=label_dtype)
    with assert_raises(ValueError, glob=f'Incorrect label in the list file*:2*got: "{label}".'):
        pipe.build()


@params(
    (dict(labels=[1 << 31]), types.INT32, "*Label 2147483648 is out of range*"),
    (dict(float_labels=[0.5]), types.INT32, "*``float_labels`` requires a floating point*"),
    (dict(float_labels=[0.5], labels=[0]), None, "*``labels`` and ``float_labels`` are mutually*"),
    (dict(float_labels=[0.5, 0.5]), None, "*Provided 2 float labels for 1 files.*"),
    (dict(), types.UINT8, "*Unsupported ``label_dtype``*"),
)
def test_file_labels_invalid_args(args, label_dtype, pattern):
    pipe = _label_pipe(file_root=g_root, files=g_files[:1], label_dtype=label_dtype, **args)
    with assert_raises(ValueError, glob=pattern):
        pipe.build()


def test_file_list_float_labels_requires_files():
    pipe = _label_pipe(file_root=g_root, float_labels=[0.5])
    with assert_raises(ValueError, glob="*``float_labels`` is valid only when*"):
        pipe.build()
