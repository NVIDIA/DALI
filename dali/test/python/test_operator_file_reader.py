import tempfile
import os
import nvidia.dali.fn as fn
from functools import partial
from nvidia.dali.pipeline import Pipeline

def ref_contents(path):
    fname = path[path.rfind('/')+1:]
    return "Contents of " + fname + ".\n"

def populate(root, files):
    for fname in files:
        with open(os.path.join(root, fname), "w") as f:
            f.write(ref_contents(fname))

g_root = None
g_tmpdir = None
g_files = None

def setup_module():
    global g_root
    global g_files
    global g_tmpdir

    g_tmpdir = tempfile.TemporaryDirectory()
    g_root = g_tmpdir.__enter__()
    g_files = [str(i)+' x.dat' for i in range(10)]  # name with a space in the middle!
    populate(g_root, g_files)

def teardown_module():
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
    files, labels = fn.file_reader(file_root=root, files=fnames, labels=lbl, random_shuffle=shuffle)
    pipe.set_outputs(files, labels)
    pipe.build()

    num_iters = (len(fnames) + 2 * batch_size) // batch_size
    for i in range(num_iters):
        out_f, out_l = pipe.run()
        for j in range(batch_size):
            contents = bytes(out_f.at(j)).decode('utf-8')
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
    files, labels = fn.file_reader(files=fnames, random_shuffle=True)
    pipe.set_outputs(files, labels)
    pipe.build()

    num_iters = (len(fnames) + 2 * batch_size) // batch_size
    for i in range(num_iters):
        out_f, out_l = pipe.run()
        for j in range(batch_size):
            contents = bytes(out_f.at(j)).decode('utf-8')
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
    files, labels = fn.file_reader(file_list=list_file, random_shuffle=True)
    pipe.set_outputs(files, labels)
    pipe.build()

    num_iters = (len(fnames) + 2 * batch_size) // batch_size
    for i in range(num_iters):
        out_f, out_l = pipe.run()
        for j in range(batch_size):
            contents = bytes(out_f.at(j)).decode('utf-8')
            label = out_l.at(j)[0]
            index = 10000 - label
            assert contents == ref_contents(fnames[index])
