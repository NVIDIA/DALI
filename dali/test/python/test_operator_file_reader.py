import tempfile
import os
import nvidia.dali.fn as fn
from functools import partial
from nvidia.dali.pipeline import Pipeline

def ref_contents(path):
    fname = path[path.rfind('/')+1:]
    return "Contents of " + fname + ".\n";l

def populate(root, files):
    for fname in files:
        with open(os.path.join(root, fname), "w") as f:
            f.write(ref_contents(fname));

def _test_env(func):
    with tempfile.TemporaryDirectory() as root:
        files = [str(i)+'.dat' for i in range(10)]
        populate(root, files)
        try:
            func(root, files)
        finally:
            for f in files:
                try:
                    os.remove(os.path.join(root, f))
                except:
                    pass

def _test_file_reader_body(root, fnames, *, use_root, use_labels, shuffle):
    if not use_root:
        fnames = [os.path.join(root, f) for f in fnames]
        root = None

    lbl = None
    if use_labels:
        lbl = [10000 + i for i in range(len(fnames))]

    batch_size = 3;
    pipe = Pipeline(batch_size, 1, 0)
    files, labels = fn.file_reader(file_root=root, files=fnames, labels=lbl, random_shuffle=shuffle)
    pipe.set_outputs(files, labels)
    pipe.build()

    num_iters = (len(fnames) + 2 * batch_size) // batch_size;
    for i in range(num_iters):
        out_f, out_l = pipe.run()
        for j in range(batch_size):
            contents = bytes(out_f.at(j)).decode('utf-8')
            label = out_l.at(j)[0]
            index = label - 10000 if use_labels else label
            assert(contents == ref_contents(fnames[index]))

def _test_file_reader(use_root, use_labels, shuffle):
    _test_env(partial(_test_file_reader_body, use_root=use_root, use_labels=use_labels, shuffle=shuffle))

def test_file_reader():
    for use_root in [False, True]:
        for use_labels in [False, True]:
            for shuffle in [False, True]:
                yield _test_file_reader, use_root, use_labels, shuffle
