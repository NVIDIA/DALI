import nvidia.dali.experimental.dynamic as ndd
import re


_graph_regex = re.compile(r".*(^|[^A-Za-z0-9_])[Gg]raph([ .,)]|$).*")


def _check_no_pipeline_mode_wording(s, schema):
    assert "TensorList" not in s, f"TensorList found in docs for {schema.Name()}:\n{s}"
    assert not _graph_regex.match(s), f'"Graph" found in the docs for {schema.Name()}:\n{s}'


def should_skip(x):
    return x._schema.IsDocHidden() or x._schema.IsDocPartiallyHidden() or x._schema.IsInternal()


def test_function_docs_present():
    assert ndd.ops._all_functions  # not empty
    for f in ndd.ops._all_functions:
        if should_skip(f):
            continue
        assert len(f.__doc__) > 20, f._schema.Name()


def test_function_docs_no_tensor_list():
    assert ndd.ops._all_functions  # not empty
    for f in ndd.ops._all_functions:
        if should_skip(f):
            continue
        _check_no_pipeline_mode_wording(f.__doc__, f._schema)


def test_op_docs_present():
    assert ndd.ops._all_ops  # not empty
    for c in ndd.ops._all_ops:
        if should_skip(c):
            continue
        assert len(c.__init__.__doc__) > 20, c._schema.Name()
        assert len(c.__call__.__doc__) > 20, c._schema.Name()


def test_op_docs_no_tensor_list():
    assert ndd.ops._all_ops  # not empty
    for c in ndd.ops._all_ops:
        if should_skip(c):
            continue
        _check_no_pipeline_mode_wording(c.__init__.__doc__, c._schema)
        _check_no_pipeline_mode_wording(c.__call__.__doc__, c._schema)
