import nvidia.dali.experimental.dynamic as ndd


def _check_no_tensor_list(string, schema):
    assert "TensorList" not in string, f"TensorList found in docs for {schema.Name()}:\n{string}"


def should_skip(x):
    return x.schema.IsDocHidden() or x.schema.IsDocPartiallyHidden() or x.schema.IsInternal()


def test_function_docs_present():
    assert ndd.ops._all_functions  # not empty
    for f in ndd.ops._all_functions:
        if should_skip(f):
            continue
        assert len(f.__doc__) > 20, f.schema.Name()


def test_function_docs_no_tensor_list():
    assert ndd.ops._all_functions  # not empty
    for f in ndd.ops._all_functions:
        if should_skip(f):
            continue
        _check_no_tensor_list(f.__doc__, f.schema)


def test_op_docs_present():
    assert ndd.ops._all_ops  # not empty
    for c in ndd.ops._all_ops:
        if should_skip(c):
            continue
        assert len(c.__init__.__doc__) > 20, c.schema.Name()
        assert len(c.__call__.__doc__) > 20, c.schema.Name()


def test_op_docs_no_tensor_list():
    assert ndd.ops._all_ops  # not empty
    for c in ndd.ops._all_ops:
        if should_skip(c):
            continue
        _check_no_tensor_list(c.__init__.__doc__, c.schema)
        _check_no_tensor_list(c.__call__.__doc__, c.schema)
