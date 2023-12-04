doc(
    title="Custom Operations",
    underline_char="=",
    entries=[
        "custom_operator/create_a_custom_operator.ipynb",
        doc_entry(
            "python_operator.ipynb",
            [
                op_reference(
                    "fn.python_function",
                    "Running custom Python code with the family of python_function operators",
                ),
                op_reference(
                    "plugin.pytorch.fn.torch_python_function",
                    "Running custom Python code with the family of python_function operators",
                ),
                op_reference(
                    "fn.dl_tensor_python_function",
                    "Running custom Python code with the family of python_function operators",
                ),
            ],
        ),
        doc_entry(
            "gpu_python_operator.ipynb",
            [
                op_reference("fn.python_function", "Processing GPU Data with Python Operators"),
                op_reference(
                    "plugin.pytorch.fn.torch_python_function",
                    "Processing GPU Data with Python Operators",
                ),
                op_reference(
                    "fn.dl_tensor_python_function", "Processing GPU Data with Python Operators"
                ),
            ],
        ),
        doc_entry(
            "numba_function.ipynb",
            op_reference(
                "plugin.numba.fn.experimental.numba_function",
                "Running custom operations written as Numba JIT-compiled functions",
            ),
        ),
    ],
)
