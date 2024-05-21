from nvidia.dali import backend as b
import nvidia.dali.ops as ops
import nvidia.dali.ops._python_def_op_utils as _python_def_op_utils
import nvidia.dali.plugin.pytorch
import nvidia.dali.plugin.numba.experimental
import nvidia.dali.plugin.jax as dax
import sys

# Dictionary with modules that can have registered Ops
ops_modules = {
    "nvidia.dali.ops": nvidia.dali.ops,
    "nvidia.dali.plugin.pytorch": nvidia.dali.plugin.pytorch,
    "nvidia.dali.plugin.numba.experimental": nvidia.dali.plugin.numba.experimental,
}

# Some operators might have a different module for the fn wrapper
module_mapping = {
    "nvidia.dali.plugin.pytorch": "nvidia.dali.plugin.pytorch.fn",
    "nvidia.dali.plugin.numba.experimental": "nvidia.dali.plugin.numba.fn.experimental",
}

no_schema_fns = {
    "nvidia.dali.plugin.jax.fn.jax_function": dax.fn._jax_function_impl._jax_function_desc,
}

# Remove ops not available in the fn API
removed_ops = ["Compose"]

cpu_ops = ops.cpu_ops()
gpu_ops = ops.gpu_ops()
mix_ops = ops.mixed_ops()
all_ops = cpu_ops.union(gpu_ops).union(mix_ops)
link_formatter = ":meth:`{op} <{module}.{op}>`"


def to_fn_module(module_name):
    if module_name in module_mapping:
        return module_mapping[module_name]
    else:
        return module_name.replace(".ops", ".fn")


def name_sort(op_name):
    if isinstance(op_name, _python_def_op_utils.PyOpDesc):
        return f"{op_name.module}.{op_name.name.upper()}"
    _, module, name = ops._process_op_name(op_name)
    return ".".join(module + [name.upper()])


def longest_fn_string():
    longest_str = ""
    for op in sorted(all_ops, key=name_sort):
        fn_string = ""
        _, submodule, op_name = ops._process_op_name(op, api="ops")
        fn_full_name = ops._op_name(op, api="fn")
        for module_name, module in ops_modules.items():
            m = module
            for part in submodule:
                m = getattr(m, part, None)
                if m is None:
                    break
            if m is not None and hasattr(m, op_name):
                fn_string = link_formatter.format(
                    op=fn_full_name, module=to_fn_module(module_name)
                )
                if len(fn_string) > len(longest_str):
                    longest_str = fn_string
    return longest_str


op_name_max_len = len(longest_fn_string())
name_bar = op_name_max_len * "="


def fn_to_op_table(out_filename):
    formater = "{:{c}<{op_name_max_len}} {:{c}<{op_name_max_len}}\n"
    doc_table = ""
    doc_table += formater.format("", "", op_name_max_len=op_name_max_len, c="=")
    doc_table += formater.format(
        "Function (fn.*)",
        "Operator Object (ops.*)",
        op_name_max_len=op_name_max_len,
        c=" ",
    )
    doc_table += formater.format("", "", op_name_max_len=op_name_max_len, c="=")
    for op in sorted(all_ops, key=name_sort):
        op_full_name, submodule, op_name = ops._process_op_name(op, api="ops")
        fn_full_name = ops._op_name(op, api="fn")
        schema = b.TryGetSchema(op)
        if schema:
            if schema.IsDocHidden():
                continue
        for module_name, module in ops_modules.items():
            m = module
            for part in submodule:
                m = getattr(m, part, None)
                if m is None:
                    break
            if m is not None and hasattr(m, op_name):
                op_string = link_formatter.format(
                    op=op_full_name, module=module_name
                )
                fn_string = link_formatter.format(
                    op=fn_full_name, module=to_fn_module(module_name)
                )
        if op_name in removed_ops:
            fn_string = "N/A"
        op_doc = formater.format(
            fn_string, op_string, op_name_max_len=op_name_max_len, c=" "
        )
        doc_table += op_doc
    doc_table += formater.format("", "", op_name_max_len=op_name_max_len, c="=")
    with open(out_filename, "w") as f:
        f.write(doc_table)


def operations_table_str(ops_to_process):
    formater = "{:{c}<{op_name_max_len}} {:{c}^48} {:{c}<150}\n"
    doc_table = ""
    doc_table += "\n.. currentmodule:: nvidia.dali.fn\n\n"
    doc_table += formater.format(
        "", "", "", op_name_max_len=op_name_max_len, c="="
    )
    doc_table += formater.format(
        "Function",
        "Device support",
        "Short description",
        op_name_max_len=op_name_max_len,
        c=" ",
    )
    doc_table += formater.format(
        "", "", "", op_name_max_len=op_name_max_len, c="="
    )
    for op in sorted(ops_to_process, key=name_sort):
        if isinstance(op, _python_def_op_utils.PyOpDesc):
            fn_string = link_formatter.format(op=op.name, module=op.module)
            devices_str = ", ".join(op.devices)
            short_descr = op.short_desc
        else:
            _, submodule, op_name = ops._process_op_name(op, api="ops")
            fn_full_name = ops._op_name(op, api="fn")
            if op_name in removed_ops:
                continue
            schema = b.TryGetSchema(op)
            short_descr = ""
            devices = []
            if op in cpu_ops:
                devices += ["CPU"]
            if op in mix_ops:
                devices += ["Mixed"]
            if op in gpu_ops:
                devices += ["GPU"]
            devices_str = ", ".join(devices)
            if schema:
                if schema.IsDocHidden():
                    continue
                full_doc = schema.Dox()
            else:
                full_doc = eval("ops." + op).__doc__
            short_descr = (
                full_doc.split("\n\n")[0].replace("\n", " ").replace("::", ".")
            )
            for module_name, module in ops_modules.items():
                m = module
                for part in submodule:
                    m = getattr(m, part, None)
                    if m is None:
                        break
                if m is not None and hasattr(m, op_name):
                    fn_string = link_formatter.format(
                        op=fn_full_name, module=to_fn_module(module_name)
                    )
        op_doc = formater.format(
            fn_string,
            devices_str,
            short_descr,
            op_name_max_len=op_name_max_len,
            c=" ",
        )
        doc_table += op_doc
    doc_table += formater.format(
        "", "", "", op_name_max_len=op_name_max_len, c="="
    )
    return doc_table


def operations_table(out_filename):
    doc_table = operations_table_str(all_ops)
    with open(out_filename, "w") as f:
        f.write(doc_table)


if __name__ == "__main__":
    assert len(sys.argv) >= 2 and len(sys.argv) <= 3
    operations_table(sys.argv[1])
    if len(sys.argv) == 3:
        fn_to_op_table(sys.argv[2])
