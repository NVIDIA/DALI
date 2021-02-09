from nvidia.dali import backend as b
import nvidia.dali.ops as ops
import nvidia.dali.plugin.pytorch
import sys
import inspect

# Dictionary with modules that can have registered Ops
ops_modules = {
    'nvidia.dali.ops': nvidia.dali.ops,
}

exclude_ops_members = {
    'nvidia.dali.ops': ["PythonFunctionBase"]
}

fn_modules = {
    'nvidia.dali.fn': nvidia.dali.fn,
    'nvidia.dali.plugin.pytorch.fn': nvidia.dali.plugin.pytorch.fn
}

exclude_fn_members = {
}

def op_autodoc(out_filename):
    s = ""
    for module in sys.modules.keys():
        for doc_module in ops_modules:
            if module.startswith(doc_module) and not module.endswith('hidden'):
                s += module + "\n"
                s += "~" * len(module) + "\n"
                s += ".. automodule:: {}\n".format(module)
                s += "   :members:\n"
                s += "   :special-members: __call__\n"
                if module in exclude_ops_members:
                    excluded = exclude_ops_members[module]
                    s += "   :exclude-members: {}\n".format(", ".join(excluded))
                s += "\n"
    with open(out_filename, 'w') as f:
        f.write(s)

def fn_autodoc(out_filename):
    s = ""
    for module in sys.modules.keys():
        for doc_module in fn_modules:
            if module.startswith(doc_module) and not module.endswith('hidden'):
                s += module + "\n"
                s += "~" * len(module) + "\n"
                s += ".. automodule:: {}\n".format(module)
                s += "   :members:\n"
                s += "   :undoc-members:\n"
                if module in exclude_fn_members:
                    excluded = exclude_fn_members[module]
                    s += "   :exclude-members: {}\n".format(", ".join(excluded))
                s += "\n"
    with open(out_filename, 'w') as f:
        f.write(s)

if __name__ == "__main__":
    assert(len(sys.argv) == 3)
    op_autodoc(sys.argv[1])
    fn_autodoc(sys.argv[2])
