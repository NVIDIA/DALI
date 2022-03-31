from nvidia.dali import backend as b
import nvidia.dali.ops as ops
import nvidia.dali.plugin.pytorch
import nvidia.dali.plugin.numba
import inspect
import sys

from inspect import getmembers, isfunction

# Dictionary with modules that can have registered Ops
ops_modules = {
    'nvidia.dali.ops': nvidia.dali.ops,
    'nvidia.dali.plugin.numba.experimental': nvidia.dali.plugin.numba.experimental,
}

exclude_ops_members = {
    'nvidia.dali.ops': ["PythonFunctionBase"]
}

fn_modules = {
    'nvidia.dali.fn': nvidia.dali.fn,
    'nvidia.dali.plugin.pytorch.fn': nvidia.dali.plugin.pytorch.fn,
    'nvidia.dali.plugin.numba.fn.experimental': nvidia.dali.plugin.numba.fn.experimental,
}

exclude_fn_members = {
}

mod_aditional_doc = {
    'nvidia.dali.fn.transforms' : "All operators in this module support only CPU device as they are meant " +
"to be provided as an input to named keyword operator arguments. Check for more details the relevant " +
":ref:`pipeline documentation section<Processing Graph Structure>`."
}

def get_modules(top_modules):
    modules = []
    for module in sys.modules.keys():
        for doc_module in top_modules:
            if module.startswith(doc_module) and not module.endswith('hidden'):
                modules += [module]
    return sorted(modules)


def get_functions(module):
    """Get all function names (so DALI API operators) from given DALI module without private
    or hidden members. No nested modules would be reported."""
    result = []
    # Take all public members of given module
    public_members = list(filter(lambda x: not str(x).startswith("_"), dir(module)))
    for member_name in public_members:
        member = getattr(module, member_name)
        # Just user-defined functions
        if inspect.isfunction(member) and not member.__module__.endswith("hidden"):
            result.append(member_name)
    return result

def op_autodoc(out_filename):
    s = ""
    for module in get_modules(ops_modules):
        s += module + "\n"
        s += "~" * len(module) + "\n"
        normalize_mod = module.replace("nvidia.dali.ops", "nvidia.dali.fn")
        if normalize_mod in mod_aditional_doc:
            s += mod_aditional_doc[normalize_mod] + "\n" + "\n"
        s += ".. automodule:: {}\n".format(module)
        s += "   :members:\n"
        s += "   :special-members: __call__\n"
        if module in exclude_ops_members:
            excluded = exclude_ops_members[module]
            s += "   :exclude-members: {}\n".format(", ".join(excluded))
        s += "\n\n"
    with open(out_filename, 'w') as f:
        f.write(s)

def get_module_references(module_name, references):
    """Generate section with references for given module"""
    module_name = module_name[12:] # remove nvidia.dali prefix
    result = ""
    if module_name in references:
        result += ".. seealso::\n"
        for desc, url in references[module_name]:
            result += f"   * `{desc} </{url}>`_\n"
    return result

def get_operator_references(op_name, references):
    """Generate section with references for given operator"""
    op_name = op_name[12:] # remove nvidia.dali prefix
    result = ""
    if op_name in references:
        result += ".. seealso::\n"
        for desc, url in references[op_name]:
            result += f"   * `{desc} </{url}>`_\n"
    return result


def fn_autodoc(out_filename, references):
    s = ""
    all_modules = get_modules(fn_modules)
    for module in all_modules:
        dali_module = sys.modules[module]
        # Take all public members of given module
        funs_in_module = get_functions(dali_module)
        s += module + "\n"
        s += "~" * len(module) + "\n"
        if module in mod_aditional_doc:
            s += mod_aditional_doc[module] + "\n" + "\n"
        s += get_module_references(module, references)
        for fun in funs_in_module:
            full_name = f"{module}.{fun}"
            if module in exclude_fn_members and fun in exclude_fn_members[module]:
                continue
            s += f".. autofunction:: {full_name}\n\n"
            s += get_operator_references(full_name, references)
        s += "\n"
    with open(out_filename, 'w') as f:
        f.write(s)

if __name__ == "__main__":
    assert(len(sys.argv) == 3)
    op_autodoc(sys.argv[1])
    fn_autodoc(sys.argv[2])
