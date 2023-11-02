from nvidia.dali import backend as b
import nvidia.dali.ops as ops
import nvidia.dali.plugin.pytorch
import nvidia.dali.plugin.numba
import inspect
import sys
from pathlib import Path

import operations_table

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


def _is_private(module):
    submodules = module.split(".")
    return any([submodule.startswith("_") for submodule in submodules])


def get_modules(top_modules):
    modules = []
    for module in sys.modules.keys():
        for doc_module in top_modules:
            if (module.startswith(doc_module) and not module.endswith('hidden')
                    and not _is_private(module)):
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

def get_schema_names(module, functions):
    return [getattr(sys.modules[module], fun)._schema_name for fun in functions]


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


def get_references(name, references):
    """Generate section with references for given operator or module"""
    name = name[12:]  # remove nvidia.dali prefix
    result = ""
    if name in references:
        result += ".. seealso::\n"
        for desc, url in references[name]:
            result += f"   * `{desc} <../{url}>`_\n"
    return result

def single_fun_file(full_name, references):
    """Generate stub page for documentation of given function from fn api.
    """
    result = f"{full_name}\n"
    result += "-" * len(full_name) + "\n\n"
    result += f".. autofunction:: {full_name}\n\n"
    result += get_references(full_name, references)
    return result

def single_module_file(module, funs_in_module, references):
    """Generate stub page for documentation of given module
    """
    result = f"{module}\n"
    result += "~" * len(module) + "\n\n"

    if module in mod_aditional_doc:
        result += mod_aditional_doc[module] + "\n\n"
    result += get_references(module, references)
    result += "\n"

    result += f"The following table lists all operations available in ``{module}`` module:\n"
    result += operations_table.operations_table_str(get_schema_names(module, funs_in_module))
    result += "\n\n"


    result += ".. toctree::\n   :hidden:\n\n"

    for fun in funs_in_module:
        if module in exclude_fn_members and fun in exclude_fn_members[module]:
            continue
        full_name = f"{module}.{fun}"
        result += f"   {full_name}\n"
    return result

def fn_autodoc(out_filename, generated_path, references):
    all_modules_str = ".. toctree::\n   :hidden:\n\n"
    all_modules = get_modules(fn_modules)
    for module in all_modules:
        dali_module = sys.modules[module]
        # Take all public members of given module
        funs_in_module = get_functions(dali_module)
        if len(funs_in_module) == 0:
            continue

        # As the top-level file is included from a directory above generated_path
        # we need to provide the relative path to the per-module files
        # the rest is within the same directory, so there is no need for that
        all_modules_str += f"   {generated_path / module}\n"

        single_module_str = single_module_file(module, funs_in_module, references)
        with open(generated_path / (module + ".rst"), 'w') as module_file:
            module_file.write(single_module_str)

        for fun in funs_in_module:
            full_name = f"{module}.{fun}"
            if module in exclude_fn_members and fun in exclude_fn_members[module]:
                continue
            with open(generated_path / (full_name + ".rst"), "w") as function_file:
                single_file_str = single_fun_file(full_name, references)
                function_file.write(single_file_str)

    with open(out_filename, 'w') as f:
        f.write(all_modules_str)
