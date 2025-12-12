# Copyright (c) 2020-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import nvidia.dali.plugin.pytorch
import nvidia.dali.plugin.numba
import nvidia.dali.plugin.jax
import nvidia.dali.experimental.dynamic
import inspect
import sys

try:
    import nvidia.dali.plugin.video
except ImportError:
    pass  # nvidia-dali-plugin not present

import operations_table

# Dictionary with modules that can have registered Ops
ops_modules = {
    "nvidia.dali.ops": nvidia.dali.ops,
    "nvidia.dali.plugin.numba.experimental": nvidia.dali.plugin.numba.experimental,
}


exclude_ops_members = {"nvidia.dali.ops": ["PythonFunctionBase"]}


fn_modules = {
    "nvidia.dali.fn": nvidia.dali.fn,
    "nvidia.dali.plugin.pytorch.fn": nvidia.dali.plugin.pytorch.fn,
    "nvidia.dali.plugin.jax.fn": nvidia.dali.plugin.jax.fn,
    "nvidia.dali.plugin.numba.fn.experimental": nvidia.dali.plugin.numba.fn.experimental,
}

dynamic_modules = {
    "nvidia.dali.experimental.dynamic": nvidia.dali.experimental.dynamic,
}

exclude_fn_members = {}

installation_page_url = (
    "https://docs.nvidia.com/deeplearning/dali/user-guide/"
    "docs/installation.html"
)

mod_aditional_doc = {
    "nvidia.dali.fn.transforms": (
        "All operators in this module support only CPU device as they are meant to be provided"
        " as an input to named keyword operator arguments. Check for more details the relevant"
        " :ref:`pipeline documentation section<Processing Graph Structure>`."
    ),
    "nvidia.dali.fn.plugin.video": (
        ".. note::\n\n    "
        "This module belongs to the `nvidia-dali-video` plugin, that needs to be installed "
        "as a separate package. Refer to the `Installation Guide "
        f"<{installation_page_url}#nvidia-dali-video>`__"
        " for more details."
    ),
    "nvidia.dali.fn.readers": (
        "Operators in this module are data-producing operators that read data from storage or a"
        " different source, and where the data locations are known at pipeline construction time"
        " via arguments. For data readers that are able to read from sources specified dynamically"
        " via regular inputs, see `nvidia.dali.fn.io` module."
    ),
    "nvidia.dali.fn.io": (
        "Operators in this module are data-reading operators that read data from a source  "
        " specified at runtime by operator inputs. For inputless data readers that are able "
        " to build the dataset at pipeline constructions, see `nvidia.dali.fn.readers` module."
    ),
}


def _is_private(module):
    submodules = module.split(".")
    return any([submodule.startswith("_") for submodule in submodules])


def get_modules(top_modules):
    modules = []
    for module in sys.modules.keys():
        for doc_module in top_modules:
            if (
                module.startswith(doc_module)
                and not module.endswith("hidden")
                and not _is_private(module)
            ):
                modules += [module]
    return sorted(modules)


def get_functions(module):
    """Get all function names (so DALI API operators) from given DALI module without private
    or hidden members. No nested modules would be reported."""
    result = []
    # Take all public members of given module
    public_members = list(
        filter(lambda x: not str(x).startswith("_"), dir(module))
    )
    for member_name in public_members:
        member = getattr(module, member_name)
        # Just user-defined functions
        if inspect.isfunction(member) and not member.__module__.endswith(
            "hidden"
        ):
            result.append(member_name)
    return result


def get_schema_names(module, functions):
    def get_schema_name_or_dummy_schema(fun):
        obj = getattr(sys.modules[module], fun)
        if hasattr(obj, "_schema_name"):
            return obj._schema_name
        else:
            return operations_table.no_schema_fns[f"{module}.{fun}"]

    return [get_schema_name_or_dummy_schema(fun) for fun in functions]


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
    with open(out_filename, "w") as f:
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
    """Generate stub page for documentation of given function from fn api."""
    result = f"{full_name}\n"
    result += "-" * len(full_name) + "\n\n"
    result += f".. autofunction:: {full_name}\n\n"
    result += get_references(full_name, references)
    return result


def single_class_op_file(full_name, references):
    """Generate stub page for documentation of class-like operator from dynamic api."""
    result = f"{full_name}\n"
    result += "-" * len(full_name) + "\n\n"
    result += f".. autoclass:: {full_name}\n"
    result += "   :members: next_epoch\n"
    result += "   :special-members: __init__\n\n"
    result += get_references(full_name, references)
    return result


def single_module_file(module, funs_in_module, references):
    """Generate stub page for documentation of given module"""
    result = f"{module}\n"
    result += "~" * len(module) + "\n\n"

    if module in mod_aditional_doc:
        result += mod_aditional_doc[module] + "\n\n"
    result += get_references(module, references)
    result += "\n"

    result += f"The following table lists all operations available in ``{module}`` module:\n"
    result += operations_table.operations_table_str(
        get_schema_names(module, funs_in_module), module_name=module
    )
    result += "\n\n"

    result += ".. toctree::\n   :hidden:\n\n"

    for fun in funs_in_module:
        if module in exclude_fn_members and fun in exclude_fn_members[module]:
            continue
        full_name = f"{module}.{fun}"
        result += f"   {full_name}\n"
    return result


def write_toctree(all_modules, relative_generated_path, out_filename):
    all_modules_str = ".. toctree::\n   :hidden:\n\n"
    for module in all_modules:
        # As the top-level file is included from a directory above generated_path
        # we need to provide the relative path to the per-module files
        # the rest is within the same directory, so there is no need for that
        all_modules_str += f"   {relative_generated_path / module}\n"
    with open(out_filename, "w") as f:
        f.write(all_modules_str)


def write_module_file(generated_path, module, funs_in_module, references):
    if len(funs_in_module) == 0:
        return
    single_module_str = single_module_file(module, funs_in_module, references)
    with open(generated_path / (module + ".rst"), "w") as module_file:
        module_file.write(single_module_str)


def write_function_files(
    module, funs_in_module, file_content_generator, generated_path, references
):
    for fun in funs_in_module:
        full_name = f"{module}.{fun}"
        with open(generated_path / (full_name + ".rst"), "w") as function_file:
            single_file_str = file_content_generator(full_name, references)
            function_file.write(single_file_str)


def fn_autodoc(out_filename, generated_path, references):
    all_modules = get_modules(fn_modules)
    write_toctree(all_modules, generated_path, out_filename)
    for module in all_modules:
        dali_module = sys.modules[module]
        funs_in_module = [
            fun
            for fun in get_functions(dali_module)
            if module not in exclude_fn_members
            or fun not in exclude_fn_members[module]
        ]
        write_module_file(generated_path, module, funs_in_module, references)
        write_function_files(
            module, funs_in_module, single_fun_file, generated_path, references
        )


def dynamic_autodoc(
    out_filename, generated_path, relative_generated_path, references
):
    all_modules = get_modules(dynamic_modules)
    write_toctree(all_modules, relative_generated_path, out_filename)
    for module in all_modules:
        dali_module = sys.modules[module]
        funs_in_module = [
            fun
            for fun in get_functions(dali_module)
            if hasattr(getattr(dali_module, fun), "_schema_name")
        ]

        write_module_file(generated_path, module, funs_in_module, references)
        write_function_files(
            module, funs_in_module, single_fun_file, generated_path, references
        )


def dynamic_readers_autodoc(
    out_filename, generated_path, relative_generated_path, references
):
    all_modules = [m for m in get_modules(dynamic_modules) if "readers" in m]
    write_toctree(all_modules, relative_generated_path, out_filename)
    for module in all_modules:
        dali_module = sys.modules[module]
        readers_in_module = [
            k
            for k, v in dali_module.__dict__.items()
            if inspect.isclass(v)
            and issubclass(v, nvidia.dali.experimental.dynamic._ops.Reader)
        ]

        write_module_file(generated_path, module, readers_in_module, references)
        write_function_files(
            module,
            readers_in_module,
            single_class_op_file,
            generated_path,
            references,
        )
