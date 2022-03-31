from nvidia.dali import backend as b
import nvidia.dali.ops as ops
import nvidia.dali.plugin.pytorch
import nvidia.dali.plugin.numba
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

def single_fun_file(fn_name):
    result = ""
    result += f".. _{fn_name}:\n{fn_name}\n"
    result += "_" * len(fn_name) + "\n\n"
    result += f".. autofunction:: {fn_name}\n\n"
    return result


def fn_autodoc(out_filename, references):
    all_modules_str = ".. toctree::\n   :hidden:\n\n"

    all_modules = get_modules(fn_modules)
    print(all_modules)
    for module in all_modules:
        single_module_str = ""
        dali_module = sys.modules[module]
        funs_in_module = list(filter(lambda x: not str(x).startswith("_"), dir(dali_module)))
        funs_in_module = list(filter(lambda x: not module + "." + str(x) in all_modules, funs_in_module))

        # TODO::
        # if module in mod_aditional_doc:
        #     s += mod_aditional_doc[module] + "\n" + "\n"


        # s += ".. automodule:: {}\n".format(module)
        # s += "   :members:\n"
        # s += "   :undoc-members:\n"
        # TODO excluded members:
        # if module in exclude_fn_members:
        #     excluded = exclude_fn_members[module]
        #     s += "   :exclude-members: {}\n".format(", ".join(excluded))
        if len(funs_in_module) == 0:
            continue
        all_modules_str += f"   {module}\n"
        single_module_str += f".. _{module}:\n{module}\n"
        single_module_str += "~" * len(module) + "\n\n"
        single_module_str += ".. toctree::\n\n"
        fn_module = module[12:]
        # if fn_module in references:
        #     s += "  See examples for this module:\n"
        #     for reference in references[fn_module]:
        #         s += "    * `{} <../examples/{}>`_\n".format(reference[0], reference[1])
        # TODO: filter internal
        for fun in funs_in_module:
            reference_key = fn_module + "." + fun
            full_name = module + "." + fun
            single_module_str += f"   {full_name}\n"

            function_file = single_fun_file(full_name)
            if reference_key in references:
                function_file += ".. seealso::\n"
                for reference in references[reference_key]:
                    function_file += "    * `{} </{}>`_\n".format(reference[0], reference[1])
            with open(full_name + ".rst", "w") as f:
                f.write(function_file)

        with open(module + ".rst", "w") as f:
            f.write(single_module_str)

    with open(out_filename, 'w') as f:
        f.write(all_modules_str)

if __name__ == "__main__":
    assert(len(sys.argv) == 3)
    op_autodoc(sys.argv[1])
    fn_autodoc(sys.argv[2])
