import sys
import types

def get_submodule(root, path):
    """Gets or creates sumbodule(s) of `root`.
If the module path contains multiple parts, multiple modules are traversed or created

Parameters
----------
    `root`
        module object or name of the root module
    `path`
        period-separated path of the submodule or a list/tuple of submodule names"""

    if isinstance(root, str):
        root = sys.modules[root]

    if not path:
        return root

    if isinstance(path, str):
        if str == '':
            return root
        path = path.split('.')

    module_name = root.__name__
    for part in path:
        m = getattr(root, part, None)
        module_name += '.' + part
        if m is None:
            m = sys.modules[module_name] = types.ModuleType(module_name)
            setattr(root, part, m)
        elif not isinstance(m, types.ModuleType):
            raise RuntimeError("The module {} already contains an attribute \"{}\", which is not a module, but {}".format(
                root, part, m))
        root = m
    return root
