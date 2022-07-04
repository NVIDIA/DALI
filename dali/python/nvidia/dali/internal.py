import importlib
import sys
import types


def get_submodule(root, path):
    """Gets or creates submodule(s) of `root`.
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
            try:
                # Try importing existing module (if not loaded yet) to not overwrite it.
                m = importlib.import_module(module_name)
            except ModuleNotFoundError:
                m = sys.modules[module_name] = types.ModuleType(module_name)
            setattr(root, part, m)
        elif not isinstance(m, types.ModuleType):
            raise RuntimeError(
                f"The module {root} already contains an attribute \"{part}\", "
                f"which is not a module, but {m}")
        root = m
    return root
