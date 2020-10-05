from nvidia.dali import backend as b
import nvidia.dali.ops as ops
import nvidia.dali.plugin.pytorch
import sys
import inspect

# Dictionary with modules that can have registered Ops
ops_modules = {
    'nvidia.dali.ops': nvidia.dali.ops,
}

exclude_members = {
    'nvidia.dali.ops': ["PythonFunctionBase"]
}

def main(argv):
    s = ""
    for module in sys.modules.keys():
        for doc_module in ops_modules:
            if module.startswith(doc_module):
                s += ".. automodule:: {}\n".format(module)
                s += "   :members:\n"
                s += "   :special-members: __call__\n"
                if module in exclude_members:
                    for excluded in exclude_members[module]:
                        s += "   :exclude-members: {}\n".format(excluded)
                s += "\n"
    with open(argv[0], 'w') as f:
        f.write(s)

if __name__ == "__main__":
    main(sys.argv[1:])
