#!/usr/bin/python3

from pathlib import Path
from typing import Type

from numpy import isin

def parse_entry(entry):
    if isinstance(entry, str) and entry.endswith('ipynb'):
        return example_entry(jupyter_name=entry)
    else:
        return entry

class Doc:
    def __init__(self, title, options, entries):
        self.title = title
        self.options = options
        self.entries = entries
        self.entries = [parse_entry(entry) for entry in entries]

def doc(title, options, entries):
    global doc_return_value
    doc_return_value = Doc(title, options, entries)
    print(title, options, entries)


class ExampleEntry:
    def __init__(self, jupyter_name, operator_ref):
        self.jupyter_name = jupyter_name
        if operator_ref is not None:
            if isinstance(operator_ref, list):
                for elem in operator_ref:
                    if not isinstance(elem, OpReference):
                        raise TypeError("Expected a single op_reference of a list of them to be provided")
            elif not isinstance(operator_ref, OpReference):
                raise TypeError("Expected a single op_reference of a list of them to be provided")
        self.operator_ref = operator_ref

    def __str__(self):
        return self.jupyter_name

class OpReference:
    def __init__(self, operator, docstring):
        self.operator = operator
        self.docstring = docstring


# example listed on the page
def example_entry(jupyter_name, operator_ref: OpReference=None):
    return ExampleEntry(jupyter_name, operator_ref)

# Add a reference for this tutorial in given operator doc with optional docstring
def op_reference(operator, docstring=None):
    return OpReference(operator, docstring)


def obtain_doc(py_file):
    with open(py_file, 'r') as f:
        doc_file = f.read()
        exec(doc_file)
        return doc_return_value



def document_examples(path, result_dict={}):

    py_file = path + ".py"
    rst_file = path + ".rst"
    print(">>> DOCUMENTING FOR", path, py_file, rst_file)

    doc_contents = obtain_doc(py_file)
    tab = " " * 3
    with open(rst_file, "w") as f:
        if isinstance(doc_contents.title, str):
            f.write(f".. title:: {doc_contents.title}\n\n")
        else:
            title, level = doc_contents.title
            f.write(f"{title}\n{level * len(title)}\n\n")

        f.write(f".. toctree::\n")
        if not isinstance(doc_contents.options, list):
            doc_contents.options = [doc_contents.options]
        for option in doc_contents.options:
            f.write(f"{tab}{option}\n")
        f.write("\n")
        for entry in doc_contents.entries:
            f.write(f"{tab}{entry}\n")

    canonical_path = Path(path)
    base_path = canonical_path.parent
    print(base_path)
    for entry in doc_contents.entries:
        if isinstance(entry, str):
            document_examples(str(base_path / entry), result_dict)
        else:
            if  entry.operator_ref is None:
                continue
            op_refs = [entry.operator_ref] if isinstance(entry.operator_ref, OpReference) else entry.operator_ref
            for op_ref in op_refs:
                if not op_ref.operator in result_dict:
                    result_dict[op_ref.operator] = []

                result_dict[op_ref.operator].append((op_ref.docstring, str(base_path / entry.jupyter_name)))

                print(f"Adding the reference for {op_ref.operator} := {result_dict[op_ref.operator][-1]}")
    return result_dict



print(document_examples('examples/index'))
