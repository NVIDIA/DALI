#!/usr/bin/python3

from pathlib import Path

def _parse_entry(entry):
    if isinstance(entry, str) and entry.endswith('ipynb'):
        return example_entry(jupyter_name=entry)
    else:
        return entry
class Doc:
    def __init__(self, title, options, entries):
        self.title = title
        self.options = options
        self.entries = entries
        self.entries = [_parse_entry(entry) for entry in entries]

class ExampleEntry:
    def __init__(self, jupyter_name, operator_ref):
        self.jupyter_name = jupyter_name
        if operator_ref is not None:
            if isinstance(operator_ref, list):
                for elem in operator_ref:
                    if not isinstance(elem, OpReference):
                        raise TypeError(
                            "Expected a single op_reference of a list of them to be provided")
            elif not isinstance(operator_ref, OpReference):
                raise TypeError("Expected a single op_reference of a list of them to be provided")
        self.operator_ref = operator_ref

    def __str__(self):
        return self.jupyter_name

class OpReference:
    def __init__(self, operator, docstring):
        self.operator = operator
        self.docstring = docstring


def doc(title, options, entries):
    """Main entry point for index.py file that replaces a standard index.rst file.

    The contents of this doc will be used to generate corresponding index.rst

    Parameters
    ----------
    title : str or tuple[str, str]
        Either a title used within `..title::` directive or a tuple of (title, underline_char).
        In the second case, the underline_char will be used to do the sphinx header by placing
        it len(title) times under the title.
    options : str or list[str]
        List of options like `:maxdepth:` for the toctree.
    entries : list[str or example_entry(...)]
        Toctree of subpages, can be either represented by regular strings or by
        `example_entry()` that allows to put the reference from operator to given notebook.
    """
    global doc_return_value
    doc_return_value = Doc(title, options, entries)


def example_entry(jupyter_name, operator_ref = None):
    """Place given notebook in the toctree and optionally add a reference from operator documentation
    to that notebook.

    Parameters
    ----------
    jupyter_name : str
        Name of jupyter notebook, another index file is not supported now.
    operator_ref : OpReference or List[OpReference], optional
        Optional reference, defined by `op_reference()` call, by default None
    """
    return ExampleEntry(jupyter_name, operator_ref)


def op_reference(operator, docstring):
    """Add a reference from operator to this notebook with specified docstring.

    Parameters
    ----------
    operator : str
        Name of operator without nvidia.dali prefix, for example fn.resize or fn.gaussian_blur
    docstring : str
        Text that would appear in the see also block for given link.
    """
    return OpReference(operator, docstring)


def _obtain_doc(py_file):
    """Extract the doc() definition from index.py file"""
    with open(py_file, 'r') as f:
        doc_file = f.read()
        exec(doc_file)
        return doc_return_value


def _document_examples(path, result_dict={}):
    py_file = path + ".py"
    rst_file = path + ".rst"

    doc_contents = _obtain_doc(py_file)
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
    for entry in doc_contents.entries:
        if isinstance(entry, str):
            _document_examples(str(base_path / entry), result_dict)
            # TODO(klecki) if someone wants to link to a index page from operator
            # it can be kinda added here
        else:
            if entry.operator_ref is None:
                continue
            # Ternary expression is super long and Python formatters are abysmal
            if isinstance(entry.operator_ref, OpReference):
                op_refs = [entry.operator_ref]
            else:
                op_refs = entry.operator_ref

            for op_ref in op_refs:
                if not op_ref.operator in result_dict:
                    result_dict[op_ref.operator] = []

                result_dict[op_ref.operator].append(
                    (op_ref.docstring, str(base_path / entry.jupyter_name)[:-6] + ".html"))

    return result_dict

def document_examples(path):
    """Main api entry point, for given path to top-level index.py file containing doc() defintion
    will generate a dictionary mapping operator/module to the list of referenced examples.

    Parameters
    ----------
    path : str
        Path to doc containing

    Returns
    -------
    Dict
        Mapping from fn.operator or fn.module to list of example references
    """
    return _document_examples(path)


# print(document_examples('examples/index'))
