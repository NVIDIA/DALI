import nvidia.dali.experimental.dynamic as ndd
import re

ndd_types = {
    name: getattr(ndd, name)
    for name in dir(ndd)
    if isinstance(getattr(ndd, name), ndd.DType)
}


def type_order(name):
    """
    Returns a tuple of (type group index, type size, type name) that determines the order
    of the types in the table.
    """
    type_groups = ["int", "uint", "float", "bfloat", "bool"]
    for i, group in enumerate(type_groups):
        if m := re.match(rf"^{group}(\d+)?$", name):
            return (i, int(m.group(1)) if m.group(1) else 0, name)
    return (len(type_groups), 0, name)


ordered_ndd_types = sorted(ndd_types.keys(), key=type_order)


def ndd_types_table(out_filename):
    table_contents = {
        f"``nvidia.dali.experimental.dynamic.{name}``": (
            ndd_types[name].__doc__ or ""
        ).split("\n")
        for name in ordered_ndd_types
    }
    name_max_len = max(len(name) for name in table_contents.keys())
    description_max_len = max(
        max(len(line) for line in doc) for doc in table_contents.values()
    )
    doc_table = ""

    def add_row(name, description, c=" "):
        nonlocal doc_table
        formatter = "{:{c}<{name_max_len}} {:{c}<{description_max_len}}\n"
        doc_table += formatter.format(
            name,
            description,
            name_max_len=name_max_len,
            description_max_len=description_max_len,
            c=c,
        )

    add_row("", "", c="=")
    add_row("Type", "Description", c=" ")
    add_row("", "", c="=")
    for name, doc in table_contents.items():
        add_row(name, doc[0], c=" ")
        for line in doc[1:]:
            add_row("", line, c=" ")
    add_row("", "", c="=")
    with open(out_filename, "w") as f:
        f.write(doc_table)
