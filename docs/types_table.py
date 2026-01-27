import nvidia.dali.experimental.dynamic as ndd

ndd_types = {
    name: getattr(ndd, name)
    for name in dir(ndd)
    if isinstance(getattr(ndd, name), ndd.DType)
}


def ndd_types_table(out_filename):
    table_contents = {
        f"``nvidia.dali.experimental.dynamic.{name}``": t.__doc__.split("\n")
        for name, t in ndd_types.items()
    }
    name_max_len = max(len(name) for name in table_contents.keys())
    description_max_len = max(
        max(len(line) for line in doc) for doc in table_contents.values()
    )
    doc_table = ""

    def add_row(name, description, c=" "):
        nonlocal doc_table
        formater = "{:{c}<{name_max_len}} {:{c}<{description_max_len}}\n"
        doc_table += formater.format(
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
