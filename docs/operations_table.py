from nvidia.dali import backend as b
import nvidia.dali.ops as ops
import nvidia.dali.fn as fn
import nvidia.dali.plugin.pytorch
import sys

# Dictionary with modules that can have registered Ops
ops_modules = {
    'nvidia.dali.ops': nvidia.dali.ops,
    'nvidia.dali.plugin.pytorch': nvidia.dali.plugin.pytorch
}

def to_fn_name(full_op_name):
    tokens = full_op_name.split('.')
    tokens[-1] = fn._to_snake_case(tokens[-1])
    return '.'.join(tokens)

def name_sort(op_name):
    _, module, name = ops._process_op_name(op_name)
    return '.'.join(module + [name.upper()])

def main(out_filename):
    cpu_ops = ops.cpu_ops()
    gpu_ops = ops.gpu_ops()
    mix_ops = ops.mixed_ops()
    all_ops = cpu_ops.union(gpu_ops).union(mix_ops)
    longest_module = max(ops_modules.keys(), key = len)
    link_formatter = ':meth:`{op} <{module}.{op}>`'
    op_name_max_len = len(link_formatter.format(op = "", module = longest_module)) + \
                      2 * len(max(all_ops, key=len))
    name_bar = op_name_max_len * '='
    formater = '{:{c}<{op_name_max_len}} {:{c}^16} {:{c}^150}\n'
    doc_table = ''
    doc_table += formater.format('', '', '', op_name_max_len = op_name_max_len, c='=')
    doc_table += formater.format('Function', 'Device support', 'Short description', op_name_max_len = op_name_max_len, c=' ')
    doc_table += formater.format('', '', '', op_name_max_len = op_name_max_len, c='=')
    for op in sorted(all_ops, key=name_sort):
        op_full_name, submodule, op_name = ops._process_op_name(op)
        schema = b.TryGetSchema(op)
        short_descr = ''
        devices = []
        if op in cpu_ops:
            devices += ['CPU']
        if op in mix_ops:
            devices += ['Mixed']
        if op in gpu_ops:
            devices += ['GPU']
        devices_str = ', '.join(devices)
        if schema:
            if schema.IsDocHidden():
                continue
            full_doc = schema.Dox()
        else:
            full_doc = eval('ops.' + op).__doc__
        short_descr = full_doc.split("\n\n")[0].replace('\n', ' ')
        for (module_name, module) in ops_modules.items():
            m = module
            for part in submodule:
                m = getattr(m, part, None)
                if m is None:
                    break
            if m is not None and hasattr(m, op_name):
                submodule_str = ".".join([*submodule])
                fn_string = link_formatter.format(op = to_fn_name(op_full_name), module = module_name.replace('.ops', '.fn'))
        op_doc = formater.format(fn_string, devices_str, short_descr, op_name_max_len = op_name_max_len, c=' ')
        doc_table += op_doc
    doc_table += formater.format('', '', '', op_name_max_len = op_name_max_len, c='=')
    with open(out_filename, 'w') as f:
        f.write(doc_table)

if __name__ == "__main__":
    assert(len(sys.argv) == 2)
    main(sys.argv[1])
