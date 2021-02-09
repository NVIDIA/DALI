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
                      5 * len(max(all_ops, key=len))
    name_bar = op_name_max_len * '='
    formater = '{:{c}<{op_name_max_len}} {:{c}^6}  {:{c}^6}  {:{c}^7} {:{c}^9} {:{c}^10}\n'
    doc_table = ''
    doc_table += '.. |v| image:: images/tick.gif\n'
    doc_table += '\n'
    doc_table += formater.format('', '', '', '', '', '', op_name_max_len = op_name_max_len, c='=')
    doc_table += formater.format('Operator name', 'CPU', 'GPU', 'Mixed', 'Sequences', 'Volumes', op_name_max_len = op_name_max_len, c=' ')
    doc_table += formater.format('', '', '', '', '', '', op_name_max_len = op_name_max_len, c='=')
    for op in sorted(all_ops, key=name_sort):
        op_full_name, submodule, op_name = ops._process_op_name(op)
        #op_name = fn._to_snake_case(op_name)
        is_cpu = '|v|' if op in cpu_ops else ''
        is_gpu = '|v|' if op in gpu_ops else ''
        is_mixed = '|v|' if op in mix_ops else ''
        schema = b.TryGetSchema(op)
        if schema:
            supports_seq = '|v|' if schema.AllowsSequences() or schema.IsSequenceOperator() else ''
            volumetric = '|v|' if schema.SupportsVolumetric() else ''
            if schema.IsDocHidden():
                continue
        else:
            supports_seq = ''
            volumetric = ''
        for (module_name, module) in ops_modules.items():
            m = module
            for part in submodule:
                m = getattr(m, part, None)
                if m is None:
                    break
            if m is not None and hasattr(m, op_name):
                submodule_str = ".".join([*submodule])
                op_string = link_formatter.format(op = op_full_name, module = module_name)
                fn_string = link_formatter.format(op = to_fn_name(op_full_name), module = module_name.replace('.ops', '.fn'))
        op_doc = formater.format(' / '.join([fn_string, op_string]), is_cpu, is_gpu, is_mixed, supports_seq, volumetric, op_name_max_len = op_name_max_len, c=' ')
        doc_table += op_doc
    doc_table += formater.format('', '', '', '', '', '', op_name_max_len = op_name_max_len, c='=')
    with open(out_filename, 'w') as f:
        f.write(doc_table)

if __name__ == "__main__":
    assert(len(sys.argv) == 2)
    main(sys.argv[1])
