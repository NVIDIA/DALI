from nvidia.dali import backend as b
import sys

def main(argv):
    cpu_ops = set(b.RegisteredCPUOps())
    if '_TFRecordReader' in cpu_ops:
        cpu_ops.remove('_TFRecordReader')
        cpu_ops.add('TFRecordReader')
    gpu_ops = set(b.RegisteredGPUOps())
    mix_ops = set(b.RegisteredMixedOps())
    mix_ops.remove('MakeContiguous')
    support_ops = set(b.RegisteredSupportOps())
    all_ops = cpu_ops.union(gpu_ops).union(mix_ops).union(support_ops)
    op_name_max_len = len(max(all_ops, key=len))
    name_bar = op_name_max_len * '='
    formater = '{:{c}>{op_name_max_len}}  {:{c}>6}  {:{c}>6}  {:{c}>6}  {:{c}>7}\n'
    doc_table = ''
    doc_table += 'Below table lists all available operators and devices they can operate on.\n\n'
    doc_table += '.. |tick| image:: images/tick.gif\n'
    doc_table += formater.format('', '', '', '', '', op_name_max_len = op_name_max_len, c='=')
    doc_table += formater.format('Operator name', 'CPU', 'GPU', 'Mixed', 'Support', op_name_max_len = op_name_max_len, c=' ')
    doc_table += formater.format('', '', '', '', '', op_name_max_len = op_name_max_len, c='=')
    for op in sorted(all_ops):
        is_cpu = '|tick|' if op in cpu_ops else ''
        is_gpu = '|tick|' if op in gpu_ops else ''
        is_mixed = '|tick|' if op in mix_ops else ''
        is_support = '|tick|' if op in support_ops else ''
        op_doc = formater.format(op, is_cpu, is_gpu, is_mixed, is_support, op_name_max_len = op_name_max_len, c=' ')
        doc_table += op_doc
    doc_table += formater.format('', '', '', '', '', op_name_max_len = op_name_max_len, c='=')
    with open(argv[0], 'w') as f:
        f.write(doc_table)

if __name__ == "__main__":
    main(sys.argv[1:])
