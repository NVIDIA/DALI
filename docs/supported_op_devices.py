from nvidia.dali import backend as b
import nvidia.dali.ops as ops
import nvidia.dali.plugin.pytorch
import sys

def main(argv):
    cpu_ops = ops.cpu_ops()
    gpu_ops = ops.gpu_ops()
    mix_ops = ops.mixed_ops()
    all_ops = cpu_ops.union(gpu_ops).union(mix_ops)
    link_string = '_'
    op_name_max_len = len(max(all_ops, key=len)) + len(link_string)
    name_bar = op_name_max_len * '='
    formater = '{:{c}<{op_name_max_len}} {:{c}^6}  {:{c}^6}  {:{c}^7} {:{c}^9} {:{c}^10}\n'
    doc_table = ''
    doc_table += 'Below table lists all available operators and devices they can operate on.\n\n'
    doc_table += '.. |v| image:: images/tick.gif\n'
    doc_table += '\n'
    doc_table += formater.format('', '', '', '', '', '', op_name_max_len = op_name_max_len, c='=')
    doc_table += formater.format('Operator name', 'CPU', 'GPU', 'Mixed', 'Sequences', 'Volumetric', op_name_max_len = op_name_max_len, c=' ')
    doc_table += formater.format('', '', '', '', '', '', op_name_max_len = op_name_max_len, c='=')
    for op in sorted(all_ops, key=lambda v: str(v).lower()):
        schema = b.GetSchema(op)
        is_cpu = '|v|' if op in cpu_ops else ''
        is_gpu = '|v|' if op in gpu_ops else ''
        is_mixed = '|v|' if op in mix_ops else ''
        supports_seq = '|v|' if schema.AllowsSequences() or schema.IsSequenceOperator() else ''
        volumetric = '|v|' if schema.SupportsVolumetric() else ''
        op_string = op + link_string
        op_doc = formater.format(op_string, is_cpu, is_gpu, is_mixed, supports_seq, volumetric, op_name_max_len = op_name_max_len, c=' ')
        doc_table += op_doc
    doc_table += formater.format('', '', '', '', '', '', op_name_max_len = op_name_max_len, c='=')
    with open(argv[0], 'w') as f:
        f.write(doc_table)

if __name__ == "__main__":
    main(sys.argv[1:])
