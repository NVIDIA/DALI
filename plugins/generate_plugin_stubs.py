import sys
import nvidia.dali as dali
import nvidia.dali.plugin_manager as plugin_manager

output_dir  = sys.argv[1]
plugin_path = sys.argv[2]
plugin_manager.load_library(plugin_path)

# TODO(janton): Filter so that only plugin specific APIs are scanned
dali.ops._signatures.gen_all_signatures(output_dir, "ops")
dali.ops._signatures.gen_all_signatures(output_dir, "fn")
