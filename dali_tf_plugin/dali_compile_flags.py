#!/usr/bin/python

import sys
import os
import argparse


def get_module_path(module_name):
    module_path = ""
    for d in sys.path:
        possible_path = os.path.join(d, module_name)
        # skip current dir as this is plugin dir
        if os.path.isdir(possible_path) and len(d) != 0:
            module_path = possible_path
            break
    return module_path


def get_dali_build_flags():
    dali_cflags = ""
    dali_lflags = ""
    dali_include_flags = ""
    try:
        import nvidia.dali.sysconfig as dali_sc

        dali_include_flags = " ".join(dali_sc.get_include_flags())
        dali_cflags = " ".join(dali_sc.get_compile_flags())
        dali_lflags = " ".join(dali_sc.get_link_flags())
    except BaseException:
        dali_path = get_module_path("nvidia/dali")
        if dali_path != "":
            dali_include_flags = " ".join(["-I" + dali_path + "/include"])
            dali_cflags = " ".join(["-I" + dali_path + "/include", "-D_GLIBCXX_USE_CXX11_ABI=0"])
            dali_lflags = " ".join(["-L" + dali_path, "-ldali"])
    if dali_include_flags == "" and dali_cflags == "" and dali_lflags == "":
        raise ImportError("Could not find DALI.")
    return (dali_include_flags, dali_cflags, dali_lflags)


parser = argparse.ArgumentParser(description="DALI TF plugin compile flags")
parser.add_argument("--include_flags", dest="include_flags", action="store_true")
parser.add_argument("--cflags", dest="cflags", action="store_true")
parser.add_argument("--lflags", dest="lflags", action="store_true")
args = parser.parse_args()

include_flags, cflags, lflags = get_dali_build_flags()

flags = []

if args.include_flags:
    flags = flags + [include_flags]

if args.cflags:
    flags = flags + [cflags]

if args.lflags:
    flags = flags + [lflags]

print(" ".join(flags))
