#!/usr/bin/env python
from __future__ import print_function, division
import argparse
import sys
try:
    import pip._internal.pep425tags as p
except:
    import pip.pep425tags as p
try:
    # For Python 3.0 and later
    from urllib.request import urlopen, HTTPError, Request
except ImportError:
    # Fall back to Python 2's urllib2
    from urllib2 import urlopen, HTTPError, Request

# keeps names of all required packages as a dict key
# required versions are list or dict with keys of CUDA version, to use default just put None
# instead of version number, direct link can be used
# put {0} in pacage link as a placeholder for python pip package version (i.e. cp27-cp27mu-linux_x86_64)
# and cuda_v for cuXX version
packages = {"numpy" : ["1.11.1"],
            "mxnet-cu90" : ["1.4.0"],
            "mxnet-cu100" : ["1.4.0"],
            "tensorflow-gpu" : {"90": ["1.7", "1.11", "1.12.0"], "100": ["1.13.1"]},
            "torch" : ["http://download.pytorch.org/whl/{cuda_v}/torch-1.0.0-{0}.whl"]
            }

parser = argparse.ArgumentParser(description='Env setup helper')
parser.add_argument('--list', '-l', help='list configs', action='store_true', default=False)
parser.add_argument('--num', '-n', help='return number of all configurations possible', action='store_true', default=False)
parser.add_argument('--install', '-i', dest='install', type=int, help="get Nth configuration", default=-1)
parser.add_argument('--remove', '-r', dest='remove', help="list packages to remove", action='store_true', default=False)
parser.add_argument('--cuda', dest='cuda', default="90", help="CUDA version to use")
parser.add_argument('--use', '-u', dest='use', default=[], help="provide only packages from this list", nargs='*')
args = parser.parse_args()

def get_package(packages, key, cuda):
    if key in packages.keys():
        if isinstance(packages[key], dict):
            return packages[key][cuda]
        else:
            return packages[key]
    else:
        return None

def get_pyvers_name(name, cuda):
    for v in [(x, y, z) for (x, y, z) in p.get_supported() if y != 'none' and 'any' not in y]:
        v = "-".join(v)
        v = name.format(v, cuda_v = "cu" + cuda)
        request = Request(v)
        request.get_method = lambda : 'HEAD'
        try:
             response = urlopen(request)
             return v
        except HTTPError:
             pass
    return ""

def print_configs(cuda):
    for key in packages.keys():
        print (key + ":")
        for val in get_package(packages, key, cuda):
            if val == None:
                val = "Default"
            elif 'http' in val:
                val = get_pyvers_name(val, cuda)
            print ('\t' + val)

def get_install_string(variant, use, cuda):
    ret = []
    for key in packages.keys():
        if key not in use:
            continue
        tmp = variant % len(get_package(packages, key, cuda))
        val = get_package(packages, key, cuda)[tmp]
        if val == None:
            ret.append(key)
        elif 'http' in val:
            ret.append(get_pyvers_name(val, cuda))
        else:
            ret.append(key + "==" + val)
        variant = variant // len(get_package(packages, key, cuda))
    # add all remaining used packages with default versions
    additional = [v for v in use if v not in packages.keys()]
    return " ".join(ret + additional)

def get_remove_string(use, cuda):
    # Remove only these which version we want to change
    to_remove = []
    for key in packages.keys():
        if key not in use:
            continue
        if len(get_package(packages, key, cuda)) > 1:
            to_remove.append(key)
    return " ".join(to_remove)

def cal_num_of_configs(use, cuda):
    ret = 1
    for key in packages.keys():
        if key not in use:
            continue
        ret *= len(get_package(packages, key, cuda))
    return ret

def main():
    global args
    if args.list:
        print_configs(args.cuda)
    elif args.num:
        print (cal_num_of_configs(args.use, args.cuda) - 1)
    elif args.remove:
        print (get_remove_string(args.use, args.cuda))
    elif args.install >= 0:
        if args.install > cal_num_of_configs(args.use, args.cuda):
            args.install = 1
        print (get_install_string(args.install, args.use, args.cuda))

if __name__ == "__main__":
    main()
