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
# required versions are list, to use default just put None
# instead of version number, direct link can be used
# put {0} in pacage link as a placeholder for python pip package version (i.e. cp27-cp27mu-linux_x86_64)
packages = {"numpy" : ["1.11.1"],
            "opencv-python" : ["3.1.0"],
            "mxnet-cu90" : ["1.3.0b20180612"],
            "tensorflow-gpu" : ["1.7", "1.8", "1.9", "1.10.0rc0"],
            "torch" : ["http://download.pytorch.org/whl/cu90/torch-0.4.0-{0}.whl"]
             }

parser = argparse.ArgumentParser(description='Env setup helper')
parser.add_argument('--list', '-l', help='list configs', action='store_true', default=False)
parser.add_argument('--num', '-n', help='return number of all configurations possible', action='store_true', default=False)
parser.add_argument('--install', '-i', dest='install', type=int, help="get Nth configuration", default=-1)
parser.add_argument('--remove', '-r', dest='remove', help="list packages to remove", action='store_true', default=False)
parser.add_argument('--use', '-u', dest='use', default=[], help="provide only packages from this list", nargs='*')
args = parser.parse_args()

def get_pyvers_name(name):
    for v in [(x, y, z) for (x, y, z) in p.get_supported() if y != 'none' and 'any' not in y]:
        v = "-".join(v)
        v = name.format(v)
        request = Request(v)
        request.get_method = lambda : 'HEAD'
        try:
             response = urlopen(request)
             return v
        except HTTPError:
             pass
    return ""

def print_configs():
    for key in packages.keys():
        print (key + ":")
        for val in packages[key]:
            if val == None:
                val = "Default"
            elif 'http' in val:
                val = get_pyvers_name(val)
            print ('\t' + val)

def print_install(variant, use):
    ret = []
    for key in packages.keys():
        if key not in use:
            continue
        tmp = variant % len(packages[key])
        val = packages[key][tmp]
        if val == None:
            ret.append(key)
        elif 'http' in val:
            ret.append(get_pyvers_name(val))
        else:
            ret.append(key + "==" + val)
        variant = variant // len(packages[key])
    # add all remaining used packages with default versions
    additional = [v for v in use if v not in packages.keys()]
    return " ".join(ret + additional)

def print_remove(use):
    # Remove only these which version we want to change
    to_remove = []
    for key in packages.keys():
        if key not in use:
            continue
        if len(packages[key]) > 1:
            to_remove.append(key)
    return " ".join(to_remove)

def cal_num_of_configs(use):
    ret = 1
    for key in packages.keys():
        if key not in use:
            continue
        ret *= len(packages[key])
    return ret

def main():
    global args
    if args.list:
        print_configs()
    elif args.num:
        print (cal_num_of_configs(args.use) - 1)
    elif args.remove:
        print (print_remove(args.use))
    elif args.install >= 0:
        if args.install > cal_num_of_configs(args.use): 
            args.install = 1
        print (print_install(args.install, args.use))

if __name__ == "__main__":
    main()
