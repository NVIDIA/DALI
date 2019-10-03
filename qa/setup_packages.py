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
# NOTE: First version will be picked in case of one_config_only
package_data = {
    "opencv-python" : ["4.1.0.25"],
    "mxnet-cu90" : ["1.5.0"],
    "mxnet-cu100" : ["1.5.0"],
    "tensorflow-gpu" : {
        "90": ["1.12.0", "1.11", "1.7"],
        "100": ["1.13.1", "1.14.0"]},
    "torch" : {
        "90": [("1.1.0", "http://download.pytorch.org/whl/90/torch_stable.html")],
        "100": [("1.2.0", "http://download.pytorch.org/whl/100/torch_stable.html")]
    },
    "torchvision" : {
        "90": [("0.3.0", "http://download.pytorch.org/whl/90/torch_stable.html")],
        "100": [("0.4.0", "http://download.pytorch.org/whl/100/torch_stable.html")],
    }
}

parser = argparse.ArgumentParser(description='Env setup helper')
parser.add_argument('--list', '-l', help='list configs', action='store_true', default=False)
parser.add_argument('--num', '-n', help='return number of all configurations possible', action='store_true', default=False)
parser.add_argument('--install', '-i', dest='install', type=int, help="get Nth configuration", default=-1)
parser.add_argument('--all', '-a', dest='getall', action='store_true', help='return packages in all versions')
parser.add_argument('--remove', '-r', dest='remove', help="list packages to remove", action='store_true', default=False)
parser.add_argument('--cuda', dest='cuda', default="90", help="CUDA version to use")
parser.add_argument('--use', '-u', dest='use', default=[], help="provide only packages from this list", nargs='*')
parser.add_argument('--include-link', dest='include_link', help='include -f links', action='store_true', default=False)
args = parser.parse_args()

def get_package_list(package_data, key, cuda):
    if key in package_data.keys():
        if isinstance(package_data[key], dict):
            return package_data[key][cuda]
        else:
            return package_data[key]
    else:
        return None

def get_version(package_version_entry):
    if isinstance(package_version_entry, tuple):
        return package_version_entry[0]
    elif isinstance(package_version_entry, str):
        return package_version_entry
    else:
        return "Default"

def print_configs(cuda):
    for key in package_data.keys():
        print (key + ":")
        for val in get_package_list(package_data, key, cuda):
            print ('\t' + get_version(val))

def get_install_string(variant, use, cuda):
    ret = []
    for key in package_data.keys():
        if key not in use:
            continue
        pkg_list_len = len(get_package_list(package_data, key, cuda))
        idx = variant % pkg_list_len
        val = get_package_list(package_data, key, cuda)[idx]
        pkg_str = key
        if isinstance(val, str):
            pkg_str = key + "==" + val
        elif isinstance(val, tuple):
            version, url = val
            pkg_str = key + '==' + version
            if url and args.include_link:
                pkg_str = pkg_str + ' -f ' + url
        ret.append(pkg_str)
        variant = variant // pkg_list_len
    # add all remaining used packages with default versions
    additional = [v for v in use if v not in package_data.keys()]
    return ret + additional

def get_remove_string(use, cuda):
    # Remove only these which version we want to change
    to_remove = []
    for key in package_data.keys():
        if key not in use:
            continue
        pkg_list_len = len(get_package_list(package_data, key, cuda))
        if pkg_list_len > 1:
            to_remove.append(key)
    return to_remove

def cal_num_of_configs(use, cuda):
    ret = 1
    for key in package_data.keys():
        if key not in use:
            continue
        ret *= len(get_package_list(package_data, key, cuda))
    return ret

def get_all_strings(use, cuda):
    ret = []
    for key in package_data.keys():
        if key not in use:
            continue
        for val in get_package_list(package_data, key, cuda):
            if val is None:
                pkg_str = key
            if isinstance(val, str):
                pkg_str = key + "==" + val
            elif isinstance(val, tuple):
                version, _ = val
                pkg_str = key + '==' + version
            ret.append(pkg_str)
    # add all remaining used packages with default versions
    additional = [v for v in use if v not in package_data.keys()]
    return ret + additional

def main():
    global args
    if args.list:
        print_configs(args.cuda)
    elif args.num:
        print (cal_num_of_configs(args.use, args.cuda) - 1)
    elif args.remove:
        print (get_remove_string(args.use, args.cuda))
    elif args.getall:
        print(get_all_strings(args.use, args.cuda))
    elif args.install >= 0:
        if args.install > cal_num_of_configs(args.use, args.cuda):
            args.install = 1
        print (get_install_string(args.install, args.use, args.cuda))

if __name__ == "__main__":
    main()
