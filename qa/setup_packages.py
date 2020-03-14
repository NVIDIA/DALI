#!/usr/bin/env python
from __future__ import print_function, division
import argparse
import sys
# use packaging from PIP as it is always present on system we are testing on
from pip._vendor.packaging.version import parse
import urllib.parse
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

packages = {
            "opencv-python" : ["4.2.0.32"],
            "cupy-cuda{cuda_v}" : {
                        "90" : ["6.6.0"],
                        "100" : ["6.6.0"]},
            "mxnet-cu{cuda_v}" : {
                        "90" : ["1.6.0"],
                        "100" : ["1.5.1"]},
            "tensorflow-gpu" : {
                        "90": ["1.12.0",],
                        "100": ["1.14.0", "1.15.2", "2.0.1", "2.1.0"]},
            "torch" : {
                        "90": ["http://download.pytorch.org/whl/{cuda_v}/torch-1.1.0-{0}.whl"],
                        "100": ["http://download.pytorch.org/whl/{cuda_v}/torch-1.4.0+{cuda_v}-{0}.whl"]},
            "torchvision" : {
                        "90": ["https://download.pytorch.org/whl/{cuda_v}/torchvision-0.3.0-{0}.whl"],
                        "100": ["https://download.pytorch.org/whl/{cuda_v}/torchvision-0.5.0+{cuda_v}-{0}.whl"]},
            "paddle" : {
                        "90": ["https://paddle-wheel.bj.bcebos.com/gcc54/latest-gpu-cuda9-cudnn7-openblas/paddlepaddle_gpu-latest-{0}.whl"],
                        "100": ["https://paddle-wheel.bj.bcebos.com/gcc54/latest-gpu-cuda10-cudnn7-openblas/paddlepaddle_gpu-latest-{0}.whl"]},
            }

parser = argparse.ArgumentParser(description='Env setup helper')
parser.add_argument('--list', '-l', help='list configs', action='store_true', default=False)
parser.add_argument('--num', '-n', help='return number of all configurations possible', action='store_true', default=False)
parser.add_argument('--install', '-i', dest='install', type=int, help="get Nth configuration", default=-1)
parser.add_argument('--all', '-a', dest='getall', action='store_true', help='return packages in all versions')
parser.add_argument('--remove', '-r', dest='remove', help="list packages to remove", action='store_true', default=False)
parser.add_argument('--cuda', dest='cuda', default="90", help="CUDA version to use")
parser.add_argument('--use', '-u', dest='use', default=[], help="provide only packages from this list", nargs='*')
args = parser.parse_args()

def filter_packages(key, vers):
    """Filter out a version of package from list. Mainly for tensorflow-gpu where Python 3.8 is
      supported only from 2.2.0"""
    if vers is None:
        return vers
    if key == "tensorflow-gpu":
        tmp = []
        for v in vers:
            # in case of tensorflow python 3.8 is supported only from 2.2.0
            python_version = ".".join([str(x) for x in sys.version_info[0:3]])
            if (parse("2.2.0") <= parse(v) and parse("3.8.0") <= parse(python_version)) or \
                parse("3.8.0") > parse(python_version):
                tmp.append(v)
        vers = tmp
    return vers

def get_key_with_cuda(key, val_dict, cuda):
    """Get the set of versions for highest matching cuda version available.

       I.e. for cuda 9.2 it will get versions for cuda 9.0, for cuda 10.1 one for 10.0"""
    key_w_cuda = key
    max_cuda = None
    if isinstance(val_dict, dict):
        for ver in sorted(val_dict.keys(), key=int):
            if int(ver) <= int(cuda):
                key_w_cuda = key.format(cuda_v=ver)
                max_cuda = ver
    return key_w_cuda, max_cuda

def get_package(package_data, key, cuda):
    """Returns a list of available versions for given package name and cuda version"""
    ret = None
    if key in package_data.keys():
        if isinstance(package_data[key], dict):
            data = None
            for ver in sorted(package_data[key].keys(), key=int):
                if int(ver) <= int(cuda):
                   data = package_data[key][ver]
            ret = data
        else:
            ret = packages[key]

    return filter_packages(key, ret)

def test_request(py_ver, url, cuda):
    """Check if given `url` to a package of version `py_ver` is available"""
    py_ver = url.format(py_ver, cuda_v = "cu" + cuda)
    py_ver = py_ver.split("://")
    py_ver[-1] = urllib.parse.quote(py_ver[-1])
    py_ver = "://".join(py_ver)
    request = Request(py_ver)
    request.get_method = lambda : 'HEAD'
    try:
        _ = urlopen(request)
        return py_ver
    except HTTPError:
        return None

def get_pyvers_name(name, cuda):
    """Test if any compatible package url exists for a platform reported by PIP"""
    if isinstance(p.get_supported()[0], tuple):
        # old PIP returns tuple
        for v in [(x, y, z) for (x, y, z) in p.get_supported() if y != 'none' and 'any' not in y]:
            v = "-".join(v)
            ret = test_request(v, name, cuda)
            if ret:
                return ret
    else:
        # new PIP returns object
        for t in [tag for tag in p.get_supported() if tag.abi != 'none' and tag.platform != 'any']:
            t = str(t)
            ret = test_request(t, name, cuda)
            if ret:
                return ret
    return ""

def print_configs(cuda):
    """Prints all available configurations"""
    for key in packages.keys():
        key_w_cuda, max_cuda = get_key_with_cuda(key, packages[key], cuda)
        print (key_w_cuda + ":")
        for val in get_package(packages, key, max_cuda):
            if val == None:
                val = "Default"
            elif val.startswith('http'):
                val = get_pyvers_name(val, max_cuda)
            print ('\t' + val)

def get_install_string(variant, use, cuda):
    """Creates pip install string for given cuda version, variant number and package list

    It supports names, http direct links and name remaping like tensorflow-gpu->tensorflow for
    some specific versions.
    """
    ret = []
    for key in packages.keys():
        if key not in use:
            continue
        key_w_cuda, max_cuda = get_key_with_cuda(key, packages[key], cuda)
        tmp = variant % len(get_package(packages, key, max_cuda))
        val = get_package(packages, key, max_cuda)[tmp]
        if val == None:
            ret.append(key_w_cuda)
        elif val.startswith('http'):
            ret.append(get_pyvers_name(val, max_cuda))
        else:
            ret.append(key_w_cuda + "==" + val)
        variant = variant // len(get_package(packages, key, max_cuda))
    # add all remaining used packages with default versions
    additional = [v for v in use if v not in packages.keys()]
    return " ".join(ret + additional)

def get_remove_string(use, cuda):
    """Creates pip remove string for given cuda version and package list"""
    # Remove only these which version we want to change
    to_remove = []
    for key in packages.keys():
        if key not in use:
            continue
        key_w_cuda, max_cuda = get_key_with_cuda(key, packages[key], cuda)
        pkg_list_len = len(get_package(packages, key, max_cuda))
        if pkg_list_len > 1:
            to_remove.append(key_w_cuda)
    return " ".join(to_remove)

def cal_num_of_configs(use, cuda):
    """Calculates how many different version configurations are available for given
       package and cuda version"""
    ret = 1
    for key in packages.keys():
        _, max_cuda = get_key_with_cuda(key, packages[key], cuda)
        if key not in use:
            continue
        values = get_package(packages, key, max_cuda)
        # make sure that there is any compatible package under listed link 
        tmp = []
        for val in values:
            if val.startswith('http'):
                if get_pyvers_name(val, max_cuda) != "":
                    tmp.append(val)
            else:
                tmp.append(val)
        values = tmp
        ret *= len(values)
    return ret

def get_all_strings(use, cuda):
    """Prints all available configurations for given package list and cuda version"""
    ret = []
    for key in packages.keys():
        if key not in use:
            continue
        key_w_cuda, max_cuda = get_key_with_cuda(key, packages[key], cuda)
        for val in get_package(packages, key, max_cuda):
            if val is None:
                ret.append(key)
            elif val.startswith('http'):
                ret.append(get_pyvers_name(val, max_cuda))
            else:
                ret.append(key_w_cuda + "==" + val)
    # add all remaining used packages with default versions
    additional = [v for v in use if v not in packages.keys()]
    return " ".join(ret + additional)

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
            args.install = 0
        elif cal_num_of_configs(args.use, args.cuda) <= 0:
            print("")
            return
        print (get_install_string(args.install, args.use, args.cuda))

if __name__ == "__main__":
    main()
