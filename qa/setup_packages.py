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

packages = {
            "opencv-python" : ["4.1.0.25"],
            "cupy-cuda{cuda_v}" : {
                        "90" : ["6.6.0"],
                        "100" : ["6.6.0"]},
            "mxnet-cu{cuda_v}" : {
                        "90" : ["1.5.0"],
                        "100" : ["1.5.0"]},
            "tensorflow-gpu" : {
                "90": ["1.12.0",],
                "100": ["1.14.0", "1.15.2", "2.0.1", "2.1.0"]},
            "torch" : {"90": ["http://download.pytorch.org/whl/{cuda_v}/torch-1.1.0-{0}.whl"],
                       "100": ["http://download.pytorch.org/whl/{cuda_v}/torch-1.2.0-{0}.whl"]},
            "torchvision" : {"90": ["https://download.pytorch.org/whl/{cuda_v}/torchvision-0.3.0-{0}.whl"],
                             "100": ["https://download.pytorch.org/whl/{cuda_v}/torchvision-0.4.0-{0}.whl"]},
            "paddle" : {"90": ["https://paddle-wheel.bj.bcebos.com/gcc54/latest-gpu-cuda9-cudnn7-openblas/paddlepaddle_gpu-latest-{0}.whl"],
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

def get_key_with_cuda(key, val_dict, cuda):
    key_w_cuda = key
    max_cuda = None
    if isinstance(val_dict, dict):
        for ver in sorted(val_dict.keys(), key=int):
            if int(ver) <= int(cuda):
                key_w_cuda = key.format(cuda_v=ver)
                max_cuda = ver
    return key_w_cuda, max_cuda

def get_package(package_data, key, cuda):
    if key in package_data.keys():
        if isinstance(package_data[key], dict):
            data = None
            for ver in sorted(package_data[key].keys(), key=int):
                if int(ver) <= int(cuda):
                   data = package_data[key][ver]
            return data
        else:
            return packages[key]
    else:
        return None

def test_request(req, name, cuda):
    req = name.format(req, cuda_v = "cu" + cuda)
    request = Request(req)
    request.get_method = lambda : 'HEAD'
    try:
        _ = urlopen(request)
        return req
    except HTTPError:
        return None

def get_pyvers_name(name, cuda):
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
    ret = 1
    for key in packages.keys():
        if key not in use:
            continue
        ret *= len(get_package(packages, key, cuda))
    return ret

def get_all_strings(use, cuda):
    ret = []
    for key in packages.keys():
        if key not in use:
            continue
        _, max_cuda = get_key_with_cuda(key, packages[key], cuda)
        for val in get_package(packages, key, max_cuda):
            if val is None:
                ret.append(key)
            elif val.startswith('http'):
                ret.append(get_pyvers_name(val, max_cuda))
            else:
                ret.append(key + "==" + val)
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
        print (get_install_string(args.install, args.use, args.cuda))

if __name__ == "__main__":
    main()
