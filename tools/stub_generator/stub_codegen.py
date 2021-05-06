# Copyright 2020 The TensorFlow Runtime Authors
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Generates dynamic loading stubs for functions in CUDA and HIP APIs."""

from __future__ import absolute_import
from __future__ import print_function

import argparse
import json
import clang.cindex


def main():
    parser = argparse.ArgumentParser(
        description='Generate dynamic loading stubs for CUDA and HIP APIs.')
    parser.add_argument('--unique_prefix', default="", type=str,
                        help='Unique prefix for used in the stub')
    parser.add_argument(
        'input', nargs='?', type=argparse.FileType('r'))
    parser.add_argument(
        'output', nargs='?', type=argparse.FileType('w'))
    parser.add_argument(
        'header', nargs='?', type=str, default=None)
    parser.add_argument(
        'extra_args', nargs='*', type=str, default=None)
    args = parser.parse_args()

    config = json.load(args.input)

    function_impl = """
{return_type} %s {1}NotFound({2}) {{
  return {not_found_error};
}}

{0} {{
  using FuncPtr = {return_type} (%s *)({2});
  static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("{1}")) ?
                           reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("{1}")) :
                           {1}NotFound;
  return func_ptr({3});
}}\n""" % (config['calling_conv'], config['calling_conv'])

    prolog = """
void *{0}LoadSymbol(const char *name);

#define LOAD_SYMBOL_FUNC {0}##LoadSymbol

"""

    index = clang.cindex.Index.create()
    header = args.header
    extra_args = args.extra_args

    translation_unit = index.parse(header, args=extra_args)

    for diag in translation_unit.diagnostics:
        if diag.severity in [diag.Warning, diag.Fatal]:
            raise Exception(str(diag))

    for extra_i in config['extra_include']:
        args.output.write('#include {}\n'.format(extra_i))
    args.output.write(prolog.format(args.unique_prefix))

    all_definition = set()
    all_declaration = set()

    for cursor in translation_unit.cursor.get_children():
        if cursor.is_definition():
            all_definition.add(cursor.spelling)

        if cursor.kind == clang.cindex.CursorKind.FUNCTION_DECL:
            all_declaration.add(cursor.spelling)

    for cursor in translation_unit.cursor.get_children():
        if cursor.kind != clang.cindex.CursorKind.FUNCTION_DECL:
            continue

        # make sure that we deal  only with functions with no definition
        if cursor.spelling not in config['functions'] or cursor.spelling in all_definition or \
           cursor.spelling not in all_declaration:
            continue

        # make sure that we deal with every function only once
        all_declaration.remove(cursor.spelling)

        with open(cursor.location.file.name, 'r', encoding='latin-1') as file:
            start = cursor.extent.start.offset
            end = cursor.extent.end.offset
            declaration = file.read()[start:end]

        arg_types = [arg.type.spelling for arg in cursor.get_arguments()]
        arg_names = [arg.spelling for arg in cursor.get_arguments()]
        return_type = config['functions'][cursor.spelling].get(
            'return_type', config['return_type'])
        not_found_error = config['functions'][cursor.spelling].get(
            'not_found_error', config['not_found_error'])
        implementation = function_impl.format(declaration, cursor.spelling, ', '.join(arg_types),
                                              ', '.join(arg_names), return_type=return_type,
                                              not_found_error=not_found_error,)

        args.output.write(implementation)


if __name__ == '__main__':
    main()
