#  Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import ast
import inspect
import re

def _intersection_of_lists(l1, l2):
    return list(set(l1) & set(l2))


class DaliPipelineAstAnalyzer(ast.NodeVisitor):
    # TODO visit recursively to traverse auxiliary functions
    def __init__(self):
        self._initialized = False
        self._dali_operator_input_names = []
        self._dali_pipeline_input_names = []
        self._dali_pipeline_def_args = []

    def get_pipeline_input_names(self):
        if not self._initialized:
            self._initialize()
        return self._dali_pipeline_input_names

    def is_pipeline_input(self, arg_name):
        if not self._initialized:
            self._initialize()
        return arg_name in self.get_pipeline_input_names()

    def _initialize(self):
        self._determine_pipeline_input_names()
        self._initialized = True

    def _is_dali_operator(self, func):
        #TODO FIGURE THIS FUNCTION OUT
        import nvidia.dali.fn as fn
        fn_full_name = self._unroll_signature(func)
        fn_object = eval(fn_full_name)  # TODO Is there a better way to obtain this?
        return re.match(r'nvidia.dali.fn', fn_object.__module__) is not None

    def _unroll_signature(self, func):
        if isinstance(func, ast.Attribute):
            return self._unroll_signature(func.value) + '.' + func.attr
        elif isinstance(func, ast.Name):
            return func.id

    def _determine_pipeline_input_names(self):
        self._dali_pipeline_input_names = _intersection_of_lists(self._dali_operator_input_names,
                                                                 self._dali_pipeline_def_args)

    def visit_FunctionDef(self, node):
        self.generic_visit(node)  # So that the children of this node will be visited too.
        self._dali_pipeline_def_args = [x.arg for x in node.args.args]

    def visit_Call(self, node):
        if not self._is_dali_operator(node.func):
            return
        for arg in node.args:
            if isinstance(arg, ast.Name):
                self._dali_operator_input_names.append(arg.id)


def _generate_fn_external_source(input_name):
    expr = f"""
{input_name} = fn.external_source(
    name='{input_name}', 
    cycle=False, 
    cuda_stream=1, 
    use_copy_kernel=False, 
    blocking=False, 
    no_copy=True, 
    batch=True, 
    batch_info=False, 
    parallel=False)
"""
    return ast.parse(source=expr).body[0]


def pprint(ast):
    import astpretty
    astpretty.pprint(ast, show_offsets=False, indent=2)


def augment_pipeline_def(pipeline_def):
    fsrc = inspect.getsource(pipeline_def)
    tree = ast.parse(source=fsrc)
    analyzer = DaliPipelineAstAnalyzer()
    analyzer.visit(tree)

    # Add `fn.external_source` nodes for every input to the pipeline.
    tree.body[0].body = [_generate_fn_external_source(iname) for iname in
                         analyzer.get_pipeline_input_names()] + tree.body[0].body

    # Add temporary import
    # TODO remove
    tree.body[0].body = [ast.parse('import nvidia.dali.fn as fn').body[0]] + tree.body[0].body

    # Remove obsolete arguments from function signature.
    tree.body[0].args.args = list(
        filter(lambda x: not analyzer.is_pipeline_input(x.arg), tree.body[0].args.args))

    # Remove the @pipeline_def decorator - it's obsolete at this point.
    # TODO ?remove?
    tree.body[0].decorator_list = list(
        filter(lambda x: x.id != 'pipeline_def', tree.body[0].decorator_list))

    # Add a call to this function so that it can be called.
    # tree.body.append(ast.Call())
    # print(ast.unparse(ast_obj=tree))
    pprint(tree)

    return tree


def get_function_from_ast(tree: ast.Module, fn_name: str):
    nm = {}
    # To obtain a function object from the ast.Module, we're compiling AST to a code object
    # and executing it with local namespace.
    exec(compile(tree, filename='<string>', mode='exec'), nm)
    return nm[fn_name]

