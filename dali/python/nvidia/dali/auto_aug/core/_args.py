# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect


class MissingArgException(Exception):
    def __init__(self, message, augmentation, missing_args):
        super().__init__(message)
        self.augmentation = augmentation
        self.missing_args = missing_args


class UnusedArgException(Exception):
    def __init__(self, message, unused_args):
        super().__init__(message)
        self.unused_args = unused_args


def filter_extra_accepted_kwargs(fun, kwargs, skip_positional=0):
    """
    Returns sub-dict of `kwargs` with the keys that match the
    names of arguments in `fun`'s signature.
    """
    sig = inspect.signature(fun)
    # the params from signature with up to skip_positional filtered out
    # (less only if there is not enough of positional args)
    params = [
        (name, param)
        for i, (name, param) in enumerate(sig.parameters.items())
        if i >= skip_positional
        or param.kind
        not in [inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.POSITIONAL_ONLY]
    ]
    extra = [
        name
        for (name, param) in params
        if param.kind in [inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY]
    ]
    return {name: value for name, value in kwargs.items() if name in extra}


def get_required_kwargs(fun, skip_positional=0):
    """
    Returns the list of names of args/kwargs without defaults from
    `fun` signature.
    """
    sig = inspect.signature(fun)
    # the params from signature with up to skip_positional filtered out
    # (less only if there is not enough of positional args)
    params = [
        (name, param)
        for i, (name, param) in enumerate(sig.parameters.items())
        if i >= skip_positional
        or param.kind
        not in [inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.POSITIONAL_ONLY]
    ]
    return [
        name
        for name, param in params
        if param.default is inspect.Parameter.empty
        and param.kind in [inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY]
    ]


def get_num_positional_args(fun):
    """
    Returns the number of arguments that can be passed positionally to the `fun` call.
    """
    sig = inspect.signature(fun)
    return len(
        [
            name
            for name, param in sig.parameters.items()
            if param.kind
            in [inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.POSITIONAL_ONLY]
        ]
    )


def get_missing_kwargs(fun, kwargs, skip_positional=0):
    required = get_required_kwargs(fun, skip_positional=skip_positional)
    return [name for name in required if name not in kwargs]


def filter_unused_args(augmentations, kwargs):
    used_kwargs = set(
        kwarg_name
        for augment in augmentations
        for kwarg_name in filter_extra_accepted_kwargs(augment.op, kwargs, 2)
    )
    return [kwarg_name for kwarg_name in kwargs if kwarg_name not in used_kwargs]


def forbid_unused_kwargs(augmentations, kwargs, call_name):
    unused_args = filter_unused_args(augmentations, kwargs)
    if unused_args:
        subject, verb = ("kwarg", "is") if len(unused_args) == 1 else ("kwargs", "are")
        unused_kwargs_str = ", ".join(unused_args)
        raise UnusedArgException(
            f"The {call_name} got unexpected {subject}. "
            f"The {subject} `{unused_kwargs_str}` {verb} not used by any of the augmentations.",
            unused_args=unused_args,
        )
