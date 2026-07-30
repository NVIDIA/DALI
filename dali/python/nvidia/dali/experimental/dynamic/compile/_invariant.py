# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial
from types import MethodType
from typing import Any, ClassVar, Generic, TypeGuard, TypeVar, cast

from typing_extensions import TypeVarTuple, Unpack

_T = TypeVar("_T")
_Ts = TypeVarTuple("_Ts")


def invariant(value: _T) -> _T:
    """Mark `value` as invariant for transparent pipelining.

    The marker is an unchecked promise that the value represented by the returned object will not
    change between compiled iterations. It lets transparent pipelining capture values whose
    stability cannot be proven from source, such as module globals.

    During compiled replay, a corresponding present argument must remain marked; removing the
    marker raises ``RuntimeError``.

    The returned object is a proxy propagating the invariant property to attributes, while
    protocol operations, calls, indexing, iteration, conversions, and method calls return their
    ordinary results.

    Proxy transparency is best-effort: identity, exact type and C-API checks, native layout and
    buffers, copying, serialization, metaclass-only protocols, and special methods supplied by
    non-callable descriptors are not preserved.

    Parameters
    ----------
    value
        Any Python value to mark as invariant.

    Returns
    -------
    Same as `value`
        A proxy for `value`, or `value` itself if it is already an invariant proxy.
    """
    if is_invariant(value):
        return cast(_T, value)

    wrapped_type = type(value)
    proxy_type = _proxy_types.get(wrapped_type)
    if proxy_type is None:
        proxy_type = _proxy_types.setdefault(wrapped_type, _make_proxy_type(wrapped_type))
    return cast(_T, proxy_type(value))


def is_dunder(name: str) -> bool:
    return len(name) > 4 and name.startswith("__") and name.endswith("__")


class _InvariantProxy(Generic[_T]):
    __slots__ = ("__value",)
    _wrapped_type: ClassVar[type]

    def __init__(self, value: _T) -> None:
        object.__setattr__(self, "_InvariantProxy__value", value)

    def __getattribute__(self, name: str) -> Any:
        value = object.__getattribute__(self, "_InvariantProxy__value")
        if is_dunder(name):
            return getattr(value, name)
        attribute = getattr(value, name)
        if isinstance(attribute, MethodType) and attribute.__self__ is value:
            return MethodType(attribute.__func__, self)
        return invariant(attribute)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(object.__getattribute__(self, "_InvariantProxy__value"), name, value)

    def __delattr__(self, name: str) -> None:
        delattr(object.__getattribute__(self, "_InvariantProxy__value"), name)


_proxy_types: dict[type, type[_InvariantProxy[Any]]] = {}


def is_invariant(value: object) -> TypeGuard[_InvariantProxy[Any]]:
    return isinstance(value, _InvariantProxy)


def unwrap_invariant(value: _T | _InvariantProxy[_T]) -> _T:
    if is_invariant(value):
        # Resolve Python's private name mangling
        return object.__getattribute__(value, "_InvariantProxy__value")
    return cast(_T, value)


def unwrap_invariant_args(*values: Unpack[_Ts]) -> tuple[Unpack[_Ts]]:
    return cast("tuple[Unpack[_Ts]]", tuple(unwrap_invariant(value) for value in values))


def unwrap_invariants(value: Any) -> Any:
    """Recursively unwrap exact built-in containers, reusing unchanged containers."""
    value = unwrap_invariant(value)
    if isinstance(value, type) and issubclass(value, _InvariantProxy):
        return value._wrapped_type

    type_ = type(value)
    if type_ not in (list, tuple, dict):
        return value

    items = tuple(value.items()) if type_ is dict else value
    unwrapped_items = tuple(unwrap_invariants(item) for item in items)
    return value if all(a is b for a, b in zip(items, unwrapped_items)) else type_(unwrapped_items)


def _is_descriptor(value: Any) -> bool:
    for base in type.__getattribute__(type(value), "__mro__"):
        attributes = type.__getattribute__(base, "__dict__")
        if "__get__" in attributes:
            return attributes["__get__"] is not None
    return False


class _DunderForwarder:
    """Forward a callable special method from the wrapped type to its invariant proxy."""

    __slots__ = ("_name", "_definition", "_wrapped_type")

    def __init__(self, name: str, definition: Any, wrapped_type: type) -> None:
        self._name = name
        self._definition = definition
        self._wrapped_type = wrapped_type

    def __get__(self, proxy: _InvariantProxy[Any] | None, owner: type | None = None) -> Any:
        return self if proxy is None else MethodType(self, proxy)

    def __call__(self, proxy: _InvariantProxy[Any], *args: Any, **kwargs: Any) -> Any:
        value = unwrap_invariant(proxy)
        # Descriptor binding uses None for class access, not for the None singleton.
        if value is None:
            method = partial(self._definition, None)
        elif _is_descriptor(self._definition):
            # Bypass the descriptor instance's attribute hooks.
            method = object.__getattribute__(self._definition, "__get__")(value, self._wrapped_type)
        else:
            method = self._definition

        if self._name == "__call__":
            return method(*args, **kwargs)
        if self._name in ("__setitem__", "__set__"):
            target, value = args
            return method(unwrap_invariants(target), value)

        return method(
            *(unwrap_invariants(arg) for arg in args),
            **{name: unwrap_invariants(arg) for name, arg in kwargs.items()},
        )


_EXCLUDED_DUNDERS = {
    # The proxy owns attribute delegation.
    "__delattr__",
    "__getattr__",
    "__getattribute__",
    "__setattr__",
    # Construction and finalization apply to the proxy itself.
    "__del__",
    "__init__",
    "__new__",
    # Class hooks apply to the generated proxy type.
    "__class_getitem__",
    "__init_subclass__",
    "__mro_entries__",
    "__prepare__",
    "__subclasshook__",
    # The proxy keeps its own identity, layout, and metadata.
    "__annotations__",
    "__base__",
    "__bases__",
    "__class__",
    "__dict__",
    "__doc__",
    "__module__",
    "__mro__",
    "__name__",
    "__qualname__",
    "__slots__",
    "__weakref__",
    # Copying and serialization cannot preserve the marker reliably.
    "__copy__",
    "__deepcopy__",
    "__getinitargs__",
    "__getnewargs__",
    "__getnewargs_ex__",
    "__getstate__",
    "__reduce__",
    "__reduce_ex__",
    "__replace__",
    "__setstate__",
}


def _make_proxy_type(wrapped_type: type) -> type[_InvariantProxy[Any]]:
    # Build a namespace to forward dunders to the new type.
    # Regular attributes and methods are forwarded by _InvariantProxy directly.
    namespace = {
        "__module__": __name__,
        "__slots__": (),
        "_wrapped_type": wrapped_type,
    }
    definitions: dict[str, Any] = {}
    for base in type.__getattribute__(wrapped_type, "__mro__"):
        for name, definition in type.__getattribute__(base, "__dict__").items():
            if is_dunder(name) and name not in _EXCLUDED_DUNDERS:
                definitions.setdefault(name, definition)

    for name, definition in definitions.items():
        if definition is None:
            namespace[name] = None
        elif callable(definition):
            namespace[name] = _DunderForwarder(name, definition, wrapped_type)

    name = f"_Invariant_{type.__getattribute__(wrapped_type, '__name__')}"
    return type(name, (_InvariantProxy,), namespace)
