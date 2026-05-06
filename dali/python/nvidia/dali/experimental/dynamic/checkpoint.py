# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Checkpointing support for DALI dynamic mode.

The dynamic mode pipelines are stateful by way of two operator categories:

* :class:`~nvidia.dali.experimental.dynamic._ops.Reader` instances - they hold
  the iteration position over the dataset.
* :class:`~nvidia.dali.experimental.dynamic.random.RNG` instances - they hold
  the random number generator state.

The :class:`Checkpoint` class collects the state of a chosen subset of these
objects so that processing can be resumed at the captured point. See
:func:`current` for a thread-local, ``EvalContext``-bound default checkpoint.
"""

from __future__ import annotations

import glob
import json
import os
import re
import string
from typing import Any, Optional

from . import _eval_context

__all__ = ["Checkpoint", "current"]


# Format-string version. Bumped if the on-disk layout becomes incompatible.
_CHECKPOINT_FORMAT_VERSION = 1


class _Sentinel:
    """A unique marker used to signal "no name was provided"."""


_NO_NAME = _Sentinel()


def _pattern_to_regex(pattern: str) -> re.Pattern:
    """Translates a Python format string with a ``{seq}`` field into a regex.

    The regex captures the value of ``seq`` as group ``seq``. Other format fields
    are not supported.
    """
    parts = []
    parts.append("^")
    seq_seen = False
    for literal, field_name, format_spec, _conv in string.Formatter().parse(pattern):
        parts.append(re.escape(literal))
        if field_name is None:
            continue
        if field_name != "seq":
            raise ValueError(
                f"Unsupported field {{{field_name}}} in checkpoint pattern. "
                "Only `{seq}` (with an optional format spec) is allowed."
            )
        if seq_seen:
            raise ValueError("`{seq}` may appear at most once in a checkpoint pattern.")
        seq_seen = True
        # Allow the digits to be padded as a numeric format spec might require, but
        # we don't try to interpret the format spec - any sequence of digits matches.
        parts.append(r"(?P<seq>\d+)")
    parts.append("$")
    if not seq_seen:
        raise ValueError("Checkpoint pattern must include `{seq}`.")
    return re.compile("".join(parts))


def _matching_files(pattern: str):
    """Yields ``(seq, path)`` pairs for paths in the filesystem matching pattern."""
    # Build a glob equivalent by replacing `{seq...}` with `*`.
    glob_parts = []
    seq_seen = False
    for literal, field_name, _format_spec, _conv in string.Formatter().parse(pattern):
        glob_parts.append(literal)
        if field_name is None:
            continue
        if field_name != "seq":
            raise ValueError(
                f"Unsupported field {{{field_name}}} in checkpoint pattern. "
                "Only `{seq}` (with an optional format spec) is allowed."
            )
        if seq_seen:
            raise ValueError("`{seq}` may appear at most once in a checkpoint pattern.")
        seq_seen = True
        glob_parts.append("*")
    if not seq_seen:
        raise ValueError("Checkpoint pattern must include `{seq}`.")
    glob_pattern = "".join(glob_parts)
    regex = _pattern_to_regex(pattern)
    for path in glob.iglob(glob_pattern):
        m = regex.match(path)
        if m is None:
            continue
        yield int(m.group("seq")), path


class Checkpoint:
    """Aggregates the state of stateful objects (readers and RNGs) for resume.

    A checkpoint stores a mapping from a key (operator name) to that operator's
    serialized state. Objects can be added with :meth:`register`, after which
    their state can be collected with :meth:`collect` or restored with
    :meth:`restore`. The serialized representation can be persisted with
    :meth:`save` and loaded back with :meth:`load`.

    The supported objects are:

    * :class:`~nvidia.dali.experimental.dynamic._ops.Reader` instances
      (the ``ndd.readers.*`` operators)
    * :class:`~nvidia.dali.experimental.dynamic.random.RNG` instances

    Both expose ``get_state`` and ``set_state`` methods, which is the duck-typed
    contract used by this class.

    Examples
    --------
    >>> import nvidia.dali.experimental.dynamic as ndd
    >>>
    >>> reader = ndd.readers.File(file_root="...")
    >>> rng = ndd.random.RNG(seed=42)
    >>>
    >>> ckpt = ndd.checkpoint.Checkpoint()
    >>> ckpt.register(reader, "reader")
    >>> ckpt.register(rng, "rng")
    >>>
    >>> # ... iterate for a while ...
    >>>
    >>> ckpt.collect()
    >>> ckpt.save("ckpt_{seq:04d}.json")
    """

    def __init__(self):
        # name -> registered op instance
        self._ops: dict[str, Any] = {}
        # name -> serialized state string (`str` for both readers and RNGs)
        self._states: dict[str, str] = {}
        # set of names whose state in `self._states` is "dirty" - has been set but
        # not yet propagated to the corresponding op via `op.set_state`.
        self._dirty: set[str] = set()
        # True after a successful :meth:`collect` (and cleared by :meth:`load`).
        self._complete: bool = False
        # True after a successful :meth:`load` (and cleared by :meth:`collect`).
        self._loaded: bool = False
        # Counter for auto-generated keys.
        self._seq_counter: int = 0
        # Counter used by :meth:`save` to pick a sequential file name. Initialized
        # lazily on first save by scanning the filesystem.
        self._save_seq: Optional[int] = None

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Resets the checkpoint to its initial state.

        Drops all registered ops, all stored states, and resets the ``complete``
        and ``loaded`` flags as well as the auto-generated key counter.
        """
        self._ops.clear()
        self._states.clear()
        self._dirty.clear()
        self._complete = False
        self._loaded = False
        self._seq_counter = 0
        self._save_seq = None

    @property
    def is_complete(self) -> bool:
        """``True`` if :meth:`collect` was the last operation to populate the state."""
        return self._complete

    @property
    def is_loaded(self) -> bool:
        """``True`` if :meth:`load` (or :meth:`deserialize`) populated the state."""
        return self._loaded

    @property
    def names(self) -> tuple[str, ...]:
        """Names of all currently registered ops."""
        return tuple(self._ops.keys())

    def __len__(self) -> int:
        return len(self._ops)

    def __contains__(self, name) -> bool:
        return str(name) in self._ops

    # ------------------------------------------------------------------
    # Registration / state access
    # ------------------------------------------------------------------

    def register(self, op, name=_NO_NAME) -> str:
        """Adds a stateful op to the checkpoint.

        Parameters
        ----------
        op : Reader or RNG
            The stateful object to register. Must expose ``get_state`` /
            ``set_state`` methods.
        name : str, optional
            The key under which to store the op. If omitted, a sequential numeric
            key is generated, unless ``op`` was already registered (under any
            name), in which case its existing key is returned. If ``name`` is
            provided, any existing op under that key is replaced.

        Returns
        -------
        str
            The key under which ``op`` is registered.

        Notes
        -----
        If a state is currently associated with ``name`` and is marked dirty
        (e.g. it came from a recent :meth:`load`), the state is immediately
        applied to ``op`` via ``op.set_state`` and the entry is marked clean.
        """
        if name is _NO_NAME:
            # Anonymous registration - reverse-lookup the op first so that
            # registering the same op twice is a no-op.
            for key, registered_op in self._ops.items():
                if registered_op is op:
                    self._maybe_apply(key)
                    return key
            if self._complete:
                raise RuntimeError(
                    "Cannot register a new op into a checkpoint that has been "
                    "completed via `collect`. Call `clear` first."
                )
            # Pick the next free sequential key.
            while True:
                key = str(self._seq_counter)
                self._seq_counter += 1
                if key not in self._ops:
                    break
            if self._loaded and key not in self._states:
                raise KeyError(
                    f"Loaded checkpoint has no state for inferred key {key!r}. "
                    "If your registration order changed since the checkpoint was "
                    "saved, use the named form `register(op, name)` instead."
                )
            self._ops[key] = op
            self._maybe_apply(key)
            return key

        key = str(name)
        if key in self._ops:
            # Existing entry - replace and apply pending state if any.
            self._ops[key] = op
            self._maybe_apply(key)
            return key

        # New named entry.
        if self._complete:
            raise RuntimeError(
                f"Cannot register a new op {key!r} into a completed checkpoint. "
                "Call `clear` first."
            )
        if self._loaded:
            if key not in self._states:
                raise KeyError(
                    f"Checkpoint was loaded but does not contain a state for {key!r}."
                )
        self._ops[key] = op
        self._maybe_apply(key)
        return key

    def _maybe_apply(self, key: str) -> None:
        if key in self._dirty:
            self._ops[key].set_state(self._states[key])
            self._dirty.discard(key)

    def get_state(self, name) -> Optional[str]:
        """Returns the stored state for the op registered under ``name``.

        Parameters
        ----------
        name : str
            The key under which the op is registered.

        Returns
        -------
        str or None
            The serialized state string, or ``None`` if no state has been
            collected/loaded for that key yet.

        Raises
        ------
        KeyError
            If no op is registered under ``name``.
        """
        key = str(name)
        if key not in self._ops:
            raise KeyError(f"No op registered under {key!r}.")
        return self._states.get(key)

    def set_state(self, name, value) -> None:
        """Manually sets the state for an already-registered op.

        Parameters
        ----------
        name : str
            The key under which the op is registered.
        value
            The state value. Will be stringified via ``str(value)``. Strings are
            stored verbatim.

        Raises
        ------
        KeyError
            If no op is registered under ``name``.

        Notes
        -----
        The state is marked *dirty*. It is applied to the op the next time
        :meth:`register` is called for that key, or when :meth:`restore` is
        invoked.
        """
        key = str(name)
        if key not in self._ops:
            raise KeyError(f"No op registered under {key!r}.")
        self._states[key] = value if isinstance(value, str) else str(value)
        self._dirty.add(key)

    # ------------------------------------------------------------------
    # Collect / restore
    # ------------------------------------------------------------------

    def collect(self) -> None:
        """Collects the state of every registered op into the checkpoint.

        Sets the ``complete`` flag, clears the ``loaded`` flag, and marks all
        states as clean. Verifies that the state dictionary does not contain any
        keys without a corresponding registered op.
        """
        extra = set(self._states.keys()) - set(self._ops.keys())
        if extra:
            raise RuntimeError(
                "Checkpoint state has entries with no corresponding registered "
                f"ops: {sorted(extra)}."
            )
        new_states: dict[str, str] = {}
        for key, op in self._ops.items():
            state = op.get_state()
            new_states[key] = state if isinstance(state, str) else str(state)
        self._states = new_states
        self._dirty.clear()
        self._complete = True
        self._loaded = False

    def restore(self) -> None:
        """Restores the state of every registered op from the checkpoint.

        Intended for manual use - typically you'd register ops one by one and
        rely on :meth:`register` to apply the state implicitly. This method
        applies all dirty states at once.

        Raises
        ------
        RuntimeError
            If the state dictionary contains entries with no corresponding
            registered ops, or if any registered op is missing a state in the
            dictionary (unless the dictionary is empty, in which case this is a
            no-op).
        """
        if not self._states:
            return
        extra = set(self._states.keys()) - set(self._ops.keys())
        if extra:
            raise RuntimeError(
                "Checkpoint state has entries with no corresponding registered "
                f"ops: {sorted(extra)}."
            )
        missing = set(self._ops.keys()) - set(self._states.keys())
        if missing:
            raise RuntimeError(
                "Checkpoint state is incomplete - missing entries for "
                f"{sorted(missing)}."
            )
        for key in list(self._dirty):
            self._ops[key].set_state(self._states[key])
        self._dirty.clear()

    # ------------------------------------------------------------------
    # (De)serialization and persistence
    # ------------------------------------------------------------------

    def serialize(self) -> str:
        """Serializes the checkpoint state dictionary to a JSON string.

        Raises
        ------
        RuntimeError
            If the state dictionary is empty.
        """
        if not self._states:
            raise RuntimeError(
                "Cannot serialize an empty checkpoint. Call `collect` (or "
                "`set_state`) first."
            )
        payload = {
            "version": _CHECKPOINT_FORMAT_VERSION,
            "states": self._states,
        }
        return json.dumps(payload)

    def deserialize(self, data: str) -> None:
        """Replaces the checkpoint state dictionary with a deserialized version.

        Parameters
        ----------
        data : str
            A string previously produced by :meth:`serialize`.

        Notes
        -----
        Sets the ``loaded`` flag and clears the ``complete`` flag. Marks every
        loaded entry as dirty so that subsequent :meth:`register` calls (or
        :meth:`restore`) will apply the state.
        """
        payload = json.loads(data)
        if not isinstance(payload, dict) or "states" not in payload:
            raise ValueError("Invalid checkpoint payload.")
        version = payload.get("version", 0)
        if version != _CHECKPOINT_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported checkpoint format version {version}; expected "
                f"{_CHECKPOINT_FORMAT_VERSION}."
            )
        states = payload["states"]
        if not isinstance(states, dict):
            raise ValueError("Invalid checkpoint payload: `states` must be a dict.")
        self._states = {str(k): str(v) for k, v in states.items()}
        self._dirty = set(self._states.keys())
        self._complete = False
        self._loaded = True

    def save(self, pattern: str) -> str:
        """Serializes the checkpoint and writes it to a file.

        Parameters
        ----------
        pattern : str
            A Python format string containing ``{seq}`` (e.g. ``"ckpt_{seq:04d}.json"``).
            The placeholder is replaced with a sequential number that does not
            collide with any existing file.

        Returns
        -------
        str
            The path of the file that was written.

        Raises
        ------
        RuntimeError
            If the state dictionary is empty.
        """
        seq = self._next_save_seq(pattern)
        path = pattern.format(seq=seq)
        # Avoid overwriting an existing file - bump the counter if one appeared in
        # the meantime.
        while os.path.exists(path):
            seq += 1
            path = pattern.format(seq=seq)
        data = self.serialize()
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(data)
        self._save_seq = seq + 1
        return path

    def load(self, pattern: str) -> str:
        """Reads a previously saved checkpoint from a file.

        Parameters
        ----------
        pattern : str
            A Python format string containing ``{seq}`` (e.g. ``"ckpt_{seq:04d}.json"``).
            If multiple files match, the one with the highest sequence number
            is loaded.

        Returns
        -------
        str
            The path of the file that was read.

        Raises
        ------
        FileNotFoundError
            If no file matching the pattern exists.

        Notes
        -----
        Marks all loaded entries as dirty (see :meth:`deserialize`).
        """
        candidates = list(_matching_files(pattern))
        if not candidates:
            raise FileNotFoundError(f"No checkpoint file matches pattern {pattern!r}.")
        candidates.sort(key=lambda x: x[0])
        seq, path = candidates[-1]
        with open(path, "r", encoding="utf-8") as f:
            data = f.read()
        self.deserialize(data)
        self._save_seq = seq + 1
        return path

    def _next_save_seq(self, pattern: str) -> int:
        """Returns the next sequence number to use for :meth:`save`."""
        if self._save_seq is not None:
            return self._save_seq
        # First save - scan the filesystem so we don't clobber existing files.
        max_existing = -1
        for seq, _ in _matching_files(pattern):
            if seq > max_existing:
                max_existing = seq
        return max_existing + 1


def current() -> Checkpoint:
    """Returns the :class:`Checkpoint` bound to the current :class:`EvalContext`."""
    return _eval_context.EvalContext.current().checkpoint
