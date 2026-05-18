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

import json
import os
import tempfile

import numpy as np
import nvidia.dali.experimental.dynamic as ndd
from nose_utils import assert_raises
from nose2.tools import params
from test_utils import get_dali_extra_path

_DALI_EXTRA = get_dali_extra_path()
_FILE_ROOT = os.path.join(_DALI_EXTRA, "db", "single", "jpeg")
_FILE_LIST = os.path.join(_FILE_ROOT, "image_list.txt")


def _make_reader(**kwargs):
    """Helper that constructs a deterministic File reader for the tests."""
    return ndd.readers.File(
        file_root=_FILE_ROOT,
        file_list=_FILE_LIST,
        random_shuffle=False,
        **kwargs,
    )


def _labels_of(batch):
    """Returns the integer labels of a (jpeg, label) batch as a list."""
    _, lbl = batch
    return [int(np.array(ndd.as_tensor(t, device="cpu"))[0]) for t in lbl.tensors]


# ---------------------------------------------------------------------------
# RNG manual get_state / set_state
# ---------------------------------------------------------------------------


def test_rng_get_set_state_methods():
    """RNG exposes get_state/set_state mirroring the `state` property."""
    rng = ndd.random.RNG(seed=1234)
    state1 = rng.get_state()
    # `state` and `get_state()` must agree.
    assert str(state1) == str(rng.state)

    # Advance.
    v_pre = [rng() for _ in range(5)]

    # Restore via set_state with a state object.
    rng.set_state(state1)
    v_post = [rng() for _ in range(5)]
    assert v_pre == v_post

    # Restore via set_state with the string representation.
    rng.set_state(str(state1))
    v_str = [rng() for _ in range(5)]
    assert v_pre == v_str


# ---------------------------------------------------------------------------
# Reader manual get_state / set_state
# ---------------------------------------------------------------------------


def test_reader_get_state_before_iteration_errors():
    """get_state must error if the reader has not been initialized yet."""
    reader = _make_reader()
    with assert_raises(RuntimeError, glob="before its first iteration"):
        reader.get_state()


def test_reader_state_str_roundtrip():
    """The state object's __str__ is a serialized form usable with set_state."""
    reader = _make_reader(enable_checkpointing=True)
    it = reader.next_epoch(batch_size=4)
    next(it)
    state = reader.get_state()
    s = str(state)
    assert isinstance(s, str)
    assert len(s) > 0
    # set_state must accept either the object or its string representation.
    # It must be called on a fresh reader (before the first iteration).
    reader2 = _make_reader()
    reader2.set_state(state)
    reader3 = _make_reader()
    reader3.set_state(s)


def test_reader_get_set_state_resume():
    """set_state on a fresh reader resumes from the captured iteration position."""
    # Reference reader, iterate through and capture state after the first batch.
    ref = _make_reader(enable_checkpointing=True)
    it_ref = ref.next_epoch(batch_size=4)
    next(it_ref)  # discard first batch
    state = ref.get_state()
    expected = _labels_of(next(it_ref))

    # Restored reader: set_state before the first iteration so that the state
    # is buffered and applied on backend creation.
    restored = _make_reader()
    # The backend hasn't been initialized - this should enable checkpointing
    restored.set_state(state)
    it_restored = restored.next_epoch(batch_size=4)
    got = _labels_of(next(it_restored))
    assert got == expected, f"Restored reader produced {got} instead of {expected}."


def test_reader_set_state_after_iteration_errors():
    """set_state after iteration has begun raises a clear error.

    The underlying C++ reader can only restore a checkpoint before its prefetch
    thread starts. Calling :meth:`set_state` after the first batch must therefore
    fail loudly rather than silently misbehave.
    """
    reader = _make_reader(enable_checkpointing=True)
    next(reader.next_epoch(batch_size=4))
    state = reader.get_state()
    with assert_raises(RuntimeError, glob="after iteration has begun"):
        reader.set_state(state)


def test_reader_set_state_invalid_type():
    """set_state rejects unsupported argument types."""
    reader = _make_reader()
    with assert_raises(TypeError, glob="ReaderState"):
        reader.set_state(42)


# ---------------------------------------------------------------------------
# Checkpoint - registration semantics
# ---------------------------------------------------------------------------


def test_checkpoint_register_named():
    """Named registration returns the supplied key."""
    ckpt = ndd.checkpoint.Checkpoint()
    rng = ndd.random.RNG(seed=1)
    key = ckpt.register(rng, "my_rng")
    assert key == "my_rng"
    assert "my_rng" in ckpt
    assert ckpt.names == ("my_rng",)


def test_checkpoint_register_anonymous_returns_sequential_keys():
    """Anonymous registration returns "0", "1", ... ."""
    ckpt = ndd.checkpoint.Checkpoint()
    rng1 = ndd.random.RNG(seed=1)
    rng2 = ndd.random.RNG(seed=2)
    assert ckpt.register(rng1) == "__op_0"
    assert ckpt.register(rng2) == "__op_1"


def test_checkpoint_register_anonymous_idempotent():
    """Re-registering the same op anonymously returns the existing key."""
    ckpt = ndd.checkpoint.Checkpoint()
    rng = ndd.random.RNG(seed=1)
    key = ckpt.register(rng)
    assert ckpt.register(rng) == key
    assert len(ckpt) == 1


def test_checkpoint_register_named_replaces():
    """Named registration replaces a previous entry at the same key."""
    ckpt = ndd.checkpoint.Checkpoint()
    rng1 = ndd.random.RNG(seed=1)
    rng2 = ndd.random.RNG(seed=2)
    ckpt.register(rng1, "rng")
    ckpt.register(rng2, "rng")
    assert len(ckpt) == 1


def test_checkpoint_get_state_unregistered_errors():
    """get_state raises KeyError if the name was not registered."""
    ckpt = ndd.checkpoint.Checkpoint()
    with assert_raises(KeyError):
        ckpt.get_state("unknown")


def test_checkpoint_get_state_no_state_yet_returns_none():
    """get_state returns None if no state has been collected/loaded yet."""
    ckpt = ndd.checkpoint.Checkpoint()
    rng = ndd.random.RNG(seed=1)
    ckpt.register(rng, "rng")
    assert ckpt.get_state("rng") is None


def test_checkpoint_set_state_unregistered_errors():
    """set_state raises KeyError if the name was not registered."""
    ckpt = ndd.checkpoint.Checkpoint()
    rng1 = ndd.random.RNG(seed=1)
    ckpt.register(rng1)
    ckpt.collect()  # it's complete now
    with assert_raises(KeyError):
        ckpt.set_state("unknown2", "...")


# ---------------------------------------------------------------------------
# Checkpoint - collect / restore
# ---------------------------------------------------------------------------


def test_checkpoint_collect_sets_complete():
    """collect populates the state dictionary and sets is_complete."""
    ckpt = ndd.checkpoint.Checkpoint()
    rng = ndd.random.RNG(seed=1)
    ckpt.register(rng, "rng")
    ckpt.collect()
    assert ckpt.is_complete
    assert not ckpt.is_loaded
    assert ckpt.get_state("rng") is not None


def test_checkpoint_collect_extra_state_errors():
    """collect raises if the state dict has entries with no registered op."""
    ckpt = ndd.checkpoint.Checkpoint()
    # Manually inject a stray key.
    ckpt._states["ghost"] = "..."
    with assert_raises(RuntimeError, glob="no corresponding registered ops"):
        ckpt.collect()


def test_checkpoint_restore_empty_is_noop():
    """restore on an empty state dict is a no-op."""
    ckpt = ndd.checkpoint.Checkpoint()
    rng = ndd.random.RNG(seed=1)
    ckpt.register(rng, "rng")
    ckpt.restore()  # should not raise


def test_checkpoint_restore_missing_state_errors():
    """restore raises if a registered op has no state in the dictionary."""
    ckpt = ndd.checkpoint.Checkpoint()
    rng1 = ndd.random.RNG(seed=1)
    rng2 = ndd.random.RNG(seed=2)
    ckpt.register(rng1, "a")
    ckpt.register(rng2, "b")
    # Manually populate state for only one of the two registered ops.
    ckpt.set_state("a", str(rng1.get_state()))
    with assert_raises(RuntimeError, glob="incomplete"):
        ckpt.restore()


def test_checkpoint_restore_extra_state_errors():
    """restore raises if the state dict has entries with no matching op."""
    ckpt = ndd.checkpoint.Checkpoint()
    ckpt._states["ghost"] = "..."
    ckpt._dirty.add("ghost")
    with assert_raises(RuntimeError, glob="no corresponding registered ops"):
        ckpt.restore()


def test_checkpoint_collect_restore_rng_roundtrip():
    """A captured RNG state can be restored to reproduce its sequence."""
    rng = ndd.random.RNG(seed=42)
    rng()  # advance a bit
    rng()
    ckpt = ndd.checkpoint.Checkpoint()
    ckpt.register(rng, "rng")
    ckpt.collect()
    expected = [rng() for _ in range(5)]

    rng2 = ndd.random.RNG(seed=999)  # different seed - state will be overwritten
    ckpt2 = ndd.checkpoint.Checkpoint()
    ckpt2.deserialize(ckpt.serialize())
    ckpt2.register(rng2, "rng")  # state applied implicitly here
    got = [rng2() for _ in range(5)]
    assert got == expected


def test_checkpoint_collect_restore_reader_roundtrip():
    """A captured Reader state can be restored to resume iteration."""
    ref = _make_reader()
    ckpt = ndd.checkpoint.Checkpoint()
    ckpt.register(ref, "reader")
    it_ref = ref.next_epoch(batch_size=4)
    next(it_ref)
    ckpt.collect()
    expected = _labels_of(next(it_ref))

    restored = _make_reader()
    ckpt2 = ndd.checkpoint.Checkpoint()
    ckpt2.deserialize(ckpt.serialize())
    ckpt2.register(restored, "reader")
    got = _labels_of(next(restored.next_epoch(batch_size=4)))
    assert got == expected


# ---------------------------------------------------------------------------
# Checkpoint - serialize / deserialize
# ---------------------------------------------------------------------------


def test_checkpoint_serialize_empty_errors():
    """serialize raises if the state dictionary is empty."""
    ckpt = ndd.checkpoint.Checkpoint()
    with assert_raises(RuntimeError, glob="empty"):
        ckpt.serialize()


def test_checkpoint_deserialize_marks_loaded():
    """deserialize sets is_loaded and clears is_complete; entries are dirty."""
    ckpt = ndd.checkpoint.Checkpoint()
    rng = ndd.random.RNG(seed=1)
    ckpt.register(rng, "rng")
    ckpt.collect()
    payload = ckpt.serialize()

    ckpt2 = ndd.checkpoint.Checkpoint()
    ckpt2.deserialize(payload)
    assert ckpt2.is_loaded
    assert not ckpt2.is_complete
    assert "rng" in ckpt2._dirty


def test_checkpoint_register_after_load_unknown_key_errors():
    """Registering an op with an unknown key after load() raises KeyError."""
    ckpt = ndd.checkpoint.Checkpoint()
    rng = ndd.random.RNG(seed=1)
    ckpt.register(rng, "rng")
    ckpt.collect()
    payload = ckpt.serialize()

    ckpt2 = ndd.checkpoint.Checkpoint()
    ckpt2.deserialize(payload)
    new_rng = ndd.random.RNG(seed=2)
    with assert_raises(KeyError):
        ckpt2.register(new_rng, "other")


def test_checkpoint_register_anonymous_after_collect_errors():
    """Anonymous registration after collect() raises RuntimeError."""
    ckpt = ndd.checkpoint.Checkpoint()
    rng = ndd.random.RNG(seed=1)
    ckpt.register(rng, "rng")
    ckpt.collect()
    new_rng = ndd.random.RNG(seed=2)
    with assert_raises(RuntimeError, glob="completed"):
        ckpt.register(new_rng)


def test_checkpoint_register_anonymous_after_load_replays_keys():
    """Anonymous registration after load() works as long as the order matches."""
    ref_rng = ndd.random.RNG(seed=1)
    ref_other = ndd.random.RNG(seed=99)

    ckpt = ndd.checkpoint.Checkpoint()
    ckpt.register(ref_rng)  # gets "0"
    ckpt.register(ref_other, "other")  # named
    ckpt.collect()
    payload = ckpt.serialize()

    expected_rng = [ref_rng() for _ in range(3)]
    expected_other = [ref_other() for _ in range(3)]

    new_rng = ndd.random.RNG(seed=12345)
    new_other = ndd.random.RNG(seed=54321)

    ckpt2 = ndd.checkpoint.Checkpoint()
    ckpt2.deserialize(payload)
    # Same registration order: anonymous first, then named.
    ckpt2.register(new_rng)  # picks up sequential key "0"
    ckpt2.register(new_other, "other")
    got_rng = [new_rng() for _ in range(3)]
    got_other = [new_other() for _ in range(3)]
    assert got_rng == expected_rng
    assert got_other == expected_other


def test_checkpoint_register_anonymous_after_load_extra_op_errors():
    """Anonymous registration of an extra op (no matching key) raises KeyError."""
    ckpt = ndd.checkpoint.Checkpoint()
    ckpt.register(ndd.random.RNG(seed=1))  # "0"
    ckpt.collect()
    payload = ckpt.serialize()

    ckpt2 = ndd.checkpoint.Checkpoint()
    ckpt2.deserialize(payload)
    ckpt2.register(ndd.random.RNG(seed=2))  # consumes "0"
    with assert_raises(KeyError, glob="Loaded checkpoint has no state"):
        ckpt2.register(ndd.random.RNG(seed=3))


# ---------------------------------------------------------------------------
# Checkpoint - save / load
# ---------------------------------------------------------------------------


@params("ckpt_{seq:04d}.json", "ckpt_{seq}.bin")
def test_checkpoint_save_load_roundtrip(name_pattern):
    """save writes a file; load returns the most recent one."""
    rng = ndd.random.RNG(seed=42)
    for _ in range(10):
        rng()  # advance
    ckpt = ndd.checkpoint.Checkpoint()
    ckpt.register(rng, "rng")
    ckpt.collect()

    expected = [rng() for _ in range(5)]

    with tempfile.TemporaryDirectory() as d:
        pattern = os.path.join(d, name_pattern)
        p1 = ckpt.save(pattern)
        p2 = ckpt.save(pattern)
        assert os.path.exists(p1)
        assert os.path.exists(p2)
        assert p1 != p2

        # Load picks the most recent (highest seq) entry.
        ckpt2 = ndd.checkpoint.Checkpoint()
        loaded_path = ckpt2.load(pattern)
        assert loaded_path == p2

    rng2 = ndd.random.RNG(seed=999)
    ckpt2.register(rng2, "rng")
    got = [rng2() for _ in range(5)]
    assert got == expected


def test_checkpoint_save_avoids_overwrite_across_instances():
    """Multiple checkpoint instances writing into the same dir produce distinct files."""
    rng = ndd.random.RNG(seed=1)
    ckpt1 = ndd.checkpoint.Checkpoint()
    ckpt1.register(rng, "rng")
    ckpt1.collect()

    with tempfile.TemporaryDirectory() as d:
        pattern = os.path.join(d, "ckpt_{seq:03d}.json")
        p1 = ckpt1.save(pattern)
        # A fresh checkpoint instance must scan the dir and avoid clobbering.
        ckpt2 = ndd.checkpoint.Checkpoint()
        ckpt2.register(rng, "rng")
        ckpt2.collect()
        p2 = ckpt2.save(pattern)
        assert p1 != p2


def test_checkpoint_load_no_match_errors():
    """load raises FileNotFoundError when no file matches."""
    ckpt = ndd.checkpoint.Checkpoint()
    with tempfile.TemporaryDirectory() as d:
        pattern = os.path.join(d, "missing_{seq:03d}.json")
        with assert_raises(FileNotFoundError):
            ckpt.load(pattern)


def test_checkpoint_pattern_requires_seq():
    """save/load require a `{seq}` placeholder in the pattern."""
    rng = ndd.random.RNG(seed=1)
    ckpt = ndd.checkpoint.Checkpoint()
    ckpt.register(rng, "rng")
    ckpt.collect()
    with assert_raises(ValueError, glob="seq"):
        ckpt.save("/tmp/no_placeholder.json")
    with assert_raises(ValueError, glob="seq"):
        ckpt.load("/tmp/no_placeholder.json")


def test_checkpoint_clear():
    """clear resets the checkpoint to its empty state."""
    ckpt = ndd.checkpoint.Checkpoint()
    rng = ndd.random.RNG(seed=1)
    ckpt.register(rng, "rng")
    ckpt.collect()
    ckpt.clear()
    assert len(ckpt) == 0
    assert not ckpt.is_complete
    assert not ckpt.is_loaded


# ---------------------------------------------------------------------------
# EvalContext-bound checkpoint
# ---------------------------------------------------------------------------


def test_eval_context_has_lazy_checkpoint():
    """EvalContext.checkpoint returns a singleton Checkpoint per context."""
    ctx = ndd.EvalContext.current()
    c1 = ctx.checkpoint
    c2 = ctx.checkpoint
    assert c1 is c2
    assert isinstance(c1, ndd.checkpoint.Checkpoint)


def test_checkpoint_current_returns_eval_context_checkpoint():
    """checkpoint.current() returns EvalContext.current().checkpoint."""
    ctx = ndd.EvalContext.current()
    assert ndd.checkpoint.current() is ctx.checkpoint


# ---------------------------------------------------------------------------
# Mixed reader + RNG end-to-end
# ---------------------------------------------------------------------------


def test_checkpoint_mixed_reader_and_rng_end_to_end():
    """Mixed reader + RNG round-trip through serialize/deserialize."""
    reader = _make_reader(enable_checkpointing=True)
    rng = ndd.random.RNG(seed=2024)

    # Run a few iterations to advance state.
    it = reader.next_epoch(batch_size=4)
    next(it)
    next(it)
    rng()
    rng()

    ckpt = ndd.checkpoint.Checkpoint()
    ckpt.register(reader, "reader")
    ckpt.register(rng, "rng")
    ckpt.collect()
    payload = ckpt.serialize()

    expected_labels = _labels_of(next(it))
    expected_rngs = [rng() for _ in range(5)]

    # Now reconstruct.
    reader2 = _make_reader()
    rng2 = ndd.random.RNG(seed=12345)  # different seed - will be overwritten
    ckpt2 = ndd.checkpoint.Checkpoint()
    ckpt2.deserialize(payload)
    ckpt2.register(reader2, "reader")
    ckpt2.register(rng2, "rng")

    got_labels = _labels_of(next(reader2.next_epoch(batch_size=4)))
    got_rngs = [rng2() for _ in range(5)]
    assert got_labels == expected_labels
    assert got_rngs == expected_rngs


# ---------------------------------------------------------------------------
# Checkpoint - operator type tracking
# ---------------------------------------------------------------------------


def test_checkpoint_collect_records_type():
    """collect stores the operator's qualified type name in each entry."""
    ckpt = ndd.checkpoint.Checkpoint()
    rng = ndd.random.RNG(seed=1)
    reader = _make_reader(enable_checkpointing=True)
    next(reader.next_epoch(batch_size=4))  # advance so get_state is valid
    ckpt.register(rng, "rng")
    ckpt.register(reader, "reader")
    ckpt.collect()
    assert ckpt._states["rng"].type_name == "RNG"
    assert ckpt._states["reader"].type_name == "File"


def test_checkpoint_serialize_includes_type():
    """The serialized payload carries the operator type alongside the state."""
    ckpt = ndd.checkpoint.Checkpoint()
    rng = ndd.random.RNG(seed=1)
    ckpt.register(rng, "rng")
    ckpt.collect()
    payload = json.loads(ckpt.serialize())
    assert payload["version"] == 2
    assert payload["states"]["rng"]["type"] == "RNG"
    assert isinstance(payload["states"]["rng"]["state"], str)


def test_checkpoint_set_state_with_op_type_class():
    """set_state accepts a class as op_type and stores its qualified name."""
    ckpt = ndd.checkpoint.Checkpoint()
    ckpt.set_state("rng", "some_state", op_type=ndd.random.RNG)
    assert ckpt._states["rng"].type_name == "RNG"


def test_checkpoint_set_state_with_op_type_string():
    """set_state accepts a string as op_type and stores it verbatim."""
    ckpt = ndd.checkpoint.Checkpoint()
    ckpt.set_state("rng", "some_state", op_type="RNG")
    assert ckpt._states["rng"].type_name == "RNG"


def test_checkpoint_set_state_invalid_op_type_errors():
    """set_state rejects an op_type that is neither a type nor a string."""
    ckpt = ndd.checkpoint.Checkpoint()
    with assert_raises(TypeError, glob="op_type"):
        ckpt.set_state("rng", "some_state", op_type=42)


def test_checkpoint_set_state_op_type_mismatch_errors():
    """Registering an op of the wrong type for a typed state raises TypeError."""
    ckpt = ndd.checkpoint.Checkpoint()
    # Pretend a state for a File reader is buffered under "x".
    ckpt.set_state("x", "irrelevant", op_type=ndd.readers.File)
    rng = ndd.random.RNG(seed=1)
    # Registering an RNG under "x" must trigger a type-mismatch error before
    # the state is propagated.
    with assert_raises(TypeError, glob="Type mismatch*'File'*'RNG'*"):
        ckpt.register(rng, "x")


def test_checkpoint_register_replace_different_type_errors():
    """Replacing a registered op with one of a different type raises TypeError."""
    ckpt = ndd.checkpoint.Checkpoint()
    rng = ndd.random.RNG(seed=1)
    reader = _make_reader()
    ckpt.register(rng, "key")
    with assert_raises(TypeError, glob="Cannot replace operator*'RNG'*'File'*"):
        ckpt.register(reader, "key")


def test_checkpoint_register_after_load_wrong_type_errors():
    """Registering an op of a different type than the one in a loaded checkpoint errors."""
    rng = ndd.random.RNG(seed=1)
    ckpt = ndd.checkpoint.Checkpoint()
    ckpt.register(rng, "x")
    ckpt.collect()
    payload = ckpt.serialize()

    ckpt2 = ndd.checkpoint.Checkpoint()
    ckpt2.deserialize(payload)
    reader = _make_reader()
    with assert_raises(TypeError, glob="Type mismatch*'RNG'*'File'*"):
        ckpt2.register(reader, "x")


def test_checkpoint_register_after_load_correct_type_ok():
    """A loaded checkpoint with type info applies cleanly to a matching op."""
    rng = ndd.random.RNG(seed=42)
    for _ in range(3):
        rng()
    ckpt = ndd.checkpoint.Checkpoint()
    ckpt.register(rng, "rng")
    ckpt.collect()
    expected = [rng() for _ in range(5)]
    payload = ckpt.serialize()

    ckpt2 = ndd.checkpoint.Checkpoint()
    ckpt2.deserialize(payload)
    rng2 = ndd.random.RNG(seed=999)
    ckpt2.register(rng2, "rng")  # types match - no error
    got = [rng2() for _ in range(5)]
    assert got == expected


# ---------------------------------------------------------------------------
# Checkpoint - deserialize payload validation
# ---------------------------------------------------------------------------


def test_checkpoint_deserialize_format_version_mismatch_errors():
    """deserialize rejects payloads with an unsupported version."""
    payload = json.dumps({"version": 1, "states": {"rng": {"state": "x", "type": "RNG"}}})
    ckpt = ndd.checkpoint.Checkpoint()
    with assert_raises(ValueError, glob="version"):
        ckpt.deserialize(payload)


def test_checkpoint_deserialize_entry_not_a_dict_errors():
    """deserialize rejects entries that are not dicts (legacy v1 inline strings)."""
    payload = json.dumps({"version": 2, "states": {"rng": "raw_string_state"}})
    ckpt = ndd.checkpoint.Checkpoint()
    with assert_raises(ValueError, glob="Invalid checkpoint entry"):
        ckpt.deserialize(payload)


def test_checkpoint_deserialize_entry_missing_state_errors():
    """deserialize rejects entries that lack the ``state`` field."""
    payload = json.dumps({"version": 2, "states": {"rng": {"type": "RNG"}}})
    ckpt = ndd.checkpoint.Checkpoint()
    with assert_raises(ValueError, glob="Invalid checkpoint entry"):
        ckpt.deserialize(payload)


def test_checkpoint_deserialize_invalid_type_field_errors():
    """deserialize rejects entries whose ``type`` is not a string."""
    payload = json.dumps({"version": 2, "states": {"rng": {"state": "x", "type": 42}}})
    ckpt = ndd.checkpoint.Checkpoint()
    with assert_raises(ValueError, glob="Invalid `type`"):
        ckpt.deserialize(payload)


def test_checkpoint_deserialize_null_type_ok():
    """deserialize accepts a null ``type`` (matches set_state without op_type)."""
    payload = json.dumps({"version": 2, "states": {"rng": {"state": "x", "type": None}}})
    ckpt = ndd.checkpoint.Checkpoint()
    ckpt.deserialize(payload)
    assert ckpt._states["rng"].type_name is None
    assert ckpt._states["rng"].state == "x"


# ---------------------------------------------------------------------------
# Reader.set_state bytes branch
# ---------------------------------------------------------------------------


def test_reader_set_state_accepts_bytes():
    """set_state accepts a raw ``bytes`` blob (ASCII-encoded serialized form)."""
    ref = _make_reader(enable_checkpointing=True)
    it_ref = ref.next_epoch(batch_size=4)
    next(it_ref)
    state_bytes = str(ref.get_state()).encode("ascii")
    expected = _labels_of(next(it_ref))

    restored = _make_reader()
    restored.set_state(state_bytes)
    got = _labels_of(next(restored.next_epoch(batch_size=4)))
    assert got == expected, f"Restored reader produced {got} instead of {expected}."


# ---------------------------------------------------------------------------
# Failure during _maybe_apply
# ---------------------------------------------------------------------------


class _RaisingRNG:
    """Stand-in op whose ``set_state`` raises. Used to test partial-restore behavior."""

    def __init__(self):
        self.calls = 0

    def get_state(self, *, cuda_stream=None):
        return "state"

    def set_state(self, value):
        self.calls += 1
        raise RuntimeError("simulated set_state failure")


def test_checkpoint_register_after_load_set_state_failure_keeps_dirty():
    """If ``op.set_state`` raises during register-after-load, the key stays dirty."""
    payload = json.dumps(
        {
            "version": 2,
            "states": {
                "good": {"state": "x", "type": "_RaisingRNG"},
                "bad": {"state": "x", "type": "_RaisingRNG"},
            },
        }
    )
    ckpt = ndd.checkpoint.Checkpoint()
    ckpt.deserialize(payload)
    assert ckpt._dirty == {"good", "bad"}

    bad = _RaisingRNG()
    with assert_raises(RuntimeError, glob="simulated set_state failure"):
        ckpt.register(bad, "bad")
    # Failed apply does not clear `_dirty` for the failed key, and the unrelated
    # key remains dirty too - the caller can retry registration without losing state.
    assert "bad" in ckpt._dirty
    assert "good" in ckpt._dirty
    assert bad.calls == 1


# ---------------------------------------------------------------------------
# End-to-end via ndd.checkpoint.current()
# ---------------------------------------------------------------------------


def test_checkpoint_current_end_to_end():
    """A full save/load cycle going through ``ndd.checkpoint.current()``."""
    ndd.checkpoint.current().clear()  # ensure no leftover registrations from prior tests
    try:
        reader = _make_reader(enable_checkpointing=True)
        rng = ndd.random.RNG(seed=2026)
        ckpt = ndd.checkpoint.current()
        ckpt.register(reader, "reader")
        ckpt.register(rng, "rng")

        it = reader.next_epoch(batch_size=4)
        next(it)
        rng()

        ckpt.collect()
        with tempfile.TemporaryDirectory() as d:
            path = ckpt.save(os.path.join(d, "ckpt_{seq:04d}.json"))
            assert os.path.exists(path)

            expected_labels = _labels_of(next(it))
            expected_rngs = [rng() for _ in range(3)]

            ndd.checkpoint.current().clear()
            ckpt2 = ndd.checkpoint.current()
            ckpt2.load(os.path.join(d, "ckpt_{seq:04d}.json"))

            reader2 = _make_reader()
            rng2 = ndd.random.RNG(seed=0)
            ckpt2.register(reader2, "reader")
            ckpt2.register(rng2, "rng")

            got_labels = _labels_of(next(reader2.next_epoch(batch_size=4)))
            got_rngs = [rng2() for _ in range(3)]
            assert got_labels == expected_labels
            assert got_rngs == expected_rngs
    finally:
        ndd.checkpoint.current().clear()


# ---------------------------------------------------------------------------
# Compiled mode is rejected
# ---------------------------------------------------------------------------


def test_reader_checkpointing_rejects_compile_mode():
    """``enable_checkpointing=True`` and ``next_epoch(compile=True)`` are mutually exclusive."""
    reader = _make_reader(enable_checkpointing=True)
    with assert_raises(NotImplementedError, glob="*compiled mode*"):
        reader.next_epoch(batch_size=4, compile=True)


def test_reader_enable_checkpointing_rejected_after_compile():
    """A reader that has entered compiled mode cannot then opt in to checkpointing."""
    reader = _make_reader()
    next(reader.next_epoch(batch_size=4, compile=True))
    ckpt = ndd.checkpoint.Checkpoint()
    with assert_raises(NotImplementedError, glob="*compiled mode*"):
        ckpt.register(reader, "reader")


# ---------------------------------------------------------------------------
# Filename pattern regression - bracket characters in literals
# ---------------------------------------------------------------------------


def test_checkpoint_save_load_with_glob_metachars_in_literal():
    """Bracket characters in the filename literal must not be expanded as glob classes.

    Without escaping, ``glob.iglob`` would interpret ``[a-z]`` as a character class
    and refuse to match a file actually named ``[a-z]_0.json``.
    """
    rng = ndd.random.RNG(seed=42)
    for _ in range(3):
        rng()
    ckpt = ndd.checkpoint.Checkpoint()
    ckpt.register(rng, "rng")
    ckpt.collect()

    with tempfile.TemporaryDirectory() as d:
        pattern = os.path.join(d, "[a-z]_{seq}.json")
        saved = ckpt.save(pattern)
        assert os.path.basename(saved) == "[a-z]_0.json"

        ckpt2 = ndd.checkpoint.Checkpoint()
        loaded = ckpt2.load(pattern)
        assert loaded == saved
