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
    reader = _make_reader()
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
    ref = _make_reader()
    it_ref = ref.next_epoch(batch_size=4)
    next(it_ref)  # discard first batch
    state = ref.get_state()
    expected = _labels_of(next(it_ref))

    # Restored reader: set_state before the first iteration so that the state
    # is buffered and applied on backend creation.
    restored = _make_reader()
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
    reader = _make_reader()
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
    assert ckpt.register(rng1) == "0"
    assert ckpt.register(rng2) == "1"


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
    with assert_raises(KeyError):
        ckpt.set_state("unknown", "...")


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
    it_ref = ref.next_epoch(batch_size=4)
    next(it_ref)
    ckpt = ndd.checkpoint.Checkpoint()
    ckpt.register(ref, "reader")
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
    ckpt.register(ref_rng)            # gets "0"
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
    ckpt2.register(new_rng)            # picks up sequential key "0"
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
    with assert_raises(KeyError, glob="inferred key"):
        ckpt2.register(ndd.random.RNG(seed=3))


# ---------------------------------------------------------------------------
# Checkpoint - save / load
# ---------------------------------------------------------------------------


@params("ckpt_{seq:04d}.json", "ckpt_{seq}.bin")
def test_checkpoint_save_load_roundtrip(name_pattern):
    """save writes a file; load returns the most recent one."""
    rng = ndd.random.RNG(seed=42)
    rng()
    expected = [rng() for _ in range(5)]
    rng.set_state(rng.get_state())  # reset to current state

    # Recreate to capture state cleanly.
    rng = ndd.random.RNG(seed=42)
    rng()
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
    reader = _make_reader()
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
