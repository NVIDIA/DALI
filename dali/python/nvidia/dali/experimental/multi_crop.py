# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Multi-crop helper.

DALI's built-in ``Crop``/``Slice`` operators produce a single output sample
per input sample. A common workflow (multi-crop inference, dense bbox
extraction, multi-region augmentation) needs N crops from one input image.
See https://github.com/NVIDIA/DALI/issues/4735.

This module provides a Python helper that fans out ``fn.slice`` N times
against the same input and stacks the results, giving an output with a
leading "crop" axis. The helper is implemented purely in terms of public
DALI operators; no new C++ kernel is required.

Example:
    @pipeline_def
    def pipe():
        img = fn.external_source(name="image")
        # Five 224x224 crops at different anchors.
        anchors = [(0, 0), (0, 100), (100, 0), (100, 100), (50, 50)]
        crops = multi_crop(img, anchors=anchors, crop=(224, 224), axes=(0, 1))
        return crops  # shape (5, 224, 224, C) when stacked.
"""

from nvidia.dali import fn


def multi_crop(
    data,
    anchors,
    crop=None,
    rel_anchors=None,
    rel_crop=None,
    axes=(0, 1),
    axis_names=None,
    stack=True,
    out_of_bounds_policy=None,
    fill_values=None,
    device=None,
):
    """Produce N crops from a single input by repeated ``fn.slice``.

    Exactly one of (``anchors`` + ``crop``) or (``rel_anchors`` + ``rel_crop``)
    must be supplied. Each entry of the list defines one output crop; all
    entries must agree on shape so the results can be stacked.

    Args:
        data: Input ``DataNode`` (the source image / sample).
        anchors: Sequence of absolute anchor coordinates, one per output
            crop. Each entry must be indexable along ``axes``.
        crop: Absolute crop shape applied to every anchor (paired with
            ``anchors``). Required when ``anchors`` is given.
        rel_anchors: Sequence of relative (0..1) anchor coordinates, one
            per output crop. Mutually exclusive with ``anchors``.
        rel_crop: Relative crop shape applied to every relative anchor.
            Required when ``rel_anchors`` is given.
        axes: Axes of ``data`` that the anchors/shapes index. Defaults to
            ``(0, 1)`` (HW for HWC images).
        axis_names: Alternative to ``axes``; passed through to ``fn.slice``.
        stack: If True (default) stack the per-crop outputs along a new
            leading axis via ``fn.stack``. If False, return a list of N
            ``DataNode`` instances.
        out_of_bounds_policy: Forwarded to ``fn.slice``.
        fill_values: Forwarded to ``fn.slice`` (for "pad" policy).
        device: Optional device override forwarded to ``fn.slice``.

    Returns:
        Either a single stacked ``DataNode`` (when ``stack=True``) with a new
        leading "crop" axis, or a Python list of ``DataNode`` instances of
        length N.

    Raises:
        ValueError: If anchor lists are missing/mismatched, or if both
            absolute and relative variants are provided.
    """
    abs_mode = anchors is not None
    rel_mode = rel_anchors is not None
    if abs_mode == rel_mode:
        raise ValueError(
            "multi_crop requires exactly one of `anchors` (with `crop`) or "
            "`rel_anchors` (with `rel_crop`)."
        )
    if abs_mode and crop is None:
        raise ValueError("`crop` must be provided when `anchors` is given.")
    if rel_mode and rel_crop is None:
        raise ValueError("`rel_crop` must be provided when `rel_anchors` is given.")

    anchor_list = list(anchors) if abs_mode else list(rel_anchors)
    if len(anchor_list) == 0:
        raise ValueError("multi_crop needs at least one anchor.")

    # Forward only the slice kwargs the user actually set; fn.slice rejects
    # `None` for some args.
    common = {}
    if axis_names is not None:
        common["axis_names"] = axis_names
    else:
        common["axes"] = axes
    if out_of_bounds_policy is not None:
        common["out_of_bounds_policy"] = out_of_bounds_policy
    if fill_values is not None:
        common["fill_values"] = fill_values
    if device is not None:
        common["device"] = device

    crops = []
    for anchor in anchor_list:
        if abs_mode:
            crops.append(fn.slice(data, start=anchor, shape=crop, **common))
        else:
            crops.append(fn.slice(data, rel_start=anchor, rel_shape=rel_crop, **common))

    if not stack:
        return crops
    if len(crops) == 1:
        # fn.stack with a single input still works, but skip the call so the
        # graph stays minimal in the trivial case.
        return fn.stack(*crops)
    return fn.stack(*crops)


__all__ = ["multi_crop"]
