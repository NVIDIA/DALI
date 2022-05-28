# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
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

from itertools import cycle, permutations
from random import randint
from nvidia.dali import pipeline, pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import nvidia.dali.ops as ops
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.backend import TensorCPU

import numpy as np

from test_utils import np_type_to_dali
from nose_utils import assert_raises

min_dim = 2

def _all_reduction_shapes(sample_dim, nbins):
    axes = range(sample_dim)
    sizes = range(min_dim, sample_dim+min_dim)
    for axes_perm in permutations(axes):
        for nred_axes in range(sample_dim+1):
            input_shape = tuple((sizes[ax] for ax in range(sample_dim)))
            reduction_axes = axes_perm[:nred_axes]
            non_reduction_axes = []
            for ax in axes:
                if not ax in reduction_axes:
                    non_reduction_axes.append(ax)
            output_shape = [sizes[axis] for axis in non_reduction_axes]
            if len(reduction_axes) != 0:
                for bin_dim in nbins:
                    output_shape.append(bin_dim)
            output_shape = tuple(output_shape)
            yield (reduction_axes, input_shape, output_shape)

def _jagged_batch(batch_size, input_sh, output_sh, np_type, axes):
    batch = []
    output_shapes = []
    for i in range(batch_size):
        map = {}
        for sz in range(min_dim, len(input_sh)+min_dim):
            map[sz] = randint(8, 8+len(input_sh))

        mapped_in = [map[d] for d in [*input_sh,]]
        if len([x for x in axes]) > 0:
            mapped_out = [map[d] for d in [*output_sh,][:-1]]
            mapped_out.append(output_sh[-1])
        else:
            mapped_out = [map[d] for d in [*output_sh,]]
        batch.append(np.ones(mapped_in, np_type))
        output_shapes.append(np.array(mapped_out))
    return (batch, output_shapes)

def _testimpl_uniform_histogram_shape(batch_size, device, in_sh, out_sh, num_bins, axes, ch_ax, np_type):
    range_01 = (np.array([0.0], dtype=np.float32), np.array([1.0], dtype=np.float32))
    ranges = {type(np.uint8) : range_01, type(np.uint16) : range_01, type(np.float32) : range_01}
    @pipeline_def(batch_size=batch_size, num_threads=3, device_id=0)
    def uniform_histogram1D_uniform_shape_pipe(np_dtype, num_bins=num_bins, device='cpu', axes=[]):
        batches = [[np.ones(in_sh, dtype = np_dtype)]*batch_size, [np.zeros(in_sh, dtype = np_dtype)]*batch_size]
        in_tensors = fn.external_source(source=batches, device=device, cycle=True)
        out_tensors = fn.histogram.uniform_histogram(in_tensors, *ranges[type(np_type)],
            num_bins=num_bins, axes=axes, channel_axis=ch_ax)
        return out_tensors

    @pipeline_def(batch_size=batch_size, num_threads=3, device_id=0)
    def uniform_histogram1D_jagged_shape_pipe(np_dtype, num_bins=num_bins, device='cpu', axes=[]):
        batch, out_sh_list = _jagged_batch(batch_size, in_sh, out_sh, np_type, axes)
        batches = [batch]*2
        out_sizes_batches = [out_sh_list]*2
        in_tensors = fn.external_source(source=batches, device=device, cycle=True)
        out_tensors = fn.histogram.uniform_histogram(in_tensors, *ranges[type(np_type)],
            num_bins=num_bins, axes=axes, channel_axis=ch_ax)
        out_sizes = fn.external_source(source = out_sizes_batches, device=device, cycle=True)
        return out_tensors, out_sizes

    # Jagged tensors not tested right now for multidimensional histograms
    # would require recalculating bin argument
    if ch_ax == -1:
        pipe = uniform_histogram1D_uniform_shape_pipe(np_dtype=np.uint8, device=device, axes=axes, num_bins=num_bins)
        pipe.build()
        for iter in range(2):
            out, = pipe.run()
            for ret_sh in out.shape():
                assert(ret_sh == out_sh)

    pipe = uniform_histogram1D_jagged_shape_pipe(np_dtype=np.uint8, device=device, axes=axes, num_bins=num_bins)
    pipe.build()
    for iter in range(2):
        out, sz = pipe.run()
        for ret_sz, expected_sz in zip(out.shape(), sz.as_array()):
            assert(ret_sz == tuple(expected_sz))

def test_uniform_hist_args():
    range_01_1ch = (np.array([0.0], dtype=np.float32), np.array([1.0], dtype=np.float32))
    range_01_2ch = (np.array([0.0, 0.0], dtype=np.float32), np.array([1.0, 1.0], dtype=np.float32))

    t_2x33 = np.array([[0]*33, [1]*33], dtype=np.uint8)
    t_2x7 = np.array([[0]*7, [1]*7], dtype=np.uint8)

    @pipeline_def(batch_size=2, num_threads=3, device_id=0)
    def pipeline_unf_1d_rng_unif():
        batches = [[t_2x33]*2]
        rng_lo, rng_hi = range_01_1ch
        in_rng_lo = fn.external_source(source=[[rng_lo]*2], device='cpu', cycle=True)
        in_rng_hi = fn.external_source(source=[[rng_hi]*2], device='cpu', cycle=True)
        in_tensors = fn.external_source(source=batches, device='cpu', cycle=True, layout='AB')
        out_tensors = fn.histogram.uniform_histogram(in_tensors, in_rng_lo, in_rng_hi, num_bins=np.array([7]))
        return out_tensors

    pipe = pipeline_unf_1d_rng_unif()
    pipe.build()
    pipe.run()

    @pipeline_def(batch_size=2, num_threads=3, device_id=0)
    def pipeline_unf_1d_rng_nonunif():
        batches = [[t_2x33]*2]
        rng_lo_1ch, rng_hi_1ch = range_01_1ch
        rng_lo_2ch, rng_hi_2ch = range_01_2ch
        in_rng_lo = fn.external_source(source=[[rng_lo_1ch, rng_lo_2ch]], device='cpu', cycle=True)
        in_rng_hi = fn.external_source(source=[[rng_hi_1ch, rng_hi_2ch]], device='cpu', cycle=True)
        in_tensors = fn.external_source(source=batches, device='cpu', cycle=True, layout='AB')
        out_tensors = fn.histogram.uniform_histogram(in_tensors, in_rng_lo, in_rng_hi, num_bins=np.array([7]))
        return out_tensors

    pipe = pipeline_unf_1d_rng_nonunif()
    with assert_raises(RuntimeError, regex="Histogram bins ranges must be uniform across batch!"):
        pipe.build()
        pipe.run()

    @pipeline_def(batch_size=2, num_threads=3, device_id=0)
    def pipeline_unf_1d_rng_mismatch():
        batches = [[t_2x33]*2]
        rng_lo_1ch, rng_hi_1ch = range_01_1ch
        rng_lo_2ch, rng_hi_2ch = range_01_2ch
        in_rng_lo = fn.external_source(source=[[rng_lo_1ch, rng_lo_1ch]], device='cpu', cycle=True)
        in_rng_hi = fn.external_source(source=[[rng_hi_2ch, rng_hi_2ch]], device='cpu', cycle=True)
        in_tensors = fn.external_source(source=batches, device='cpu', cycle=True, layout='AB')
        out_tensors = fn.histogram.uniform_histogram(in_tensors, in_rng_lo, in_rng_hi, num_bins=np.array([7]))
        return out_tensors

    pipe = pipeline_unf_1d_rng_mismatch()
    with assert_raises(RuntimeError, regex="Expected matching histogram bin upper and lower range shapes!"):
        pipe.build()
        pipe.run()

    @pipeline_def(batch_size=2, num_threads=3, device_id=0)
    def pipeline_unf_1d_num_bins2():
        batches = [[t_2x33]*2]
        in_tensors = fn.external_source(source=batches, device='cpu', cycle=True, layout='AB')
        out_tensors = fn.histogram.uniform_histogram(in_tensors, *range_01_1ch, num_bins=np.array([7, 2]))
        return out_tensors

    pipe = pipeline_unf_1d_num_bins2()
    with assert_raises(RuntimeError, regex="Expected uniform shape for argument \"num_bins\" but got shape"):
        pipe.build()
        pipe.run()

    @pipeline_def(batch_size=2, num_threads=3, device_id=0)
    def pipeline_unf_nd_num_bins1():
        batches = [[t_2x33]*2]
        in_tensors = fn.external_source(source=batches, device='cpu', cycle=True, layout='AB')
        out_tensors = fn.histogram.uniform_histogram(in_tensors, *range_01_2ch, num_bins=np.array([7]),
            channel_axis_name='A')
        return out_tensors

    @pipeline_def(batch_size=2, num_threads=3, device_id=0)
    def pipeline_unf_ch_no_unsupported():
        ranges_lo_33ch = np.array([0.0]*33, dtype=np.float32)
        ranges_hi_33ch = np.array([1.0]*33, dtype=np.float32)
        batches = [[t_2x33]*2]
        in_tensors = fn.external_source(source=batches, device='cpu', cycle=True, layout='AB')
        out_tensors = fn.histogram.uniform_histogram(in_tensors, ranges_lo_33ch, ranges_hi_33ch,
            num_bins=np.array([1]), channel_axis_name='A')
        return out_tensors
    
    pipe = pipeline_unf_ch_no_unsupported()
    with assert_raises(RuntimeError, regex="Unsupported histogram dimensionality, should be not greater than 32"):
        pipe.build()
        pipe.run()

    @pipeline_def(batch_size=2, num_threads=3, device_id=0)
    def pipeline_unf_no_channels():
        ranges_lo_0ch = np.array([], dtype=np.float32)
        ranges_hi_0ch = np.array([], dtype=np.float32)
        batches = [[t_2x33]*2]
        in_tensors = fn.external_source(source=batches, device='cpu', cycle=True, layout='AB')
        out_tensors = fn.histogram.uniform_histogram(in_tensors, ranges_lo_0ch, ranges_hi_0ch,
            num_bins=np.array([1]), channel_axis_name='A')
        return out_tensors

    pipe = pipeline_unf_no_channels()
    with assert_raises(RuntimeError, regex="Expected histogram bin ranges for at least one histogram dimension"):
        pipe.build()
        pipe.run()

    # FIXME:
    # We probably shouldn't allow broadcast of num_bins in such case, but no easy way to do that with
    # current behaviour of ArgHelper
    pipe = pipeline_unf_nd_num_bins1()
    #with assert_raises(RuntimeError, regex="Expected uniform shape for argument \"num_bins\" but got shape"):
    pipe.build()
    pipe.run()

    @pipeline_def(batch_size=2, num_threads=3, device_id=0)
    def pipeline_unf_1d_chax_name():
        batches = [[t_2x33]*2]
        in_tensors = fn.external_source(source=batches, device='cpu', cycle=True, layout='AB')
        out_tensors = fn.histogram.uniform_histogram(in_tensors, *range_01_1ch, num_bins=np.array([7]),
            channel_axis_name='A')
        return out_tensors

    pipe = pipeline_unf_1d_chax_name()
    with assert_raises(RuntimeError, regex="None of `channel_axis` and `channel_axis_name` arguments should be specified for single dimensional histograms!"):
        pipe.build()
        pipe.run()

    @pipeline_def(batch_size=2, num_threads=3, device_id=0)
    def pipeline_unf_1d_chax_idx():
        batches = [[t_2x33]*2]
        in_tensors = fn.external_source(source=batches, device='cpu', cycle=True, layout='AB')
        out_tensors = fn.histogram.uniform_histogram(in_tensors, *range_01_1ch, num_bins=np.array([7]),
            channel_axis=0)
        return out_tensors

    pipe = pipeline_unf_1d_chax_idx()
    with assert_raises(RuntimeError, regex="None of `channel_axis` and `channel_axis_name` arguments should be specified for single dimensional histograms!"):
        pipe.build()
        pipe.run()

    @pipeline_def(batch_size=2, num_threads=3, device_id=0)
    def pipeline_unf_nd_chax_both():
        batches = [[t_2x33]*2]
        in_tensors = fn.external_source(source=batches, device='cpu', cycle=True)
        out_tensors = fn.histogram.uniform_histogram(in_tensors, *range_01_2ch, num_bins=np.array([7, 2]),
            channel_axis=0, channel_axis_name="B")
        return out_tensors

    pipe = pipeline_unf_nd_chax_both()
    with assert_raises(RuntimeError, regex="Arguments `channel_axis` and `channel_axis_name` are mutually exclusive"):
        pipe.build()
        pipe.run()

    @pipeline_def(batch_size=2, num_threads=3, device_id=0)
    def pipeline_unf_nd_no_chax():
        batches = [[t_2x33]*2]
        in_tensors = fn.external_source(source=batches, device='cpu', cycle=True)
        out_tensors = fn.histogram.uniform_histogram(in_tensors, *range_01_2ch, num_bins=np.array([7, 2]))
        return out_tensors

    pipe = pipeline_unf_nd_no_chax()
    with assert_raises(RuntimeError, regex="One of arguments `channel_axis` and `channel_axis_name` should be specified"):
        pipe.build()
        pipe.run()

    @pipeline_def(batch_size=2, num_threads=3, device_id=0)
    def pipeline_unf_nd_chax_bad_idx():
        batches = [[t_2x33]*2]
        in_tensors = fn.external_source(source=batches, device='cpu', cycle=True, layout="AB")
        out_tensors = fn.histogram.uniform_histogram(in_tensors, *range_01_2ch, num_bins=np.array([7, 2]), channel_axis=2)
        return out_tensors

    pipe = pipeline_unf_nd_chax_bad_idx()
    with assert_raises(RuntimeError, regex="Invalid axis specified for argument `channel_axis`"):
        pipe.build()
        pipe.run()

    @pipeline_def(batch_size=2, num_threads=3, device_id=0)
    def pipeline_unf_nd_chax_many():
        in_tensors = fn.external_source(source=[[t_2x33]*2], device='cpu', cycle=True, layout='AB')
        out_tensors = fn.histogram.uniform_histogram(in_tensors, *range_01_2ch, num_bins=np.array([7, 2]),
            channel_axis_name="AB")
        return out_tensors

    pipe = pipeline_unf_nd_chax_many()
    with assert_raises(RuntimeError, regex="Single axis name should be specified as `channel_axis_name`"):
        pipe.build()
        pipe.run()

    @pipeline_def(batch_size=2, num_threads=3, device_id=0)
    def pipeline_unf_nd_chax_red_AA():
        in_tensors = fn.external_source(source=[[t_2x33]*2], device='cpu', cycle=True, layout='AB')
        out_tensors = fn.histogram.uniform_histogram(in_tensors, *range_01_2ch, num_bins=np.array([7, 2]),
            channel_axis_name='A', axis_names='A')
        return out_tensors

    @pipeline_def(batch_size=2, num_threads=3, device_id=0)
    def pipeline_unf_nd_chax_red_A0():
        in_tensors = fn.external_source(source=[[t_2x33]*2], device='cpu', cycle=True, layout='AB')
        out_tensors = fn.histogram.uniform_histogram(in_tensors, *range_01_2ch, num_bins=np.array([7, 2]),
            channel_axis_name='A', axes=0)
        return out_tensors

    @pipeline_def(batch_size=2, num_threads=3, device_id=0)
    def pipeline_unf_nd_chax_red_00():
        in_tensors = fn.external_source(source=[[t_2x33]*2], device='cpu', cycle=True, layout='AB')
        out_tensors = fn.histogram.uniform_histogram(in_tensors, *range_01_2ch, num_bins=np.array([7, 2]),
            channel_axis=0, axes=0)
        return out_tensors

    @pipeline_def(batch_size=2, num_threads=3, device_id=0)
    def pipeline_unf_nd_chax_red_0A():
        in_tensors = fn.external_source(source=[[t_2x33]*2], device='cpu', cycle=True, layout='AB')
        out_tensors = fn.histogram.uniform_histogram(in_tensors, *range_01_2ch, num_bins=np.array([7, 2]),
            channel_axis=0, axis_names='A')
        return out_tensors

    pipe = pipeline_unf_nd_chax_red_AA()
    with assert_raises(RuntimeError, regex="Axis 0 can be eigther reduction axis `axes` or `channel_axis`, not both"):
        pipe.build()
        pipe.run()

    pipe = pipeline_unf_nd_chax_red_A0()
    with assert_raises(RuntimeError, regex="Axis 0 can be eigther reduction axis `axes` or `channel_axis`, not both"):
        pipe.build()
        pipe.run()

    pipe = pipeline_unf_nd_chax_red_00()
    with assert_raises(RuntimeError, regex="Axis 0 can be eigther reduction axis `axes` or `channel_axis`, not both"):
        pipe.build()
        pipe.run()

    pipe = pipeline_unf_nd_chax_red_0A()
    with assert_raises(RuntimeError, regex="Axis 0 can be eigther reduction axis `axes` or `channel_axis`, not both"):
        pipe.build()
        pipe.run()
    
    @pipeline_def(batch_size=2, num_threads=3, device_id=0)
    def pipeline_unf_nd_ch_mismatch(ranges, bins, ch_axis=0):
        batches = [[t_2x33]*2]
        in_tensors = fn.external_source(source=batches, device='cpu', cycle=True)
        out_tensors = fn.histogram.uniform_histogram(in_tensors, *ranges, num_bins=bins,
            channel_axis=ch_axis)
        return out_tensors

    bins_2ch = [1]*2
    pipe = pipeline_unf_nd_ch_mismatch(range_01_2ch, bins_2ch, ch_axis=1)
    with assert_raises(RuntimeError,
        regex="Number of channels in dimension specified as channel axis \(1\) doesn't match histogram dimensionality, \(33 vs 2\)"):
        pipe.build()
        pipe.run()

    pipe = pipeline_unf_nd_ch_mismatch(range_01_2ch, bins_2ch, ch_axis=0)
    pipe.build()
    pipe.run()

def test_reduce_shape_histogram():
    batch_size = 10
    for device in ['cpu']:
        for sample_dim in range(1, 4):
            for nbins in [[1], [16], [1024]]:
                for type in [np.uint8, np.uint16, np.float32]:
                    for axes, in_sh, out_sh in _all_reduction_shapes(sample_dim, nbins):
                        yield _testimpl_uniform_histogram_shape, batch_size, device, in_sh, out_sh, nbins, axes, -1, type

