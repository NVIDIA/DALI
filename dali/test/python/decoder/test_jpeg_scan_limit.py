# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import nvidia.dali.fn as fn
import os
import tempfile
import unittest
from nvidia.dali import pipeline_def

from nose_utils import assert_raises, attr
from nose2.tools import cartesian_params


class ProgressiveJpeg(unittest.TestCase):

    @classmethod
    def generate_file(cls, decoding_method, decoding_step):
        # based on
        # https://www.libjpeg-turbo.org/pmwiki/uploads/About/TwoIssueswiththeJPEGStandard.pdf
        assert decoding_method in ("huffman", "arithmetic")
        # fmt: off
        extent = (8192).to_bytes(2, "big")
        # sky's the limit, in the doc it's 8000k, but there's no need to inflate
        # the file size
        scans_num = 1024
        quant_table = b"".join(
            v.to_bytes(1, "big") for v in
            [
                16, 11, 10, 16, 24, 40, 51, 61,
                12, 12, 14, 19, 26, 58, 60, 55,
                14, 13, 16, 24, 40, 57, 69, 56,
                14, 17, 22, 29, 51, 87, 80, 62,
                18, 22, 37, 56, 68, 109, 103, 77,
                24, 35, 55, 64, 81, 104, 113, 92,
                49, 64, 78, 87, 103, 121, 120, 101,
                72, 92, 95, 98, 112, 100, 103, 99
            ])
        # Huffman tables ([dc|ac]_lum_[bits|vals])
        dc_lum_bits = [
            0, 1, 5, 1, 1, 1, 1, 1,
            1, 0, 0, 0, 0, 0, 0, 0
        ]
        dc_lum_bits_sum = sum(dc_lum_bits)
        dc_lum_bits = b"".join(v.to_bytes(1, "big") for v in dc_lum_bits)
        dc_lum_vals = b"".join(
            v.to_bytes(1, "big") for v in
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        ac_lum_bits = [
            0, 2, 1, 3, 3, 2, 4, 3,
            5, 5, 4, 4, 0, 0, 1, 0x7d
        ]
        ac_lum_bits_sum = sum(ac_lum_bits)
        ac_lum_bits = b"".join(v.to_bytes(1, "big") for v in ac_lum_bits)
        ac_lum_vals = b"".join(
            v.to_bytes(1, "big") for v in
            [
                0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12,
                0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07,
                0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xa1, 0x08,
                0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52, 0xd1, 0xf0,
                0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0a, 0x16,
                0x17, 0x18, 0x19, 0x1a, 0x25, 0x26, 0x27, 0x28,
                0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39,
                0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
                0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59,
                0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
                0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79,
                0x7a, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
                0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98,
                0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7,
                0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6,
                0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5,
                0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4,
                0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1, 0xe2,
                0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea,
                0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
                0xf9, 0xfa
            ])
        decoding_steps = {
            "ac_first": b"\xFF\xDA\x00\x08\x01\x00\x00\x01\x01\x00",
            "ac_refine": b"\xFF\xDA\x00\x08\x01\x00\x00\x01\x01\x10",
            "dc_first": b"\xFF\xDA\x00\x08\x01\x00\x00\x00\x00\x10",
            "dc_refine": b"\xFF\xDA\x00\x08\x01\x00\x00\x00\x00\x00",
        }
        # fmt: on
        file = tempfile.NamedTemporaryFile(mode="w+b", delete=True)
        try:
            file.write(b"\xff\xd8")
            file.write(b"\xff")
            file.write(b"\xc2" if decoding_method == "huffman" else b"\xca")
            file.write(b"\x00\x0b\x08")
            file.write(extent)
            file.write(extent)
            file.write(b"\x01\x00\x11\x00")
            file.write(b"\xff\xdb\x00\x43\x00")
            file.write(quant_table)
            if decoding_method == "huffman":
                file.write(b"\xff\xc4")  # DHT marker
                # 19 = 16 dc_lum_bits + 2 byte len + 1 byte Huffman class marker
                file.write((dc_lum_bits_sum + 19).to_bytes(2, "big"))
                file.write(b"\x00")  # Huffman class
                assert len(dc_lum_bits) == 16
                file.write(dc_lum_bits)
                assert len(dc_lum_vals) == dc_lum_bits_sum
                file.write(dc_lum_vals)

                file.write(b"\xff\xc4")  # DHT marker
                # 19 = 16 dc_lum_bits + 2 byte len + 1 byte Huffman class marker
                file.write((ac_lum_bits_sum + 19).to_bytes(2, "big"))
                file.write(b"\x10")  # Huffman class
                assert len(ac_lum_bits) == 16
                file.write(ac_lum_bits)
                assert len(ac_lum_vals) == ac_lum_bits_sum
                file.write(ac_lum_vals)
            for _ in range(scans_num):
                file.write(decoding_steps[decoding_step])
            file.flush()
            return file
        except:  # noqa: E722
            file.close()
            raise

    @classmethod
    def setUpClass(cls):
        cls.files = {
            (decoding_method, decoding_step): cls.generate_file(decoding_method, decoding_step)
            for decoding_method in ("huffman", "arithmetic")
            for decoding_step in ("ac_first", "ac_refine", "dc_first", "dc_refine")
        }

    @classmethod
    def tearDownClass(cls):
        for file in cls.files.values():
            file.close()

    @attr("jpeg_scans_limit")
    @cartesian_params(
        ("cpu", "mixed"),
        ("huffman", "arithmetic"),
        ("ac_first", "ac_refine", "dc_first", "dc_refine"),
    )
    def test_scans_limit(self, decoding_device, decoding_method, decoding_step):
        max_scans = int(os.environ.get("DALI_MAX_JPEG_SCANS", "256"))

        @pipeline_def(batch_size=1, device_id=0, num_threads=4)
        def pipeline():
            data, _ = fn.readers.file(files=self.files[(decoding_method, decoding_step)].name)
            return fn.decoders.image(data, device=decoding_device)

        pretty_decoding_dev = "CPU" if decoding_device == "cpu" else "MIXED"
        with assert_raises(
            RuntimeError,
            glob=f"Error in {pretty_decoding_dev} * "
            f"The number of scans ({max_scans + 1}) during progressive decoding *",
        ):
            p = pipeline()
            p.run()
