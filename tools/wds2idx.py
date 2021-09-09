#!/usr/bin/python3
# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import sys
import time
import argparse
import subprocess
from shutil import which
import tarfile
import math


class IndexCreator:
    """Reads `Webdataset` data format, and creates index file
    that enables random access.

    Example usage:
    ----------
    >>> creator = IndexCreator('data/test.tar','data/test.idx')
    >>> creator.create_index()
    >>> creator.close()
    >>> !ls data/
    test.tar  test.idx

    Parameters
    ----------
    uri : str
        Path to the archive file.
    idx_path : str
        Path to the index file, that will be created/overwritten.
    """

    tar_block_size = 512

    def __init__(self, uri, idx_path):
        self.uri = uri
        self.idx_path = idx_path
        self.fidx = open(self.idx_path, "w")

    def open(self):
        """Opens the archive and index files and sets their read heads to 0."""
        if self.fidx.closed:
            self.fidx = open(self.idx_path, "w")
        else:
            self.fidx.seek(0)

    def close(self):
        """Closes the archive and index files."""
        if not self.fidx.closed:
            self.fidx.close()

    def reset(self):
        """Resets the archive and index files."""
        self.close()
        self.open()

    @staticmethod
    def split_name(filepath):  # translated from the matching function in c++
        """Splits the webdataset into the basename and the extension"""
        base_name_pos = filepath.rfind("\\") + 1
        dot_pos = filepath.find(".", base_name_pos + 1)
        if dot_pos == -1:
            return filepath, ""
        return filepath[:dot_pos], filepath[dot_pos + 1 :]

    def _get_data_tar(self):
        tar_blocks_proc = subprocess.Popen(
            ["tar", "--list", "--block-num", "--file", self.uri],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        tar_types_sizes_proc = subprocess.Popen(
            ["tar", "--verbose", "--list", "--file", self.uri],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        tar_blocks = tar_blocks_proc.communicate()[0].split(b"\n")  # block <n>: <filepath>
        tar_types_sizes = tar_types_sizes_proc.communicate()[0].split(
            b"\n"
        )  # <type>... <size> <date> <name>

        # Extracting total size
        last_blocks_line = None
        for blocks_line in reversed(tar_blocks):
            if not not blocks_line:
                last_blocks_line = blocks_line
                break

        total_size = (
            int(last_blocks_line[last_blocks_line.find(b"block") + 6 : last_blocks_line.find(b":")])
            * IndexCreator.tar_block_size
        )

        yield total_size

        # Extracting
        tar_data = zip(tar_blocks, tar_types_sizes)
        for blocks_line, types_sizes_line in zip(tar_blocks, tar_types_sizes):
            if not blocks_line or not types_sizes_line:
                continue

            name = str(blocks_line[blocks_line.find(b":") + 2 :], "ascii")
            entry_type = types_sizes_line[0:1]

            if entry_type != b"-" or name.startswith("."):
                continue

            offset = int(blocks_line[blocks_line.find(b"block") + 6 : blocks_line.find(b":")]) * 512
            size = types_sizes_line[: -len(name)]
            size = size[: size.rfind(b"-") - 8]  # "... <size> 20yy-mm-...."
            size = int(size[size.rfind(b" ") :])

            yield offset, name, size

    def _get_data_tarfile(self):
        print("Warning: tar utility not found. Falling back to tarfile", file=sys.stderr)

        members = []
        total_size = 0
        farchive = tarfile.open(self.uri)
        for member in iter(farchive):
            total_size = (
                farchive.fileobj.tell()
                + math.ceil(member.size / IndexCreator.tar_block_size)
                * IndexCreator.tar_block_size
            )

        yield total_size

        for member in iter(farchive):
            if member.type != tarfile.REGTYPE:
                continue
            yield member.offset, member.name, member.size

    def create_index(self):
        """Creates the index file from open record file"""
        self.reset()

        pre_time = time.time()
        counter = 0
        report_step = 100000

        print(f"time: {time.time() - pre_time:.2f} count: {counter} stage: collect")

        # Aggregates extensions in samples
        data_generator = (
            self._get_data_tar() if which("tar") is not None else self._get_data_tarfile()
        )
        total_size = next(data_generator)

        aggregated_data = []
        last_basename = None

        for offset, name, size in data_generator:
            if counter % report_step == 0:
                cur_time = time.time()
                print(f"time: {cur_time - pre_time:.2f} count: {counter} stage: collect")
            counter += 1

            basename, extension = IndexCreator.split_name(name)

            if last_basename != basename:
                aggregated_data.append((offset, [(extension, size)]))
                last_basename = basename
            else:
                aggregated_data[-1][1].append((extension, size))

        if not aggregated_data:
            raise ValueError("Webdataset Tar File empty")

        # Constructs the index file out of the aggregated extensions
        self.fidx.write(f"{total_size} {len(aggregated_data)}\n")
        for offset, extensions_sizes in aggregated_data:
            if counter % report_step == 0:
                cur_time = time.time()
                print(f"time: {cur_time - pre_time:.2f} count: {counter} stage: index")
            self.fidx.write(
                f"""{offset} {' '.join(
                map(lambda ext_size: str(ext_size[0]) + ' ' + str(ext_size[1]), extensions_sizes)
            )}\n"""
            )
            counter += 1

        cur_time = time.time()
        print(f"time: {cur_time - pre_time:.2f} count: {counter} stage: done")


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Creates a webdataset index file for the use with the fn.readers.webdataset from DALI.",
    )
    parser.add_argument("archive", help="path to .tar file.")
    parser.add_argument(
        "index",
        help="path to index file",
        nargs="?",
    )
    args = parser.parse_args()
    if args.index is None:
        args.index = args.archive[: args.archive.find(".", args.archive.rfind("/") + 2)] + ".idx"
    args.archive = os.path.abspath(args.archive)
    args.index = os.path.abspath(args.index)
    return args


def main():
    args = parse_args()
    creator = IndexCreator(args.archive, args.index)
    creator.create_index()
    creator.close()


if __name__ == "__main__":
    main()
