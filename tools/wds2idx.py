#!/usr/bin/python3
# Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import argparse
import os
import subprocess  # nosec B404
import sys
import tarfile
import time
from shutil import which


class IndexCreator:
    """Reads `Webdataset` data format, and creates index file
    that enables random access.

    Example usage:
    ----------
    >> with IndexCreator('data/test.tar','data/test.idx') as ic:
    >>     ic.create_index()
    >> !ls data/
    test.tar  test.idx

    Parameters
    ----------
    uri : str
        Path to the archive file.
    idx_path : str
        Path to the index file, that will be created/overwritten.
    """

    tar_block_size = 512
    index_file_version = "v1.2"

    def __init__(self, uri, idx_path, verbose=True):
        self.uri = uri
        self.idx_path = idx_path
        self.fidx = open(self.idx_path, "w")
        self.verbose = verbose

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close()

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
    def split_name(filepath):
        """Splits the webdataset into the basename and the extension"""
        dot_pos = filepath.find(".", filepath.rfind("/") + 1)
        return filepath[:dot_pos], filepath[dot_pos + 1 :]

    def _get_data_tar(self):
        """Retrieves the data about the offset, name and size of each component
        using the gnu tar utility, while also filtering out non-file entries"""

        tar_blocks_proc = subprocess.Popen(  # nosec B603, B607
            ["tar", "--list", "--block-num", "--file", self.uri],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        tar_types_sizes_proc = subprocess.Popen(  # nosec B603, B607
            ["tar", "--verbose", "--list", "--file", self.uri],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        tar_blocks = tar_blocks_proc.communicate()[0].split(b"\n")  # block <n>: <filepath>
        tar_types_sizes = tar_types_sizes_proc.communicate()[0].split(
            b"\n"
        )  # <type>... <size> <date> <name>

        # Extracting
        for blocks_line, types_sizes_line in zip(tar_blocks, tar_types_sizes):
            if not blocks_line or not types_sizes_line:
                continue

            name = str(blocks_line[blocks_line.find(b":") + 2 :], "ascii")
            entry_type = types_sizes_line[0:1]

            if entry_type != b"-":
                continue

            offset = int(blocks_line[blocks_line.find(b"block") + 6 : blocks_line.find(b":")])
            # according to https://www.loc.gov/preservation/digital/formats/fdd/fdd000531.shtml#:~:text=A%20tar%20(tape%20archive)%20file,are%20not%20compressed%20archive%20files. # noqa: E501, W505
            # each data record is preceded by 512-byte header. `tar --list --block-num --file`
            # return the position (counted in 512-byte blocks) of the header for a given entry.
            # So the size of the header needs to be added to get the data offset
            offset = (offset + 1) * 512

            size = types_sizes_line[: -len(name)]
            size = size[: size.rfind(b"-") - 8]  # "... <size> 20yy-mm-...."
            size = int(size[size.rfind(b" ") :])

            yield offset, name, size

    def _get_data_tarfile(self):
        """Retreives the data about the offset, name and size of each component
        using the tarfile module, while also filtering out non-file entries
        Intended as a fallback for the gnu tar version (since it is much slower)"""

        print(
            "Warning: tar utility not found. Falling back to tarfile."
            + " Processing will most likely take much longer",
            file=sys.stderr,
        )
        farchive = tarfile.open(self.uri)
        for member in iter(farchive):
            if member.type != tarfile.REGTYPE:
                continue
            offset = farchive.fileobj.tell()
            yield offset, member.name, member.size

    def create_index(self):
        """Creates the index file from a tar archive"""
        self.reset()

        pre_time = time.time()
        counter = 0
        report_step = 100000

        if self.verbose:
            print(f"time: {time.time() - pre_time:.2f} count: {counter} stage: collect")

        # Aggregates extensions in samples
        aggregated_data = []
        last_basename = None

        for offset, name, size in (
            self._get_data_tar() if which("tar") is not None else self._get_data_tarfile()
        ):
            if counter % report_step == 0 and counter > 0:
                cur_time = time.time()
                if self.verbose:
                    print(f"time: {cur_time - pre_time:.2f} count: {counter} stage: collect")
            counter += 1

            basename, extension = IndexCreator.split_name(name)

            # check for the files starting with a dot (hidden files)
            if not basename or basename.endswith("/"):
                continue

            if last_basename != basename:
                aggregated_data.append([(extension, offset, size, name)])
                last_basename = basename
            else:
                aggregated_data[-1].append((extension, offset, size, name))

        if not aggregated_data:
            raise ValueError("Webdataset Tar File empty")

        # Constructs the index file out of the aggregated extensions
        self.fidx.write(f"{IndexCreator.index_file_version} {len(aggregated_data)}\n")
        for bundle in aggregated_data:
            if counter % report_step == 0:
                cur_time = time.time()
                if self.verbose:
                    print(f"time: {cur_time - pre_time:.2f} count: {counter} stage: index")
            self.fidx.write(" ".join(map(lambda component: " ".join(map(str, component)), bundle)))
            self.fidx.write("\n")
            counter += 1

        cur_time = time.time()
        if self.verbose:
            print(f"time: {cur_time - pre_time:.2f} count: {counter} stage: done")


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Creates a webdataset index file for the use with the `fn.readers.webdataset`.",
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
