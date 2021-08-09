#!/usr/bin/python3
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import os
import time
import argparse
import tarfile

class IndexCreator():
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
    def __init__(self, uri, idx_path):
        self.uri = uri
        self.idx_path = idx_path
        self.farchive = tarfile.TarFile(self.uri)
        self.fidx = open(self.idx_path, 'w')

    def open(self):
        """Opens the archive and index files and sets their read heads to 0."""
        if self.farchive.closed:
            self.farchive = tarfile.TarFile(self.uri)
        else:
            self.farchive.fileobj.seek(0)

        if self.fidx.closed:
            self.fidx = open(self.idx_path, 'w')
        else:
            self.fidx.seek(0)

    def close(self):
        """Closes the archive and index files."""
        if not self.farchive.closed:
            self.farchive.close()
        if not self.fidx.closed:
            self.fidx.close()

    def reset(self):
        """Resets the archive and index files."""
        self.close()
        self.open()

    def create_index(self):
        """Creates the index file from open record file
        """
        self.reset()

        # Has to parse the archive first because needs the number of files in the archive
        pre_time = time.time()
        data = []
        counter = 0
        report_step = 100000
        for member in iter(self.farchive):
            if counter % report_step == 0:
                  cur_time = time.time()
                  print(f"time: {cur_time - pre_time:.2f} count: {counter} stage: collect")
            data.append((member.name, member.offset))
            counter += 1

        self.fidx.write(f"{len(data)}\n")
        for name, offset in data:
            if counter % report_step == 0:
                  cur_time = time.time()
                  print(f"time: {cur_time - pre_time:.2f} count: {counter} stage: index")
            self.fidx.write(f"{offset} {name}\n")
            counter += 1
        
        cur_time = time.time()
        print(f"time: {cur_time - pre_time:.2f} count: {counter} stage: done")

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Create an index file from .tar file')
    parser.add_argument('archive', help='path to .tar file.')
    parser.add_argument('index', help='path to index file.')
    args = parser.parse_args()
    args.archive = os.path.abspath(args.archive)
    args.index = os.path.abspath(args.index)
    return args

def main():
    args = parse_args()
    creator = IndexCreator(args.archive, args.index)
    creator.create_index()
    creator.close()

if __name__ == '__main__':
    main()
