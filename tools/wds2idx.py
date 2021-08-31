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
        self.farchive = tarfile.open(self.uri)
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

    @staticmethod
    def split_name(filepath): # translated from the matching function in c++
        """Splits the webdataset into the basename and the extension
        """
        base_name_pos = filepath.rfind('\\') + 1
        dot_pos = filepath.find('.', base_name_pos + 1)
        if dot_pos == -1:
            return filepath
        return filepath[:dot_pos], filepath[dot_pos + 1:]


    def create_index(self):
        """Creates the index file from open record file
        """
        self.reset()

        pre_time = time.time()
        counter = 0
        report_step = 10000

        # Parse the archive first for the file offsets
        data = []
        last_skipped = 0
        for member in self.farchive:
            if counter % report_step == 0:
                cur_time = time.time()
                print(f"time: {cur_time - pre_time:.2f} count: {counter} stage: collect")
            counter += 1

            #if member.type != tarfile.REGTYPE or member.name.startswith('.'):
            #    last_skipped = member.offset
            #    continue
            #last_skipped = self.farchive.fileobj.tell()
            #basename, extension = IndexCreator.split_name(member.name)
            #offset = member.offset
            #if not data or data[-1][0] != basename:
            #    data.append((offset, [extension]))
            #else:
            #    data[-1][1].append(extension)
        return

        if not data:
            raise ValueError("Webdataset Tar File empty")

        # Then construct the index file out of it
        self.fidx.write(f"{last_skipped} {len(data)}\n")
        for offset, extensions in data:
            if counter % report_step == 0:
                cur_time = time.time()
                print(f"time: {cur_time - pre_time:.2f} count: {counter} stage: index")
            self.fidx.write(f"{offset} {' '.join(extensions)}\n")
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