#!/usr/bin/env python
import sys
import struct

if len(sys.argv) < 3:
    print("Usage: tfrecord2idx <tfrecord filename> <index filename>")
    exit()

f = open(sys.argv[1], 'rb')
idx = open(sys.argv[2], 'w')

while True:
    current = f.tell()
    try:
        # length
        byte_len = f.read(8)
        if len(byte_len) == 0:
            break
        # crc
        f.read(4)
        proto_len = struct.unpack('q', byte_len)[0]
        # proto
        f.read(proto_len)
        # crc
        f.read(4)
        idx.write(str(current) + ' ' + str(f.tell() - current) + '\n')
    except Exception:
        print("Not a valid TFRecord file")
        break

f.close()
idx.close()
