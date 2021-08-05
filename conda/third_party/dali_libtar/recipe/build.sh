#!/bin/bash

autoreconf --force --install
./configure --prefix=$PREFIX --disable-debug --disable-dependency-tracking
make install
