#!/bin/bash -e

optflow_so=`find /usr -name libnvidia-opticalflow.so`
optflow_so_1=`find /usr -name libnvidia-opticalflow.so.1`
if [ -z $optflow_so ]; then
    if [ ! -z $optflow_so_1 ]; then
        echo "INFO: Could not find libnvidia-opticalflow.so"
        echo "INFO: libnvidia-opticalflow.so.1 found: $optflow_so_1"
        echo "IFNO: Creating a symlink $optflow_so_1 => $optflow_so"
        optflow_so=`dirname $optflow_so_1`/libnvidia-opticalflow.so
        ln -s $optflow_so_1 $optflow_so
    else
        echo "ERROR: Cannot find libnvidia-opticalflow.so or libnvidia-opticalflow.so.1"
        exit 1
    fi
fi
if ! readelf -s $optflow_so | grep NvOFAPICreateInstanceCuda > /dev/null; then
    echo "ERROR: $optflow_so does not contain the API entry point NvOFAPICreateInstanceCuda"
    exit 1
fi
