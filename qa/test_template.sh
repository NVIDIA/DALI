#!/bin/bash

set -o pipefail
if [ -n "$gather_pip_packages" ]
then
    # perl breaks the population of the outside variables from the inside of the sourced
    # script. Turn this off to the gather_pip_packages process
    source qa/test_template_impl.sh
else
    source qa/test_template_impl.sh 2>&1 | perl -pe 'use POSIX strftime;
                    $|=1;
                    select((select(STDERR), $| = 1)[0]);
                    print strftime "[%Y-%m-%d %H:%M:%S] ", localtime'
fi
