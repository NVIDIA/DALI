#!/bin/bash

set -o pipefail
source /opt/dali/qa/test_template_impl.sh 2>&1 | perl -pe 'use POSIX strftime;
                    $|=1;
                    select((select(STDERR), $| = 1)[0]);
                    print strftime "[%Y-%m-%d %H:%M:%S] ", localtime'
