#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Keep notebook stdout/stderr attached to CI and retain reports for the final
# process_sanitizers_logs check. Diagnostic tools do not need sanitizer preloads.
set -u

topdir=$1
python_major=$2
notebook=$3
overall_status=0

run_notebook() (
    capture_fd=$1
    repetition=$2
    kernel_args=()
    if [ "$capture_fd" != default ]; then
        # This option belongs to the kernel, not the nbconvert application.
        kernel_args+=("--ExecutePreprocessor.extra_arguments=[\"--IPKernelApp.capture_fd_output=$capture_fd\"]")
    fi
    log_dir=$(LD_PRELOAD= mktemp -d "$topdir/notebook-sanitizer.XXXXXX") || exit $?
    export ASAN_OPTIONS="${ASAN_OPTIONS:-}:log_path=$log_dir/sanitizer.log"

    printf '\n=== Notebook start: %s ===\n' "$notebook"
    printf 'FD capture: %s; repetition: %s\n' "$capture_fd" "$repetition"
    LD_PRELOAD= date -u '+UTC: %Y-%m-%dT%H:%M:%SZ'
    printf 'Wrapper PID: %s; sanitizer logs: %s\n' "$$" "$log_dir"
    started=$SECONDS
    status=0
    jupyter nbconvert --to notebook --inplace --execute \
        --ExecutePreprocessor.kernel_name=python"$python_major" \
        --ExecutePreprocessor.timeout=300 "${kernel_args[@]}" "$notebook" || status=$?

    printf '\n=== Notebook end: %s; exit: %s; elapsed: %ss ===\n' \
        "$notebook" "$status" "$((SECONDS - started))"
    printf 'FD capture: %s; repetition: %s\n' "$capture_fd" "$repetition"
    LD_PRELOAD= date -u '+UTC: %Y-%m-%dT%H:%M:%SZ'

    # The last log_path option takes precedence, so nbconvert and its kernel write
    # into this notebook-specific directory regardless of their working directories.
    shopt -s nullglob
    logs=("$log_dir"/sanitizer.log.*)
    printf 'Sanitizer reports for %s: %s\n' "$notebook" "${#logs[@]}"
    for log in "${logs[@]}"; do
        printf '\n--- %s ---\n' "$log"
        LD_PRELOAD= cat "$log"
    done

    exit "$status"
)

if [ "${notebook#./}" = sequence_processing/video/video_reader_simple/pipeline_mode.ipynb ]; then
    # Alternate fresh kernels to investigate the intermittent os_read_impl leak.
    # Keep all reports (including the baseline) for the final sanitizer check.
    for repetition in 1 2 3; do
        for capture_fd in True False; do
            run_notebook "$capture_fd" "$repetition" || overall_status=$?
        done
    done
else
    run_notebook default 1 || overall_status=$?
fi

exit "$overall_status"
