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

"""Helpers for running DALI tests against a mock S3 server started by the test itself.

Environment variables, all optional:
    DALI_TEST_S3_VERBOSE       - if set, don't silence the mock server's own stdout/stderr.
"""

import atexit
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid

# The mock server accepts any credentials/region; these are fixed rather than configurable since
# nothing here talks to a real endpoint.
ACCESS_KEY = "dalitestaccesskey"
SECRET_KEY = "dalitestsecretkey"
REGION = "us-east-1"
BUCKET = "dali-test-bucket"

# Everything this module uploads goes under a prefix unique to the process, even though each test
# run gets its own private mock server - keeps the object keys self-documenting and collision-proof
# if this module is ever imported more than once in the same interpreter.
PREFIX = f"dali-test-{uuid.uuid4().hex[:12]}"

# A bucket nothing ever creates, for the error paths.
MISSING_BUCKET = f"{PREFIX}-missing"

# The mock server intentionally runs in a SEPARATE PROCESS. In dali/python/backend_impl.cc,
# Pipeline.Build() is bound without py::call_guard<py::gil_scoped_release> (unlike Run and
# Shutdown), and S3 listing happens inside Build(), so an in-process server thread is
# GIL-starved and DALI eventually fails with "curlCode: 28, Timeout was reached".
_SERVER_MAIN = r"""
import ctypes, logging, os, signal, sys

try:  # die with the parent even if it is SIGKILLed and never closes our stdin
    ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG
except Exception:
    pass
if os.getppid() == 1:
    sys.exit(0)

# werkzeug is the WSGI server moto.server runs on; it logs every request at INFO by default.
logging.getLogger("werkzeug").setLevel(logging.ERROR)
from moto.server import ThreadedMotoServer

server = ThreadedMotoServer(ip_address="127.0.0.1", port=0, verbose=False)
server.start()
print(server.get_host_and_port()[1], flush=True)
os.dup2(os.open(os.devnull, os.O_WRONLY), 1)  # nothing else may ever fill the pipe
try:
    sys.stdin.read()   # EOF when the parent exits or closes the pipe
finally:
    server.stop()
"""


def _with_loopback(no_proxy):
    entries = set(e for e in no_proxy.split(",") if e)
    entries.update(["localhost", "127.0.0.1"])
    return ",".join(entries)


class MockS3Server:
    """A mock S3 server in a subprocess, bound to an ephemeral port on 127.0.0.1.

    The endpoint MUST be an IP literal: aws-sdk-cpp defaults to virtual-host addressing and DALI
    never sets useVirtualAddressing=false in dali/util/s3_client_manager.h, so
    "http://localhost:<port>" would be addressed as "http://<bucket>.localhost:<port>".
    """

    def __init__(self, startup_timeout_s=60):
        self._proc = None
        self._startup_timeout_s = startup_timeout_s
        self.endpoint_url = None

    def start(self):
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        # Both keys are set because consumers disagree on casing; both are read from for the same
        # reason - the environment may already carry a bypass list under either one.
        no_proxy = env.get("no_proxy") or env.get("NO_PROXY", "")
        env["NO_PROXY"] = env["no_proxy"] = _with_loopback(no_proxy)
        self._proc = subprocess.Popen(
            [sys.executable, "-c", _SERVER_MAIN],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            # Set DALI_TEST_S3_VERBOSE to see the mock server's own stdout/stderr.
            stderr=None if os.environ.get("DALI_TEST_S3_VERBOSE") else subprocess.DEVNULL,
            env=env,
            text=True,
        )
        atexit.register(self.stop)
        # A child that stalls before it prints - ThreadedMotoServer.start() waits forever on an
        # event that its server thread never sets if make_server() raises - would block this read
        # forever, and the deadline below is only armed afterwards. Killing the child closes the
        # pipe, which turns the read into the EOF that the check already reports.
        watchdog = threading.Timer(self._startup_timeout_s, self._proc.kill)
        watchdog.start()
        try:
            line = self._proc.stdout.readline()  # the child reports the port it bound
        finally:
            watchdog.cancel()
        if not line.strip().isdigit():
            status = self._proc.poll()  # -9 when the watchdog killed it
            self.stop()
            raise RuntimeError(
                f"mock S3 server did not report a port, got {line!r} (exit {status})"
            )
        self.endpoint_url = f"http://127.0.0.1:{int(line)}"
        self._wait_until_serving()
        return self.endpoint_url

    def _wait_until_serving(self):
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        deadline = time.monotonic() + self._startup_timeout_s
        last_err = None
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError(f"mock S3 server exited with {self._proc.returncode}")
            try:
                opener.open(self.endpoint_url + "/", timeout=2)
                return
            except urllib.error.HTTPError:
                return  # any HTTP response means it is serving
            except Exception as e:
                last_err = e
                time.sleep(0.05)
        self.stop()
        raise RuntimeError(f"mock S3 server not ready in {self._startup_timeout_s}s: {last_err}")

    def stop(self):
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            proc.stdin.close()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
        finally:
            proc.stdout.close()


def make_server():
    return MockS3Server()


def export_s3_env(endpoint_url):
    """Exports what DALI's S3 client reads. MUST run before the first s3:// access: DALI builds
    one process-wide S3 client from a getenv snapshot (dali/util/s3_client_manager.h:33-41).
    """
    os.environ["AWS_ENDPOINT_URL"] = endpoint_url
    os.environ["AWS_DEFAULT_REGION"] = REGION
    os.environ["AWS_REGION"] = REGION
    os.environ["AWS_ACCESS_KEY_ID"] = ACCESS_KEY
    os.environ["AWS_SECRET_ACCESS_KEY"] = SECRET_KEY
    os.environ["AWS_EC2_METADATA_DISABLED"] = "true"
    os.environ["NO_PROXY"] = os.environ["no_proxy"] = _with_loopback(os.environ.get("no_proxy", ""))


def s3_client(endpoint_url):
    import boto3

    # None means "resolve it yourself": botocore builds explicit credentials only when it gets
    # both the key and the secret, and otherwise falls back to the chain that honours
    # AWS_SESSION_TOKEN, profiles and instance roles - none of which fit in these two arguments.
    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        region_name=REGION,
    )


def create_bucket(client, bucket):
    """Creates `bucket` unless the endpoint already has it.

    The location has to be stated exactly once: outside us-east-1 a region-specific endpoint
    rejects a CreateBucket that omits it (IllegalLocationConstraintException), while us-east-1
    must NOT be sent as a LocationConstraint (InvalidLocationConstraint). Real S3, MinIO and moto
    all follow that rule.
    """
    kwargs = {"Bucket": bucket}
    if REGION != "us-east-1":
        kwargs["CreateBucketConfiguration"] = {"LocationConstraint": REGION}
    try:
        client.create_bucket(**kwargs)
    except client.exceptions.BucketAlreadyOwnedByYou:
        pass


def upload_dir(client, bucket, local_dir, key_prefix):
    keys = []
    for dirpath, _, filenames in os.walk(local_dir):
        for name in sorted(filenames):
            local = os.path.join(dirpath, name)
            rel = os.path.relpath(local, local_dir).replace(os.sep, "/")
            key = f"{key_prefix.rstrip('/')}/{rel}"
            client.upload_file(local, bucket, key)
            keys.append(key)
    return keys


def require_mock_server():
    """boto3 and moto are required dependencies, not optional ones.

    A missing package is an error rather than a skip: every suite that runs
    dali/test/python/reader installs them (see
    qa/TL0_python-self-test-readers-decoders/test_nofw.sh). Were this a skip, the package
    silently disappearing from the environment would silently delete the coverage with it.
    Builds without S3 support are a separate, required precondition, checked by
    require_s3_support.
    """
    for mod in ["boto3", "moto.server"]:
        try:
            __import__(mod)
        except ImportError:
            raise RuntimeError(
                f"{mod} is required to run the S3 tests. It is installed by "
                f"qa/TL0_python-self-test-readers-decoders/test_nofw.sh; install it to run them."
            )


def require_s3_support():
    """DALI built with BUILD_AWSSDK=OFF raises a fixed message, in discover_files.cc.
    Must be called AFTER export_s3_env, otherwise the probe talks to real AWS.

    S3 support is required, not optional: BUILD_AWSSDK only turns itself off when the AWS SDK
    isn't found on the build machine (cmake/Dependencies.common.cmake), which is a build
    regression in any configuration this test suite runs in, not an intentional choice.

    The probe uses file_root, not files: only file_root makes the loader list the bucket while
    the pipeline is being built, and listing is what raises on a BUILD_AWSSDK=OFF build. With
    files= nothing is opened until the pipeline runs, so the probe would pass and the tests
    would fail instead of being caught here."""
    import nvidia.dali.fn as fn
    from nvidia.dali import pipeline_def

    @pipeline_def(batch_size=1, num_threads=1, device_id=None)
    def probe():
        return tuple(fn.readers.file(file_root=f"s3://{MISSING_BUCKET}/no-such-prefix"))

    try:
        probe().build()
    except Exception as e:
        # Every other failure - NoSuchBucket, "No files found.", a transport error - means the S3
        # code path is compiled in, which is all this probe asks. Deliberately not re-raised: a
        # build WITH S3 support has to fail here, and a broken endpoint or bad credentials show up
        # in setUpModule immediately afterwards anyway.
        if "not built with AWS S3 storage support" in str(e):
            raise RuntimeError(
                "DALI was built without AWS S3 storage support (BUILD_AWSSDK=OFF), which this "
                "test suite requires. If a build variant is intentionally without S3 support, "
                "exclude reader/test_s3.py from it explicitly instead of relying on a skip here."
            )
