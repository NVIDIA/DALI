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

"""Helpers for running DALI tests against a mock S3 server started by the test itself."""

import atexit
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

from nose_utils import SkipTest

ACCESS_KEY = os.environ.get("DALI_TEST_S3_ACCESS_KEY", "dalitestaccesskey")
SECRET_KEY = os.environ.get("DALI_TEST_S3_SECRET_KEY", "dalitestsecretkey")
REGION = "us-east-1"
BUCKET = os.environ.get("DALI_TEST_S3_BUCKET", "dali-test-bucket")

# The mock server intentionally runs in a SEPARATE PROCESS. Pipeline.Build() is bound without
# py::call_guard<py::gil_scoped_release> (dali/python/backend_impl.cc:2512, 2520 - unlike Run at
# :2548 and Shutdown at :2521), and S3 listing happens inside Build(), so an in-process server
# thread is GIL-starved and DALI eventually fails with "curlCode: 28, Timeout was reached".
_SERVER_MAIN = r"""
import ctypes, logging, os, signal, sys

try:  # die with the parent even if it is SIGKILLed and never closes our stdin
    ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG
except Exception:
    pass
if os.getppid() == 1:
    sys.exit(0)

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
    entries = [e for e in no_proxy.split(",") if e]
    for host in ("localhost", "127.0.0.1"):
        if host not in entries:
            entries.insert(0, host)
    return ",".join(entries)


class MockS3Server:
    """A mock S3 server in a subprocess, bound to an ephemeral port on 127.0.0.1.

    The endpoint MUST be an IP literal: aws-sdk-cpp defaults to virtual-host addressing and DALI
    never sets useVirtualAddressing=false (dali/util/s3_client_manager.h:58-72), so
    "http://localhost:<port>" would be addressed as "http://<bucket>.localhost:<port>".
    """

    def __init__(self, startup_timeout_s=60):
        self._proc = None
        self._startup_timeout_s = startup_timeout_s
        self.endpoint_url = None

    def start(self):
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        env["NO_PROXY"] = env["no_proxy"] = _with_loopback(env.get("no_proxy", ""))
        self._proc = subprocess.Popen(
            [sys.executable, "-c", _SERVER_MAIN],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None if os.environ.get("DALI_TEST_S3_VERBOSE") else subprocess.DEVNULL,
            env=env,
            text=True,
        )
        atexit.register(self.stop)
        line = self._proc.stdout.readline()  # the child reports the port it bound
        if not line.strip().isdigit():
            self.stop()
            raise RuntimeError(f"mock S3 server did not report a port, got {line!r}")
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


class ExternalS3Server:
    """Escape hatch: DALI_TEST_S3_ENDPOINT points the same tests at minio or at real S3."""

    def __init__(self, endpoint_url):
        self.endpoint_url = endpoint_url

    def start(self):
        return self.endpoint_url

    def stop(self):
        pass


def make_server():
    external = os.environ.get("DALI_TEST_S3_ENDPOINT")
    return ExternalS3Server(external) if external else MockS3Server()


def export_s3_env(endpoint_url):
    """Exports what DALI's S3 client reads. MUST run before the first s3:// access: DALI builds
    one process-wide S3 client from a getenv snapshot (dali/util/s3_client_manager.h:33-41)."""
    os.environ["AWS_ENDPOINT_URL"] = endpoint_url
    os.environ["AWS_ACCESS_KEY_ID"] = ACCESS_KEY
    os.environ["AWS_SECRET_ACCESS_KEY"] = SECRET_KEY
    os.environ["AWS_DEFAULT_REGION"] = REGION
    os.environ["AWS_REGION"] = REGION
    os.environ["AWS_EC2_METADATA_DISABLED"] = "true"
    os.environ["NO_PROXY"] = os.environ["no_proxy"] = _with_loopback(os.environ.get("no_proxy", ""))


def s3_client(endpoint_url):
    import boto3

    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        region_name=REGION,
    )


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


def skip_if_no_mock_server():
    if os.environ.get("DALI_ENABLE_SANITIZERS"):
        raise SkipTest("the S3 tests are not run under sanitizers")
    if os.environ.get("DALI_TEST_S3_ENDPOINT"):
        return
    for mod in ("boto3", "moto.server"):
        try:
            __import__(mod)
        except ImportError:
            raise SkipTest(f"{mod} is required to run the S3 tests")


def skip_if_no_s3_support():
    """DALI built with BUILD_AWSSDK=OFF raises a fixed message (dali/util/file.cc:37).
    Must be called AFTER export_s3_env, otherwise the probe talks to real AWS."""
    import nvidia.dali.fn as fn
    from nvidia.dali import pipeline_def

    @pipeline_def(batch_size=1, num_threads=1, device_id=None)
    def probe():
        return tuple(fn.readers.file(files=["s3://dali-no-such-bucket/no-such-key"]))

    try:
        probe().build()
    except Exception as e:
        if "not built with AWS S3 storage support" in str(e):
            raise SkipTest("DALI was built without AWS S3 storage support (BUILD_AWSSDK=OFF)")
