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
import threading
import time
import urllib.error
import urllib.request
import uuid

from nose_utils import SkipTest

# Escape hatch: points the same tests at minio or at real S3 instead of at the mock server.
EXTERNAL_ENDPOINT = os.environ.get("DALI_TEST_S3_ENDPOINT")


def _cred(dali_var, aws_var, mock_default):
    """The mock server accepts anything, a real endpoint does not.

    DALI_TEST_S3_* wins, then whatever the environment already carries for the AWS SDK. Against an
    external endpoint an unresolved value stays unset instead of falling back to the mock default,
    so that both SDKs resolve it through their own credential chain: a dummy AWS_ACCESS_KEY_ID
    exported here would shadow a profile, an SSO session or an EC2 instance role.
    """
    value = os.environ.get(dali_var) or os.environ.get(aws_var)
    return value or (None if EXTERNAL_ENDPOINT else mock_default)


ACCESS_KEY = _cred("DALI_TEST_S3_ACCESS_KEY", "AWS_ACCESS_KEY_ID", "dalitestaccesskey")
SECRET_KEY = _cred("DALI_TEST_S3_SECRET_KEY", "AWS_SECRET_ACCESS_KEY", "dalitestsecretkey")
# Not a credential, and CreateBucket has to state it, so this one always has a value.
REGION = (
    os.environ.get("DALI_TEST_S3_REGION")
    or os.environ.get("AWS_DEFAULT_REGION")
    or os.environ.get("AWS_REGION")
    or "us-east-1"
)
# The tests create this bucket if it is missing, and remove only what they uploaded - never the
# bucket itself, which a concurrent run may still be using and which a killed run cannot clean up
# either. Point DALI_TEST_S3_BUCKET at a bucket you do not mind keeping around.
BUCKET = os.environ.get("DALI_TEST_S3_BUCKET", "dali-test-bucket")

# Everything this module uploads goes under a prefix unique to the process, so that a run against
# a shared endpoint neither collides with a concurrent run nor inherits its leftovers.
PREFIX = f"dali-test-{uuid.uuid4().hex[:12]}"

# A bucket nothing ever creates, for the error paths. Derived from the per-run uuid rather than
# hardcoded because bucket names are one global namespace: a fixed name may be owned by somebody
# else on real S3, which answers AccessDenied and not NoSuchBucket, or be left over on a shared
# MinIO, where listing it would simply succeed.
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
    entries = [e for e in no_proxy.split(",") if e]
    for host in ("localhost", "127.0.0.1"):
        if host not in entries:
            entries.insert(0, host)
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


class ExternalS3Server:
    """Escape hatch: DALI_TEST_S3_ENDPOINT points the same tests at minio or at real S3."""

    def __init__(self, endpoint_url):
        self.endpoint_url = endpoint_url

    def start(self):
        return self.endpoint_url

    def stop(self):
        pass


def make_server():
    return ExternalS3Server(EXTERNAL_ENDPOINT) if EXTERNAL_ENDPOINT else MockS3Server()


def export_s3_env(endpoint_url):
    """Exports what DALI's S3 client reads. MUST run before the first s3:// access: DALI builds
    one process-wide S3 client from a getenv snapshot (dali/util/s3_client_manager.h:33-41).

    Only what this module actually resolved is exported: against an external endpoint the
    credentials belong to the SDK's own chain, so nothing here may overwrite them.
    """
    os.environ["AWS_ENDPOINT_URL"] = endpoint_url
    os.environ["AWS_DEFAULT_REGION"] = REGION
    os.environ["AWS_REGION"] = REGION
    if ACCESS_KEY and SECRET_KEY:
        os.environ["AWS_ACCESS_KEY_ID"] = ACCESS_KEY
        os.environ["AWS_SECRET_ACCESS_KEY"] = SECRET_KEY
    if not EXTERNAL_ENDPOINT:  # setting it would cut off an EC2 instance role
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
        pass  # a shared endpoint may already have it; our keys are prefixed anyway


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


def delete_prefix(client, bucket, prefix):
    """Deletes every object under `prefix`. Paginates: a page holds at most 1000 keys, which is
    also as many as DeleteObjects takes per request."""
    paginator = client.get_paginator("list_objects_v2")
    errors = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        keys = [{"Key": obj["Key"]} for obj in page.get("Contents", [])]
        if keys:
            # a key that cannot be deleted is reported in the response, not raised
            response = client.delete_objects(Bucket=bucket, Delete={"Objects": keys})
            errors += response.get("Errors", [])
    if errors:
        raise RuntimeError(f"could not delete {len(errors)} object(s) under {prefix}: {errors[:5]}")


def skip_if_no_mock_server():
    if os.environ.get("DALI_ENABLE_SANITIZERS"):
        raise SkipTest("the S3 tests are not run under sanitizers")
    # boto3 seeds the bucket in either mode; only the mock server itself is optional.
    for mod in ["boto3"] if EXTERNAL_ENDPOINT else ["boto3", "moto.server"]:
        try:
            __import__(mod)
        except ImportError:
            raise SkipTest(f"{mod} is required to run the S3 tests")


def skip_if_no_s3_support():
    """DALI built with BUILD_AWSSDK=OFF raises a fixed message (discover_files.cc:121).
    Must be called AFTER export_s3_env, otherwise the probe talks to real AWS.

    The probe uses file_root, not files: only file_root makes the loader list the bucket while
    the pipeline is being built, and listing is what raises on a BUILD_AWSSDK=OFF build. With
    files= nothing is opened until the pipeline runs, so the probe would pass and the tests
    would fail instead of being skipped."""
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
            raise SkipTest("DALI was built without AWS S3 storage support (BUILD_AWSSDK=OFF)")
