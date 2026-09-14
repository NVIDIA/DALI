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

"""Helpers for running DALI tests against a GCS emulator started by the test itself.

Seeding goes through plain urllib against the GCS JSON API, so the suite needs no pip packages
of its own. It also has to: a directory marker is a zero-byte object whose *name* ends with "/",
which no filesystem preload can express - only a REST upload can create one.
"""

import atexit
import json
import os
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

from nose_utils import SkipTest

# fake-gcs-server ignores the project, but the JSON API requires the parameter to be present.
PROJECT = "dali-test-project"
BUCKET = os.environ.get("DALI_TEST_GCS_BUCKET", "dali-test-bucket")

# Everything this module uploads goes under a prefix unique to the process, so that a run against
# a shared endpoint neither collides with a concurrent run nor inherits its leftovers.
PREFIX = f"dali-test-{uuid.uuid4().hex[:12]}"

SERVER_BINARY = os.environ.get("DALI_TEST_FAKE_GCS_SERVER", "fake-gcs-server")


def _with_loopback(no_proxy):
    entries = [e for e in no_proxy.split(",") if e]
    for host in ("localhost", "127.0.0.1"):
        if host not in entries:
            entries.insert(0, host)
    return ",".join(entries)


def _opener():
    # Never go through a proxy: the emulator is on loopback and a corporate proxy would swallow it.
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _request(method, url, data=None, content_type=None, timeout=30):
    request = urllib.request.Request(url, data=data, method=method)
    if content_type is not None:
        request.add_header("Content-Type", content_type)
    with _opener().open(request, timeout=timeout) as response:
        return response.status, response.read()


def _free_port():
    """Reserves a port by binding and immediately releasing it.

    fake-gcs-server accepts `-port 0`, but then logs the literal "http://127.0.0.1:0" and never
    reports the port it actually got, so the port has to be chosen here. The window between
    closing this socket and the server binding is a race in theory; start() retries around it.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class FakeGCSServer:
    """fake-gcs-server, in memory, bound to loopback.

    It runs as a SEPARATE PROCESS, and not merely because it happens to be a Go binary - do not
    replace it with an in-process Python emulator thread. Pipeline.Build() is bound without
    py::call_guard<py::gil_scoped_release> (dali/python/backend_impl.cc), and GCS object listing
    happens inside Build(), so an in-process server thread is GIL-starved and the request
    eventually times out in libcurl. The same failure mode applies to the S3 suite.

    Unlike S3, the endpoint does not have to be an IP literal: google-cloud-cpp puts the bucket in
    the request path and never uses virtual-host addressing.
    """

    def __init__(self, startup_timeout_s=60):
        self._proc = None
        self._startup_timeout_s = startup_timeout_s
        self.endpoint_url = None

    def start(self):
        binary = shutil.which(SERVER_BINARY) or SERVER_BINARY
        last_err = None
        for _ in range(5):  # retry: the port picked below may be taken before we bind it
            port = _free_port()
            self._proc = subprocess.Popen(
                [
                    binary,
                    "-scheme",
                    "http",
                    "-host",
                    "127.0.0.1",
                    "-port",
                    str(port),
                    "-backend",
                    "memory",
                    "-log-level",
                    "error",
                ],
                stdout=subprocess.DEVNULL,
                stderr=None if os.environ.get("DALI_TEST_GCS_VERBOSE") else subprocess.DEVNULL,
            )
            atexit.register(self.stop)
            self.endpoint_url = f"http://127.0.0.1:{port}"
            try:
                self._wait_until_serving()
                return self.endpoint_url
            except RuntimeError as e:
                last_err = e
                self.stop()
        raise RuntimeError(f"fake-gcs-server did not come up: {last_err}")

    def _wait_until_serving(self):
        deadline = time.monotonic() + self._startup_timeout_s
        last_err = None
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError(f"fake-gcs-server exited with {self._proc.returncode}")
            try:
                _request("GET", f"{self.endpoint_url}/storage/v1/b?project={PROJECT}", timeout=2)
                return
            except urllib.error.HTTPError:
                return  # any HTTP response means it is serving
            except Exception as e:
                last_err = e
                time.sleep(0.05)
        raise RuntimeError(f"not ready in {self._startup_timeout_s}s: {last_err}")

    def stop(self):
        proc, self._proc = self._proc, None
        if proc is None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)


class ExternalGCSServer:
    """Escape hatch: DALI_TEST_GCS_ENDPOINT points the tests at another emulator or at real GCS."""

    def __init__(self, endpoint_url):
        self.endpoint_url = endpoint_url

    def start(self):
        return self.endpoint_url

    def stop(self):
        pass


def make_server():
    external = os.environ.get("DALI_TEST_GCS_ENDPOINT")
    return ExternalGCSServer(external) if external else FakeGCSServer()


def export_gcs_env(endpoint_url):
    """Exports what DALI's GCS client reads. MUST run before the first gs:// access: DALI builds
    one process-wide client from a getenv snapshot (dali/util/gcs_client_manager.h)."""
    os.environ["DALI_GCS_ENDPOINT_URL"] = endpoint_url
    # google-cloud-cpp reads this one by itself; keeping both set means a helper that bypasses
    # DALI's own option plumbing still talks to the emulator rather than to real GCS.
    os.environ["CLOUD_STORAGE_EMULATOR_ENDPOINT"] = endpoint_url
    # The default is Application Default Credentials, which an emulator cannot satisfy.
    os.environ.setdefault("DALI_GCS_ANONYMOUS", "1")
    os.environ["NO_PROXY"] = os.environ["no_proxy"] = _with_loopback(os.environ.get("no_proxy", ""))
    if _is_loopback(endpoint_url):
        _drop_proxy_vars()


def _is_loopback(endpoint_url):
    host = urllib.parse.urlsplit(endpoint_url).hostname or ""
    return host in ("127.0.0.1", "::1", "localhost")


def _drop_proxy_vars():
    """Removes the proxy from this process, for a loopback endpoint.

    Setting `no_proxy` is not enough. google-cloud-cpp hands libcurl an explicit
    `CURLOPT_NOPROXY` of exactly "metadata.google.internal"
    (google/cloud/internal/curl_impl.cc), and that *replaces* the list libcurl would otherwise
    derive from `no_proxy`. So in a proxied environment every request except the metadata server
    is sent to the proxy in absolute-URI form - including the one addressed to the emulator on
    127.0.0.1, which the proxy cannot route back to us. The request never arrives and the
    listing inside Pipeline.build() retries until it times out.

    Only done for a loopback endpoint: an external DALI_TEST_GCS_ENDPOINT may well need the
    proxy to be reachable.
    """
    for var in ("http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        os.environ.pop(var, None)


def create_bucket(endpoint_url, bucket=BUCKET):
    """Creates the bucket, tolerating one that is already there (a shared endpoint may have it)."""
    try:
        _request(
            "POST",
            f"{endpoint_url}/storage/v1/b?project={PROJECT}",
            json.dumps({"name": bucket}).encode(),
            "application/json",
        )
    except urllib.error.HTTPError as e:
        if e.code != 409:
            raise


def put_object(endpoint_url, key, body, bucket=BUCKET):
    quoted = urllib.parse.quote(key, safe="")
    _request(
        "POST",
        f"{endpoint_url}/upload/storage/v1/b/{bucket}/o?uploadType=media&name={quoted}",
        body,
        "application/octet-stream",
    )


def put_directory_marker(endpoint_url, key, bucket=BUCKET):
    """Creates a zero-byte object whose name ends with '/' - what consoles show as a folder."""
    assert key.endswith("/"), key
    put_object(endpoint_url, key, b"", bucket)


def upload_dir(endpoint_url, local_dir, key_prefix, bucket=BUCKET):
    keys = []
    for dirpath, _, filenames in os.walk(local_dir):
        for name in sorted(filenames):
            local = os.path.join(dirpath, name)
            rel = os.path.relpath(local, local_dir).replace(os.sep, "/")
            key = f"{key_prefix.rstrip('/')}/{rel}"
            with open(local, "rb") as f:
                put_object(endpoint_url, key, f.read(), bucket)
            keys.append(key)
    return keys


def list_objects(endpoint_url, prefix, bucket=BUCKET):
    """Lists every object under `prefix`, following pageToken."""
    names = []
    page_token = None
    while True:
        url = f"{endpoint_url}/storage/v1/b/{bucket}/o?prefix={urllib.parse.quote(prefix, safe='')}"
        if page_token:
            url += f"&pageToken={urllib.parse.quote(page_token, safe='')}"
        _, body = _request("GET", url)
        payload = json.loads(body)
        names.extend(item["name"] for item in payload.get("items", []))
        page_token = payload.get("nextPageToken")
        if not page_token:
            break
    return names


def delete_prefix(endpoint_url, prefix, bucket=BUCKET):
    for name in list_objects(endpoint_url, prefix, bucket):
        try:
            quoted = urllib.parse.quote(name, safe="")
            _request("DELETE", f"{endpoint_url}/storage/v1/b/{bucket}/o/{quoted}")
        except urllib.error.HTTPError as e:
            if e.code != 404:
                raise


def delete_bucket(endpoint_url, bucket=BUCKET):
    try:
        _request("DELETE", f"{endpoint_url}/storage/v1/b/{bucket}")
    except urllib.error.HTTPError as e:
        if e.code != 404:
            raise


def skip_if_no_mock_server():
    if os.environ.get("DALI_ENABLE_SANITIZERS"):
        raise SkipTest("the GCS tests are not run under sanitizers")
    if os.environ.get("DALI_TEST_GCS_ENDPOINT"):
        return
    if shutil.which(SERVER_BINARY) is None and not os.path.isfile(SERVER_BINARY):
        raise SkipTest(f"{SERVER_BINARY} is required to run the GCS tests")


def skip_if_no_gcs_support():
    """DALI built without BUILD_GCS raises a fixed message (discover_files.cc).
    Must be called AFTER export_gcs_env, otherwise the probe talks to real GCS.

    The probe uses file_root, not files: only file_root makes the loader list the bucket while
    the pipeline is being built, and listing is what raises on a build without GCS support. With
    files= nothing is opened until the pipeline runs, so the probe would pass and the tests would
    fail instead of being skipped."""
    import nvidia.dali.fn as fn
    from nvidia.dali import pipeline_def

    @pipeline_def(batch_size=1, num_threads=1, device_id=None)
    def probe():
        return tuple(fn.readers.file(file_root="gs://dali-no-such-bucket/no-such-prefix"))

    try:
        probe().build()
    except Exception as e:
        if "not built with Google Cloud Storage support" in str(e):
            raise SkipTest("DALI was built without Google Cloud Storage support (BUILD_GCS=OFF)")
