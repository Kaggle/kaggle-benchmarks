# This test is vendored from docker-python
# (https://github.com/Kaggle/docker-python/blob/main/tests/test_user_secrets.py)
# and modified for the kaggle-benchmarks environment:
#
# - Replaced `test.support.os_helper.EnvironmentVarGuard` with `unittest.mock.patch.dict`
#   (python:3.11-slim strips the `test` module)
#   Affected: _test_client, test_no_token_fails, test_get_secret_validates_label
#
# - Mocked `subprocess.run` to avoid requiring the gcloud CLI
#   Affected: test_set_gcloud_credentials_succeeds
#
# - Updated credential path from /tmp/ to ~/ to match HOME=/root
#   (docker-python sets HOME=/tmp)
#   Affected: test_set_tensorflow_credential

import json
import os
import threading
import unittest
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import Mock, patch
from unittest.mock import patch as _patch
from urllib.parse import urlparse

from kaggle_secrets import GcpTarget, NotFoundError, UserSecretsClient, ValidationError
from kaggle_web_client import (
    _KAGGLE_URL_BASE_ENV_VAR_NAME,
    _KAGGLE_USER_SECRETS_TOKEN_ENV_VAR_NAME,
    BackendError,
    CredentialError,
)

_TEST_JWT = "test-secrets-key"


class UserSecretsHTTPHandler(BaseHTTPRequestHandler):
    def set_request(self):
        raise NotImplementedError()

    def get_response(self):
        raise NotImplementedError()

    def do_HEAD(s):
        s.send_response(200)

    def do_POST(s):
        s.set_request()
        s.send_response(200)
        s.send_header("Content-type", "application/json")
        s.end_headers()
        s.wfile.write(json.dumps(s.get_response()).encode("utf-8"))


class TestUserSecrets(unittest.TestCase):
    SERVER_ADDRESS = urlparse(
        os.getenv(_KAGGLE_URL_BASE_ENV_VAR_NAME, default="http://127.0.0.1:0")
    )

    def _test_client(
        self, client_func, expected_path, expected_body, secret=None, success=True
    ):
        _request = {}

        class AccessTokenHandler(UserSecretsHTTPHandler):
            def set_request(self):
                _request["path"] = self.path
                content_len = int(self.headers.get("Content-Length"))
                _request["body"] = json.loads(self.rfile.read(content_len))
                _request["headers"] = self.headers

            def get_response(self):
                if success:
                    return {
                        "result": {
                            "secret": secret,
                            "secretType": "refreshToken",
                            "secretProvider": "google",
                            "expiresInSeconds": 3600,
                        },
                        "wasSuccessful": "true",
                    }
                else:
                    return {
                        "wasSuccessful": "false",
                        "errors": ["No user secrets exist for kernel"],
                    }

        with _patch.dict(
            os.environ, {_KAGGLE_USER_SECRETS_TOKEN_ENV_VAR_NAME: _TEST_JWT}
        ):
            with HTTPServer(
                (self.SERVER_ADDRESS.hostname, self.SERVER_ADDRESS.port),
                AccessTokenHandler,
            ) as httpd:
                threading.Thread(target=httpd.serve_forever).start()
                try:
                    os.environ[_KAGGLE_URL_BASE_ENV_VAR_NAME] = (
                        "http://"
                        + httpd.server_address[0]
                        + ":"
                        + str(httpd.server_address[1])
                    )
                    client_func()
                finally:
                    httpd.shutdown()

                path, body = _request["path"], _request["body"]
                self.assertEqual(
                    path,
                    expected_path,
                    msg="Fake server did not receive the right request from the UserSecrets client.",
                )
                self.assertEqual(
                    body,
                    expected_body,
                    msg="Fake server did not receive the right body from the UserSecrets client.",
                )

    def test_no_token_fails(self):
        env = os.environ.copy()
        env.pop(_KAGGLE_USER_SECRETS_TOKEN_ENV_VAR_NAME, None)
        with _patch.dict(os.environ, env, clear=True):
            with self.assertRaises(CredentialError):
                UserSecretsClient()

    def test_get_secret_succeeds(self):
        secret = "12345"

        def call_get_secret():
            client = UserSecretsClient()
            secret_response = client.get_secret("secret_label")
            self.assertEqual(secret_response, secret)

        self._test_client(
            call_get_secret,
            "/requests/GetUserSecretByLabelRequest",
            {"Label": "secret_label"},
            secret=secret,
        )

    def test_get_secret_handles_unsuccessful(self):
        def call_get_secret():
            client = UserSecretsClient()
            with self.assertRaises(BackendError):
                client.get_secret("secret_label")

        self._test_client(
            call_get_secret,
            "/requests/GetUserSecretByLabelRequest",
            {"Label": "secret_label"},
            success=False,
        )

    def test_get_secret_validates_label(self):
        with _patch.dict(
            os.environ, {_KAGGLE_USER_SECRETS_TOKEN_ENV_VAR_NAME: _TEST_JWT}
        ):
            client = UserSecretsClient()
            with self.assertRaises(ValidationError):
                client.get_secret("")

    def test_get_gcloud_secret_succeeds(self):
        secret = '{"client_id":"gcloud","type":"authorized_user"}'

        def call_get_secret():
            client = UserSecretsClient()
            secret_response = client.get_gcloud_credential()
            self.assertEqual(secret_response, secret)

        self._test_client(
            call_get_secret,
            "/requests/GetUserSecretByLabelRequest",
            {"Label": "__gcloud_sdk_auth__"},
            secret=secret,
        )

    def test_get_gcloud_secret_handles_unsuccessful(self):
        def call_get_secret():
            client = UserSecretsClient()
            with self.assertRaises(NotFoundError):
                client.get_gcloud_credential()

        self._test_client(
            call_get_secret,
            "/requests/GetUserSecretByLabelRequest",
            {"Label": "__gcloud_sdk_auth__"},
            success=False,
        )

    @patch("subprocess.run")
    def test_set_gcloud_credentials_succeeds(self, mock_run):
        secret = '{"client_id":"gcloud","type":"authorized_user","refresh_token":"refresh_token"}'
        project = "foo"
        account = "bar"

        mock_run.return_value = Mock(returncode=0)

        def test_fn():
            client = UserSecretsClient()
            client.set_gcloud_credentials(project=project, account=account)

            self.assertEqual(project, os.environ["GOOGLE_CLOUD_PROJECT"])
            self.assertEqual(account, os.environ["GOOGLE_ACCOUNT"])

            expected_creds_file = os.path.join(
                os.environ["HOME"], "gcloud_credential.json"
            )
            self.assertEqual(
                expected_creds_file, os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
            )

            with open(expected_creds_file, "r") as f:
                self.assertEqual(secret, "\n".join(f.readlines()))

            # Verify gcloud was called
            self.assertTrue(mock_run.called)

        self._test_client(
            test_fn,
            "/requests/GetUserSecretByLabelRequest",
            {"Label": "__gcloud_sdk_auth__"},
            secret=secret,
        )

    def test_set_tensorflow_credential(self):
        secret = '{"client_id":"gcloud","type":"authorized_user","refresh_token":"refresh_token"}'

        def test_fn():
            client = UserSecretsClient()
            creds = client.get_gcloud_credential()
            client.set_tensorflow_credential(creds)

            expected_creds_file = os.path.join(
                os.environ["HOME"], "gcloud_credential.json"
            )
            self.assertEqual(
                expected_creds_file, os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
            )

            with open(expected_creds_file, "r") as f:
                self.assertEqual(secret, "\n".join(f.readlines()))

        self._test_client(
            test_fn,
            "/requests/GetUserSecretByLabelRequest",
            {"Label": "__gcloud_sdk_auth__"},
            secret=secret,
        )

    @patch("kaggle_secrets.datetime")
    def test_get_access_token_succeeds(self, mock_dt):
        secret = "12345"
        now = datetime(1993, 4, 24)
        mock_dt.utcnow = Mock(return_value=now)

        def call_get_bigquery_access_token():
            client = UserSecretsClient()
            secret_response = client.get_bigquery_access_token()
            self.assertEqual(secret_response, (secret, now + timedelta(seconds=3600)))

        def call_get_gcs_access_token():
            client = UserSecretsClient()
            secret_response = client._get_gcs_access_token()
            self.assertEqual(secret_response, (secret, now + timedelta(seconds=3600)))

        def call_get_cloudai_access_token():
            client = UserSecretsClient()
            secret_response = client._get_cloudai_access_token()
            self.assertEqual(secret_response, (secret, now + timedelta(seconds=3600)))

        self._test_client(
            call_get_bigquery_access_token,
            "/requests/GetUserSecretRequest",
            {"Target": GcpTarget.BIGQUERY.target},
            secret=secret,
        )
        self._test_client(
            call_get_gcs_access_token,
            "/requests/GetUserSecretRequest",
            {"Target": GcpTarget.GCS.target},
            secret=secret,
        )
        self._test_client(
            call_get_cloudai_access_token,
            "/requests/GetUserSecretRequest",
            {"Target": GcpTarget.CLOUDAI.target},
            secret=secret,
        )

    def test_get_access_token_handles_unsuccessful(self):
        def call_get_access_token():
            client = UserSecretsClient()
            with self.assertRaises(BackendError):
                client.get_bigquery_access_token()

        self._test_client(
            call_get_access_token,
            "/requests/GetUserSecretRequest",
            {"Target": GcpTarget.BIGQUERY.target},
            success=False,
        )
