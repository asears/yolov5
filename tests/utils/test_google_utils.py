from utils import google_utils
import pytest


class TestGoogleUtils:
    def test_gsutil_getsize(self):
        url = "http://127.0.0.1"
        google_utils.gsutil_getsize(url)

    def test_attempt_download(self):
        weights = "test"
        google_utils.attempt_download(weights)

    def test_gdrive_download(self):
        id = "abc"
        name = "123"
        google_utils.gdrive_download(id, name)

    def test_get_token(self):
        cookie = "test"
        google_utils.get_token(cookie)
