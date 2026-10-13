import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

from weather_risk.cache import Cache, CacheError


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def test_offline_replay_does_not_fetch_and_verifies_checksum(self):
        data, original = Cache(self.root).get_or_create({"series": "x"}, lambda: b"one")
        replay = Cache(self.root, offline=True)
        again, metadata = replay.get_or_create({"series": "x"}, lambda: self.fail("network"))
        self.assertEqual(data, again)
        self.assertEqual(original["retrieved_at"], metadata["retrieved_at"])
        self.assertTrue(metadata["cache_hit"])
        (self.root / "blobs" / original["sha256"]).write_bytes(b"tampered")
        with self.assertRaisesRegex(CacheError, "Corrupt"):
            Cache(self.root, offline=True).get_or_create({"series": "x"}, lambda: b"never")

    def test_refresh_fetches_shared_request_once_per_run(self):
        cache = Cache(self.root, refresh=True)
        first, meta = cache.get_or_create({"series": "x"}, lambda: b"one")
        again, repeated = cache.get_or_create({"series": "x"}, lambda: self.fail("refetched"))
        self.assertEqual(first, again)
        self.assertEqual(meta["retrieved_at"], repeated["retrieved_at"])

    def test_manifest_replay_survives_pointer_refresh(self):
        old_cache = Cache(self.root)
        old_cache.get_or_create({"series": "x"}, lambda: b"one")
        Cache(self.root, refresh=True).get_or_create({"series": "x"}, lambda: b"two")
        replay = Cache(self.root, offline=True, pinned_records=old_cache.records)
        data, _ = replay.get_or_create({"series": "x"}, lambda: self.fail("network"))
        self.assertEqual(data, b"one")
        with self.assertRaises(CacheError):
            replay.get_or_create({"series": "not-in-manifest"}, lambda: b"never")

    def test_refresh_preserves_previous_content_blob(self):
        first, old = Cache(self.root).get_or_create({"series": "x"}, lambda: b"one")
        second, new = Cache(self.root, refresh=True).get_or_create({"series": "x"}, lambda: b"two")
        self.assertNotEqual(old["sha256"], new["sha256"])
        self.assertEqual((self.root / "blobs" / old["sha256"]).read_bytes(), first)
        self.assertEqual((self.root / "blobs" / new["sha256"]).read_bytes(), second)

    def test_offline_miss_and_incompatible_flags(self):
        with self.assertRaisesRegex(CacheError, "Offline cache miss"):
            Cache(self.root, offline=True).get("https://example.test/data")
        with self.assertRaises(ValueError):
            Cache(self.root, offline=True, refresh=True)

    def test_api_key_is_excluded_from_identity_and_redacted_from_bytes(self):
        secret = "private-demo-key"
        body = json.dumps({"request": {"api_key": secret}, "data": [1, 2]}).encode()
        cache = Cache(self.root)
        with patch("weather_risk.cache.urlopen", return_value=io.BytesIO(body)) as request:
            data, metadata = cache.get("https://example.test/data", {"api_key": secret, "series": "x"})
        self.assertEqual(request.call_count, 1)
        self.assertNotIn(secret.encode(), data)
        self.assertEqual(metadata["sha256"], hashlib.sha256(data).hexdigest())
        self.assertNotIn("api_key", json.dumps(metadata))
        replay, _ = Cache(self.root, offline=True).get(
            "https://example.test/data?series=x&api_key=other-key")
        self.assertEqual(replay, data)
        for path in self.root.rglob("*"):
            if path.is_file():
                self.assertNotIn(secret.encode(), path.read_bytes())

    def test_error_never_exposes_credential_and_does_not_retry_http_401(self):
        url = "https://example.test/data?api_key=very-private"
        error = HTTPError(url, 401, "unauthorized very-private", {}, None)
        with patch("weather_risk.cache.urlopen", side_effect=error) as request:
            with self.assertRaises(CacheError) as caught:
                Cache(self.root).get(url)
        self.assertEqual(request.call_count, 1)
        self.assertNotIn("very-private", str(caught.exception))
        self.assertIsNone(caught.exception.__cause__)

    def test_retry_is_bounded_and_empty_payload_fails(self):
        with patch("weather_risk.cache.urlopen", side_effect=TimeoutError) as request:
            with patch("weather_risk.cache.time.sleep"):
                with self.assertRaises(CacheError):
                    Cache(self.root).get("https://example.test/data")
        self.assertEqual(request.call_count, 3)
        with self.assertRaisesRegex(CacheError, "Empty"):
            Cache(self.root).get_or_create({"series": "empty"}, lambda: b"")


if __name__ == "__main__":
    unittest.main()
