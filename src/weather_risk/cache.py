"""Verified, content-addressed response cache; credentials are never persisted.

SHA-256 identifies stored bytes. If a server echoes a supplied credential, that
credential is redacted before persistence, so the hash is of sanitized bytes.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen


class CacheError(RuntimeError):
    """Cache miss, corrupt snapshot, or unavailable upstream response."""


_SECRET_KEYS = {"api_key", "apikey", "token", "access_token", "authorization"}


def _safe_key(value):
    if isinstance(value, dict):
        return {k: _safe_key(v) for k, v in value.items()
                if str(k).lower() not in _SECRET_KEYS}
    if isinstance(value, (list, tuple)):
        return [_safe_key(v) for v in value]
    if isinstance(value, str) and value.startswith(("https://", "http://")):
        parts = urlsplit(value)
        if parts.username or parts.password:
            raise CacheError("Credentials in URL user information are unsupported")
        query = sorted((k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
                       if k.lower() not in _SECRET_KEYS)
        return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), ""))
    return value


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=".pending-")
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(data)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class Cache:
    def __init__(self, root: Path, offline: bool = False, refresh: bool = False,
                 pinned_records: list[dict] | None = None):
        if offline and refresh:
            raise ValueError("offline and refresh are mutually exclusive")
        self.root = Path(root)
        self.offline = offline
        self.refresh = refresh
        self.records: list[dict] = []
        self._memo: dict[str, tuple[bytes, dict]] = {}
        self._pinned = None if pinned_records is None else {r["request_id"]: r for r in pinned_records}
        if self._pinned is not None and (not offline or refresh):
            raise ValueError("Pinned snapshots require offline=True and refresh=False")
        for record in pinned_records or []:
            if record["sha256"] != self._pinned[record["request_id"]]["sha256"]:
                raise CacheError("Manifest has inconsistent snapshots for one request")

    def get_or_create(self, key: dict, fetch: Callable[[], bytes]) -> tuple[bytes, dict]:
        safe = _safe_key(key)
        encoded = json.dumps(safe, sort_keys=True, separators=(",", ":")).encode()
        request_id = hashlib.sha256(encoded).hexdigest()
        if request_id in self._memo:
            data, saved = self._memo[request_id]
            metadata = dict(saved, cache_hit=True)
            self.records.append(metadata.copy())
            return data, metadata
        pointer = self.root / "requests" / f"{request_id}.json"
        if self._pinned is not None or pointer.exists() and not self.refresh:
            try:
                metadata = (self._pinned[request_id] if self._pinned is not None
                            else json.loads(pointer.read_text()))
                digest = metadata["sha256"]
                if (len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)
                        or metadata["request"] != safe):
                    raise ValueError("invalid request pointer")
                data = (self.root / "blobs" / digest).read_bytes()
                if hashlib.sha256(data).hexdigest() != digest:
                    raise ValueError("content checksum mismatch")
                stamp = datetime.fromisoformat(metadata["retrieved_at"])
                if stamp.tzinfo is None:
                    raise ValueError("missing retrieval timezone")
            except (OSError, ValueError, KeyError, TypeError) as exc:
                raise CacheError(f"Corrupt cache entry {request_id}; use --refresh") from exc
            metadata = dict(metadata, cache_hit=True)
        else:
            if self.offline:
                raise CacheError(f"Offline cache miss for request {request_id}")
            data = fetch()
            if not isinstance(data, bytes) or not data:
                raise CacheError(f"Empty or non-byte response for request {request_id}")
            digest = hashlib.sha256(data).hexdigest()
            blob = self.root / "blobs" / digest
            if blob.exists() and blob.read_bytes() != data:
                raise CacheError(f"Corrupt existing blob {digest}; remove it before refresh")
            if not blob.exists():
                _atomic_write(blob, data)
            metadata = {"request_id": request_id, "request": safe, "sha256": digest,
                        "retrieved_at": datetime.now(timezone.utc).isoformat(),
                        "cache_hit": False}
            _atomic_write(pointer, json.dumps(metadata, indent=2, sort_keys=True).encode())
        self.records.append(metadata.copy())
        self._memo[request_id] = (data, metadata.copy())
        return data, metadata

    def get(self, url: str, params: dict | None = None) -> tuple[bytes, dict]:
        parts = urlsplit(url)
        if parts.scheme != "https":
            raise ValueError("Only HTTPS source URLs are supported")
        query = parse_qsl(parts.query, keep_blank_values=True)
        query += parse_qsl(urlencode(params or {}, doseq=True), keep_blank_values=True)
        full_url = urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), ""))
        secrets = [v for k, v in query if k.lower() in _SECRET_KEYS and v]
        safe_url = _safe_key(full_url)

        def fetch() -> bytes:
            reason = "network error"
            for attempt in range(3):
                try:
                    request = Request(full_url, headers={"User-Agent": "weather-risk-research/0.1"})
                    with urlopen(request, timeout=30) as response:
                        payload = response.read(50_000_001)
                    if len(payload) > 50_000_000:
                        raise CacheError("Response exceeded the 50 MB safety limit")
                    # EIA responses may echo request parameters. Preserve other bytes.
                    for secret in secrets:
                        for token in {secret, json.dumps(secret)[1:-1], urlencode({"x": secret})[2:]}:
                            payload = payload.replace(token.encode(), b"[REDACTED]")
                    return payload
                except HTTPError as exc:
                    reason = f"HTTP {exc.code}"
                    exc.close()
                    if exc.code not in {408, 429, 500, 502, 503, 504}:
                        break
                except (URLError, TimeoutError, OSError):
                    reason = "network/timeout error"
                if attempt < 2:
                    time.sleep(2 ** attempt)
            # Do not attach upstream exceptions, which may contain a secret URL.
            raise CacheError(f"Unable to fetch {safe_url}: {reason}") from None

        return self.get_or_create({"url": safe_url}, fetch)
