"""
data_feeds/cache.py — TTL-based disk cache shared by all feed clients.

Design
------
All external API calls are decorated with @cached(ttl_seconds=N).
The cache key is a SHA-256 hash of the function name + all arguments,
so calls with different ISO-3 codes or indicator codes are cached separately.

TTL recommendations:
  World Bank indicators (GNI, WGI, Findex)  : 90 days  (annual release cycle)
  CryptoCompare exchange volume              : 24 hours (intraday moves don't matter)

The cache directory defaults to .cache/data_feeds/ relative to the caller's CWD.
Override by setting MARKETSPHERE_CACHE_DIR in the environment.

Cache files are plain JSON — human-readable, manually deleteable if needed.
"""

import functools
import hashlib
import json
import os
import time
from pathlib import Path

_DEFAULT_CACHE_DIR = ".cache/data_feeds"


def _ensure_cache_dir() -> Path:
    # Read env var at call time so monkeypatch.setenv works in tests
    cache_dir = Path(os.environ.get("MARKETSPHERE_CACHE_DIR", _DEFAULT_CACHE_DIR))
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def cached(ttl_seconds: int):
    """
    Decorator: cache the return value of fn to disk for ttl_seconds.

    The decorated function must return a JSON-serializable value.
    On cache miss (first call or TTL expired), fn is called and the result
    written to disk. On cache hit, the file is read and returned.

    Usage::

        @cached(ttl_seconds=90 * 86_400)   # 90 days
        def fetch_something(iso3: str) -> dict:
            ...
    """
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            key = hashlib.sha256(
                f"{fn.__module__}.{fn.__name__}|{args!r}|{sorted(kwargs.items())!r}".encode()
            ).hexdigest()
            cache_path = _ensure_cache_dir() / f"{key}.json"

            if cache_path.exists():
                age = time.time() - cache_path.stat().st_mtime
                if age < ttl_seconds:
                    return json.loads(cache_path.read_text())

            result = fn(*args, **kwargs)
            cache_path.write_text(json.dumps(result))
            return result

        # Attach a .bust() helper for tests that need a clean state
        def bust(*args, **kwargs):
            key = hashlib.sha256(
                f"{fn.__module__}.{fn.__name__}|{args!r}|{sorted(kwargs.items())!r}".encode()
            ).hexdigest()
            p = _ensure_cache_dir() / f"{key}.json"
            if p.exists():
                p.unlink()

        wrapper.bust = bust
        return wrapper

    return decorator
