"""model_discovery.py — Dynamic model list fetching for VCF GUI.

Queries the OpenAI-compatible ``GET /v1/models`` endpoint of the configured
proxy (e.g. OmniRoute) and returns a plain list of model ID strings.

Design constraints
------------------
- Uses only the Python standard library (``urllib.request``) — no extra deps.
- Hard 3-second timeout on the network call.
- Every exception path returns an empty list; this function *never* raises.
- API keys and secrets are **never** logged or printed.
- URL normalization: if ``api_base`` already ends with ``/v1``, request
  ``{api_base}/models``; otherwise request ``{api_base}/v1/models``.
  This prevents the double-path ``/v1/v1/models`` bug.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any

log = logging.getLogger(__name__)

_FETCH_TIMEOUT_SECONDS: float = 3.0


def _build_models_url(api_base: str) -> str:
    """Return the correct ``/v1/models`` URL for a given ``api_base``.

    Normalizes the base URL so the path never becomes ``/v1/v1/models``:

    * If ``api_base`` already ends with ``/v1``  → ``{api_base}/models``
    * Otherwise                                  → ``{api_base}/v1/models``

    Leading/trailing whitespace is stripped; a trailing slash is removed
    before the suffix is appended.

    Examples
    --------
    >>> _build_models_url("http://localhost:20128/v1")
    'http://localhost:20128/v1/models'
    >>> _build_models_url("http://localhost:20128")
    'http://localhost:20128/v1/models'
    >>> _build_models_url("http://localhost:20128/")
    'http://localhost:20128/v1/models'
    >>> _build_models_url("https://api.openai.com/v1/")
    'https://api.openai.com/v1/models'
    """
    base = api_base.strip().rstrip("/")
    if base.endswith("/v1"):
        return f"{base}/models"
    return f"{base}/v1/models"


def fetch_available_models(
    api_base: str | None = None,
    api_key: str | None = None,
) -> list[str]:
    """Fetch the list of available model IDs from the configured proxy.

    Parameters
    ----------
    api_base:
        Base URL of the OpenAI-compatible proxy endpoint, e.g.
        ``http://localhost:20128/v1``.  If ``None`` or empty, returns ``[]``
        immediately.
    api_key:
        Bearer token to send in the ``Authorization`` header.  May be
        ``None`` for proxies that do not require authentication.
        **Never logged or printed.**

    Returns
    -------
    list[str]
        Model ID strings exactly as returned by the proxy — no renaming,
        no filtering.  Returns ``[]`` on any error without raising.
    """
    if not api_base or not api_base.strip():
        return []

    url = _build_models_url(api_base)

    headers: dict[str, str] = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        req = urllib.request.Request(url, headers=headers, method="GET")
        with urllib.request.urlopen(req, timeout=_FETCH_TIMEOUT_SECONDS) as resp:
            if resp.status != 200:
                log.debug(
                    "fetch_available_models: non-200 response %d from %s",
                    resp.status,
                    url,
                )
                return []
            raw: bytes = resp.read()
    except urllib.error.HTTPError as exc:
        log.debug("fetch_available_models: HTTP error %d from %s", exc.code, url)
        return []
    except urllib.error.URLError as exc:
        log.debug("fetch_available_models: connection error for %s — %s", url, exc.reason)
        return []
    except TimeoutError:
        log.debug("fetch_available_models: timeout reaching %s", url)
        return []
    except OSError as exc:
        log.debug("fetch_available_models: OS-level error for %s — %s", url, exc)
        return []

    try:
        payload: Any = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        log.debug("fetch_available_models: malformed JSON from %s — %s", url, exc)
        return []

    if not isinstance(payload, dict):
        log.debug(
            "fetch_available_models: unexpected top-level type %s from %s",
            type(payload).__name__,
            url,
        )
        return []

    data = payload.get("data")
    if not isinstance(data, list):
        log.debug(
            "fetch_available_models: 'data' key missing or not a list in response from %s",
            url,
        )
        return []

    model_ids: list[str] = []
    for item in data:
        if isinstance(item, dict):
            model_id = item.get("id")
            if isinstance(model_id, str) and model_id.strip():
                model_ids.append(model_id)

    return model_ids
