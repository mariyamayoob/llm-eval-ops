from __future__ import annotations

import json
from dataclasses import dataclass
from urllib import error, request


@dataclass(frozen=True)
class ApiClient:
    api_base: str

    def get(self, path: str):
        try:
            with request.urlopen(f"{self.api_base}{path}") as response:
                return json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            raise RuntimeError(format_http_error(exc)) from exc

    def post(self, path: str, payload: dict):
        req = request.Request(
            f"{self.api_base}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req) as response:
                return json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            raise RuntimeError(format_http_error(exc)) from exc


def format_http_error(exc: error.HTTPError) -> str:
    raw_body = exc.read().decode("utf-8", errors="replace")
    if raw_body:
        try:
            payload = json.loads(raw_body)
        except json.JSONDecodeError:
            payload = raw_body
        detail = payload.get("detail", payload) if isinstance(payload, dict) else payload
        return f"HTTP {exc.code}: {detail}"
    return f"HTTP {exc.code}: {exc.reason}"

