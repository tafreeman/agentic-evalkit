"""A minimal stub HTTP agent server for the http_agent example.

This is a demonstration fixture, not a real agent -- it exists so
docs/guides/http-agent-example.md and this directory's ``eval.yaml`` can be
run end to end with no external dependencies or network access. It speaks
exactly the request/response envelope ``HttpTarget`` sends
(``agentic_evalkit.targets.http``): a POST body containing
``schema_version``/``sample_id``/``input``/``attempt``/``trace_id``, and a
JSON response containing a matching ``sample_id`` and an ``output`` object.

It uses only the Python standard library (``http.server``) so the example
has zero additional dependencies beyond ``agentic-evalkit`` itself. It is
not part of this repository's public API and is not covered by the
dependency-boundary or public-docs contract tests -- it is example content,
not framework code.

Usage:
    python stub_agent_server.py [--port 8765]

Then, in another terminal, from this same directory:
    agentic-evalkit run eval.yaml --limit 3 --yes
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

_DEFAULT_PORT = 8765

# A tiny fixed lookup table so the stub agent gives a deterministic,
# sometimes-correct answer -- enough to show mixed pass/fail grading
# without needing a real model backend.
_KNOWN_ANSWERS = {
    "what is the capital of france?": "Paris",
    "what is 2 + 2?": "4",
    "what color is the sky on a clear day?": "blue",
}


def _answer_for(question: str) -> str:
    return _KNOWN_ANSWERS.get(question.strip().lower(), "I don't know")


class _StubAgentHandler(BaseHTTPRequestHandler):
    # Silence the default per-request stderr logging; this is a
    # demonstration fixture, not a production server.
    def log_message(self, format: str, *args: Any) -> None:
        del format, args

    def do_POST(self) -> None:
        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length)
        try:
            request = json.loads(raw_body)
        except json.JSONDecodeError:
            self._respond(400, {"error": "invalid JSON request body"})
            return

        sample_id = request.get("sample_id")
        question = request.get("input", {}).get("question", "")
        response_body = {
            "sample_id": sample_id,
            "output": {
                "answer": _answer_for(question),
                "tool_calls": ["lookup"],
            },
        }
        self._respond(200, response_body)

    def _respond(self, status: int, body: dict[str, Any]) -> None:
        payload = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=_DEFAULT_PORT)
    args = parser.parse_args()

    server = HTTPServer(("127.0.0.1", args.port), _StubAgentHandler)
    print(f"stub agent server listening on http://127.0.0.1:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
