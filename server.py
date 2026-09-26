#!/usr/bin/env python3
"""Small production-friendly HTTP server and Whisper proxy for Callnote."""

from __future__ import annotations

import json
import os
import re
import sys
import uuid
from email.parser import BytesParser
from email.policy import default
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


HOST = "0.0.0.0"
PORT = int(os.environ.get("PORT", "5000"))
ROOT = Path(__file__).resolve().parent
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_WEBHOOK_BYTES = 512 * 1024
OPENAI_TRANSCRIPTION_URL = "https://api.openai.com/v1/audio/transcriptions"


def json_bytes(payload: object) -> bytes:
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def api_error(message: str, status: int = HTTPStatus.BAD_REQUEST) -> tuple[int, dict]:
    return int(status), {"error": {"message": message}}


def parse_multipart(content_type: str, body: bytes) -> tuple[bytes, str, str]:
    """Extract the first file field from a multipart/form-data request."""
    if not content_type.lower().startswith("multipart/form-data"):
        raise ValueError("A kérésnek multipart/form-data formátumúnak kell lennie.")

    raw_message = (
        f"Content-Type: {content_type}\r\n"
        "MIME-Version: 1.0\r\n"
        "\r\n"
    ).encode("utf-8") + body
    message = BytesParser(policy=default).parsebytes(raw_message)
    if not message.is_multipart():
        raise ValueError("A feltöltött kérés nem értelmezhető multipart adatként.")

    for part in message.iter_parts():
        disposition = part.get("Content-Disposition", "")
        field_name = part.get_param("name", header="content-disposition")
        if field_name not in {"file", "audio"} or "filename" not in disposition:
            continue

        file_bytes = part.get_payload(decode=True) or b""
        filename = Path(part.get_filename() or "hangjegyzet.webm").name
        file_content_type = part.get_content_type() or "application/octet-stream"
        if not file_bytes:
            raise ValueError("A feltöltött hangfájl üres.")
        return file_bytes, filename, file_content_type

    raise ValueError("A kérés nem tartalmazott file mezőt.")


def build_openai_multipart(
    file_bytes: bytes,
    filename: str,
    file_content_type: str,
) -> tuple[bytes, str]:
    boundary = f"----CallnoteBoundary{uuid.uuid4().hex}"
    chunks: list[bytes] = []

    def add_field(name: str, value: str) -> None:
        chunks.extend(
            [
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                value.encode("utf-8"),
                b"\r\n",
            ]
        )

    add_field("model", "whisper-1")
    add_field("language", "hu")
    chunks.extend(
        [
            f"--{boundary}\r\n".encode(),
            (
                f'Content-Disposition: form-data; name="file"; '
                f'filename="{filename}"\r\n'
            ).encode(),
            f"Content-Type: {file_content_type}\r\n\r\n".encode(),
            file_bytes,
            b"\r\n",
            f"--{boundary}--\r\n".encode(),
        ]
    )
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def transcribe_with_openai(
    api_key: str,
    file_bytes: bytes,
    filename: str,
    file_content_type: str,
) -> tuple[int, dict]:
    request_body, request_content_type = build_openai_multipart(
        file_bytes, filename, file_content_type
    )
    request = Request(
        OPENAI_TRANSCRIPTION_URL,
        data=request_body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key.strip()}",
            "Content-Type": request_content_type,
            "Accept": "application/json",
        },
    )

    try:
        with urlopen(request, timeout=90) as response:
            payload = json.loads(response.read().decode("utf-8"))
            text = str(payload.get("text", "")).strip()
            if not text:
                return api_error("Az OpenAI Whisper nem adott vissza leiratot.", HTTPStatus.BAD_GATEWAY)
            return HTTPStatus.OK, {"text": text}
    except HTTPError as error:
        raw_error = error.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(raw_error)
        except json.JSONDecodeError:
            payload = {"error": {"message": raw_error or str(error)}}
        return error.code, payload
    except URLError as error:
        return api_error(f"Az OpenAI Whisper nem érhető el: {error.reason}", HTTPStatus.BAD_GATEWAY)
    except TimeoutError:
        return api_error("Az OpenAI Whisper kérés időtúllépés miatt megszakadt.", HTTPStatus.GATEWAY_TIMEOUT)


class CallnoteHandler(SimpleHTTPRequestHandler):
    """Serve the single-page app and handle its same-origin transcription proxy."""

    server_version = "CallnoteHTTP/1.0"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def send_json(self, status: int, payload: object) -> None:
        response = json_bytes(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(response)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(response)

    def proxy_webhook(self) -> None:
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.send_json(*api_error("Érvénytelen Content-Length fejléc.", HTTPStatus.BAD_REQUEST))
            return

        if content_length <= 0:
            self.send_json(*api_error("Nem érkezett webhook adat.", HTTPStatus.BAD_REQUEST))
            return
        if content_length > MAX_WEBHOOK_BYTES:
            self.send_json(
                *api_error(
                    "A webhook csomag túl nagy.",
                    HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                )
            )
            return

        try:
            request_payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self.send_json(*api_error("A webhook kérés törzse érvénytelen JSON.", HTTPStatus.BAD_REQUEST))
            return

        webhook_url = str(request_payload.get("webhook_url", "")).strip()
        webhook_payload = request_payload.get("payload")
        parsed_url = urlparse(webhook_url)
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
            self.send_json(
                *api_error(
                    "A webhook URL-nek érvényes http vagy https címnek kell lennie.",
                    HTTPStatus.BAD_REQUEST,
                )
            )
            return
        if not isinstance(webhook_payload, dict):
            self.send_json(*api_error("A webhook payload objektum kell legyen.", HTTPStatus.BAD_REQUEST))
            return

        outbound_body = json_bytes(webhook_payload)
        request = Request(
            webhook_url,
            data=outbound_body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json, text/plain, */*",
                "User-Agent": "Callnote-Webhook/1.0",
            },
        )
        try:
            with urlopen(request, timeout=20) as response:
                upstream_status = int(response.status)
            self.send_json(
                HTTPStatus.OK,
                {"ok": True, "upstream_status": upstream_status},
            )
        except HTTPError as error:
            self.send_json(
                *api_error(
                    f"A webhook szolgáltató HTTP {error.code} választ adott.",
                    HTTPStatus.BAD_GATEWAY,
                )
            )
        except URLError as error:
            self.send_json(
                *api_error(
                    f"A webhook nem érhető el: {error.reason}",
                    HTTPStatus.BAD_GATEWAY,
                )
            )
        except TimeoutError:
            self.send_json(
                *api_error(
                    "A webhook kérés időtúllépés miatt megszakadt.",
                    HTTPStatus.GATEWAY_TIMEOUT,
                )
            )

    def do_POST(self) -> None:
        if self.path == "/api/webhook":
            self.proxy_webhook()
            return
        if self.path != "/api/transcribe":
            self.send_json(*api_error("Az API útvonal nem található.", HTTPStatus.NOT_FOUND))
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.send_json(*api_error("Érvénytelen Content-Length fejléc.", HTTPStatus.BAD_REQUEST))
            return

        if content_length <= 0:
            self.send_json(*api_error("Nem érkezett hangfájl.", HTTPStatus.BAD_REQUEST))
            return
        if content_length > MAX_UPLOAD_BYTES:
            self.send_json(
                *api_error(
                    "A hangfájl túl nagy. A maximális méret 25 MB.",
                    HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                )
            )
            return

        try:
            body = self.rfile.read(content_length)
            file_bytes, filename, file_content_type = parse_multipart(
                self.headers.get("Content-Type", ""), body
            )
        except (ValueError, UnicodeError) as error:
            self.send_json(*api_error(str(error), HTTPStatus.BAD_REQUEST))
            return

        # Prefer the server-side secret. The header fallback keeps the existing
        # localStorage-based settings compatible until OPENAI_API_KEY is added.
        api_key = (
            os.environ.get("OPENAI_API_KEY", "").strip()
            or self.headers.get("X-OpenAI-API-Key", "").strip()
        )
        if not api_key:
            self.send_json(
                *api_error(
                    "Nincs beállítva OPENAI_API_KEY a szerveren.",
                    HTTPStatus.SERVICE_UNAVAILABLE,
                )
            )
            return

        status, payload = transcribe_with_openai(
            api_key, file_bytes, filename, file_content_type
        )
        self.send_json(status, payload)

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Allow", "GET, POST, OPTIONS")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

    def log_message(self, format: str, *args) -> None:
        # Keep the normal server access log, but never log request headers or
        # request bodies where a client-provided API key could appear.
        sys.stderr.write(f"{self.address_string()} - {format % args}\n")


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), CallnoteHandler)
    print(f"Callnote server listening on http://{HOST}:{PORT}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()