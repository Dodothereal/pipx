from __future__ import annotations

import importlib
from email.message import Message
from http.client import HTTPResponse
from io import BytesIO
from typing import TYPE_CHECKING, cast
from urllib.response import addinfourl

import pytest

from pipx.util import PipxError

if TYPE_CHECKING:
    from collections.abc import Callable
    from socket import socket

    from pytest_mock import MockerFixture


@pytest.fixture(
    params=[("pipx.script", "_read_url"), ("pipx.commands.run", "_http_get_request")], ids=["install", "run"]
)
def download(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Callable[[str], str]:
    module_name, function_name = request.param
    module = importlib.import_module(module_name)
    monkeypatch.setattr(module, "_MAX_SCRIPT_BYTES", 7)
    return cast("Callable[[str], str]", getattr(module, function_name))


def _response(mocker: MockerFixture, headers: bytes, body: bytes) -> HTTPResponse:
    # Use the actual HTTP parser and bounded read, without opening a socket or running the downloaded script.
    stream = BytesIO(b"HTTP/1.1 200 OK\r\n" + headers + b"\r\n" + body)
    connection = mocker.Mock()
    connection.makefile.return_value = stream
    response = HTTPResponse(cast("socket", connection))
    response.begin()
    mocker.patch("urllib.request.urlopen", return_value=response)
    return response


@pytest.mark.parametrize(
    ("headers", "body", "expected"),
    [
        pytest.param(b"Content-Length: 7\r\n", b"content", "content", id="exact-limit"),
        pytest.param(b"Content-Length: 0\r\n", b"", "", id="empty"),
        pytest.param(b"", b"content", "content", id="unknown-length"),
        pytest.param(
            b"Transfer-Encoding: chunked\r\nContent-Length: 100\r\n",
            b"7\r\ncontent\r\n0\r\n\r\n",
            "content",
            id="chunked-ignores-content-length",
        ),
        pytest.param(
            b"Content-Length: 1\r\nContent-Type: text/plain; charset=iso-8859-1\r\n",
            b"\xe9",
            "\u00e9",
            id="declared-charset",
        ),
        pytest.param(b"Content-Length: 2\r\n", b"\xc3\xa9", "\u00e9", id="multibyte-length"),
    ],
)
def test_download_complete_script(
    mocker: MockerFixture, download: Callable[[str], str], headers: bytes, body: bytes, expected: str
) -> None:
    response = _response(mocker, headers, body)

    assert download("https://example.invalid/app.py") == expected
    assert response.isclosed()


@pytest.mark.parametrize("body", [pytest.param(b"content", id="partial"), pytest.param(b"", id="empty")])
def test_download_rejects_incomplete_script(
    mocker: MockerFixture, download: Callable[[str], str], body: bytes
) -> None:
    response = _response(mocker, b"Content-Length: 8\r\n", body)

    with pytest.raises(PipxError, match="download is incomplete"):
        download("https://example.invalid/app.py")
    assert response.isclosed()


@pytest.mark.parametrize(
    "headers",
    [pytest.param(b"Content-Length: 8\r\n", id="known-length"), pytest.param(b"", id="unknown-length")],
)
def test_download_preserves_size_limit(
    mocker: MockerFixture, download: Callable[[str], str], headers: bytes
) -> None:
    response = _response(mocker, headers, b"too long")

    with pytest.raises(PipxError, match="larger than"):
        download("https://example.invalid/app.py")
    assert response.isclosed()


def test_download_incomplete_error_masks_url_password(mocker: MockerFixture, download: Callable[[str], str]) -> None:
    _response(mocker, b"Content-Length: 8\r\n", b"content")

    with pytest.raises(PipxError, match="download is incomplete") as error:
        download("https://user:secret@example.invalid/app.py")
    assert "secret" not in str(error.value)
    assert "user:****@example.invalid" in str(error.value)


def test_download_accepts_non_http_response(mocker: MockerFixture, download: Callable[[str], str]) -> None:
    response = addinfourl(BytesIO(b"content"), Message(), "file:///app.py")
    mocker.patch("urllib.request.urlopen", return_value=response)

    assert download("file:///app.py") == "content"
    assert response.closed
