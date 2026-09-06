"""Synthetic Git blob fixtures; never fetch or read real RNNoise weights."""

from __future__ import annotations

import io
from dataclasses import replace

import pytest
from package_test_support import FakeToolchain

from audio_cli import packages as pkg
from audio_cli.packages import catalog, fetcher

PAYLOAD = b"hello\n"
# Independent, canonical Git hash-object test vector, including blob header and six bytes.
BLOB_SHA1 = "ce013625030ba8dba906f756967f9e9ca394464a"


class Response(io.BytesIO):
    def geturl(self):
        return "https://raw.githubusercontent.com/synthetic/response"


@pytest.fixture
def rnnoise(monkeypatch):
    declared = pkg.packages()["rnnoise-voice"]
    package = replace(
        declared,
        bytes=len(PAYLOAD),
        source={**declared.source, "git_blob_sha1": BLOB_SHA1},
    )
    packages = {**pkg.packages(), package.id: package}
    monkeypatch.setattr(catalog, "packages", lambda: packages)
    monkeypatch.setattr(fetcher.Fetcher, "cached_revisions", lambda self: set())
    requested = []

    def respond(request, **_kwargs):
        requested.append(request.full_url)
        return Response(PAYLOAD)

    monkeypatch.setattr(fetcher.urllib.request, "urlopen", respond)
    provisioner = pkg.Provisioner(toolchain=FakeToolchain())
    return package, provisioner, requested
