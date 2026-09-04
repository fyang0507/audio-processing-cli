"""Hash-pinned downloads and partial provisioning behavior."""

from __future__ import annotations

from package_test_support import (
    FakeFetcher,
    FakeToolchain,
    hashlib,
    io,
    os,
    package_fetcher,
    paths,
    pkg,
    pytest,
    types,
)
from package_test_support import (
    isolated_root as isolated_root,
)

from audio_cli.media import publication as media_publication


def test_a_url_package_needs_no_forced_download_because_it_is_hash_pinned() -> None:
    """The one place `--repair` does nothing, and the reason it does not have to.

    `url_file` re-hashes what is on disk against the manifest pin and downloads again unless it
    matches, so a match is already the strongest re-materialization available. This runs the real
    fetcher, not the fake, because the claim is about that code path: a matching file returns
    without touching the network at all.
    """
    import urllib.request

    target = paths.models_dir() / "already-correct.onnx"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"onnx-bytes")

    def explode(*args, **kwargs):  # noqa: ANN002, ANN003 - a tripwire, never called
        raise AssertionError("url_file went to the network for a file that matches its pin")

    original = urllib.request.urlopen
    urllib.request.urlopen = explode
    try:
        resolved = pkg.Fetcher().url_file(
            "https://example.invalid/x.onnx", hashlib.sha256(b"onnx-bytes").hexdigest(), target
        )
    finally:
        urllib.request.urlopen = original
    assert resolved == target

    # And it is a check, not a shortcut: a file that does not match is re-fetched.
    target.write_bytes(b"corrupt")
    with pytest.raises(AssertionError, match="went to the network"):
        urllib.request.urlopen = explode
        try:
            pkg.Fetcher().url_file(
                "https://example.invalid/x.onnx", hashlib.sha256(b"onnx-bytes").hexdigest(), target
            )
        finally:
            urllib.request.urlopen = original


def test_url_download_temporary_never_follows_a_precreated_symlink(
    tmp_path,
    monkeypatch,
) -> None:
    target = paths.models_dir() / "model.onnx"
    target.parent.mkdir(parents=True, exist_ok=True)
    victim = tmp_path / "victim.bin"
    victim.write_bytes(b"owned elsewhere")
    monkeypatch.setattr(package_fetcher.uuid, "uuid4", lambda: types.SimpleNamespace(hex="fixed"))
    temporary = target.with_name(f".audio-download-{os.getpid()}-fixed.part")
    temporary.symlink_to(victim)

    with pytest.raises(pkg.ProvisioningError) as raised:
        pkg.Fetcher().url_file(
            "https://example.invalid/model.onnx",
            hashlib.sha256(b"model").hexdigest(),
            target,
        )

    assert raised.value.code == "download_failed"
    assert victim.read_bytes() == b"owned elsewhere"
    assert temporary.is_symlink()
    assert not target.exists()


def test_hash_matching_url_target_symlink_is_replaced_not_accepted(
    tmp_path,
    monkeypatch,
) -> None:
    payload = b"model bytes"
    target = paths.models_dir() / "model.onnx"
    target.parent.mkdir(parents=True, exist_ok=True)
    victim = tmp_path / "victim.onnx"
    victim.write_bytes(payload)
    target.symlink_to(victim)

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.close()

    monkeypatch.setattr(
        package_fetcher.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: Response(payload),
    )

    resolved = pkg.Fetcher().url_file(
        "https://example.invalid/model.onnx",
        hashlib.sha256(payload).hexdigest(),
        target,
    )

    assert resolved == target
    assert target.is_file() and not target.is_symlink()
    assert target.read_bytes() == payload
    assert victim.read_bytes() == payload


def test_url_download_rolls_back_a_private_temporary_substitution_at_publication(
    tmp_path,
    monkeypatch,
) -> None:
    payload = b"new model bytes"
    target = paths.models_dir() / "model.onnx"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"prior managed cache")
    victim = tmp_path / "victim.bin"
    victim.write_bytes(b"owned elsewhere")
    monkeypatch.setattr(package_fetcher.uuid, "uuid4", lambda: types.SimpleNamespace(hex="fixed"))

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.close()

    monkeypatch.setattr(
        package_fetcher.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: Response(payload),
    )
    real_exchange = media_publication._rename_exchange
    substituted = False

    def substitute_at_exchange(directory_descriptor, left_name, right_name):
        nonlocal substituted
        if not substituted:
            substituted = True
            os.rename(
                left_name,
                "held-legitimate.part",
                src_dir_fd=directory_descriptor,
                dst_dir_fd=directory_descriptor,
            )
            os.symlink(
                victim,
                left_name,
                dir_fd=directory_descriptor,
            )
        real_exchange(directory_descriptor, left_name, right_name)

    monkeypatch.setattr(media_publication, "_rename_exchange", substitute_at_exchange)

    with pytest.raises(pkg.ProvisioningError) as raised:
        pkg.Fetcher().url_file(
            "https://example.invalid/model.onnx",
            hashlib.sha256(payload).hexdigest(),
            target,
        )

    temporary = target.with_name(f".audio-download-{os.getpid()}-fixed.part")
    assert raised.value.code == "download_failed"
    assert substituted is True
    assert target.read_bytes() == b"prior managed cache"
    assert victim.read_bytes() == b"owned elsewhere"
    assert temporary.is_symlink()
    assert (target.parent / "held-legitimate.part").read_bytes() == payload


def test_url_download_never_replaces_a_directory_leaf(
    monkeypatch,
) -> None:
    payload = b"model bytes"
    target = paths.models_dir() / "model.onnx"
    target.mkdir(parents=True)
    marker = target / "owned.txt"
    marker.write_bytes(b"preserve me")

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.close()

    monkeypatch.setattr(
        package_fetcher.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: Response(payload),
    )

    with pytest.raises(pkg.ProvisioningError) as raised:
        pkg.Fetcher().url_file(
            "https://example.invalid/model.onnx",
            hashlib.sha256(payload).hexdigest(),
            target,
        )

    assert raised.value.code == "download_failed"
    assert target.is_dir()
    assert marker.read_bytes() == b"preserve me"


def test_url_download_replaces_an_unreadable_cache_entry(
    tmp_path,
    monkeypatch,
) -> None:
    payload = b"model bytes"
    target = paths.models_dir() / "model.onnx"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"unreadable")
    digest_probes: list[tuple[object, ...]] = []

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.close()

    def unreadable_digest(*args):
        digest_probes.append(args)
        raise PermissionError("unreadable")

    monkeypatch.setattr(package_fetcher, "sha256_regular_file_at", unreadable_digest)
    monkeypatch.setattr(
        package_fetcher.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: Response(payload),
    )

    assert (
        pkg.Fetcher().url_file(
            "https://example.invalid/model.onnx",
            hashlib.sha256(payload).hexdigest(),
            target,
        )
        == target
    )
    assert len(digest_probes) == 1, "the facade hash probe override was not consulted"
    assert target.read_bytes() == payload


def test_url_download_cannot_follow_a_parent_swapped_after_open(
    tmp_path,
    monkeypatch,
) -> None:
    payload = b"model bytes"
    models = paths.models_dir()
    models.mkdir(parents=True, exist_ok=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    external = outside / "model.onnx"
    external.write_bytes(b"owned elsewhere")

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.close()

    def swap_parent():
        models.rename(paths.root() / "models-old")
        models.symlink_to(outside, target_is_directory=True)
        return types.SimpleNamespace(hex="fixed")

    monkeypatch.setattr(package_fetcher.uuid, "uuid4", swap_parent)
    monkeypatch.setattr(
        package_fetcher.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: Response(payload),
    )

    with pytest.raises(pkg.ProvisioningError) as raised:
        pkg.Fetcher().url_file(
            "https://example.invalid/model.onnx",
            hashlib.sha256(payload).hexdigest(),
            models / "model.onnx",
        )

    assert raised.value.code == "download_failed"
    assert external.read_bytes() == b"owned elsewhere"


def test_a_stack_pull_provisions_around_a_missing_toolchain(tmp_path) -> None:
    """Measured before this fix: `pull --stack qwen-1.7b` with no `swift` left `ready` empty.

    `select` sorts by package id, `fluidaudio` sorts first, and `pull` raised on it — so a machine
    without a Swift toolchain got none of the ASR weights it could have had. §0 of
    TRANSCRIBE_HAPPY_PATH.md promises the opposite.
    """
    provisioner = pkg.Provisioner(
        toolchain=FakeToolchain(missing=("swift",)), fetcher=FakeFetcher(tmp_path)
    )
    receipt = provisioner.pull(pkg.select(stack="qwen-1.7b"), stack="qwen-1.7b")

    pulled = [entry["package"] for entry in receipt["pulled"]]
    assert "qwen3-asr-1.7b-8bit" in pulled and "qwen3-forcedaligner" in pulled
    assert "fluidaudio" not in pulled

    blocked = next(item for item in receipt["warnings"] if item["code"] == "toolchain_missing")
    assert blocked["blocking"] is True
    assert blocked["packages"] == ["fluidaudio"]
    assert blocked["requires_tool"] == ["swift"]
    assert "qwen-1.7b" in blocked["detail"]

    document = pkg.load_registry()
    assert pkg.is_ready(document, "qwen3-asr-1.7b-8bit")
    assert "fluidaudio" not in document["packages"], "a blocked package left a registry entry"
    # A blocked package provisioned nothing, so it claims no license and no bytes.
    licenses = next(item for item in receipt["warnings"] if item["code"] == "license_unreviewed")
    assert "fluidaudio" not in licenses["packages"]
    assert receipt["pulled_known_bytes"] > 0


def test_naming_a_toolchain_blocked_package_is_still_exit_three(tmp_path) -> None:
    """The asymmetry: a stack is a superset guess, a named package is an instruction.

    Named *beside a package that did provision*, so this cannot pass by way of the
    nothing-was-provisionable rule below — the refusal has to come from the naming.
    """
    provisioner = pkg.Provisioner(
        toolchain=FakeToolchain(missing=("swift",)), fetcher=FakeFetcher(tmp_path)
    )
    with pytest.raises(pkg.ProvisioningError) as caught:
        provisioner.pull(pkg.select(["qwen3-asr-1.7b-8bit", "fluidaudio"]))
    assert caught.value.code == "toolchain_missing"
    assert caught.value.exit_code == 3
    assert caught.value.payload["package"] == "fluidaudio"
    assert caught.value.payload["requires_tool"] == ["swift"]
    # What it did provision before refusing stays provisioned, as after any interrupted pull.
    assert pkg.is_ready(pkg.load_registry(), "qwen3-asr-1.7b-8bit")

    # Order does not soften it either: blocked first is the case that used to abort a stack.
    with pytest.raises(pkg.ProvisioningError):
        provisioner.pull(pkg.select(["fluidaudio", "qwen3-asr-0.6b-8bit"]))
    assert "qwen3-asr-0.6b-8bit" not in pkg.load_registry()["packages"]

    # And a stack in which nothing at all was provisionable is exit 3 too, because there is no
    # partial success to report. No shipped stack is one package wide, so the selection is
    # constructed rather than selected.
    with pytest.raises(pkg.ProvisioningError) as caught:
        provisioner.pull(pkg.select(["fluidaudio"]), stack="qwen-1.7b")
    assert caught.value.exit_code == 3
