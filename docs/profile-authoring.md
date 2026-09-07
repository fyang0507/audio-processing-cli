# Developing bundled enhancement profiles

This document is for contributors extending the CLI. Agents using the CLI should use the shipped [audio skill](../skills/audio-cli/SKILL.md) to choose and apply existing profiles; authoring presets is not part of that workflow.

Complete preset settings live in [transcription.toml](../src/audio_cli/profile_configs/transcription.toml) and [product-demo.toml](../src/audio_cli/profile_configs/product-demo.toml). TOML supports comments and uses Python’s standard-library `tomllib` parser without an additional dependency. The immutable `Profile` type and strict resource loader remain in [profiles.py](../src/audio_cli/profiles.py); algorithms and processing order remain in Python. Bundled TOML files are discovered automatically and their names become the `inspect` and `enhance` profile choices.

To add a bundled preset, copy a complete TOML definition into `src/audio_cli/profile_configs/<name>.toml`, use the same lowercase hyphenated `name` inside it, and set its positive integer string `version`. Specify every `Profile` field, including speech-region settings; the loader supplies no inherited or implicit defaults. Keep `processing_order` out of the file: reports add that fixed implementation contract. User-supplied profile paths are not supported.

The loader rejects duplicate, unknown or missing fields, incorrect types, nonfinite numbers, invalid bounds and inconsistent target ranges. Validate the preset with the [profile tests](../tests/audio_cli/test_profiles.py) and applicable [agent acceptance](agent-acceptance.md) before shipping. Keep these development procedures out of the shipped skill and its references.
