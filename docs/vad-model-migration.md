# Silero VAD model configuration

`audio inspect` and `audio enhance` use the same pinned Silero VAD model. Their former `--vad-model` option has been removed: it selected a file location, never an alternative model. Existing invocations that include it now fail with an argument error (exit 2); remove the flag and configure the location through the environment.

For a prepopulated copy outside the managed cache, replace `--vad-model /path/to/silero_vad.onnx` with a command-local environment assignment:

```sh
AUDIO_PROCESSING_VAD_MODEL=/path/to/silero_vad.onnx audio inspect recording.wav
AUDIO_PROCESSING_VAD_MODEL=/path/to/silero_vad.onnx audio enhance recording.wav --profile product-demo --output enhanced.wav
```

The override must contain the exact pinned Silero VAD 6.2.1 ONNX artifact. The [resolver](../src/audio_cli/vad.py) checks its SHA-256 before loading it and refuses a missing or mismatched override. It does not replace a bad override by downloading another file.

To provision the managed copy in advance, use:

```sh
audio packages pull silero-vad
audio inspect recording.wav
audio enhance recording.wav --profile product-demo --output enhanced.wav
```

`AUDIO_PROCESSING_MODEL_CACHE` selects the shared provisioning root. Set it consistently for provisioning and later commands; the VAD model lives under that root's `models/` directory. `AUDIO_PROCESSING_VAD_MODEL` takes precedence when set. Without an override or a valid managed copy, first VAD use retains the existing pinned download and SHA-256 verification. No source media is modified by model resolution. See [environments](ENVIRONMENTS.md) for provisioning details.

Inspection and enhancement still use VAD. Transcription's separate `--vad` selector and saved-report commands are unchanged.
