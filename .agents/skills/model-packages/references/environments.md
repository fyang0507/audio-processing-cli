# Why a request spans the environments it does

Background for explaining the layout to a user. Not needed to run any command.

## Four provisioned environments, and the reason is one dependency

| Environment | Packages | Interpreter |
| --- | --- | --- |
| `core` | `silero-vad` | the tool's own; nothing to provision |
| `mlx` | both Qwen ASR checkpoints, `qwen3-forcedaligner` | 3.13.9, no PyTorch |
| `torch-firered` | `firered-asr2s` | 3.12.12 |
| `torch-vibevoice` | `vibevoice-asr-7b` | 3.12.12 |
| `swift` | `fluidaudio`, `speaker-diarization-coreml` | none — a build product |

They are separate because four packages pin three mutually exclusive `transformers` ranges:
`>=5.5.0,<5.13.0` for the MLX runtime, `==5.1.0` for FireRed, `==4.57.6` for the aligner's
PyTorch path, and `<5.0.0` for VibeVoice. Only the last two intersect. This was resolved rather
than decided: every possible grouping was put to the resolver, ten of fifteen conflict, and the
smallest workable partition is unique. So "could these two share one environment" has a
measured answer, and it is no.

The aligner then left PyTorch entirely, because the MLX runtime implements it and reproduces
the recorded alignment token for token. That is why word timing on a Qwen stack costs weights
but not a second runtime.

## What common requests span

| Request | Environments | Known download |
| --- | --- | --- |
| `qwen-1.7b`, nothing optional | `mlx` | 2.30 GiB |
| `qwen-1.7b` + diarization + word timestamps | `mlx`, `swift` | 3.61 GiB + one unsized build |
| `vibevoice` + word timestamps | `mlx`, `torch-vibevoice` | 17.35 GiB |
| `firered` whole pipeline | `torch-firered` | 8.93 GiB |

No request spans more than two. Stages run strictly sequentially with one model resident at a
time, each in its own process, so a request costs the **sum** of the stage times and the
**maximum** of the per-stage memory peaks — never the sum of both.

## Two things a user may ask about

**Why two PyTorch environments?** Because FireRed and VibeVoice cannot share one. The benefit is
that a FireRed user never installs VibeVoice's dependency set and vice versa; the cost is a
second PyTorch install on a machine that uses both stacks.

**Is `torch-vibevoice` permanent?** It is marked provisional. The MLX runtime can also run
VibeVoice, at roughly half the memory, but it produces a different transcript — so adopting it
would invalidate every recorded VibeVoice measurement, and that decision is open. Nothing about
the commands changes either way.

`ENVIRONMENTS.md` at the repository root has the full derivation, the registry schema, and the
recorded evidence.
