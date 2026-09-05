
## 6. Teardown

Everything provisioned is discoverable from the registry, so a session that never
ran `pull` can still find and remove it.

```bash
audio packages path
audio packages list
audio packages remove vibevoice-asr-7b     # removes its unique torch environment and checkout
audio packages purge --dry-run             # reports reclaimable bytes
audio packages purge
uv tool uninstall audio-processing-cli
```

`packages path` uses `location` for a single materialized source and a repository-to-path
`locations` map for a multi-repository Hub package; it never emits a null singular alias beside
that map. A native package also carries `checkout`, because the executable source is part of the
live provenance that `verify` checks. Fields for materializations a package does not have stay
absent.

`remove` and `purge` treat the registry as an ownership receipt, not path authority. Local
package and environment targets come from the installed manifest and stay inside the managed
root. A Hub revision is deletion-eligible only when the receipt says this root downloaded it,
the current manifest still pins it for that package, and it was not already cached before pull;
all other claimed revisions are retained. An unknown or retired registry package loses its
entry without following any recorded path. A failed local deletion leaves its registry owner
and contributes no reclaimed bytes. Neither command touches user media or output artifacts.

Purge before uninstalling, or the resolved root outlives the only tool that knows
how to describe it.

## Capability coverage

Where each capability in the namespace is exercised above:

| Capability | Exercised | Shown as |
| --- | --- | --- |
| `verbatim` | requested in §2 and §3; evidence divergence in §1.5 | native on all four; `quality: "refuted"` on `vibevoice` and `qwen-0.6b` |
| `diarization` | §1, §2, §3 | derived on Qwen and FireRed, native on VibeVoice; yields `segments[].speaker` and `turns[]` together |
| `overlapped_speech` | §1.4, §3 | derived |
| `vad` | §1.4, §3 | derived on Qwen via `silero-vad`, native on FireRed |
| `segment_timestamps` | §2, §3, §5 | native on VibeVoice and FireRed; exit 2 `unsatisfiable_on_stack` on Qwen |
| `word_timestamps` | §1.4, §2, §3 | derived on Qwen and VibeVoice, native on FireRed |
| `lid` | §3 | native on FireRed only, with its inference cost and region granularity |
| `token_lid` | `capabilities`, §5 | `impossible` in the catalog; exit 2 `unsupported` when requested |

All seven roles appear in a resolved plan: `decode` and `asr` in §1.1, §2 and §3;
`diarizer` in §1.1; `vad` in §1.4 (`silero-vad`) and §3
(`firered-vad`); `aligner` in §1.4 and §2; `punctuator` in §3; `lid` in §3's LID
variant. Four of the five `selected_by` forms appear in a resolved plan above —
`stack`, `requirement`, `add_on_required_by`, and `floor`; the fifth, `pin`, appears
only in §5 where a pin is rejected. Every error code in the table at the top of this
document is shown with its payload or its field list.
