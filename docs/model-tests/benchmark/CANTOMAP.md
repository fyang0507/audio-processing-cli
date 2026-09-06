# CantoMap runnable slice

This fixture prepares one 149.9-second Hong Kong Cantonese MapTask exchange
without tracking corpus media or derived transcripts. It is a first runnable
dialect slice, not yet the full multi-conversation evaluation set described in
`DATASETS.md`.

## Fetch and prepare

Clone metadata without downloading all of the Git LFS audio, then fetch only
the frozen source recording:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 \
  https://github.com/gwinterstein/CantoMap.git \
  model_tests/benchmark_data/cantomap

git -C model_tests/benchmark_data/cantomap lfs pull \
  --include='ConversationData/Subjects-37_38/160818_009F37_38_D.wav' \
  --exclude=''

python model_tests/benchmark/prepare_cantomap.py
```

The command verifies the frozen repository revision and source hashes, checks
that the three ELAN tiers agree by timestamp, and writes these ignored files:

- `model_tests/benchmark_data/prepared/cantomap_yue_hk_37_38_d_030500_180400/audio.wav`
- `model_tests/benchmark_data/prepared/cantomap_yue_hk_37_38_d_030500_180400/reference.json`

Use `--reference-only` to validate and inspect the ELAN output without running
ffmpeg. Re-running does not replace an existing WAV unless `--force` is set.

## Why this span

The frozen window is `00:30.500–03:00.400` of Subjects 37/38 session D. Its
boundaries fall between aligned annotations, so no reference utterance is cut.
Within the window are 83 F/G activity segments, 75 chronological speaker
transitions, and 10 cross-speaker overlap pairs. The exchange contains dense
route negotiation, short acknowledgements/backchannels, repairs, unknown-word
notation, and corpus annotation tokens. F and G remain opaque corpus speaker
IDs; the fixture does not infer participant identity or conversational role.

## Reference contract

`reference.json` follows `cantomap_reference.schema.json` and contains:

- clip-relative and source-absolute millisecond intervals;
- speaker IDs and all three aligned source texts: unsegmented characters,
  manually segmented characters, and Jyutping;
- a conservative `characters_cer` view plus chronological and per-speaker
  concatenations;
- exact cross-speaker overlap regions and structural counts.

The CER view removes pause markers, `xxx`, and ampersand annotation tokens, but
the raw source is retained. This prevents corpus markup from dominating a first
orthographic CER while keeping evidence available for a separately defined,
human-reviewed particle/filler metric. The ELAN annotations provide useful
speaker-activity and approximate boundary labels; they are utterance-level
alignment units, not independently adjudicated conversational turns. Do not
report a definitive conversational-turn score, dialect-token recall, or semantic
preservation score until those labels are frozen by a Cantonese reviewer.

Corpus source and license: <https://github.com/gwinterstein/CantoMap>, GPL-3.0.
The required scholarly citation is recorded in the slice manifest and the
corpus README.
