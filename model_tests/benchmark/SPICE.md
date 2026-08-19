# Preparing the SpiCE interview fixtures

[SpiCE](https://doi.org/10.5683/SP2/MJOXP3) is CC BY 4.0. The frozen
VF19A Cantonese session comes from Scholars Portal Dataverse file IDs 165798
(WAV) and 165805 (TextGrid). Download into the ignored data directory:

```bash
mkdir -p model_tests/benchmark_data/spice

curl -fL \
  https://dataverse.scholarsportal.info/api/access/datafile/165798 \
  -o model_tests/benchmark_data/spice/VF19A_Cantonese_I2_20181114.wav

curl -fL \
  https://dataverse.scholarsportal.info/api/access/datafile/165805 \
  -o model_tests/benchmark_data/spice/VF19A_Cantonese_I2_20181114.TextGrid
```

The frozen source SHA-256 values are:

```text
56773f36702a74a114e9086e2ce9a0314151c67a2b453ad128bcefcc8b702236  VF19A_Cantonese_I2_20181114.wav
1e878f633a64f97832bc88f82ec477b7096ca0ab3eb77af6fbee73ca510fbf0f  VF19A_Cantonese_I2_20181114.TextGrid
```

The source is 44.1 kHz stereo. The interview starts at source time
271.130649013 seconds. FL is the participant microphone; FR is the interviewer
microphone, and both contain audible cross-talk. Prepare the participant-mic
30-minute stress/quality fixture:

```bash
ffmpeg -nostdin -hide_banner -loglevel error -y \
  -ss 271.130649013 -t 1800 \
  -i model_tests/benchmark_data/spice/VF19A_Cantonese_I2_20181114.wav \
  -af 'pan=mono|c0=FL' -ar 24000 -c:a pcm_s16le \
  model_tests/benchmark_data/spice/VF19A_Cantonese_interview30m_participant_24k.wav

python3 model_tests/benchmark/prepare_spice.py \
  --textgrid model_tests/benchmark_data/spice/VF19A_Cantonese_I2_20181114.TextGrid \
  --audio model_tests/benchmark_data/spice/VF19A_Cantonese_interview30m_participant_24k.wav \
  --output model_tests/benchmark/manifests/spice_vf19a_cantonese_interview30m.json \
  --start 271.130649013 --duration 1800 \
  --source-url https://doi.org/10.5683/SP2/MJOXP3
```

The prepared audio SHA-256 is
`0769ce2478709153d6e7ba5c10078f49edc58a76351189fcf92d1d8e7b6f233c`.
The manifest freezes 153 hand-corrected participant utterances. The
interviewer's audible speech is intentionally absent from the corpus
transcript, so it is valid for participant ASR/utterance-interval scoring only—not
all-speaker MER or full DER.

For diarization, generate the canonical two-channel downmix with an explicit
seek and conversion. It contains both dedicated microphone channels **plus
their bleed**; it is not isolated-speaker audio:

```bash
ffmpeg -nostdin -hide_banner -loglevel error -y \
  -ss 271.130649013 -t 1800 \
  -i model_tests/benchmark_data/spice/VF19A_Cantonese_I2_20181114.wav \
  -ac 1 -ar 16000 -c:a pcm_s16le \
  model_tests/benchmark_data/spice/VF19A_Cantonese_interview30m_canonical_mix_16k.wav
```

With FFmpeg 8.1, the output is 1,800 seconds of 16 kHz mono PCM16,
57,600,078 bytes, SHA-256
`79c28259614253cb589c968c4d27693eea98ceaf7c2db372a4cdbeb8a77eb511`.
The full provenance and partial-reference restriction are frozen in
`manifests/spice_vf19a_cantonese_interview30m_canonical_mix.json`. An earlier
local mix with unrecoverable command provenance is intentionally not canonical.

For a one-hour resource/stability stress, repeat the complete participant
fixture twice. It is duplicated evidence and must never be quality-scored:

```bash
ffmpeg -nostdin -hide_banner -loglevel error -y \
  -stream_loop 1 \
  -i model_tests/benchmark_data/spice/VF19A_Cantonese_interview30m_participant_24k.wav \
  -t 3600 -ac 1 -ar 24000 -c:a pcm_s16le \
  model_tests/benchmark_data/spice/VF19A_Cantonese_interview60m_participant_concat_24k.wav
```

The expected one-hour fixture SHA-256 is
`d0049e2bd6080dbd2125ff2981acc8d8e25c09e63a5a6413d0e2d088e43c549e`.
Its construction and epistemic restriction are also frozen in
`manifests/spice_vf19a_cantonese_interview60m_concat.json`.

The corresponding diarization-only hour repeats the canonical mixed interview
without re-encoding:

```bash
ffmpeg -nostdin -hide_banner -loglevel error -y \
  -stream_loop 1 \
  -i model_tests/benchmark_data/spice/VF19A_Cantonese_interview30m_canonical_mix_16k.wav \
  -t 3600 -map 0:a:0 -c:a copy \
  model_tests/benchmark_data/spice/VF19A_Cantonese_interview60m_canonical_mix_concat_16k.wav
```

Its SHA-256 is
`dfc02923f19b0da6140f8289399be70047151f491a23d2cd9759e84877a022a8`.
Decoded PCM for both halves and the 30-minute input has the same SHA-256,
`465cae97476254ae563420fc11e503b4cc2a7e09b687b5a7b12b158590384c9d`.
See `manifests/spice_vf19a_cantonese_interview60m_canonical_mix_concat.json`;
this is repeat/resource evidence, never an additional quality sample.
