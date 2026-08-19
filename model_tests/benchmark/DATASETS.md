# Benchmark datasets

## Chinese dialect conversation

The dialect benchmark must use conversation rather than read speech and must
report each variety separately. Do not collapse Cantonese, Wu, Xiang, Gan,
Sichuanese/Southwestern Mandarin, and regional Mandarin into one score.

### Primary, ungated material

- **CantoMap** — 12 h 48 min of contemporary Hong Kong Cantonese MapTask
  conversations, with audio and ELAN transcriptions in Chinese characters and
  Jyutping. GPL-3.0. Source: <https://github.com/gwinterstein/CantoMap>.
- **SpiCE** — 34 early Cantonese-English bilinguals, each recorded in
  Cantonese and English: 68 stereo sessions containing sentence, storyboard,
  and interview tasks, with four-tier TextGrids. This benchmark freezes one
  30-minute Cantonese interview excerpt. CC BY 4.0. DOI:
  <https://doi.org/10.5683/SP2/MJOXP3>. Dedicated participant/interviewer
  microphones occupy separate stereo channels but contain audible bleed; only
  participant speech is transcribed, so a
  small manually labeled subset is still required for diarization scoring.

CantoMap is the primary Cantonese conversation test because FireRedASR2S does
not list it in its published dialect table. SpiCE adds interview shape and
natural bilingual behavior. Code-switch metrics must be computed only on
reference spans that actually switch languages; bilingual speakers and two
language sessions do not by themselves imply intra-utterance code-switching.

### Breadth/calibration material (login gated)

MagicHub provides small free conversation sets spanning non-Mandarin Sinitic
varieties—Shanghai Wu (4.19 h), Guangzhou Yue (4.25 h), Changsha Xiang (4.1 h),
and Nanchang Gan (4 h)—plus Mandarin varieties Sichuan/Southwestern Mandarin
(4.53 h, 12 speaker-pair conversations) and Zhengzhou regional Mandarin (4 h).
The Sichuan set is CC BY-NC-ND 4.0 and described at
<https://magichub.com/cn/datasets/sichuan-dialect-conversational-speech-corpus/>.

These sets are appropriate for cross-variety calibration, but FireRed reports
results on several of the same MagicHub sets. Treat them as benchmark-familiar
for FireRed, not a blind head-to-head. A final product decision requires a
small, consented, post-release held-out set of actual target conversations.

### Minimum frozen evaluation slice

Select at least two 2–3 minute conversation spans per variety, from different
speakers/conversations, with one relatively clean span and one span containing
turn-taking, overlap/noise, discourse particles, or code-switching. Freeze:

- verbatim reference text preserving particles, fillers, repeats, and repairs;
- a frozen orthography-equivalence map kept separate from the verbatim
  reference, plus separate human semantic-preservation/hallucination labels;
- speaker-labeled intervals and overlap regions;
- marked dialect-bearing lexical spans, discourse particles/fillers, and
  code-switch spans;
- corpus, recording, time range, speaker IDs, license, and SHA-256 provenance.

Report verbatim error rate, orthography-normalized error rate, dialect-span recall,
particle/filler precision and recall, code-switch retention/translation count,
hallucinated-clause count, speaker-attributed error, ELAN-agreement DER where
the label source supports it, and annotation-order speaker-change F1.
Human review remains necessary for whether a transcript preserves
conversational meaning; deterministic edit distance is not a semantic metric
and can penalize valid dialect orthographies or hide normalization into
Standard Mandarin. Verify each corpus license separately before inclusion; the
license above is currently frozen only for the Sichuan MagicHub set.
