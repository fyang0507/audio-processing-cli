# Documentation

- [Vocabulary](VOCABULARY.md), [transcription contract](TRANSCRIBE_CONTRACT.md), and [happy path](TRANSCRIBE_HAPPY_PATH.md): authoritative names, payloads, and examples.
- [Environments](ENVIRONMENTS.md): declared model packages, runtimes, and provisioning.
- [Enhancement](enhancement.md), [broadband denoising](dsp/denoise/ALGORITHM.md), and [RNNoise](rnnoise-denoising.md): processing behavior and evidence limits.
- [CLI feedback](cli-feedback.md), [progress and logs](cli-progress-logging.md), [report comparison](report-comparison.md), and [timing evidence](timing-evidence.md): command feedback, saved evidence, and timeline validation.
- [Model decisions](model-tests/DECISION_REPORT.md), [findings](model-tests/FINDINGS.md), and [experiments](model-tests/EXPERIMENT_RESULTS.md): research documentation; claims about measured output still require runner or artifact evidence.
- [Benchmark guide](model-tests/benchmark/README.md): reproducible research workflows. Runners, manifests, tracked results, and ignored raw runs remain under `model_tests/`.
- [Test layout](testing.md): ownership and local artifact conventions. [Agent acceptance](agent-acceptance.md): manual checks with fresh operators, declared clip requirements and saved evidence. [Clip report template](clip-report-template.md): delivery checks, optional observations and operator/reporting outcomes without a shared overall verdict.
- [Open-issue stack acceptance, 6 September 2026](model-tests/2026-09-06-open-issues-acceptance.md): earlier scoped feature verdicts, original-source artifacts and the strict/default Qwen timing-delivery failure.
- [Alignment recovery and subtitle acceptance, 6 September 2026](model-tests/2026-09-06-alignment-recovery-acceptance.md): explicit clipping and word-preserving export results, retained failures, operator trace limits and human-judgment scope.
- [Audio CLI skill](../skills/audio-cli/SKILL.md): CLI user guidance developed and shipped with the source distribution, outside development-agent auto-discovery.
