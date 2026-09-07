# Documentation

- [Vocabulary](VOCABULARY.md), [transcription contract](TRANSCRIBE_CONTRACT.md), and [happy path](TRANSCRIBE_HAPPY_PATH.md): authoritative names, payloads, and examples.
- [Environments](ENVIRONMENTS.md): declared model packages, runtimes, and provisioning.
- [Silero VAD configuration migration](vad-model-migration.md): replace the removed inspection/enhancement `--vad-model` option with shared environment configuration.
- [Enhancement](enhancement.md), [broadband denoising](dsp/denoise/ALGORITHM.md), and [RNNoise](rnnoise-denoising.md): processing behavior and evidence limits.
- [Developing bundled profiles](profile-authoring.md): contributor guidance for extending presets; separate from the shipped agent-user skill.
- [CLI feedback](cli-feedback.md), [progress and logs](cli-progress-logging.md), [report comparison](report-comparison.md), and [timing evidence](timing-evidence.md): command feedback, saved evidence, and timeline validation.
- [Model decisions](model-tests/DECISION_REPORT.md), [findings](model-tests/FINDINGS.md), and [experiments](model-tests/EXPERIMENT_RESULTS.md): research documentation; claims about measured output still require runner or artifact evidence.
- [Benchmark guide](model-tests/benchmark/README.md): reproducible research workflows. Runners, manifests, tracked results, and ignored raw runs remain under `model_tests/`.
- [Test layout](testing.md): ownership and local artifact conventions. [Agent acceptance](agent-acceptance.md): manual checks with fresh operators, declared clip requirements and saved evidence. [Clip report template](clip-report-template.md): delivery checks, optional observations and operator/reporting outcomes without a shared overall verdict.
- [Open-issue stack acceptance, 6 September 2026](model-tests/2026-09-06-open-issues-acceptance.md): earlier scoped feature verdicts, original-source artifacts and the strict/default Qwen timing-delivery failure.
- [Alignment recovery and subtitle acceptance, 6 September 2026](model-tests/2026-09-06-alignment-recovery-acceptance.md): explicit clipping and word-preserving export results, retained failures, operator trace limits and human-judgment scope.
- [Custom-clip report interpretation acceptance, 7 September 2026](model-tests/2026-09-07-report-interpretation-acceptance.md): scoped acceptance reporting, independent measurement-interval findings and fresh-agent saved-report checks on the demo video and mixed-language clip.
- [JSON run and offline export acceptance, 7 September 2026](model-tests/2026-09-07-json-run-acceptance.md): one full-source recognition, reusable canonical JSON, readable exports, expected refusals and a separately disclosed setup-protocol deviation.
- [Full post-change acceptance, 7 September 2026](model-tests/2026-09-07-post-change-acceptance.md): the complete original-source matrix, feature-specific refusals, unchanged baseline artifacts and the retained strict/default Qwen timing failure.
- [Audio CLI skill](../skills/audio-cli/SKILL.md): CLI user guidance developed and shipped with the source distribution, outside development-agent auto-discovery.
