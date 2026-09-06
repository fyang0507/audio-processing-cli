# Documentation

- [Vocabulary](VOCABULARY.md), [transcription contract](TRANSCRIBE_CONTRACT.md), and [happy path](TRANSCRIBE_HAPPY_PATH.md): authoritative names, payloads, and examples.
- [Environments](ENVIRONMENTS.md): declared model packages, runtimes, and provisioning.
- [Enhancement](enhancement.md), [broadband denoising](dsp/denoise/ALGORITHM.md), and [RNNoise](rnnoise-denoising.md): processing behavior and evidence limits.
- [CLI feedback](cli-feedback.md), [progress and logs](cli-progress-logging.md), [report comparison](report-comparison.md), and [timing evidence](timing-evidence.md): command feedback, saved evidence, and timeline validation.
- [Model decisions](model-tests/DECISION_REPORT.md), [findings](model-tests/FINDINGS.md), and [experiments](model-tests/EXPERIMENT_RESULTS.md): research documentation; claims about measured output still require runner or artifact evidence.
- [Benchmark guide](model-tests/benchmark/README.md): reproducible research workflows. Runners, manifests, tracked results, and ignored raw runs remain under `model_tests/`.
- [Test layout](testing.md): ownership and local artifact conventions. [Agent acceptance](agent-acceptance.md): the final development gate, fresh operator contexts, media matrix, evidence and pass/fail criteria.
- [Audio CLI skill](../skills/audio-cli/SKILL.md): CLI user guidance developed and shipped with the source distribution, outside development-agent auto-discovery.
