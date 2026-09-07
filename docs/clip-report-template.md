# Clip acceptance report template

Copy the blank template below into a report file and fill it with actual evidence. The development agent may include the blank format in the fresh operator's brief; do not give the operator developer history or prior verdicts. Declare required checks before running the clip. This is a report format, not an evaluation harness or scoring system.

Save the report as native Markdown with a file-writing tool such as `apply_patch`. If JSON is requested, save native JSON data directly and parse it once to check validity. Do not paste JSON as Python code, embed Markdown in JavaScript template literals, or generate a script just to write the report. Reopen the saved report and check its statuses and links.

```markdown
# Clip report

Task and scope: <source clip/range, requested operation, configuration>.
Candidate and evidence: <candidate/CLI and skill identity, source hashes, command/raw-result paths>.

Required delivery: <PASS / FAIL / BLOCKED>.

| Check and scope | Required? | Status | Evidence |
| --- | --- | --- | --- |
| <requested deliverable or check> | Yes | <PASS / FAIL / BLOCKED> | <actual artifact, measurement or blocker> |

Optional observations:

| Diagnostic and scope | Observation | Evidence |
| --- | --- | --- |
| <optional target or diagnostic> | <measured value, miss or unavailable evidence> | <raw result and relevant limit> |

Operator, protocol and reporting:

| Activity | Status | Effect on delivery or evidence |
| --- | --- | --- |
| <execution / stop compliance / report writing> | <PASS / FAIL / BLOCKED> | <exact error or evidence; name an affected required check, or state no effect on validated outputs> |

Human judgment: <recorded decision, required but unavailable, or not required>.
Unattempted work: <dependent steps and why, or none>.
Next action: <required fix or user decision, or none>.
```

Put every required check, including required human judgment, in the required delivery table. Aggregate those rows: a failed row makes delivery FAIL; otherwise an unavailable required check or judgment makes it BLOCKED; otherwise all required rows must pass. The separate human-judgment note does not exempt it from this rule. Do not produce a passing verdict from an empty required-check list. Optional LRA/residual-noise misses do not enter that verdict unless attainment was required before the run. Keep those measurements visible without creating an automatic fine-tuning backlog.

For example, valid requested subtitles followed by a report-writing parse error are `delivery PASS; reporting FAIL`. If source identity cannot be established, the required source-preservation check is `BLOCKED`, even if a playable file exists. If required subtitle words are omitted, that export check is `FAIL`; successful ASR remains separately recorded. An unexpected command failure still stops further CLI work under the [acceptance procedure](agent-acceptance.md).
