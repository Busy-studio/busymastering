# v8.5.3.64.0.4 Commercial Maximum Density Push Foundation

## Purpose

This hotfix changes Commercial / Maximum ownership from "stop once the commercial floor is safe" to "render and evaluate an actual loud WAV master candidate near the requested true-peak ceiling".

The previous v64.0.2 run could pass the commercial floor while leaving the delivered WAV around `-1.5 dBTP`. That is now reported as under-pushed for Commercial / Maximum modes.

## Main behavior

- Commercial / Maximum modes now target a WAV true-peak ceiling near `-0.1 dBTP`.
- Final true peak at or below `-1.0 dBTP` is marked `commercial_maximum_under_pushed=true` through `under_pushed` telemetry.
- The engine renders actual audio candidates instead of only reporting a shadow recommendation:
  - `main_clean_candidate`
  - `commercial_density_candidate`
  - `clipper_limiter_candidate`
  - `codec_safe_loud_candidate`
  - `maximum_push_candidate`
- The density chain distributes work across:
  - subtle density saturation
  - lightweight parallel density compression
  - soft clipper before limiter
  - limiter input / makeup gain
  - oversampled true-peak limiting and final TP pinning
- Codec stress remains measured and reported, but does not automatically block the WAV maximum candidate in Commercial / Maximum modes.
- A separate `codec_safe_loud_candidate` is rendered for distribution-risk visibility.

## Warning policy

Warnings are not hidden and thresholds are not relaxed.

Mode policy is explicit:

- `Safe`: warnings may block.
- `Balanced`: warnings can block when damage exceeds budget.
- `Commercial`: warnings are advisory except severe clipping / distortion / collapse.
- `Maximum`: warnings are advisory except severe clipping / distortion / collapse; the `-0.1 dBTP` WAV target is honored.

## Decision semantics

v64.0.3 selection semantics remain preserved:

- `accepted` / `rejected` = delivery selection state.
- `quality_gate` / `quality_rejected` = codec, translation, or damage quality state.

v64.0.4 adds decision replay fields for:

- selected density candidate
- mode policy
- true-peak target honored state
- commercial density priority
- warnings-as-advisory state
- rejected candidates
- selected reason

## Telemetry

Top-level report keys added or mirrored:

- `v6404_commercial_maximum_density`
- `commercial_maximum_density`
- `commercial_density_candidate_lineage`
- `clipper_limiter_candidate_lineage`
- `maximum_push_policy`
- `warning_advisory_policy`
- `codec_safe_vs_maximum_wav_decision`

Debug brief adds:

```text
## 11C. v64.0.4 Commercial Maximum Density Push
```

RCF marks these as implemented:

- `commercial_maximum_density`
- `clipper_before_limiter`
- `commercial_density_candidate`
- `warning_advisory_policy`
- `maximum_true_peak_ownership`

## Non-goals / forbidden behavior

This patch does not:

- hide warnings
- loosen thresholds to fake a pass
- rewrite `commercial_floor_met` as `target_reached`
- mark codec proxy as actual
- add Blossom-specific exceptions
- replace the density chain with simple limiter-only gain
- change v63.6-v63.9 DSP strengths
- break DML target semantics

## Validation performed

- `py_compile`
- `compileall`
- import smoke for:
  - `master_job`
  - `audio_engine.mastering`
  - `audio_engine.busy_auto_mixing`
  - `audio_engine.research_capability_framework`
  - `audio_engine.finishing_intelligence`
  - `audio_engine.commercial_density`
- synthetic density module smoke
- synthetic routed `master_audio` smoke in Maximum mode
- `supabase-files (38)(2).zip` report/debug replay smoke
- RCF attach smoke

Cloud Run production verification is still required after deployment.
