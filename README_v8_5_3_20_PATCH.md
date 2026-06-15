# Busy Auto Mastering Private Worker v8.5.3.20 Patch

## Purpose
Fix the remaining hot-commercial path problem seen after v8.5.3.19:

- v19 confirmed that virtual strategy gain was applied before limiting.
- However, high-gain candidates hit the limiter with several dB of peak overshoot.
- The transparent limiter then attenuated the entire candidate, so LUFS fell back around -13 LUFS and the commercial candidate did not replace the standard final master.

## Changes

### 1. Pre-Clean makeup gain compensation
`residue_pre_clean_stage` now includes `pre_clean_makeup_gain`.

If residue pre-clean lowers LUFS more than a small threshold, the worker applies limited makeup gain before premastering, only when true-peak headroom allows it.

Default env behavior:

```text
BUSY_PRE_CLEAN_MAKEUP_GAIN=1
BUSY_PRE_CLEAN_MAKEUP_MIN_LOSS_DB=0.45
BUSY_PRE_CLEAN_MAKEUP_MAX_DB=1.75
BUSY_PRE_CLEAN_MAKEUP_TARGET_TP_DB=-1.05
```

No required env changes; defaults are active.

### 2. Hot candidate render trace
Commercial candidates now use a traced renderer:

```text
after_strategy_input_gain
peak_absorb_soft_clip_N
after_oversampled_candidate_limiter
after_candidate_true_peak_normalize
density_recovery_N
```

This records why limiting is or is not translating virtual gain into final LUFS.

### 3. Peak absorption before limiter
Before final candidate limiting, the worker absorbs excessive true-peak overshoot using stronger soft clipping. This prevents the limiter from simply turning the whole song down.

### 4. Density recovery loop
If the candidate is still below the commercial floor after limiting, the worker attempts small gain + clip + oversampled limit recovery passes. This stops when:

- LUFS no longer improves,
- a hard blocker appears,
- or the candidate approaches the target/floor.

### 5. Role-aware crest tolerance for electronic/club hot candidates
For dense EDM/club/electronic hot-commercial candidates, crest around 4.4-5.0 dB is treated as a QC-visible condition rather than an automatic rejection. True failure still blocks the candidate.

### 6. Standard-final replacement guard
A commercial candidate will not replace the already rendered standard final if it is not louder, unless it reaches the commercial floor. The report now includes:

```text
standard_final_reference_analysis
candidate_rejected_against_standard_final
candidate_vs_standard_lufs_delta_db
```

## Check in stage_state

Look for:

```text
residue_pre_clean_stage.pre_clean_makeup_gain
commercial_loudness_finish.candidates[].conditioning.candidate_render_trace
commercial_loudness_finish.candidates[].conditioning.candidate_render_trace.trace
commercial_loudness_finish.standard_final_reference_analysis
commercial_loudness_finish.selected_candidate_replaced_standard_final_limiter
```

The important diagnostic is whether candidate trace shows LUFS moving like:

```text
after_strategy_input_gain: high LUFS but high TP overshoot
peak_absorb: TP overshoot reduced
after_limiter: LUFS no longer collapses back to -13
recovery: candidate approaches commercial floor unless a real blocker appears
```
