# Busy Auto Mastering v8.5.3.64.0.4.1

## Decision Replay Selected Delivery Semantics Hotfix

This hotfix reinforces the v64.0.3 selection semantics after the v64.0.4 Commercial/Maximum Density Push Foundation.

### Fix

`finishing_intelligence._decision_replay()` now treats a v64.0.4 `selected_delivery_candidate` or matching `selected_candidate_id` as the actual delivery selection even when a compact/synthetic candidate payload does not explicitly carry `accepted=true`.

The selected delivery candidate is serialized as:

```text
accepted = true
rejected = false
```

Quality or translation concerns remain separated as:

```text
quality_gate = pass / caution / fail
quality_rejected = true / false
warning_advisory = true / false
warning_blocking = true / false
```

### Why

v64.0.4 introduced actual loud candidates such as `commercial_density_candidate` and `maximum_push_candidate`. Decision replay must not regress to the old contradiction where the selected delivery candidate can appear rejected merely because codec/translation quality telemetry exists or because compact candidate telemetry omitted explicit accepted/rejected fields.

### Tests

- `py_compile` passed
- `compileall` passed
- import smoke passed
- Commercial/Maximum density smoke passed
- v64.0.3/v64.0.4 selected delivery semantics smoke passed
- `master_audio()` routed smoke confirmed v64.0.4 activates and selects `maximum_push_candidate` at approximately `-0.1 dBTP`
