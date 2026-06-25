# Busy Auto Mastering Public v8.5.4.20.1

## Busy Auto Mixing Start Error Handoff Hotfix

Applies on top of public v8.5.4.20 Busy Auto Mixing checkbox patch.

### Fixes

- Keeps the Busy Auto Mixing checkbox behavior:
  - visible only when a stem ZIP is uploaded
  - default OFF
  - sends `busy_auto_mixing=true` only when selected with a stem ZIP
- Fixes the async bootstrap start path so `/v1/jobs/{job_id}/start` errors are surfaced instead of silently swallowed.
- This matters for Busy Auto Mixing because the private worker correctly returns a clear 409 if Busy Auto Mixing was selected but the stem ZIP is not visible in storage.
- Build ID updated to `v8.5.4.20.1-public-busy-auto-mixing-start-error-hotfix-20260625`.

### Files

- `app.py`
