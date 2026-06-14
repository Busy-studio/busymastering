# Busy Auto Mastering - Public Frontend Only

This repository is safe for public GitHub. It contains only the Streamlit UI and private worker client.

The actual mastering engine must run in a private Cloud Run worker.

## Streamlit Secrets

```toml
PRIVATE_WORKER_URL = "https://your-cloud-run-worker-url"
PRIVATE_WORKER_TOKEN = "same-token-as-worker"
PRIVATE_WORKER_TIMEOUT_SEC = "60"
PRIVATE_WORKER_START_TIMEOUT_SEC = "3600"
```

This version uses signed upload flow:

```text
Streamlit -> private worker /v1/jobs/init -> signed upload URL
Streamlit -> object storage signed URL -> WAV upload
Streamlit -> private worker /v1/jobs/{job_id}/start -> mastering
Streamlit -> signed result URL -> download
```

The public repo does not include analysis, role extraction, DSP mapping, model prompts, private rulepack, calibration data, or rendering code.
