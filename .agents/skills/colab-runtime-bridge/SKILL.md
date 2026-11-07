---
name: colab-runtime-bridge
description: Connect to the user's TEBDIL Colab runtime when given a trycloudflare URL or bridge connection JSON; run and recover authorized research jobs.
---

# TEBDIL Colab bridge

Paste all of [scripts/cell.py](scripts/cell.py) into one Colab cell. No repository checkout is needed for bootstrap. The cell selects an available localhost port, rotates a random bearer token, replaces only its own server/tunnel, retries tunnel startup, and checks authenticated public health before printing connection JSON.

The URL **and token** grant shell access to the runtime, including mounted Drive. Share privately; never commit either or put the token in URL query parameters. Rerunning rotates credentials and the URL. A URL alone is insufficient: request the connection JSON, not a description of what to do.

## Agent procedure

1. Read the current task and repo instructions. Treat supplied connection JSON as authorization to connect for that task, not permission for unrelated work.
2. GET the URL with `Authorization: Bearer <token>`. Require JSON `status=ready, protocol=1`.
3. POST JSON `{"cmd":"pwd; nvidia-smi; ls /content", "timeout":20}` with the same header. Inspect checkout, Git revision, storage and active jobs before changing anything.
4. Sync the intended code revision without overwriting runtime results. Use `scripts/run_colab_l4.py` for supported suites. Run one GPU worker first; do not assume L4 memory supports three model copies.
5. Launch long jobs detached, redirect output to an explicit log, and capture PID. Requests must finish within 30 seconds. Poll job status and log tails with short requests. Never rerun a launch merely because its HTTP response was lost: inspect PID, command and output first.
6. Use `--checkpoint-dir` on persistent storage; generation checkpoints bind code, model and inputs and resume exact matches only. Choose a fresh `--output` summary path when resuming. Keep results in uniquely named run directories. Copy completed records to durable storage after each task. `/content` and background jobs do not survive runtime replacement. Preserve incomplete records separately; never score interrupted generations as completed.
7. On tunnel failure retry health with bounded backoff, then request rerunning the same cell. On runtime replacement reconnect, inspect durable checkpoints and resume only missing tasks. Never claim a lost process resumed.
8. Download results, verify hashes and completion states, run local validators, then update existing site sections only when evidence changes.

## Lifecycle and limits

States: starting → ready → disconnected → reconnecting → ready; runtime loss → user reruns cell → new runtime. Google can revoke the VM. No cell can guarantee uptime or restart a deleted runtime without user action. This is not a keepalive or quota bypass. The tunnel process reconnects transient transport failures; a dead process needs a cell rerun. Command timeout is not proof all descendant processes stopped. Inspect before retrying.
