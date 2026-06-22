# JOURNAL — TRINITY replication lab notebook

This is the running log of **mistakes, findings, and decisions**. See `AGENTS.md` §6 for the
protocol. **Newest entries at the top.** Tag each entry with one or more of:
`#mistake` `#finding` `#decision` `#repro` `#gotcha` `#todo`.

### Entry template

```
## YYYY-MM-DD — short title  #tag #tag
**Context:** what we were doing.
**Expected:** what we thought would happen.
**Actual:** what happened (paste the error/number).
**Root cause:** why.
**Fix / decision:** what we changed and why.
**Follow-up:** anything left open.
```

---

## 2026-06-22 — Remote H200 box inventory  #finding

**Context:** Inventoried `trinity-gpu` (read-only) before writing the remote setup path.

**Findings:**
- **Ubuntu 24.04**, **192 vCPU**, **~2 TB RAM**. `$HOME` = `/mnt/data/harshal` on a 12 TB array
  with **3.2 TB free** — ample for HF model caches.
- **8× NVIDIA H200 NVL (143 GB each)**, driver `595.71.05`. **We use index 5 only.**
- Python **3.12.3** present; **no `uv`, no `conda`, no `torch`** system-wide → `setup_remote.sh`
  installs `uv` and builds a project `.venv`.
- Network: `huggingface.co` → 200 (model downloads OK). `api.fireworks.ai` root → 404, which is
  expected (the API lives under `/inference/v1`; root has no handler). Not a problem.
- No pre-existing `~/trinity`; we sync there fresh.

**Decision:** default `TRINITY_REMOTE_DIR=$HOME/trinity` (= `/mnt/data/harshal/trinity`),
`HF_HOME` under the project dir to avoid polluting shared `$HOME`.

---

## 2026-06-22 — Project bootstrap & environment verification  #finding #decision

**Context:** Kicking off the replication. Verified the full toolchain before writing code.

**Findings:**
- **GPU box reachable.** `ssh trinity-gpu` works. `nvidia-smi -i 5` reports **NVIDIA H200 NVL,
  143771 MiB, idle**. We are allocated **GPU 5 only** — all CUDA work pins
  `CUDA_VISIBLE_DEVICES=5`.
- **All three Fireworks models answer** a chat completion (HTTP 200):
  `accounts/fireworks/models/deepseek-v4-pro`, `.../glm-5p2`, `.../kimi-k2p6`.
- **GitHub:** authed as `harrrshall` with `repo` scope; repo will be created **private**.

**Decisions:**
- Secrets live **outside the repo**: SSH key at `~/.ssh/trinity_gpu` (600) behind the
  `trinity-gpu` host alias; Fireworks key at `~/.config/trinity/secrets.env` (600), read via
  `FIREWORKS_API_KEY`. `.gitignore` blocks all secret patterns + the 11 MB paper PDF.
- Replication target is the paper's **relative** claims (TRINITY > best single model > random
  routing; sep-CMA-ES > RL/IL/random), since our open-source model pool differs from the
  paper's. See `AGENTS.md` §1.

## 2026-06-22 — Fireworks account model-list endpoint 500s  #gotcha

**Context:** Tried `GET /inference/v1/models` to enumerate available models.
**Actual:** `HTTP 500 — "Error listing deployed models"`.
**Root cause:** That account-level listing endpoint is flaky / not enabled for this key; it is
not needed.
**Fix / decision:** Probe model IDs directly with a tiny `chat/completions` call instead. All
three target IDs returned 200, so we hardcode them in `configs/models.yaml`. Do **not** depend
on the list endpoint.
