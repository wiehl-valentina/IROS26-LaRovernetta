# AGENTS.md

Guide for agents (Claude Code, Codex, etc.) working in this repo.

## Language

Everything in this repo is written in **English**: code, comments,
docstrings, log messages, docs and commit messages.

## Python environment

This project uses the pyenv virtualenv **venv313** (Python 3.13). The
committed `.python-version` selects it automatically on `cd`; otherwise:

```bash
pyenv activate venv313
```

- Do not create new venvs and do not use the system Python (3.14 breaks the
  pydantic-core build, an SDK dependency).
- With venv313 active, `./scripts/setup.sh` installs everything into it
  (torch, the vendored `genie/` and `traversability/` packages, SDK deps).
  Only when no environment is active does setup.sh create a local `.venv/`
  as a fallback.
- README commands written as `.venv/bin/python ...` are equivalent to plain
  `python ...` with venv313 active.

## Project rules

- 100% self-contained: `genie/` and `traversability/` are vendored here. Do
  not reference paths in other repos and do not install those packages from
  git. `traversability/` mirrors an external PR to the UNLP team's repo — do
  not modify it; mission-v2 code lives in `sana/`.
- `lib/` holds only the Earth Rovers SDK (black box, cloned by setup.sh) and
  is not versioned.
- Anything that moves the rover requires the `--yes-i-want-the-rover-to-move`
  flag; keep that contract in any new script. `--dry-run` must never POST
  `/control` or `/checkpoint-reached`.
- Only the control (main) thread writes to `/control` — keep the
  single-writer invariant if you touch `sana/`.
- Tests (`tests/` and `traversability/tests/`) must keep running without
  torch, checkpoint or network.
- Commits: authored by Santiago only, no co-author trailers or automated
  signatures of any kind.
