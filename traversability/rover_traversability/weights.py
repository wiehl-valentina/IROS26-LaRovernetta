"""Checkpoint and config resolution for the SAM-TP traversability model.

The fine-tuned Mini+ checkpoint is ~130 MB and is NOT committed to this repo.
Resolution order:

1. Explicit ``checkpoint=`` argument.
2. ``SAMTP_CHECKPOINT`` environment variable (path to a local .pt file).
3. The local cache dir (``~/.cache/rover_traversability/``).
4. Hugging Face Hub download from ``SAMTP_HF_REPO`` (default:
   ``sanatem/samtp-mini-traversability``, public) — requires
   ``huggingface_hub`` (``pip install rover-traversability[hf]``).

The inference config ships inside the vendored ``genie`` package (installed as
top-level ``sam2``) — we point hydra at the real directory on disk.
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_CHECKPOINT_FILENAME = "checkpoint_finetuned_v2.pt"
DEFAULT_HF_REPO = "sanatem/samtp-mini-traversability"
CONFIG_RELPATH = "configs/sam2.1_inference_tiny/sam2.1_custom2.yaml"

ENV_CHECKPOINT = "SAMTP_CHECKPOINT"
ENV_HF_REPO = "SAMTP_HF_REPO"
ENV_HF_FILENAME = "SAMTP_HF_FILENAME"
ENV_CONFIG = "SAMTP_CONFIG"

# sha256 of the known-good Mini+ fine-tuned checkpoint (v2, ~50k frames).
CHECKPOINT_V2_SHA256 = "44e508da3d36a63431f8197f16784c980abf43ea94fc4e524bcd19d0646692bd"


class CheckpointNotFoundError(FileNotFoundError):
    """No usable SAM-TP checkpoint could be located."""


class SamNotInstalledError(ImportError):
    """The vendored sam2 package (from ./genie) is not installed."""


def default_cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return Path(base) / "rover_traversability"


def _not_found_message(hf_error: str | None = None) -> str:
    lines = [
        f"SAM-TP checkpoint not found ({DEFAULT_CHECKPOINT_FILENAME}, ~130 MB, "
        "SAM2.1 Hiera-tiny fine-tuned on Mini+ footage). Options:",
        f"  1. Set {ENV_CHECKPOINT}=/path/to/{DEFAULT_CHECKPOINT_FILENAME}",
        f"  2. Drop the file into {default_cache_dir()}/",
        f"  3. Enable auto-download: pip install 'rover-traversability[hf]' "
        f"(fetches {os.environ.get(ENV_HF_REPO, DEFAULT_HF_REPO)} from the Hugging Face Hub)",
        "See traversability/README.md § Weights for how to get access.",
    ]
    if hf_error:
        lines.append(f"(Hugging Face download was attempted and failed: {hf_error})")
    return "\n".join(lines)


def resolve_checkpoint(
    explicit: str | Path | None = None,
    hf_repo: str | None = None,
    hf_filename: str | None = None,
    auto_download: bool = True,
) -> Path:
    """Locate the checkpoint file, downloading from HF Hub as a last resort.

    An explicitly-provided path (argument or env var) that does not exist is an
    error, never a silent fall-through — a typo'd path must not quietly resolve
    to some other checkpoint.
    """
    if explicit is not None:
        p = Path(explicit).expanduser()
        if p.is_file():
            return p
        raise CheckpointNotFoundError(f"checkpoint path given explicitly but not found: {p}")

    env_path = os.environ.get(ENV_CHECKPOINT)
    if env_path:
        p = Path(env_path).expanduser()
        if p.is_file():
            return p
        raise CheckpointNotFoundError(f"${ENV_CHECKPOINT} is set but the file does not exist: {p}")

    filename = hf_filename or os.environ.get(ENV_HF_FILENAME) or DEFAULT_CHECKPOINT_FILENAME
    cached = default_cache_dir() / filename
    if cached.is_file():
        return cached

    hf_error: str | None = None
    if auto_download:
        repo = hf_repo or os.environ.get(ENV_HF_REPO) or DEFAULT_HF_REPO
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            hf_error = "huggingface_hub is not installed (pip install 'rover-traversability[hf]')"
        else:
            try:
                return Path(hf_hub_download(repo_id=repo, filename=filename))
            except Exception as exc:  # gated repo, no token, offline, 404 ...
                hf_error = f"{type(exc).__name__}: {exc}"

    raise CheckpointNotFoundError(_not_found_message(hf_error))


def resolve_config(explicit: str | Path | None = None) -> Path:
    """Locate the tiny SAM-TP inference config inside the installed sam2 package.

    Returns a real filesystem path — sam2's ``build_sam2`` hands the directory
    to hydra's ``initialize_config_dir``, which cannot read from a zip.
    """
    if explicit is not None:
        p = Path(explicit).expanduser()
        if p.is_file():
            return p
        raise FileNotFoundError(f"config path given explicitly but not found: {p}")

    env_path = os.environ.get(ENV_CONFIG)
    if env_path:
        p = Path(env_path).expanduser()
        if p.is_file():
            return p
        raise FileNotFoundError(f"${ENV_CONFIG} is set but the file does not exist: {p}")

    try:
        import sam2
    except ImportError as exc:
        raise SamNotInstalledError(
            "The sam2 package is not installed. It is vendored in this repo — "
            "from the repository root run:\n"
            "    pip install torch torchvision\n"
            "    pip install --no-build-isolation -e ./genie\n"
            "then retry."
        ) from exc

    cfg = Path(sam2.__path__[0]) / CONFIG_RELPATH
    if not cfg.is_file():
        raise FileNotFoundError(
            f"sam2 is installed at {sam2.__path__[0]} but the SAM-TP inference config "
            f"is missing ({CONFIG_RELPATH}). Your sam2 install likely dropped its "
            "package data — reinstall from this repo's ./genie directory."
        )
    return cfg
