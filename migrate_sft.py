"""Migrate verified legacy SFT data once, from the current main environment."""

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent
LEGACY = ROOT / "legacy" / "sft_data"
WORKER = LEGACY / "worker.py"


@contextmanager
def exclusive_lock(path):
    """OS-released locks survive interruption without leaving a stale PID lock."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        print(f"Waiting for migration lock: {path}", flush=True)
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def compatibility_profile(metadata):
    profile = json.loads((LEGACY / "profile.json").read_text(encoding="utf-8"))
    saved = metadata.get("data_environment")
    if saved and any(saved.get(name) != expected for name, expected in profile["data_environment"].items()):
        raise ValueError("No supported legacy compatibility profile matches this SFT data environment.")
    if metadata.get("text_format", {}).get("protocol") != "qwen3-instruct-openr1-v1":
        raise ValueError("No supported legacy compatibility profile matches this SFT text protocol.")
    return profile


def managed_root():
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "qwen-post-training" / "sft-data"


def compatibility_key(profile):
    digest = hashlib.sha256(json.dumps(profile, sort_keys=True).encode())
    for name in ("requirements.txt", "requirements-original.txt"):
        digest.update((LEGACY / name).read_bytes())
    return f"{profile['id']}-{digest.hexdigest()[:12]}"


def find_conda(explicit=None):
    candidate = explicit or os.environ.get("CONDA_EXE") or shutil.which("conda")
    executable = shutil.which(str(candidate)) if candidate else None
    if not executable:
        raise ValueError("Conda was not found. Install Conda or pass --conda-exe /path/to/conda; "
                         "no changes were made to the main environment.")
    return str(Path(executable).resolve())


def child_environment():
    env = os.environ.copy()
    for key in ("PYTHONHOME", "PYTHONPATH"):
        env.pop(key, None)
    env.update(CUDA_VISIBLE_DEVICES="", PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1")
    return env


def run_child(command):
    subprocess.run([str(part) for part in command], check=True, env=child_environment(), cwd=ROOT)


def write_state(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def ensure_compatibility_environment(profile, root, conda_exe=None):
    root = Path(root).expanduser().resolve()
    key = compatibility_key(profile)
    prefix = root / "environments" / key
    python = prefix / "bin" / "python"
    ready = prefix / "sft-data-ready.json"
    state = root / "states" / f"{key}.json"
    with exclusive_lock(root / "locks" / f"{key}.lock"):
        if ready.is_file():
            marker = json.loads(ready.read_text(encoding="utf-8"))
            if marker.get("key") != key or not python.is_file():
                raise ValueError(f"Invalid managed environment marker: {ready}. Use a different --compat-root.")
            try:
                run_child([python, WORKER, "--check-environment"])
            except subprocess.CalledProcessError as error:
                raise ValueError(f"Managed CPU environment verification failed: {prefix}. "
                                 "Use a fresh --compat-root to rebuild it.") from error
            return python
        conda = find_conda(conda_exe)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        if prefix.exists() and not state.is_file():
            raise ValueError(f"Refusing to change an unmanaged existing prefix: {prefix}. Use a fresh --compat-root.")
        if state.is_file() and json.loads(state.read_text(encoding="utf-8")).get("key") != key:
            raise ValueError(f"Invalid managed environment state: {state}. Use a fresh --compat-root.")
        state.parent.mkdir(parents=True, exist_ok=True)
        write_state(state, {"key": key, "state": "creating"})
        if not python.is_file():
            if prefix.exists():
                raise ValueError(f"Interrupted Conda creation left an incomplete prefix: {prefix}. "
                                 "Use a fresh --compat-root to retry; existing files were preserved.")
            print(f"Creating CPU data compatibility environment: {prefix}", flush=True)
            run_child([conda, "create", "--prefix", prefix, "--override-channels", "-c", "conda-forge",
                       "python=3.12", "pip", "-y"])
        print("Installing pinned CPU data dependencies (an interrupted installation can be retried).", flush=True)
        write_state(state, {"key": key, "state": "installing"})
        run_child([python, "-m", "pip", "install", "--isolated", "-r", LEGACY / "requirements.txt"])
        run_child([python, "-m", "pip", "check"])
        run_child([python, WORKER, "--check-environment"])
        write_state(ready, {"key": key, "profile": profile})
        write_state(state, {"key": key, "state": "ready"})
    return python


def checked_cache(target, metadata, tokenizer):
    from prepared_sft_data import load_prepared_sft_data
    from training_workflow import data_plan
    dataset, info = load_prepared_sft_data(target, metadata["config"], tokenizer)
    data_plan(dataset, metadata, info)
    return target


def migrate_sft(source, *, compat_root=None, conda_exe=None):
    from prepare_sft_data import load_tokenizer
    from training_workflow import read_source
    source, metadata, _ = read_source(source)
    if metadata.get("stage", "sft") != "sft" or source.name != "final":
        raise ValueError("Migration requires the original SFT final/ artifact.")
    target = source.parent / "prepared-sft"
    tokenizer = load_tokenizer(source, local_files_only=True)
    with exclusive_lock(source.parent / ".sft-data-migration.lock"):
        if target.exists():
            checked_cache(target, metadata, tokenizer)
            print(f"Reusing verified SFT data: {target}", flush=True)
            return target
        profile = compatibility_profile(metadata)
        python = ensure_compatibility_environment(profile, compat_root or managed_root(), conda_exe)
        # Only a fully checked directory is published. Interrupted staging directories
        # never become a cache; a retry uses another staging directory.
        with tempfile.TemporaryDirectory(prefix=".sft-data-staging-", dir=source.parent) as staging:
            prepared = Path(staging) / "prepared"
            print(f"Preparing and verifying legacy SFT data on CPU: {source}", flush=True)
            run_child([python, WORKER, "--sft-source", source, "--output", prepared])
            checked_cache(prepared, metadata, tokenizer)
            prepared.rename(target)
        print(f"Migrated SFT data: {target}", flush=True)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft-source", type=Path, required=True, help="Legacy SFT final/ directory.")
    parser.add_argument("--conda-exe", help="Conda executable; default: CONDA_EXE or PATH.")
    parser.add_argument("--compat-root", type=Path, help="Managed CPU environment location; default: XDG user cache.")
    args = parser.parse_args()
    try:
        migrate_sft(args.sft_source, compat_root=args.compat_root, conda_exe=args.conda_exe)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        parser.error(f"Migration failed: {error}. The SFT weights and run metadata were preserved.")


if __name__ == "__main__":
    main()
