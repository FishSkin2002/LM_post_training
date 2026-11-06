"""Internal Conda/CPU preparation called by existing training and evaluation entries."""

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

LEGACY = Path(__file__).resolve().parent
ROOT = LEGACY.parents[1]
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
        raise ValueError("Conda was not found. Install Conda or set CONDA_EXE to its executable; "
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


def conda_info(conda):
    result = subprocess.run([conda, "info", "--json"], check=True, capture_output=True, text=True,
                            env=child_environment(), cwd=ROOT, timeout=60)
    return json.loads(result.stdout)


def environment_name(profile):
    return "qwen-sft-data-" + compatibility_key(profile)


def named_prefix(info, name):
    # These directories come from Conda itself, including user envs_dirs settings.
    matches = {Path(directory).expanduser().resolve() / name for directory in info["envs_dirs"]
               if (Path(directory).expanduser() / name).exists()}
    if len(matches) > 1:
        raise ValueError(f"Multiple Conda environments named {name}; resolve the duplicate names in Conda first.")
    return next(iter(matches), None)


def environment_python(conda, name, prefix):
    result = subprocess.run([conda, "run", "--no-capture-output", "-n", name, "python", "-c",
                             "import sys,json;print(json.dumps({'prefix':sys.prefix,'python':sys.executable}))"],
                            check=True, capture_output=True, text=True, env=child_environment(), cwd=ROOT, timeout=60)
    runtime = json.loads(result.stdout)
    python = Path(runtime["python"]).resolve()
    if Path(runtime["prefix"]).resolve() != prefix or not python.is_file():
        raise ValueError(f"Conda resolved a different or incomplete environment for {name}.")
    return python


def ensure_compatibility_environment(profile, conda_exe=None):
    conda = find_conda(conda_exe)
    info = conda_info(conda)
    key, name = compatibility_key(profile), environment_name(profile)
    conda_root = str(Path(info["root_prefix"]).resolve())
    root_key = hashlib.sha256(conda_root.encode()).hexdigest()[:12]
    root = managed_root() / root_key
    state = root / "states" / f"{name}.json"
    identity = {"key": key, "name": name, "conda_root": conda_root}
    with exclusive_lock(root / "locks" / f"{name}.lock"):
        prefix = named_prefix(conda_info(conda), name)
        saved = json.loads(state.read_text(encoding="utf-8")) if state.is_file() else None
        ready = prefix / "sft-data-ready.json" if prefix else None
        marker = json.loads(ready.read_text(encoding="utf-8")) if ready and ready.is_file() else None
        for record in (saved, marker):
            if record and any(record.get(field) != value for field, value in identity.items()):
                raise ValueError(f"Invalid managed environment state for {name}; existing files were preserved.")
        if prefix is not None and not (saved or marker):
            raise ValueError(f"Refusing to change an unmanaged existing Conda environment: {name} ({prefix}).")
        if saved and saved.get("prefix") and prefix and Path(saved["prefix"]) != prefix:
            raise ValueError(f"Managed Conda environment {name} moved; restore its original Conda configuration.")
        state.parent.mkdir(parents=True, exist_ok=True)
        if prefix is None:
            write_state(state, {**identity, "state": "creating"})
            print(f"Creating CPU compatibility environment with Conda: {name}", flush=True)
            run_child([conda, "create", "-n", name, "--override-channels", "-c", "conda-forge",
                       "python=3.12", "pip", "-y"])
            prefix = named_prefix(conda_info(conda), name)
            if prefix is None:
                raise ValueError(f"Conda did not report the newly created environment {name}.")
        try:
            python = environment_python(conda, name, prefix)
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            raise ValueError(f"Conda environment {name} is incomplete. Repair or remove this managed environment "
                             "with Conda, then rerun your original training/evaluation command.") from error
        if marker:
            try:
                run_child([python, WORKER, "--check-environment"])
            except subprocess.CalledProcessError as error:
                raise ValueError(f"Managed CPU environment {name} failed verification. Repair it with Conda, "
                                 "then rerun your original command.") from error
            return python
        print(f"Installing pinned CPU data dependencies: {name} ({prefix})", flush=True)
        write_state(state, {**identity, "prefix": str(prefix), "state": "installing"})
        run_child([python, "-m", "pip", "install", "--isolated", "-r", LEGACY / "requirements.txt"])
        run_child([python, "-m", "pip", "check"])
        run_child([python, WORKER, "--check-environment"])
        write_state(prefix / "sft-data-ready.json", {**identity, "profile": profile})
        write_state(state, {**identity, "prefix": str(prefix), "state": "ready"})
    return python


def checked_cache(target, metadata, tokenizer):
    from prepared_sft_data import load_prepared_sft_data
    from training_workflow import data_plan
    dataset, info = load_prepared_sft_data(target, metadata["config"], tokenizer)
    data_plan(dataset, metadata, info)
    return target


def migrate_sft(source, *, conda_exe=None):
    from prepare_sft_data import load_tokenizer
    from training_workflow import read_source
    source, metadata, _ = read_source(source)
    if metadata.get("stage", "sft") != "sft":
        raise ValueError("Legacy data preparation requires an SFT artifact.")
    target = source.parent / "prepared-sft"
    tokenizer = load_tokenizer(source, local_files_only=True)
    with exclusive_lock(source.parent / ".sft-data-migration.lock"):
        if target.exists():
            checked_cache(target, metadata, tokenizer)
            print(f"Reusing verified SFT data: {target}", flush=True)
            return target
        profile = compatibility_profile(metadata)
        python = ensure_compatibility_environment(profile, conda_exe)
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
