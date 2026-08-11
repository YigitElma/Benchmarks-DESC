"""Shared result storage for the benchmark scripts.

Results live in one folder per branch (or commit), e.g. results/master/. Inside,
there is one JSON file per script, device and profile mode:

    results/<branch>/<script>_<device>_<profile mode>.json

Each file holds a "runs" dict keyed by the script's settings, so re-running a
script with the same resolution and chunk sizes overwrites that entry while a
different setting is stored next to it. Memory traces from memory-profile.py go
next to it as <script>_<device>_memory_<settings id>.npz, one per setting, the
id being a short hash of the same key. Read it all back with compare-results.py.
"""

import datetime
import glob
import hashlib
import json
import os
import subprocess

REPO_DIR = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))


def config_key(config):
    """Stable one line key for a settings dict, e.g. 'n_coils=5,res=8'."""
    return ",".join(f"{k}={_plain(config[k])}" for k in sorted(config))


def _plain(value):
    """Coerce a setting to something JSON can hold."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if hasattr(value, "item"):  # numpy scalar
        return value.item()
    return str(value)


def stem(script):
    """Script name without directory or extension."""
    return os.path.splitext(os.path.basename(script))[0]


def result_path(save_dir, script, device, profile_mode):
    """Path of the JSON file holding the runs of one script."""
    return os.path.join(save_dir, f"{stem(script)}_{device}_{profile_mode}.json")


def config_id(key):
    """Short stable id of a settings key, the keys are too long for a file name."""
    return hashlib.sha1(key.encode()).hexdigest()[:8]


def trace_path(save_dir, script, device, key=None):
    """Path of the memory trace of one script at one setting."""
    suffix = f"_{config_id(key)}" if key else ""
    return os.path.join(save_dir, f"{stem(script)}_{device}_memory{suffix}.npz")


def last_run_key(save_dir, script, device):
    """Settings key the script stored last, so the trace can be named after it."""
    data = load_file(result_path(save_dir, script, device, "memory"))
    return (data or {}).get("last_run")


def git_commit():
    """Short hash of the DESC commit being benchmarked, None if unavailable."""
    try:
        out = subprocess.run(
            ["git", "-C", REPO_DIR, "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def load_file(path):
    """Contents of one result file, None if it does not exist."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def save_result(save_dir, script, device, profile_mode, config, t_compile, times):
    """Store one timing run, replacing any earlier run with the same settings."""
    if not save_dir:
        return None
    os.makedirs(save_dir, exist_ok=True)
    path = result_path(save_dir, script, device, profile_mode)
    data = load_file(path) or {
        "script": stem(script),
        "device": device,
        "profile_mode": profile_mode,
        "runs": {},
    }
    key = config_key(config)
    data["runs"][key] = {
        "config": {k: _plain(v) for k, v in config.items()},
        "n_repeat": len(times),
        "t_compile": t_compile,
        "times": list(times),
        "commit": git_commit(),
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    # memory-profile.py names the trace it is about to write after this
    data["last_run"] = key
    with open(path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    print(f"saved result to {path}")
    return path


def branch_dir(results_dir, branch):
    """Folder holding the results of one branch, / -> - as in the drivers."""
    return os.path.join(results_dir, branch.replace("/", "-"))


def load_branch(results_dir, branch, device, profile_mode, script=None):
    """All runs of one branch as {script: {config key: run}}."""
    pattern = f"{script or '*'}_{device}_{profile_mode}.json"
    folder = branch_dir(results_dir, branch)
    out = {}
    for path in sorted(glob.glob(os.path.join(folder, pattern))):
        data = load_file(path)
        if data and data.get("runs"):
            out[data["script"]] = data["runs"]
    return out
