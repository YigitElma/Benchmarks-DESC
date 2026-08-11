#!/usr/bin/env python3
"""Plot the memory traces of consecutive versions, one figure per pair.

Makes one figure for every consecutive pair of VERSIONS below, saved as

    memory-<script>-<device>-<version1>-<version2>.png

Every version gets its own color from PALETTE and keeps it in each figure it
appears in, and all figures of a script share their axes, so the frames stitch
into a gif, memory-<script>-<device>.gif, written next to them. Set WINDOW to
put more than two versions in one figure. VERSIONS name the result folders
written by the drivers.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import bench_io

# --- Configuration: edit these as needed ---
VERSIONS = [
    "master",
    "v0.17.3",
    "v0.17.2",
    "v0.17.1",
    "v0.17.0",
    "v0.16.0",
    "v0.15.0",
    "v0.14.2",
    "v0.14.1",
    "v0.14.0",
    "v0.13.0",
    "v0.12.3",
    "v0.12.2",
]
DEVICE = "gpu"  # cpu or gpu
SCRIPT = None  # None for every script found, or e.g. "07_prox_jac_qa_coils"
RESULTS_DIR = "results"
OUT_DIR = None  # None puts the figures in RESULTS_DIR
WINDOW = 2  # versions per figure
CONFIG = None  # None for any settings, or a settings key to pin
SHARED_AXES = True  # same axes on every figure of a script, for the gif
THRESHOLD = 100.0  # memory rise (MB) that marks the start of a run
DPI = 150
MAKE_GIF = True  # stitch the figures of a script into a gif
FRAME_SECONDS = 1.0  # how long one frame is shown
GIF_WIDTH = 1200  # px, None keeps the figure size
# a version's color is its position in VERSIONS, so it stays the same in every
# figure it appears in and neighbours never share one
PALETTE = [
    "tab:red",
    "tab:blue",
    "tab:green",
    "tab:orange",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:olive",
    "tab:cyan",
    "black",
]
# --------------------------------------------


def auto_trim(mem, threshold=100.0):
    """Index of the first sample where memory rises above the initial baseline."""
    baseline = np.median(mem[: max(5, len(mem) // 100)])
    above = mem > baseline + threshold
    return int(np.argmax(above)) if above.any() else 0


def pick_run(runs, prefer_key):
    """Which settings of a script to plot, defaults change between versions."""
    if not runs:
        return None, None
    if prefer_key in runs:
        return prefer_key, runs[prefer_key]
    if len(runs) == 1:
        key = next(iter(runs))
        return key, runs[key]
    # several settings and none matches, take the one that ran last
    key = max(runs, key=lambda k: runs[k].get("timestamp", ""))
    return key, runs[key]


def load_traces(script):
    """{version: (key, run, t, mem)} for one script, versions without it dropped."""
    out = {}
    prefer_key = CONFIG
    for version in VERSIONS:
        runs = bench_io.load_branch(RESULTS_DIR, version, DEVICE, "memory", script).get(
            script, {}
        )
        key, run = pick_run(runs, prefer_key)
        if key is None:
            continue
        path = bench_io.trace_path(
            bench_io.branch_dir(RESULTS_DIR, version), script, DEVICE, key
        )
        if not os.path.exists(path):
            continue
        trace = np.load(path)
        t, mem = trace["time"], trace["memory"]
        trim = auto_trim(mem, THRESHOLD)
        out[version] = (key, run, t[trim:] - t[trim], mem[trim:])
        if CONFIG is None:
            prefer_key = key  # keep the next version on these settings if it has them
    return out


if __name__ == "__main__":
    if WINDOW < 2 or WINDOW > len(VERSIONS):
        raise SystemExit(f"WINDOW must be between 2 and {len(VERSIONS)}")
    if WINDOW > len(PALETTE):
        raise SystemExit(f"PALETTE needs at least WINDOW={WINDOW} colors")

    out_dir = OUT_DIR or RESULTS_DIR
    os.makedirs(out_dir, exist_ok=True)
    color_of = {v: PALETTE[i % len(PALETTE)] for i, v in enumerate(VERSIONS)}

    scripts = sorted(
        {
            s
            for v in VERSIONS
            for s in bench_io.load_branch(RESULTS_DIR, v, DEVICE, "memory", SCRIPT)
        }
    )
    if not scripts:
        raise SystemExit(f"no memory results for {DEVICE} in {RESULTS_DIR}")

    for script in scripts:
        traces = load_traces(script)
        missing = [v for v in VERSIONS if v not in traces]
        if missing:
            print(f"{script}: no trace for {', '.join(missing)}")
        if len(traces) < 2:
            continue

        xlim = ylim = None
        if SHARED_AXES:
            xlim = (0, max(t[-1] for _, _, t, _ in traces.values()) * 1.02)
            ylim = (0, max(m.max() for _, _, _, m in traces.values()) * 1.05)

        frames = []
        for start in range(len(VERSIONS) - WINDOW + 1):
            window = VERSIONS[start : start + WINDOW]
            if not any(v in traces for v in window):
                continue
            plt.figure(figsize=(16, 6))
            for version in window:
                if version not in traces:
                    continue
                key, run, t, mem = traces[version]
                label = f"{version} ({run['commit']})" if run.get("commit") else version
                plt.plot(t, mem, label=label, color=color_of[version])
                print(
                    f"  {version:16s} {color_of[version]:12s} "
                    f"peak {mem.max():7.0f} MB  {key}"
                )
            if SHARED_AXES:
                plt.xlim(*xlim)
                plt.ylim(*ylim)
            plt.xlabel("Time (s)", fontsize=14)
            plt.ylabel(f"{DEVICE.upper()} Memory (MB)", fontsize=14)
            plt.title(f"{script}  [{DEVICE}]  {' vs '.join(window)}")
            plt.grid(True)
            plt.legend(loc="upper left")
            plt.tight_layout()

            tag = "-".join(v.replace("/", "-") for v in window)
            out = os.path.join(out_dir, f"memory-{script}-{DEVICE}-{tag}.png")
            plt.savefig(out, dpi=DPI)
            plt.close()
            frames.append(out)
            print(f"saved {out}\n")

        if MAKE_GIF and len(frames) > 1:
            images = []
            for path in frames:
                img = Image.open(path).convert("RGB")
                if GIF_WIDTH:
                    w, h = img.size
                    size = (GIF_WIDTH, round(h * GIF_WIDTH / w))
                    img = img.resize(size, Image.LANCZOS)
                images.append(img)
            gif = os.path.join(out_dir, f"memory-{script}-{DEVICE}.gif")
            images[0].save(
                gif,
                save_all=True,
                append_images=images[1:],
                duration=round(FRAME_SECONDS * 1000),
                loop=0,
            )
            print(f"saved {gif} ({len(images)} frames, {FRAME_SECONDS}s each)\n")
