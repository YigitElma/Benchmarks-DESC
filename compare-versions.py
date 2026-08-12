#!/usr/bin/env python3
"""Plot the memory traces of consecutive versions, one figure per pair.

Makes one figure for every consecutive pair of VERSIONS below, saved as

    memory-<script>-<device>-<version1>-<version2>-<settings id>.png

Every figure, and so the whole gif, is one single setting: CONFIG_ID below is
the short hash the traces are named after, and only runs with that hash are
plotted. Leave it None to list the hashes available for each script and stop.

Every version gets its own color from PALETTE and keeps it in each figure it
appears in, and all figures of a script share their axes, so the frames stitch
into a gif, memory-<script>-<device>-<settings id>.gif, written next to them.
Set WINDOW to put more than two versions in one figure. VERSIONS name the
result folders written by the drivers.
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
    "v0.12.2",
    "v0.12.3",
    "v0.13.0",
    "v0.14.0",
    "v0.14.1",
    "v0.14.2",
    "v0.15.0",
    "v0.16.0",
    "v0.17.0",
    "v0.17.1",
    "v0.17.2",
    "v0.17.3",
    "master",
]
DEVICE = "gpu"  # cpu or gpu
SCRIPT = None  # None for every script found, or e.g. "07_prox_jac_qa_coils"
RESULTS_DIR = "results"
OUT_DIR = None  # None puts the figures in RESULTS_DIR
WINDOW = 2  # versions per figure
CONFIG_ID = "553aa658"  # settings hash to plot, None lists the ones available and stops
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


def runs_of(script, version):
    """The stored memory runs of one script and version, {settings key: run}."""
    return bench_io.load_branch(RESULTS_DIR, version, DEVICE, "memory", script).get(
        script, {}
    )


def load_traces(script):
    """{version: (key, run, t, mem)} at CONFIG_ID, versions without it dropped."""
    out = {}
    for version in VERSIONS:
        for key, run in runs_of(script, version).items():
            if bench_io.config_id(key) != CONFIG_ID:
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
    return out


def list_configs(scripts):
    """What CONFIG_ID can be set to, and how many versions each one covers."""
    for script in scripts:
        seen = {}
        for version in VERSIONS:
            for key in runs_of(script, version):
                folder = bench_io.branch_dir(RESULTS_DIR, version)
                if os.path.exists(bench_io.trace_path(folder, script, DEVICE, key)):
                    seen.setdefault(key, []).append(version)
        print(f"\n{script}  [{DEVICE}]")
        for key, versions in sorted(seen.items(), key=lambda kv: -len(kv[1])):
            print(
                f"  {bench_io.config_id(key)}  {len(versions):2d}/{len(VERSIONS)} "
                f"versions  {key}"
            )


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

    if CONFIG_ID is None:
        list_configs(scripts)
        raise SystemExit("\nset CONFIG_ID to one of the hashes above")

    for script in scripts:
        traces = load_traces(script)
        missing = [v for v in VERSIONS if v not in traces]
        if missing:
            print(f"{script}: nothing at {CONFIG_ID} for {', '.join(missing)}")
        if len(traces) < 2:
            continue

        xlim = ylim = None
        if SHARED_AXES:
            xlim = (0, max(t[-1] for _, _, t, _ in traces.values()) * 1.02)
            ylim = (0, max(m.max() for _, _, _, m in traces.values()) * 1.05)

        # every trace is at CONFIG_ID, so they all carry the same settings
        settings = next(iter(traces.values()))[0]

        frames = []
        for start in range(len(VERSIONS) - WINDOW + 1):
            window = VERSIONS[start : start + WINDOW]
            if not any(v in traces for v in window):
                continue
            plt.figure(figsize=(16, 6))
            for version in window:
                if version not in traces:
                    continue
                _, run, t, mem = traces[version]
                label = f"{version} ({run['commit']})" if run.get("commit") else version
                plt.plot(t, mem, label=label, color=color_of[version])
                print(
                    f"  {version:16s} {color_of[version]:12s} peak {mem.max():7.0f} MB"
                )
            if SHARED_AXES:
                plt.xlim(*xlim)
                plt.ylim(*ylim)
            plt.xlabel("Time (s)", fontsize=14)
            plt.ylabel(f"{DEVICE.upper()} Memory (MB)", fontsize=14)
            plt.title(
                f"{script}  [{DEVICE}]  {' vs '.join(window)}\n{settings}", fontsize=12
            )
            plt.grid(True)
            plt.legend(loc="upper left")
            plt.tight_layout()

            tag = "-".join(v.replace("/", "-") for v in window)
            name = f"memory-{script}-{DEVICE}-{tag}-{CONFIG_ID}.png"
            out = os.path.join(out_dir, name)
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
            gif = os.path.join(out_dir, f"memory-{script}-{DEVICE}-{CONFIG_ID}.gif")
            images[0].save(
                gif,
                save_all=True,
                append_images=images[1:],
                duration=round(FRAME_SECONDS * 1000),
                loop=0,
            )
            print(f"saved {gif} ({len(images)} frames, {FRAME_SECONDS}s each)\n")
