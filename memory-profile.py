#!/usr/bin/env python3
"""Sample RAM/VRAM usage of a script over time and save the trace to <save_dir>.

Usage: python memory-profile.py <cpu|gpu> <save_dir> <script> [n_repeat] [interval]

<save_dir> is the branch folder the trace is written to, e.g. results/master.
<script> is the script to profile, e.g. scripts/06_prox_jac_qa.py. It is called
as `<script> <cpu|gpu> memory <n_repeat> <save_dir>`, so it stores its own
settings next to the trace; all other run settings are defined inside it.

CPU reads the child's RSS straight from /proc, GPU queries NVML in-process, so
both reach the kHz range and can resolve the ~ms transients of a single XLA op.
<interval> is the seconds to sleep between samples, default 1e-4 (~5 kHz, a few
percent of one core). Setting it to 0 spins at ~400 kHz, which saturates a core
and produces multi-GB traces, so only use it to chase sub-ms transients.
Plot the saved traces with compare-results.py.
"""

import os
import subprocess
import sys
import time
import threading
import numpy as np

from bench_io import last_run_key, trace_path


def monitor_ram(proc, interval, ram_usage, timestamps):
    """Sample the child's resident set size until *proc* finishes."""
    page_mb = os.sysconf("SC_PAGE_SIZE") / 1024 / 1024
    statm = open(f"/proc/{proc.pid}/statm", "rb")

    def sample():
        statm.seek(0)
        ram_usage.append(int(statm.read(64).split()[1]) * page_mb)
        timestamps.append(time.time())

    try:
        while proc.poll() is None:  # child still running?
            sample()
            if interval:
                time.sleep(interval)

        # keep watching for an extra second
        end = time.time() + 2.0
        while time.time() < end:
            sample()
            time.sleep(max(interval, 1e-4))
    except (OSError, IndexError):  # child exited, /proc entry is gone
        pass
    finally:
        statm.close()


def monitor_vram(proc, interval, vram_usage, timestamps):
    """Sample total GPU memory until *proc* finishes."""
    import pynvml

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    def sample():
        vram_usage.append(pynvml.nvmlDeviceGetMemoryInfo(handle).used / 1024 / 1024)
        timestamps.append(time.time())

    while proc.poll() is None:
        sample()
        if interval:
            time.sleep(interval)

    # keep watching for an extra second
    end = time.time() + 2.0
    while time.time() < end:
        sample()
        time.sleep(max(interval, 1e-4))
    pynvml.nvmlShutdown()


if __name__ == "__main__":
    if len(sys.argv) not in (4, 5, 6) or sys.argv[1].lower() not in ("cpu", "gpu"):
        sys.exit(
            f"usage: python {sys.argv[0]} <cpu|gpu> <save_dir> <script> "
            "[n_repeat] [interval]"
        )
    device = sys.argv[1].lower()
    save_dir = sys.argv[2]
    script = sys.argv[3]
    n_repeat = sys.argv[4] if len(sys.argv) >= 5 else "1"
    interval = float(sys.argv[5]) if len(sys.argv) == 6 else 1e-5
    os.makedirs(save_dir, exist_ok=True)

    mem_usage = []
    ts = []

    # launch the script to be profiled, always in memory profiling mode
    child = subprocess.Popen(
        [sys.executable, script, device, "memory", n_repeat, save_dir]
    )

    # start the sampler thread for the memory of the requested device
    sampler = threading.Thread(
        target=monitor_vram if device == "gpu" else monitor_ram,
        args=(child, interval, mem_usage, ts),
        daemon=True,
    )
    sampler.start()

    # wait until the child exits, then join the sampler
    child.wait()
    sampler.join()
    if child.returncode != 0:
        sys.exit(f"{script} failed with exit code {child.returncode}, no trace saved")

    # save the trace; plotting is done separately by compare-results.py
    mem_usage = np.asarray(mem_usage)
    if device == "cpu":
        # drop the interpreter's own baseline RSS
        mem_usage = mem_usage - min(mem_usage)
    times = np.asarray(ts) - ts[0]
    # one trace per setting, named after what the script just stored
    path = trace_path(save_dir, script, device, last_run_key(save_dir, script, device))
    np.savez_compressed(path, time=times, memory=mem_usage)
    rate = len(times) / times[-1] / 1e3
    print(f"saved memory trace to {path} ({rate:.1f} kHz)")
