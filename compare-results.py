#!/usr/bin/env python3
"""Compare the benchmark results of several branches.

Usage: python compare-results.py <branch1> <branch2> [...] [options]

Reads the result folders written by run-benchmark.sh and run-memory-profile.sh,
see bench_io.py for the layout. The first branch given is the baseline every
other branch is compared against.

In speed mode this prints one table per script and per set of settings, so runs
with a different resolution or chunk size are compared separately. In memory
mode it overlays the memory traces of each branch, one figure per script, and
prints the peak usage.
"""

import argparse
import os

import numpy as np

import bench_io


def auto_trim(mem, threshold=100.0):
    """Index of the first sample where memory rises above the initial baseline."""
    baseline = np.median(mem[: max(5, len(mem) // 100)])
    above = mem > baseline + threshold
    return int(np.argmax(above)) if above.any() else 0


def render(headers, rows, markdown=False):
    """Print an aligned text table, or a markdown one."""
    cols = [
        max([len(str(h))] + [len(str(r[i])) for r in rows])
        for i, h in enumerate(headers)
    ]
    align = ["<"] + [">"] * (len(headers) - 1)  # first column is a name
    if markdown:
        cells = (f"{h:{a}{c}}" for h, a, c in zip(headers, align, cols))
        print("| " + " | ".join(cells) + " |")
        print("| " + " | ".join("-" * c for c in cols) + " |")
    else:
        print("  ".join(f"{h:{a}{c}}" for h, a, c in zip(headers, align, cols)))
        print("  ".join("-" * c for c in cols))
    for row in rows:
        cells = (f"{str(v):{a}{c}}" for v, a, c in zip(row, align, cols))
        print(("| " + " | ".join(cells) + " |") if markdown else "  ".join(cells))


def speed_tables(args):
    """One timing table per script and settings."""
    found = {
        b: bench_io.load_branch(args.results_dir, b, args.device, "speed", args.script)
        for b in args.branches
    }
    scripts = sorted({s for runs in found.values() for s in runs})
    if not scripts:
        print(f"no speed results for {args.device} in {args.results_dir}")
        return

    header = ["branch", "commit", "n", "compile", "best", "mean", "worst", "vs base"]
    for script in scripts:
        keys = sorted({k for b in args.branches for k in found[b].get(script, {})})
        for key in keys:
            rows = []
            base = None
            for branch in args.branches:
                run = found[branch].get(script, {}).get(key)
                if run is None:
                    rows.append([branch, "-", "-", "-", "-", "-", "-", "-"])
                    continue
                times = np.asarray(run["times"], dtype=float)
                best = times.min()
                base = base if base is not None else best
                rows.append(
                    [
                        branch,
                        run.get("commit") or "-",
                        run.get("n_repeat", times.size),
                        f"{run['t_compile']:.3f}",
                        f"{best:.4f}",
                        f"{times.mean():.4f}",
                        f"{times.max():.4f}",
                        f"{base / best:.2f}x",
                    ]
                )
            print(f"\n{script}  [{args.device}]\nsettings: {key}")
            render(header, rows, args.markdown)


def memory_plots(args):
    """Overlay every branch, one figure per script and one row per setting."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    found = {
        b: bench_io.load_branch(args.results_dir, b, args.device, "memory", args.script)
        for b in args.branches
    }
    scripts = sorted({s for runs in found.values() for s in runs})
    if not scripts:
        print(f"no memory results for {args.device} in {args.results_dir}")
        return

    header = ["branch", "commit", "peak MB", "vs base"]
    for script in scripts:
        keys = sorted({k for b in args.branches for k in found[b].get(script, {})})
        fig, axes = plt.subplots(
            len(keys), 1, figsize=(20, 6 * len(keys)), squeeze=False
        )
        for ax, key in zip(axes[:, 0], keys):
            rows = []
            base = None
            for branch in args.branches:
                folder = bench_io.branch_dir(args.results_dir, branch)
                path = bench_io.trace_path(folder, script, args.device, key)
                if not os.path.exists(path):
                    rows.append([branch, "-", "-", "-"])
                    continue
                trace = np.load(path)
                t, m = trace["time"], trace["memory"]
                trim = auto_trim(m, args.threshold)
                ax.plot(t[trim:] - t[trim], m[trim:], label=branch)
                peak = m.max()
                base = base if base is not None else peak
                run = found[branch][script][key]
                rows.append(
                    [
                        branch,
                        run.get("commit") or "-",
                        f"{peak:.0f}",
                        f"{peak / base:.2f}x",
                    ]
                )
            ax.set_xlabel("Time (s)", fontsize=14)
            ax.set_ylabel(f"{args.device.upper()} Memory (MB)", fontsize=14)
            ax.set_title(key, fontsize=12)
            ax.grid(True)
            ax.legend()

            print(f"\n{script}  [{args.device}]\nsettings: {key}")
            render(header, rows, args.markdown)

        fig.suptitle(f"{script}  [{args.device}]", fontsize=16)
        fig.tight_layout()
        out = os.path.join(args.results_dir, f"memory-{script}-{args.device}.png")
        fig.savefig(out, dpi=200)
        plt.close(fig)
        print(f"saved plot to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "branches", nargs="+", help="branch names, the first one is the baseline"
    )
    parser.add_argument("--mode", default="speed", choices=["speed", "memory"])
    parser.add_argument("--device", default="gpu", choices=["cpu", "gpu"])
    parser.add_argument("--script", default=None, help="only this script, e.g. 07_*")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--markdown", action="store_true", help="markdown tables")
    parser.add_argument(
        "--threshold",
        type=float,
        default=100.0,
        help="memory rise (MB) over the baseline that marks the start of a run",
    )
    args = parser.parse_args()

    if args.mode == "speed":
        speed_tables(args)
    else:
        memory_plots(args)
