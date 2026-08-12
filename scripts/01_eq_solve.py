import sys
import os
import timeit

_BENCH_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(_BENCH_DIR))  # DESC repo root
sys.path.insert(0, _BENCH_DIR)  # bench_io, scripts.universal

# --- command line: <cpu|gpu> <memory|speed> [n_repeat] [save_dir] ---
DEVICE = (sys.argv[1] if len(sys.argv) > 1 else "gpu").lower()
PROFILE_MODE = (sys.argv[2] if len(sys.argv) > 2 else "speed").lower()
assert DEVICE in ["cpu", "gpu"], f"unknown device '{DEVICE}'"
assert PROFILE_MODE in ["memory", "speed"], f"unknown profile mode '{PROFILE_MODE}'"
N_REPEAT = 0 if PROFILE_MODE == "memory" else 5
if len(sys.argv) > 3:
    N_REPEAT = int(sys.argv[3])
    if PROFILE_MODE == "memory":
        N_REPEAT = 0 if N_REPEAT == 1 else N_REPEAT
SAVE_DIR = sys.argv[4] if len(sys.argv) > 4 else None

if PROFILE_MODE == "memory":
    # per-allocation cudaMalloc, so the sampled VRAM tracks the live usage
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
else:
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.00"

from desc import set_device

set_device(DEVICE)

from packaging.version import Version
import desc

from desc.basis import *
from desc.backend import *

from desc.objectives import *
from desc.optimize import *
from desc.io import load

from bench_io import config_key, save_result

if Version(desc.__version__) >= Version("0.14.0"):
    print_backend_info()
print(f"device : {DEVICE}, profile mode : {PROFILE_MODE}, N_REPEAT : {N_REPEAT}")
print(f"save dir : {SAVE_DIR}")

res = 14
maxiter = 5
jac_chunk_size = 500
deriv_mode = "batched"
tr_method = "qr"
name = "precise_QA"

N = res
# keep the initial values the same
eq = load(f"./inputs/{name}_output.h5")[-1]
eq.change_resolution(L=N, M=N, N=N, L_grid=2 * N, M_grid=2 * N, N_grid=2 * N)
eq.set_initial_guess()
if Version(desc.__version__) >= Version("0.12.3"):
    obj = ObjectiveFunction(
        ForceBalance(eq), jac_chunk_size=jac_chunk_size, deriv_mode=deriv_mode
    )
else:
    obj = ObjectiveFunction(ForceBalance(eq), deriv_mode=deriv_mode)
obj.build()

eq.resolution_summary()
print(f"\neq.solve for {name}, maxiter={maxiter}")
if Version(desc.__version__) >= Version("0.12.3"):
    print(f"deriv mode : {obj._deriv_mode}, jac_chunk_size : {obj._jac_chunk_size}")
else:
    print(f"deriv mode : {obj._deriv_mode}, jac_chunk_size : doesn't exist")

CONFIG = {
    "name": name,
    "eq_L": eq.L,
    "eq_M": eq.M,
    "eq_N": eq.N,
    "eq_L_grid": eq.L_grid,
    "eq_M_grid": eq.M_grid,
    "eq_N_grid": eq.N_grid,
    "maxiter": maxiter,
    "jac_chunk_size": jac_chunk_size,
    "deriv_mode": deriv_mode,
    "tr_method": tr_method,
}
print(f"config : {config_key(CONFIG)}")


def run():
    # copy=True leaves eq untouched, so every repeat starts from the same state
    return eq.solve(
        objective=obj,
        constraints=None,
        optimizer="lsq-exact",
        ftol=0,
        xtol=0,
        gtol=0,
        maxiter=maxiter,
        verbose=3,
        copy=True,
        options={"tr_method": tr_method},
    )


# first call includes lowering + compilation
t_compile = timeit.timeit(run, number=1)
print(f"compile + first run: {t_compile:7.3f} s")

if N_REPEAT > 0:
    times = timeit.repeat(run, number=1, repeat=N_REPEAT)
    print(
        f"run: best {min(times):7.4f} s, mean {np.mean(times):7.4f} s, "
        f"worst {max(times):7.4f} s (over {N_REPEAT} runs)"
    )
else:
    times = [t_compile]
save_result(SAVE_DIR, __file__, DEVICE, PROFILE_MODE, CONFIG, t_compile, times)
