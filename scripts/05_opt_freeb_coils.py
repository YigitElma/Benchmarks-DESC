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

from desc.backend import *
from desc.grid import *
from desc.io import *

from desc.objectives import *
from desc.optimize._constraint_wrappers import *

from desc.optimize import *
from desc.magnetic_fields import *

from scripts.universal import init_modular
from bench_io import config_key, save_result

if Version(desc.__version__) >= Version("0.14.0"):
    print_backend_info()
print(f"device : {DEVICE}, profile mode : {PROFILE_MODE}, N_REPEAT : {N_REPEAT}")
print(f"save dir : {SAVE_DIR}")


n_coils = 10
r_over_a = 2
jac_chunk_size = 60
force_chunk_size = None
bs_chunk_size = None
maxiter = 3
B_plasma_chunk_size = None
res = 10
name = "precise_QA"

eq = load(f"./inputs/{name}_output.h5")[-1]
eq.change_resolution(res, res, res, res * 2, res * 2, res * 2)
field = init_modular(eq, n_coils, r_over_a)
field_grid = LinearGrid(N=50)
field = field.to_FourierXYZ(N=8, grid=field_grid, check_intersection=False)

obj = ObjectiveFunction(
    BoundaryError(eq, field=field, B_plasma_chunk_size=B_plasma_chunk_size),
    jac_chunk_size=jac_chunk_size,
)
cons = (
    ForceBalance(eq, jac_chunk_size=force_chunk_size),
    FixPsi(eq),
    FixPressure(eq),
    FixCurrent(eq),
)
print(f"bs_chunk_size : {bs_chunk_size}, jac_chunk_size : {jac_chunk_size}")
print(f"B_plasma_chunk_size : {B_plasma_chunk_size}")
opt = Optimizer("proximal-lsq-exact")

CONFIG = {
    "name": name,
    "eq_L": eq.L,
    "eq_M": eq.M,
    "eq_N": eq.N,
    "eq_L_grid": eq.L_grid,
    "eq_M_grid": eq.M_grid,
    "eq_N_grid": eq.N_grid,
    "n_coils": n_coils,
    "r_over_a": r_over_a,
    "maxiter": maxiter,
    "jac_chunk_size": jac_chunk_size,
    "bs_chunk_size": bs_chunk_size,
    "B_plasma_chunk_size": B_plasma_chunk_size,
    "force_chunk_size": force_chunk_size,
}
print(f"config : {config_key(CONFIG)}")

if Version(desc.__version__) >= Version("0.16.0"):
    options = {"solve_options": {"solve_during_proximal_build": False}}
else:
    options = {}


def run():
    return opt.optimize(
        (eq, field),
        objective=obj,
        constraints=cons,
        ftol=0,
        xtol=0,
        gtol=0,
        maxiter=maxiter,
        verbose=3,
        options=options,
        copy=True,
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
