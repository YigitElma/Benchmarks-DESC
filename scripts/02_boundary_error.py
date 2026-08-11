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
N_REPEAT = 1 if PROFILE_MODE == "memory" else 5
if len(sys.argv) > 3:
    N_REPEAT = int(sys.argv[3])
SAVE_DIR = sys.argv[4] if len(sys.argv) > 4 else None

if PROFILE_MODE == "memory":
    # per-allocation cudaMalloc, so the sampled VRAM tracks the live usage
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
else:
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.00"

from desc import set_device

set_device(DEVICE)


import desc

from desc.basis import *
from desc.backend import *
from desc.compute import *
from desc.coils import *
from desc.equilibrium import *
from desc.examples import *
from desc.grid import *
from desc.geometry import *
from desc.io import *

from desc.objectives import *
from desc.objectives.objective_funs import *
from desc.objectives.getters import *
from desc.objectives.normalization import compute_scaling_factors
from desc.objectives.utils import *
from desc.optimize._constraint_wrappers import *

from desc.transform import Transform
from desc.plotting import *
from desc.optimize import *
from desc.perturbations import *
from desc.profiles import *
from desc.compat import *
from desc.utils import *
from desc.magnetic_fields import *
from desc.particles import *

from desc.__main__ import main
from desc.vmec_utils import vmec_boundary_subspace
from desc.input_reader import InputReader
from desc.continuation import solve_continuation_automatic
from desc.compute.data_index import register_compute_fun
from desc.optimize.utils import solve_triangular_regularized

from scripts.universal import init_modular
from bench_io import config_key, save_result

print_backend_info()
print(f"device : {DEVICE}, profile mode : {PROFILE_MODE}, N_REPEAT : {N_REPEAT}")
print(f"save dir : {SAVE_DIR}")


n_coils = 10
r_over_a = 2
jac_chunk_size = 160
bs_chunk_size = None

name = "ESTELL"
eq = get(name)

# field = init_modular(eq, 5, 1.5)
field = init_modular(eq, n_coils, r_over_a)
field_grid = LinearGrid(N=20)
field = field.to_FourierXYZ(N=8, grid=field_grid, check_intersection=False)

objective = ObjectiveFunction(BoundaryError(eq, field=field))
constraint = ObjectiveFunction(ForceBalance(eq))
prox = ProximalProjection(
    objective, constraint, eq, solve_options={"solve_during_proximal_build": False}
)
obj = LinearConstraintProjection(
    prox, ObjectiveFunction((FixCurrent(eq), FixPressure(eq), FixPsi(eq)))
)
obj.build()
x = obj.x(eq)
print(
    f"\nBoundaryError.jac_scaled_error for {name}, n_coils={n_coils} "
    f"({field.num_coils} total incl. virtual coils)"
)
print(f"dim_x : {x.size}")
print(f"bs_chunk_size : {bs_chunk_size}, jac_chunk_size : {jac_chunk_size}")


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
    "jac_chunk_size": jac_chunk_size,
    "bs_chunk_size": bs_chunk_size,
}
print(f"config : {config_key(CONFIG)}")


def run():
    return obj.jac_scaled_error(x).block_until_ready()


# first call includes lowering + compilation
t_compile = timeit.timeit(run, number=1)
print(f"compile + first run: {t_compile:7.3f} s")

times = timeit.repeat(run, number=1, repeat=N_REPEAT)
print(
    f"run: best {min(times):7.4f} s, mean {np.mean(times):7.4f} s, "
    f"worst {max(times):7.4f} s (over {N_REPEAT} runs)"
)
save_result(SAVE_DIR, __file__, DEVICE, PROFILE_MODE, CONFIG, t_compile, times)
