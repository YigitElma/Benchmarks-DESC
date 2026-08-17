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
N_REPEAT = 0 if PROFILE_MODE == "memory" else 3
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
from diffrax import *
import equinox as eqx

from desc.__main__ import main
from desc.vmec_utils import vmec_boundary_subspace
from desc.input_reader import InputReader
from desc.continuation import solve_continuation_automatic
from desc.compute.data_index import register_compute_fun
from desc.optimize.utils import solve_triangular_regularized
from desc.magnetic_fields._core import _field_line_integrate

from scripts.universal import init_modular
from bench_io import config_key, save_result

print_backend_info()
print(f"device : {DEVICE}, profile mode : {PROFILE_MODE}, N_REPEAT : {N_REPEAT}")
print(f"save dir : {SAVE_DIR}")


N = 20
num_coils = 10
r_over_a = 2
name = "precise_QH"
res = 8
ntransit = 10

eq = get(name)
# keep the initial values the same
eq = load(f"./inputs/{name}_output.h5")[-1]
eq.change_resolution(
    L=res, M=res, N=res, L_grid=2 * res, M_grid=2 * res, N_grid=2 * res
)
field_grid = LinearGrid(N=20)

# field = init_modular(eq, 5, 1.5)
field = init_modular(eq, num_coils, r_over_a)
field = field.to_FourierXYZ(N=8, grid=field_grid, check_intersection=False)

# r0 = jnp.linspace(10.5, 11.0, N)
r0 = jnp.linspace(1, 1.2, N)
z0 = jnp.zeros(N)
phis = jnp.asarray([0.0, ntransit * 2 * np.pi])

# Build diffrax objects ONCE and reuse.
bounds_R = (0.0, np.inf)
bounds_Z = (-np.inf, np.inf)


def terminating_event(t, y, args, **kwargs):
    R_out = jnp.logical_or(y[0] < bounds_R[0], y[0] > bounds_R[1])
    Z_out = jnp.logical_or(y[2] < bounds_Z[0], y[2] > bounds_Z[1])
    return jnp.logical_or(R_out, Z_out)


rtol, atol = 1e-8, 1e-8
min_step_size = 1e-8
OPTIONS = {
    "saveat": SaveAt(ts=phis),
    "event": Event(terminating_event),
    "adjoint": RecursiveCheckpointAdjoint(),
    "stepsize_controller": PIDController(rtol=rtol, atol=atol, dtmin=min_step_size),
}


@eqx.filter_jit
def _run_jit(r0, z0, phis, field):
    return _field_line_integrate(
        r0=r0,
        z0=z0,
        phis=phis,
        field=field,
        params=field.params_dict,
        source_grid=field_grid,
        solver=Tsit5(),
        max_steps=1000000,
        min_step_size=min_step_size,
        saveat=OPTIONS["saveat"],
        stepsize_controller=OPTIONS["stepsize_controller"],
        event=OPTIONS["event"],
        adjoint=OPTIONS["adjoint"],
        chunk_size=None,
        bs_chunk_size=None,
        options={},
        return_aux=False,
    )


print(f"\nfield_line_integrate for {name}, {N} field lines, phis={phis.tolist()}")
print(f"num_coils : {num_coils} ({field.num_coils} total incl. virtual coils)")


CONFIG = {
    "name": name,
    "eq_L": eq.L,
    "eq_M": eq.M,
    "eq_N": eq.N,
    "eq_L_grid": eq.L_grid,
    "eq_M_grid": eq.M_grid,
    "eq_N_grid": eq.N_grid,
    "N": N,
    "num_coils": num_coils,
    "r_over_a": r_over_a,
    "rtol": rtol,
    "atol": atol,
    "ntransit": ntransit,
}
print(f"config : {config_key(CONFIG)}")


def run():
    r, z = _run_jit(r0, z0, phis, field)
    return jax.block_until_ready((r, z))


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
