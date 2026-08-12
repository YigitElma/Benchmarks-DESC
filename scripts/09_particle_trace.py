import sys
import os
import timeit

# Enable XLA command buffers (CUDA graphs): the solver while loop body (and
# with WHILE/CONDITIONAL, ideally the whole loop) is captured once and replayed
# as a single graph launch, instead of the CPU dispatching each small kernel
# individually every step. Must be set BEFORE jax is imported.
# WHILE/CONDITIONAL capture needs CUDA >= 12.3 and a recent driver.
# Set to False to benchmark without capture.
ENABLE_CUDA_GRAPHS = True
if ENABLE_CUDA_GRAPHS:
    os.environ["XLA_FLAGS"] = " ".join(
        [
            os.environ.get("XLA_FLAGS", ""),
            # DYNAMIC_SLICE_FUSION covers the SaveAt buffer updates, which
            # otherwise fragment the capture at every save step
            "--xla_gpu_enable_command_buffer="
            "FUSION,CUBLAS,CUBLASLT,CUSTOM_CALL,CONDITIONAL,WHILE,"
            "DYNAMIC_SLICE_FUSION",
            # if many tiny command buffers form, try the default (5) or 8
            # instead of 1: graph replay has its own launch cost, so capturing
            # 2-op segments can be a wash
            "--xla_gpu_graph_min_graph_size=1",
        ]
    ).strip()
# to log what XLA captured into command buffers (and why not), uncomment:
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
# os.environ["TF_CPP_VMODULE"] = "command_buffer_scheduling=3"

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
from desc.particles import _trace_particles
from desc.particles import _precompute_zernike_bases

from bench_io import config_key, save_result

print_backend_info()
print(f"device : {DEVICE}, profile mode : {PROFILE_MODE}, N_REPEAT : {N_REPEAT}")
print(f"save dir : {SAVE_DIR}")

# "poly" evaluates the Zernike radial basis with precomputed polynomial
# coefficients, "jacobi" uses the standard sequential Jacobi recurrence
ZERNIKE_MODE = "poly"
# None -> unbounded plain lax.while_loop: no checkpointed-loop bookkeeping
# inside the solve, the friendliest shape for WHILE command buffer capture.
# Forward only (cannot backprop through the solve). Set an int to restore the
# bounded/checkpointed loop.
MAX_STEPS = None
name = "precise_QH"
eq = get(name)
eq = rescale(eq=eq, L=("a", 1.7044), B=("<B>", 5.86))
if eq.iota is None:
    # single point grids used during tracing cannot compute iota from current
    eq.iota = eq.get_profile("iota").to_powerseries(order=eq.L)

N = 1000
particles = ManualParticleInitializerFlux(
    rho0=jnp.linspace(0.2, 0.7, N),
    theta0=jnp.zeros(N),
    zeta0=jnp.zeros(N),
    xi0=jnp.linspace(-0.9, 0.9, N),
    E=3.5e6,
)
model = VacuumGuidingCenterTrajectory(frame="flux")
ts = jnp.linspace(0, 1e-3, 101)

# Build diffrax objects ONCE and reuse, so that eqx.filter_jit on diffeqsolve
# does not recompile on every call due to fresh closures.
bounds = jnp.array([[0, 1.0], [-jnp.inf, jnp.inf], [-jnp.inf, jnp.inf]])


def terminating_event(t, y, args, **kwargs):
    i = jnp.sqrt(y[0] ** 2 + y[1] ** 2)
    return jnp.logical_or(i < bounds[0, 0], i > bounds[0, 1])


rtol, atol = 1e-4, 1e-4
min_step_size = 1e-8
OPTIONS = {
    "saveat": SaveAt(ts=ts),
    "event": Event(terminating_event),
    "adjoint": RecursiveCheckpointAdjoint(),
    "stepsize_controller": PIDController(
        rtol=rtol, atol=atol, dtmin=min_step_size, pcoeff=0.3, icoeff=0.3, dcoeff=0
    ),
}


y0, model_args = particles.init_particles(model, eq)
# precomputed polynomial coefficient tables for the Zernike radial basis,
# built once outside the solve (this is what trace_particles does internally
# for zernike_mode="poly")
options = (
    {"zernike_bases": _precompute_zernike_bases(eq)} if ZERNIKE_MODE == "poly" else {}
)


@eqx.filter_jit
def _run_jit(y0, model_args, ts, field, options):
    return _trace_particles(
        field=field,
        y0=y0,
        model=model,
        model_args=model_args,
        ts=ts,
        params=field.params_dict,
        max_steps=MAX_STEPS,
        min_step_size=min_step_size,
        stepsize_controller=OPTIONS["stepsize_controller"],
        saveat=OPTIONS["saveat"],
        solver=Tsit5(),
        adjoint=OPTIONS["adjoint"],
        event=OPTIONS["event"],
        options=options,
        chunk_size=None,
        throw=False,
        return_aux=False,
    )


print(
    f"\ntrace_particles for {name}, {N} particles, t_final={ts[-1]:.1e}s, "
    f"zernike_mode={ZERNIKE_MODE}, cuda_graphs={ENABLE_CUDA_GRAPHS}, "
    f"max_steps={MAX_STEPS}"
)
print(f"XLA_FLAGS = {os.environ.get('XLA_FLAGS', '')}")


CONFIG = {
    "name": name,
    "eq_L": eq.L,
    "eq_M": eq.M,
    "eq_N": eq.N,
    "eq_L_grid": eq.L_grid,
    "eq_M_grid": eq.M_grid,
    "eq_N_grid": eq.N_grid,
    "N": N,
    "ZERNIKE_MODE": ZERNIKE_MODE,
    "MAX_STEPS": MAX_STEPS,
    "ENABLE_CUDA_GRAPHS": ENABLE_CUDA_GRAPHS,
    "rtol": rtol,
    "atol": atol,
}
print(f"config : {config_key(CONFIG)}")


def run():
    x, v = _run_jit(y0, model_args, ts, eq, options)
    return jax.block_until_ready((x, v))


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
