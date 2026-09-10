"""Isolated timing of the trust region subproblem, in the setting of 01_eq_solve.

Same problem as 01_eq_solve (precise_QA force balance, fixed boundary constraints,
LinearConstraintProjection), so the Jacobian handed to the trust region step has
the shape, scaling and conditioning of the real solve rather than of a random
matrix. Everything up to the subproblem is set up exactly as lsqtr does it.

Lowering, compilation and run time are reported separately, because the first
lsqtr iteration pays all three and compilation dominates it. HLO instruction
counts are printed alongside, since that is what compile time tracks.

Set TR_RES for the resolution (default 16, matching 01_eq_solve) and TR_RTOL
for the subproblem tolerance (default 0.01, the lsqtr default).
"""

import sys
import os
import timeit
from time import perf_counter

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

from desc.optimize._constraint_wrappers import LinearConstraintProjection
from desc.optimize.bound_utils import cl_scaling_vector
from desc.optimize.tr_subproblems import trust_region_step_exact_qr
from desc.optimize.utils import compute_jac_scale, solve_triangular_regularized

from bench_io import config_key, save_result

if Version(desc.__version__) >= Version("0.14.0"):
    print_backend_info()
print(f"device : {DEVICE}, profile mode : {PROFILE_MODE}, N_REPEAT : {N_REPEAT}")
print(f"save dir : {SAVE_DIR}")

res = int(os.environ.get("TR_RES", 16))
rtol = float(os.environ.get("TR_RTOL", 0.01))
jac_chunk_size = 500
deriv_mode = "batched"
name = "precise_QA"

N = res
# keep the initial values the same
eq = load(f"./inputs/{name}_output.h5")[-1]
eq.change_resolution(L=N, M=N, N=N, L_grid=2 * N, M_grid=2 * N, N_grid=2 * N)
eq.set_initial_guess()
obj = ObjectiveFunction(
    ForceBalance(eq), jac_chunk_size=jac_chunk_size, deriv_mode=deriv_mode
)
con = get_fixed_boundary_constraints(eq)
con = ObjectiveFunction(maybe_add_self_consistency(eq, con))
lc = LinearConstraintProjection(obj, con)
lc.build()

eq.resolution_summary()

# --- everything lsqtr does before the first trust region step -----------------
# eq.solve passes no bounds, so `bounded` is False and J_a/f_a are just J_h/f
x = lc.x(eq)
f = lc.compute_scaled_error(x)
J = lc.jac_scaled_error(x).block_until_ready()
g = jnp.dot(f, J)
scale, scale_inv = compute_jac_scale(J)  # x_scale="auto"
v, dv = cl_scaling_vector(x, g, -jnp.inf * jnp.ones_like(x), jnp.inf * jnp.ones_like(x))
v = jnp.where(dv != 0, v * scale_inv, v)
d = v**0.5 * scale
J_a = J * d
f_a = f
# "scipy" is the default initial_trust_radius
trust_radius = jnp.linalg.norm(x * scale_inv / v**0.5)
alpha = jnp.float64(0.0)

dr = jnp.abs(jnp.diag(jnp.linalg.qr(J_a, mode="r")))
print(f"\nJ_a {J_a.shape}, |diag R| max/min {dr.max() / dr.min():.2e} "
      f"(rough conditioning), trust_radius {trust_radius:.3e}")

CONFIG = {
    "name": name,
    "eq_L": eq.L,
    "eq_M": eq.M,
    "eq_N": eq.N,
    "eq_L_grid": eq.L_grid,
    "eq_M_grid": eq.M_grid,
    "eq_N_grid": eq.N_grid,
    "jac_chunk_size": jac_chunk_size,
    "deriv_mode": deriv_mode,
    "rtol": rtol,
    "m": J_a.shape[0],
    "n": J_a.shape[1],
}
print(f"config : {config_key(CONFIG)}")


def breakdown(label, fn, *args):
    """Lower, compile and run once, timing each stage separately."""
    t0 = perf_counter()
    lowered = fn.lower(*args)
    t1 = perf_counter()
    compiled = lowered.compile()
    t2 = perf_counter()
    out = jax.block_until_ready(compiled(*args))
    t3 = perf_counter()
    n_hlo = compiled.as_text().count("\n")
    print(
        f"{label:26s} lower {t1 - t0:7.3f} s  compile {t2 - t1:7.3f} s  "
        f"run {t3 - t2:7.3f} s  ({n_hlo} hlo lines)"
    )
    return out, compiled, (t1 - t0, t2 - t1, t3 - t2)


print("")
# the big m x n factorization, which lsqtr does once per Jacobian
big_qr = jit(lambda a, c: qr_multiply(a, c, mode="right"))
(Qt_fa, R), _, t_qr = breakdown("qr_multiply(J_a, f_a)", big_qr, J_a, f_a)
p_newton = solve_triangular_regularized(R, -Qt_fa)

# the subproblem itself, called once per inner trust region iteration
tr_args = (p_newton, Qt_fa, R, trust_radius, alpha, rtol)
_, tr, t_tr = breakdown(
    "trust_region_step_exact_qr", trust_region_step_exact_qr, *tr_args
)

t_compile = sum(t_tr)
print(f"\ntrust region step, compile + first run: {t_compile:7.3f} s")

# The first call returns p_newton untouched whenever it fits inside the (large)
# initial radius, so it never enters the alpha loop and says nothing about the
# cost of the factorizations. Walk the radii the solve actually visits instead:
# update_tr_radius sets trust_radius = 0.25*step_norm on a rejected step, and
# lsqtr rescales alpha by tr_old/trust_radius.
pn_norm = float(jnp.linalg.norm(p_newton))
print(f"\n||p_newton|| {pn_norm:.4e} vs trust_radius {trust_radius:.4e} -> first "
      f"call {'returns p_newton' if pn_norm <= trust_radius else 'enters the loop'}")
print(f"{'call':>5s} {'delta/||p_N||':>14s} {'alpha_in':>11s} {'time (s)':>10s}")
d_tr, d_alpha = float(trust_radius), 0.0
for call in range(6):
    step, hit, a_out = tr(p_newton, Qt_fa, R, d_tr, d_alpha, rtol)
    jax.block_until_ready(step)
    t0 = perf_counter()
    for _ in range(3):
        jax.block_until_ready(tr(p_newton, Qt_fa, R, d_tr, d_alpha, rtol)[0])
    dt = (perf_counter() - t0) / 3
    print(f"{call + 1:>5d} {d_tr / pn_norm:>14.4e} {d_alpha:>11.3e} {dt:>10.4f}")
    step_norm = min(pn_norm, d_tr)  # rejected step: p_newton, or on the boundary
    tr_old, d_tr = d_tr, 0.25 * step_norm
    d_alpha = float(a_out) * tr_old / d_tr


def run():
    return jax.block_until_ready(tr(*tr_args))


if N_REPEAT > 0:
    times = timeit.repeat(run, number=1, repeat=N_REPEAT)
    print(
        f"run: best {min(times):7.4f} s, mean {np.mean(times):7.4f} s, "
        f"worst {max(times):7.4f} s (over {N_REPEAT} runs)"
    )
else:
    times = [t_compile]
save_result(SAVE_DIR, __file__, DEVICE, PROFILE_MODE, CONFIG, t_compile, times)
