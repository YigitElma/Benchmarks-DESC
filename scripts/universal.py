"""Helpers that must behave the same on every branch being benchmarked.

DESC functions that do not exist in older versions, or whose defaults changed,
are reimplemented here so a benchmark measures the same problem no matter which
commit is checked out. Import from here instead of from desc.
"""

import numpy as np
from scipy.constants import mu_0

from desc.coils import CoilSet, FourierPlanarCoil
from desc.grid import LinearGrid
from desc.utils import errorif


def init_modular(eq, num_coils, r_over_a=2.0):
    """Planar circular coils on the axis. Drop in for initialize_modular_coils."""
    extent = 2 * np.pi / (eq.NFP * (eq.sym + 1))
    zeta = np.linspace(0, extent, num_coils, endpoint=False) + extent / (2 * num_coils)
    grid = LinearGrid(rho=[0.0], M=0, zeta=zeta, NFP=eq.NFP)

    minor_radius = eq.compute("a")["a"]
    G = eq.compute("G", grid=LinearGrid(rho=1.0))["G"]
    data = eq.axis.compute(["x", "x_s"], grid=grid, basis="rpz")

    centers = data["x"]  # center coils on axis position
    normals = data["x_s"]  # make normal to coil align with tangent along axis

    unique_coils = []
    for k in range(num_coils):
        coil = FourierPlanarCoil(
            current=2 * np.pi * G / (mu_0 * eq.NFP * num_coils * (eq.sym + 1)),
            center=centers[k, :],
            normal=normals[k, :],
            r_n=minor_radius * r_over_a,
            basis="rpz",
        )
        unique_coils.append(coil)
    # older versions always check intersections, which is slow and not the point
    return CoilSet(unique_coils, NFP=eq.NFP, sym=eq.sym, check_intersection=False)


def init_saddle(eq, num_coils, r_over_a=0.5, offset=2.0, position="outer"):
    """Planar circular coils around the plasma. Drop in for initialize_saddle_coils."""
    errorif(
        position not in {"outer", "inner", "top", "bottom"},
        ValueError,
        f"position must be one of 'outer', 'inner'', 'top', 'bottom', got {position}",
    )
    extent = 2 * np.pi / (eq.NFP * (eq.sym + 1))
    zeta = np.linspace(0, extent, num_coils, endpoint=False) + extent / (2 * num_coils)
    grid = LinearGrid(rho=[0.0], M=0, zeta=zeta, NFP=eq.NFP)

    minor_radius = eq.compute("a")["a"]
    data = eq.axis.compute(["x", "x_s"], grid=grid, basis="rpz")

    centers = data["x"]  # center coils on axis position
    normals = data["x_s"]  # make normal to coil align with tangent along axis

    offset_vecs = {
        "outer": np.array([1, 0, 0]),
        "inner": np.array([-1, 0, 0]),
        "top": np.array([0, 0, 1]),
        "bottom": np.array([0, 0, -1]),
    }
    normal_vecs = {
        "outer": np.array([0, 0, -1]),
        "inner": np.array([0, 0, 1]),
        "top": np.array([1, 0, 0]),
        "bottom": np.array([-1, 0, 0]),
    }

    windowpane_coils = []
    for k in range(num_coils):
        coil = FourierPlanarCoil(
            current=0.0,
            center=centers[k, :] + offset_vecs[position] * offset * minor_radius,
            normal=np.cross(normals[k, :], normal_vecs[position]),
            r_n=minor_radius * r_over_a,
            basis="rpz",
        )
        windowpane_coils.append(coil)

    # older versions always check intersections, which is slow and not the point
    return CoilSet(
        windowpane_coils, NFP=int(eq.NFP), sym=eq.sym, check_intersection=False
    )
