"""
ES4G4 Assignment 1 (Python rebuild) — shared quadrotor model.

Single source of truth for:
  * Table 1 nominal parameters
  * the nonlinear state equations f(x, u)   (numeric, for simulation)
  * the linearised (A, B) about hover       (numeric, for the linear MPC)

State vector  x = [phi, theta, psi, wx, wy, wz]^T   (n = 6)
Input vector  u = [tau_x, tau_y, tau_z]^T           (m = 3)
Angles in rad, rates in rad/s, torques in N·m.
"""

import numpy as np

# ----------------------------------------------------------------------
# Table 1 nominal parameters
# ----------------------------------------------------------------------
PARAMS = dict(
    Ix=6e-5, Iy=6e-5, Iz=1.2e-4,   # kg·m^2
    dx=1e-5, dy=1e-5, dz=2e-5,     # N·m·s
)

TAU_MAX = 0.25                     # N·m, rotor torque limit

# Scalar aliases, so notebooks can import the nominal values by name.
Ix, Iy, Iz = PARAMS["Ix"], PARAMS["Iy"], PARAMS["Iz"]
dx, dy, dz = PARAMS["dx"], PARAMS["dy"], PARAMS["dz"]

DEG = np.pi / 180.0                # rad per degree


N_STATES = 6
N_INPUTS = 3


def f_nonlinear(x, u, p=PARAMS):
    """Nonlinear state equations xdot = f(x, u).  ('func1' of Part C.)

    Vectorised over nothing: x is shape (6,), u is shape (3,).
    Returns shape (6,).
    """
    phi, theta, _psi, wx, wy, wz = x
    tx, ty, tz = u
    Ix, Iy, Iz = p["Ix"], p["Iy"], p["Iz"]
    dx, dy, dz = p["dx"], p["dy"], p["dz"]

    s, c, t = np.sin(phi), np.cos(phi), np.tan(theta)
    sec = 1.0 / np.cos(theta)

    return np.array([
        wx + s * t * wy + c * t * wz,
        c * wy - s * wz,
        s * sec * wy + c * sec * wz,
        (tx + (Iy - Iz) * wy * wz - dx * wx) / Ix,
        (ty + (Iz - Ix) * wz * wx - dy * wy) / Iy,
        (tz + (Ix - Iy) * wx * wy - dz * wz) / Iz,
    ])


def euler_step(x, u, dt, p=PARAMS):
    """One forward-Euler step of the nonlinear dynamics. ('func2' of Part C.)"""
    return x + dt * f_nonlinear(x, u, p)


def linearised_matrices(p=PARAMS):
    """(A, B) of the Jacobian linearisation about hover (x*, u*) = (0, 0).

    Derived symbolically in part_a.ipynb; hard-coded here for speed.
    """
    Ix, Iy, Iz = p["Ix"], p["Iy"], p["Iz"]
    dx, dy, dz = p["dx"], p["dy"], p["dz"]

    A = np.zeros((6, 6))
    A[0:3, 3:6] = np.eye(3)
    A[3, 3] = -dx / Ix
    A[4, 4] = -dy / Iy
    A[5, 5] = -dz / Iz

    B = np.zeros((6, 3))
    B[3, 0] = 1.0 / Ix
    B[4, 1] = 1.0 / Iy
    B[5, 2] = 1.0 / Iz
    return A, B