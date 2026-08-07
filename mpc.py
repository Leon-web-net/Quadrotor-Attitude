"""
ES4G4 Assignment 1 (Python rebuild) — linear MPC toolkit.

Everything the notebooks share once the plant model (`quadrotor.py`) is fixed:

  * `Plant`         — (Ad, Bd, Ts), the three things that always travel together
  * `discretise`    — exact ZOH discretisation via one Van Loan matrix exponential
  * `weights_for`   — (Q, R, RD, P) synthesised from a target closed-loop bandwidth
  * `MPC`           — the receding-horizon controller of Part B-1
  * `simulate`      — closed loop on the exact ZOH plant, with a disturbance hook
  * `step_metrics`  — rise / overshoot / settling / saturation for a step transient
  * `impulse_of`    — momentum-preserving rectangular torque pulse
  * `show`          — fixed-width table printer for dicts of metrics

The derivations behind these choices live in `part_b.ipynb`; section numbers in the
docstrings refer to it.

State  x = [phi, theta, psi, wx, wy, wz]^T   (n = 6)
Input  u = [tau_x, tau_y, tau_z]^T           (m = 3)
"""

import time
from typing import NamedTuple

import cvxpy as cp
import numpy as np
import scipy.linalg as sla

from quadrotor import DEG, PARAMS, TAU_MAX

P_H, C_H = 25, 10          # prediction, control horizon (§5)


class Plant(NamedTuple):
    """A discrete-time plant: x[k+1] = Ad x[k] + Bd u[k], sampled every Ts seconds."""

    Ad: np.ndarray
    Bd: np.ndarray
    Ts: float


def discretise(Ac, Bc, Ts):
    """Exact ZOH discretisation from one Van Loan matrix exponential (§3).

    MPC holds each input constant across a sample, so this is not an approximation
    of the linear plant but its exact solution at the sample instants. Reused for
    the perturbed plants of B-4, so uncertain models are discretised identically.
    """
    nx, nu = Bc.shape
    Maug = np.zeros((nx + nu, nx + nu))
    Maug[:nx, :nx], Maug[:nx, nx:] = Ac * Ts, Bc * Ts
    Eaug = sla.expm(Maug)
    return Plant(Eaug[:nx, :nx], Eaug[:nx, nx:], Ts)


def weights_for(wn, plant, zeta=1.0, p=PARAMS, q1=None):
    """(Q, R, RD, P) for a target bandwidth — the §6/§7 formulas, wn as the knob.

    Because r = rho / tau_max^2 = q1 / (Ix^2 wn^4), the torque normaliser cancels and
    the bandwidth really is the only quantity being set. q2 is chosen per axis so the
    damping ratio holds on yaw as well as roll/pitch (Iz = 2 Ix).

    P is the DARE solution — computed, never tuned (§4b).
    """
    Ix, Iz = p["Ix"], p["Iz"]
    q1 = 1 / (10 * DEG) ** 2 if q1 is None else q1        # Bryson-style normaliser
    r = q1 / (Ix**2 * wn**4)
    q2 = lambda I_: 4 * I_ * np.sqrt(q1 * r) * (zeta**2 - 0.5)

    Q = np.diag([q1, q1, q1, q2(Ix), q2(Ix), q2(Iz)])
    R = np.diag([r] * 3)
    RD = 1.0 * R                # slew weight: light; empirical, verified in B-3
    return Q, R, RD, sla.solve_discrete_are(plant.Ad, plant.Bd, Q, R)


_chol = lambda W: np.linalg.cholesky(W).T             # W = S'S  ->  v'Wv = ||Sv||^2


class MPC:
    """Receding-horizon controller for the linear attitude model (§4).

    Posed in error coordinates e = x - r: a constant attitude needs no torque, so r
    is an exact fixed point of (Ad, Bd) at u = 0 and the error obeys the same
    dynamics. The reference therefore enters only through e0 (§8a).

    budget="l2"  -> ||u||_2 <= tau_max, a shared actuator budget (SOCP, Clarabel)
    budget="box" -> |u_i| <= tau_max per axis  (QP, OSQP) — the §11 fallback
    """

    def __init__(self, plant, Q, R, RD, P, p=P_H, c=C_H, tau_max=TAU_MAX,
                 budget="l2", solver=None):
        Ad, Bd = plant.Ad, plant.Bd
        n, m = Bd.shape
        self.solver = solver or (cp.CLARABEL if budget == "l2" else cp.OSQP)
        E = cp.Variable((n, p + 1))                   # predicted error trajectory
        U = cp.Variable((m, c))                       # free moves
        self.e0 = cp.Parameter(n)                     # x - r
        self.up = cp.Parameter((m, 1))                # torque applied last sample

        Ub = U[:, np.minimum(np.arange(p), c - 1)]    # move blocking: hold u_{c-1}
        # slew: dU_k = u_k - u_{k-1}, with u_{-1} the torque applied last sample
        dU = U - (cp.hstack([self.up, U[:, :c - 1]]) if c > 1 else self.up)
        Qs, Rs, RDs, Ps = map(_chol, (Q, R, RD, P))

        cost = (cp.sum_squares(Qs @ E[:, :p]) + cp.sum_squares(Ps @ E[:, p])
                + cp.sum_squares(Rs @ Ub) + cp.sum_squares(RDs @ dU))
        cons = [E[:, 0] == self.e0,
                E[:, 1:] == Ad @ E[:, :p] + Bd @ Ub,
                cp.norm(U, 2, axis=0) <= tau_max if budget == "l2"
                else cp.abs(U) <= tau_max]
        self.U, self.prob = U, cp.Problem(cp.Minimize(cost), cons)

    def step(self, x, ref, u_prev):
        """One receding-horizon solve; returns u_0* only."""
        self.e0.value = x - ref
        self.up.value = np.reshape(u_prev, (-1, 1))
        self.prob.solve(solver=self.solver, warm_start=True)
        if self.prob.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"solver status: {self.prob.status}")
        return self.U.value[:, 0]


def simulate(ctrl, plant, x0, ref_of_t, T_end, dist_of_t=None):
    """Closed loop on the exact ZOH plant. Returns t, X (N+1,n), U (N,m), solve_ms (N,).

    `plant` may differ from the one the controller was built on — that is how B-4
    drives the nominal controller with an uncertain plant.

    dist_of_t(t) -> torque added at the plant input only: the controller never sees
    it, so rejection is pure feedback (the mechanism B-3 needs).
    """
    Ad, Bd, Ts = plant
    N = int(round(T_end / Ts))
    x, u_prev = np.asarray(x0, float).copy(), np.zeros(Bd.shape[1])
    X, U, solve_ms = np.zeros((N + 1, x.size)), np.zeros((N, u_prev.size)), np.zeros(N)
    X[0] = x
    for k in range(N):
        tk = k * Ts
        t0 = time.perf_counter()
        u = ctrl.step(x, ref_of_t(tk), u_prev)
        solve_ms[k] = (time.perf_counter() - t0) * 1e3
        x = Ad @ x + Bd @ (u + (dist_of_t(tk) if dist_of_t else 0.0))
        u_prev, X[k + 1], U[k] = u, x, u
    return np.arange(N + 1) * Ts, X, U, solve_ms


def step_metrics(X, U, Ts, t0, target, axis, t1=None, band=0.02, tau_max=TAU_MAX):
    """Metrics for the step transient starting at t0 on one axis, sign-robust.

    Progress is measured as a fraction of the signed step, so the +10 deg roll and
    the -10 deg pitch targets are handled by the same code.
    """
    k0, k1 = int(round(t0 / Ts)), None if t1 is None else int(round(t1 / Ts))
    y, u = X[k0:k1, axis], U[k0:k1]
    frac = (y - y[0]) / (target - y[0])           # 0 -> 1 progress, sign-independent
    outside = np.abs(frac - 1) > band             # last exit from the +/-2% band
    un = np.linalg.norm(u, axis=1)
    sat = un > 0.999 * tau_max
    return dict(
        rise_ms=(np.argmax(frac >= 0.9) - np.argmax(frac >= 0.1)) * Ts * 1e3,
        overshoot_pc=max(frac.max() - 1.0, 0.0) * 100,
        settle_ms=((np.flatnonzero(outside)[-1] + 1) * Ts * 1e3) if outside.any() else 0.0,
        sse_deg=(y[-1] - target) / DEG,
        peak_u=float(un.max()),
        sat_ms=float(sat.sum()) * Ts * 1e3,
        sat_episodes=int(np.diff(sat.astype(int)).clip(0).sum() + sat[0]),
    )


def impulse_of(amp, t0, Ts, width):
    """A rectangular torque pulse of amplitude `amp` and duration `width` from t0,
    returned as dist_of_t(tk) giving its *mean over the sample* [tk, tk + Ts).

    The B-3 pulse is 5 ms against Ts = 2 ms, i.e. 2.5 samples, so it cannot sit on
    the sample grid. Rounding to 2 or 3 samples would change the delivered angular
    momentum by +/-20% and invalidate every prediction in §2 and §7; averaging over
    the overlap makes the ZOH plant absorb exactly amp * width.
    """
    return lambda tk: amp * max(0.0, min(tk + Ts, t0 + width) - max(tk, t0)) / Ts


def show(rows, label="transient"):
    """Print a dict of {row name: {metric: value}} as a fixed-width table."""
    keys = list(next(iter(rows.values())))
    print(f"{label:<24}" + "".join(f"{k:>14}" for k in keys))
    for name, d in rows.items():
        print(f"{name:<24}" + "".join(f"{d[k]:>14.4g}" for k in keys))
