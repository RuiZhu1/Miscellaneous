#!/usr/bin/env python3
"""Majorana-Coulomb-Island: parity-aligned core model v11.

Main correction relative to v10
--------------------------------
For a single Majorana pair, with
    d = (gamma_L + i gamma_R)/2,
    n_d = d^dagger d,
    Z = 1 - 2 n_d,
we have
    i gamma_L gamma_R = 2 n_d - 1 = -Z.

Thus a conventional positive BdG splitting E_M enters the two-level
Hamiltonian longitudinally, not transversely:
    H_M = (i E_M/2) gamma_L gamma_R = -(E_M/2) Z.

The effective logical model is therefore
    H_log = [h_z(n_g) - E_M/2] Z
            + (Omega/2) [cos(phi) X + sin(phi) Y],
for the standard Z = 1 - 2 n_d convention.
The overall sign of the E_M term is a convention if Z is redefined, so the
code exposes `majorana_z_sign` explicitly.

The central engineering statement is no longer a transverse-coupling law in
eta_M. Instead:

    same-pair Majorana splitting is a longitudinal detuning;
    exact knowledge of E_M lets charge control compensate it;
    residual gate error is therefore controlled by calibration detuning,
    not by E_M itself, within the valid two-level/control range.

The external equatorial drive supplies a non-commuting control axis. This is
necessary for targets outside the Z-rotation orbit (e.g. the H-type magic
state), but it is not correct to say that a continuously tunable Z Hamiltonian
can only generate Clifford gates. The Clifford-only limitation applies to the
standard topologically protected braiding gate set; deliberate dynamical
couplings can implement unprotected phase gates.

The BdG layer remains a physical input generator. It does not claim a full
microscopic projection from a four-Majorana island Hamiltonian.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
from scipy.optimize import least_squares
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------


def pauli() -> Dict[str, np.ndarray]:
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    return {"I": I, "X": X, "Y": Y, "Z": Z}


def dagger(a: np.ndarray) -> np.ndarray:
    return a.conj().T


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    if trials <= 0:
        return 0.0, 1.0
    p = successes / trials
    denom = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / denom
    half = z * math.sqrt((p * (1.0 - p) / trials) + z * z / (4.0 * trials * trials)) / denom
    return max(0.0, center - half), min(1.0, center + half)


# -----------------------------------------------------------------------------
# 0. Majorana algebra / encoding validation
# -----------------------------------------------------------------------------


def verify_majorana_pair_algebra() -> Dict[str, float]:
    """Numerically verify the sign convention implied by d=(gamma_L+i gamma_R)/2."""
    d = np.array([[0, 1], [0, 0]], dtype=complex)
    dd = dagger(d)
    gL = d + dd
    gR = -1j * (d - dd)
    n = dd @ d
    I = np.eye(2, dtype=complex)
    Z = I - 2.0 * n
    parity = 1j * gL @ gR

    return {
        "CAR_error": float(la.norm(d @ dd + dd @ d - I)),
        "gamma_L_hermiticity_error": float(la.norm(gL - dagger(gL))),
        "gamma_R_hermiticity_error": float(la.norm(gR - dagger(gR))),
        "gamma_L_square_error": float(la.norm(gL @ gL - I)),
        "gamma_R_square_error": float(la.norm(gR @ gR - I)),
        "parity_minus_Z_error": float(la.norm(parity + Z)),
        "parity_minus_2n_minus_1_error": float(la.norm(parity - (2.0 * n - I))),
    }


# -----------------------------------------------------------------------------
# 1. BdG layer
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class BdGParams:
    L: int = 80
    alpha: float = 0.15
    Delta: float = 1.0
    Ez: float = 2.5
    t0: float = 1.0
    mu: float = 0.5


class BdGWire:
    def __init__(self, p: BdGParams):
        self.p = p
        P = pauli()
        self.I2, self.sx, self.sy = P["I"], P["X"], P["Y"]
        self.tx, self.ty, self.tz = P["X"], P["Y"], P["Z"]
        self.U_C = np.kron(self.ty, self.sy)
        self._bulk_gap_cache: Optional[float] = None

    def bloch_H(self, k: float) -> np.ndarray:
        p = self.p
        xi = 2 * p.t0 - 2 * p.t0 * np.cos(k) - p.mu
        return (
            xi * np.kron(self.tz, self.I2)
            + p.alpha * np.sin(k) * np.kron(self.tz, self.sy)
            + p.Ez * np.kron(self.I2, self.sx)
            + p.Delta * np.kron(self.tx, self.I2)
        )

    def bulk_gap(self) -> float:
        if self._bulk_gap_cache is None:
            ks = np.linspace(-np.pi, np.pi, 401)
            self._bulk_gap_cache = float(
                min(np.min(np.abs(np.linalg.eigvalsh(self.bloch_H(k)))) for k in ks)
            )
        return self._bulk_gap_cache

    @staticmethod
    def pf4(A: np.ndarray) -> complex:
        return A[0, 1] * A[2, 3] - A[0, 2] * A[1, 3] + A[0, 3] * A[1, 2]

    def z2_invariant(self) -> int:
        B0 = self.bloch_H(0.0) @ self.U_C
        Bpi = self.bloch_H(np.pi) @ self.U_C
        prod = complex(self.pf4(B0) * self.pf4(Bpi))
        real_prod = float(np.real_if_close(prod))
        if abs(real_prod) < 1e-12:
            raise RuntimeError("Pfaffian product too close to zero")
        return -1 if real_prod < 0 else 1

    def build_finite(self, disorder: Optional[np.ndarray] = None) -> np.ndarray:
        p = self.p
        tzI = np.kron(self.tz, self.I2)
        H0 = (
            (2 * p.t0 - p.mu) * tzI
            + p.Ez * np.kron(self.I2, self.sx)
            + p.Delta * np.kron(self.tx, self.I2)
        )
        Hhop = -p.t0 * tzI - 0.5j * p.alpha * np.kron(self.tz, self.sy)
        H = np.zeros((4 * p.L, 4 * p.L), dtype=complex)
        for i in range(p.L):
            sl = slice(4 * i, 4 * (i + 1))
            site = H0.copy()
            if disorder is not None:
                site = site - float(disorder[i]) * tzI
            H[sl, sl] = site
            if i < p.L - 1:
                sl2 = slice(4 * (i + 1), 4 * (i + 2))
                H[sl, sl2] = Hhop
                H[sl2, sl] = dagger(Hhop)
        return (H + dagger(H)) / 2.0

    def central_spectrum(self, disorder: Optional[np.ndarray] = None, n_each: int = 4):
        H = self.build_finite(disorder)
        n = H.shape[0]
        k = min(max(2 * n_each + 2, 6), n - 2)
        try:
            vals, vecs = eigsh(
                csr_matrix(H), k=k, sigma=0.0, which="LM",
                return_eigenvectors=True, tol=1e-10, maxiter=5000,
            )
            order = np.argsort(vals)
            return vals[order], vecs[:, order]
        except Exception:
            mid = n // 2
            return la.eigh(H, subset_by_index=[max(0, mid - n_each), min(n - 1, mid + n_each - 1)])

    def majorana_pair(self, disorder: Optional[np.ndarray] = None) -> Dict[str, float]:
        evals, evecs = self.central_spectrum(disorder)
        pos = np.where(evals > 1e-12)[0]
        neg = np.where(evals < -1e-12)[0]
        if len(pos) < 1 or len(neg) < 1:
            raise RuntimeError("Insufficient positive/negative-energy levels")

        pidx = pos[0]
        E_low = float(evals[pidx])
        E_next = float(evals[pos[1]]) if len(pos) > 1 else float("nan")
        psi_plus = evecs[:, pidx]
        U_C_full = np.kron(np.eye(self.p.L, dtype=complex), self.U_C)
        psi_minus = U_C_full @ psi_plus.conj()
        psi_minus /= max(la.norm(psi_minus), 1e-15)

        g1 = (psi_plus + psi_minus) / np.sqrt(2.0)
        g2 = -1j * (psi_plus - psi_minus) / np.sqrt(2.0)
        g1 /= max(la.norm(g1), 1e-15)
        g2 /= max(la.norm(g2), 1e-15)

        site_pos = np.repeat(np.arange(self.p.L, dtype=float), 4)
        Xop = np.diag(site_pos)
        G = np.column_stack([g1, g2])
        _, U = la.eigh((G.conj().T @ Xop @ G + (G.conj().T @ Xop @ G).conj().T) / 2.0)
        modes = G @ U
        m1, m2 = modes[:, 0], modes[:, 1]

        def weights(mode):
            w = np.array([np.sum(np.abs(mode[4 * i:4 * (i + 1)]) ** 2) for i in range(self.p.L)])
            return w / max(w.sum(), 1e-15)

        w1, w2 = weights(m1), weights(m2)
        edge = max(4, int(round(self.p.L * 0.10)))
        w1L, w1R = float(w1[:edge].sum()), float(w1[-edge:].sum())
        w2L, w2R = float(w2[:edge].sum()), float(w2[-edge:].sum())
        x1, x2 = float(np.dot(np.arange(self.p.L), w1)), float(np.dot(np.arange(self.p.L), w2))
        if x1 <= x2:
            R_L = w1L / (w1R + 1e-12)
            R_R = w2R / (w2L + 1e-12)
        else:
            R_L = w2L / (w2R + 1e-12)
            R_R = w1R / (w1L + 1e-12)

        gap = self.bulk_gap()
        return {
            "E_low": E_low,
            "E_next": E_next,
            "bulk_gap": gap,
            "E_low_over_gap": E_low / max(gap, 1e-15),
            "E_low_over_next": E_low / max(E_next, 1e-15),
            "R_L": R_L,
            "R_R": R_R,
        }

    @staticmethod
    def generate_disorder(strength: float, seed: int, L: int) -> np.ndarray:
        return np.random.default_rng(seed).uniform(-strength, strength, size=L)


def fit_oscillatory_splitting(
    lengths: Sequence[int], energies: Sequence[float],
    n_restart_batches: int = 4, restarts_per_batch: int = 30,
) -> Dict[str, float]:
    """Optional descriptor only; it is deliberately outside the core theorem."""
    L = np.asarray(lengths, float)
    E = np.asarray(energies, float)
    mask = (E > 1e-12) & np.isfinite(E)
    L, E = L[mask], E[mask]
    if L.size < 8:
        return {"fit_success": 0.0}

    def predict(p):
        return np.exp(p[0]) * np.exp(-L / np.exp(p[1])) * np.abs(np.cos(p[2] * L + p[3])) + 1e-12

    def resid(p):
        return np.log(predict(p)) - np.log(E)

    lo = np.array([-12.0, np.log(2.0), 0.0, -np.pi])
    hi = np.array([1.0, np.log(1000.0), np.pi, np.pi])
    fixed = [
        np.array([np.log(max(E)), np.log(30), 0.15, 0.0]),
        np.array([np.log(max(E)), np.log(60), 0.40, 1.0]),
        np.array([np.log(max(E)), np.log(80), 0.60, -1.0]),
    ]

    batches = []
    for batch in range(n_restart_batches):
        rng = np.random.default_rng(5000 + batch)
        starts = list(fixed)
        starts.extend(np.array([rng.uniform(lo[i], hi[i]) for i in range(4)]) for _ in range(restarts_per_batch))
        best = None
        for x0 in starts:
            try:
                r = least_squares(resid, x0, bounds=(lo, hi), max_nfev=4000)
            except Exception:
                continue
            score = float(np.mean(r.fun ** 2))
            if best is None or score < best[0]:
                best = (score, r.x)
        if best is not None:
            score, p = best
            batches.append({
                "A": float(np.exp(p[0])),
                "xi": float(np.exp(p[1])),
                "k": float(p[2]),
                "phi": float(p[3]),
                "log_rmse": float(np.sqrt(score)),
            })

    if not batches:
        return {"fit_success": 0.0}
    xis = np.array([b["xi"] for b in batches])
    rmses = np.array([b["log_rmse"] for b in batches])
    best = batches[int(np.argmin(rmses))]
    return {
        "fit_success": 1.0,
        **best,
        "xi_mean": float(xis.mean()),
        "xi_std": float(xis.std()),
        "log_rmse_mean": float(rmses.mean()),
        "log_rmse_std": float(rmses.std()),
        "n_restart_batches": len(batches),
        "fit_is_fragile": bool(xis.std() / max(xis.mean(), 1e-12) > 0.15),
    }


def operating_window_scan(
    base: BdGParams, L_list: Sequence[int], W_list: Sequence[float], n_real: int = 40,
    thresholds: Optional[Dict[str, float]] = None,
):
    """Random-disorder support; W=0 is deterministic and gets an exact status."""
    thr = thresholds or {"max_gap_ratio": 0.25, "max_next_ratio": 0.25, "min_R": 3.0}
    shape = (len(L_list), len(W_list))
    support = np.full(shape, np.nan)
    ci_lo = np.full(shape, np.nan)
    ci_hi = np.full(shape, np.nan)
    cells = []
    eps_pass = []

    for i, L in enumerate(L_list):
        wire = BdGWire(BdGParams(L=L, alpha=base.alpha, Delta=base.Delta, Ez=base.Ez, t0=base.t0, mu=base.mu))
        for j, W in enumerate(W_list):
            trials = 1 if W == 0 else n_real
            ok = 0
            for r in range(trials):
                dis = None if W == 0 else wire.generate_disorder(W, 7000 + i * 997 + j * 89 + r, L)
                d = wire.majorana_pair(dis)
                passed = (
                    d["E_low_over_gap"] <= thr["max_gap_ratio"]
                    and d["E_low_over_next"] <= thr["max_next_ratio"]
                    and d["R_L"] >= thr["min_R"]
                    and d["R_R"] >= thr["min_R"]
                )
                ok += int(passed)
                if passed:
                    eps_pass.append(d["E_low"])
            support[i, j] = ok / trials
            if W == 0:
                ci_lo[i, j] = ci_hi[i, j] = support[i, j]
                ci_type = "exact"
            else:
                ci_lo[i, j], ci_hi[i, j] = wilson_interval(ok, trials)
                ci_type = "Wilson-95%"
            cells.append({
                "L": int(L), "W": float(W), "support": float(support[i, j]),
                "ci_lo": float(ci_lo[i, j]), "ci_hi": float(ci_hi[i, j]),
                "n": int(trials), "ci_type": ci_type,
            })
    return support, ci_lo, ci_hi, np.asarray(eps_pass), cells


# -----------------------------------------------------------------------------
# 2. Effective qubit model: longitudinal Majorana term + external drive
# -----------------------------------------------------------------------------


def h_z_of_ng(Ec: float, n_g: float) -> float:
    """For Z=1-2n in the {|0>,|1>} charge basis."""
    return 2.0 * Ec * (2.0 * n_g - 1.0)


def ng_for_target_total_z(Ec: float, target_h_total: float, E_M: float,
                          t: Optional[float] = None, majorana_z_sign: float = -1.0) -> Tuple[float, float]:
    """Choose n_g so that h_z(n_g)+majorana_z_sign*E_M/2 = target_h_total.

    If t is given, target_h_total is instead interpreted as theta/(2t).
    majorana_z_sign=-1 is the standard positive-E_M BdG convention for
    H_M=(i E_M/2) gamma_L gamma_R with Z=1-2n_d.
    """
    if t is not None:
        if abs(t) < 1e-15:
            raise ValueError("Pulse duration must be nonzero")
        target_h_total = target_h_total / (2.0 * t)
    h_z_required = target_h_total - majorana_z_sign * E_M / 2.0
    n_g = 0.5 + h_z_required / (4.0 * Ec)
    return float(n_g), float(h_z_required)


def ng_for_majorana_compensation(Ec: float, E_M: float, majorana_z_sign: float = -1.0) -> float:
    """Set total longitudinal field h_z + sign*E_M/2 to zero."""
    h_z_required = -majorana_z_sign * E_M / 2.0
    return float(0.5 + h_z_required / (4.0 * Ec))


def logical_hamiltonian(h_z: float, E_M: float, Omega: float = 0.0, phi: float = 0.0,
                       majorana_z_sign: float = -1.0) -> np.ndarray:
    """H = [h_z + sign*E_M/2] Z + Omega/2(cos phi X + sin phi Y)."""
    P = pauli()
    H = (h_z + majorana_z_sign * E_M / 2.0) * P["Z"]
    H += 0.5 * Omega * (math.cos(phi) * P["X"] + math.sin(phi) * P["Y"])
    return H


def fidelity_from_bloch(v: np.ndarray, target: np.ndarray) -> float:
    return float(np.clip(0.5 * (1.0 + np.dot(v, target)), 0.0, 1.0))


def rotate_bloch(v0: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, float)
    nrm = np.linalg.norm(axis)
    if nrm < 1e-15:
        return np.asarray(v0, float).copy()
    axis = axis / nrm
    return (
        v0 * math.cos(angle)
        + np.cross(axis, v0) * math.sin(angle)
        + axis * np.dot(axis, v0) * (1.0 - math.cos(angle))
    )


def evolve_bloch_constant_hamiltonian(v0: np.ndarray, h_z_total: float, Omega: float,
                                       phi: float, t: float) -> np.ndarray:
    """Exact Bloch evolution for H=h_z_total Z + Omega/2*(cos phi X+sin phi Y)."""
    omega_vec = np.array([Omega * math.cos(phi), Omega * math.sin(phi), 2.0 * h_z_total], dtype=float)
    speed = float(np.linalg.norm(omega_vec))
    if speed < 1e-15:
        return np.asarray(v0, float).copy()
    return rotate_bloch(np.asarray(v0, float), omega_vec, speed * t)


def t_gate_infidelity_from_calibration_error(delta_E_M: float, t: float) -> float:
    """T/equatorial Z-rotation infidelity after E_M miscalibration.

    The implemented pulse has phase error delta_theta = delta_E_M * t when
    the charge bias was calibrated using the wrong E_M by delta_E_M.
    """
    return float(math.sin(0.5 * delta_E_M * t) ** 2)


def h_state_infidelity_from_detuning(delta_E_M: float, t: float, Omega: float,
                                      target: np.ndarray, majorana_factor: float = 0.5) -> float:
    """H-type state preparation error from E_M calibration residual.

    The ideal pulse cancels the nominal longitudinal splitting. A calibration
    error delta_E_M therefore leaves a longitudinal detuning
    delta_z = -majorana_factor*delta_E_M (for sign -1 convention).
    """
    delta_z = majorana_factor * delta_E_M
    # Use phase -pi/2 so positive Omega implements a -Y rotation.
    v0 = np.array([1.0, 0.0, 0.0])
    v = evolve_bloch_constant_hamiltonian(v0, delta_z, Omega, -math.pi / 2.0, t)
    return 1.0 - fidelity_from_bloch(v, target)


def arbitrary_pure_target_geometry_floor(target_bloch: np.ndarray,
                                          control_axis: np.ndarray = np.array([0.0, 0.0, 1.0])) -> float:
    target = np.asarray(target_bloch, float)
    axis = np.asarray(control_axis, float)
    axis = axis / np.linalg.norm(axis)
    axial = float(np.dot(target, axis))
    return float(0.5 * (1.0 - math.sqrt(max(0.0, 1.0 - axial * axial))))


# -----------------------------------------------------------------------------
# 3. Regression tests for the corrected model
# -----------------------------------------------------------------------------


def run_regression_checks() -> Dict[str, float]:
    alg = verify_majorana_pair_algebra()
    assert alg["CAR_error"] < 1e-12
    assert alg["parity_minus_Z_error"] < 1e-12
    assert alg["parity_minus_2n_minus_1_error"] < 1e-12

    P = pauli()
    plus = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2.0)
    rho0 = np.outer(plus, plus.conj())
    target_T = np.array([math.cos(math.pi / 4.0), math.sin(math.pi / 4.0), 0.0])
    theta = math.pi / 4.0
    t = 0.5
    target_h_total = theta / (2.0 * t)

    max_matrix_error = 0.0
    for _ in range(40):
        rng = np.random.default_rng(9000 + _)
        E = float(10 ** rng.uniform(-5, -1))
        # Pick a random nominal h_z then calibrate it so the total Z coefficient is exact.
        h_z = target_h_total + E / 2.0
        H = logical_hamiltonian(h_z, E, majorana_z_sign=-1.0)
        U = la.expm(-1j * H * t)
        rho = U @ rho0 @ dagger(U)
        target_rho = 0.5 * (P["I"] + target_T[0] * P["X"] + target_T[1] * P["Y"])
        F = float(np.real(np.trace(target_rho @ rho)))
        max_matrix_error = max(max_matrix_error, abs(F - 1.0))

    # Exact phase-error law: compare matrix evolution with sin^2(delta E t / 2).
    max_phase_error = 0.0
    for delta_E in np.geomspace(1e-5, 0.2, 30):
        actual_h = target_h_total - delta_E / 2.0
        H = actual_h * P["Z"]
        U = la.expm(-1j * H * t)
        rho = U @ rho0 @ dagger(U)
        target_rho = 0.5 * (P["I"] + target_T[0] * P["X"] + target_T[1] * P["Y"])
        F = float(np.real(np.trace(target_rho @ rho)))
        pred = t_gate_infidelity_from_calibration_error(delta_E, t)
        max_phase_error = max(max_phase_error, abs((1.0 - F) - pred))

    assert max_matrix_error < 1e-10
    assert max_phase_error < 1e-10
    return {
        **alg,
        "max_exact_compensation_infidelity": max_matrix_error,
        "max_phase_error_formula_error": max_phase_error,
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main(outdir: str = "outputs_v11", n_real: int = 40, include_fit: bool = False,
         quick: bool = False, robustness: bool = False) -> None:
    os.makedirs(outdir, exist_ok=True)

    print("=" * 96)
    print("Majorana-Coulomb-Island: parity-aligned core model v11")
    print("Majorana splitting is longitudinal; calibration, not E_M itself, sets the residual error.")
    print("=" * 96)

    checks = run_regression_checks()
    print("\n[MAJORANA ALGEBRA]")
    print("  d=(gamma_L+i gamma_R)/2, Z=1-2 n_d")
    print("  i gamma_L gamma_R = 2 n_d - 1 = -Z  (verified numerically)")
    print(f"  algebra max errors = {max(checks[k] for k in checks if k.endswith('_error')):.2e}")

    # --- BdG physical input --------------------------------------------------
    bdg = BdGParams(L=80, alpha=0.15, Delta=1.0, Ez=2.5, t0=1.0, mu=0.5)
    wire = BdGWire(bdg)
    z2 = wire.z2_invariant()
    nominal = wire.majorana_pair()
    E_M = nominal["E_low"]
    print("\n[BDG]")
    print(f"  Z2 invariant nu = {z2:+d}")
    print(
        f"  E_M=E_low={E_M:.6e}, E_next={nominal['E_next']:.6e}, "
        f"bulk_gap={nominal['bulk_gap']:.6e}, R_L/R_R={nominal['R_L']:.1f}/{nominal['R_R']:.1f}"
    )

    # --- Charge + Majorana calibration -------------------------------------
    Ec = 0.8
    theta_T = math.pi / 4.0
    t_T = 0.5
    target_total_z = theta_T / (2.0 * t_T)
    ng_ignore, hz_ignore = ng_for_target_total_z(Ec, target_total_z, 0.0, majorana_z_sign=-1.0)
    ng_correct, hz_correct = ng_for_target_total_z(Ec, target_total_z, E_M, majorana_z_sign=-1.0)
    actual_total_if_ignore = hz_ignore - E_M / 2.0
    exact_total_after_correction = hz_correct - E_M / 2.0

    print("\n[CALIBRATION]")
    print(f"  target total Z coefficient = {target_total_z:.9f}")
    print(f"  ignoring E_M:  n_g={ng_ignore:.9f}, total Z={actual_total_if_ignore:.9f}")
    print(f"  correcting E_M: n_g={ng_correct:.9f}, total Z={exact_total_after_correction:.9f}")
    print(f"  exact corrected T infidelity = {t_gate_infidelity_from_calibration_error(0.0, t_T):.3e}")
    if not (0.0 <= ng_correct <= 1.0):
        print("  WARNING: compensation lies outside the nominal two-charge-state window [0,1].")
    else:
        print("  compensation is inside the nominal [0,1] offset-charge window for this example.")

    # --- H-state preparation with an external non-commuting drive ------------
    H_target = np.array([math.cos(math.pi / 4.0), 0.0, math.sin(math.pi / 4.0)])
    ng_cancel = ng_for_majorana_compensation(Ec, E_M, majorana_z_sign=-1.0)
    hz_cancel = h_z_of_ng(Ec, ng_cancel)
    Omega_H = math.pi / (4.0 * t_T)
    Hmat = logical_hamiltonian(hz_cancel, E_M, Omega_H, -math.pi / 2.0, majorana_z_sign=-1.0)
    plus = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2.0)
    rho = la.expm(-1j * Hmat * t_T) @ np.outer(plus, plus.conj()) @ dagger(la.expm(-1j * Hmat * t_T))
    P = pauli()
    target_rho_H = 0.5 * (P["I"] + H_target[0] * P["X"] + H_target[1] * P["Y"] + H_target[2] * P["Z"])
    F_H = float(np.real(np.trace(target_rho_H @ rho)))
    print("\n[EXTERNAL DRIVE]")
    print("  H-type target is off the pure-Z orbit, so a non-commuting drive is required.")
    print(f"  compensated n_g = {ng_cancel:.9f}, Omega*t = {Omega_H*t_T:.9f} (= pi/4)")
    print(f"  exact compensated H-state infidelity = {1.0-F_H:.3e}")

    # --- Core calibration-error curves --------------------------------------
    x = np.geomspace(1e-5, 1.5, 180)  # x = |delta E_M| * t
    delta_E = x / t_T
    inf_T = np.array([t_gate_infidelity_from_calibration_error(d, t_T) for d in delta_E])
    inf_H = np.array([h_state_infidelity_from_detuning(d, t_T, Omega_H, H_target) for d in delta_E])

    print("\n[CORE LAW]")
    print("  T gate, exact compensation: 1-F = 0 for any E_M that lies within the control model.")
    print("  T gate, calibration error deltaE_M: 1-F = sin^2(deltaE_M*t/2) ~ (deltaE_M*t)^2/4.")
    print("  H state with Y drive: exact compensation also gives zero error; residual detuning controls the error.")
    print(f"  For the nominal BdG E_M={E_M:.3e}, one-half splitting is {E_M/2:.3e} in code units.")

    # --- Optional BdG scaling diagnostic ------------------------------------
    lengths = list(range(20, 121, 5))
    E_of_L = []
    for L in lengths:
        w = BdGWire(BdGParams(L=L, alpha=bdg.alpha, Delta=bdg.Delta, Ez=bdg.Ez, t0=bdg.t0, mu=bdg.mu))
        E_of_L.append(w.majorana_pair()["E_low"])
    E_of_L = np.asarray(E_of_L)
    fit = fit_oscillatory_splitting(lengths, E_of_L) if include_fit else {"fit_success": 0.0, "note": "optional diagnostic skipped"}

    # --- Optional disorder window -------------------------------------------
    support = ci_lo = ci_hi = eps_pass = np.array([])
    cells = []
    if robustness:
        L_grid = [60, 80, 100] if quick else [40, 60, 80, 100, 120]
        W_grid = [0.0, 0.2, 0.4] if quick else [0.0, 0.1, 0.2, 0.3, 0.4]
        support, ci_lo, ci_hi, eps_pass, cells = operating_window_scan(bdg, L_grid, W_grid, n_real=n_real)
        print("\n[OPERATING WINDOW]")
        for i, L in enumerate(L_grid):
            row = []
            for j, W in enumerate(W_grid):
                if W == 0:
                    row.append(f"W={W:.1f}:{support[i,j]:.2f}(exact)")
                else:
                    row.append(f"W={W:.1f}:{support[i,j]:.2f}[{ci_lo[i,j]:.2f},{ci_hi[i,j]:.2f}]")
            print(f"  L={L:3d}  " + "  ".join(row))

    # --- Figures -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.semilogy(lengths, E_of_L, "o-", label=r"$E_M(L)=E_{low}(L)$")
    ax.axhline(nominal["bulk_gap"], ls=":", label="bulk gap")
    ax.set_xlabel("wire length L")
    ax.set_ylabel("energy")
    ax.set_title(f"BdG input layer: Z2={z2:+d}; finite-size splitting")
    ax.grid(True, which="both", ls=":", alpha=0.6)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig1_bdg_v11.png"), dpi=220)
    plt.close(fig)

    if robustness:
        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        im = ax.imshow(
            support, origin="lower", aspect="auto",
            extent=[min(W_grid), max(W_grid), min(L_grid), max(L_grid)], vmin=0, vmax=1,
        )
        ax.set_xlabel("disorder strength W")
        ax.set_ylabel("wire length L")
        ax.set_title("Majorana operating window: support P_M(L,W)")
        fig.colorbar(im, ax=ax, label="support fraction")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "fig2_operating_window_v11.png"), dpi=220)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.loglog(x, inf_T, label=r"T state: residual calibration error")
    ax.loglog(x, inf_H, label=r"H state: residual longitudinal detuning + Y drive")
    ax.loglog(x, x**2 / 4.0, "--", label=r"small-error T asymptote $x^2/4$")
    ax.set_xlabel(r"$x=|\delta E_M|t$")
    ax.set_ylabel("infidelity")
    ax.set_title("Core result: Majorana splitting is calibratable longitudinal detuning")
    ax.grid(True, which="both", ls=":", alpha=0.6)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig3_core_calibration_v11.png"), dpi=220)
    plt.close(fig)

    # --- Save summary --------------------------------------------------------
    summary = {
        "encoding": {
            "fermion_definition": "d=(gamma_L+i gamma_R)/2",
            "n_d": "d^dagger d",
            "logical_Z": "1-2 n_d",
            "identity": "i gamma_L gamma_R = 2 n_d - 1 = -Z",
            "algebra_checks": checks,
        },
        "effective_hamiltonian": {
            "form": "[h_z(n_g)+sign*E_M/2]Z + Omega/2(cos(phi)X+sin(phi)Y)",
            "majorana_z_sign_default": -1.0,
            "interpretation": "For positive BdG E_low and Z=1-2n_d, H_M=(i E_M/2)gamma_L gamma_R gives sign=-1.",
            "microscopic_projection_claim": False,
        },
        "nominal_bdg": nominal,
        "calibration": {
            "Ec": Ec,
            "theta_T": theta_T,
            "t_T": t_T,
            "E_M": E_M,
            "n_g_ignore_E_M": ng_ignore,
            "n_g_correct_E_M": ng_correct,
            "n_g_majorana_compensation": ng_cancel,
            "T_error_from_exact_compensation": 0.0,
            "H_error_from_exact_compensation": float(1.0-F_H),
        },
        "core_law": {
            "T_exact": "1-F=0 under exact longitudinal compensation",
            "T_calibration_error": "1-F=sin^2(delta_E_M*t/2) ~ (delta_E_M*t)^2/4",
            "H_control": "non-commuting drive supplies a second Bloch axis; exact compensation removes the nominal E_M detuning",
            "not_claimed": [
                "E_M is not an intrinsic T-gate error floor",
                "braiding-only Clifford limitation does not imply a continuously tunable longitudinal Hamiltonian is Clifford-only",
                "no vendor roadmap comparison",
            ],
        },
        "bdg_fit_optional": fit,
        "operating_window": ({
            "support": support.tolist(),
            "ci_lo": ci_lo.tolist(),
            "ci_hi": ci_hi.tolist(),
            "cells": cells,
        } if robustness else None),
    }
    with open(os.path.join(outdir, "core_results_v11.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n[CORE RESULT]")
    print("  The old transverse E_M X term has been removed.")
    print("  Same-pair Majorana splitting is represented as a longitudinal Z detuning.")
    print(f"  Exact E_M compensation gives T infidelity = 0.0 and H infidelity = {1.0-F_H:.3e} (drive sign convention checked).")
    print(f"  outputs -> {outdir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Majorana-Coulomb-Island parity-aligned core model")
    parser.add_argument("--outdir", default="outputs_v11")
    parser.add_argument("--n-real", type=int, default=40)
    parser.add_argument("--fit-descriptor", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--robustness", action="store_true")
    args = parser.parse_args()
    main(
        outdir=args.outdir,
        n_real=args.n_real,
        include_fit=args.fit_descriptor,
        quick=args.quick,
        robustness=args.robustness,
    )
