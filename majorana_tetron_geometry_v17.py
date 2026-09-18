#!/usr/bin/env python3
"""Majorana-tetron geometric noise / covariance law v17.

v17 is a stricter, falsifiable successor to v16.  It is designed around one
specific scientific question:

    Does Majorana geometry act as a reproducible transfer function that maps
    microscopic correlated disorder into an effective covariance between
    intrawire Majorana splittings?

The central microscopic chain is

    delta mu  ->  delta epsilon_a = K_a^T delta mu
               -> C_epsilon = K^T Sigma_mu K
               -> parity-resolved dephasing / lifetime asymmetry.

v17 adds four major upgrades over v16:

1. Actual intrawire sweet-spot search:
   each finite wire is scanned in chemical potential and a signed-splitting
   zero is bracketed with Brent's method when possible. If no sign-changing
   root exists in the requested interval, the code falls back to minimizing
   |epsilon_M| and explicitly labels the result approximate.

2. Full-BdG falsification ladder:
   direct disordered BdG samples are evaluated at the tuned operating points.
   The linear kernel is tested against several disorder strengths W, not just
   one W.  The code reports the scaling of residual nonlinearity and the
   direct-vs-kernel covariance error.

3. General spatial noise covariance:
   the geometry factor is evaluated for finite correlation length xi_noise,
   while the special white-noise case is recovered as xi_noise=0.

4. Device-design metric:
   in addition to rho_epsilon and eta, v17 reports parity-channel noise
   susceptibilities S_+ / W^2 and S_- / W^2 and a geometry objective

       J_geom = max(S_+, S_-) / W^2,

   which can be scanned over geometry without changing the microscopic noise
   strength. This is a design metric, not an optimized-device claim.

Core effective equations
------------------------
For total Majorana parity p = +/-1:

    H_p = 1/2 [ g X - (epsilon_12 + p epsilon_34) Z ].

At the joint intrawire sweet spot epsilon_12 = epsilon_34 = 0,

    delta z_p = -(delta epsilon_12 + p delta epsilon_34),

    S_p = sigma_12^2 + sigma_34^2 + 2 p C_12,34.

For centered jointly Gaussian quasistatic coupling noise and sigma << g,

    |W_p(t)| = [1 + (S_p t / g)^2]^(-1/4),

    T_1/2,p = sqrt(15) g / S_p,

    eta = (T_+ - T_-) / (T_+ + T_-)
        = -2 C_12,34 / (sigma_12^2 + sigma_34^2).

Microscopic transfer
--------------------
    delta epsilon_a = K_a^T delta mu,

    C_12,34 = K_12^T Sigma_mu,12 K_34.

For the separable paired-noise model

    Sigma_12 = rho_site W^2 R_12,

and therefore

    rho_epsilon = rho_site * chi_geom,

with

    chi_geom = (K_12^T R_12 K_34)
               / sqrt[(K_12^T R_11 K_12)(K_34^T R_22 K_34)].

What v17 does NOT claim
-----------------------
- p=+/- are total-parity sectors, not automatically two logical states of a
  fixed-parity tetron.
- The code does not derive the bridge coupling g microscopically.
- Direct finite-disorder BdG tests are numerical evidence, not a theorem.
- The covariance-lifetime inversion is an effective-model measurement protocol,
  not an experimental validation by itself.
- The geometry law is exact for the separable Gaussian site-noise construction;
  its usefulness as a device law must be tested across disorder scales,
  correlation lengths, and BdG parameter ranges.

Outputs
-------
The run writes JSON/NPZ/PNG files into --outdir and prints a compact scientific
report. The most important files are:

    core_results_v17.json
    sensitivity_kernels_v17.npz
    fig1_v17_operating_point_and_kernels.png
    fig2_v17_direct_vs_kernel_covariance.png
    fig3_v17_W_scaling_falsification.png
    fig4_v17_rho_transfer.png
    fig5_v17_geometry_design_map.png
    fig6_v17_dephasing_and_inversion.png

Example
-------
python majorana_tetron_geometry_v17.py \\
    --outdir outputs_v17 \\
    --n-real 80 \\
    --mc 100000 \\
    --kernel-ensemble 1000000 \\
    --rho-site 0.60 \\
    --xi-noise 0.0 \\
    --mu-scan-min 0.0 \\
    --mu-scan-max 1.5

"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
from scipy.optimize import brentq, minimize_scalar
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh


# -----------------------------------------------------------------------------
# Basic linear algebra
# -----------------------------------------------------------------------------


def paulis() -> Dict[str, np.ndarray]:
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    return {"I": I, "X": X, "Y": Y, "Z": Z}


def dagger(a: np.ndarray) -> np.ndarray:
    return a.conj().T


def frob(a: np.ndarray) -> float:
    return float(la.norm(a))


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    sa = float(np.std(a))
    sb = float(np.std(b))
    if sa <= 0 or sb <= 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def distribution_moments(x: np.ndarray, ddof: int = 1) -> Dict[str, float]:
    x = np.asarray(x, dtype=float)
    mu = float(np.mean(x))
    if x.size <= ddof:
        return {
            "mean": mu,
            "std": float("nan"),
            "skew": float("nan"),
            "excess_kurtosis": float("nan"),
        }
    s = float(np.std(x, ddof=ddof))
    if s <= 0:
        return {
            "mean": mu,
            "std": s,
            "skew": float("nan"),
            "excess_kurtosis": float("nan"),
        }
    z = (x - mu) / s
    return {
        "mean": mu,
        "std": s,
        "skew": float(np.mean(z**3)),
        "excess_kurtosis": float(np.mean(z**4) - 3.0),
    }


def percentile_summary(x: np.ndarray, q=(2.5, 50.0, 97.5)) -> Dict[str, float]:
    x = np.asarray(x, dtype=float)
    mask = np.isfinite(x)
    vals = np.percentile(x[mask], q) if np.any(mask) else np.full(len(q), np.nan)
    return {f"q{int(qq)}": float(v) for qq, v in zip(q, vals)}


def relative_error(actual: float, target: float, floor: float = 1e-30) -> float:
    return float((actual - target) / max(abs(target), floor))


# -----------------------------------------------------------------------------
# Four-Majorana algebra
# -----------------------------------------------------------------------------


def jw_four_majoranas() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    P = paulis()
    sm = np.array([[0, 1], [0, 0]], dtype=complex)
    c1 = np.kron(sm, P["I"])
    c2 = np.kron(P["Z"], sm)
    g1 = c1 + dagger(c1)
    g2 = -1j * (c1 - dagger(c1))
    g3 = c2 + dagger(c2)
    g4 = -1j * (c2 - dagger(c2))
    return g1, g2, g3, g4


def four_majorana_pair_ops() -> Dict[str, np.ndarray]:
    g = jw_four_majoranas()
    names = ("12", "13", "14", "23", "24", "34")
    out: Dict[str, np.ndarray] = {}
    for name in names:
        a, b = int(name[0]) - 1, int(name[1]) - 1
        out[name] = 1j * g[a] @ g[b]
    return out


def parity_projector_basis(parity: int) -> np.ndarray:
    if parity not in (+1, -1):
        raise ValueError("parity must be +1 or -1")
    return np.array([0, 3] if parity == +1 else [1, 2], dtype=int)


def projected(block: np.ndarray, parity: int) -> np.ndarray:
    idx = parity_projector_basis(parity)
    return block[np.ix_(idx, idx)]


def validate_logical_paulis() -> Dict[str, float]:
    ops = four_majorana_pair_ops()
    P = paulis()
    max_err = 0.0
    for p in (+1, -1):
        Z = -projected(ops["12"], p)
        X = -projected(ops["23"], p)
        Y = projected(ops["13"], p)
        max_err = max(
            max_err,
            frob(Z @ Z - P["I"]),
            frob(X @ X - P["I"]),
            frob(Y @ Y - P["I"]),
            frob(X @ Y - 1j * Z),
            frob(Y @ Z - 1j * X),
            frob(Z @ X - 1j * Y),
            frob(projected(ops["34"], p) + p * Z),
            frob(projected(ops["14"], p) + p * X),
            frob(projected(ops["24"], p) + p * Y),
        )
    return {"logical_pauli_max_error": float(max_err)}


@dataclass(frozen=True)
class Couplings:
    e12: float = 0.0
    e13: float = 0.0
    e14: float = 0.0
    e23: float = 0.0
    e24: float = 0.0
    e34: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


_COUPLING_NAMES = ("e12", "e13", "e14", "e23", "e24", "e34")


def effective_field(c: Couplings, parity: int) -> np.ndarray:
    if parity not in (+1, -1):
        raise ValueError("parity must be +1 or -1")
    return np.array(
        [
            -(c.e23 + parity * c.e14),
            c.e13 - parity * c.e24,
            -(c.e12 + parity * c.e34),
        ],
        dtype=float,
    )


def effective_hamiltonian(c: Couplings, parity: int) -> np.ndarray:
    P = paulis()
    b = effective_field(c, parity)
    return 0.5 * (b[0] * P["X"] + b[1] * P["Y"] + b[2] * P["Z"])


def qubit_gap(c: Couplings, parity: int) -> float:
    return float(np.linalg.norm(effective_field(c, parity)))


def full_majorana_hamiltonian(c: Couplings) -> np.ndarray:
    g = jw_four_majoranas()
    pairs = {
        "12": c.e12,
        "13": c.e13,
        "14": c.e14,
        "23": c.e23,
        "24": c.e24,
        "34": c.e34,
    }
    H = np.zeros((4, 4), dtype=complex)
    for name, eps in pairs.items():
        a, b = int(name[0]) - 1, int(name[1]) - 1
        H += 0.5j * eps * g[a] @ g[b]
    return (H + dagger(H)) / 2.0


def validate_effective_projection(c: Couplings) -> Dict[str, float]:
    Hfull = full_majorana_hamiltonian(c)
    errs: Dict[str, float] = {}
    for p in (+1, -1):
        Hproj = projected(Hfull, p)
        Heff = effective_hamiltonian(c, p)
        errs[f"parity_{p}_projection_error"] = frob(Hproj - Heff)
        evals = np.sort(np.real_if_close(np.linalg.eigvalsh(Hproj)).astype(float))
        expected = np.array([-0.5 * qubit_gap(c, p), 0.5 * qubit_gap(c, p)])
        errs[f"parity_{p}_spectrum_error"] = float(np.max(np.abs(evals - expected)))
    return errs


def frequency_gradient(c: Couplings, parity: int, eps: float = 1e-14) -> np.ndarray:
    b = effective_field(c, parity)
    Om = float(np.linalg.norm(b))
    if Om < eps:
        return np.full(6, np.nan)
    bx, by, bz = b
    return np.array(
        [
            -bz / Om,
            +by / Om,
            -parity * bx / Om,
            -bx / Om,
            -parity * by / Om,
            -parity * bz / Om,
        ],
        dtype=float,
    )


def frequency_hessian(c: Couplings, parity: int, eps: float = 1e-14) -> np.ndarray:
    b = effective_field(c, parity)
    Om = float(np.linalg.norm(b))
    if Om < eps:
        return np.full((6, 6), np.nan)
    p = float(parity)
    J = np.array(
        [
            [0, 0, -p, -1, 0, 0],
            [0, 1, 0, 0, -p, 0],
            [-1, 0, 0, 0, 0, -p],
        ],
        dtype=float,
    )
    metric = np.eye(3) / Om - np.outer(b, b) / Om**3
    H = J.T @ metric @ J
    return 0.5 * (H + H.T)


def finite_difference_gradient(c: Couplings, parity: int, step: float = 1e-7) -> np.ndarray:
    vals = np.array(list(c.as_dict().values()), dtype=float)
    out = np.zeros(6, dtype=float)
    for k in range(6):
        vp = vals.copy(); vm = vals.copy()
        vp[k] += step; vm[k] -= step
        out[k] = (
            qubit_gap(Couplings(*vp.tolist()), parity)
            - qubit_gap(Couplings(*vm.tolist()), parity)
        ) / (2.0 * step)
    return out


def validate_frequency_derivatives(c: Couplings, parity: int) -> Dict[str, float]:
    ga = frequency_gradient(c, parity)
    gf = finite_difference_gradient(c, parity)
    gerr = float(np.max(np.abs(ga - gf)))
    H = frequency_hessian(c, parity)
    hstep = 2e-5
    Hfd = np.zeros((6, 6), dtype=float)
    base = np.array(list(c.as_dict().values()), dtype=float)
    for k in range(6):
        vp = base.copy(); vm = base.copy()
        vp[k] += hstep; vm[k] -= hstep
        gp = frequency_gradient(Couplings(*vp.tolist()), parity)
        gm = frequency_gradient(Couplings(*vm.tolist()), parity)
        Hfd[:, k] = (gp - gm) / (2.0 * hstep)
    herr = float(np.nanmax(np.abs(H - Hfd)))
    return {"gradient_max_error": gerr, "hessian_max_error": herr}


# -----------------------------------------------------------------------------
# Gaussian quasistatic dephasing / covariance spectroscopy
# -----------------------------------------------------------------------------


def covariance_psd_clipped(
    sigma12: float,
    sigma34: float,
    covariance: float,
    floor: float = 1e-15,
) -> np.ndarray:
    C = np.array([[sigma12**2, covariance], [covariance, sigma34**2]], dtype=float)
    evals, vecs = la.eigh(C)
    if np.min(evals) < -floor:
        raise ValueError(f"Noise covariance is not PSD: eigenvalues={evals}")
    evals = np.clip(evals, 0.0, None)
    return vecs @ np.diag(evals) @ vecs.T


def joint_two_channel_variance(
    sigma12: float,
    sigma34: float,
    covariance: float = 0.0,
    parity: int = +1,
) -> float:
    if sigma12 < 0 or sigma34 < 0:
        raise ValueError("sigmas must be non-negative")
    if parity not in (+1, -1):
        raise ValueError("parity must be +/-1")
    S = sigma12**2 + sigma34**2 + 2.0 * parity * covariance
    if S < -1e-12:
        raise ValueError(f"Negative parity-resolved variance S_p={S}")
    return float(max(S, 0.0))


def joint_sweetspot_coherence(
    sigma12: float,
    sigma34: float,
    g: float,
    t: Sequence[float],
    covariance: float = 0.0,
    parity: int = +1,
) -> np.ndarray:
    if g <= 0:
        raise ValueError("g must be positive")
    S = joint_two_channel_variance(sigma12, sigma34, covariance, parity)
    t = np.asarray(t, dtype=float)
    q = (S * t / g) ** 2
    return (1.0 + q) ** (-0.25)


def exact_gap_coherence_from_noise(
    d12: np.ndarray,
    d34: np.ndarray,
    g: float,
    parity: int,
    times: Sequence[float],
) -> np.ndarray:
    if g <= 0 or parity not in (+1, -1):
        raise ValueError("g must be positive and parity must be +/-1")
    d12 = np.asarray(d12, dtype=float)
    d34 = np.asarray(d34, dtype=float)
    dz = -(d12 + parity * d34)
    omega = np.sqrt(g**2 + dz**2)
    out = np.empty(len(times), dtype=float)
    for j, t in enumerate(np.asarray(times, dtype=float)):
        out[j] = abs(np.mean(np.exp(-1j * (omega - g) * t)))
    return out


def covariance_to_rho(sigma12: float, sigma34: float, covariance: float) -> float:
    denom = sigma12 * sigma34
    if denom <= 0:
        if abs(covariance) > 1e-14:
            raise ValueError("Nonzero covariance requires both sigmas > 0")
        return 0.0
    rho = covariance / denom
    if rho < -1.0 - 1e-12 or rho > 1.0 + 1e-12:
        raise ValueError(f"Covariance incompatible with sigmas: rho={rho}")
    return float(np.clip(rho, -1.0, 1.0))


def draw_correlated_gaussian_pair(
    sigma12: float,
    sigma34: float,
    correlation: float,
    n_samples: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if sigma12 < 0 or sigma34 < 0:
        raise ValueError("sigmas must be non-negative")
    if not (-1.0 <= correlation <= 1.0):
        raise ValueError("correlation must lie in [-1,1]")
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n_samples)
    y = rng.normal(size=n_samples)
    d12 = sigma12 * x
    d34 = sigma34 * (
        correlation * x
        + math.sqrt(max(0.0, 1.0 - correlation**2)) * y
    )
    return d12, d34


def quadratic_mc_coherence(
    sigma12: float,
    sigma34: float,
    covariance: float,
    g: float,
    parity: int,
    times: Sequence[float],
    n_samples: int = 200_000,
    seed: int = 2024,
) -> np.ndarray:
    d12, d34 = draw_correlated_gaussian_pair(
        sigma12,
        sigma34,
        covariance_to_rho(sigma12, sigma34, covariance),
        n_samples,
        seed,
    )
    dz = -(d12 + parity * d34)
    omega_shift = dz**2 / (2.0 * g)
    out = np.empty(len(times), dtype=float)
    for j, t in enumerate(np.asarray(times, dtype=float)):
        out[j] = abs(np.mean(np.exp(-1j * omega_shift * t)))
    return out


def exact_gaussian_mc_coherence(
    sigma12: float,
    sigma34: float,
    covariance: float,
    g: float,
    parity: int,
    times: Sequence[float],
    n_samples: int = 200_000,
    seed: int = 2024,
) -> np.ndarray:
    d12, d34 = draw_correlated_gaussian_pair(
        sigma12,
        sigma34,
        covariance_to_rho(sigma12, sigma34, covariance),
        n_samples,
        seed,
    )
    return exact_gap_coherence_from_noise(d12, d34, g, parity, times)


def quadratic_half_life(g: float, S: float) -> float:
    if g <= 0 or S <= 0:
        return float("inf")
    return float(math.sqrt(15.0) * g / S)


def covariance_inversion_from_half_lives(g: float, t_plus: float, t_minus: float) -> float:
    if g <= 0 or t_plus <= 0 or t_minus <= 0:
        raise ValueError("g and half-lives must be positive")
    return float(
        (math.sqrt(15.0) * g / 4.0)
        * (1.0 / t_plus - 1.0 / t_minus)
    )


def eta_from_covariance(sigma12: float, sigma34: float, covariance: float) -> float:
    den = sigma12**2 + sigma34**2
    if den <= 0:
        return float("nan")
    return float(-2.0 * covariance / den)


def eta_from_half_lives(t_plus: float, t_minus: float) -> float:
    den = t_plus + t_minus
    if den <= 0:
        return float("nan")
    return float((t_plus - t_minus) / den)


# -----------------------------------------------------------------------------
# Microscopic spinful-Rashba BdG model
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
    """Finite spinful Rashba nanowire BdG model."""

    def __init__(self, p: BdGParams):
        self.p = p
        P = paulis()
        self.I2 = P["I"]
        self.sx = P["X"]
        self.sy = P["Y"]
        self.tx = P["X"]
        self.ty = P["Y"]
        self.tz = P["Z"]
        self.U_C = np.kron(self.ty, self.sy)
        self._bulk_gap_cache: Dict[Tuple[int, float], float] = {}
        self._clean_majorana_basis_cache: Optional[np.ndarray] = None

    def with_mu(self, mu: float) -> "BdGWire":
        return BdGWire(
            BdGParams(
                L=self.p.L,
                alpha=self.p.alpha,
                Delta=self.p.Delta,
                Ez=self.p.Ez,
                t0=self.p.t0,
                mu=float(mu),
            )
        )

    def bloch_H(self, k: float) -> np.ndarray:
        p = self.p
        xi = 2 * p.t0 - 2 * p.t0 * np.cos(k) - p.mu
        return (
            xi * np.kron(self.tz, self.I2)
            + p.alpha * np.sin(k) * np.kron(self.tz, self.sy)
            + p.Ez * np.kron(self.I2, self.sx)
            + p.Delta * np.kron(self.tx, self.I2)
        )

    def bulk_gap(self, nk: int = 401) -> float:
        key = (int(nk), round(float(self.p.mu), 12))
        if key not in self._bulk_gap_cache:
            ks = np.linspace(-np.pi, np.pi, nk)
            self._bulk_gap_cache[key] = float(
                min(np.min(np.abs(np.linalg.eigvalsh(self.bloch_H(k)))) for k in ks)
            )
        return self._bulk_gap_cache[key]

    @staticmethod
    def pf4(A: np.ndarray) -> complex:
        return A[0, 1] * A[2, 3] - A[0, 2] * A[1, 3] + A[0, 3] * A[1, 2]

    def z2_invariant(self) -> int:
        B0 = self.bloch_H(0.0) @ self.U_C
        Bpi = self.bloch_H(np.pi) @ self.U_C
        prod = complex(self.pf4(B0) * self.pf4(Bpi))
        real_prod = float(np.real(prod))
        if abs(real_prod) < 1e-12:
            raise RuntimeError("Pfaffian product too close to zero")
        return -1 if real_prod < 0 else 1

    def build_finite(self, disorder: Optional[np.ndarray] = None) -> np.ndarray:
        p = self.p
        if disorder is not None:
            disorder = np.asarray(disorder, dtype=float)
            if disorder.shape != (p.L,):
                raise ValueError(f"disorder must have shape ({p.L},)")
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

    def central_spectrum(
        self,
        disorder: Optional[np.ndarray] = None,
        n_each: int = 4,
    ) -> Tuple[np.ndarray, np.ndarray]:
        H = self.build_finite(disorder)
        n = H.shape[0]
        k = min(max(2 * n_each + 2, 6), n - 2)
        try:
            vals, vecs = eigsh(
                csr_matrix(H),
                k=k,
                sigma=0.0,
                which="LM",
                return_eigenvectors=True,
                tol=1e-10,
                maxiter=5000,
            )
            order = np.argsort(vals)
            return vals[order], vecs[:, order]
        except Exception:
            mid = n // 2
            return la.eigh(
                H,
                subset_by_index=[
                    max(0, mid - n_each),
                    min(n - 1, mid + n_each - 1),
                ],
            )

    def lowest_positive_state(
        self,
        disorder: Optional[np.ndarray] = None,
    ) -> Tuple[float, np.ndarray, float]:
        vals, vecs = self.central_spectrum(disorder)
        pos = np.where(vals > 1e-12)[0]
        if len(pos) < 1:
            raise RuntimeError("No positive low-energy level found")
        i = int(pos[0])
        next_pos = float(vals[pos[1]]) if len(pos) > 1 else float("nan")
        return float(vals[i]), vecs[:, i], next_pos

    def localized_majoranas(
        self,
        disorder: Optional[np.ndarray] = None,
    ) -> Dict[str, object]:
        H = self.build_finite(disorder)
        vals, evecs = self.central_spectrum(disorder)
        pos = np.where(vals > 1e-12)[0]
        if len(pos) < 1:
            raise RuntimeError("No positive low-energy level found")
        psi_plus = evecs[:, pos[0]]
        U_C_full = np.kron(np.eye(self.p.L, dtype=complex), self.U_C)
        psi_minus = U_C_full @ psi_plus.conj()
        psi_minus /= max(la.norm(psi_minus), 1e-15)
        g1 = (psi_plus + psi_minus) / np.sqrt(2.0)
        g2 = -1j * (psi_plus - psi_minus) / np.sqrt(2.0)
        G = np.column_stack([g1, g2])

        x = np.repeat(np.arange(self.p.L, dtype=float), 4)
        xmat = np.real(G.conj().T @ (x[:, None] * G))
        xmat = 0.5 * (xmat + xmat.T)
        _, U = la.eigh(xmat)
        M = G @ U
        centers = np.array(
            [float(np.real(np.vdot(M[:, j], x * M[:, j]))) for j in range(2)]
        )
        if centers[0] > centers[1]:
            M = M[:, ::-1]
            centers = centers[::-1]
        for j in range(2):
            kmax = int(np.argmax(np.abs(M[:, j])))
            ref = M[kmax, j]
            if abs(ref) > 1e-14 and np.real(ref) < 0:
                M[:, j] *= -1.0
        coupling_matrix = M.conj().T @ H @ M
        signed_eps = float(np.imag(coupling_matrix[0, 1]))
        pos_energy = float(vals[pos[0]])
        next_pos = float(vals[pos[1]]) if len(pos) > 1 else float("nan")
        return {
            "M": M,
            "centers": centers,
            "signed_epsilon": signed_eps,
            "positive_energy": pos_energy,
            "next_positive_energy": next_pos,
            "bulk_gap": self.bulk_gap(),
            "z2": self.z2_invariant(),
            "splitting_over_gap": pos_energy / max(self.bulk_gap(), 1e-15),
            "splitting_over_next": pos_energy / max(next_pos, 1e-15),
        }

    def reference_majorana_basis(self) -> np.ndarray:
        if self._clean_majorana_basis_cache is None:
            self._clean_majorana_basis_cache = np.asarray(
                self.localized_majoranas(None)["M"], dtype=complex
            )
        return self._clean_majorana_basis_cache.copy()

    @staticmethod
    def real_procrustes(
        reference: np.ndarray,
        current: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        overlap = np.real(reference.conj().T @ current)
        U, _, Vh = la.svd(overlap)
        O = U @ Vh
        score = float(np.trace(O.T @ overlap) / max(reference.shape[1], 1))
        return O, score

    def signed_epsilon(
        self,
        disorder: Optional[np.ndarray] = None,
        reference_basis: Optional[np.ndarray] = None,
        return_diagnostics: bool = False,
    ):
        d = self.localized_majoranas(disorder)
        M = np.asarray(d["M"], dtype=complex)
        H = self.build_finite(disorder)
        score = float("nan")
        if reference_basis is not None:
            O, score = self.real_procrustes(reference_basis, M)
            M = M @ O
        coupling_matrix = M.conj().T @ H @ M
        eps_signed = float(np.imag(coupling_matrix[0, 1]))
        if not return_diagnostics:
            return eps_signed
        return {
            "signed_epsilon": eps_signed,
            "E_low": float(abs(d["positive_energy"])),
            "E_next": float(d["next_positive_energy"]),
            "bulk_gap": float(d["bulk_gap"]),
            "z2": int(d["z2"]),
            "E_low_over_gap": float(d["splitting_over_gap"]),
            "E_low_over_next": float(d["splitting_over_next"]),
            "x_left": float(d["centers"][0]),
            "x_right": float(d["centers"][1]),
            "gauge_overlap_score": score,
        }


# -----------------------------------------------------------------------------
# Sweet-spot search
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class SweetSpotResult:
    mu_star: float
    epsilon_star: float
    E_low: float
    bulk_gap: float
    z2: int
    method: str
    bracket: Optional[Tuple[float, float]]
    residual_abs_epsilon: float
    topological: bool
    scan_mu: Tuple[float, ...]
    scan_epsilon: Tuple[float, ...]

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


def tune_wire_to_sweet_spot(
    wire_template: BdGWire,
    mu_min: float,
    mu_max: float,
    n_scan: int = 25,
    root_tol: float = 1e-8,
    choose: str = "best_ratio",
) -> Tuple[BdGWire, np.ndarray, SweetSpotResult]:
    """Find a finite-wire intrawire sweet spot without a gauge artifact.

    Strategy:
      1. Scan the physical nonnegative splitting E_low(mu).
      2. Refine several low-E candidates.
      3. Around each candidate, freeze a *local* Majorana reference gauge and
         look for a continuous signed-epsilon zero. This prevents artificial
         sign changes caused by independently re-gauging every scan point.
      4. If no zero is found locally, retain the best refined minimum of E_low
         and label it minimum_abs_epsilon.

    This is more robust than using a separately redefined signed epsilon as the
    global scan objective.
    """
    if n_scan < 5:
        raise ValueError("n_scan must be >= 5")
    if not (mu_min < mu_max):
        raise ValueError("mu_min must be < mu_max")

    mus = np.linspace(mu_min, mu_max, n_scan)
    E = np.empty(n_scan, dtype=float)
    signed_local = np.empty(n_scan, dtype=float)
    for i, mu in enumerate(mus):
        w = wire_template.with_mu(float(mu))
        ref = w.reference_majorana_basis()
        d = w.signed_epsilon(None, reference_basis=ref, return_diagnostics=True)
        E[i] = float(d["E_low"])
        signed_local[i] = float(d["signed_epsilon"])

    # Candidate minima: all strict grid minima plus the endpoints if they are
    # competitive. Limiting to ~6 candidates keeps the tuning cost reasonable.
    minima_idx = [
        i for i in range(1, n_scan - 1)
        if E[i] <= E[i - 1] and E[i] <= E[i + 1]
    ]
    minima_idx += [0, n_scan - 1]
    minima_idx = sorted(set(minima_idx), key=lambda i: E[i])[:6]

    root_candidates: List[Tuple[float, np.ndarray, Tuple[float, float], str]] = []
    min_candidates: List[Tuple[float, np.ndarray, Tuple[float, float], str]] = []

    for idx in minima_idx:
        lo = float(mus[max(0, idx - 1)])
        hi = float(mus[min(n_scan - 1, idx + 1)])
        mu0 = float(mus[idx])
        w0 = wire_template.with_mu(mu0)
        ref0 = w0.reference_majorana_basis()

        def local_signed(mu: float) -> float:
            w = wire_template.with_mu(float(mu))
            return float(w.signed_epsilon(None, reference_basis=ref0))

        # Refine the physical minimum E_low first.
        res = minimize_scalar(
            lambda mu: float(wire_template.with_mu(float(mu)).lowest_positive_state(None)[0]),
            bounds=(lo, hi),
            method="bounded",
            options={"xatol": root_tol},
        )
        mu_minimum = float(res.x)
        w_minimum = wire_template.with_mu(mu_minimum)
        min_candidates.append((float(res.fun), ref0.copy(), (lo, hi), "minimum_abs_epsilon"))

        # Build a small local signed scan around the refined minimum. If a
        # sign-changing root exists, it is the preferred exact operating point.
        local_half = max((hi - lo) * 0.55, 10.0 * root_tol)
        a = max(mu_min, mu_minimum - local_half)
        b = min(mu_max, mu_minimum + local_half)
        local_grid = np.linspace(a, b, 7)
        local_vals = np.array([local_signed(float(m)) for m in local_grid], dtype=float)
        for j in range(len(local_grid) - 1):
            f0, f1 = float(local_vals[j]), float(local_vals[j + 1])
            if abs(f0) <= root_tol:
                root_candidates.append((float(local_grid[j]), ref0.copy(), (float(local_grid[j]), float(local_grid[j])), "bracketed_root"))
            elif f0 * f1 < 0.0:
                try:
                    root = brentq(
                        local_signed,
                        float(local_grid[j]),
                        float(local_grid[j + 1]),
                        xtol=root_tol,
                        rtol=1e-10,
                        maxiter=60,
                    )
                    root_candidates.append((float(root), ref0.copy(), (float(local_grid[j]), float(local_grid[j + 1])), "bracketed_root"))
                except Exception:
                    pass

    # Evaluate all exact roots using a fresh final local basis, and choose the
    # one with the smallest residual / best isolation ratio.
    scored_roots = []
    for mu_star, _, bracket, method in root_candidates:
        w = wire_template.with_mu(mu_star)
        ref = w.reference_majorana_basis()
        d = w.signed_epsilon(None, reference_basis=ref, return_diagnostics=True)
        isolation = float(d["E_low"] / max(d["bulk_gap"], 1e-15))
        score = -abs(float(d["signed_epsilon"])) + 1e-9 * isolation
        scored_roots.append((score, w, ref, d, method, bracket))

    if scored_roots:
        scored_roots.sort(key=lambda x: x[0], reverse=True)
        _, wire_star, ref_star, d_star, method, bracket = scored_roots[0]
    else:
        # No local zero found: take the physical E_low minimum with the best
        # topological/isolation quality among the refined candidates.
        scored_mins = []
        for _, _, bracket, method in min_candidates:
            lo, hi = bracket
            res = minimize_scalar(
                lambda mu: float(wire_template.with_mu(float(mu)).lowest_positive_state(None)[0]),
                bounds=(lo, hi),
                method="bounded",
                options={"xatol": root_tol},
            )
            w = wire_template.with_mu(float(res.x))
            ref = w.reference_majorana_basis()
            d = w.signed_epsilon(None, reference_basis=ref, return_diagnostics=True)
            isolation = float(d["E_low"] / max(d["bulk_gap"], 1e-15))
            score = -float(d["E_low"]) + 1e-9 * isolation
            scored_mins.append((score, w, ref, d, method, bracket))
        scored_mins.sort(key=lambda x: x[0], reverse=True)
        _, wire_star, ref_star, d_star, method, bracket = scored_mins[0]

    return wire_star, ref_star, SweetSpotResult(
        mu_star=float(wire_star.p.mu),
        epsilon_star=float(d_star["signed_epsilon"]),
        E_low=float(d_star["E_low"]),
        bulk_gap=float(d_star["bulk_gap"]),
        z2=int(d_star["z2"]),
        method=str(method),
        bracket=(float(bracket[0]), float(bracket[1])) if bracket is not None else None,
        residual_abs_epsilon=float(abs(d_star["signed_epsilon"])),
        topological=bool(int(d_star["z2"]) == -1),
        scan_mu=tuple(float(x) for x in mus),
        scan_epsilon=tuple(float(x) for x in signed_local),
    )


# -----------------------------------------------------------------------------
# Spatial noise model
# -----------------------------------------------------------------------------


def spatial_covariance_matrix(L: int, xi_noise: float) -> np.ndarray:
    """Unit-variance stationary covariance R_ij = exp(-|i-j|/xi)."""
    if xi_noise <= 0:
        return np.eye(L, dtype=float)
    idx = np.arange(L, dtype=float)
    R = np.exp(-np.abs(idx[:, None] - idx[None, :]) / float(xi_noise))
    return 0.5 * (R + R.T)


def cross_spatial_covariance_matrix(
    L1: int,
    L2: int,
    xi_noise: float,
) -> np.ndarray:
    if xi_noise <= 0:
        return np.eye(L1, L2, dtype=float)
    idx1 = np.arange(L1, dtype=float)
    idx2 = np.arange(L2, dtype=float)
    return np.exp(-np.abs(idx1[:, None] - idx2[None, :]) / float(xi_noise))


class PairedSiteNoiseSampler:
    """Joint Gaussian site-noise sampler with optional spatial correlation."""

    def __init__(
        self,
        L1: int,
        L2: int,
        W: float,
        rho_site: float,
        xi_noise: float = 0.0,
        seed: Optional[int] = None,
    ):
        if L1 <= 0 or L2 <= 0:
            raise ValueError("wire lengths must be positive")
        if W < 0:
            raise ValueError("W must be non-negative")
        if not (-1.0 <= rho_site <= 1.0):
            raise ValueError("rho_site must lie in [-1,1]")
        if xi_noise < 0:
            raise ValueError("xi_noise must be >= 0")
        self.L1 = int(L1)
        self.L2 = int(L2)
        self.W = float(W)
        self.rho = float(rho_site)
        self.xi = float(xi_noise)
        self.rng = np.random.default_rng(seed)
        self.Lmax = max(self.L1, self.L2)

        Rmax = spatial_covariance_matrix(self.Lmax, self.xi)
        R2 = spatial_covariance_matrix(self.L2, self.xi)
        jitter = 1e-12
        self.chol_max = la.cholesky(Rmax + jitter * np.eye(self.Lmax), lower=True)
        self.chol_2 = la.cholesky(R2 + jitter * np.eye(self.L2), lower=True)

    def sample(self, n: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        zc = self.rng.normal(size=(n, self.Lmax)) @ self.chol_max.T
        z2 = self.rng.normal(size=(n, self.L2)) @ self.chol_2.T
        d1 = self.W * zc[:, :self.L1]
        d2 = self.W * (
            self.rho * zc[:, :self.L2]
            + math.sqrt(max(0.0, 1.0 - self.rho**2)) * z2
        )
        if n == 1:
            return d1[0], d2[0]
        return d1, d2


def site_covariance_components(
    L1: int,
    L2: int,
    W: float,
    rho_site: float,
    xi_noise: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    R1 = spatial_covariance_matrix(L1, xi_noise)
    R2 = spatial_covariance_matrix(L2, xi_noise)
    R12 = cross_spatial_covariance_matrix(L1, L2, xi_noise)
    return W**2 * R1, W**2 * R2, rho_site * W**2 * R12


def full_site_covariance_matrix(
    L1: int,
    L2: int,
    W: float,
    rho_site: float,
    xi_noise: float = 0.0,
) -> np.ndarray:
    S11, S22, S12 = site_covariance_components(L1, L2, W, rho_site, xi_noise)
    S = np.zeros((L1 + L2, L1 + L2), dtype=float)
    S[:L1, :L1] = S11
    S[L1:, L1:] = S22
    S[:L1, L1:] = S12
    S[L1:, :L1] = S12.T
    return 0.5 * (S + S.T)


def validate_disorder_generator(
    L1: int,
    L2: int,
    W: float,
    rhos: Sequence[float],
    n_samples: int,
    seed: int,
    xi_noise: float = 0.0,
) -> Dict[str, object]:
    out: Dict[str, object] = {}
    n_shared = min(L1, L2)
    for rho in rhos:
        sampler = PairedSiteNoiseSampler(
            L1,
            L2,
            W,
            float(rho),
            xi_noise,
            seed + int((rho + 1.0) * 1000),
        )
        a, b = sampler.sample(n_samples)
        aa = a[:, :n_shared].reshape(-1)
        bb = b[:, :n_shared].reshape(-1)
        out[str(float(rho))] = {
            "requested_rho": float(rho),
            "sample_rho": safe_corr(aa, bb),
            "sample_std_1": float(np.std(aa, ddof=1)),
            "sample_std_2": float(np.std(bb, ddof=1)),
            "expected_std": float(W),
            "xi_noise": float(xi_noise),
        }
    return {"by_rho": out}


# -----------------------------------------------------------------------------
# BdG sensitivity kernels
# -----------------------------------------------------------------------------


def compute_sensitivity_kernel(
    wire: BdGWire,
    reference_basis: np.ndarray,
    step: float = 2e-4,
    zero_tol: float = 1e-7,
    force_fd: bool = False,
) -> Dict[str, object]:
    """Compute K_i = d epsilon_M / d disorder_i at the operating point.

    At an exact (or sufficiently accurate) intrawire sweet spot, the signed
    coupling derivative is obtained from degenerate first-order perturbation
    theory inside the two-dimensional Majorana subspace:

        K_i = Im[(M^† (dH/dmu_i) M)_{01}].

    This avoids the non-differentiability of E_low=|epsilon_M| at epsilon_M=0.
    An independent full finite-difference check is run separately. If the
    residual signed splitting is not small enough, v17 automatically switches
    to the direct finite-difference derivative unless force_fd=False and the
    caller explicitly accepts the off-sweet-spot local derivative.
    """
    if step <= 0:
        raise ValueError("step must be > 0")
    Mdata = wire.localized_majoranas(None)
    M = np.asarray(Mdata["M"], dtype=complex)
    if reference_basis is not None:
        O, gauge_score = wire.real_procrustes(reference_basis, M)
        M = M @ O
    else:
        gauge_score = float("nan")

    eps0 = float(wire.signed_epsilon(None, reference_basis=reference_basis))
    gap = float(Mdata["bulk_gap"])
    rel_residual = abs(eps0) / max(gap, 1e-15)

    if force_fd or rel_residual > zero_tol:
        K = finite_difference_kernel_direct(wire, reference_basis, step=step)
        method = "full finite difference of signed epsilon"
    else:
        tzI = np.kron(wire.tz, wire.I2)
        K = np.empty(wire.p.L, dtype=float)
        for i in range(wire.p.L):
            sl = slice(4 * i, 4 * (i + 1))
            Mi = M[sl, :]
            # dH/d(disorder_i) = -tau_z on site i.
            local = Mi.conj().T @ (-tzI) @ Mi
            K[i] = float(np.imag(local[0, 1]))
        method = "Majorana-subspace degenerate perturbation theory"

    E0 = float(Mdata["positive_energy"])
    return {
        "K": np.asarray(K, dtype=float),
        "epsilon_clean": eps0,
        "positive_energy_clean": E0,
        "bulk_gap": gap,
        "relative_residual_to_gap": rel_residual,
        "step_for_fd_validation": float(step),
        "norm": float(np.linalg.norm(K)),
        "max_abs": float(np.max(np.abs(K))),
        "gauge_overlap_score": float(gauge_score),
        "method": method,
    }


def validate_kernel_against_finite_difference(
    wire: BdGWire,
    reference_basis: np.ndarray,
    kernel: np.ndarray,
    step: float = 2e-4,
    sites: Optional[Sequence[int]] = None,
) -> Dict[str, object]:
    if sites is None:
        sites = np.unique(
            np.linspace(0, wire.p.L - 1, min(10, wire.p.L), dtype=int)
        ).tolist()
    clean = np.zeros(wire.p.L, dtype=float)
    vals = []
    for i in sites:
        dp = clean.copy(); dm = clean.copy()
        dp[int(i)] = step; dm[int(i)] = -step
        ep = float(wire.signed_epsilon(dp, reference_basis=reference_basis))
        em = float(wire.signed_epsilon(dm, reference_basis=reference_basis))
        kfd = (ep - em) / (2.0 * step)
        khf = float(kernel[int(i)])
        vals.append(
            {
                "site": int(i),
                "K_HF": khf,
                "K_FD": float(kfd),
                "abs_error": float(abs(khf - kfd)),
                "rel_error": float(abs(khf - kfd) / max(abs(kfd), 1e-15)),
            }
        )
    return {
        "sites": vals,
        "max_abs_error": float(max(v["abs_error"] for v in vals)) if vals else float("nan"),
        "max_relative_error": float(max(v["rel_error"] for v in vals)) if vals else float("nan"),
    }


def finite_difference_kernel_direct(
    wire: BdGWire,
    reference_basis: np.ndarray,
    step: float = 2e-4,
) -> np.ndarray:
    """Full-site finite-difference derivative of signed epsilon.

    This is used as a production cross-check of the HF kernel, not for the
    large ensemble calculation.
    """
    L = wire.p.L
    K = np.empty(L, dtype=float)
    zero = np.zeros(L, dtype=float)
    for i in range(L):
        dp = zero.copy(); dm = zero.copy()
        dp[i] = step; dm[i] = -step
        ep = float(wire.signed_epsilon(dp, reference_basis=reference_basis))
        em = float(wire.signed_epsilon(dm, reference_basis=reference_basis))
        K[i] = (ep - em) / (2.0 * step)
    return K


# -----------------------------------------------------------------------------
# Covariance transfer and design metric
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class KernelCovariance:
    sigma12: float
    sigma34: float
    covariance: float
    correlation: float
    geometry_factor: float
    eta: float
    S_plus: float
    S_minus: float
    J_geom: float
    chi_common: float
    chi_differential: float

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


def kernel_covariance(
    K1: np.ndarray,
    K2: np.ndarray,
    W: float,
    rho_site: float,
    xi_noise: float = 0.0,
) -> KernelCovariance:
    K1 = np.asarray(K1, dtype=float)
    K2 = np.asarray(K2, dtype=float)
    S11, S22, S12 = site_covariance_components(
        K1.size, K2.size, W, rho_site, xi_noise
    )
    var1 = float(K1 @ S11 @ K1)
    var2 = float(K2 @ S22 @ K2)
    cov = float(K1 @ S12 @ K2)
    sigma12 = math.sqrt(max(var1, 0.0))
    sigma34 = math.sqrt(max(var2, 0.0))
    den = sigma12 * sigma34
    corr = float(cov / den) if den > 0 else float("nan")

    R1 = spatial_covariance_matrix(K1.size, xi_noise)
    R2 = spatial_covariance_matrix(K2.size, xi_noise)
    R12 = cross_spatial_covariance_matrix(K1.size, K2.size, xi_noise)
    v1_unit = max(float(K1 @ R1 @ K1), 0.0)
    v2_unit = max(float(K2 @ R2 @ K2), 0.0)
    cross_unit = float(K1 @ R12 @ K2)
    geometry = (
        float(cross_unit / math.sqrt(v1_unit * v2_unit))
        if v1_unit > 0 and v2_unit > 0
        else float("nan")
    )

    S_plus = joint_two_channel_variance(sigma12, sigma34, cov, +1)
    S_minus = joint_two_channel_variance(sigma12, sigma34, cov, -1)
    eta = eta_from_covariance(sigma12, sigma34, cov)

    denom_geom = max(W**2, 1e-30)
    J_geom = max(S_plus, S_minus) / denom_geom
    # Dimensionless common/differential channel overlaps at the K level.
    chi_common = max(S_plus, 0.0) / denom_geom
    chi_differential = max(S_minus, 0.0) / denom_geom

    return KernelCovariance(
        sigma12=sigma12,
        sigma34=sigma34,
        covariance=cov,
        correlation=corr,
        geometry_factor=geometry,
        eta=eta,
        S_plus=S_plus,
        S_minus=S_minus,
        J_geom=J_geom,
        chi_common=chi_common,
        chi_differential=chi_differential,
    )


# -----------------------------------------------------------------------------
# Linearized and direct disorder ensembles
# -----------------------------------------------------------------------------


def generate_kernel_disorder_ensemble(
    K1: np.ndarray,
    K2: np.ndarray,
    eps1_clean: float,
    eps2_clean: float,
    L1: int,
    L2: int,
    W: float,
    rho_site: float,
    n_real: int,
    seed: int,
    batch_size: int = 20_000,
    xi_noise: float = 0.0,
) -> Dict[str, np.ndarray]:
    if n_real <= 0:
        raise ValueError("n_real must be positive")
    sampler = PairedSiteNoiseSampler(L1, L2, W, rho_site, xi_noise, seed)
    out1 = np.empty(n_real, dtype=float)
    out2 = np.empty(n_real, dtype=float)
    start = 0
    while start < n_real:
        n = min(batch_size, n_real - start)
        d1, d2 = sampler.sample(n)
        out1[start : start + n] = eps1_clean + d1 @ K1
        out2[start : start + n] = eps2_clean + d2 @ K2
        start += n
    return {"epsilon_1": out1, "epsilon_2": out2}


def paired_direct_bdg_ensemble(
    wire1: BdGWire,
    wire2: BdGWire,
    ref1: np.ndarray,
    ref2: np.ndarray,
    W: float,
    rho_site: float,
    n_real: int,
    seed: int,
    xi_noise: float = 0.0,
) -> Dict[str, object]:
    e1 = np.empty(n_real, dtype=float)
    e2 = np.empty(n_real, dtype=float)
    scores1 = np.empty(n_real, dtype=float)
    scores2 = np.empty(n_real, dtype=float)
    low1 = np.empty(n_real, dtype=float)
    low2 = np.empty(n_real, dtype=float)

    sampler = PairedSiteNoiseSampler(
        wire1.p.L,
        wire2.p.L,
        W,
        rho_site,
        xi_noise,
        seed,
    )
    for r in range(n_real):
        d1, d2 = sampler.sample(1)
        q1 = wire1.signed_epsilon(d1, reference_basis=ref1, return_diagnostics=True)
        q2 = wire2.signed_epsilon(d2, reference_basis=ref2, return_diagnostics=True)
        e1[r] = q1["signed_epsilon"]
        e2[r] = q2["signed_epsilon"]
        scores1[r] = q1["gauge_overlap_score"]
        scores2[r] = q2["gauge_overlap_score"]
        low1[r] = q1["E_low"]
        low2[r] = q2["E_low"]

    d1c = e1 - np.mean(e1)
    d2c = e2 - np.mean(e2)
    cov = float(np.cov(d1c, d2c, ddof=1)[0, 1])
    s1 = float(np.std(d1c, ddof=1))
    s2 = float(np.std(d2c, ddof=1))
    return {
        "epsilon_1": e1,
        "epsilon_2": e2,
        "sigma12": s1,
        "sigma34": s2,
        "covariance": cov,
        "correlation": safe_corr(d1c, d2c),
        "wire1_moments": distribution_moments(e1),
        "wire2_moments": distribution_moments(e2),
        "gauge_score_wire1": {"min": float(np.min(scores1)), "mean": float(np.mean(scores1))},
        "gauge_score_wire2": {"min": float(np.min(scores2)), "mean": float(np.mean(scores2))},
        "E_low_wire1": distribution_moments(low1),
        "E_low_wire2": distribution_moments(low2),
        "rho_site": float(rho_site),
        "xi_noise": float(xi_noise),
        "W": float(W),
        "n_real": int(n_real),
    }


def direct_linearization_realization_test(
    wire1: BdGWire,
    wire2: BdGWire,
    ref1: np.ndarray,
    ref2: np.ndarray,
    K1: np.ndarray,
    K2: np.ndarray,
    W: float,
    rho_site: float,
    n_real: int,
    seed: int,
    eps1_clean: float,
    eps2_clean: float,
    xi_noise: float = 0.0,
) -> Dict[str, float]:
    sampler = PairedSiteNoiseSampler(
        wire1.p.L,
        wire2.p.L,
        W,
        rho_site,
        xi_noise,
        seed,
    )
    direct1 = np.empty(n_real, dtype=float)
    direct2 = np.empty(n_real, dtype=float)
    linear1 = np.empty(n_real, dtype=float)
    linear2 = np.empty(n_real, dtype=float)
    for r in range(n_real):
        d1, d2 = sampler.sample(1)
        direct1[r] = wire1.signed_epsilon(d1, reference_basis=ref1)
        direct2[r] = wire2.signed_epsilon(d2, reference_basis=ref2)
        linear1[r] = eps1_clean + float(K1 @ d1)
        linear2[r] = eps2_clean + float(K2 @ d2)
    err1 = direct1 - linear1
    err2 = direct2 - linear2
    denom1 = max(float(np.std(direct1 - np.mean(direct1), ddof=1)), 1e-15)
    denom2 = max(float(np.std(direct2 - np.mean(direct2), ddof=1)), 1e-15)
    return {
        "wire1_rmse": float(np.sqrt(np.mean(err1**2))),
        "wire2_rmse": float(np.sqrt(np.mean(err2**2))),
        "wire1_rmse_over_direct_std": float(np.sqrt(np.mean(err1**2)) / denom1),
        "wire2_rmse_over_direct_std": float(np.sqrt(np.mean(err2**2)) / denom2),
        "wire1_corr": safe_corr(direct1, linear1),
        "wire2_corr": safe_corr(direct2, linear2),
    }


def empirical_coherence_from_epsilon_samples(
    e12: np.ndarray,
    e34: np.ndarray,
    g: float,
    parity: int,
    times: Sequence[float],
    center: bool = True,
) -> np.ndarray:
    e12 = np.asarray(e12, dtype=float)
    e34 = np.asarray(e34, dtype=float)
    d12 = e12 - np.mean(e12) if center else e12
    d34 = e34 - np.mean(e34) if center else e34
    return exact_gap_coherence_from_noise(d12, d34, g, parity, times)


# -----------------------------------------------------------------------------
# Direct W-scaling falsification experiment
# -----------------------------------------------------------------------------


def run_direct_W_scaling(
    wire1: BdGWire,
    wire2: BdGWire,
    ref1: np.ndarray,
    ref2: np.ndarray,
    K1: np.ndarray,
    K2: np.ndarray,
    W_values: Sequence[float],
    rho_site: float,
    xi_noise: float,
    n_real: int,
    seed: int,
) -> Dict[str, object]:
    rows = []
    for j, W in enumerate(W_values):
        direct = paired_direct_bdg_ensemble(
            wire1,
            wire2,
            ref1,
            ref2,
            float(W),
            rho_site,
            n_real,
            seed + 1000 * j,
            xi_noise=xi_noise,
        )
        kc = kernel_covariance(K1, K2, float(W), rho_site, xi_noise)
        d1 = np.asarray(direct["epsilon_1"], dtype=float)
        d2 = np.asarray(direct["epsilon_2"], dtype=float)
        dd1 = d1 - np.mean(d1)
        dd2 = d2 - np.mean(d2)
        cov = float(np.cov(dd1, dd2, ddof=1)[0, 1])
        s1 = float(np.std(dd1, ddof=1))
        s2 = float(np.std(dd2, ddof=1))
        rows.append(
            {
                "W": float(W),
                "direct_sigma12": s1,
                "direct_sigma34": s2,
                "direct_covariance": cov,
                "direct_rho_epsilon": safe_corr(dd1, dd2),
                "kernel_sigma12": kc.sigma12,
                "kernel_sigma34": kc.sigma34,
                "kernel_covariance": kc.covariance,
                "kernel_rho_epsilon": kc.correlation,
                "sigma12_rel_error": relative_error(s1, kc.sigma12),
                "sigma34_rel_error": relative_error(s2, kc.sigma34),
                "covariance_rel_error": relative_error(cov, kc.covariance),
            }
        )

    W_arr = np.asarray([r["W"] for r in rows], dtype=float)
    cov_abs = np.asarray([abs(r["direct_covariance"]) for r in rows], dtype=float)
    s1_arr = np.asarray([r["direct_sigma12"] for r in rows], dtype=float)
    s2_arr = np.asarray([r["direct_sigma34"] for r in rows], dtype=float)

    def loglog_slope(x: np.ndarray, y: np.ndarray) -> float:
        mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
        if np.count_nonzero(mask) < 2:
            return float("nan")
        return float(np.polyfit(np.log(x[mask]), np.log(y[mask]), 1)[0])

    return {
        "rows": rows,
        "power_law_exponent_sigma12": loglog_slope(W_arr, s1_arr),
        "power_law_exponent_sigma34": loglog_slope(W_arr, s2_arr),
        "power_law_exponent_covariance_abs": loglog_slope(W_arr, cov_abs),
    }


# -----------------------------------------------------------------------------
# Half-life extraction and bootstrap inversion
# -----------------------------------------------------------------------------


def first_crossing(
    curve: Sequence[float],
    times: Sequence[float],
    level: float = 0.5,
) -> float:
    curve = np.asarray(curve, dtype=float)
    times = np.asarray(times, dtype=float)
    idx = np.where(curve <= level)[0]
    if idx.size == 0:
        return float("inf")
    k = int(idx[0])
    if k == 0:
        return float(times[0])
    y0, y1 = float(curve[k - 1]), float(curve[k])
    t0, t1 = float(times[k - 1]), float(times[k])
    if y1 == y0:
        return t1
    return float(t0 + (level - y0) * (t1 - t0) / (y1 - y0))


def bootstrap_half_lives_and_covariance(
    e12: np.ndarray,
    e34: np.ndarray,
    g: float,
    times: Sequence[float],
    n_boot: int,
    seed: int,
    max_batch: int = 20,
) -> Dict[str, np.ndarray]:
    e12 = np.asarray(e12, dtype=float)
    e34 = np.asarray(e34, dtype=float)
    n = e12.size
    rng = np.random.default_rng(seed)
    times = np.asarray(times, dtype=float)

    tplus = np.empty(n_boot, dtype=float)
    tminus = np.empty(n_boot, dtype=float)
    cinf = np.empty(n_boot, dtype=float)
    eta = np.empty(n_boot, dtype=float)

    for start in range(0, n_boot, max_batch):
        B = min(max_batch, n_boot - start)
        idx = rng.integers(0, n, size=(B, n))
        for j in range(B):
            a = e12[idx[j]]
            b = e34[idx[j]]
            cp = empirical_coherence_from_epsilon_samples(a, b, g, +1, times)
            cm = empirical_coherence_from_epsilon_samples(a, b, g, -1, times)
            tp = first_crossing(cp, times)
            tm = first_crossing(cm, times)
            tplus[start + j] = tp
            tminus[start + j] = tm
            if np.isfinite(tp) and np.isfinite(tm):
                cinf[start + j] = covariance_inversion_from_half_lives(g, tp, tm)
                eta[start + j] = eta_from_half_lives(tp, tm)
            else:
                cinf[start + j] = np.nan
                eta[start + j] = np.nan
    return {
        "t_plus": tplus,
        "t_minus": tminus,
        "C_inferred": cinf,
        "eta_inferred": eta,
    }


# -----------------------------------------------------------------------------
# Geometry / transfer scan
# -----------------------------------------------------------------------------


def compute_geometry_transfer_scan(
    L1: int,
    L2_values: Sequence[int],
    alpha: float,
    Delta: float,
    Ez: float,
    t0: float,
    mu1: float,
    mu2_values: Sequence[float],
    W: float,
    rho_values: Sequence[float],
    xi_noise_values: Sequence[float],
    kernel_step: float,
    mu_scan_min: Optional[float] = None,
    mu_scan_max: Optional[float] = None,
    mu_scan_points: int = 13,
) -> Dict[str, object]:
    """Scan geometry factor over L2, mu2, rho_site, and xi_noise.

    The scan uses independent tuned or fixed operating points supplied by the
    caller.  It is intentionally separated from the sweet-spot search so that
    the scientific report can distinguish operating-point tuning from geometry
    variation.
    """
    wire1 = BdGWire(BdGParams(L=L1, alpha=alpha, Delta=Delta, Ez=Ez, t0=t0, mu=mu1))
    ref1 = wire1.reference_majorana_basis()
    kd1 = compute_sensitivity_kernel(wire1, ref1, kernel_step)
    K1 = np.asarray(kd1["K"], dtype=float)

    records = []
    for L2 in L2_values:
        candidates_mu2 = list(mu2_values)
        tuned_mu2 = None
        if mu_scan_min is not None and mu_scan_max is not None:
            template2 = BdGWire(BdGParams(L=L2, alpha=alpha, Delta=Delta, Ez=Ez, t0=t0, mu=float(mu2_values[0]) if len(mu2_values) else 0.5))
            wire2_tuned, ref2_tuned, sweet2 = tune_wire_to_sweet_spot(
                template2,
                mu_scan_min,
                mu_scan_max,
                n_scan=mu_scan_points,
            )
            candidates_mu2 = [sweet2.mu_star]
            tuned_mu2 = sweet2.mu_star
        for mu2 in candidates_mu2:
            wire2 = BdGWire(BdGParams(L=L2, alpha=alpha, Delta=Delta, Ez=Ez, t0=t0, mu=mu2))
            if tuned_mu2 is not None:
                ref2 = wire2.reference_majorana_basis()
            else:
                ref2 = wire2.reference_majorana_basis()
            kd2 = compute_sensitivity_kernel(wire2, ref2, kernel_step)
            K2 = np.asarray(kd2["K"], dtype=float)
            for xi in xi_noise_values:
                for rho in rho_values:
                    kc = kernel_covariance(K1, K2, W, rho, xi_noise=xi)
                    records.append(
                        {
                            "L1": int(L1),
                            "L2": int(L2),
                            "mu1": float(mu1),
                            "mu2": float(mu2),
                            "xi_noise": float(xi),
                            "rho_site": float(rho),
                            "geometry_factor": kc.geometry_factor,
                            "rho_epsilon": kc.correlation,
                            "sigma12": kc.sigma12,
                            "sigma34": kc.sigma34,
                            "C": kc.covariance,
                            "S_plus": kc.S_plus,
                            "S_minus": kc.S_minus,
                            "J_geom": kc.J_geom,
                            "eta": kc.eta,
                        }
                    )
    return {"records": records}


# -----------------------------------------------------------------------------
# Main research run
# -----------------------------------------------------------------------------


def run_core_demonstration(
    outdir: str = "outputs_v17",
    mc: int = 100_000,
    n_real: int = 80,
    kernel_ensemble: int = 1_000_000,
    L1: int = 80,
    L2: int = 65,
    W_noise: float = 0.03,
    g_real: float = 0.120,
    rho_site: float = 0.60,
    xi_noise: float = 0.0,
    alpha: float = 0.15,
    Delta: float = 1.0,
    Ez: float = 2.5,
    t0: float = 1.0,
    mu_initial_1: float = 0.50,
    mu_initial_2: float = 0.50,
    mu_scan_min: float = 0.0,
    mu_scan_max: float = 1.5,
    mu_scan_points: int = 19,
    kernel_step: float = 2e-4,
    bootstrap: int = 300,
    seed_direct: int = 7000,
    seed_kernel: int = 17000,
    seed_bootstrap: int = 27000,
    W_scaling_points: int = 4,
    do_geometry_scan: bool = True,
    geometry_L2_values: Sequence[int] = (55, 65, 75, 85),
    geometry_rho_points: int = 11,
    geometry_xi_values: Sequence[float] = (0.0, 1.0, 3.0),
) -> Dict[str, object]:
    os.makedirs(outdir, exist_ok=True)

    # A. Exact algebraic checks ----------------------------------------------
    alg = validate_logical_paulis()
    if alg["logical_pauli_max_error"] > 1e-12:
        raise AssertionError(f"Logical Pauli algebra failed: {alg}")

    c_test = Couplings(
        e12=0.013,
        e13=-0.007,
        e14=0.009,
        e23=0.081,
        e24=0.005,
        e34=-0.021,
    )
    proj = validate_effective_projection(c_test)
    if max(proj.values()) > 1e-12:
        raise AssertionError(f"Effective projection failed: {proj}")
    deriv_plus = validate_frequency_derivatives(c_test, +1)
    deriv_minus = validate_frequency_derivatives(c_test, -1)
    if max(max(deriv_plus.values()), max(deriv_minus.values())) > 2e-6:
        raise AssertionError(f"Frequency derivative checks failed: {deriv_plus}, {deriv_minus}")

    # B. Noise generator check ------------------------------------------------
    print("\n[v17] Correlated site-noise generator validation")
    gen_check = validate_disorder_generator(
        L1,
        L2,
        W_noise,
        rhos=(-0.6, 0.0, +0.6),
        n_samples=300,
        seed=4100,
        xi_noise=xi_noise,
    )
    for rho_key, row in gen_check["by_rho"].items():
        print(
            f"  rho_site={float(rho_key):+.2f}: sample_rho={row['sample_rho']:+.4f}, "
            f"std1={row['sample_std_1']:.4e}, std2={row['sample_std_2']:.4e}"
        )

    # C. Tune actual operating points ----------------------------------------
    print("\n[v17] Searching actual intrawire sweet spots")
    template1 = BdGWire(
        BdGParams(L=L1, alpha=alpha, Delta=Delta, Ez=Ez, t0=t0, mu=mu_initial_1)
    )
    template2 = BdGWire(
        BdGParams(L=L2, alpha=alpha, Delta=Delta, Ez=Ez, t0=t0, mu=mu_initial_2)
    )
    wire1, ref1, sweet1 = tune_wire_to_sweet_spot(
        template1,
        mu_scan_min,
        mu_scan_max,
        n_scan=mu_scan_points,
    )
    wire2, ref2, sweet2 = tune_wire_to_sweet_spot(
        template2,
        mu_scan_min,
        mu_scan_max,
        n_scan=mu_scan_points,
    )
    print(
        f"  wire1: mu*={sweet1.mu_star:+.8f}, eps*={sweet1.epsilon_star:+.4e}, "
        f"E_low={sweet1.E_low:.4e}, bulk_gap={sweet1.bulk_gap:.4e}, "
        f"method={sweet1.method}, topological={sweet1.topological}"
    )
    print(
        f"  wire2: mu*={sweet2.mu_star:+.8f}, eps*={sweet2.epsilon_star:+.4e}, "
        f"E_low={sweet2.E_low:.4e}, bulk_gap={sweet2.bulk_gap:.4e}, "
        f"method={sweet2.method}, topological={sweet2.topological}"
    )

    # D. Sensitivity kernels and exact HF/FD check ---------------------------
    print("\n[v17] Building sensitivity kernels at the tuned operating points")
    kd1 = compute_sensitivity_kernel(wire1, ref1, kernel_step)
    kd2 = compute_sensitivity_kernel(wire2, ref2, kernel_step)
    K1 = np.asarray(kd1["K"], dtype=float)
    K2 = np.asarray(kd2["K"], dtype=float)

    kcheck1 = validate_kernel_against_finite_difference(wire1, ref1, K1, kernel_step)
    kcheck2 = validate_kernel_against_finite_difference(wire2, ref2, K2, kernel_step)

    # Full-site finite-difference kernel check can be expensive, so run it only
    # when L is modest.  This is still a useful independent derivative route.
    full_fd_checks: Dict[str, object] = {}
    if L1 <= 100:
        K1_fd = finite_difference_kernel_direct(wire1, ref1, kernel_step)
        full_fd_checks["wire1"] = {
            "max_abs_error": float(np.max(np.abs(K1 - K1_fd))),
            "relative_l2_error": float(
                la.norm(K1 - K1_fd) / max(la.norm(K1_fd), 1e-15)
            ),
        }
    if L2 <= 100:
        K2_fd = finite_difference_kernel_direct(wire2, ref2, kernel_step)
        full_fd_checks["wire2"] = {
            "max_abs_error": float(np.max(np.abs(K2 - K2_fd))),
            "relative_l2_error": float(
                la.norm(K2 - K2_fd) / max(la.norm(K2_fd), 1e-15)
            ),
        }

    kc = kernel_covariance(K1, K2, W_noise, rho_site, xi_noise)
    S_mu = full_site_covariance_matrix(L1, L2, W_noise, rho_site, xi_noise)
    Kstack = np.concatenate([K1, K2])
    cov_full = Kstack @ S_mu @ Kstack
    expected_cov_matrix = np.array(
        [[kc.sigma12**2, kc.covariance], [kc.covariance, kc.sigma34**2]],
        dtype=float,
    )
    _ = covariance_psd_clipped(kc.sigma12, kc.sigma34, kc.covariance)

    print("\n  [TUNED KERNEL COVARIANCE]")
    print(f"    ||K12||={la.norm(K1):.6e}, ||K34||={la.norm(K2):.6e}")
    print(f"    chi_geom={kc.geometry_factor:+.6f}")
    print(f"    sigma12={kc.sigma12:.6e}, sigma34={kc.sigma34:.6e}")
    print(f"    C12,34={kc.covariance:+.6e}, rho_epsilon={kc.correlation:+.6f}")
    print(f"    S_plus={kc.S_plus:.6e}, S_minus={kc.S_minus:.6e}, eta={kc.eta:+.6f}")
    print(f"    J_geom={kc.J_geom:.6e}")
    # The stacked covariance quadratic form is the total variance of a
    # particular linear combination.  We also check the cross block directly.
    S11, S22, S12 = site_covariance_components(L1, L2, W_noise, rho_site, xi_noise)
    cross_direct = float(K1 @ S12 @ K2)
    print(f"    cross K^T Sigma_mu K error={cross_direct-kc.covariance:+.4e}")

    # E. Large kernel ensemble -----------------------------------------------
    print(f"\n[v17] Generating {kernel_ensemble:,} linearized microscopic realizations")
    large = generate_kernel_disorder_ensemble(
        K1,
        K2,
        sweet1.epsilon_star,
        sweet2.epsilon_star,
        L1,
        L2,
        W_noise,
        rho_site,
        kernel_ensemble,
        seed_kernel,
        xi_noise=xi_noise,
    )
    e12_large = np.asarray(large["epsilon_1"], dtype=float)
    e34_large = np.asarray(large["epsilon_2"], dtype=float)
    d12_large = e12_large - np.mean(e12_large)
    d34_large = e34_large - np.mean(e34_large)
    sigma12_large = float(np.std(d12_large, ddof=1))
    sigma34_large = float(np.std(d34_large, ddof=1))
    cov_large = float(np.cov(d12_large, d34_large, ddof=1)[0, 1])
    rho_large = safe_corr(d12_large, d34_large)
    print(
        f"  sigma12={sigma12_large:.6e}, sigma34={sigma34_large:.6e}, "
        f"C={cov_large:+.6e}, rho_epsilon={rho_large:+.6f}"
    )

    # F. Direct finite BdG ensemble ------------------------------------------
    print("\n[v17] Direct full-BdG disorder ensemble at the tuned points")
    direct = paired_direct_bdg_ensemble(
        wire1,
        wire2,
        ref1,
        ref2,
        W_noise,
        rho_site,
        n_real,
        seed_direct,
        xi_noise=xi_noise,
    )
    print(
        f"  direct sigma12={direct['sigma12']:.6e}, sigma34={direct['sigma34']:.6e}, "
        f"C={direct['covariance']:+.6e}, rho_epsilon={direct['correlation']:+.6f}"
    )
    print(
        f"  moments wire1 skew/kurt={direct['wire1_moments']['skew']:+.3f}/"
        f"{direct['wire1_moments']['excess_kurtosis']:+.3f}; "
        f"wire2 skew/kurt={direct['wire2_moments']['skew']:+.3f}/"
        f"{direct['wire2_moments']['excess_kurtosis']:+.3f}"
    )
    print(
        f"  gauge min/mean wire1={direct['gauge_score_wire1']['min']:.4f}/"
        f"{direct['gauge_score_wire1']['mean']:.4f}; "
        f"wire2={direct['gauge_score_wire2']['min']:.4f}/"
        f"{direct['gauge_score_wire2']['mean']:.4f}"
    )

    linearity = direct_linearization_realization_test(
        wire1,
        wire2,
        ref1,
        ref2,
        K1,
        K2,
        W_noise,
        rho_site,
        max(20, min(40, n_real)),
        seed_direct + 10000,
        sweet1.epsilon_star,
        sweet2.epsilon_star,
        xi_noise=xi_noise,
    )
    print("\n  [DIRECT BdG vs LINEAR KERNEL]")
    print(
        f"    wire1 RMSE/std={linearity['wire1_rmse_over_direct_std']:.4f}, "
        f"corr={linearity['wire1_corr']:.5f}"
    )
    print(
        f"    wire2 RMSE/std={linearity['wire2_rmse_over_direct_std']:.4f}, "
        f"corr={linearity['wire2_corr']:.5f}"
    )

    direct_rel_errors = {
        "sigma12": relative_error(float(direct["sigma12"]), kc.sigma12),
        "sigma34": relative_error(float(direct["sigma34"]), kc.sigma34),
        "covariance": relative_error(float(direct["covariance"]), kc.covariance),
        "correlation": float(direct["correlation"] - kc.correlation),
    }

    # G. W-scaling falsification ---------------------------------------------
    W_lo = max(W_noise / 4.0, 1e-4)
    W_hi = max(W_noise * 1.5, W_lo * 2.0)
    W_values = np.linspace(W_lo, W_hi, max(3, W_scaling_points)).tolist()
    print("\n[v17] Full-BdG disorder-strength scaling test")
    wscale = run_direct_W_scaling(
        wire1,
        wire2,
        ref1,
        ref2,
        K1,
        K2,
        W_values,
        rho_site,
        xi_noise,
        max(16, min(35, n_real)),
        seed_direct + 20000,
    )
    print(
        f"  sigma scaling exponents: wire1={wscale['power_law_exponent_sigma12']:.3f}, "
        f"wire2={wscale['power_law_exponent_sigma34']:.3f}"
    )
    print(
        f"  covariance magnitude scaling exponent={wscale['power_law_exponent_covariance_abs']:.3f}"
    )

    # H. Coherence / covariance spectroscopy ---------------------------------
    # Choose a time grid from both the kernel prediction and the directly
    # observed full-BdG variances so that both 0.5 crossings are visible when
    # they exist.
    t_half_plus = quadratic_half_life(g_real, kc.S_plus)
    t_half_minus = quadratic_half_life(g_real, kc.S_minus)
    S_direct_plus = float(joint_two_channel_variance(direct["sigma12"], direct["sigma34"], direct["covariance"], +1))
    S_direct_minus = float(joint_two_channel_variance(direct["sigma12"], direct["sigma34"], direct["covariance"], -1))
    t_direct_est_plus = quadratic_half_life(g_real, S_direct_plus)
    t_direct_est_minus = quadratic_half_life(g_real, S_direct_minus)
    finite_half_lives = [
        x for x in (t_half_plus, t_half_minus, t_direct_est_plus, t_direct_est_minus)
        if np.isfinite(x) and x > 0
    ]
    t_scale = max(max(finite_half_lives) if finite_half_lives else 1.0, 1.0)
    times = np.linspace(0.0, 1.25 * t_scale, 520)

    analytic_plus = joint_sweetspot_coherence(
        kc.sigma12, kc.sigma34, g_real, times, kc.covariance, +1
    )
    analytic_minus = joint_sweetspot_coherence(
        kc.sigma12, kc.sigma34, g_real, times, kc.covariance, -1
    )
    kernel_emp_plus = empirical_coherence_from_epsilon_samples(
        e12_large, e34_large, g_real, +1, times
    )
    kernel_emp_minus = empirical_coherence_from_epsilon_samples(
        e12_large, e34_large, g_real, -1, times
    )

    mc_quad_plus = quadratic_mc_coherence(
        kc.sigma12, kc.sigma34, kc.covariance, g_real, +1, times, mc, 111
    )
    mc_quad_minus = quadratic_mc_coherence(
        kc.sigma12, kc.sigma34, kc.covariance, g_real, -1, times, mc, 111
    )
    mc_exact_plus = exact_gaussian_mc_coherence(
        kc.sigma12, kc.sigma34, kc.covariance, g_real, +1, times, mc, 211
    )
    mc_exact_minus = exact_gaussian_mc_coherence(
        kc.sigma12, kc.sigma34, kc.covariance, g_real, -1, times, mc, 211
    )

    direct_coh_plus = empirical_coherence_from_epsilon_samples(
        np.asarray(direct["epsilon_1"]),
        np.asarray(direct["epsilon_2"]),
        g_real,
        +1,
        times,
    )
    direct_coh_minus = empirical_coherence_from_epsilon_samples(
        np.asarray(direct["epsilon_1"]),
        np.asarray(direct["epsilon_2"]),
        g_real,
        -1,
        times,
    )

    tplus_large = first_crossing(kernel_emp_plus, times)
    tminus_large = first_crossing(kernel_emp_minus, times)
    C_from_large_half = covariance_inversion_from_half_lives(
        g_real, tplus_large, tminus_large
    )
    eta_from_large_half = eta_from_half_lives(tplus_large, tminus_large)

    tplus_direct = first_crossing(direct_coh_plus, times)
    tminus_direct = first_crossing(direct_coh_minus, times)
    C_from_direct_half = (
        covariance_inversion_from_half_lives(g_real, tplus_direct, tminus_direct)
        if np.isfinite(tplus_direct) and np.isfinite(tminus_direct)
        else float("nan")
    )
    eta_direct_half = (
        eta_from_half_lives(tplus_direct, tminus_direct)
        if np.isfinite(tplus_direct) and np.isfinite(tminus_direct)
        else float("nan")
    )

    bootstrap_out = bootstrap_half_lives_and_covariance(
        np.asarray(direct["epsilon_1"], dtype=float),
        np.asarray(direct["epsilon_2"], dtype=float),
        g_real,
        times,
        n_boot=bootstrap,
        seed=seed_bootstrap,
    )
    C_inf = bootstrap_out["C_inferred"]
    eta_inf = bootstrap_out["eta_inferred"]
    C_ci = percentile_summary(C_inf)
    eta_ci = percentile_summary(eta_inf)

    print("\n[v17] Dephasing / covariance spectroscopy")
    print(
        f"  analytic T1/2(+)= {t_half_plus:.6e}, T1/2(-)= {t_half_minus:.6e}"
    )
    print(
        f"  kernel-ensemble T+={tplus_large:.6e}, T-={tminus_large:.6e}, "
        f"C_inferred={C_from_large_half:+.6e}"
    )
    print(
        f"  direct-BdG T+={tplus_direct:.6e}, T-={tminus_direct:.6e}, "
        f"C_inferred={C_from_direct_half:+.6e}"
    )
    print(
        f"  bootstrap C median={C_ci['q50']:+.6e}, "
        f"95% CI=[{C_ci['q2']:+.6e},{C_ci['q97']:+.6e}]"
    )

    coherence_errors = {
        "kernel_ensemble_vs_analytic_plus_max_abs": float(np.max(np.abs(kernel_emp_plus - analytic_plus))),
        "kernel_ensemble_vs_analytic_minus_max_abs": float(np.max(np.abs(kernel_emp_minus - analytic_minus))),
        "quad_mc_vs_analytic_plus_max_abs": float(np.max(np.abs(mc_quad_plus - analytic_plus))),
        "quad_mc_vs_analytic_minus_max_abs": float(np.max(np.abs(mc_quad_minus - analytic_minus))),
        "exact_gaussian_vs_quadratic_plus_max_abs": float(np.max(np.abs(mc_exact_plus - analytic_plus))),
        "exact_gaussian_vs_quadratic_minus_max_abs": float(np.max(np.abs(mc_exact_minus - analytic_minus))),
    }

    # I. Transfer-law scan ---------------------------------------------------
    geometry = None
    if do_geometry_scan:
        rho_values = np.linspace(-1.0, 1.0, geometry_rho_points)
        mu2_values = [sweet2.mu_star]
        geometry = compute_geometry_transfer_scan(
            L1=L1,
            L2_values=geometry_L2_values,
            alpha=alpha,
            Delta=Delta,
            Ez=Ez,
            t0=t0,
            mu1=sweet1.mu_star,
            mu2_values=mu2_values,
            W=W_noise,
            rho_values=rho_values,
            xi_noise_values=geometry_xi_values,
            kernel_step=kernel_step,
            mu_scan_min=mu_scan_min,
            mu_scan_max=mu_scan_max,
            mu_scan_points=mu_scan_points,
        )

    # J. Save figures ---------------------------------------------------------
    # Figure 1: operating-point scan and kernels.
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    ax.semilogy(sweet1.scan_mu, np.abs(sweet1.scan_epsilon) + 1e-16, "o-", label="wire 1 |signed splitting|")
    ax.semilogy(sweet2.scan_mu, np.abs(sweet2.scan_epsilon) + 1e-16, "s-", label="wire 2 |signed splitting|")
    ax.axvline(sweet1.mu_star, ls=":", linewidth=0.9, label=fr"$\mu_1^*={sweet1.mu_star:.4f}$")
    ax.axvline(sweet2.mu_star, ls="-.", linewidth=0.9, label=fr"$\mu_2^*={sweet2.mu_star:.4f}$")
    ax.set_xlabel(r"chemical potential $\mu$")
    ax.set_ylabel(r"$|\epsilon_M|$")
    ax.set_title("v17: intrawire operating-point search")
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig1_v17_operating_point_and_kernels.png"), dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    ax.plot(np.arange(L1), K1, label=r"$K_{12}(x)$")
    ax.plot(np.arange(L2), K2, label=r"$K_{34}(x)$")
    ax.axhline(0.0, ls=":", linewidth=0.8)
    ax.set_xlabel("site index")
    ax.set_ylabel(r"$K_a(x)=\partial\epsilon_a/\partial\mu_x$")
    ax.set_title(fr"Noise-transfer kernels; $\chi_{{geom}}={kc.geometry_factor:+.3f}$")
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig1b_v17_sensitivity_kernels.png"), dpi=220)
    plt.close(fig)

    # Figure 2: direct-vs-kernel covariance summary.
    labels = ["sigma12", "sigma34", "C"]
    kernel_vals = np.array([kc.sigma12, kc.sigma34, kc.covariance], dtype=float)
    direct_vals = np.array([direct["sigma12"], direct["sigma34"], direct["covariance"]], dtype=float)
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.bar(x - width / 2, kernel_vals, width, label="kernel prediction")
    ax.bar(x + width / 2, direct_vals, width, label="direct BdG")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("value")
    ax.set_title("v17: full-BdG vs kernel covariance transfer")
    ax.grid(True, axis="y", ls=":", alpha=0.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig2_v17_direct_vs_kernel_covariance.png"), dpi=220)
    plt.close(fig)

    # Figure 3: W scaling. Use log-log plot for sigma/covariance.
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    ws = np.array([r["W"] for r in wscale["rows"]], dtype=float)
    s1d = np.array([r["direct_sigma12"] for r in wscale["rows"]], dtype=float)
    s2d = np.array([r["direct_sigma34"] for r in wscale["rows"]], dtype=float)
    cd = np.abs(np.array([r["direct_covariance"] for r in wscale["rows"]], dtype=float))
    ax.loglog(ws, s1d, "o-", label=r"direct $\sigma_{12}$")
    ax.loglog(ws, s2d, "s-", label=r"direct $\sigma_{34}$")
    ax.loglog(ws, cd, "^-", label=r"direct $|C_{12,34}|$")
    ax.set_xlabel(r"disorder RMS $W$")
    ax.set_ylabel("magnitude")
    ax.set_title("v17: full-BdG disorder-strength scaling")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig3_v17_W_scaling_falsification.png"), dpi=220)
    plt.close(fig)

    # Figure 4: rho_site -> rho_epsilon and eta for the nominal L2/xi.
    rho_dense = np.linspace(-1.0, 1.0, 101)
    rho_eps_dense = []
    eta_dense = []
    for r in rho_dense:
        kcr = kernel_covariance(K1, K2, W_noise, float(r), xi_noise)
        rho_eps_dense.append(kcr.correlation)
        eta_dense.append(kcr.eta)
    fig, ax = plt.subplots(figsize=(8.3, 5.2))
    ax.plot(rho_dense, rho_dense, "--", label=r"identity $\rho_\epsilon=\rho_{site}$")
    ax.plot(rho_dense, rho_eps_dense, label=r"kernel prediction")
    ax.plot(rho_dense, eta_dense, ":", label=r"$\eta$")
    ax.axhline(0.0, ls="--", linewidth=0.7)
    ax.axvline(0.0, ls="--", linewidth=0.7)
    ax.set_xlabel(r"microscopic correlation $\rho_{site}$")
    ax.set_ylabel("dimensionless transfer")
    ax.set_title(fr"v17: correlation transfer at $L_1={L1}$, $L_2={L2}$, $\xi={xi_noise}$")
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig4_v17_rho_transfer.png"), dpi=220)
    plt.close(fig)

    # Figure 5: geometry design map for the scanned xi=0 case if present.
    if geometry is not None:
        recs = geometry["records"]
        xi0 = min(geometry_xi_values, key=lambda x: abs(float(x) - 0.0))
        rec0 = [r for r in recs if abs(float(r["xi_noise"]) - float(xi0)) < 1e-12 and abs(float(r["rho_site"]) - float(rho_site)) < 1e-12]
        if rec0:
            byL = sorted(set(int(r["L2"]) for r in rec0))
            J = [float(next(r["J_geom"] for r in rec0 if int(r["L2"]) == L)) for L in byL]
            chi = [float(next(r["geometry_factor"] for r in rec0 if int(r["L2"]) == L)) for L in byL]
            fig, ax = plt.subplots(figsize=(8.3, 5.2))
            ax.plot(byL, J, "o-", label=r"$J_{geom}=\max(S_+,S_-)/W^2$")
            ax.plot(byL, np.abs(chi), "s--", label=r"$|\chi_{geom}|$")
            ax.set_xlabel(r"second-wire length $L_2$")
            ax.set_ylabel("dimensionless metric")
            ax.set_title("v17: geometry-dependent noise-transfer design metric")
            ax.grid(True, ls=":", alpha=0.5)
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(os.path.join(outdir, "fig5_v17_geometry_design_map.png"), dpi=220)
            plt.close(fig)

    # Figure 6: dephasing + inversion.
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    ax.plot(times, analytic_plus, label=r"analytic $p=+1$")
    ax.plot(times, analytic_minus, label=r"analytic $p=-1$")
    ax.plot(times[::12], kernel_emp_plus[::12], "o", ms=3, label="1M kernel ensemble +")
    ax.plot(times[::12], kernel_emp_minus[::12], "x", ms=3, label="1M kernel ensemble -")
    ax.plot(times[::14], direct_coh_plus[::14], ".", ms=4, label="direct BdG +")
    ax.plot(times[::14], direct_coh_minus[::14], "+", ms=4, label="direct BdG -")
    ax.axhline(0.5, ls="--", linewidth=0.8)
    ax.set_xlabel("time")
    ax.set_ylabel(r"$|W_p(t)|$")
    ax.set_ylim(0, 1.03)
    ax.set_title("v17: sweet-spot dephasing and covariance inversion")
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig6_v17_dephasing_and_inversion.png"), dpi=220)
    plt.close(fig)

    # Figure 7: clean finite-size splitting for context.
    lengths = list(range(max(30, min(L1, L2) - 20), max(L1, L2) + 31, 5))
    split_scan = []
    for L in lengths:
        w = BdGWire(
            BdGParams(
                L=L,
                alpha=alpha,
                Delta=Delta,
                Ez=Ez,
                t0=t0,
                mu=sweet1.mu_star if abs(L - L1) <= abs(L - L2) else sweet2.mu_star,
            )
        )
        ref = w.reference_majorana_basis()
        split_scan.append(abs(float(w.signed_epsilon(None, reference_basis=ref))))
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.semilogy(lengths, split_scan, "o-")
    ax.axvline(L1, ls="--", linewidth=0.9, label=f"L1={L1}")
    ax.axvline(L2, ls=":", linewidth=0.9, label=f"L2={L2}")
    ax.set_xlabel("wire length L")
    ax.set_ylabel(r"$|\epsilon_M|$")
    ax.set_title("v17: clean finite-size splitting around the chosen geometries")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig7_v17_clean_length_scan.png"), dpi=220)
    plt.close(fig)

    # K. Save arrays ----------------------------------------------------------
    np.savez(
        os.path.join(outdir, "sensitivity_kernels_v17.npz"),
        K12=K1,
        K34=K2,
        L1=np.array([L1]),
        L2=np.array([L2]),
        mu1_star=np.array([sweet1.mu_star]),
        mu2_star=np.array([sweet2.mu_star]),
        kernel_step=np.array([kernel_step]),
    )

    # L. Compact scientific status -------------------------------------------
    transfer_claim_conditions = {
        "linear_response": "supported only locally; quantified by direct-vs-kernel RMSE/correlation",
        "Gaussian_quasistatic": "assumed for closed-form envelope, not proved by direct BdG",
        "site_noise_model": "paired Gaussian with separable cross-covariance rho_site * W^2 * R12",
        "sweet_spot": "searched numerically in mu; result tagged exact root or minimum_abs_epsilon",
        "bridge_g": "external effective control parameter",
    }

    direct_target_comparison = {
        "sigma12_relative_error": relative_error(float(direct["sigma12"]), kc.sigma12),
        "sigma34_relative_error": relative_error(float(direct["sigma34"]), kc.sigma34),
        "covariance_relative_error": relative_error(float(direct["covariance"]), kc.covariance),
        "rho_epsilon_absolute_error": float(direct["correlation"] - kc.correlation),
    }

    scientific_readout = {
        "sweet_spot_operating_points_found": bool(sweet1.residual_abs_epsilon < max(1e-7, 0.01 * min(sweet1.bulk_gap, 1.0)))
        and bool(sweet2.residual_abs_epsilon < max(1e-7, 0.01 * min(sweet2.bulk_gap, 1.0))),
        "both_wires_topological": bool(sweet1.topological and sweet2.topological),
        "kernel_covariance_closed_form_verified": bool(abs(cross_direct - kc.covariance) < 1e-12 * max(abs(kc.covariance), 1e-15)),
        "direct_full_BdG_agreement": direct_target_comparison,
        "W_scaling": {
            "sigma12_exponent": wscale["power_law_exponent_sigma12"],
            "sigma34_exponent": wscale["power_law_exponent_sigma34"],
            "covariance_abs_exponent": wscale["power_law_exponent_covariance_abs"],
        },
        "design_metric": {
            "J_geom": kc.J_geom,
            "chi_common": kc.chi_common,
            "chi_differential": kc.chi_differential,
        },
    }

    summary = {
        "version": "v17",
        "status": "tuned-sweetspot_BdG_kernel_transfer_falsification_and_geometry_design_metric",
        "core_rules": {
            "effective_hamiltonian": "H_p = 1/2 [g X - (epsilon_12 + p epsilon_34) Z]",
            "noise_coordinate": "delta z_p = -(delta_e12 + p delta_e34)",
            "variance_rule": "S_p = sigma_12^2 + sigma_34^2 + 2 p C_12,34",
            "quadratic_envelope": "|W_p(t)| = [1 + (S_p t/g)^2]^(-1/4)",
            "half_life": "T_1/2,p = sqrt(15) g / S_p",
            "covariance_spectroscopy": "C_12,34 = sqrt(15) g / 4 * [1/T_+ - 1/T_-]",
            "parity_lifetime_asymmetry": "eta = -2 C_12,34 / (sigma_12^2 + sigma_34^2)",
            "microscopic_transfer": "delta epsilon_a = K_a^T delta mu",
            "microscopic_covariance": "C_12,34 = K_12^T Sigma_mu,12 K_34",
            "geometry_factor": "rho_epsilon = rho_site * chi_geom",
            "geometry_factor_definition": "chi_geom = (K1^T R12 K2) / sqrt[(K1^T R1 K1)(K2^T R2 K2)]",
            "design_metric": "J_geom = max(S_plus,S_minus)/W^2",
        },
        "scientific_readout": scientific_readout,
        "transfer_claim_conditions": transfer_claim_conditions,
        "operating_points": {
            "wire1": sweet1.as_dict(),
            "wire2": sweet2.as_dict(),
        },
        "algebra": {**alg, **proj},
        "derivative_checks": {"even": deriv_plus, "odd": deriv_minus},
        "disorder_generator_check": gen_check,
        "kernels": {
            "wire1": {**kd1, "K": K1.tolist()},
            "wire2": {**kd2, "K": K2.tolist()},
            "selected_sites_validation_wire1": kcheck1,
            "selected_sites_validation_wire2": kcheck2,
            "full_site_fd_check": full_fd_checks,
            "sigma12": kc.sigma12,
            "sigma34": kc.sigma34,
            "covariance": kc.covariance,
            "correlation": kc.correlation,
            "geometry_factor": kc.geometry_factor,
            "eta": kc.eta,
            "S_plus": kc.S_plus,
            "S_minus": kc.S_minus,
            "J_geom": kc.J_geom,
            "cross_K_Sigma_K_error": cross_direct - kc.covariance,
            "stacked_quadratic_form": float(cov_full),
            "expected_covariance_matrix": expected_cov_matrix.tolist(),
            "xi_noise": xi_noise,
        },
        "direct_bdg": {
            "sigma12": direct["sigma12"],
            "sigma34": direct["sigma34"],
            "covariance": direct["covariance"],
            "correlation": direct["correlation"],
            "wire1_moments": direct["wire1_moments"],
            "wire2_moments": direct["wire2_moments"],
            "gauge_score_wire1": direct["gauge_score_wire1"],
            "gauge_score_wire2": direct["gauge_score_wire2"],
            "linearity_test": linearity,
            "target_relative_errors": direct_rel_errors,
            "n_real": n_real,
            "W": W_noise,
            "rho_site": rho_site,
            "xi_noise": xi_noise,
        },
        "large_kernel_ensemble": {
            "n_real": kernel_ensemble,
            "sigma12": sigma12_large,
            "sigma34": sigma34_large,
            "covariance": cov_large,
            "correlation": rho_large,
            "relative_errors_vs_kernel": {
                "sigma12": relative_error(sigma12_large, kc.sigma12),
                "sigma34": relative_error(sigma34_large, kc.sigma34),
                "covariance": relative_error(cov_large, kc.covariance),
            },
        },
        "W_scaling_falsification": wscale,
        "coherence": {
            "g": g_real,
            "analytic_T_half_plus": t_half_plus,
            "analytic_T_half_minus": t_half_minus,
            "kernel_ensemble_T_plus": tplus_large,
            "kernel_ensemble_T_minus": tminus_large,
            "C_from_kernel_ensemble_half_lives": C_from_large_half,
            "eta_from_kernel_ensemble_half_lives": eta_from_large_half,
            "direct_BdG_estimated_T_plus_from_variance": t_direct_est_plus,
            "direct_BdG_estimated_T_minus_from_variance": t_direct_est_minus,
            "direct_BdG_T_plus": tplus_direct,
            "direct_BdG_T_minus": tminus_direct,
            "C_from_direct_BdG_half_lives": C_from_direct_half,
            "eta_from_direct_BdG_half_lives": eta_direct_half,
            "bootstrap_n": bootstrap,
            "bootstrap_C_95pct": C_ci,
            "bootstrap_eta_95pct": eta_ci,
            "errors": coherence_errors,
        },
        "geometry_scan": geometry,
        "clean_length_scan": {
            "lengths": lengths,
            "abs_signed_epsilon": split_scan,
        },
        "modeling_caveats": [
            "p=+/- are total-parity sectors; they are not automatically the two logical states of one fixed-parity tetron.",
            "The sweet spot is searched in finite-length BdG using the signed intrawire splitting; no claim is made if only a minimum_abs_epsilon fallback is found.",
            "The HF kernel is a local derivative. Full-site FD checks and direct-disorder scaling quantify its regime of validity.",
            "The Gaussian closed-form coherence assumes centered jointly Gaussian quasistatic coupling noise and sigma << g.",
            "Direct-BdG coherence here uses the extracted effective intrawire splittings plus externally specified bridge coupling g; it is not a full dynamical microscopic open-system simulation.",
            "The geometry law is exact for the paired separable Gaussian site-noise covariance used here; universality beyond that class remains an empirical question.",
            "The design metric J_geom is a device-screening metric, not a proof of lower logical error rate until a gate/readout error model is attached.",
            "No claim is made that this framework is experimentally validated by existing data without an explicit mapping from measured observables to the model parameters.",
        ],
    }

    with open(os.path.join(outdir, "core_results_v17.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 100)
    print("v17 SCIENTIFIC SUMMARY")
    print("=" * 100)
    print("Microscopic disorder -> tuned BdG kernels -> effective covariance -> parity dephasing")
    print(f"  mu1*                     = {sweet1.mu_star:+.8f} ({sweet1.method})")
    print(f"  mu2*                     = {sweet2.mu_star:+.8f} ({sweet2.method})")
    print(f"  epsilon1*, epsilon2*     = {sweet1.epsilon_star:+.3e}, {sweet2.epsilon_star:+.3e}")
    print(f"  topological wires         = {sweet1.topological} / {sweet2.topological}")
    print(f"  chi_geom                 = {kc.geometry_factor:+.6f}")
    print(f"  rho_site                 = {rho_site:+.3f}")
    print(f"  rho_epsilon (kernel)     = {kc.correlation:+.6f}")
    print(f"  rho_epsilon (direct)     = {direct['correlation']:+.6f}")
    print(f"  C12,34 kernel/direct     = {kc.covariance:+.6e} / {direct['covariance']:+.6e}")
    print(f"  eta kernel               = {kc.eta:+.6f}")
    print(f"  J_geom                  = {kc.J_geom:.6e}")
    print(f"  sigma W exponents        = {wscale['power_law_exponent_sigma12']:.3f}, {wscale['power_law_exponent_sigma34']:.3f}")
    print(f"  |C| W exponent           = {wscale['power_law_exponent_covariance_abs']:.3f}")
    print(f"  C from kernel half-lives = {C_from_large_half:+.6e}")
    print(f"  C from direct half-lives = {C_from_direct_half:+.6e}")
    print(f"\n[OUTPUTS] -> {outdir}/")
    return summary


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_int_list(spec: str) -> Tuple[int, ...]:
    vals: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if part:
            vals.append(int(part))
    if not vals:
        raise ValueError("empty integer list")
    return tuple(vals)


def parse_float_list(spec: str) -> Tuple[float, ...]:
    vals: List[float] = []
    for part in spec.split(","):
        part = part.strip()
        if part:
            vals.append(float(part))
    if not vals:
        raise ValueError("empty float list")
    return tuple(vals)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Majorana tetron v17: tuned sweet spots, BdG kernel falsification, covariance transfer, and geometry design metric"
    )
    parser.add_argument("--outdir", default="outputs_v17")
    parser.add_argument("--mc", type=int, default=100_000)
    parser.add_argument("--n-real", type=int, default=80,
                        help="direct finite-BdG realizations for the nominal point")
    parser.add_argument("--kernel-ensemble", type=int, default=1_000_000,
                        help="large vectorized linearized ensemble")
    parser.add_argument("--bootstrap", type=int, default=300)
    parser.add_argument("--L1", type=int, default=80)
    parser.add_argument("--L2", type=int, default=65)
    parser.add_argument("--W", type=float, default=0.03)
    parser.add_argument("--g", type=float, default=0.120)
    parser.add_argument("--rho-site", type=float, default=0.60)
    parser.add_argument("--xi-noise", type=float, default=0.0)
    parser.add_argument("--alpha", type=float, default=0.15)
    parser.add_argument("--Delta", type=float, default=1.0)
    parser.add_argument("--Ez", type=float, default=2.5)
    parser.add_argument("--t0", type=float, default=1.0)
    parser.add_argument("--mu-initial-1", type=float, default=0.50)
    parser.add_argument("--mu-initial-2", type=float, default=0.50)
    parser.add_argument("--mu-scan-min", type=float, default=0.0)
    parser.add_argument("--mu-scan-max", type=float, default=1.5)
    parser.add_argument("--mu-scan-points", type=int, default=19)
    parser.add_argument("--kernel-step", type=float, default=2e-4)
    parser.add_argument("--seed-direct", type=int, default=7000)
    parser.add_argument("--seed-kernel", type=int, default=17000)
    parser.add_argument("--seed-bootstrap", type=int, default=27000)
    parser.add_argument("--W-scaling-points", type=int, default=4)
    parser.add_argument("--no-geometry-scan", action="store_true")
    parser.add_argument("--geometry-L2", default="55,65,75,85")
    parser.add_argument("--geometry-rho-points", type=int, default=11)
    parser.add_argument("--geometry-xi", default="0.0,1.0,3.0")
    args = parser.parse_args()

    geometry_L2 = parse_int_list(args.geometry_L2)
    geometry_xi = parse_float_list(args.geometry_xi)
    run_core_demonstration(
        outdir=args.outdir,
        mc=args.mc,
        n_real=args.n_real,
        kernel_ensemble=args.kernel_ensemble,
        L1=args.L1,
        L2=args.L2,
        W_noise=args.W,
        g_real=args.g,
        rho_site=args.rho_site,
        xi_noise=args.xi_noise,
        alpha=args.alpha,
        Delta=args.Delta,
        Ez=args.Ez,
        t0=args.t0,
        mu_initial_1=args.mu_initial_1,
        mu_initial_2=args.mu_initial_2,
        mu_scan_min=args.mu_scan_min,
        mu_scan_max=args.mu_scan_max,
        mu_scan_points=args.mu_scan_points,
        kernel_step=args.kernel_step,
        bootstrap=args.bootstrap,
        seed_direct=args.seed_direct,
        seed_kernel=args.seed_kernel,
        seed_bootstrap=args.seed_bootstrap,
        W_scaling_points=args.W_scaling_points,
        do_geometry_scan=not args.no_geometry_scan,
        geometry_L2_values=geometry_L2,
        geometry_rho_points=args.geometry_rho_points,
        geometry_xi_values=geometry_xi,
    )


if __name__ == "__main__":
    main()
