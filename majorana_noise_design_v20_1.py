#!/usr/bin/env python3
"""Majorana-noise geometry v20.1

Core idea
---------
    geometry -> Majorana response K -> Cbar = K R K^T -> noise eigenmodes

For microscopic disorder covariance Sigma_mu = W^2 R,

    Sigma_epsilon = W^2 Cbar,
    Cbar = K R K^T.

The central gauge-robust observables are the eigenvalues of Cbar.  The main
scalar design metric is

    N_geom = lambda_max(Cbar).

The same 2x2 covariance gives physical parity-sector noise strengths

    S_p = v_p^T Sigma_epsilon v_p,   v_p=(1,p).

v20.1 adds an explicit Majorana-separation constraint to matched-geometry
searches.  The goal is to test whether effective-noise differences survive
when bulk gap, Majorana localization, AND Majorana separation are all matched.

The main pipeline deliberately excludes lifetime inversion, W^4 fitting,
bootstrap, antithetic estimators, and large Monte-Carlo ensembles.  Full-BdG
sampling is retained only as an applicability check for the linear-response
law.

Outputs
-------
  core_results_v20_1.json
  kernels_v20_1.npz
  operating_point_v20_1.png
  noise_modes_v20_1.png
  geometry_design_v20_1.png
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
from scipy.optimize import minimize_scalar


# -----------------------------------------------------------------------------
# Linear algebra
# -----------------------------------------------------------------------------


def paulis() -> Dict[str, np.ndarray]:
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], complex)
    Y = np.array([[0, -1j], [1j, 0]], complex)
    Z = np.array([[1, 0], [0, -1]], complex)
    return {"I": I, "X": X, "Y": Y, "Z": Z}


def dagger(a: np.ndarray) -> np.ndarray:
    return a.conj().T


def cov(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float) - np.mean(a)
    b = np.asarray(b, float) - np.mean(b)
    return float(a @ b / max(a.size - 1, 1))


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    sa = float(np.std(a, ddof=1))
    sb = float(np.std(b, ddof=1))
    return float(cov(a, b) / (sa * sb)) if sa > 0 and sb > 0 else float("nan")


def relative_difference(a: float, b: float) -> float:
    return float(abs(a - b) / max(abs(a), abs(b), 1e-30))


# -----------------------------------------------------------------------------
# Finite spinful Rashba BdG model
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class BdGParams:
    L: int
    alpha: float = 0.15
    Delta: float = 1.0
    Ez: float = 2.5
    t0: float = 1.0
    mu: float = 0.5


@dataclass(frozen=True)
class OperatingPoint:
    L: int
    alpha: float
    mu: float
    epsilon: float
    E_low: float
    E_gap: float
    z2: int
    x_left: float
    x_right: float
    xi_left: float
    xi_right: float
    xi_mean: float
    separation: float

    def as_dict(self) -> Dict[str, float | int]:
        return asdict(self)


class BdGWire:
    """Finite spinful Rashba nanowire with s-wave pairing."""

    def __init__(self, p: BdGParams):
        self.p = p
        P = paulis()
        self.I2, self.sx, self.sy = P["I"], P["X"], P["Y"]
        self.tx, self.ty, self.tz = P["X"], P["Y"], P["Z"]
        self.U_C = np.kron(self.ty, self.sy)

    def bloch_H(self, k: float) -> np.ndarray:
        p = self.p
        xi = 2 * p.t0 - 2 * p.t0 * np.cos(k) - p.mu
        return (
            xi * np.kron(self.tz, self.I2)
            + p.alpha * np.sin(k) * np.kron(self.tz, self.sy)
            + p.Ez * np.kron(self.I2, self.sx)
            + p.Delta * np.kron(self.tx, self.I2)
        )

    def bulk_gap(self, nk: int = 121) -> float:
        ks = np.linspace(-np.pi, np.pi, nk)
        return float(min(np.min(np.abs(np.linalg.eigvalsh(self.bloch_H(k)))) for k in ks))

    @staticmethod
    def pf4(A: np.ndarray) -> complex:
        return A[0, 1] * A[2, 3] - A[0, 2] * A[1, 3] + A[0, 3] * A[1, 2]

    def z2(self) -> int:
        b0 = self.bloch_H(0.0) @ self.U_C
        bp = self.bloch_H(np.pi) @ self.U_C
        q = float(np.real(self.pf4(b0) * self.pf4(bp)))
        if abs(q) < 1e-12:
            raise RuntimeError("Pfaffian product too close to zero")
        return -1 if q < 0 else 1

    def finite_H(self, disorder: Optional[np.ndarray] = None) -> np.ndarray:
        p = self.p
        if disorder is not None:
            disorder = np.asarray(disorder, float)
            if disorder.shape != (p.L,):
                raise ValueError(f"disorder must have shape ({p.L},)")
        tzI = np.kron(self.tz, self.I2)
        H0 = (
            (2 * p.t0 - p.mu) * tzI
            + p.Ez * np.kron(self.I2, self.sx)
            + p.Delta * np.kron(self.tx, self.I2)
        )
        hop = -p.t0 * tzI - 0.5j * p.alpha * np.kron(self.tz, self.sy)
        H = np.zeros((4 * p.L, 4 * p.L), complex)
        for i in range(p.L):
            s = slice(4 * i, 4 * (i + 1))
            site = H0.copy()
            if disorder is not None:
                site -= float(disorder[i]) * tzI
            H[s, s] = site
            if i + 1 < p.L:
                t = slice(4 * (i + 1), 4 * (i + 2))
                H[s, t] = hop
                H[t, s] = dagger(hop)
        return 0.5 * (H + dagger(H))

    def central(self, disorder: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        H = self.finite_H(disorder)
        n = H.shape[0]
        mid = n // 2
        lo, hi = max(0, mid - 5), min(n - 1, mid + 4)
        return la.eigh(H, subset_by_index=[lo, hi])

    def majoranas(self, disorder: Optional[np.ndarray] = None) -> Dict[str, object]:
        H = self.finite_H(disorder)
        vals, vecs = self.central(disorder)
        pos = np.where(vals > 1e-12)[0]
        if len(pos) == 0:
            raise RuntimeError("No positive low-energy state")
        psi = vecs[:, pos[0]]
        Uc = np.kron(np.eye(self.p.L), self.U_C)
        psi_h = Uc @ psi.conj()
        psi_h /= max(la.norm(psi_h), 1e-15)
        G = np.column_stack(((psi + psi_h) / np.sqrt(2), -1j * (psi - psi_h) / np.sqrt(2)))

        x = np.repeat(np.arange(self.p.L, dtype=float), 4)
        xmat = np.real(G.conj().T @ (x[:, None] * G))
        _, U = la.eigh(0.5 * (xmat + xmat.T))
        M = G @ U
        centers = np.array([float(np.real(np.vdot(M[:, j], x * M[:, j]))) for j in range(2)])
        if centers[0] > centers[1]:
            M = M[:, ::-1]
            centers = centers[::-1]
        for j in range(2):
            k = int(np.argmax(np.abs(M[:, j])))
            if np.real(M[k, j]) < 0:
                M[:, j] *= -1
        widths = np.array([
            math.sqrt(max(0.0, float(np.real(np.vdot(M[:, j], (x - centers[j]) ** 2 * M[:, j])))))
            for j in range(2)
        ])
        eps = float(np.imag((M.conj().T @ H @ M)[0, 1]))
        return {
            "M": M,
            "epsilon": eps,
            "E_low": float(vals[pos[0]]),
            "E_gap": self.bulk_gap(),
            "z2": self.z2(),
            "centers": centers,
            "widths": widths,
        }

    @staticmethod
    def align(reference: np.ndarray, current: np.ndarray) -> np.ndarray:
        U, _, Vh = la.svd(np.real(reference.conj().T @ current))
        return U @ Vh

    def epsilon(self, disorder: Optional[np.ndarray], reference: np.ndarray) -> float:
        d = self.majoranas(disorder)
        M = np.asarray(d["M"], complex)
        M = M @ self.align(reference, M)
        return float(np.imag((M.conj().T @ self.finite_H(disorder) @ M)[0, 1]))


# -----------------------------------------------------------------------------
# Operating point and response kernel
# -----------------------------------------------------------------------------


def make_wire(template: BdGWire, mu: float) -> BdGWire:
    p = template.p
    return BdGWire(BdGParams(p.L, p.alpha, p.Delta, p.Ez, p.t0, float(mu)))


def tune_operating_point(
    template: BdGWire,
    mu_min: float,
    mu_max: float,
    points: int = 11,
) -> Tuple[BdGWire, np.ndarray, OperatingPoint]:
    if points < 5:
        raise ValueError("mu-points must be >= 5")
    if not mu_min < mu_max:
        raise ValueError("mu-min must be < mu-max")

    mus = np.linspace(mu_min, mu_max, points)
    E = np.array([make_wire(template, mu).majoranas()["E_low"] for mu in mus], float)
    i = int(np.argmin(E))
    a = float(mus[max(0, i - 1)])
    b = float(mus[min(points - 1, i + 1)])
    if a == b:
        a, b = mu_min, mu_max

    res = minimize_scalar(
        lambda mu: float(make_wire(template, mu).majoranas()["E_low"]),
        bounds=(a, b), method="bounded", options={"xatol": 1e-8},
    )
    wire = make_wire(template, float(res.x))
    d = wire.majoranas()
    ref = np.asarray(d["M"], complex)
    eps = float(wire.epsilon(None, ref))
    c = np.asarray(d["centers"], float)
    w = np.asarray(d["widths"], float)
    op = OperatingPoint(
        L=wire.p.L, alpha=wire.p.alpha, mu=float(res.x), epsilon=eps,
        E_low=float(d["E_low"]), E_gap=float(d["E_gap"]), z2=int(d["z2"]),
        x_left=float(c[0]), x_right=float(c[1]),
        xi_left=float(w[0]), xi_right=float(w[1]), xi_mean=float(np.mean(w)),
        separation=float(c[1] - c[0]),
    )
    return wire, ref, op


def majorana_kernel(wire: BdGWire, reference: np.ndarray) -> np.ndarray:
    d = wire.majoranas()
    M = np.asarray(d["M"], complex)
    M = M @ wire.align(reference, M)
    tzI = np.kron(wire.tz, wire.I2)
    K = np.zeros(wire.p.L)
    for i in range(wire.p.L):
        s = slice(4 * i, 4 * (i + 1))
        Mi = M[s, :]
        K[i] = float(np.imag((Mi.conj().T @ (-tzI) @ Mi)[0, 1]))
    return K


def kernel_fd_check(
    wire: BdGWire,
    reference: np.ndarray,
    K: np.ndarray,
    sites: Sequence[int],
    step: float,
) -> float:
    errs = []
    zero = np.zeros(wire.p.L)
    for i in sites:
        dp = zero.copy(); dm = zero.copy()
        dp[int(i)] = step; dm[int(i)] = -step
        fd = (wire.epsilon(dp, reference) - wire.epsilon(dm, reference)) / (2 * step)
        errs.append(abs(float(fd) - float(K[int(i)])) / max(abs(float(fd)), 1e-30))
    return float(max(errs, default=0.0))


# -----------------------------------------------------------------------------
# Microscopic noise covariance -> effective 2x2 covariance
# -----------------------------------------------------------------------------


def spatial_R(L: int, xi: float) -> np.ndarray:
    x = np.arange(L, dtype=float)
    if xi <= 0:
        return np.eye(L)
    return np.exp(-np.abs(x[:, None] - x[None, :]) / float(xi))


def cross_R(L1: int, L2: int, xi: float) -> np.ndarray:
    a = np.arange(L1, dtype=float)[:, None]
    b = np.arange(L2, dtype=float)[None, :]
    if xi <= 0:
        return np.eye(L1, L2)
    return np.exp(-np.abs(a - b) / float(xi))


def noise_R(L1: int, L2: int, rho: float, xi: float) -> np.ndarray:
    if not -1.0 <= rho <= 1.0:
        raise ValueError("rho-site must lie in [-1,1]")
    R12 = rho * cross_R(L1, L2, xi)
    R = np.block([[spatial_R(L1, xi), R12], [R12.T, spatial_R(L2, xi)]])
    return 0.5 * (R + R.T)


def effective_noise(K1: np.ndarray, K2: np.ndarray, R: np.ndarray) -> Dict[str, np.ndarray | float]:
    K1 = np.asarray(K1, float)
    K2 = np.asarray(K2, float)
    K = np.block([
        [K1[None, :], np.zeros((1, K2.size))],
        [np.zeros((1, K1.size)), K2[None, :]],
    ])
    C = 0.5 * (K @ R @ K.T + (K @ R @ K.T).T)
    vals, vecs = la.eigh(C)
    vp = np.array([1.0, 1.0])
    vm = np.array([1.0, -1.0])
    lam_weak, lam_strong = float(vals[0]), float(vals[-1])
    return {
        "K": K,
        "Cbar": C,
        "eigenvalues": vals,
        "eigenvectors": vecs,
        "S_plus_bar": float(vp @ C @ vp),
        "S_minus_bar": float(vm @ C @ vm),
        "lambda_weak": lam_weak,
        "lambda_strong": lam_strong,
        "mode_ratio": float(lam_strong / max(lam_weak, 1e-30)),
        "N_geom": lam_strong,
    }


# -----------------------------------------------------------------------------
# Full-BdG applicability check
# -----------------------------------------------------------------------------


def sample_disorder(
    L1: int,
    L2: int,
    rho: float,
    xi: float,
    n: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    R = noise_R(L1, L2, rho, xi)
    # The specified block covariance should be PSD for |rho|<=1 for this model.
    L = la.cholesky(R + 1e-11 * np.eye(L1 + L2), lower=True)
    z = np.random.default_rng(seed).normal(size=(n, L1 + L2)) @ L.T
    return z[:, :L1], z[:, L1:]


def nonlinear_window(
    wire1: BdGWire,
    wire2: BdGWire,
    ref1: np.ndarray,
    ref2: np.ndarray,
    Cbar: np.ndarray,
    rho: float,
    xi: float,
    W_values: Sequence[float],
    n: int,
    seed: int,
) -> Dict[str, object]:
    if n < 2:
        raise ValueError("validation-real must be >= 2")
    W_values = sorted(set(float(w) for w in W_values if float(w) > 0))
    if not W_values:
        raise ValueError("validation-W must contain at least one positive value")

    d1u, d2u = sample_disorder(wire1.p.L, wire2.p.L, rho, xi, n, seed)
    rows = []
    for W in W_values:
        e1 = np.empty(n)
        e2 = np.empty(n)
        for j in range(n):
            e1[j] = wire1.epsilon(W * d1u[j], ref1)
            e2[j] = wire2.epsilon(W * d2u[j], ref2)
        E = np.column_stack((e1 - e1.mean(), e2 - e2.mean()))
        Cdir = (E.T @ E) / max(n - 1, 1)
        target = W * W * Cbar
        denom = max(la.norm(Cdir, "fro"), 1e-30)
        eta = float(la.norm(Cdir - target, "fro") / denom)
        rows.append({
            "W": float(W),
            "eta_nl": eta,
            "C_direct": Cdir.tolist(),
            "C_target": target.tolist(),
            "rho_epsilon": corr(e1, e2),
            "n_real": int(n),
        })
    return {"rows": rows}


# -----------------------------------------------------------------------------
# Geometry scan and matched-pair search
# -----------------------------------------------------------------------------


def scan_geometry(
    L_values: Sequence[int],
    alpha_values: Sequence[float],
    Delta: float,
    Ez: float,
    t0: float,
    mu_min: float,
    mu_max: float,
    mu_points: int,
    K_reference: np.ndarray,
    L_reference: int,
    rho: float,
    xi: float,
) -> Sequence[Dict[str, float]]:
    records = []
    for alpha in alpha_values:
        for L in L_values:
            try:
                template = BdGWire(BdGParams(int(L), float(alpha), Delta, Ez, t0, 0.5))
                wire, ref, op = tune_operating_point(template, mu_min, mu_max, mu_points)
                if op.z2 != -1:
                    continue
                K = majorana_kernel(wire, ref)
                R = noise_R(L_reference, int(L), rho, xi)
                tr = effective_noise(K_reference, K, R)
                vals = np.asarray(tr["eigenvalues"], float)
                records.append({
                    "L": int(L),
                    "alpha": float(alpha),
                    "mu": float(op.mu),
                    "E_gap": float(op.E_gap),
                    "E_low": float(op.E_low),
                    "xi_mean": float(op.xi_mean),
                    "separation": float(op.separation),
                    "N_geom": float(tr["N_geom"]),
                    "lambda_weak": float(vals[0]),
                    "lambda_strong": float(vals[-1]),
                    "mode_ratio": float(tr["mode_ratio"]),
                })
            except (RuntimeError, ValueError, la.LinAlgError):
                continue
    return records


def matched_pair(
    records: Sequence[Dict[str, float]],
    gap_tol: float,
    xi_tol: float,
    separation_tol: float,
) -> Optional[Dict[str, object]]:
    if gap_tol <= 0 or xi_tol <= 0 or separation_tol <= 0:
        raise ValueError("match tolerances must be positive")

    best = None
    best_score = -np.inf
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            a, b = records[i], records[j]
            gd = relative_difference(a["E_gap"], b["E_gap"])
            xd = relative_difference(a["xi_mean"], b["xi_mean"])
            sd = relative_difference(a["separation"], b["separation"])
            if gd > gap_tol or xd > xi_tol or sd > separation_tol:
                continue
            ratio = max(a["N_geom"], b["N_geom"]) / max(min(a["N_geom"], b["N_geom"]), 1e-30)
            # Prefer a large noise separation, but mildly penalize imperfect matching.
            mismatch = (gd / gap_tol) + (xd / xi_tol) + (sd / separation_tol)
            score = math.log(max(ratio, 1.0)) - 0.10 * mismatch
            if score > best_score:
                best_score = score
                better = a if a["N_geom"] >= b["N_geom"] else b
                quieter = b if better is a else a
                best = {
                    "A": a,
                    "B": b,
                    "relative_gap_difference": float(gd),
                    "relative_xi_difference": float(xd),
                    "relative_separation_difference": float(sd),
                    "N_geom_ratio": float(ratio),
                    "mode_ratio_A": float(a["mode_ratio"]),
                    "mode_ratio_B": float(b["mode_ratio"]),
                    "higher_noise_geometry": better,
                    "lower_noise_geometry": quieter,
                    "match_score": float(score),
                }
    return best


# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------


def make_figures(
    outdir: str,
    K1: np.ndarray,
    K2: np.ndarray,
    transfer: Dict[str, object],
    geometry: Sequence[Dict[str, float]],
    pair: Optional[Dict[str, object]],
) -> None:
    # Fig 1: response kernels
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.plot(np.arange(K1.size), np.abs(K1), label="|K12(x)|")
    ax.plot(np.arange(K2.size), np.abs(K2), label="|K34(x)|")
    ax.set_xlabel("site index")
    ax.set_ylabel("|d epsilon_M / d mu_x|")
    ax.set_title("Majorana noise-response geometry")
    ax.grid(True, ls=":", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "operating_point_v20_1.png"), dpi=220)
    plt.close(fig)

    # Fig 2: effective covariance and modes
    C = np.asarray(transfer["Cbar"])
    vals = np.asarray(transfer["eigenvalues"])
    vecs = np.asarray(transfer["eigenvectors"])
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    im = ax.imshow(C, origin="lower", aspect="equal", interpolation="nearest")
    ax.set_xticks([0, 1], ["epsilon12", "epsilon34"])
    ax.set_yticks([0, 1], ["epsilon12", "epsilon34"])
    ax.set_title(f"Cbar; lambda_weak={vals[0]:.2e}, lambda_strong={vals[-1]:.2e}")
    fig.colorbar(im, ax=ax, label="Cbar")
    # Draw eigenvector directions in the same 2D coordinate frame.
    scale = 0.45 * max(np.max(np.abs(C)), 1e-12) ** 0.5
    center = np.array([0.5, 0.5])
    for k in range(2):
        v = vecs[:, k]
        ax.plot(
            [center[0] - scale * v[0], center[0] + scale * v[0]],
            [center[1] - scale * v[1], center[1] + scale * v[1]],
            lw=2,
        )
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "noise_modes_v20_1.png"), dpi=220)
    plt.close(fig)

    # Fig 3: geometry metric with optional matched pair highlight
    if geometry:
        gap = np.array([r["E_gap"] for r in geometry])
        N = np.array([r["N_geom"] for r in geometry])
        xi = np.array([r["xi_mean"] for r in geometry])
        fig, ax = plt.subplots(figsize=(7.8, 5.0))
        sc = ax.scatter(gap, N, c=xi, s=45)
        if pair is not None:
            A = pair["A"]; B = pair["B"]
            ax.scatter([A["E_gap"], B["E_gap"]], [A["N_geom"], B["N_geom"]], s=110)
            ax.annotate("A", (A["E_gap"], A["N_geom"]), xytext=(5, 5), textcoords="offset points")
            ax.annotate("B", (B["E_gap"], B["N_geom"]), xytext=(5, -15), textcoords="offset points")
        ax.set_yscale("log")
        ax.set_xlabel("bulk gap E_gap")
        ax.set_ylabel("N_geom = lambda_strong(Cbar)")
        ax.set_title("Geometry-dependent effective noise")
        ax.grid(True, ls=":", alpha=0.4)
        fig.colorbar(sc, ax=ax, label="mean Majorana width xi_M")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "geometry_design_v20_1.png"), dpi=220)
        plt.close(fig)


# -----------------------------------------------------------------------------
# Main research run
# -----------------------------------------------------------------------------


def run(args: argparse.Namespace) -> Dict[str, object]:
    os.makedirs(args.outdir, exist_ok=True)

    # 1. Operating points.
    template1 = BdGWire(BdGParams(args.L1, args.alpha, args.Delta, args.Ez, args.t0, 0.5))
    template2 = BdGWire(BdGParams(args.L2, args.alpha, args.Delta, args.Ez, args.t0, 0.5))
    wire1, ref1, op1 = tune_operating_point(template1, args.mu_min, args.mu_max, args.mu_points)
    wire2, ref2, op2 = tune_operating_point(template2, args.mu_min, args.mu_max, args.mu_points)

    if op1.z2 != -1 or op2.z2 != -1:
        raise RuntimeError("At least one baseline wire is not in the topological phase")

    print("=" * 84)
    print("Majorana Noise Geometry v20.1")
    print("=" * 84)
    for tag, op in (("wire1", op1), ("wire2", op2)):
        print(
            f"{tag}: L={op.L}, alpha={op.alpha:.3f}, mu*={op.mu:.7f}, "
            f"gap={op.E_gap:.3e}, xi={op.xi_mean:.3e}, separation={op.separation:.3e}, Z2={op.z2}"
        )

    # 2. First-order kernels and finite-difference check.
    K1 = majorana_kernel(wire1, ref1)
    K2 = majorana_kernel(wire2, ref2)
    sites1 = np.unique(np.linspace(0, args.L1 - 1, min(6, args.L1), dtype=int))
    sites2 = np.unique(np.linspace(0, args.L2 - 1, min(6, args.L2), dtype=int))
    fd1 = kernel_fd_check(wire1, ref1, K1, sites1, args.fd_step)
    fd2 = kernel_fd_check(wire2, ref2, K2, sites2, args.fd_step)
    print(f"kernel FD relative error: {fd1:.3%} / {fd2:.3%}")

    # 3. Core effective covariance.
    R = noise_R(args.L1, args.L2, args.rho_site, args.xi_noise)
    transfer = effective_noise(K1, K2, R)
    vals = np.asarray(transfer["eigenvalues"], float)
    print(f"eigenvalues(Cbar): [{vals[0]:.8e} {vals[-1]:.8e}]")
    print(f"mode anisotropy lambda_strong/lambda_weak: {transfer['mode_ratio']:.4f}")
    print(f"N_geom: {transfer['N_geom']:.8e}")

    # 4. Full-BdG applicability window.
    validation = nonlinear_window(
        wire1, wire2, ref1, ref2, np.asarray(transfer["Cbar"]),
        args.rho_site, args.xi_noise, args.validation_W, args.validation_real, args.seed,
    )
    print("eta_nl(W):", [(r["W"], r["eta_nl"]) for r in validation["rows"]])

    # 5. Geometry search.
    geometry = scan_geometry(
        args.scan_lengths, args.scan_alpha,
        args.Delta, args.Ez, args.t0,
        args.mu_min, args.mu_max, args.mu_points,
        K1, args.L1, args.rho_site, args.xi_noise,
    )
    pair = matched_pair(
        geometry,
        args.gap_match_tol,
        args.xi_match_tol,
        args.separation_match_tol,
    )
    print(f"geometry candidates: {len(geometry)}")
    print(f"matched geometry pair (gap + xi + separation): {'YES' if pair else 'NO'}")
    if pair:
        print(
            f"  pair noise ratio = {pair['N_geom_ratio']:.6f}; "
            f"gap diff={pair['relative_gap_difference']:.3%}; "
            f"xi diff={pair['relative_xi_difference']:.3%}; "
            f"separation diff={pair['relative_separation_difference']:.3%}"
        )

    # 6. Figures.
    make_figures(args.outdir, K1, K2, transfer, geometry, pair)

    # 7. Save compact machine-readable report.
    report = {
        "version": "v20.1",
        "core": {
            "effective_hamiltonian": "H_p = 1/2 [g X - z_p Z], z_p=-(epsilon_12+p epsilon_34)",
            "linear_response": "delta epsilon = K delta mu",
            "normalized_noise_transfer": "Cbar = K R K^T",
            "physical_covariance": "Sigma_epsilon = W^2 Cbar",
            "noise_eigenmodes": "Cbar = Q diag(lambda_weak, lambda_strong) Q^T",
            "parity_projection": "S_p = v_p^T Sigma_epsilon v_p, v_p=(1,p)",
            "design_metric": "N_geom = lambda_strong(Cbar)",
        },
        "parameters": {
            k: getattr(args, k) for k in (
                "L1", "L2", "alpha", "Delta", "Ez", "t0",
                "rho_site", "xi_noise", "W", "mu_min", "mu_max",
                "mu_points", "fd_step", "validation_real",
                "gap_match_tol", "xi_match_tol", "separation_match_tol",
            )
        },
        "operating_points": {"wire1": op1.as_dict(), "wire2": op2.as_dict()},
        "kernels": {
            "K1": K1.tolist(),
            "K2": K2.tolist(),
            "finite_difference_relative_error": {"wire1": fd1, "wire2": fd2},
        },
        "effective_noise": {
            "Cbar": np.asarray(transfer["Cbar"]).tolist(),
            "eigenvalues": vals.tolist(),
            "eigenvectors": np.asarray(transfer["eigenvectors"]).tolist(),
            "lambda_weak": float(transfer["lambda_weak"]),
            "lambda_strong": float(transfer["lambda_strong"]),
            "mode_ratio": float(transfer["mode_ratio"]),
            "N_geom": float(transfer["N_geom"]),
            "S_plus_bar": float(transfer["S_plus_bar"]),
            "S_minus_bar": float(transfer["S_minus_bar"]),
            "physical_S_plus": float(args.W ** 2 * transfer["S_plus_bar"]),
            "physical_S_minus": float(args.W ** 2 * transfer["S_minus_bar"]),
        },
        "validation": validation,
        "geometry_scan": list(geometry),
        "matched_geometry_pair": pair,
        "scientific_readout": {
            "both_topological": bool(op1.z2 == -1 and op2.z2 == -1),
            "kernel_fd_supported": bool(fd1 < 0.08 and fd2 < 0.08),
            "matched_pair_found": pair is not None,
            "matching_constraints": {
                "relative_gap": float(args.gap_match_tol),
                "relative_xi": float(args.xi_match_tol),
                "relative_separation": float(args.separation_match_tol),
            },
            "universal_claim": False,
            "interpretation": (
                "A matched geometry pair exists under gap, localization, and separation constraints; "
                "this is a candidate geometry-level noise effect, not yet a universality claim."
                if pair is not None else
                "No pair met all matching constraints; broaden the geometry grid before drawing a design-law conclusion."
            ),
        },
        "notes": [
            "Cbar is the central gauge-robust effective covariance; its eigenvalues are invariant under Majorana sign flips.",
            "The parity sectors are projections of the same Cbar, not separate noise models.",
            "W is factored out of the geometry law; Cbar depends on geometry and normalized noise correlations.",
            "Full-BdG validation checks the range in which Sigma_epsilon ~= W^2 Cbar is accurate.",
            "The matched-pair test now constrains bulk gap, mean Majorana width, and Majorana separation simultaneously.",
            "Lifetime inversion, W^4 identification, bootstrap, and antithetic sampling are outside the core model.",
        ],
    }

    np.savez(
        os.path.join(args.outdir, "kernels_v20_1.npz"),
        K12=K1,
        K34=K2,
        L1=np.array([args.L1]),
        L2=np.array([args.L2]),
        mu1_star=np.array([op1.mu]),
        mu2_star=np.array([op2.mu]),
        Cbar=np.asarray(transfer["Cbar"]),
    )
    with open(os.path.join(args.outdir, "core_results_v20_1.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"outputs -> {args.outdir}/")
    return report


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_int_list(spec: str) -> Tuple[int, ...]:
    vals = tuple(int(x.strip()) for x in spec.split(",") if x.strip())
    if not vals:
        raise ValueError("empty integer list")
    return vals


def parse_float_list(spec: str) -> Tuple[float, ...]:
    vals = tuple(float(x.strip()) for x in spec.split(",") if x.strip())
    if not vals:
        raise ValueError("empty float list")
    return vals


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Majorana noise geometry v20.1: matched geometry + effective noise eigenmodes"
    )
    parser.add_argument("--outdir", default="outputs_v20_1")
    parser.add_argument("--L1", type=int, default=80)
    parser.add_argument("--L2", type=int, default=65)
    parser.add_argument("--W", type=float, default=0.03)
    parser.add_argument("--g", type=float, default=0.120, help="retained only for parameter provenance; not used in the core metric")
    parser.add_argument("--rho-site", type=float, default=0.60)
    parser.add_argument("--xi-noise", type=float, default=0.0)
    parser.add_argument("--alpha", type=float, default=0.15)
    parser.add_argument("--Delta", type=float, default=1.0)
    parser.add_argument("--Ez", type=float, default=2.5)
    parser.add_argument("--t0", type=float, default=1.0)
    parser.add_argument("--mu-min", type=float, default=0.0)
    parser.add_argument("--mu-max", type=float, default=1.5)
    parser.add_argument("--mu-points", type=int, default=19)
    parser.add_argument("--fd-step", type=float, default=2e-4)
    parser.add_argument("--validation-W", default="0.001,0.002,0.005,0.01,0.02,0.03")
    parser.add_argument("--validation-real", type=int, default=2000)
    parser.add_argument("--scan-lengths", default="50,60,70,80,90,100")
    parser.add_argument("--scan-alpha", default="0.12,0.15,0.18")
    parser.add_argument("--gap-match-tol", type=float, default=0.05)
    parser.add_argument("--xi-match-tol", type=float, default=0.05)
    parser.add_argument("--separation-match-tol", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=2028)
    args = parser.parse_args()

    args.validation_W = parse_float_list(args.validation_W)
    args.scan_lengths = parse_int_list(args.scan_lengths)
    args.scan_alpha = parse_float_list(args.scan_alpha)
    run(args)


if __name__ == "__main__":
    main()
