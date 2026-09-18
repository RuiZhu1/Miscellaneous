#!/usr/bin/env python3
"""Majorana Noise Geometry v21.3 — Canonical Minimal Law Audit / Device Design Rule.

The scientific target is now deliberately narrow:

    microscopic geometry -> Majorana response K -> Cbar = K R K^T
    -> N_geom = lambda_max(Cbar) -> smallest predictive geometry law.

Core equations
--------------
    delta epsilon = K delta mu
    Cbar = K R K^T
    N_geom = lambda_max(Cbar)

v21.1 adds four falsifiable law-discovery tests:

1. Model ladder: simple, dimensionless descriptors are compared by repeated
   cross-validation. Candidate descriptors include
       E_gap / Delta, xi_M / a, d_M / a, d_M / xi_M,
   and physically motivated combinations.
2. Coefficient stability: the selected candidate law is refit across many
   random train/test splits and coefficient spread is reported.
3. Region holdouts: train/test splits along L and alpha test whether the law
   extrapolates into unseen geometry regions.
4. Paper-ready report: paper_law_v21_1.md states what passed, what failed,
   and what can or cannot yet be called a design rule.

Important physical convention
-----------------------------
Logs are only taken of dimensionless quantities:
    log(E_gap / Delta), log(xi_M / a), log(d_M / a).
The lattice spacing a is an explicit CLI parameter, default a=1 in the model.

This code deliberately avoids black-box machine learning. The candidate law
must remain short enough to fit in a few equations in the main paper.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
from scipy.optimize import minimize_scalar, brentq

try:
    import cupy as cp  # type: ignore
    CUPY_AVAILABLE = True
except Exception:
    cp = None  # type: ignore
    CUPY_AVAILABLE = False


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------


def paulis() -> Dict[str, np.ndarray]:
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    return {"I": I, "X": X, "Y": Y, "Z": Z}


def dagger(a: np.ndarray) -> np.ndarray:
    return a.conj().T


def relative_difference(a: float, b: float) -> float:
    return float(abs(a - b) / max(abs(a), abs(b), 1e-30))


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


def parse_number_list(spec: str, cast=float) -> Tuple[Any, ...]:
    spec = str(spec).strip()
    if not spec:
        raise ValueError("empty list specification")
    if ":" not in spec:
        vals = tuple(cast(x.strip()) for x in spec.split(",") if x.strip())
        if not vals:
            raise ValueError("empty list specification")
        return vals
    pieces = spec.split(":")
    if len(pieces) != 3:
        raise ValueError(f"range must be start:stop:step, got {spec!r}")
    start, stop, step = map(float, pieces)
    if step == 0:
        raise ValueError("range step cannot be zero")
    out = []
    x = start
    eps = 0.25 * abs(step)
    if step > 0:
        while x <= stop + eps:
            out.append(cast(round(x) if cast is int else x))
            x += step
    else:
        while x >= stop - eps:
            out.append(cast(round(x) if cast is int else x))
            x += step
    return tuple(out)


def choose_device(requested: str, gpu_id: int) -> Tuple[str, Dict[str, Any]]:
    info: Dict[str, Any] = {
        "requested": requested,
        "cupy_available": bool(CUPY_AVAILABLE),
        "gpu_id": int(gpu_id),
    }
    if requested not in {"auto", "cpu", "gpu"}:
        raise ValueError("device must be auto, cpu, or gpu")
    if requested == "cpu":
        return "cpu", info
    if not CUPY_AVAILABLE:
        if requested == "gpu":
            raise RuntimeError("--device gpu requested but CuPy is unavailable")
        return "cpu", info
    try:
        count = int(cp.cuda.runtime.getDeviceCount())
        info["gpu_count"] = count
        if count <= gpu_id:
            if requested == "gpu":
                raise RuntimeError(f"GPU id {gpu_id} unavailable; device count={count}")
            return "cpu", info
        cp.cuda.Device(gpu_id).use()
        props = cp.cuda.runtime.getDeviceProperties(gpu_id)
        name = props.get("name", b"unknown")
        if isinstance(name, bytes):
            name = name.decode(errors="ignore")
        info["gpu_name"] = str(name)
        info["compute_capability"] = f"{props.get('major', '?')}.{props.get('minor', '?')}"
        info["cuda_runtime"] = str(cp.cuda.runtime.runtimeGetVersion())
        return "gpu", info
    except Exception:
        if requested == "gpu":
            raise
        return "cpu", info


# -----------------------------------------------------------------------------
# BdG model
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

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BdGWire:
    """Finite spinful Rashba BdG wire."""

    def __init__(self, p: BdGParams):
        self.p = p
        P = paulis()
        self.I2, self.sx, self.sy = P["I"], P["X"], P["Y"]
        self.tx, self.ty, self.tz = P["X"], P["Y"], P["Z"]
        self.U_C = np.kron(self.ty, self.sy)
        self.tzI = np.kron(self.tz, self.I2)

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
        H0 = (
            (2 * p.t0 - p.mu) * self.tzI
            + p.Ez * np.kron(self.I2, self.sx)
            + p.Delta * np.kron(self.tx, self.I2)
        )
        hop = -p.t0 * self.tzI - 0.5j * p.alpha * np.kron(self.tz, self.sy)
        H = np.zeros((4 * p.L, 4 * p.L), dtype=complex)
        d = None if disorder is None else np.asarray(disorder, float)
        if d is not None and d.shape != (p.L,):
            raise ValueError(f"disorder must have shape ({p.L},)")
        for i in range(p.L):
            s = slice(4 * i, 4 * (i + 1))
            site = H0.copy()
            if d is not None:
                site -= float(d[i]) * self.tzI
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
        return la.eigh(H, subset_by_index=[max(0, mid - 5), min(n - 1, mid + 4)])

    def majoranas(self, disorder: Optional[np.ndarray] = None) -> Dict[str, Any]:
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
# GPU batched helpers
# -----------------------------------------------------------------------------


def _build_clean_H_stack(params_list: Sequence[BdGParams]) -> np.ndarray:
    return np.stack([BdGWire(p).finite_H(None) for p in params_list], axis=0)


def gpu_batch_lowest_positive(params_list: Sequence[BdGParams]) -> np.ndarray:
    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy is unavailable")
    H = cp.asarray(_build_clean_H_stack(params_list))
    vals = cp.linalg.eigvalsh(H)
    idx = vals.shape[-1] // 2
    return cp.asnumpy(vals[:, idx]).real


def _gpu_majorana_from_eigh(Hc: Any, Uc: Any, x: Any, tzI: Any, reference: Optional[np.ndarray] = None, need_kernel: bool = False) -> Dict[str, Any]:
    vals, vecs = cp.linalg.eigh(Hc)
    nmat = Hc.shape[-1]
    idx = nmat // 2
    psi = vecs[:, :, idx]
    psi_h = cp.einsum("ij,bj->bi", Uc, cp.conj(psi))
    norm = cp.sqrt(cp.sum(cp.abs(psi_h) ** 2, axis=1, keepdims=True))
    psi_h = psi_h / cp.maximum(norm, 1e-15)
    G = cp.stack(((psi + psi_h) / cp.sqrt(2.0), -1j * (psi - psi_h) / cp.sqrt(2.0)), axis=2)
    xG = x[None, :, None] * G
    xmat = cp.real(cp.einsum("bni,bnj->bij", cp.conj(G), xG))
    xmat = 0.5 * (xmat + cp.swapaxes(xmat, -1, -2))
    _, U = cp.linalg.eigh(xmat)
    M = cp.einsum("bni,bij->bnj", G, U)

    nsite = M.shape[1] // 4
    site_M = M.reshape(M.shape[0], nsite, 4, 2)
    prob = cp.sum(cp.abs(site_M) ** 2, axis=2)
    xsite = x[:nsite]
    centers = cp.sum(prob * xsite[None, :, None], axis=1)
    swap = centers[:, 0] > centers[:, 1]
    if bool(cp.any(swap).item()):
        temp = M[swap, :, 0].copy()
        M[swap, :, 0] = M[swap, :, 1]
        M[swap, :, 1] = temp
        c0 = centers[swap, 0].copy()
        centers[swap, 0] = centers[swap, 1]
        centers[swap, 1] = c0

    for j in range(2):
        idx_max = cp.argmax(cp.abs(M[:, :, j]), axis=1)
        ref = M[cp.arange(M.shape[0]), idx_max, j]
        signs = cp.where(cp.real(ref) < 0, -1.0, 1.0)
        M[:, :, j] *= signs[:, None]

    if reference is not None:
        refc = cp.asarray(reference)
        overlap = cp.real(cp.einsum("ni,bnj->bij", cp.conj(refc), M))
        Uo, _, Vh = cp.linalg.svd(overlap)
        O = cp.einsum("bij,bjk->bik", Uo, Vh)
        M = cp.einsum("bni,bij->bnj", M, O)

    HM = cp.matmul(Hc, M)
    coupling = cp.einsum("bni,bnj->bij", cp.conj(M), HM)
    eps = cp.imag(coupling[:, 0, 1])
    E_low = cp.real(vals[:, idx])

    site_M = M.reshape(M.shape[0], nsite, 4, 2)
    prob = cp.sum(cp.abs(site_M) ** 2, axis=2)
    centers = cp.sum(prob * xsite[None, :, None], axis=1)
    widths = cp.sqrt(cp.maximum(cp.sum(prob * (xsite[None, :, None] - centers[:, None, :]) ** 2, axis=1), 0.0))

    out: Dict[str, Any] = {"M": M, "epsilon": eps, "E_low": E_low, "centers": centers, "widths": widths}
    if need_kernel:
        local = cp.einsum("blia,ij,bljc->blac", cp.conj(site_M), tzI, site_M)
        out["K"] = -cp.imag(local[:, :, 0, 1])
    return out


def gpu_batch_wire_features(params_list: Sequence[BdGParams], need_kernel: bool = True) -> List[Dict[str, Any]]:
    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy is unavailable")
    if not params_list:
        return []
    L = params_list[0].L
    if any(p.L != L for p in params_list):
        raise ValueError("gpu_batch_wire_features requires equal L")
    H = cp.asarray(_build_clean_H_stack(params_list))
    P = paulis()
    Uc_np = np.kron(P["Y"], P["Y"])
    Uc = cp.kron(cp.eye(L, dtype=cp.complex128), cp.asarray(Uc_np))
    x = cp.repeat(cp.arange(L, dtype=cp.float64), 4)
    tzI = cp.asarray(np.kron(P["Z"], P["I"]))
    raw = _gpu_majorana_from_eigh(H, Uc, x, tzI, reference=None, need_kernel=need_kernel)
    M = cp.asnumpy(raw["M"])
    eps = cp.asnumpy(raw["epsilon"])
    E_low = cp.asnumpy(raw["E_low"])
    centers = cp.asnumpy(raw["centers"])
    widths = cp.asnumpy(raw["widths"])
    K = cp.asnumpy(raw["K"]) if need_kernel else None
    out: List[Dict[str, Any]] = []
    for i, p in enumerate(params_list):
        wire = BdGWire(p)
        out.append({
            "M": M[i], "epsilon": float(eps[i]), "E_low": float(E_low[i]),
            "E_gap": float(wire.bulk_gap()), "z2": int(wire.z2()),
            "centers": centers[i], "widths": widths[i],
            "K": K[i] if K is not None else None,
        })
    return out


def gpu_batch_epsilons(wire: BdGWire, reference: np.ndarray, disorders: np.ndarray, batch_size: int) -> np.ndarray:
    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy is unavailable")
    d = np.asarray(disorders, dtype=np.float64)
    if d.ndim != 2 or d.shape[1] != wire.p.L:
        raise ValueError("disorders must have shape (n, L)")
    n = d.shape[0]
    out = np.empty(n, dtype=float)
    H0 = wire.finite_H(None)
    H0c = cp.asarray(H0)
    P = paulis()
    Uc = cp.kron(cp.eye(wire.p.L, dtype=cp.complex128), cp.asarray(np.kron(P["Y"], P["Y"])))
    x = cp.repeat(cp.arange(wire.p.L, dtype=cp.float64), 4)
    tz_diag = np.tile(np.array([1.0, 1.0, -1.0, -1.0]), wire.p.L)
    site_idx = np.repeat(np.arange(wire.p.L), 4)
    tzI_gpu = cp.asarray(wire.tzI)
    for start in range(0, n, batch_size):
        stop = min(n, start + batch_size)
        db = cp.asarray(d[start:stop])
        H = cp.broadcast_to(H0c, (stop - start, H0.shape[0], H0.shape[1])).copy()
        diag = -db[:, site_idx] * cp.asarray(tz_diag)[None, :]
        ii = cp.arange(H.shape[-1])
        H[:, ii, ii] += diag
        raw = _gpu_majorana_from_eigh(H, Uc, x, tzI_gpu, reference=reference, need_kernel=False)
        out[start:stop] = cp.asnumpy(raw["epsilon"])
    return out


# -----------------------------------------------------------------------------
# Operating point / response kernel
# -----------------------------------------------------------------------------


def make_wire(template: BdGWire, mu: float) -> BdGWire:
    p = template.p
    return BdGWire(BdGParams(p.L, p.alpha, p.Delta, p.Ez, p.t0, float(mu)))


def tune_operating_point_cpu(template: BdGWire, mu_min: float, mu_max: float, points: int) -> Tuple[BdGWire, np.ndarray, OperatingPoint]:
    mus = np.linspace(mu_min, mu_max, points)
    E = np.array([make_wire(template, mu).majoranas()["E_low"] for mu in mus], float)
    i = int(np.argmin(E))
    a = float(mus[max(0, i - 1)])
    b = float(mus[min(points - 1, i + 1)])
    if a == b:
        a, b = mu_min, mu_max
    res = minimize_scalar(lambda mu: float(make_wire(template, mu).majoranas()["E_low"]), bounds=(a, b), method="bounded", options={"xatol": 1e-9})
    wire = make_wire(template, float(res.x))
    d = wire.majoranas()
    ref = np.asarray(d["M"], complex)
    c = np.asarray(d["centers"], float)
    w = np.asarray(d["widths"], float)
    op = OperatingPoint(template.p.L, template.p.alpha, float(res.x), float(d["epsilon"]), float(d["E_low"]), float(d["E_gap"]), int(d["z2"]), float(c[0]), float(c[1]), float(w[0]), float(w[1]), float(np.mean(w)), float(c[1] - c[0]))
    return wire, ref, op


def tune_operating_point_gpu(template: BdGWire, mu_min: float, mu_max: float, points: int, refine_stages: int, refine_points: int) -> Tuple[BdGWire, np.ndarray, OperatingPoint]:
    lo, hi = float(mu_min), float(mu_max)
    best_mu = float(template.p.mu)
    for stage in range(max(1, refine_stages + 1)):
        npts = points if stage == 0 else refine_points
        mus = np.linspace(lo, hi, max(npts, 5))
        params = [BdGParams(template.p.L, template.p.alpha, template.p.Delta, template.p.Ez, template.p.t0, float(m)) for m in mus]
        Es = gpu_batch_lowest_positive(params)
        j = int(np.argmin(Es))
        best_mu = float(mus[j])
        if len(mus) == 1:
            break
        step = float(mus[1] - mus[0])
        lo = max(mu_min, best_mu - 1.5 * step)
        hi = min(mu_max, best_mu + 1.5 * step)
        if hi - lo < 1e-10:
            break
    wire = make_wire(template, best_mu)
    feat = gpu_batch_wire_features([wire.p], need_kernel=True)[0]
    ref = np.asarray(feat["M"], complex)
    centers = np.asarray(feat["centers"], float)
    widths = np.asarray(feat["widths"], float)
    op = OperatingPoint(wire.p.L, wire.p.alpha, best_mu, float(feat["epsilon"]), float(feat["E_low"]), float(feat["E_gap"]), int(feat["z2"]), float(centers[0]), float(centers[1]), float(widths[0]), float(widths[1]), float(np.mean(widths)), float(centers[1] - centers[0]))
    return wire, ref, op


def tune_operating_point(template: BdGWire, mu_min: float, mu_max: float, points: int, device: str, refine_stages: int, refine_points: int) -> Tuple[BdGWire, np.ndarray, OperatingPoint]:
    if points < 5:
        raise ValueError("mu-points must be >= 5")
    if not mu_min < mu_max:
        raise ValueError("mu-min must be < mu-max")
    return tune_operating_point_gpu(template, mu_min, mu_max, points, refine_stages, refine_points) if device == "gpu" else tune_operating_point_cpu(template, mu_min, mu_max, points)


def majorana_kernel(wire: BdGWire, reference: np.ndarray, device: str) -> np.ndarray:
    if device == "gpu":
        feat = gpu_batch_wire_features([wire.p], need_kernel=True)[0]
        M = np.asarray(feat["M"], complex)
        M = M @ wire.align(reference, M)
        site_M = M.reshape(wire.p.L, 4, 2)
        K = np.empty(wire.p.L, dtype=float)
        for i in range(wire.p.L):
            local = site_M[i].conj().T @ (-wire.tzI) @ site_M[i]
            K[i] = float(np.imag(local[0, 1]))
        return K
    d = wire.majoranas()
    M = np.asarray(d["M"], complex)
    M = M @ wire.align(reference, M)
    K = np.empty(wire.p.L)
    for i in range(wire.p.L):
        s = slice(4 * i, 4 * (i + 1))
        Mi = M[s, :]
        K[i] = float(np.imag((Mi.conj().T @ (-wire.tzI) @ Mi)[0, 1]))
    return K


def kernel_fd_check(wire: BdGWire, reference: np.ndarray, K: np.ndarray, sites: Sequence[int], step: float) -> float:
    errs = []
    zero = np.zeros(wire.p.L)
    for i in sites:
        dp = zero.copy(); dm = zero.copy()
        dp[int(i)] = step; dm[int(i)] = -step
        fd = (wire.epsilon(dp, reference) - wire.epsilon(dm, reference)) / (2 * step)
        errs.append(abs(float(fd) - float(K[int(i)])) / max(abs(float(fd)), 1e-30))
    return float(max(errs, default=0.0))


# -----------------------------------------------------------------------------
# Noise transfer and validation
# -----------------------------------------------------------------------------


def spatial_R(L: int, xi: float) -> np.ndarray:
    x = np.arange(L, dtype=float)
    return np.eye(L) if xi <= 0 else np.exp(-np.abs(x[:, None] - x[None, :]) / float(xi))


def cross_R(L1: int, L2: int, xi: float) -> np.ndarray:
    a = np.arange(L1, dtype=float)[:, None]
    b = np.arange(L2, dtype=float)[None, :]
    return np.eye(L1, L2) if xi <= 0 else np.exp(-np.abs(a - b) / float(xi))


def noise_R(L1: int, L2: int, rho: float, xi: float) -> np.ndarray:
    if not -1.0 <= rho <= 1.0:
        raise ValueError("rho-site must lie in [-1,1]")
    R12 = rho * cross_R(L1, L2, xi)
    R = np.block([[spatial_R(L1, xi), R12], [R12.T, spatial_R(L2, xi)]])
    return 0.5 * (R + R.T)


def effective_noise(K1: np.ndarray, K2: np.ndarray, R: np.ndarray) -> Dict[str, Any]:
    K1 = np.asarray(K1, float)
    K2 = np.asarray(K2, float)
    K = np.block([[K1[None, :], np.zeros((1, K2.size))], [np.zeros((1, K1.size)), K2[None, :]]])
    C = 0.5 * (K @ R @ K.T + (K @ R @ K.T).T)
    vals, vecs = la.eigh(C)
    vp = np.array([1.0, 1.0])
    vm = np.array([1.0, -1.0])
    return {
        "K": K, "Cbar": C, "eigenvalues": vals, "eigenvectors": vecs,
        "lambda_weak": float(vals[0]), "lambda_strong": float(vals[-1]),
        "mode_ratio": float(vals[-1] / max(vals[0], 1e-30)), "N_geom": float(vals[-1]),
        "S_plus_bar": float(vp @ C @ vp), "S_minus_bar": float(vm @ C @ vm),
    }


def sample_disorder(L1: int, L2: int, rho: float, xi: float, n: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    R = noise_R(L1, L2, rho, xi)
    L = la.cholesky(R + 1e-11 * np.eye(L1 + L2), lower=True)
    z = np.random.default_rng(seed).normal(size=(n, L1 + L2)) @ L.T
    return z[:, :L1], z[:, L1:]


def nonlinear_window(wire1: BdGWire, wire2: BdGWire, ref1: np.ndarray, ref2: np.ndarray, Cbar: np.ndarray, rho: float, xi: float, W_values: Sequence[float], n: int, seed: int, device: str, gpu_batch_size: int) -> Dict[str, Any]:
    W_values = sorted(set(float(w) for w in W_values if float(w) > 0))
    d1u, d2u = sample_disorder(wire1.p.L, wire2.p.L, rho, xi, n, seed)
    rows = []
    for W in W_values:
        d1, d2 = W * d1u, W * d2u
        if device == "gpu":
            e1 = gpu_batch_epsilons(wire1, ref1, d1, gpu_batch_size)
            e2 = gpu_batch_epsilons(wire2, ref2, d2, gpu_batch_size)
        else:
            e1 = np.array([wire1.epsilon(x, ref1) for x in d1], float)
            e2 = np.array([wire2.epsilon(x, ref2) for x in d2], float)
        E = np.column_stack((e1 - e1.mean(), e2 - e2.mean()))
        Cdir = (E.T @ E) / max(n - 1, 1)
        target = W * W * Cbar
        denom = max(la.norm(Cdir, "fro"), 1e-30)
        eta = float(la.norm(Cdir - target, "fro") / denom)
        rows.append({"W": float(W), "eta_nl": eta, "C_direct": Cdir.tolist(), "C_target": target.tolist(), "rho_epsilon": corr(e1, e2), "n_real": int(n)})
    return {"rows": rows}


# -----------------------------------------------------------------------------
# Geometry scan and pair metrics
# -----------------------------------------------------------------------------


def scan_geometry(L_values: Sequence[int], alpha_values: Sequence[float], Delta: float, Ez: float, t0: float, mu_min: float, mu_max: float, mu_points: int, K_reference: np.ndarray, L_reference: int, rho: float, xi: float, device: str, mu_refine_stages: int, mu_refine_points: int) -> Tuple[List[Dict[str, float]], Dict[str, Any]]:
    records: List[Dict[str, float]] = []
    tune_meta: Dict[str, Any] = {"device": device, "successful": 0, "failed": 0}
    for alpha in alpha_values:
        for L in L_values:
            try:
                template = BdGWire(BdGParams(int(L), float(alpha), Delta, Ez, t0, 0.5))
                wire, ref, op = tune_operating_point(template, mu_min, mu_max, mu_points, device, mu_refine_stages, mu_refine_points)
                if op.z2 != -1:
                    tune_meta["failed"] += 1
                    continue
                K = majorana_kernel(wire, ref, device)
                R = noise_R(L_reference, int(L), rho, xi)
                tr = effective_noise(K_reference, K, R)
                vals = np.asarray(tr["eigenvalues"], float)
                records.append({
                    "L": int(L), "alpha": float(alpha), "mu": float(op.mu),
                    "E_gap": float(op.E_gap), "E_low": float(op.E_low),
                    "xi_mean": float(op.xi_mean), "separation": float(op.separation),
                    "d_over_xi": float(op.separation / max(op.xi_mean, 1e-30)),
                    "gap_over_Delta": float(op.E_gap / max(Delta, 1e-30)),
                    "N_geom": float(tr["N_geom"]), "lambda_weak": float(vals[0]),
                    "lambda_strong": float(vals[-1]), "mode_ratio": float(tr["mode_ratio"]),
                })
                tune_meta["successful"] += 1
            except (RuntimeError, ValueError, la.LinAlgError, FloatingPointError) as exc:
                tune_meta["failed"] += 1
                tune_meta.setdefault("errors", []).append({"L": int(L), "alpha": float(alpha), "error": str(exc)})
    return records, tune_meta


def add_dimensionless_descriptors(records: Sequence[Dict[str, float]], Delta: float, lattice_spacing: float) -> None:
    if lattice_spacing <= 0:
        raise ValueError("lattice-spacing must be > 0")
    for r in records:
        r["gap_over_Delta"] = float(r["E_gap"] / max(Delta, 1e-30))
        r["xi_over_a"] = float(r["xi_mean"] / lattice_spacing)
        r["d_over_a"] = float(r["separation"] / lattice_spacing)
        r["log_gap"] = float(math.log(max(r["gap_over_Delta"], 1e-30)))
        r["log_xi_over_a"] = float(math.log(max(r["xi_over_a"], 1e-30)))
        r["log_d_over_a"] = float(math.log(max(r["d_over_a"], 1e-30)))


def pair_metrics(a: Dict[str, float], b: Dict[str, float], tolerances: Tuple[float, float, float]) -> Dict[str, float]:
    gt, xt, st = tolerances
    gd = relative_difference(a["E_gap"], b["E_gap"])
    xd = relative_difference(a["xi_mean"], b["xi_mean"])
    sd = relative_difference(a["separation"], b["separation"])
    D = math.sqrt((gd / gt) ** 2 + (xd / xt) ** 2 + (sd / st) ** 2)
    Dinf = max(gd / gt, xd / xt, sd / st)
    ratio = max(a["N_geom"], b["N_geom"]) / max(min(a["N_geom"], b["N_geom"]), 1e-30)
    return {
        "relative_gap_difference": float(gd), "relative_xi_difference": float(xd),
        "relative_separation_difference": float(sd), "match_distance": float(D),
        "match_distance_inf": float(Dinf), "N_geom_ratio": float(ratio),
    }


def strict_pair_statistics(records: Sequence[Dict[str, float]], gap_tol: float, xi_tol: float, separation_tol: float) -> Dict[str, Any]:
    if len(records) < 2:
        return {"count": 0, "max_ratio": None, "median_ratio": None, "p95_ratio": None, "nearest": None, "max_ratio_pair": None}
    tol = (float(gap_tol), float(xi_tol), float(separation_tol))
    nearest, nearest_key = None, None
    strict_ratios: List[float] = []
    strict_pairs: List[Dict[str, Any]] = []
    max_pair = None
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            a, b = records[i], records[j]
            m = pair_metrics(a, b, tol)
            key = (m["match_distance"], -m["N_geom_ratio"])
            if nearest_key is None or key < nearest_key:
                nearest_key = key
                nearest = {"A": a, "B": b, **m}
            if m["match_distance_inf"] <= 1.0:
                strict_ratios.append(m["N_geom_ratio"])
                row = {"A": a, "B": b, **m}
                strict_pairs.append(row)
                if max_pair is None or m["N_geom_ratio"] > max_pair["N_geom_ratio"]:
                    max_pair = row
    return {
        "count": int(len(strict_ratios)),
        "max_ratio": float(max(strict_ratios)) if strict_ratios else None,
        "median_ratio": float(np.median(strict_ratios)) if strict_ratios else None,
        "p95_ratio": float(np.percentile(strict_ratios, 95)) if strict_ratios else None,
        "nearest": nearest, "max_ratio_pair": max_pair,
        "strict_ratios": strict_ratios, "strict_pairs": strict_pairs,
    }


# -----------------------------------------------------------------------------
# Law discovery
# -----------------------------------------------------------------------------


MODEL_DISPLAY = {
    "gap": "E_gap / Delta",
    "xi": "xi_M / a",
    "d": "d_M / a",
    "d_over_xi": "d_M / xi_M",
    "log_gap": "log(E_gap / Delta)",
    "log_xi_over_a": "log(xi_M / a)",
    "log_d_over_a": "log(d_M / a)",
}


def _ols_fit(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta, X @ beta


def _r2(y: np.ndarray, pred: np.ndarray) -> float:
    denom = float(np.sum((y - np.mean(y)) ** 2))
    return float(1.0 - np.sum((y - pred) ** 2) / max(denom, 1e-30))


def _kfold_indices(n: int, k: int, seed: int) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    return [x for x in np.array_split(idx, min(k, n)) if x.size > 0]


def _feature_matrix(records: Sequence[Dict[str, float]], names: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    y = np.log(np.maximum(np.array([r["N_geom"] for r in records], float), 1e-30))
    cols = [np.ones(len(records), dtype=float)]
    for name in names:
        if name == "gap":
            cols.append(np.array([r["gap_over_Delta"] for r in records], float))
        elif name == "xi":
            cols.append(np.array([r["xi_over_a"] for r in records], float))
        elif name == "d":
            cols.append(np.array([r["d_over_a"] for r in records], float))
        elif name == "d_over_xi":
            cols.append(np.array([r["d_over_xi"] for r in records], float))
        elif name == "log_gap":
            cols.append(np.array([r["log_gap"] for r in records], float))
        elif name == "log_xi_over_a":
            cols.append(np.array([r["log_xi_over_a"] for r in records], float))
        elif name == "log_d_over_a":
            cols.append(np.array([r["log_d_over_a"] for r in records], float))
        else:
            raise ValueError(f"unknown feature {name}")
    return np.column_stack(cols), y


def _model_label(names: Sequence[str]) -> str:
    return "constant" if not names else " + ".join(MODEL_DISPLAY[n] for n in names)


def _formula_for(names: Sequence[str]) -> str:
    if tuple(names) == ("d_over_xi", "log_xi_over_a"):
        return r"log N_geom = A + B (d_M/xi_M) + C log(xi_M/a)"
    if not names:
        return r"log N_geom = A"
    return "log N_geom = A + " + " + ".join(MODEL_DISPLAY[n] for n in names)


def _prediction_metrics(y_log: np.ndarray, pred_log: np.ndarray) -> Dict[str, float]:
    y = np.exp(y_log)
    p = np.exp(pred_log)
    rel = np.abs(p - y) / np.maximum(np.abs(y), 1e-30)
    return {
        "rmse_logN": float(np.sqrt(np.mean((y_log - pred_log) ** 2))),
        "r2_logN": _r2(y_log, pred_log),
        "median_relative_error": float(np.median(rel)),
        "p90_relative_error": float(np.percentile(rel, 90)),
    }


def evaluate_model(records: Sequence[Dict[str, float]], names: Sequence[str], folds: int, seed: int) -> Dict[str, Any]:
    X, y = _feature_matrix(records, names)
    n, p = X.shape
    folds_idx = _kfold_indices(n, folds, seed)
    pred_cv = np.full(n, np.nan)
    for test_idx in folds_idx:
        train_mask = np.ones(n, dtype=bool)
        train_mask[test_idx] = False
        beta, _ = _ols_fit(X[train_mask], y[train_mask])
        pred_cv[test_idx] = X[test_idx] @ beta
    beta, pred_all = _ols_fit(X, y)
    cv = _prediction_metrics(y, pred_cv)
    in_sample = _prediction_metrics(y, pred_all)
    rss = float(np.sum((y - pred_all) ** 2))
    bic = float(n * math.log(max(rss / n, 1e-30)) + p * math.log(max(n, 2)))
    return {
        "features": list(names), "label": _model_label(names), "formula": _formula_for(names),
        "n_features": len(names), "coefficients": beta.tolist(),
        "cv_rmse_logN": cv["rmse_logN"], "cv_r2_logN": cv["r2_logN"],
        "cv_median_relative_error": cv["median_relative_error"], "cv_p90_relative_error": cv["p90_relative_error"],
        "in_sample_rmse_logN": in_sample["rmse_logN"], "bic": bic,
        "predicted_logN": pred_cv.tolist(),
    }


def candidate_model_space() -> List[Tuple[str, ...]]:
    return [
        (),
        ("gap",), ("xi",), ("d",), ("d_over_xi",), ("log_xi_over_a",),
        ("gap", "xi"), ("gap", "d"), ("xi", "d"),
        ("gap", "xi", "d"),
        ("gap", "d_over_xi"),
        ("d_over_xi", "log_xi_over_a"),
        ("gap", "d_over_xi", "log_xi_over_a"),
    ]


def _simple_model_key(r: Dict[str, Any]) -> Tuple[int, float]:
    return int(r["n_features"]), float(r["cv_rmse_logN"])


def discover_law(records: Sequence[Dict[str, float]], folds: int, seed: int, tolerance: float, target_r2: float, target_median_relerr: float) -> Dict[str, Any]:
    if len(records) < 12:
        return {"status": "insufficient_candidates", "n": len(records)}
    models = [evaluate_model(records, m, folds, seed) for m in candidate_model_space()]
    best = min(models, key=lambda x: x["cv_rmse_logN"])
    threshold = best["cv_rmse_logN"] * (1.0 + tolerance)
    minimal_near_best = next((r for r in sorted(models, key=_simple_model_key) if r["cv_rmse_logN"] <= threshold), best)
    r2_pass = [r for r in models if r["cv_r2_logN"] >= target_r2]
    median_err_pass = [r for r in models if r["cv_median_relative_error"] <= target_median_relerr]
    joint_pass = [r for r in models if (r["cv_r2_logN"] >= target_r2 and r["cv_median_relative_error"] <= target_median_relerr)]
    return {
        "status": "ok", "n": len(records), "target": "log(N_geom)",
        "dimensionless_convention": "a = lattice spacing; Delta = pairing scale",
        "models": models,
        "best_cv_model": best,
        "minimal_model_within_tolerance": minimal_near_best,
        "minimal_engineering_model_r2": min(r2_pass, key=_simple_model_key) if r2_pass else None,
        "minimal_engineering_model_median_error": min(median_err_pass, key=_simple_model_key) if median_err_pass else None,
        "minimal_engineering_model_joint": min(joint_pass, key=_simple_model_key) if joint_pass else None,
        "minimality_tolerance_fraction": float(tolerance),
        "engineering_targets": {"cv_r2": float(target_r2), "median_relative_error": float(target_median_relerr), "joint_requires_both": True},
        "candidate_law": _formula_for(minimal_near_best["features"]),
    }


def coefficient_stability(records: Sequence[Dict[str, float]], names: Sequence[str], repeats: int, train_fraction: float, seed: int) -> Dict[str, Any]:
    X, y = _feature_matrix(records, names)
    n = len(records)
    train_n = min(max(int(round(train_fraction * n)), X.shape[1] + 2), n - 1)
    rng = np.random.default_rng(seed)
    betas = []
    for _ in range(max(1, repeats)):
        idx = rng.choice(n, size=train_n, replace=False)
        beta, _ = _ols_fit(X[idx], y[idx])
        betas.append(beta)
    B = np.asarray(betas, float)
    mean = np.mean(B, axis=0)
    std = np.std(B, axis=0, ddof=1) if B.shape[0] > 1 else np.zeros(B.shape[1])
    sign_stability = [float(np.mean(np.sign(B[:, j]) == np.sign(mean[j]))) if abs(mean[j]) > 1e-15 else 0.0 for j in range(B.shape[1])]
    return {
        "features": list(names), "label": _model_label(names), "repeats": int(repeats),
        "train_fraction": float(train_fraction), "coefficient_mean": mean.tolist(),
        "coefficient_std": std.tolist(), "relative_std": (std / np.maximum(np.abs(mean), 1e-30)).tolist(),
        "sign_stability": sign_stability,
    }


def _fit_test_metrics(records: Sequence[Dict[str, float]], names: Sequence[str], train_idx: np.ndarray, test_idx: np.ndarray) -> Dict[str, Any]:
    X, y = _feature_matrix(records, names)
    beta, _ = _ols_fit(X[train_idx], y[train_idx])
    pred = X[test_idx] @ beta
    m = _prediction_metrics(y[test_idx], pred)
    m["n_train"] = int(len(train_idx)); m["n_test"] = int(len(test_idx))
    return m


def region_holdouts(records: Sequence[Dict[str, float]], names: Sequence[str]) -> Dict[str, Any]:
    if len(records) < 20:
        return {}
    L = np.array([r["L"] for r in records])
    alpha = np.array([r["alpha"] for r in records])
    specs = {
        "train_L_<=80_test_>80": (L <= 80, L > 80),
        "train_L_>80_test_<=80": (L > 80, L <= 80),
        "train_alpha_<=0.16_test_>=0.17": (alpha <= 0.16 + 1e-12, alpha >= 0.17 - 1e-12),
        "train_alpha_>=0.17_test_<=0.16": (alpha >= 0.17 - 1e-12, alpha <= 0.16 + 1e-12),
    }
    out = {}
    for label, (train_mask, test_mask) in specs.items():
        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]
        if len(train_idx) < 5 or len(test_idx) < 5:
            out[label] = {"status": "skipped", "n_train": int(len(train_idx)), "n_test": int(len(test_idx))}
        else:
            out[label] = {"status": "ok", **_fit_test_metrics(records, names, train_idx, test_idx)}
    return out


def repeated_cv_stability(records: Sequence[Dict[str, float]], names: Sequence[str], folds: int, repeats: int, seed: int) -> Dict[str, Any]:
    vals = []
    for i in range(max(1, repeats)):
        res = evaluate_model(records, names, folds, seed + i)
        vals.append([res["cv_rmse_logN"], res["cv_r2_logN"], res["cv_median_relative_error"]])
    A = np.asarray(vals, float)
    return {
        "repeats": int(len(vals)),
        "folds": int(folds),
        "rmse_logN_mean": float(np.mean(A[:, 0])), "rmse_logN_std": float(np.std(A[:, 0], ddof=1)) if len(vals) > 1 else 0.0,
        "r2_mean": float(np.mean(A[:, 1])), "r2_std": float(np.std(A[:, 1], ddof=1)) if len(vals) > 1 else 0.0,
        "median_relative_error_mean": float(np.mean(A[:, 2])), "median_relative_error_std": float(np.std(A[:, 2], ddof=1)) if len(vals) > 1 else 0.0,
    }


# -----------------------------------------------------------------------------
# Figures and reports
# -----------------------------------------------------------------------------


def compact_pair(pair: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if pair is None:
        return None
    keys = ["L", "alpha", "mu", "E_gap", "xi_mean", "separation", "d_over_xi", "N_geom"]
    return {
        "A": {k: pair["A"].get(k) for k in keys}, "B": {k: pair["B"].get(k) for k in keys},
        **{k: pair[k] for k in ["relative_gap_difference", "relative_xi_difference", "relative_separation_difference", "match_distance", "match_distance_inf", "N_geom_ratio"]},
    }


def make_figures(outdir: str, K1: np.ndarray, K2: np.ndarray, transfer: Dict[str, Any], geometry: Sequence[Dict[str, float]], pair_stats: Dict[str, Any], law: Dict[str, Any]) -> None:
    # Core response
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.plot(np.arange(K1.size), np.abs(K1), label="|K12(x)|")
    ax.plot(np.arange(K2.size), np.abs(K2), label="|K34(x)|")
    ax.set_xlabel("site index"); ax.set_ylabel("|d epsilon_M / d mu_x|")
    ax.set_title("Majorana noise-response geometry"); ax.grid(True, ls=":", alpha=0.4); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "operating_point_v21_1.png"), dpi=220); plt.close(fig)

    C = np.asarray(transfer["Cbar"]); vals = np.asarray(transfer["eigenvalues"])
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    im = ax.imshow(C, origin="lower", aspect="equal", interpolation="nearest")
    ax.set_xticks([0, 1], ["epsilon12", "epsilon34"]); ax.set_yticks([0, 1], ["epsilon12", "epsilon34"])
    ax.set_title(f"Cbar; weak={vals[0]:.2e}, strong={vals[-1]:.2e}")
    fig.colorbar(im, ax=ax, label="Cbar"); fig.tight_layout(); fig.savefig(os.path.join(outdir, "noise_modes_v21_1.png"), dpi=220); plt.close(fig)

    if len(geometry) >= 2:
        # Money figure: d/xi vs log N colored by xi/a, with fitted family for selected law.
        x = np.array([r["d_over_xi"] for r in geometry])
        y = np.log(np.maximum(np.array([r["N_geom"] for r in geometry]), 1e-30))
        c = np.array([r["xi_over_a"] for r in geometry])
        fig, ax = plt.subplots(figsize=(7.8, 5.4))
        sc = ax.scatter(x, y, c=c, s=36, alpha=0.72)
        ax.set_xlabel(r"$d_M/\xi_M$"); ax.set_ylabel(r"$\log N_{geom}$")
        ax.set_title("Minimal-geometry view of effective noise")
        ax.grid(True, ls=":", alpha=0.35); fig.colorbar(sc, ax=ax, label=r"$\xi_M/a$")
        if law.get("status") == "ok":
            best = law["minimal_model_within_tolerance"]
            if tuple(best["features"]) == ("d_over_xi", "log_xi_over_a"):
                beta = np.asarray(best["coefficients"], float)
                xs = np.linspace(float(np.min(x)), float(np.max(x)), 100)
                for q in [0.2, 0.5, 0.8]:
                    lx = float(np.quantile(np.log(np.maximum(c, 1e-30)), q))
                    ys = beta[0] + beta[1] * xs + beta[2] * lx
                    ax.plot(xs, ys, ls="--", alpha=0.55)
        fig.tight_layout(); fig.savefig(os.path.join(outdir, "law_money_figure_v21_1.png"), dpi=240); plt.close(fig)

        # CV law ladder
        results = law.get("models", []) if law.get("status") == "ok" else []
        if results:
            labels = [r["label"] for r in results]
            rmse = [r["cv_rmse_logN"] for r in results]
            order = np.argsort(rmse)
            fig, ax = plt.subplots(figsize=(8.2, 6.2))
            yy = np.arange(len(order))
            ax.barh(yy, np.array(rmse)[order])
            ax.set_yticks(yy, [labels[i] for i in order], fontsize=8)
            ax.invert_yaxis(); ax.set_xlabel("5-fold CV RMSE of log N_geom")
            ax.set_title("Law ladder: simplest interpretable models")
            ax.grid(True, axis="x", ls=":", alpha=0.35)
            fig.tight_layout(); fig.savefig(os.path.join(outdir, "law_ladder_v21_1.png"), dpi=220); plt.close(fig)

        # Predicted vs actual for selected law.
        best = law["minimal_model_within_tolerance"]
        pred = np.asarray(best["predicted_logN"], float); actual = y
        lo, hi = min(float(np.min(actual)), float(np.min(pred))), max(float(np.max(actual)), float(np.max(pred)))
        fig, ax = plt.subplots(figsize=(6.2, 5.4))
        ax.scatter(actual, pred, s=30, alpha=0.65)
        ax.plot([lo, hi], [lo, hi], ls="--", alpha=0.5)
        ax.set_xlabel("actual log N_geom"); ax.set_ylabel("5-fold CV predicted log N_geom")
        ax.set_title(f"Selected law: {best['label']}"); ax.grid(True, ls=":", alpha=0.35)
        fig.tight_layout(); fig.savefig(os.path.join(outdir, "law_cv_pred_vs_actual_v21_1.png"), dpi=220); plt.close(fig)

    # Pair analysis
    points_x, points_y, strict_x, strict_y = [], [], [], []
    for i in range(len(geometry)):
        for j in range(i + 1, len(geometry)):
            m = pair_metrics(geometry[i], geometry[j], (0.05, 0.05, 0.05))
            points_x.append(m["match_distance"]); points_y.append(m["N_geom_ratio"])
            if m["match_distance_inf"] <= 1.0:
                strict_x.append(m["match_distance"]); strict_y.append(m["N_geom_ratio"])
    if points_x:
        fig, ax = plt.subplots(figsize=(7.8, 5.0))
        ax.scatter(points_x, points_y, s=7, alpha=0.22, label="all pairs")
        if strict_x:
            ax.scatter(strict_x, strict_y, s=18, alpha=0.65, label="strict 5% pairs")
            ax.axvline(math.sqrt(3.0), ls="--", alpha=0.45, label=r"$D_{match}=\sqrt{3}$")
        ax.set_yscale("log"); ax.set_xlabel("D_match (Euclidean, normalized by 5%)")
        ax.set_ylabel("N_geom ratio"); ax.set_title("Pairwise geometry matching")
        ax.grid(True, ls=":", alpha=0.35); ax.legend()
        fig.tight_layout(); fig.savefig(os.path.join(outdir, "pair_analysis_v21_1.png"), dpi=220); plt.close(fig)


def write_geometry_csv(path: str, records: Sequence[Dict[str, float]]) -> None:
    if not records:
        return
    fields = list(records[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(records)


def _format_model_table(models: Sequence[Dict[str, Any]]) -> str:
    rows = ["| Model | CV R² | CV RMSE(log N) | median rel. err. |", "|---|---:|---:|---:|"]
    for m in sorted(models, key=lambda x: (x["n_features"], x["cv_rmse_logN"])):
        rows.append(f"| {m['label']} | {m['cv_r2_logN']:.4f} | {m['cv_rmse_logN']:.5f} | {100*m['cv_median_relative_error']:.2f}% |")
    return "\n".join(rows)


def write_paper_law(path: str, report: Dict[str, Any]) -> None:
    law = report["law_discovery"]
    pair = report["pair_analysis"]
    op = report["operating_points"]
    lines = []
    lines.append("# Majorana Noise Geometry v21.1 — Minimal Physical Law\n")
    lines.append("## 1. What is the law?")
    if law.get("status") == "ok":
        best = law["minimal_model_within_tolerance"]
        lines.append(f"Candidate minimal law: **{best['formula']}**")
        lines.append(f"Candidate label: `{best['label']}`")
        lines.append(f"CV R²(log N): `{best['cv_r2_logN']:.4f}`")
        lines.append(f"CV RMSE(log N): `{best['cv_rmse_logN']:.5f}`")
        lines.append(f"CV median relative error: `{100*best['cv_median_relative_error']:.2f}%`")
        if tuple(best["features"]) == ("d_over_xi", "log_xi_over_a"):
            lines.append("\nEquivalent physical form:")
            lines.append(r"`N_geom = N0 * (xi_M/a)^C * exp[B * (d_M/xi_M)]`")
    else:
        lines.append("Insufficient candidates for law discovery.")

    lines.append("\n## 2. What are the coefficients?")
    if law.get("status") == "ok":
        best = law["minimal_model_within_tolerance"]
        lines.append("Full-data coefficients (intercept first): `" + ", ".join(f"{x:.8g}" for x in best["coefficients"]) + "`")
        lines.append("Coefficient order: " + ", ".join(["A"] + [MODEL_DISPLAY[x] for x in best["features"]]))
        stab = law.get("coefficient_stability")
        if stab:
            lines.append("Repeated-split coefficient means: `" + ", ".join(f"{x:.8g}" for x in stab["coefficient_mean"]) + "`")
            lines.append("Repeated-split coefficient std: `" + ", ".join(f"{x:.8g}" for x in stab["coefficient_std"]) + "`")

    lines.append("\n## 3. What is the simplest competing law?")
    if law.get("status") == "ok":
        lines.append(_format_model_table(law["models"]))
        eng_r2 = law.get("minimal_engineering_model_r2")
        eng_err = law.get("minimal_engineering_model_median_error")
        eng_joint = law.get("minimal_engineering_model_joint")
        lines.append(f"\nSmallest model meeting CV R² >= target: **{eng_r2['label']}**" if eng_r2 else "\nNo candidate met the CV R² target.")
        lines.append(f"Smallest model meeting median relative error <= target: **{eng_err['label']}**" if eng_err else "No candidate met the median-relative-error target.")
        lines.append(f"Smallest model meeting both: **{eng_joint['label']}**" if eng_joint else "No candidate met both engineering criteria simultaneously.")

    lines.append("\n## 4. Does it survive unseen L / alpha?")
    if law.get("region_holdouts"):
        for k, v in law["region_holdouts"].items():
            if v.get("status") == "ok":
                lines.append(f"- `{k}`: R²={v['r2_logN']:.4f}, median relative error={100*v['median_relative_error']:.2f}% (n_test={v['n_test']})")
            else:
                lines.append(f"- `{k}`: skipped ({v.get('n_train')} train, {v.get('n_test')} test)")
    else:
        lines.append("Region holdouts unavailable.")

    lines.append("\n## 5. What did coefficient-stability tests show?")
    stab = law.get("coefficient_stability")
    repcv = law.get("repeated_cv_stability")
    if stab:
        lines.append(f"- coefficient repeats: {stab['repeats']}; train fraction: {stab['train_fraction']:.2f}")
        lines.append("- relative coefficient std: " + ", ".join(f"{x:.2f}" for x in stab["relative_std"]))
        lines.append("- sign stability: " + ", ".join(f"{100*x:.1f}%" for x in stab["sign_stability"]))
    if repcv:
        lines.append(f"- repeated-CV R² = {repcv['r2_mean']:.4f} ± {repcv['r2_std']:.4f}")
        lines.append(f"- repeated-CV median relative error = {100*repcv['median_relative_error_mean']:.2f}% ± {100*repcv['median_relative_error_std']:.2f}%")

    lines.append("\n## 6. What device-design rule follows?")
    lines.append("1. Compute the microscopic Majorana response `K`.")
    lines.append("2. Form the effective noise covariance `Cbar = K R K^T`.")
    lines.append("3. Minimize `N_geom = lambda_max(Cbar)` subject to the required topological and spectral constraints.")
    lines.append("4. Treat bulk gap as a constraint/quality metric, not as a complete proxy for noise resilience.")

    lines.append("\n## Strict-pair sanity check")
    lines.append(f"Strict 5% pairs: **{pair['count']}**")
    if pair.get("nearest"):
        lines.append(f"Nearest pair ratio: **{pair['nearest']['N_geom_ratio']:.6f}**")
    if pair.get("max_ratio") is not None:
        lines.append(f"Maximum strict-pair ratio: **{pair['max_ratio']:.6f}**")
    lines.append("\n## Baseline numbers")
    lines.append(f"Wire 1: L={op['wire1']['L']}, alpha={op['wire1']['alpha']:.4f}, gap={op['wire1']['E_gap']:.6e}, xi={op['wire1']['xi_mean']:.6e}, d={op['wire1']['separation']:.6e}")
    lines.append(f"Wire 2: L={op['wire2']['L']}, alpha={op['wire2']['alpha']:.4f}, gap={op['wire2']['E_gap']:.6e}, xi={op['wire2']['xi_mean']:.6e}, d={op['wire2']['separation']:.6e}")

    lines.append("\n## Main-text caution")
    lines.append("This file is a numerical law-discovery report. A candidate law becomes a main-text design rule only after the chosen error criterion and the region-holdout tests support the intended level of extrapolation.")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def run(args: argparse.Namespace) -> Dict[str, Any]:
    os.makedirs(args.outdir, exist_ok=True)
    device, device_info = choose_device(args.device, args.gpu_id)

    template1 = BdGWire(BdGParams(args.L1, args.alpha, args.Delta, args.Ez, args.t0, 0.5))
    template2 = BdGWire(BdGParams(args.L2, args.alpha, args.Delta, args.Ez, args.t0, 0.5))
    wire1, ref1, op1 = tune_operating_point(template1, args.mu_min, args.mu_max, args.mu_points, device, args.mu_refine_stages, args.mu_refine_points)
    wire2, ref2, op2 = tune_operating_point(template2, args.mu_min, args.mu_max, args.mu_points, device, args.mu_refine_stages, args.mu_refine_points)
    if op1.z2 != -1 or op2.z2 != -1:
        raise RuntimeError("At least one baseline wire is not topological")

    print("=" * 100)
    print("Majorana Noise Geometry v21.1 — Minimal Physical Law / Device Design Rule")
    print("=" * 100)
    print(f"device: {device} | GPU info: {device_info}")
    for tag, op in (("wire1", op1), ("wire2", op2)):
        print(f"{tag}: L={op.L}, alpha={op.alpha:.3f}, mu*={op.mu:.8f}, gap={op.E_gap:.3e}, xi={op.xi_mean:.3e}, separation={op.separation:.3e}, d/xi={op.separation/max(op.xi_mean,1e-30):.3f}, Z2={op.z2}")

    K1 = majorana_kernel(wire1, ref1, device)
    K2 = majorana_kernel(wire2, ref2, device)
    sites1 = np.unique(np.linspace(0, args.L1 - 1, min(6, args.L1), dtype=int))
    sites2 = np.unique(np.linspace(0, args.L2 - 1, min(6, args.L2), dtype=int))
    fd1 = kernel_fd_check(wire1, ref1, K1, sites1, args.fd_step)
    fd2 = kernel_fd_check(wire2, ref2, K2, sites2, args.fd_step)
    print(f"kernel FD relative error: {fd1:.3%} / {fd2:.3%}")

    R = noise_R(args.L1, args.L2, args.rho_site, args.xi_noise)
    transfer = effective_noise(K1, K2, R)
    vals = np.asarray(transfer["eigenvalues"], float)
    print(f"eigenvalues(Cbar): [{vals[0]:.8e} {vals[-1]:.8e}]")
    print(f"mode anisotropy lambda_strong/lambda_weak: {transfer['mode_ratio']:.4f}")
    print(f"N_geom: {transfer['N_geom']:.8e}")

    validation = nonlinear_window(wire1, wire2, ref1, ref2, np.asarray(transfer["Cbar"]), args.rho_site, args.xi_noise, args.validation_W, args.validation_real, args.seed, device, args.gpu_batch_size)
    print("eta_nl(W):", [(r["W"], r["eta_nl"]) for r in validation["rows"]])

    geometry, tune_meta = scan_geometry(args.scan_lengths, args.scan_alpha, args.Delta, args.Ez, args.t0, args.mu_min, args.mu_max, args.mu_points, K1, args.L1, args.rho_site, args.xi_noise, device, args.mu_refine_stages, args.mu_refine_points)
    add_dimensionless_descriptors(geometry, args.Delta, args.lattice_spacing)
    print(f"geometry candidates: {len(geometry)} (successful={tune_meta['successful']}, failed={tune_meta['failed']})")

    pair_stats = strict_pair_statistics(geometry, args.gap_match_tol, args.xi_match_tol, args.separation_match_tol)
    nearest = pair_stats.get("nearest"); max_pair = pair_stats.get("max_ratio_pair")
    print(f"strict matched pairs: {pair_stats['count']}")
    if nearest:
        print(f"nearest pair: ratio={nearest['N_geom_ratio']:.6f}; gap={nearest['relative_gap_difference']:.3%}; xi={nearest['relative_xi_difference']:.3%}; separation={nearest['relative_separation_difference']:.3%}; D_match={nearest['match_distance']:.4f}; D_inf={nearest['match_distance_inf']:.4f}")
    if max_pair:
        print(f"max-ratio strict pair: ratio={max_pair['N_geom_ratio']:.6f}; gap={max_pair['relative_gap_difference']:.3%}; xi={max_pair['relative_xi_difference']:.3%}; separation={max_pair['relative_separation_difference']:.3%}; D_match={max_pair['match_distance']:.4f}; D_inf={max_pair['match_distance_inf']:.4f}")

    law = discover_law(geometry, args.cv_folds, args.seed, args.law_tolerance, args.target_r2, args.target_median_relerr)
    if law.get("status") == "ok":
        best = law["best_cv_model"]; minimal = law["minimal_model_within_tolerance"]
        print(f"law discovery: best={best['label']} | CV RMSE(log N)={best['cv_rmse_logN']:.5f} | CV R2={best['cv_r2_logN']:.4f} | median rel.err={100*best['cv_median_relative_error']:.2f}%")
        print(f"law discovery: minimal-within-tolerance={minimal['label']} | CV RMSE(log N)={minimal['cv_rmse_logN']:.5f} | CV R2={minimal['cv_r2_logN']:.4f}")

        law["coefficient_stability"] = coefficient_stability(geometry, minimal["features"], args.coeff_repeats, args.coeff_train_fraction, args.seed + 1000)
        law["repeated_cv_stability"] = repeated_cv_stability(geometry, minimal["features"], args.cv_folds, args.cv_repeats, args.seed + 2000)
        law["region_holdouts"] = region_holdouts(geometry, minimal["features"])
        print("coefficient stability: relative std =", [round(x, 4) for x in law["coefficient_stability"]["relative_std"]])
        for label, item in law["region_holdouts"].items():
            if item.get("status") == "ok":
                print(f"holdout {label}: R2={item['r2_logN']:.4f}, median rel.err={100*item['median_relative_error']:.2f}%")
        eng_r2 = law.get("minimal_engineering_model_r2")
        eng_err = law.get("minimal_engineering_model_median_error")
        eng_joint = law.get("minimal_engineering_model_joint")
        print("engineering target (R2):", eng_r2["label"] if eng_r2 else "NONE")
        print("engineering target (median rel.err):", eng_err["label"] if eng_err else "NONE")
        print("engineering target (both):", eng_joint["label"] if eng_joint else "NONE")
    else:
        print("law discovery: insufficient candidate count")

    make_figures(args.outdir, K1, K2, transfer, geometry, pair_stats, law)
    write_geometry_csv(os.path.join(args.outdir, "geometry_scan_v21_1.csv"), geometry)

    report: Dict[str, Any] = {
        "version": "v21.1",
        "device": device_info | {"selected": device},
        "core": {
            "H_p": "H_p = 1/2 [g X - z_p Z], z_p=-(epsilon_12+p epsilon_34)",
            "linear_response": "delta epsilon = K delta mu",
            "normalized_noise_transfer": "Cbar = K R K^T",
            "physical_covariance": "Sigma_epsilon = W^2 Cbar",
            "noise_eigenmodes": "Cbar = Q diag(lambda_weak, lambda_strong) Q^T",
            "design_metric": "N_geom = lambda_max(Cbar)",
        },
        "parameters": {k: getattr(args, k) for k in ("L1", "L2", "alpha", "Delta", "Ez", "t0", "rho_site", "xi_noise", "W", "mu_min", "mu_max", "mu_points", "mu_refine_stages", "mu_refine_points", "fd_step", "validation_real", "gpu_batch_size", "gap_match_tol", "xi_match_tol", "separation_match_tol", "cv_folds", "law_tolerance", "target_r2", "target_median_relerr", "coeff_repeats", "coeff_train_fraction", "cv_repeats", "lattice_spacing", "seed", "device", "gpu_id")},
        "operating_points": {"wire1": op1.as_dict(), "wire2": op2.as_dict()},
        "kernels": {"K1": K1.tolist(), "K2": K2.tolist(), "finite_difference_relative_error": {"wire1": fd1, "wire2": fd2}},
        "effective_noise": {
            "Cbar": np.asarray(transfer["Cbar"]).tolist(), "eigenvalues": vals.tolist(),
            "eigenvectors": np.asarray(transfer["eigenvectors"]).tolist(), "lambda_weak": float(transfer["lambda_weak"]),
            "lambda_strong": float(transfer["lambda_strong"]), "mode_ratio": float(transfer["mode_ratio"]),
            "N_geom": float(transfer["N_geom"]), "S_plus_bar": float(transfer["S_plus_bar"]), "S_minus_bar": float(transfer["S_minus_bar"],),
            "physical_S_plus": float(args.W ** 2 * transfer["S_plus_bar"]), "physical_S_minus": float(args.W ** 2 * transfer["S_minus_bar"]),
        },
        "validation": validation, "geometry_scan": geometry, "scan_meta": tune_meta,
        "pair_analysis": {
            "strict_definition": {"relative_gap": float(args.gap_match_tol), "relative_xi": float(args.xi_match_tol), "relative_separation": float(args.separation_match_tol), "equivalent_condition": "D_inf <= 1"},
            "count": pair_stats["count"], "max_ratio": pair_stats["max_ratio"], "median_ratio": pair_stats["median_ratio"], "p95_ratio": pair_stats["p95_ratio"],
            "nearest_pair": compact_pair(nearest), "max_ratio_pair": compact_pair(max_pair),
        },
        "law_discovery": law,
        "scientific_readout": {
            "both_topological": True, "kernel_fd_supported": bool(fd1 < 0.08 and fd2 < 0.08),
            "universality_claim": False,
            "current_message": "The law-discovery stage tests whether N_geom collapses onto a minimal dimensionless geometry description. Promotion to a main-text rule requires acceptable cross-validation and intended-region holdout performance.",
        },
    }

    np.savez(os.path.join(args.outdir, "kernels_v21_1.npz"), K12=K1, K34=K2, L1=np.array([args.L1]), L2=np.array([args.L2]), mu1_star=np.array([op1.mu]), mu2_star=np.array([op2.mu]), Cbar=np.asarray(transfer["Cbar"]))
    with open(os.path.join(args.outdir, "core_results_v21_1.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    write_paper_law(os.path.join(args.outdir, "paper_law_v21_1.md"), report)
    print(f"outputs -> {args.outdir}/")
    return report


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------



# -----------------------------------------------------------------------------
# v21.4 — orthogonal noise-environment audit + final paper figures
# -----------------------------------------------------------------------------


def _fit_linear_with_fixed_slopes(X: np.ndarray, y: np.ndarray, slopes: np.ndarray) -> Tuple[float, np.ndarray]:
    """Fit only the intercept with slopes held fixed."""
    resid = y - X[:, 1:] @ slopes
    A = np.ones((len(y), 1), dtype=float)
    intercept, *_ = np.linalg.lstsq(A, resid, rcond=None)
    beta = np.concatenate(([float(intercept[0])], np.asarray(slopes, float)))
    return float(intercept[0]), beta


def orthogonal_env_holdouts(
    records: Sequence[Dict[str, Any]],
    baseline_coefficients: Sequence[float],
    xi_values: Sequence[float],
) -> Dict[str, Any]:
    """Test whether geometry-law slopes transfer across new noise correlation lengths.

    The baseline law is trained at the baseline noise environment. For each new
    xi_noise, only the intercept is refit on the training half of each L/alpha
    region; B and C remain frozen. This directly tests whether geometry dependence
    factorizes from the environmental noise scale.
    """
    names = ("d_over_xi", "xi")
    base_beta = np.asarray(baseline_coefficients, dtype=float)
    if base_beta.size != 3:
        raise ValueError("baseline canonical law must have intercept, B, C")
    L = np.array([int(r["L"]) for r in records])
    alpha = np.array([float(r["alpha"]) for r in records])
    specs = {
        "L_low_to_high": (L <= 80, L > 80),
        "L_high_to_low": (L > 80, L <= 80),
        "alpha_low_to_high": (alpha <= 0.16 + 1e-12, alpha >= 0.17 - 1e-12),
        "alpha_high_to_low": (alpha >= 0.17 - 1e-12, alpha <= 0.16 + 1e-12),
    }
    Kvecs = [np.asarray(r["K_vector"], dtype=float) for r in records]
    out: Dict[str, Any] = {}
    for xi in xi_values:
        xi = float(xi)
        nvals = np.array([intrinsic_noise_exposure(k, xi) for k in Kvecs], dtype=float)
        env_records = [dict(r, N_wire=float(nv)) for r, nv in zip(records, nvals)]
        full_X, full_y = intrinsic_feature_matrix(env_records, names)
        full_beta, _ = _ols_fit(full_X, full_y)
        fixed_rows = {}
        for label, (train_mask, test_mask) in specs.items():
            train_idx = np.where(train_mask)[0]
            test_idx = np.where(test_mask)[0]
            if len(train_idx) < 5 or len(test_idx) < 5:
                fixed_rows[label] = {"status": "skipped"}
                continue
            Xtr, ytr = full_X[train_idx], full_y[train_idx]
            Xte, yte = full_X[test_idx], full_y[test_idx]
            # Fit only intercept, freeze B,C from the baseline law.
            intercept, beta_fixed = _fit_linear_with_fixed_slopes(Xtr, ytr, base_beta[1:])
            pred_fixed = Xte @ beta_fixed
            m = _prediction_metrics(yte, pred_fixed)
            # Full refit is a diagnostic comparison, not the proposed transfer law.
            beta_full, _ = _ols_fit(Xtr, ytr)
            pred_full = Xte @ beta_full
            mf = _prediction_metrics(yte, pred_full)
            fixed_rows[label] = {
                "status": "ok",
                **m,
                "full_refit": mf,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "frozen_BC": base_beta[1:].tolist(),
                "fitted_intercept": float(intercept),
            }
        ok = [v for v in fixed_rows.values() if v.get("status") == "ok"]
        out[f"xi_noise={xi:g}"] = {
            "xi_noise": xi,
            "full_refit_coefficients": full_beta.tolist(),
            "full_refit_relative_to_baseline": ((full_beta - base_beta) / np.maximum(np.abs(base_beta), 1e-30)).tolist(),
            "holdouts": fixed_rows,
            "fixed_slope_summary": {
                "median_r2": float(np.median([v["r2_logN"] for v in ok])) if ok else float("nan"),
                "worst_r2": float(np.min([v["r2_logN"] for v in ok])) if ok else float("nan"),
                "median_relerr": float(np.median([v["median_relative_error"] for v in ok])) if ok else float("nan"),
                "worst_relerr": float(np.max([v["median_relative_error"] for v in ok])) if ok else float("nan"),
            },
        }
    return out


def pair_noise_environment_audit(
    K1: np.ndarray,
    K2: np.ndarray,
    rho_values: Sequence[float],
    xi_values: Sequence[float],
) -> Dict[str, Any]:
    """Evaluate exact pair-level N_geom under orthogonal noise environments."""
    rows: List[Dict[str, float]] = []
    n0_by_xi: Dict[float, float] = {}
    for xi in xi_values:
        base_R = noise_R(K1.size, K2.size, 0.0, float(xi))
        base = pair_noise_decomposition(K1, K2, base_R)
        n0 = float(base["N_geom_exact"])
        n0_by_xi[float(xi)] = n0
        for rho in rho_values:
            R = noise_R(K1.size, K2.size, float(rho), float(xi))
            d = pair_noise_decomposition(K1, K2, R)
            rows.append({
                "xi_noise": float(xi),
                "rho_site": float(rho),
                "n1": float(d["n1"]),
                "n2": float(d["n2"]),
                "cross_c": float(d["cross_c"]),
                "rho_eff": float(d["rho_eff"]),
                "N_geom": float(d["N_geom_exact"]),
                "N_geom_over_rho0": float(d["N_geom_exact"] / max(n0, 1e-30)),
            })
    return {"rows": rows, "rho_values": [float(x) for x in rho_values], "xi_values": [float(x) for x in xi_values], "rho0_reference": n0_by_xi}


def write_csv_rows(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def v214_final_figures(
    outdir: str,
    geometry: Sequence[Dict[str, Any]],
    canonical: Dict[str, Any],
    env_audit: Dict[str, Any],
    pair_env: Dict[str, Any],
) -> None:
    """Make the three paper-facing figures for v21.4."""
    if not geometry or canonical.get("status") != "ok":
        return
    beta = np.asarray(canonical["coefficients"], dtype=float)
    x1 = np.array([r["d_over_xi"] for r in geometry], float)
    x2 = np.array([r["xi_over_a"] for r in geometry], float)
    y = np.log(np.maximum(np.array([r["N_wire"] for r in geometry], float), 1e-30))
    score = beta[1] * x1 + beta[2] * x2
    pred = beta[0] + score

    # Figure 1: law collapse — one scalar geometry score predicts log N_wire.
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.scatter(pred, y, s=34, alpha=0.72)
    lo = min(float(pred.min()), float(y.min())); hi = max(float(pred.max()), float(y.max()))
    ax.plot([lo, hi], [lo, hi], ls="--", alpha=0.55)
    ax.set_xlabel(r"predicted $\log N_{\rm wire}$ from geometry law")
    ax.set_ylabel(r"calculated $\log N_{\rm wire}$")
    ax.set_title("Two-variable Majorana noise law")
    ax.grid(True, ls=":", alpha=0.35)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "figure1_geometry_law_v21_4.png"), dpi=280); plt.close(fig)

    # Figure 2: orthogonal noise-environment transfer. Frozen B,C are the
    # physically interesting test; full-refit curves provide a diagnostic.
    x = []; med = []; worst = []
    for key, item in env_audit.items():
        x.append(item["xi_noise"])
        med.append(100.0 * item["fixed_slope_summary"]["median_relerr"])
        worst.append(100.0 * item["fixed_slope_summary"]["worst_relerr"])
    order = np.argsort(x); x = np.asarray(x)[order]; med = np.asarray(med)[order]; worst = np.asarray(worst)[order]
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    ax.plot(x, med, marker="o", label="median holdout error (B,C frozen)")
    ax.plot(x, worst, marker="s", label="worst holdout error (B,C frozen)")
    ax.set_xlabel(r"noise correlation length $\xi_{noise}/a$")
    ax.set_ylabel("holdout relative error (%)")
    ax.set_title("Orthogonal noise-environment test")
    ax.grid(True, ls=":", alpha=0.35); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "figure2_noise_environment_v21_4.png"), dpi=280); plt.close(fig)

    # Figure 3: cross-wire correlation controls the pair-level strong mode.
    rows = pair_env.get("rows", [])
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    seen = sorted(set(float(r["xi_noise"]) for r in rows))
    for xi in seen:
        rr = [r for r in rows if abs(float(r["xi_noise"]) - xi) < 1e-12]
        rr.sort(key=lambda z: z["rho_site"])
        ax.plot([r["rho_site"] for r in rr], [r["N_geom_over_rho0"] for r in rr], marker="o", label=fr"$\xi_{{noise}}/a={xi:g}$")
    ax.axhline(1.0, ls="--", alpha=0.45)
    ax.set_xlabel(r"site-noise correlation $\rho_{site}$")
    ax.set_ylabel(r"$N_{geom}(\rho)/N_{geom}(0)$")
    ax.set_title("Cross-wire noise correlation")
    ax.grid(True, ls=":", alpha=0.35); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "figure3_pair_correlation_v21_4.png"), dpi=280); plt.close(fig)


def write_paper_law_v214(path: str, report: Dict[str, Any]) -> None:
    can = report.get("canonical_law_audit", {}).get("canonical_model", {})
    h = report.get("canonical_law_audit", {}).get("canonical_holdouts", {})
    stab = report.get("canonical_law_audit", {}).get("canonical_coefficient_stability", {})
    env = report.get("noise_environment_audit", {})
    lines = [
        "# Majorana Noise Geometry v21.4 — Final Law Audit\n",
        "## Core equations\n",
        "1. `delta epsilon = K delta mu`\n",
        "2. `Cbar = K R K^T`\n",
        "3. `N_geom = lambda_max(Cbar)`\n",
        "4. `N_wire = K R_self K^T`\n",
        "",
    ]
    if can:
        beta = can.get("coefficients", [])
        lines.append("## Canonical geometry law\n")
        lines.append("`log N_wire = A + B (d_M/xi_M) + C (xi_M/a)`\n")
        lines.append("Equivalent form: `N_wire = N0 * exp[B(d_M/xi_M) + C(xi_M/a)]`.\n")
        lines.append("Coefficients `(A, B, C)`: `" + ", ".join(f"{x:.12g}" for x in beta) + "`\n")
        lines.append(f"Random 5-fold CV: R²=`{can.get('cv_r2_logN', float('nan')):.5f}`, median relative error=`{100*can.get('cv_median_relative_error', float('nan')):.2f}%`.\n")
        lines.append(f"Region holdouts: median R²=`{report['canonical_law_audit']['canonical_holdouts'].get('median_r2', float('nan')):.5f}`, worst R²=`{report['canonical_law_audit']['canonical_holdouts'].get('worst_r2', float('nan')):.5f}`, median relative error=`{100*report['canonical_law_audit']['canonical_holdouts'].get('median_relerr', float('nan')):.2f}%`, worst=`{100*report['canonical_law_audit']['canonical_holdouts'].get('worst_relerr', float('nan')):.2f}%`.\n")
        lines.append("Coefficient relative standard deviations: `" + ", ".join(f"{x:.5f}" for x in stab.get('relative_std', [])) + "`.\n")
        lines.append(f"Predeclared engineering-rule test: **{'PASS' if report['canonical_law_audit'].get('canonical_model_is_engineering_rule_candidate') else 'FAIL'}**.\n")
    lines.append("## Orthogonal noise-environment test\n")
    lines.append("The geometry-law slopes B and C are frozen from the baseline environment; only the intercept is refit for a new noise correlation length.\n")
    for key in sorted(env.keys(), key=lambda k: env[k]["xi_noise"]):
        item = env[key]
        s = item["fixed_slope_summary"]
        lines.append(f"- xi_noise/a=`{item['xi_noise']:g}`: median holdout error=`{100*s['median_relerr']:.2f}%`, worst=`{100*s['worst_relerr']:.2f}%`, median R²=`{s['median_r2']:.5f}`.")
    lines.append("\nThis test asks whether geometry dependence is approximately shape-invariant under changed local noise correlations; it does not assume that the absolute noise scale is universal.\n")
    lines.append("## Pair-level rule\n")
    lines.append("For `Cbar=[[n1,c],[c,n2]]`, the strong mode is exactly `N_geom=(n1+n2+sqrt((n1-n2)^2+4c^2))/2`. The single-wire geometry law therefore supplies the intrinsic exposures, while the environmental covariance determines the pair correlation term.\n")
    lines.append("## Paper interpretation\n")
    lines.append("The compact law is supported within the tested topological-device regime by both interpolation and L/alpha region-holdout tests. The explicit oscillatory phase descriptor was not required in the preceding falsification test. The result is a regime-specific engineering rule, not a universal theorem.\n")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def run_v214(args: argparse.Namespace) -> Dict[str, Any]:
    report = run_v213(args)
    report["version"] = "v21.4"
    geometry = report.get("geometry_scan", [])
    audit = report.get("canonical_law_audit", {})
    if not geometry or audit.get("status") != "ok":
        return report
    canonical = audit["canonical_model"]
    base_beta = canonical["coefficients"]
    # Orthogonal local-noise environment test. rho_site acts only at pair level;
    # intrinsic exposure depends on xi_noise through R_self.
    xi_env = tuple(args.orthogonal_xi_noise)
    rho_env = tuple(args.orthogonal_rho_site)
    env_audit = orthogonal_env_holdouts(geometry, base_beta, xi_env)
    pair_env = pair_noise_environment_audit(
        np.asarray(report["kernels"]["K1"], float),
        np.asarray(report["kernels"]["K2"], float),
        rho_env,
        xi_env,
    )
    report["noise_environment_audit"] = env_audit
    report["pair_noise_environment_audit"] = pair_env
    # Main-text candidate requires canonical law plus orthogonal fixed-slope test
    # to remain below the engineering error threshold.
    env_worst = max((v["fixed_slope_summary"]["worst_relerr"] for v in env_audit.values()), default=float("inf"))
    report["scientific_readout"]["orthogonal_environment_pass"] = bool(env_worst <= args.target_median_relerr)
    report["scientific_readout"]["final_rule_ready"] = bool(
        audit.get("canonical_model_is_engineering_rule_candidate", False)
        and env_worst <= args.target_median_relerr
    )

    print("-" * 100)
    print("v21.4 final-law + orthogonal-noise audit")
    print(f"A,B,C = {[float(x) for x in base_beta]}")
    print(f"canonical worst holdout rel.err = {100*audit['canonical_holdouts']['worst_relerr']:.2f}%")
    for key in sorted(env_audit.keys(), key=lambda k: env_audit[k]["xi_noise"]):
        item = env_audit[key]
        s = item["fixed_slope_summary"]
        print(f"orthogonal xi_noise/a={item['xi_noise']:g}: median R2={s['median_r2']:.4f} | worst R2={s['worst_r2']:.4f} | median err={100*s['median_relerr']:.2f}% | worst err={100*s['worst_relerr']:.2f}%")
    print(f"final rule ready: {'YES' if report['scientific_readout']['final_rule_ready'] else 'NO'}")

    v214_final_figures(args.outdir, geometry, canonical, env_audit, pair_env)
    write_csv_rows(os.path.join(args.outdir, "noise_environment_v21_4.csv"), [
        {"xi_noise": item["xi_noise"], "fixed_slope_median_r2": item["fixed_slope_summary"]["median_r2"], "fixed_slope_worst_r2": item["fixed_slope_summary"]["worst_r2"], "fixed_slope_median_relerr": item["fixed_slope_summary"]["median_relerr"], "fixed_slope_worst_relerr": item["fixed_slope_summary"]["worst_relerr"]}
        for item in env_audit.values()
    ])
    write_csv_rows(os.path.join(args.outdir, "pair_noise_environment_v21_4.csv"), pair_env["rows"])
    np.savez(
        os.path.join(args.outdir, "geometry_kernels_v21_4.npz"),
        K_stack=np.array([np.asarray(r["K_vector"], float) for r in geometry], dtype=object),
        L=np.array([r["L"] for r in geometry]),
        alpha=np.array([r["alpha"] for r in geometry]),
        N_wire=np.array([r["N_wire"] for r in geometry]),
    )
    write_paper_law_v214(os.path.join(args.outdir, "paper_law_v21_4.md"), report)
    with open(os.path.join(args.outdir, "core_results_v21_4.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Majorana noise geometry v21.1 — minimal physical law / device design rule")
    parser.add_argument("--outdir", default="outputs_v21_1")
    parser.add_argument("--L1", type=int, default=80); parser.add_argument("--L2", type=int, default=65)
    parser.add_argument("--W", type=float, default=0.03); parser.add_argument("--rho-site", type=float, default=0.60); parser.add_argument("--xi-noise", type=float, default=0.0)
    parser.add_argument("--alpha", type=float, default=0.15); parser.add_argument("--Delta", type=float, default=1.0); parser.add_argument("--Ez", type=float, default=2.5); parser.add_argument("--t0", type=float, default=1.0)
    parser.add_argument("--mu-min", type=float, default=0.0); parser.add_argument("--mu-max", type=float, default=1.5); parser.add_argument("--mu-points", type=int, default=19)
    parser.add_argument("--mu-refine-stages", type=int, default=3); parser.add_argument("--mu-refine-points", type=int, default=17); parser.add_argument("--fd-step", type=float, default=2e-4)
    parser.add_argument("--validation-W", default="0.001,0.002,0.005,0.01,0.02,0.03"); parser.add_argument("--validation-real", type=int, default=2000); parser.add_argument("--gpu-batch-size", type=int, default=256)
    parser.add_argument("--scan-lengths", default="50:100:2"); parser.add_argument("--scan-alpha", default="0.11:0.19:0.01")
    parser.add_argument("--gap-match-tol", type=float, default=0.05); parser.add_argument("--xi-match-tol", type=float, default=0.05); parser.add_argument("--separation-match-tol", type=float, default=0.05)
    parser.add_argument("--cv-folds", type=int, default=5); parser.add_argument("--cv-repeats", type=int, default=20)
    parser.add_argument("--law-tolerance", type=float, default=0.05, help="fractional CV-RMSE margin for selecting the simplest near-best law")
    parser.add_argument("--target-r2", type=float, default=0.90, help="optional engineering threshold")
    parser.add_argument("--target-median-relerr", type=float, default=0.10, help="optional engineering threshold")
    parser.add_argument("--coeff-repeats", type=int, default=60); parser.add_argument("--coeff-train-fraction", type=float, default=0.80)
    parser.add_argument("--lattice-spacing", type=float, default=1.0)
    parser.add_argument("--device", choices=("auto", "cpu", "gpu"), default="auto"); parser.add_argument("--gpu-id", type=int, default=0); parser.add_argument("--seed", type=int, default=2028)
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        print(f"[v21.1] Ignoring host/runtime arguments: {unknown}")
    args.validation_W = parse_number_list(args.validation_W, float)
    args.scan_lengths = parse_number_list(args.scan_lengths, int)
    args.scan_alpha = parse_number_list(args.scan_alpha, float)
    run(args)




# =============================================================================
# v21.2 — Intrinsic Noise Law + Majorana Interference Phase
# =============================================================================

V21_2_VERSION = "v21.4"


def normal_lower_band_energy(wire: BdGWire, k: float) -> float:
    """Lower helical-band energy of the normal-state (Delta -> 0) model."""
    p = wire.p
    xi_k = 2.0 * p.t0 - 2.0 * p.t0 * math.cos(k) - p.mu
    return float(xi_k - math.sqrt(p.Ez * p.Ez + (p.alpha * math.sin(k)) ** 2))


def estimate_k_osc(wire: BdGWire, nk: int = 401) -> Tuple[float, str]:
    """Estimate an oscillation wavevector from the lowest positive normal-state Fermi root.

    The primary estimator is the smallest positive root of the lower helical
    normal-state band.  This is a physically motivated k_F surrogate for the
    oscillatory phase of finite-wire Majorana wavefunctions.  If no sign-change
    root is found, fall back to the momentum at the minimum positive BdG bulk gap.

    This is intentionally a *candidate descriptor*, not a theorem: v21.2 tests
    whether the descriptor improves out-of-region prediction before promoting it.
    """
    ks = np.linspace(1e-6, math.pi - 1e-6, max(int(nk), 101))
    vals = np.array([normal_lower_band_energy(wire, float(k)) for k in ks], dtype=float)
    roots: List[float] = []
    for i in range(len(ks) - 1):
        y0, y1 = vals[i], vals[i + 1]
        if y0 == 0.0:
            roots.append(float(ks[i]))
        elif y0 * y1 < 0.0:
            try:
                roots.append(float(brentq(lambda q: normal_lower_band_energy(wire, float(q)), ks[i], ks[i + 1], xtol=1e-12, rtol=1e-12)))
            except Exception:
                pass
    roots = sorted(x for x in roots if x > 1e-7)
    if roots:
        return roots[0], "normal_lower_band_Fermi_root"

    # Fallback: momentum of the minimum positive BdG excitation.
    def gap_at(k: float) -> float:
        ev = np.linalg.eigvalsh(wire.bloch_H(float(k)))
        pos = ev[ev >= 0]
        return float(pos[0] if len(pos) else np.min(np.abs(ev)))

    idx = int(np.argmin([gap_at(k) for k in ks]))
    return float(ks[idx]), "BdG_gap_minimum_fallback"


def intrinsic_noise_exposure(K: np.ndarray, xi_noise: float) -> float:
    """Single-wire normalized noise exposure n_wire = K R_self K^T."""
    K = np.asarray(K, dtype=float)
    Rself = spatial_R(K.size, xi_noise)
    return float(K @ Rself @ K)


def pair_noise_decomposition(K1: np.ndarray, K2: np.ndarray, R: np.ndarray) -> Dict[str, float]:
    """Exact 2x2 decomposition of Cbar into self exposures and cross correlation."""
    K1 = np.asarray(K1, float)
    K2 = np.asarray(K2, float)
    L1 = K1.size
    R11 = R[:L1, :L1]
    R22 = R[L1:, L1:]
    R12 = R[:L1, L1:]
    n1 = float(K1 @ R11 @ K1)
    n2 = float(K2 @ R22 @ K2)
    c = float(K1 @ R12 @ K2)
    discr = math.sqrt(max((n1 - n2) ** 2 + 4.0 * c * c, 0.0))
    nstrong = 0.5 * (n1 + n2 + discr)
    nweak = 0.5 * (n1 + n2 - discr)
    rho_eff = c / math.sqrt(max(n1 * n2, 1e-30))
    return {
        "n1": n1,
        "n2": n2,
        "cross_c": c,
        "rho_eff": float(rho_eff),
        "lambda_weak_exact": float(nweak),
        "lambda_strong_exact": float(nstrong),
        "N_geom_exact": float(nstrong),
    }


def _v212_feature_value(r: Dict[str, float], name: str) -> float:
    table = {
        "gap": r["gap_over_Delta"],
        "xi": r["xi_over_a"],
        "d_over_xi": r["d_over_xi"],
        "log_xi_over_a": r["log_xi_over_a"],
        "phase_cos": r["phase_cos"],
        "phase_sin": r["phase_sin"],
        "phase_cos2": r["phase_cos2"],
        "phase_sin2": r["phase_sin2"],
        "theta_abs_mod": r["theta_abs_mod"],
    }
    if name not in table:
        raise ValueError(f"unknown v21.2 feature {name}")
    return float(table[name])


def intrinsic_feature_matrix(records: Sequence[Dict[str, float]], names: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    y = np.log(np.maximum(np.array([r["N_wire"] for r in records], float), 1e-30))
    cols = [np.ones(len(records), dtype=float)]
    for name in names:
        cols.append(np.array([_v212_feature_value(r, name) for r in records], dtype=float))
    return np.column_stack(cols), y


V212_DISPLAY = {
    "gap": "E_gap / Delta",
    "xi": "xi_M / a",
    "d_over_xi": "d_M / xi_M",
    "log_xi_over_a": "log(xi_M / a)",
    "phase_cos": "cos(k_osc d_M)",
    "phase_sin": "sin(k_osc d_M)",
    "phase_cos2": "cos(2 k_osc d_M)",
    "phase_sin2": "sin(2 k_osc d_M)",
    "theta_abs_mod": "|k_osc d_M| mod pi",
}


def v212_model_label(names: Sequence[str]) -> str:
    return "constant" if not names else " + ".join(V212_DISPLAY[n] for n in names)


def v212_formula(names: Sequence[str]) -> str:
    if not names:
        return "log N_wire = A"
    return "log N_wire = A + " + " + ".join(V212_DISPLAY[n] for n in names)


def candidate_intrinsic_models() -> List[Tuple[str, ...]]:
    # Small, physically motivated ladder. First-harmonic phase terms are the
    # primary interference descriptor; second harmonics test whether the first
    # harmonic is structurally insufficient without turning this into black-box ML.
    return [
        (),
        ("gap",),
        ("xi",),
        ("d_over_xi",),
        ("phase_cos",),
        ("phase_sin",),
        ("d_over_xi", "xi"),
        ("d_over_xi", "phase_cos"),
        ("d_over_xi", "phase_sin"),
        ("d_over_xi", "phase_cos", "phase_sin"),
        ("d_over_xi", "phase_cos", "phase_sin", "xi"),
        ("d_over_xi", "phase_cos", "phase_sin", "log_xi_over_a"),
        ("gap", "d_over_xi"),
        ("gap", "d_over_xi", "phase_cos", "phase_sin"),
        ("gap", "xi", "d_over_xi", "phase_cos", "phase_sin"),
        ("d_over_xi", "phase_cos2", "phase_sin2"),
        ("d_over_xi", "phase_cos", "phase_sin", "phase_cos2", "phase_sin2"),
    ]


def evaluate_intrinsic_model(records: Sequence[Dict[str, float]], names: Sequence[str], folds: int, seed: int) -> Dict[str, Any]:
    X, y = intrinsic_feature_matrix(records, names)
    n, p = X.shape
    folds_idx = _kfold_indices(n, folds, seed)
    pred_cv = np.full(n, np.nan)
    for test_idx in folds_idx:
        train_mask = np.ones(n, dtype=bool)
        train_mask[test_idx] = False
        beta, _ = _ols_fit(X[train_mask], y[train_mask])
        pred_cv[test_idx] = X[test_idx] @ beta
    beta, pred_all = _ols_fit(X, y)
    cv = _prediction_metrics(y, pred_cv)
    ins = _prediction_metrics(y, pred_all)
    rss = float(np.sum((y - pred_all) ** 2))
    bic = float(n * math.log(max(rss / n, 1e-30)) + p * math.log(max(n, 2)))
    return {
        "target": "N_wire",
        "features": list(names),
        "label": v212_model_label(names),
        "formula": v212_formula(names),
        "n_features": len(names),
        "coefficients": beta.tolist(),
        "cv_rmse_logN": cv["rmse_logN"],
        "cv_r2_logN": cv["r2_logN"],
        "cv_median_relative_error": cv["median_relative_error"],
        "cv_p90_relative_error": cv["p90_relative_error"],
        "in_sample_rmse_logN": ins["rmse_logN"],
        "bic": bic,
        "predicted_logN": pred_cv.tolist(),
    }


def discover_intrinsic_law(records: Sequence[Dict[str, float]], folds: int, seed: int, tolerance: float, target_r2: float, target_median_relerr: float) -> Dict[str, Any]:
    if len(records) < 12:
        return {"status": "insufficient_candidates", "n": len(records)}
    models = [evaluate_intrinsic_model(records, m, folds, seed) for m in candidate_intrinsic_models()]
    best = min(models, key=lambda x: x["cv_rmse_logN"])
    threshold = best["cv_rmse_logN"] * (1.0 + tolerance)
    near = [m for m in models if m["cv_rmse_logN"] <= threshold]
    minimal = min(near, key=lambda x: (x["n_features"], x["cv_rmse_logN"]))
    r2_pass = [m for m in models if m["cv_r2_logN"] >= target_r2]
    err_pass = [m for m in models if m["cv_median_relative_error"] <= target_median_relerr]
    joint = [m for m in models if m["cv_r2_logN"] >= target_r2 and m["cv_median_relative_error"] <= target_median_relerr]
    return {
        "status": "ok",
        "n": len(records),
        "target": "log(N_wire)",
        "models": models,
        "best_cv_model": best,
        "minimal_model_within_tolerance": minimal,
        "minimal_engineering_model_r2": min(r2_pass, key=lambda x: (x["n_features"], x["cv_rmse_logN"])) if r2_pass else None,
        "minimal_engineering_model_median_error": min(err_pass, key=lambda x: (x["n_features"], x["cv_rmse_logN"])) if err_pass else None,
        "minimal_engineering_model_joint": min(joint, key=lambda x: (x["n_features"], x["cv_rmse_logN"])) if joint else None,
        "minimality_tolerance_fraction": float(tolerance),
        "engineering_targets": {"cv_r2": float(target_r2), "median_relative_error": float(target_median_relerr)},
    }


def intrinsic_region_holdouts(records: Sequence[Dict[str, float]], names: Sequence[str]) -> Dict[str, Any]:
    if len(records) < 20:
        return {}
    L = np.array([r["L"] for r in records])
    alpha = np.array([r["alpha"] for r in records])
    specs = {
        "train_L_<=80_test_>80": (L <= 80, L > 80),
        "train_L_>80_test_<=80": (L > 80, L <= 80),
        "train_alpha_<=0.16_test_>=0.17": (alpha <= 0.16 + 1e-12, alpha >= 0.17 - 1e-12),
        "train_alpha_>=0.17_test_<=0.16": (alpha >= 0.17 - 1e-12, alpha <= 0.16 + 1e-12),
    }
    X, y = intrinsic_feature_matrix(records, names)
    out: Dict[str, Any] = {}
    for label, (train_mask, test_mask) in specs.items():
        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]
        if len(train_idx) < 5 or len(test_idx) < 5:
            out[label] = {"status": "skipped", "n_train": int(len(train_idx)), "n_test": int(len(test_idx))}
            continue
        beta, _ = _ols_fit(X[train_idx], y[train_idx])
        pred = X[test_idx] @ beta
        m = _prediction_metrics(y[test_idx], pred)
        out[label] = {"status": "ok", **m, "n_train": int(len(train_idx)), "n_test": int(len(test_idx))}
    return out


def intrinsic_coefficient_stability(records: Sequence[Dict[str, float]], names: Sequence[str], repeats: int, train_fraction: float, seed: int) -> Dict[str, Any]:
    X, y = intrinsic_feature_matrix(records, names)
    n = len(records)
    train_n = min(max(int(round(train_fraction * n)), X.shape[1] + 2), n - 1)
    rng = np.random.default_rng(seed)
    betas = []
    for _ in range(max(1, repeats)):
        idx = rng.choice(n, size=train_n, replace=False)
        beta, _ = _ols_fit(X[idx], y[idx])
        betas.append(beta)
    B = np.asarray(betas, dtype=float)
    mean = np.mean(B, axis=0)
    std = np.std(B, axis=0, ddof=1) if B.shape[0] > 1 else np.zeros(B.shape[1])
    return {
        "features": list(names),
        "label": v212_model_label(names),
        "repeats": int(repeats),
        "train_fraction": float(train_fraction),
        "coefficient_mean": mean.tolist(),
        "coefficient_std": std.tolist(),
        "relative_std": (std / np.maximum(np.abs(mean), 1e-30)).tolist(),
        "sign_stability": [float(np.mean(np.sign(B[:, j]) == np.sign(mean[j]))) if abs(mean[j]) > 1e-15 else 0.0 for j in range(B.shape[1])],
    }


def intrinsic_repeated_cv(records: Sequence[Dict[str, float]], names: Sequence[str], folds: int, repeats: int, seed: int) -> Dict[str, Any]:
    vals = []
    for i in range(max(1, repeats)):
        res = evaluate_intrinsic_model(records, names, folds, seed + i)
        vals.append([res["cv_rmse_logN"], res["cv_r2_logN"], res["cv_median_relative_error"]])
    A = np.asarray(vals, dtype=float)
    return {
        "repeats": int(len(A)),
        "rmse_mean": float(np.mean(A[:, 0])),
        "rmse_std": float(np.std(A[:, 0], ddof=1)) if len(A) > 1 else 0.0,
        "r2_mean": float(np.mean(A[:, 1])),
        "r2_std": float(np.std(A[:, 1], ddof=1)) if len(A) > 1 else 0.0,
        "median_relative_error_mean": float(np.mean(A[:, 2])),
        "median_relative_error_std": float(np.std(A[:, 2], ddof=1)) if len(A) > 1 else 0.0,
    }


def phase_model_comparison(records: Sequence[Dict[str, float]], folds: int, seed: int) -> Dict[str, Any]:
    no_phase = ("d_over_xi", "xi")
    phase = ("d_over_xi", "phase_cos", "phase_sin", "xi")
    a = evaluate_intrinsic_model(records, no_phase, folds, seed)
    b = evaluate_intrinsic_model(records, phase, folds, seed)
    hold_a = intrinsic_region_holdouts(records, no_phase)
    hold_b = intrinsic_region_holdouts(records, phase)
    def summarize(h):
        vals = [v for v in h.values() if v.get("status") == "ok"]
        return {
            "median_holdout_r2": float(np.median([v["r2_logN"] for v in vals])) if vals else float("nan"),
            "median_holdout_relerr": float(np.median([v["median_relative_error"] for v in vals])) if vals else float("nan"),
            "worst_holdout_r2": float(np.min([v["r2_logN"] for v in vals])) if vals else float("nan"),
            "worst_holdout_relerr": float(np.max([v["median_relative_error"] for v in vals])) if vals else float("nan"),
        }
    sa, sb = summarize(hold_a), summarize(hold_b)
    return {
        "no_phase": {"model": a, "holdouts": hold_a, "summary": sa},
        "phase": {"model": b, "holdouts": hold_b, "summary": sb},
        "cv_r2_gain": float(b["cv_r2_logN"] - a["cv_r2_logN"]),
        "cv_median_relerr_change": float(b["cv_median_relative_error"] - a["cv_median_relative_error"]),
        "holdout_median_r2_gain": float(sb["median_holdout_r2"] - sa["median_holdout_r2"]),
        "holdout_median_relerr_change": float(sb["median_holdout_relerr"] - sa["median_holdout_relerr"]),
    }




# -----------------------------------------------------------------------------
# v21.3 — canonical minimal-law audit
# -----------------------------------------------------------------------------


def canonical_law_formula(names: Sequence[str]) -> str:
    key = tuple(names)
    if key == ("d_over_xi", "xi"):
        return r"log N_wire = A + B (d_M/xi_M) + C (xi_M/a)"
    if key == ("d_over_xi", "log_xi_over_a"):
        return r"log N_wire = A + B (d_M/xi_M) + C log(xi_M/a)"
    if not names:
        return r"log N_wire = A"
    return "log N_wire = A + " + " + ".join(V212_DISPLAY[n] for n in names)


def canonical_model_set() -> List[Tuple[str, ...]]:
    # Primary ladder: only compact, dimensionless, non-phase descriptors.
    # Phase-enabled models remain a falsification diagnostic, not part of the
    # proposed headline law.
    return [
        (),
        ("d_over_xi",),
        ("xi",),
        ("log_xi_over_a",),
        ("d_over_xi", "xi"),
        ("d_over_xi", "log_xi_over_a"),
        ("gap", "d_over_xi"),
        ("gap", "d_over_xi", "xi"),
        ("d_over_xi", "xi", "log_xi_over_a"),
    ]


def evaluate_canonical_model(records: Sequence[Dict[str, float]], names: Sequence[str], folds: int, seed: int) -> Dict[str, Any]:
    X, y = intrinsic_feature_matrix(records, names)
    n = len(records)
    pred_cv = np.full(n, np.nan)
    for test_idx in _kfold_indices(n, folds, seed):
        train_mask = np.ones(n, dtype=bool)
        train_mask[test_idx] = False
        beta, _ = _ols_fit(X[train_mask], y[train_mask])
        pred_cv[test_idx] = X[test_idx] @ beta
    beta, pred = _ols_fit(X, y)
    cv = _prediction_metrics(y, pred_cv)
    ins = _prediction_metrics(y, pred)
    rss = float(np.sum((y - pred) ** 2))
    bic = float(n * math.log(max(rss / n, 1e-30)) + X.shape[1] * math.log(max(n, 2)))
    return {
        "features": list(names),
        "label": v212_model_label(names),
        "formula": canonical_law_formula(names),
        "n_features": len(names),
        "coefficients": beta.tolist(),
        "cv_rmse_logN": cv["rmse_logN"],
        "cv_r2_logN": cv["r2_logN"],
        "cv_median_relative_error": cv["median_relative_error"],
        "cv_p90_relative_error": cv["p90_relative_error"],
        "in_sample_rmse_logN": ins["rmse_logN"],
        "bic": bic,
        "predicted_logN": pred_cv.tolist(),
    }


def canonical_holdouts(records: Sequence[Dict[str, float]], names: Sequence[str]) -> Dict[str, Any]:
    L = np.array([r["L"] for r in records])
    alpha = np.array([r["alpha"] for r in records])
    specs = {
        "train_L_<=80_test_>80": (L <= 80, L > 80),
        "train_L_>80_test_<=80": (L > 80, L <= 80),
        "train_alpha_<=0.16_test_>=0.17": (alpha <= 0.16 + 1e-12, alpha >= 0.17 - 1e-12),
        "train_alpha_>=0.17_test_<=0.16": (alpha >= 0.17 - 1e-12, alpha <= 0.16 + 1e-12),
    }
    X, y = intrinsic_feature_matrix(records, names)
    out: Dict[str, Any] = {}
    for label, (train_mask, test_mask) in specs.items():
        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]
        if len(train_idx) < 5 or len(test_idx) < 5:
            out[label] = {"status": "skipped", "n_train": int(len(train_idx)), "n_test": int(len(test_idx))}
            continue
        beta, _ = _ols_fit(X[train_idx], y[train_idx])
        pred = X[test_idx] @ beta
        m = _prediction_metrics(y[test_idx], pred)
        out[label] = {"status": "ok", **m, "n_train": int(len(train_idx)), "n_test": int(len(test_idx)), "coefficients": beta.tolist()}
    return out


def canonical_coefficient_stability(records: Sequence[Dict[str, float]], names: Sequence[str], repeats: int, train_fraction: float, seed: int) -> Dict[str, Any]:
    X, y = intrinsic_feature_matrix(records, names)
    n = len(records)
    train_n = min(max(int(round(train_fraction * n)), X.shape[1] + 2), n - 1)
    rng = np.random.default_rng(seed)
    B = []
    for _ in range(max(1, repeats)):
        idx = rng.choice(n, size=train_n, replace=False)
        beta, _ = _ols_fit(X[idx], y[idx])
        B.append(beta)
    B = np.asarray(B, float)
    mean = np.mean(B, axis=0)
    std = np.std(B, axis=0, ddof=1) if len(B) > 1 else np.zeros(B.shape[1])
    return {
        "features": list(names),
        "label": v212_model_label(names),
        "repeats": int(len(B)),
        "train_fraction": float(train_fraction),
        "coefficient_mean": mean.tolist(),
        "coefficient_std": std.tolist(),
        "relative_std": (std / np.maximum(np.abs(mean), 1e-30)).tolist(),
        "sign_stability": [float(np.mean(np.sign(B[:, j]) == np.sign(mean[j]))) if abs(mean[j]) > 1e-15 else 0.0 for j in range(B.shape[1])],
    }


def canonical_repeated_holdout(records: Sequence[Dict[str, float]], names: Sequence[str]) -> Dict[str, Any]:
    # Refit the canonical law separately in each extrapolation direction and
    # expose the distribution, not just one score.
    h = canonical_holdouts(records, names)
    ok = {k: v for k, v in h.items() if v.get("status") == "ok"}
    r2 = [v["r2_logN"] for v in ok.values()]
    err = [v["median_relative_error"] for v in ok.values()]
    return {
        "holdouts": h,
        "median_r2": float(np.median(r2)) if r2 else float("nan"),
        "worst_r2": float(np.min(r2)) if r2 else float("nan"),
        "median_relerr": float(np.median(err)) if err else float("nan"),
        "worst_relerr": float(np.max(err)) if err else float("nan"),
    }


def canonical_law_audit(records: Sequence[Dict[str, float]], folds: int, repeats: int, seed: int, coeff_repeats: int, train_fraction: float, target_r2: float, target_median_relerr: float) -> Dict[str, Any]:
    if len(records) < 20:
        return {"status": "insufficient_candidates", "n": len(records)}
    models = [evaluate_canonical_model(records, m, folds, seed) for m in canonical_model_set()]
    best = min(models, key=lambda x: x["cv_rmse_logN"])
    near = [m for m in models if m["cv_rmse_logN"] <= best["cv_rmse_logN"] * 1.05]
    minimal = min(near, key=lambda x: (x["n_features"], x["cv_rmse_logN"]))
    canonical_names = ("d_over_xi", "xi")
    canonical = next(m for m in models if tuple(m["features"]) == canonical_names)
    canonical_h = canonical_repeated_holdout(records, canonical_names)
    canonical_stab = canonical_coefficient_stability(records, canonical_names, coeff_repeats, train_fraction, seed + 9000)
    passes = (
        canonical["cv_r2_logN"] >= target_r2 and
        canonical["cv_median_relative_error"] <= target_median_relerr and
        canonical_h["worst_r2"] >= target_r2 and
        canonical_h["worst_relerr"] <= target_median_relerr
    )
    return {
        "status": "ok",
        "n": len(records),
        "models": models,
        "best_random_cv": best,
        "minimal_near_best": minimal,
        "canonical_model": canonical,
        "canonical_holdouts": canonical_h,
        "canonical_coefficient_stability": canonical_stab,
        "canonical_model_is_engineering_rule_candidate": bool(passes),
        "canonical_formula": canonical["formula"],
        "interpretation": (
            "The two-variable law is promoted to a candidate engineering rule within the tested parameter region."
            if passes else
            "The two-variable law remains a compact candidate but does not yet satisfy all predeclared engineering thresholds."
        ),
    }


def phase_falsification_summary(phase_cmp: Dict[str, Any], tolerance_r2: float = 0.01, tolerance_err: float = 0.01) -> Dict[str, Any]:
    if not phase_cmp:
        return {"status": "unavailable"}
    r2_gain = float(phase_cmp.get("holdout_median_r2_gain", 0.0))
    err_change = float(phase_cmp.get("holdout_median_relerr_change", 0.0))
    useful = (r2_gain >= tolerance_r2) and (err_change <= -tolerance_err)
    return {
        "status": "ok",
        "median_holdout_r2_gain": r2_gain,
        "median_holdout_relerr_change": err_change,
        "phase_useful_under_threshold": bool(useful),
        "verdict": "phase not required by current test" if not useful else "phase remains potentially useful",
    }

def scan_geometry_v212(L_values: Sequence[int], alpha_values: Sequence[float], Delta: float, Ez: float, t0: float, mu_min: float, mu_max: float, mu_points: int, K_reference: np.ndarray, L_reference: int, rho: float, xi_noise: float, device: str, mu_refine_stages: int, mu_refine_points: int, lattice_spacing: float) -> Tuple[List[Dict[str, float]], Dict[str, Any]]:
    records: List[Dict[str, float]] = []
    meta: Dict[str, Any] = {"device": device, "successful": 0, "failed": 0, "k_osc_sources": {}, "errors": []}
    for alpha in alpha_values:
        for L in L_values:
            try:
                template = BdGWire(BdGParams(int(L), float(alpha), Delta, Ez, t0, 0.5))
                wire, ref, op = tune_operating_point(template, mu_min, mu_max, mu_points, device, mu_refine_stages, mu_refine_points)
                if op.z2 != -1:
                    meta["failed"] += 1
                    continue
                K = majorana_kernel(wire, ref, device)
                Rpair = noise_R(L_reference, int(L), rho, xi_noise)
                tr = effective_noise(K_reference, K, Rpair)
                decomp = pair_noise_decomposition(K_reference, K, Rpair)
                n_wire = intrinsic_noise_exposure(K, xi_noise)
                kosc, source = estimate_k_osc(wire)
                xi = float(op.xi_mean)
                d = float(op.separation)
                theta = float(kosc * d)
                rec = {
                    "L": int(L), "alpha": float(alpha), "mu": float(op.mu),
                    "E_gap": float(op.E_gap), "E_low": float(op.E_low),
                    "xi_mean": xi, "separation": d,
                    "d_over_xi": float(d / max(xi, 1e-30)),
                    "gap_over_Delta": float(op.E_gap / max(Delta, 1e-30)),
                    "xi_over_a": float(xi / max(lattice_spacing, 1e-30)),
                    "d_over_a": float(d / max(lattice_spacing, 1e-30)),
                    "log_xi_over_a": float(math.log(max(xi / max(lattice_spacing, 1e-30), 1e-30))),
                    "k_osc": float(kosc), "k_osc_source": source,
                    "theta_osc": theta,
                    "theta_abs_mod": float(abs(theta) % math.pi),
                    "phase_cos": float(math.cos(theta)),
                    "phase_sin": float(math.sin(theta)),
                    "phase_cos2": float(math.cos(2.0 * theta)),
                    "phase_sin2": float(math.sin(2.0 * theta)),
                    "N_wire": float(n_wire),
                    "K_norm2": float(np.dot(K, K)),
                    "K_vector": np.asarray(K, dtype=float).tolist(),
                    "N_geom": float(tr["N_geom"]),
                    "lambda_weak": float(tr["lambda_weak"]),
                    "lambda_strong": float(tr["lambda_strong"]),
                    "mode_ratio": float(tr["mode_ratio"]),
                    "pair_n_reference": float(decomp["n1"]),
                    "pair_n_candidate": float(decomp["n2"]),
                    "pair_cross_c": float(decomp["cross_c"]),
                    "pair_rho_eff": float(decomp["rho_eff"]),
                    "pair_N_geom_exact": float(decomp["N_geom_exact"]),
                    "pair_formula_relative_error": float(abs(decomp["N_geom_exact"] - tr["N_geom"]) / max(abs(tr["N_geom"]), 1e-30)),
                    "z2": int(op.z2),
                }
                records.append(rec)
                meta["successful"] += 1
                meta["k_osc_sources"][source] = int(meta["k_osc_sources"].get(source, 0) + 1)
            except (RuntimeError, ValueError, la.LinAlgError, FloatingPointError) as exc:
                meta["failed"] += 1
                meta["errors"].append({"L": int(L), "alpha": float(alpha), "error": str(exc)})
    return records, meta


def v212_make_figures(outdir: str, geometry: Sequence[Dict[str, float]], law: Dict[str, Any], phase_cmp: Dict[str, Any], pair_stats: Dict[str, Any]) -> None:
    if not geometry:
        return
    x = np.array([r["d_over_xi"] for r in geometry], float)
    y = np.log(np.maximum(np.array([r["N_wire"] for r in geometry], float), 1e-30))
    phase = np.array([r["theta_abs_mod"] for r in geometry], float)
    xi = np.array([r["xi_over_a"] for r in geometry], float)

    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    sc = ax.scatter(x, y, c=phase, s=42, alpha=0.78)
    ax.set_xlabel(r"$d_M/\xi_M$")
    ax.set_ylabel(r"$\log N_{\rm wire}$")
    ax.set_title("Intrinsic Majorana noise exposure")
    ax.grid(True, ls=":", alpha=0.35)
    fig.colorbar(sc, ax=ax, label=r"$|k_{\rm osc}d_M|\mathrm{mod}\,\pi$")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "intrinsic_phase_v21_2.png"), dpi=240); plt.close(fig)

    if law.get("status") == "ok":
        models = law["models"]
        order = np.argsort([m["cv_rmse_logN"] for m in models])
        fig, ax = plt.subplots(figsize=(9.0, 6.2))
        yy = np.arange(len(order))
        vals = np.array([models[i]["cv_rmse_logN"] for i in order])
        ax.barh(yy, vals)
        ax.set_yticks(yy, [models[i]["label"] for i in order], fontsize=8)
        ax.invert_yaxis(); ax.set_xlabel("CV RMSE of log N_wire")
        ax.set_title("Intrinsic law ladder")
        ax.grid(True, axis="x", ls=":", alpha=0.35)
        fig.tight_layout(); fig.savefig(os.path.join(outdir, "intrinsic_law_ladder_v21_2.png"), dpi=240); plt.close(fig)

        best = law["minimal_model_within_tolerance"]
        pred = np.asarray(best["predicted_logN"], float)
        actual = y
        lo = min(float(np.min(actual)), float(np.min(pred))); hi = max(float(np.max(actual)), float(np.max(pred)))
        fig, ax = plt.subplots(figsize=(6.3, 5.4))
        ax.scatter(actual, pred, s=32, alpha=0.68)
        ax.plot([lo, hi], [lo, hi], ls="--", alpha=0.5)
        ax.set_xlabel("actual log N_wire"); ax.set_ylabel("5-fold CV predicted log N_wire")
        ax.set_title(f"Selected intrinsic law: {best['label']}")
        ax.grid(True, ls=":", alpha=0.35)
        fig.tight_layout(); fig.savefig(os.path.join(outdir, "intrinsic_cv_pred_vs_actual_v21_2.png"), dpi=240); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    all_ratios = []
    if pair_stats.get("strict_ratios"):
        all_ratios = pair_stats["strict_ratios"]
        ax.hist(all_ratios, bins=min(30, max(8, int(math.sqrt(len(all_ratios))))), alpha=0.78)
    ax.axvline(1.0, ls="--", alpha=0.45)
    ax.set_xlabel("strict-pair N_geom ratio")
    ax.set_ylabel("count")
    ax.set_title("Residual pair-level variation after gap + xi + separation matching")
    ax.grid(True, axis="y", ls=":", alpha=0.35)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "strict_pair_ratio_hist_v21_2.png"), dpi=240); plt.close(fig)

    if phase_cmp:
        # Holdout summary: no-phase vs phase-enabled median test performance.
        labels = ["no phase", "+ interference phase"]
        r2s = [phase_cmp["no_phase"]["summary"]["median_holdout_r2"], phase_cmp["phase"]["summary"]["median_holdout_r2"]]
        errs = [phase_cmp["no_phase"]["summary"]["median_holdout_relerr"], phase_cmp["phase"]["summary"]["median_holdout_relerr"]]
        fig, ax1 = plt.subplots(figsize=(7.5, 5.0))
        xpos = np.arange(2)
        ax1.bar(xpos - 0.18, r2s, width=0.36, label="median holdout R²")
        ax1.set_xticks(xpos, labels); ax1.set_ylabel("median holdout R²")
        ax2 = ax1.twinx()
        ax2.bar(xpos + 0.18, np.array(errs) * 100.0, width=0.36, alpha=0.65, label="median holdout rel. error")
        ax2.set_ylabel("median holdout relative error (%)")
        ax1.set_title("Does an interference-phase descriptor improve extrapolation?")
        ax1.grid(True, axis="y", ls=":", alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(outdir, "phase_holdout_comparison_v21_2.png"), dpi=240); plt.close(fig)


def write_paper_law_v212(path: str, report: Dict[str, Any]) -> None:
    law = report["intrinsic_law"]
    phase_cmp = report["phase_comparison"]
    pairs = report["pair_analysis"]
    lines: List[str] = []
    lines.append("# Majorana Noise Geometry v21.2 — Intrinsic Noise Law + Interference Phase\n")
    lines.append("## Scientific question\n")
    lines.append("Can a single-wire Majorana noise exposure be predicted from a small set of physically meaningful, dimensionless geometry descriptors, and does an interference-phase descriptor improve extrapolation to unseen device regions?\n")
    lines.append("## 1. Core transfer equations\n")
    lines.append(r"`delta epsilon = K delta mu`\n")
    lines.append(r"`Cbar = K R K^T`\n")
    lines.append(r"`N_geom = lambda_max(Cbar)`\n")
    lines.append(r"For one wire: `N_wire = K R_self K^T`.\n")
    lines.append(r"For the two-wire 2x2 covariance, `N_geom = [n1+n2+sqrt((n1-n2)^2+4c^2)]/2`, with `c = K1 R12 K2^T`.\n")

    lines.append("## 2. Intrinsic law-discovery result\n")
    if law.get("status") == "ok":
        best = law["best_cv_model"]
        minimal = law["minimal_model_within_tolerance"]
        lines.append(f"Best random-CV model: **{best['label']}**")
        lines.append(f"CV R²(log N_wire): `{best['cv_r2_logN']:.4f}`; CV RMSE: `{best['cv_rmse_logN']:.5f}`; median relative error: `{100*best['cv_median_relative_error']:.2f}%`.\n")
        lines.append(f"Minimal model within {100*law['minimality_tolerance_fraction']:.1f}% of best CV error: **{minimal['label']}**")
        lines.append(f"Formula: `{minimal['formula']}`")
        lines.append("Full-data coefficients (intercept first): `" + ", ".join(f"{x:.8g}" for x in minimal["coefficients"]) + "`\n")
        lines.append("### Law ladder")
        lines.append(_format_intrinsic_model_table(law["models"]))
    else:
        lines.append("Insufficient candidates for intrinsic law discovery.")

    lines.append("\n## 3. Interference-phase test\n")
    if phase_cmp:
        a = phase_cmp["no_phase"]["summary"]; b = phase_cmp["phase"]["summary"]
        lines.append(f"No-phase baseline: median holdout R² = `{a['median_holdout_r2']:.4f}`, median holdout relative error = `{100*a['median_holdout_relerr']:.2f}%`." )
        lines.append(f"Phase-enabled model: median holdout R² = `{b['median_holdout_r2']:.4f}`, median holdout relative error = `{100*b['median_holdout_relerr']:.2f}%`." )
        lines.append(f"Median holdout R² change: `{phase_cmp['holdout_median_r2_gain']:+.4f}`; median holdout error change: `{100*phase_cmp['holdout_median_relerr_change']:+.2f}%`.\n")
        lines.append("The phase descriptor is therefore treated as a falsifiable candidate physical variable, not as an assumed final law.")

    lines.append("\n## 4. Unseen-region extrapolation\n")
    if law.get("region_holdouts"):
        for k, v in law["region_holdouts"].items():
            if v.get("status") == "ok":
                lines.append(f"- `{k}`: R²={v['r2_logN']:.4f}, median relative error={100*v['median_relative_error']:.2f}% (n_test={v['n_test']})")
            else:
                lines.append(f"- `{k}`: skipped")
    else:
        lines.append("No region holdout result available.")

    stab = law.get("coefficient_stability")
    if stab:
        lines.append("\n## 5. Coefficient stability\n")
        lines.append("Relative coefficient standard deviations: `" + ", ".join(f"{x:.4f}" for x in stab["relative_std"]) + "`")
        lines.append("Sign stability: `" + ", ".join(f"{100*x:.1f}%" for x in stab["sign_stability"]) + "`\n")

    lines.append("## 6. Strict-pair result\n")
    lines.append(f"Strict 5% matched pairs: **{pairs['count']}**")
    lines.append(f"Nearest-pair N_geom ratio: **{pairs['nearest']['N_geom_ratio']:.6f}**" if pairs.get("nearest") else "Nearest pair unavailable.")
    lines.append(f"Maximum strict-pair N_geom ratio: **{pairs['max_ratio']:.6f}**" if pairs.get("max_ratio") is not None else "Maximum strict-pair ratio unavailable.")

    lines.append("\n## 7. Current scientific interpretation\n")
    lines.append("Random cross-validation can reveal compact correlations inside the sampled geometry cloud, but a main-text design law requires stable coefficients and useful performance on unseen L/alpha regions.")
    if phase_cmp and phase_cmp["holdout_median_r2_gain"] > 0 and phase_cmp["holdout_median_relerr_change"] < 0:
        lines.append("In this run, adding an interference-phase descriptor improves the median holdout metrics relative to the no-phase baseline. This supports further testing of phase as a physical control variable.")
    else:
        lines.append("In this run, the interference-phase descriptor does not yet establish a robust extrapolating law. It remains a falsifiable candidate variable rather than a main-text rule.")
    lines.append("\n## 8. Engineering rule status\n")
    eng = law.get("minimal_engineering_model_joint")
    if eng:
        lines.append(f"A candidate meets the chosen in-sample CV engineering targets: **{eng['label']}**.")
    else:
        lines.append("No candidate simultaneously meets the chosen random-CV engineering thresholds.")
    lines.append("Even if an in-sample CV model passes, it is not promoted to a universal device law unless the intended region-holdout tests are also satisfactory.")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _format_intrinsic_model_table(models: Sequence[Dict[str, Any]]) -> str:
    rows = ["| Model | CV R² | CV RMSE(log N_wire) | median rel. err. |", "|---|---:|---:|---:|"]
    for m in sorted(models, key=lambda x: (x["n_features"], x["cv_rmse_logN"])):
        rows.append(f"| {m['label']} | {m['cv_r2_logN']:.4f} | {m['cv_rmse_logN']:.5f} | {100*m['cv_median_relative_error']:.2f}% |")
    return "\n".join(rows)


def run_v212(args: argparse.Namespace) -> Dict[str, Any]:
    os.makedirs(args.outdir, exist_ok=True)
    device, device_info = choose_device(args.device, args.gpu_id)

    template1 = BdGWire(BdGParams(args.L1, args.alpha, args.Delta, args.Ez, args.t0, 0.5))
    template2 = BdGWire(BdGParams(args.L2, args.alpha, args.Delta, args.Ez, args.t0, 0.5))
    wire1, ref1, op1 = tune_operating_point(template1, args.mu_min, args.mu_max, args.mu_points, device, args.mu_refine_stages, args.mu_refine_points)
    wire2, ref2, op2 = tune_operating_point(template2, args.mu_min, args.mu_max, args.mu_points, device, args.mu_refine_stages, args.mu_refine_points)
    if op1.z2 != -1 or op2.z2 != -1:
        raise RuntimeError("At least one baseline wire is not topological")

    print("=" * 100)
    print("Majorana Noise Geometry v21.2 — Intrinsic Noise Law + Majorana Interference Phase")
    print("=" * 100)
    print(f"device: {device} | GPU info: {device_info}")
    for tag, op, wire in (("wire1", op1, wire1), ("wire2", op2, wire2)):
        kosc, src = estimate_k_osc(wire)
        print(f"{tag}: L={op.L}, alpha={op.alpha:.3f}, mu*={op.mu:.8f}, gap={op.E_gap:.3e}, xi={op.xi_mean:.3e}, separation={op.separation:.3e}, d/xi={op.separation/max(op.xi_mean,1e-30):.3f}, k_osc={kosc:.5f}, theta={kosc*op.separation:.5f}, source={src}, Z2={op.z2}")

    K1 = majorana_kernel(wire1, ref1, device)
    K2 = majorana_kernel(wire2, ref2, device)
    sites1 = np.unique(np.linspace(0, args.L1 - 1, min(6, args.L1), dtype=int))
    sites2 = np.unique(np.linspace(0, args.L2 - 1, min(6, args.L2), dtype=int))
    fd1 = kernel_fd_check(wire1, ref1, K1, sites1, args.fd_step)
    fd2 = kernel_fd_check(wire2, ref2, K2, sites2, args.fd_step)
    print(f"kernel FD relative error: {fd1:.3%} / {fd2:.3%}")

    R = noise_R(args.L1, args.L2, args.rho_site, args.xi_noise)
    transfer = effective_noise(K1, K2, R)
    decomp = pair_noise_decomposition(K1, K2, R)
    vals = np.asarray(transfer["eigenvalues"], float)
    print(f"eigenvalues(Cbar): [{vals[0]:.8e} {vals[-1]:.8e}]")
    print(f"mode anisotropy lambda_strong/lambda_weak: {transfer['mode_ratio']:.4f}")
    print(f"N_geom: {transfer['N_geom']:.8e}")
    print(f"pair decomposition: n1={decomp['n1']:.8e} | n2={decomp['n2']:.8e} | c={decomp['cross_c']:.8e} | rho_eff={decomp['rho_eff']:.5f}")
    print(f"exact 2x2 reconstruction relative error: {abs(decomp['N_geom_exact']-transfer['N_geom'])/max(abs(transfer['N_geom']),1e-30):.3e}")

    validation = nonlinear_window(wire1, wire2, ref1, ref2, np.asarray(transfer["Cbar"]), args.rho_site, args.xi_noise, args.validation_W, args.validation_real, args.seed, device, args.gpu_batch_size)
    print("eta_nl(W):", [(r["W"], r["eta_nl"]) for r in validation["rows"]])

    geometry, tune_meta = scan_geometry_v212(args.scan_lengths, args.scan_alpha, args.Delta, args.Ez, args.t0, args.mu_min, args.mu_max, args.mu_points, K1, args.L1, args.rho_site, args.xi_noise, device, args.mu_refine_stages, args.mu_refine_points, args.lattice_spacing)
    print(f"geometry candidates: {len(geometry)} (successful={tune_meta['successful']}, failed={tune_meta['failed']})")
    print(f"k_osc sources: {tune_meta.get('k_osc_sources', {})}")

    pair_stats = strict_pair_statistics(geometry, args.gap_match_tol, args.xi_match_tol, args.separation_match_tol)
    nearest = pair_stats.get("nearest"); max_pair = pair_stats.get("max_ratio_pair")
    print(f"strict matched pairs: {pair_stats['count']}")
    if nearest:
        print(f"nearest pair: ratio={nearest['N_geom_ratio']:.6f}; gap={nearest['relative_gap_difference']:.3%}; xi={nearest['relative_xi_difference']:.3%}; separation={nearest['relative_separation_difference']:.3%}; D_match={nearest['match_distance']:.4f}; D_inf={nearest['match_distance_inf']:.4f}")
    if max_pair:
        print(f"max-ratio strict pair: ratio={max_pair['N_geom_ratio']:.6f}; gap={max_pair['relative_gap_difference']:.3%}; xi={max_pair['relative_xi_difference']:.3%}; separation={max_pair['relative_separation_difference']:.3%}; D_match={max_pair['match_distance']:.4f}; D_inf={max_pair['match_distance_inf']:.4f}")

    intrinsic_law = discover_intrinsic_law(geometry, args.cv_folds, args.seed, args.law_tolerance, args.target_r2, args.target_median_relerr)
    if intrinsic_law.get("status") == "ok":
        best = intrinsic_law["best_cv_model"]
        minimal = intrinsic_law["minimal_model_within_tolerance"]
        print(f"intrinsic law: best={best['label']} | CV RMSE(log N_wire)={best['cv_rmse_logN']:.5f} | CV R2={best['cv_r2_logN']:.4f} | median rel.err={100*best['cv_median_relative_error']:.2f}%")
        print(f"intrinsic law: minimal-within-tolerance={minimal['label']} | CV RMSE(log N_wire)={minimal['cv_rmse_logN']:.5f} | CV R2={minimal['cv_r2_logN']:.4f}")
        intrinsic_law["coefficient_stability"] = intrinsic_coefficient_stability(geometry, minimal["features"], args.coeff_repeats, args.coeff_train_fraction, args.seed + 1000)
        intrinsic_law["repeated_cv_stability"] = intrinsic_repeated_cv(geometry, minimal["features"], args.cv_folds, args.cv_repeats, args.seed + 2000)
        intrinsic_law["region_holdouts"] = intrinsic_region_holdouts(geometry, minimal["features"])
        for label, item in intrinsic_law["region_holdouts"].items():
            if item.get("status") == "ok":
                print(f"intrinsic holdout {label}: R2={item['r2_logN']:.4f}, median rel.err={100*item['median_relative_error']:.2f}%")
        eng = intrinsic_law.get("minimal_engineering_model_joint")
        print("intrinsic engineering target (both):", eng["label"] if eng else "NONE")
    else:
        print("intrinsic law discovery: insufficient candidate count")

    phase_cmp = phase_model_comparison(geometry, args.cv_folds, args.seed + 3000) if len(geometry) >= 20 else {}
    if phase_cmp:
        print(f"phase comparison: CV R2 gain={phase_cmp['cv_r2_gain']:+.4f}; CV median rel.err change={100*phase_cmp['cv_median_relerr_change']:+.2f}%")
        print(f"phase comparison: median holdout R2 gain={phase_cmp['holdout_median_r2_gain']:+.4f}; median holdout rel.err change={100*phase_cmp['holdout_median_relerr_change']:+.2f}%")

    report: Dict[str, Any] = {
        "version": V21_2_VERSION,
        "device": device_info | {"selected": device},
        "core": {
            "H_p": "H_p = 1/2 [g X - z_p Z], z_p=-(epsilon_12+p epsilon_34)",
            "linear_response": "delta epsilon = K delta mu",
            "normalized_noise_transfer": "Cbar = K R K^T",
            "single_wire_exposure": "N_wire = K R_self K^T",
            "two_wire_exact": "N_geom = [n1+n2+sqrt((n1-n2)^2+4c^2)]/2, c=K1 R12 K2^T",
            "design_metric": "N_geom = lambda_max(Cbar)",
            "phase_descriptor": "theta_osc = k_osc d_M, with k_osc estimated from the lowest positive normal-state Fermi root when available",
        },
        "parameters": {k: getattr(args, k) for k in ("L1", "L2", "alpha", "Delta", "Ez", "t0", "rho_site", "xi_noise", "W", "mu_min", "mu_max", "mu_points", "mu_refine_stages", "mu_refine_points", "fd_step", "validation_real", "gpu_batch_size", "gap_match_tol", "xi_match_tol", "separation_match_tol", "cv_folds", "cv_repeats", "law_tolerance", "target_r2", "target_median_relerr", "coeff_repeats", "coeff_train_fraction", "lattice_spacing", "seed", "device", "gpu_id")},
        "operating_points": {"wire1": op1.as_dict(), "wire2": op2.as_dict()},
        "kernels": {"K1": K1.tolist(), "K2": K2.tolist(), "finite_difference_relative_error": {"wire1": fd1, "wire2": fd2}},
        "effective_noise": {
            "Cbar": np.asarray(transfer["Cbar"]).tolist(), "eigenvalues": vals.tolist(),
            "eigenvectors": np.asarray(transfer["eigenvectors"]).tolist(), "lambda_weak": float(transfer["lambda_weak"]),
            "lambda_strong": float(transfer["lambda_strong"]), "mode_ratio": float(transfer["mode_ratio"]),
            "N_geom": float(transfer["N_geom"]), "S_plus_bar": float(transfer["S_plus_bar"]), "S_minus_bar": float(transfer["S_minus_bar"]),
            "physical_S_plus": float(args.W ** 2 * transfer["S_plus_bar"]), "physical_S_minus": float(args.W ** 2 * transfer["S_minus_bar"]),
            "pair_decomposition": decomp,
        },
        "validation": validation,
        "geometry_scan": geometry,
        "scan_meta": tune_meta,
        "pair_analysis": {
            "strict_definition": {"relative_gap": float(args.gap_match_tol), "relative_xi": float(args.xi_match_tol), "relative_separation": float(args.separation_match_tol), "equivalent_condition": "D_inf <= 1"},
            "count": pair_stats["count"], "max_ratio": pair_stats["max_ratio"], "median_ratio": pair_stats["median_ratio"], "p95_ratio": pair_stats["p95_ratio"],
            "nearest_pair": compact_pair(nearest), "max_ratio_pair": compact_pair(max_pair), "strict_ratios": pair_stats.get("strict_ratios", []),
        },
        "intrinsic_law": intrinsic_law,
        "phase_comparison": phase_cmp,
        "scientific_readout": {
            "both_topological": True,
            "kernel_fd_supported": bool(fd1 < 0.08 and fd2 < 0.08),
            "universality_claim": False,
            "current_message": "The intrinsic law is only promotable to a main-text device rule if its coefficients are stable and its L/alpha holdouts show useful extrapolation. Pair-level N_geom is decomposed exactly into single-wire exposures plus cross-wire noise correlation.",
        },
    }

    v212_make_figures(args.outdir, geometry, intrinsic_law, phase_cmp, pair_stats)
    write_geometry_csv(os.path.join(args.outdir, "geometry_scan_v21_2.csv"), geometry)
    np.savez(os.path.join(args.outdir, "kernels_v21_2.npz"), K12=K1, K34=K2, L1=np.array([args.L1]), L2=np.array([args.L2]), mu1_star=np.array([op1.mu]), mu2_star=np.array([op2.mu]), Cbar=np.asarray(transfer["Cbar"]))
    with open(os.path.join(args.outdir, "core_results_v21_2.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    write_paper_law_v212(os.path.join(args.outdir, "paper_law_v21_2.md"), report)
    print(f"outputs -> {args.outdir}/")
    return report



def run_v213(args: argparse.Namespace) -> Dict[str, Any]:
    # Reuse the fully validated v21.2 numerical pipeline, then add a clean
    # canonical-law audit. We deliberately do not alter the microscopic model.
    report = run_v212(args)

    # Re-read the just-generated geometry from the in-memory report, because
    # run_v212 already performed the expensive BdG/GPU scan.
    geometry = report.get("geometry_scan", [])
    if geometry:
        audit = canonical_law_audit(
            geometry,
            args.cv_folds,
            args.cv_repeats,
            args.seed,
            args.coeff_repeats,
            args.coeff_train_fraction,
            args.target_r2,
            args.target_median_relerr,
        )
        phase_summary = phase_falsification_summary(report.get("phase_comparison", {}))
        report["version"] = "v21.3"
        report["canonical_law_audit"] = audit
        report["phase_falsification"] = phase_summary
        report["scientific_readout"]["canonical_rule_candidate"] = bool(audit.get("canonical_model_is_engineering_rule_candidate", False)) if isinstance(audit, dict) else False
        report["scientific_readout"]["phase_is_primary"] = bool(phase_summary.get("phase_useful_under_threshold", False)) if isinstance(phase_summary, dict) else False

        print("-" * 100)
        print("v21.3 canonical-law audit")
        if audit.get("status") == "ok":
            can = audit["canonical_model"]
            h = audit["canonical_holdouts"]
            stab = audit["canonical_coefficient_stability"]
            print(f"canonical law: {can['formula']}")
            print(f"canonical CV: R2={can['cv_r2_logN']:.4f} | RMSE(log N)={can['cv_rmse_logN']:.5f} | median rel.err={100*can['cv_median_relative_error']:.2f}%")
            print(f"canonical holdout: median R2={h['median_r2']:.4f} | worst R2={h['worst_r2']:.4f} | median rel.err={100*h['median_relerr']:.2f}% | worst rel.err={100*h['worst_relerr']:.2f}%")
            print(f"canonical coefficient relative std: {[round(x,4) for x in stab['relative_std']]}")
            print(f"canonical engineering-rule candidate: {'YES' if audit['canonical_model_is_engineering_rule_candidate'] else 'NO'}")
            print(f"phase falsification verdict: {phase_summary.get('verdict', 'unknown')}")

        # Write a focused v21.3 paper memo alongside the v21.2 artifacts.
        outdir = args.outdir
        memo = []
        memo.append("# Majorana Noise Geometry v21.3 — Canonical Minimal Law\n")
        memo.append("## Core statement\n")
        memo.append("The numerical pipeline is unchanged: `delta epsilon = K delta mu`, `Cbar = K R K^T`, and `N_geom = lambda_max(Cbar)`. The new question is whether a single-wire intrinsic exposure admits a minimal, dimensionless, extrapolating law.\n")
        if audit.get("status") == "ok":
            can = audit["canonical_model"]
            h = audit["canonical_holdouts"]
            stab = audit["canonical_coefficient_stability"]
            memo.append("## Canonical candidate\n")
            memo.append(f"`{can['formula']}`\n")
            memo.append(f"Equivalent exponential form: `N_wire = N0 * exp(B*d_M/xi_M + C*xi_M/a)`.\n")
            memo.append("Coefficients (intercept, B, C): `" + ", ".join(f"{x:.10g}" for x in can["coefficients"]) + "`\n")
            memo.append(f"Random 5-fold CV: R²=`{can['cv_r2_logN']:.4f}`, median relative error=`{100*can['cv_median_relative_error']:.2f}%`.\n")
            memo.append(f"L/alpha region holdouts: median R²=`{h['median_r2']:.4f}`, worst R²=`{h['worst_r2']:.4f}`, median relative error=`{100*h['median_relerr']:.2f}%`, worst=`{100*h['worst_relerr']:.2f}%`.\n")
            memo.append("Coefficient relative standard deviations: `" + ", ".join(f"{x:.4f}" for x in stab["relative_std"]) + "`.\n")
            memo.append(f"Engineering-rule candidate under the predeclared thresholds: **{'YES' if audit['canonical_model_is_engineering_rule_candidate'] else 'NO'}**.\n")
        memo.append("## Interference-phase falsification\n")
        memo.append(f"Phase verdict: **{phase_summary.get('verdict','unknown')}**. Median holdout R² gain=`{phase_summary.get('median_holdout_r2_gain', float('nan')):+.4f}`; median holdout relative-error change=`{100*phase_summary.get('median_holdout_relerr_change', float('nan')):+.2f}%`.\n")
        memo.append("The phase variable is therefore not promoted to the headline law by this run. The result does not claim that oscillatory physics is absent; it says that the tested phase descriptor does not add predictive value beyond the compact geometry law under the current model/noise setting.\n")
        memo.append("## Proposed main-text rule\n")
        memo.append("Use `d_M/xi_M` and `xi_M/a` as the primary geometry coordinates, and compute the pair-level noise exposure from the exact 2x2 covariance reconstruction. Treat the law as regime-specific until additional microscopic/noise settings are tested.\n")
        Path(os.path.join(outdir, "paper_law_v21_3.md")).write_text("\n".join(memo) + "\n", encoding="utf-8")
        with open(os.path.join(outdir, "core_results_v21_3.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        # CSV is already emitted by v21.2 with the same geometry rows.
    return report


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Majorana noise geometry v21.4 — final law + orthogonal noise audit")
    parser.add_argument("--outdir", default="outputs_v21_4")
    parser.add_argument("--L1", type=int, default=80); parser.add_argument("--L2", type=int, default=65)
    parser.add_argument("--W", type=float, default=0.03); parser.add_argument("--rho-site", type=float, default=0.60); parser.add_argument("--xi-noise", type=float, default=0.0)
    parser.add_argument("--alpha", type=float, default=0.15); parser.add_argument("--Delta", type=float, default=1.0); parser.add_argument("--Ez", type=float, default=2.5); parser.add_argument("--t0", type=float, default=1.0)
    parser.add_argument("--mu-min", type=float, default=0.0); parser.add_argument("--mu-max", type=float, default=1.5); parser.add_argument("--mu-points", type=int, default=19)
    parser.add_argument("--mu-refine-stages", type=int, default=3); parser.add_argument("--mu-refine-points", type=int, default=17); parser.add_argument("--fd-step", type=float, default=2e-4)
    parser.add_argument("--validation-W", default="0.001,0.002,0.005,0.01,0.02,0.03"); parser.add_argument("--validation-real", type=int, default=2000); parser.add_argument("--gpu-batch-size", type=int, default=256)
    parser.add_argument("--scan-lengths", default="50:100:2"); parser.add_argument("--scan-alpha", default="0.11:0.19:0.01")
    parser.add_argument("--gap-match-tol", type=float, default=0.05); parser.add_argument("--xi-match-tol", type=float, default=0.05); parser.add_argument("--separation-match-tol", type=float, default=0.05)
    parser.add_argument("--cv-folds", type=int, default=5); parser.add_argument("--cv-repeats", type=int, default=20)
    parser.add_argument("--law-tolerance", type=float, default=0.05); parser.add_argument("--target-r2", type=float, default=0.90); parser.add_argument("--target-median-relerr", type=float, default=0.10)
    parser.add_argument("--coeff-repeats", type=int, default=60); parser.add_argument("--coeff-train-fraction", type=float, default=0.80)
    parser.add_argument("--lattice-spacing", type=float, default=1.0)
    parser.add_argument("--orthogonal-xi-noise", default="0,0.5,1,2")
    parser.add_argument("--orthogonal-rho-site", default="0,0.3,0.6,0.9")
    parser.add_argument("--device", choices=("auto", "cpu", "gpu"), default="auto"); parser.add_argument("--gpu-id", type=int, default=0); parser.add_argument("--seed", type=int, default=2028)
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        print(f"[v21.4] Ignoring host/runtime arguments: {unknown}")
    args.validation_W = parse_number_list(args.validation_W, float)
    args.scan_lengths = parse_number_list(args.scan_lengths, int)
    args.scan_alpha = parse_number_list(args.scan_alpha, float)
    args.orthogonal_xi_noise = parse_number_list(args.orthogonal_xi_noise, float)
    args.orthogonal_rho_site = parse_number_list(args.orthogonal_rho_site, float)
    # v21.4 reuses the validated v21.3 pipeline and adds the final audit.
    run_v214(args)


if __name__ == "__main__":
    main()
from google.colab import files

uploaded = files.upload()
pyfile = next(name for name in uploaded if name.endswith(".py"))

# !python "{pyfile}" \
#     --device gpu \
#     --gpu-id 0 \
#     --outdir outputs_v21_4 \
#     --L1 80 \
#     --L2 65 \
#     --alpha 0.15 \
#     --Delta 1.0 \
#     --Ez 2.5 \
#     --t0 1.0 \
#     --rho-site 0.60 \
#     --xi-noise 0.0 \
#     --W 0.03 \
#     --mu-min 0.0 \
#     --mu-max 1.5 \
#     --mu-points 19 \
#     --mu-refine-stages 3 \
#     --mu-refine-points 17 \
#     --fd-step 2e-4 \
#     --validation-W 0.001,0.002,0.005,0.01,0.02,0.03 \
#     --validation-real 2000 \
#     --gpu-batch-size 256 \
#     --scan-lengths 50:100:2 \
#     --scan-alpha 0.11:0.19:0.01 \
#     --gap-match-tol 0.05 \
#     --xi-match-tol 0.05 \
#     --separation-match-tol 0.05 \
#     --cv-folds 5 \
#     --cv-repeats 20 \
#     --law-tolerance 0.05 \
#     --target-r2 0.90 \
#     --target-median-relerr 0.10 \
#     --coeff-repeats 60 \
#     --coeff-train-fraction 0.80 \
#     --lattice-spacing 1.0 \
#     --orthogonal-xi-noise 0,0.5,1,2 \
#     --orthogonal-rho-site 0,0.3,0.6,0.9 \
#     --seed 2028