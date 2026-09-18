#!/usr/bin/env python3
"""Majorana-noise geometry v20.2

Core physics
------------
    geometry -> Majorana response K -> Cbar = K R K^T -> noise eigenmodes

For microscopic disorder covariance Sigma_mu = W^2 R,

    Sigma_epsilon = W^2 Cbar,
    Cbar = K R K^T.

The gauge-robust effective-noise eigenvalues are

    lambda_weak <= lambda_strong,

and the core design metric is

    N_geom = lambda_strong(Cbar).

The same Cbar gives the parity-sector projections

    S_p = v_p^T Sigma_epsilon v_p,  v_p=(1,p).

v20.2 changes the research test in two important ways:

1. Geometry matching now uses bulk gap, Majorana localization width, AND
   Majorana separation.
2. Even when no strict matched pair exists, the code always reports the
   nearest pair and a normalized match distance, so a coarse candidate grid
   cannot be mistaken for a physical null result.

GPU acceleration
----------------
The expensive full-BdG eigensolves are optionally accelerated with CuPy and
batched cuSOLVER calls. CuPy's Hermitian eigensolver supports batched matrices,
which is used here for both parameter-grid tuning and disorder validation.
For small 2x2 transfer calculations, CPU NumPy is retained because the GPU
launch/transfer overhead would dominate.

Device modes:
    --device auto   use GPU when CuPy+CUDA is available, otherwise CPU
    --device gpu    require GPU
    --device cpu    force CPU

The code is written to preserve float64/complex128 arithmetic for numerical
reliability. On an NVIDIA A100 this is useful because the problem sizes here
are small-to-medium dense Hermitian matrices and the workload is highly
batchable.

Outputs
-------
    core_results_v20_2.json
    kernels_v20_2.npz
    operating_point_v20_2.png
    noise_modes_v20_2.png
    geometry_design_v20_2.png
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Sequence, Tuple, List, Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
from scipy.optimize import minimize_scalar

try:
    import cupy as cp  # type: ignore
    CUPY_AVAILABLE = True
except Exception:
    cp = None  # type: ignore
    CUPY_AVAILABLE = False


# -----------------------------------------------------------------------------
# Small linear-algebra helpers
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
    """Parse comma lists or start:stop:step ranges (inclusive when possible)."""
    spec = spec.strip()
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
    if requested not in {"auto", "cpu", "gpu"}:
        raise ValueError("device must be auto, cpu, or gpu")
    info: Dict[str, Any] = {
        "requested": requested,
        "cupy_available": bool(CUPY_AVAILABLE),
        "gpu_id": int(gpu_id),
    }
    if requested == "cpu":
        return "cpu", info
    if not CUPY_AVAILABLE:
        if requested == "gpu":
            raise RuntimeError("--device gpu requested but CuPy is not installed")
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
# CPU/GPU batched BdG helpers
# -----------------------------------------------------------------------------


def _build_clean_H_stack(params_list: Sequence[BdGParams]) -> np.ndarray:
    mats = []
    for p in params_list:
        mats.append(BdGWire(p).finite_H(None))
    return np.stack(mats, axis=0)


def _build_clean_H_stack_fixed_L(base: BdGParams, mus: Sequence[float]) -> np.ndarray:
    params = [BdGParams(base.L, base.alpha, base.Delta, base.Ez, base.t0, float(mu)) for mu in mus]
    return _build_clean_H_stack(params)


def gpu_batch_lowest_positive(params_list: Sequence[BdGParams]) -> np.ndarray:
    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy is unavailable")
    H = cp.asarray(_build_clean_H_stack(params_list))
    vals = cp.linalg.eigvalsh(H)
    idx = vals.shape[-1] // 2
    return cp.asnumpy(vals[:, idx]).real


def _gpu_majorana_from_eigh(
    Hc: Any,
    Uc: Any,
    x: Any,
    tzI: Any,
    reference: Optional[np.ndarray] = None,
    need_kernel: bool = False,
) -> Dict[str, Any]:
    """Build Majorana modes from a batch of Hermitian eigenproblems on GPU."""
    vals, vecs = cp.linalg.eigh(Hc)
    nmat = Hc.shape[-1]
    idx = nmat // 2
    psi = vecs[:, :, idx]
    psi_h = cp.einsum("ij,bj->bi", Uc, cp.conj(psi))
    norm = cp.sqrt(cp.sum(cp.abs(psi_h) ** 2, axis=1, keepdims=True))
    psi_h = psi_h / cp.maximum(norm, 1e-15)
    G = cp.stack(((psi + psi_h) / cp.sqrt(2.0), -1j * (psi - psi_h) / cp.sqrt(2.0)), axis=2)
    # G: (B,N,2)

    xG = x[None, :, None] * G
    xmat = cp.real(cp.einsum("bni,bnj->bij", cp.conj(G), xG))
    xmat = 0.5 * (xmat + cp.swapaxes(xmat, -1, -2))
    _, U = cp.linalg.eigh(xmat)
    M = cp.einsum("bni,bij->bnj", G, U)

    site_M = M.reshape(M.shape[0], -1, 4, 2)
    prob = cp.sum(cp.abs(site_M) ** 2, axis=2)
    centers = cp.sum(prob * x[: (M.shape[1] // 4)][None, :, None], axis=1)
    swap = centers[:, 0] > centers[:, 1]
    if bool(cp.any(swap).item()):
        temp = M[swap, :, 0].copy()
        M[swap, :, 0] = M[swap, :, 1]
        M[swap, :, 1] = temp
        c0 = centers[swap, 0].copy()
        centers[swap, 0] = centers[swap, 1]
        centers[swap, 1] = c0
        site_M = M.reshape(M.shape[0], -1, 4, 2)
        prob = cp.sum(cp.abs(site_M) ** 2, axis=2)

    # Fixed-sign convention before optional reference alignment.
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

    nsite = M.shape[1] // 4
    xsite = x[:nsite]
    site_M = M.reshape(M.shape[0], nsite, 4, 2)
    prob = cp.sum(cp.abs(site_M) ** 2, axis=2)
    centers = cp.sum(prob * xsite[None, :, None], axis=1)
    widths = cp.sqrt(cp.maximum(
        cp.sum(prob * (xsite[None, :, None] - centers[:, None, :]) ** 2, axis=1),
        0.0,
    ))
    out: Dict[str, Any] = {
        "M": M,
        "epsilon": eps,
        "E_low": E_low,
        "centers": centers,
        "widths": widths,
    }
    if need_kernel:
        local = cp.einsum(
            "blia,ij,bljc->blac",
            cp.conj(site_M), tzI, site_M,
        )
        # dH/dmu_i = -tau_z, so insert minus sign.
        K = -cp.imag(local[:, :, 0, 1])
        out["K"] = K
    return out


def gpu_batch_wire_features(
    params_list: Sequence[BdGParams],
    reference_list: Optional[Sequence[np.ndarray]] = None,
    need_kernel: bool = True,
) -> List[Dict[str, Any]]:
    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy is unavailable")
    if not params_list:
        return []
    L = params_list[0].L
    if any(p.L != L for p in params_list):
        raise ValueError("gpu_batch_wire_features requires equal L within a batch")
    H = cp.asarray(_build_clean_H_stack(params_list))
    P = paulis()
    Uc_np = np.kron(P["Y"], P["Y"])
    Uc = cp.kron(cp.eye(L, dtype=cp.complex128), cp.asarray(Uc_np))
    x = cp.repeat(cp.arange(L, dtype=cp.float64), 4)
    tzI = cp.asarray(np.kron(P["Z"], P["I"]))
    refs = reference_list
    if refs is None:
        refs = [None] * len(params_list)
    # We need one common reference per batch only for final alignment; for clean
    # features the natural localized sign convention is already deterministic.
    raw = _gpu_majorana_from_eigh(H, Uc, x, tzI, reference=None, need_kernel=need_kernel)
    M = cp.asnumpy(raw["M"])
    eps = cp.asnumpy(raw["epsilon"])
    E_low = cp.asnumpy(raw["E_low"])
    centers = cp.asnumpy(raw["centers"])
    widths = cp.asnumpy(raw["widths"])
    K = cp.asnumpy(raw["K"]) if need_kernel else None
    out = []
    for i, p in enumerate(params_list):
        wire = BdGWire(p)
        # The batch-local M is already gauge fixed; make a reference from that
        # same basis for a numerically stable single-sample epsilon check.
        z2 = wire.z2()
        gap = wire.bulk_gap()
        out.append({
            "M": M[i],
            "epsilon": float(eps[i]),
            "E_low": float(E_low[i]),
            "E_gap": float(gap),
            "z2": int(z2),
            "centers": centers[i],
            "widths": widths[i],
            "K": K[i] if K is not None else None,
        })
    return out


def gpu_batch_epsilons(
    wire: BdGWire,
    reference: np.ndarray,
    disorders: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    """Evaluate signed epsilon for many disorder realizations on GPU."""
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
    for start in range(0, n, batch_size):
        stop = min(n, start + batch_size)
        db = cp.asarray(d[start:stop])
        H = cp.broadcast_to(H0c, (stop - start, H0.shape[0], H0.shape[1])).copy()
        diag = -db[:, site_idx] * cp.asarray(tz_diag)[None, :]
        ii = cp.arange(H.shape[-1])
        H[:, ii, ii] += diag
        raw = _gpu_majorana_from_eigh(H, Uc, x, cp.asarray(wire.tzI), reference=reference, need_kernel=False)
        out[start:stop] = cp.asnumpy(raw["epsilon"])
    return out


# -----------------------------------------------------------------------------
# Operating point / kernel
# -----------------------------------------------------------------------------


def make_wire(template: BdGWire, mu: float) -> BdGWire:
    p = template.p
    return BdGWire(BdGParams(p.L, p.alpha, p.Delta, p.Ez, p.t0, float(mu)))


def tune_operating_point_cpu(
    template: BdGWire,
    mu_min: float,
    mu_max: float,
    points: int,
) -> Tuple[BdGWire, np.ndarray, OperatingPoint]:
    mus = np.linspace(mu_min, mu_max, points)
    E = np.array([make_wire(template, mu).majoranas()["E_low"] for mu in mus], float)
    i = int(np.argmin(E))
    a = float(mus[max(0, i - 1)])
    b = float(mus[min(points - 1, i + 1)])
    if a == b:
        a, b = mu_min, mu_max
    res = minimize_scalar(
        lambda mu: float(make_wire(template, mu).majoranas()["E_low"]),
        bounds=(a, b), method="bounded", options={"xatol": 1e-9},
    )
    wire = make_wire(template, float(res.x))
    d = wire.majoranas()
    ref = np.asarray(d["M"], complex)
    c = np.asarray(d["centers"], float)
    w = np.asarray(d["widths"], float)
    op = OperatingPoint(
        template.p.L, template.p.alpha, float(res.x), float(d["epsilon"]),
        float(d["E_low"]), float(d["E_gap"]), int(d["z2"]),
        float(c[0]), float(c[1]), float(w[0]), float(w[1]), float(np.mean(w)),
        float(c[1] - c[0]),
    )
    return wire, ref, op


def tune_operating_point_gpu(
    template: BdGWire,
    mu_min: float,
    mu_max: float,
    points: int,
    refine_stages: int,
    refine_points: int,
) -> Tuple[BdGWire, np.ndarray, OperatingPoint]:
    lo, hi = float(mu_min), float(mu_max)
    best_mu = float(template.p.mu)
    for _ in range(max(1, refine_stages + 1)):
        mus = np.linspace(lo, hi, max(points if _ == 0 else refine_points, 5))
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
    # One final GPU eigensolve gives clean Majorana basis and kernel consistently.
    feat = gpu_batch_wire_features([wire.p], need_kernel=True)[0]
    ref = np.asarray(feat["M"], complex)
    centers = np.asarray(feat["centers"], float)
    widths = np.asarray(feat["widths"], float)
    op = OperatingPoint(
        wire.p.L, wire.p.alpha, best_mu, float(feat["epsilon"]),
        float(feat["E_low"]), float(feat["E_gap"]), int(feat["z2"]),
        float(centers[0]), float(centers[1]), float(widths[0]), float(widths[1]),
        float(np.mean(widths)), float(centers[1] - centers[0]),
    )
    return wire, ref, op


def tune_operating_point(
    template: BdGWire,
    mu_min: float,
    mu_max: float,
    points: int,
    device: str,
    refine_stages: int,
    refine_points: int,
) -> Tuple[BdGWire, np.ndarray, OperatingPoint]:
    if points < 5:
        raise ValueError("mu-points must be >= 5")
    if not mu_min < mu_max:
        raise ValueError("mu-min must be < mu-max")
    if device == "gpu":
        return tune_operating_point_gpu(template, mu_min, mu_max, points, refine_stages, refine_points)
    return tune_operating_point_cpu(template, mu_min, mu_max, points)


def majorana_kernel(wire: BdGWire, reference: np.ndarray, device: str) -> np.ndarray:
    if device == "gpu":
        feat = gpu_batch_wire_features([wire.p], need_kernel=True)[0]
        M = np.asarray(feat["M"], complex)
        # Align the batch basis to the supplied reference to ensure the same local convention.
        M = M @ wire.align(reference, M)
        # Recompute K from aligned M.
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
# Normalized microscopic noise covariance and transfer
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


def effective_noise(K1: np.ndarray, K2: np.ndarray, R: np.ndarray) -> Dict[str, Any]:
    K1 = np.asarray(K1, float)
    K2 = np.asarray(K2, float)
    K = np.block([
        [K1[None, :], np.zeros((1, K2.size))],
        [np.zeros((1, K1.size)), K2[None, :]],
    ])
    C = K @ R @ K.T
    C = 0.5 * (C + C.T)
    vals, vecs = la.eigh(C)
    vp = np.array([1.0, 1.0])
    vm = np.array([1.0, -1.0])
    lam_weak, lam_strong = float(vals[0]), float(vals[-1])
    return {
        "K": K,
        "Cbar": C,
        "eigenvalues": vals,
        "eigenvectors": vecs,
        "lambda_weak": lam_weak,
        "lambda_strong": lam_strong,
        "mode_ratio": float(lam_strong / max(lam_weak, 1e-30)),
        "N_geom": lam_strong,
        "S_plus_bar": float(vp @ C @ vp),
        "S_minus_bar": float(vm @ C @ vm),
    }


# -----------------------------------------------------------------------------
# Validation window
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
    device: str,
    gpu_batch_size: int,
) -> Dict[str, Any]:
    W_values = sorted(set(float(w) for w in W_values if float(w) > 0))
    d1u, d2u = sample_disorder(wire1.p.L, wire2.p.L, rho, xi, n, seed)
    rows = []
    for W in W_values:
        d1 = W * d1u
        d2 = W * d2u
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
# Geometry scan, pair matching, nearest-pair analysis
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
    device: str,
    mu_refine_stages: int,
    mu_refine_points: int,
) -> Tuple[List[Dict[str, float]], Dict[str, Any]]:
    records: List[Dict[str, float]] = []
    tune_meta: Dict[str, Any] = {"device": device, "successful": 0, "failed": 0}
    # Tune each (alpha,L), then evaluate the clean feature/K. GPU batches the
    # mu-grid search and final eigensolve within each candidate.
    for alpha in alpha_values:
        for L in L_values:
            try:
                template = BdGWire(BdGParams(int(L), float(alpha), Delta, Ez, t0, 0.5))
                wire, ref, op = tune_operating_point(
                    template, mu_min, mu_max, mu_points, device,
                    mu_refine_stages, mu_refine_points,
                )
                if op.z2 != -1:
                    tune_meta["failed"] += 1
                    continue
                K = majorana_kernel(wire, ref, device)
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
                tune_meta["successful"] += 1
            except (RuntimeError, ValueError, la.LinAlgError, FloatingPointError):
                tune_meta["failed"] += 1
    return records, tune_meta


def pair_metrics(a: Dict[str, float], b: Dict[str, float], tolerances: Tuple[float, float, float]) -> Dict[str, float]:
    gt, xt, st = tolerances
    gd = relative_difference(a["E_gap"], b["E_gap"])
    xd = relative_difference(a["xi_mean"], b["xi_mean"])
    sd = relative_difference(a["separation"], b["separation"])
    D = math.sqrt((gd / gt) ** 2 + (xd / xt) ** 2 + (sd / st) ** 2)
    ratio = max(a["N_geom"], b["N_geom"]) / max(min(a["N_geom"], b["N_geom"]), 1e-30)
    return {
        "relative_gap_difference": float(gd),
        "relative_xi_difference": float(xd),
        "relative_separation_difference": float(sd),
        "match_distance": float(D),
        "N_geom_ratio": float(ratio),
    }


def matched_pair(
    records: Sequence[Dict[str, float]],
    gap_tol: float,
    xi_tol: float,
    separation_tol: float,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    if len(records) < 2:
        return None, None
    tol = (float(gap_tol), float(xi_tol), float(separation_tol))
    strict_best = None
    strict_score = -np.inf
    nearest = None
    nearest_D = np.inf
    nearest_ratio = -np.inf
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            a, b = records[i], records[j]
            m = pair_metrics(a, b, tol)
            if m["match_distance"] < nearest_D - 1e-12 or (
                abs(m["match_distance"] - nearest_D) <= 1e-12 and m["N_geom_ratio"] > nearest_ratio
            ):
                nearest_D = m["match_distance"]
                nearest_ratio = m["N_geom_ratio"]
                nearest = {"A": a, "B": b, **m}
            strict = (
                m["relative_gap_difference"] <= gap_tol
                and m["relative_xi_difference"] <= xi_tol
                and m["relative_separation_difference"] <= separation_tol
            )
            if strict:
                # Reward ratio strongly, but lightly prefer tighter matching.
                score = math.log(max(m["N_geom_ratio"], 1.0)) - 0.05 * m["match_distance"]
                if score > strict_score:
                    strict_score = score
                    better = a if a["N_geom"] >= b["N_geom"] else b
                    quieter = b if better is a else a
                    strict_best = {
                        "A": a, "B": b, **m,
                        "higher_noise_geometry": better,
                        "lower_noise_geometry": quieter,
                        "match_score": float(score),
                    }
    return strict_best, nearest


# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------


def make_figures(
    outdir: str,
    K1: np.ndarray,
    K2: np.ndarray,
    transfer: Dict[str, Any],
    geometry: Sequence[Dict[str, float]],
    strict_pair: Optional[Dict[str, Any]],
    nearest_pair: Optional[Dict[str, Any]],
) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.plot(np.arange(K1.size), np.abs(K1), label="|K12(x)|")
    ax.plot(np.arange(K2.size), np.abs(K2), label="|K34(x)|")
    ax.set_xlabel("site index")
    ax.set_ylabel("|d epsilon_M / d mu_x|")
    ax.set_title("Majorana noise-response geometry")
    ax.grid(True, ls=":", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "operating_point_v20_2.png"), dpi=220)
    plt.close(fig)

    C = np.asarray(transfer["Cbar"])
    vals = np.asarray(transfer["eigenvalues"])
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    im = ax.imshow(C, origin="lower", aspect="equal", interpolation="nearest")
    ax.set_xticks([0, 1], ["epsilon12", "epsilon34"])
    ax.set_yticks([0, 1], ["epsilon12", "epsilon34"])
    ax.set_title(f"Cbar; weak={vals[0]:.2e}, strong={vals[-1]:.2e}")
    fig.colorbar(im, ax=ax, label="Cbar")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "noise_modes_v20_2.png"), dpi=220)
    plt.close(fig)

    if geometry:
        gap = np.array([r["E_gap"] for r in geometry])
        N = np.array([r["N_geom"] for r in geometry])
        xi = np.array([r["xi_mean"] for r in geometry])
        sep = np.array([r["separation"] for r in geometry])
        fig, ax = plt.subplots(figsize=(7.8, 5.0))
        sc = ax.scatter(gap, N, c=sep / np.maximum(xi, 1e-30), s=48)
        if strict_pair is not None:
            for label, key in (("strict A", "A"), ("strict B", "B")):
                r = strict_pair[key]
                ax.scatter([r["E_gap"]], [r["N_geom"]], s=120)
                ax.annotate(label, (r["E_gap"], r["N_geom"]), xytext=(5, 5), textcoords="offset points")
        elif nearest_pair is not None:
            for label, key in (("near A", "A"), ("near B", "B")):
                r = nearest_pair[key]
                ax.scatter([r["E_gap"]], [r["N_geom"]], s=120)
                ax.annotate(label, (r["E_gap"], r["N_geom"]), xytext=(5, 5), textcoords="offset points")
        ax.set_yscale("log")
        ax.set_xlabel("bulk gap E_gap")
        ax.set_ylabel("N_geom = lambda_strong(Cbar)")
        ax.set_title("Geometry-dependent effective noise")
        ax.grid(True, ls=":", alpha=0.4)
        fig.colorbar(sc, ax=ax, label="separation / xi_M")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "geometry_design_v20_2.png"), dpi=220)
        plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def run(args: argparse.Namespace) -> Dict[str, Any]:
    os.makedirs(args.outdir, exist_ok=True)
    device, device_info = choose_device(args.device, args.gpu_id)

    template1 = BdGWire(BdGParams(args.L1, args.alpha, args.Delta, args.Ez, args.t0, 0.5))
    template2 = BdGWire(BdGParams(args.L2, args.alpha, args.Delta, args.Ez, args.t0, 0.5))
    wire1, ref1, op1 = tune_operating_point(
        template1, args.mu_min, args.mu_max, args.mu_points, device,
        args.mu_refine_stages, args.mu_refine_points,
    )
    wire2, ref2, op2 = tune_operating_point(
        template2, args.mu_min, args.mu_max, args.mu_points, device,
        args.mu_refine_stages, args.mu_refine_points,
    )
    if op1.z2 != -1 or op2.z2 != -1:
        raise RuntimeError("At least one baseline wire is not topological")

    print("=" * 94)
    print("Majorana Noise Geometry v20.2")
    print("=" * 94)
    print(f"device: {device} | GPU info: {device_info}")
    for tag, op in (("wire1", op1), ("wire2", op2)):
        print(
            f"{tag}: L={op.L}, alpha={op.alpha:.3f}, mu*={op.mu:.8f}, "
            f"gap={op.E_gap:.3e}, xi={op.xi_mean:.3e}, separation={op.separation:.3e}, Z2={op.z2}"
        )

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

    validation = nonlinear_window(
        wire1, wire2, ref1, ref2, np.asarray(transfer["Cbar"]),
        args.rho_site, args.xi_noise, args.validation_W,
        args.validation_real, args.seed, device, args.gpu_batch_size,
    )
    print("eta_nl(W):", [(r["W"], r["eta_nl"]) for r in validation["rows"]])

    geometry, tune_meta = scan_geometry(
        args.scan_lengths, args.scan_alpha,
        args.Delta, args.Ez, args.t0,
        args.mu_min, args.mu_max, args.mu_points,
        K1, args.L1, args.rho_site, args.xi_noise,
        device, args.mu_refine_stages, args.mu_refine_points,
    )
    strict_pair, nearest_pair = matched_pair(
        geometry, args.gap_match_tol, args.xi_match_tol, args.separation_match_tol,
    )
    print(f"geometry candidates: {len(geometry)} (successful={tune_meta['successful']}, failed={tune_meta['failed']})")
    print(f"strict matched geometry pair: {'YES' if strict_pair else 'NO'}")
    if strict_pair:
        print(
            f"  strict ratio={strict_pair['N_geom_ratio']:.6f}; "
            f"gap={strict_pair['relative_gap_difference']:.3%}; "
            f"xi={strict_pair['relative_xi_difference']:.3%}; "
            f"separation={strict_pair['relative_separation_difference']:.3%}; "
            f"D_match={strict_pair['match_distance']:.4f}"
        )
    if nearest_pair:
        print(
            f"nearest pair: ratio={nearest_pair['N_geom_ratio']:.6f}; "
            f"gap={nearest_pair['relative_gap_difference']:.3%}; "
            f"xi={nearest_pair['relative_xi_difference']:.3%}; "
            f"separation={nearest_pair['relative_separation_difference']:.3%}; "
            f"D_match={nearest_pair['match_distance']:.4f}"
        )

    make_figures(args.outdir, K1, K2, transfer, geometry, strict_pair, nearest_pair)

    report: Dict[str, Any] = {
        "version": "v20.2",
        "device": device_info | {"selected": device},
        "core": {
            "effective_hamiltonian": "H_p = 1/2 [g X - z_p Z], z_p=-(epsilon_12+p epsilon_34)",
            "linear_response": "delta epsilon = K delta mu",
            "normalized_noise_transfer": "Cbar = K R K^T",
            "physical_covariance": "Sigma_epsilon = W^2 Cbar",
            "noise_eigenmodes": "Cbar = Q diag(lambda_weak, lambda_strong) Q^T",
            "parity_projection": "S_p = v_p^T Sigma_epsilon v_p, v_p=(1,p)",
            "design_metric": "N_geom = lambda_strong(Cbar)",
            "match_distance": "D = sqrt[(d_gap/t_gap)^2+(d_xi/t_xi)^2+(d_sep/t_sep)^2]",
        },
        "parameters": {
            k: getattr(args, k) for k in (
                "L1", "L2", "alpha", "Delta", "Ez", "t0", "rho_site", "xi_noise", "W",
                "mu_min", "mu_max", "mu_points", "mu_refine_stages", "mu_refine_points", "fd_step",
                "validation_real", "gpu_batch_size", "gap_match_tol", "xi_match_tol", "separation_match_tol",
                "seed", "device", "gpu_id",
            )
        },
        "operating_points": {"wire1": op1.as_dict(), "wire2": op2.as_dict()},
        "kernels": {
            "K1": K1.tolist(), "K2": K2.tolist(),
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
        "geometry_scan": geometry,
        "matched_geometry_pair": strict_pair,
        "nearest_geometry_pair": nearest_pair,
        "scientific_readout": {
            "both_topological": True,
            "kernel_fd_supported": bool(fd1 < 0.08 and fd2 < 0.08),
            "strict_pair_found": strict_pair is not None,
            "nearest_pair_found": nearest_pair is not None,
            "matching_constraints": {
                "relative_gap": float(args.gap_match_tol),
                "relative_xi": float(args.xi_match_tol),
                "relative_separation": float(args.separation_match_tol),
            },
            "universal_claim": False,
            "interpretation": (
                "A strict matched pair exists under gap, localization, and separation constraints; "
                "its N_geom ratio is a candidate geometry-level effect, not yet a universality claim."
                if strict_pair is not None else
                "No strict matched pair was found on this grid; nearest-pair distance is reported, "
                "so a coarse-grid null is not interpreted as a physical null."
            ),
        },
        "notes": [
            "Cbar is the central gauge-robust effective covariance; its eigenvalues are invariant under Majorana sign flips.",
            "The parity sectors are projections of the same Cbar, not separate noise models.",
            "W is factored out of the geometry law; Cbar depends on geometry and normalized noise correlations.",
            "Full-BdG validation checks the range in which Sigma_epsilon ~= W^2 Cbar is accurate.",
            "The matched-pair test constrains bulk gap, mean Majorana width, and Majorana separation simultaneously.",
            "Even without a strict pair, the nearest pair and normalized match distance are always reported.",
            "GPU mode batches dense Hermitian eigensolves; transfer-level 2x2 calculations stay on CPU to avoid needless transfers.",
        ],
    }

    np.savez(
        os.path.join(args.outdir, "kernels_v20_2.npz"),
        K12=K1, K34=K2,
        L1=np.array([args.L1]), L2=np.array([args.L2]),
        mu1_star=np.array([op1.mu]), mu2_star=np.array([op2.mu]),
        Cbar=np.asarray(transfer["Cbar"]),
    )
    with open(os.path.join(args.outdir, "core_results_v20_2.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"outputs -> {args.outdir}/")
    return report


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Majorana noise geometry v20.2 with dense matched-pair search and optional GPU acceleration")
    parser.add_argument("--outdir", default="outputs_v20_2")
    parser.add_argument("--L1", type=int, default=80)
    parser.add_argument("--L2", type=int, default=65)
    parser.add_argument("--W", type=float, default=0.03)
    parser.add_argument("--rho-site", type=float, default=0.60)
    parser.add_argument("--xi-noise", type=float, default=0.0)
    parser.add_argument("--alpha", type=float, default=0.15)
    parser.add_argument("--Delta", type=float, default=1.0)
    parser.add_argument("--Ez", type=float, default=2.5)
    parser.add_argument("--t0", type=float, default=1.0)
    parser.add_argument("--mu-min", type=float, default=0.0)
    parser.add_argument("--mu-max", type=float, default=1.5)
    parser.add_argument("--mu-points", type=int, default=19)
    parser.add_argument("--mu-refine-stages", type=int, default=3)
    parser.add_argument("--mu-refine-points", type=int, default=17)
    parser.add_argument("--fd-step", type=float, default=2e-4)
    parser.add_argument("--validation-W", default="0.001,0.002,0.005,0.01,0.02,0.03")
    parser.add_argument("--validation-real", type=int, default=2000)
    parser.add_argument("--gpu-batch-size", type=int, default=256)
    parser.add_argument("--scan-lengths", default="50:100:2")
    parser.add_argument("--scan-alpha", default="0.11:0.19:0.01")
    parser.add_argument("--gap-match-tol", type=float, default=0.05)
    parser.add_argument("--xi-match-tol", type=float, default=0.05)
    parser.add_argument("--separation-match-tol", type=float, default=0.05)
    parser.add_argument("--device", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2028)
    # Colab/Jupyter injects kernel arguments such as `-f <kernel.json>`.
    # Keep strict validation for our own CLI flags while ignoring host/runtime args.
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        print(f"[v20.2] Ignoring host/runtime arguments: {unknown}")

    args.validation_W = parse_number_list(args.validation_W, float)
    args.scan_lengths = parse_number_list(args.scan_lengths, int)
    args.scan_alpha = parse_number_list(args.scan_alpha, float)
    run(args)


if __name__ == "__main__":
    main()
