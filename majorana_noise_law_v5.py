#!/usr/bin/env python3
"""Majorana noise-kernel law v5.

Three layers (nothing else is "theory"; everything after Layer 3 is validation):

  Layer 1  microscopic model   : Model(A, B0, T, UC)  ->  H_k, H_finite, E_gap, Z2
  Layer 2  root law            : det[A z^2 + B_E z + C] = 0,  z = e^{ik}
                                 -> boundary matching -> E_pred -> psi_E(x)
  Layer 3  noise observable    : K(x) = <psi|-T_x|psi> = Im<gamma_L(x)|-T|gamma_R(x)>
                                 delta_eps = sum_x K(x) dmu(x),   N(xi) = K^T R(xi) K

Changes w.r.t. v4
  * K(x) einsum bug fixed ('la,ij,lj->l' summed a and i independently).  K is now the
    proper local matrix element; the Majorana-pair form is kept only as a cross-check.
  * Model-agnostic (n = 4 Rashba-Zeeman-SC wire, n = 2 Kitaev chain).
  * Independent checks: (i) sum K vs finite-difference d(eps)/d(mu),
    (ii) K(x) vs Im<gL|-T|gR>, (iii) end-to-end disorder test: exact diagonalisation with
    correlated random mu(x) vs first-order prediction |E0 + K.dmu|,
    (iv) q_K = fold(2 Re k_*) vs a refined DTFT peak of K(x).
  * GPU path removed (CPU only) to keep the code minimal.

NOTE: the root construction is an exact solution of the open chain (generalised Bloch /
transfer-matrix).  Agreement with exact diagonalisation is therefore a consistency check of
the construction, not a statistical prediction.
"""
from __future__ import annotations
import argparse, csv, json, math, time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import numpy as np
import scipy.linalg as la
from scipy.optimize import minimize_scalar

# =============================================================================
# Layer 1: microscopic model
# =============================================================================
I2 = np.eye(2); SX = np.array([[0., 1.], [1., 0.]]); SZ = np.diag([1., -1.])
SY = np.array([[0., -1j], [1j, 0.]])
TZI = np.kron(SZ, I2); TXI = np.kron(SX, I2); ISX = np.kron(I2, SX); TZSY = np.kron(SZ, SY)
UC4 = np.kron(SY, SY)


def pf2(M): return M[0, 1]
def pf4(M): return M[0, 1] * M[2, 3] - M[0, 2] * M[1, 3] + M[0, 3] * M[1, 2]


@dataclass
class Model:
    """H = sum_x [ c_x^† B(mu) c_x + c_x^† A c_{x+1} + h.c. ],  B(mu) = B0 - mu T (BdG blocks)."""
    name: str; n: int; A: np.ndarray; B0: np.ndarray; T: np.ndarray; UC: np.ndarray
    pf: Callable; params: dict

    def B(self, mu, E=0.0): return self.B0 - mu * self.T - E * np.eye(self.n)

    def parts(self, L):
        H0 = (np.kron(np.eye(L), self.B0).astype(complex) + np.kron(np.eye(L, k=1), self.A)
              + np.kron(np.eye(L, k=-1), self.A.conj().T))
        return H0, np.kron(np.eye(L), self.T)          # H(mu) = H0 - mu * Tf


def rashba(alpha, Ez, D, t):
    return Model('rashba', 4, -t * TZI - 0.5j * alpha * TZSY, 2 * t * TZI + Ez * ISX + D * TXI,
                 TZI, UC4, pf4, dict(alpha=alpha, Ez=Ez, Delta=D, t0=t))


def kitaev(D, t):
    return Model('kitaev', 2, -t * SZ - 0.5j * D * SY, 2 * t * SZ, SZ, SX, pf2, dict(Delta=D, t0=t))


def Hk(m, k, mu):
    k = np.atleast_1d(k).astype(float)
    return (m.B(mu)[None] + m.A[None] * np.exp(1j * k)[:, None, None]
            + m.A.conj().T[None] * np.exp(-1j * k)[:, None, None])


def bulk_gap(m, mu, nk=1201):
    return float(np.abs(np.linalg.eigvalsh(Hk(m, np.linspace(-np.pi, np.pi, nk), mu))).min())


def z2(m, mu):
    p0 = m.pf(Hk(m, 0., mu)[0] @ m.UC); p1 = m.pf(Hk(m, np.pi, mu)[0] @ m.UC)
    return -1 if np.real(p0 * p1) < 0 else 1


def exact_state(m, H, L):
    """lowest positive BdG level (E_+) and its eigenvector"""
    h = m.n * L // 2
    w, V = la.eigh(H, subset_by_index=[h - 1, h])
    return float(w[1]), V[:, 1]


def eplus(m, L, H0, Tf, mu):
    h = m.n * L // 2
    return float(la.eigvalsh(H0 - mu * Tf, subset_by_index=[h, h])[0])


# =============================================================================
# Layer 2: complex-root law
# =============================================================================
def fold(q): return abs((q + np.pi) % (2 * np.pi) - np.pi)


def roots_at(m, mu, E, strict=False):
    """n decaying (|z|<1) and n growing (|z|>1) roots of det[A z^2 + B_E z + C] = 0."""
    n = m.n; A = m.A; B = m.B(mu, E); C = m.A.conj().T
    Z = np.zeros((n, n))
    vals, vecs = la.eig(np.block([[-B, -C], [np.eye(n), Z]]), np.block([[A, Z], [Z, np.eye(n)]]))
    good = np.isfinite(vals); vals, vecs = vals[good], vecs[:, good]
    if len(vals) < 2 * n: return None
    o = np.argsort(np.abs(vals)); ii, oo = o[:n], o[-n:]
    if strict and not (abs(vals[o[n - 1]]) < 1 - 1e-8 and abs(vals[o[n]]) > 1 + 1e-8): return None

    def pack(idx):
        zs = vals[idx].astype(complex); vs = vecs[n:, idx].T.astype(complex)
        return zs, vs / np.maximum(la.norm(vs, axis=1), 1e-30)[:, None]
    zi, vi = pack(ii); zo, vo = pack(oo)
    k = np.argsort(-np.abs(zi))                               # zi[0] = slowest-decaying root k_*
    return dict(A=A, B=B, C=C, zi=zi[k], vi=vi[k], zo=zo, vo=vo)


def zero_root(m, mu):
    r = roots_at(m, mu, 0.0, strict=True)
    if r is None: return None
    z = r['zi'][0]
    return dict(z=z, qK=fold(2 * np.angle(z)), gamma=-math.log(max(abs(z), 1e-300)))


def kstar(m, mu):
    r = zero_root(m, mu)
    return None if r is None else abs(np.angle(r['z']))


def secular_matrix(r, L):
    A, B, C = r['A'], r['B'], r['C']; cols = []
    for z, v in zip(r['zi'], r['vi']):
        cols.append(np.concatenate([(B + A * z) @ v, (z ** (L - 1)) * ((B + C / z) @ v)]))
    for z, v in zip(r['zo'], r['vo']):
        cols.append(np.concatenate([(z ** (-(L - 1))) * ((B + A * z) @ v), (B + C / z) @ v]))
    return np.column_stack(cols)


def secular(m, L, mu, E):
    r = roots_at(m, mu, E)
    if r is None: return np.nan, None
    s = la.svdvals(secular_matrix(r, L))
    return float(s[-1] / max(s[0], 1e-300)), r


def predict_E(m, L, mu, gap):
    """open-boundary quantisation: smallest E>0 where the boundary matrix is singular"""
    g = max(gap, 1e-12); floor = max(1e-15, 1e-13 * g)
    grid = np.geomspace(floor, 0.999 * g, 320)
    vals = np.array([secular(m, L, mu, float(E))[0] for E in grid]); good = np.isfinite(vals)
    if not good.any(): return dict(E=np.nan, s=np.nan, status='no_grid', candidates=0)
    cand = []
    for j in range(1, len(grid) - 1):
        if good[j - 1] and good[j] and good[j + 1] and vals[j] <= vals[j - 1] and vals[j] <= vals[j + 1]:
            try:
                f = lambda le: secular(m, L, mu, math.exp(float(le)))[0]
                z = minimize_scalar(f, bounds=(math.log(grid[j - 1]), math.log(grid[j + 1])),
                                    method='bounded', options={'xatol': 1e-10})
                if np.isfinite(z.fun): cand.append((math.exp(z.x), float(z.fun)))
            except Exception:
                pass
    j = int(np.nanargmin(np.where(good, vals, np.inf))); cand.append((float(grid[j]), float(vals[j])))
    cand.sort(key=lambda c: c[1]); best = cand[0][1]
    near = [c for c in cand if c[1] <= max(100 * best, best + 1e-12)]
    E, s = min(near, key=lambda c: c[0])
    return dict(E=float(E), s=float(s), status='ok' if E < .999 * g else 'edge', candidates=len(cand))


def root_state(m, L, mu, E):
    """psi(x) = sum_j c_j z_j^x u_j at energy E; c = null vector of the boundary matrix"""
    r = roots_at(m, mu, E)
    if r is None: return None
    M = secular_matrix(r, L); _, sv, vh = la.svd(M); c = vh.conj().T[:, -1]
    n = m.n; x = np.arange(L); psi = np.zeros((L, n), complex)
    for cc, z, v in zip(c[:n], r['zi'], r['vi']): psi += cc * (z ** x)[:, None] * v[None, :]
    for cc, z, v in zip(c[n:], r['zo'], r['vo']): psi += cc * (z ** (x - L + 1))[:, None] * v[None, :]
    psi = psi.reshape(-1); psi /= max(la.norm(psi), 1e-30)
    return dict(psi=psi, secular=float(sv[-1] / max(sv[0], 1e-300)), roots=r)


# =============================================================================
# Layer 3: noise observable
# =============================================================================
def kernel_density(m, psi, L):
    """K(x) = <psi| -T_x |psi>  (local first-order coupling of the level to mu(x))"""
    p = psi.reshape(L, m.n)
    return np.real(np.einsum('xa,ab,xb->x', p.conj(), -m.T, p))


def R(L, xi):
    if xi == 0: return np.eye(L)
    if math.isinf(xi): return np.ones((L, L))
    x = np.arange(L); return np.exp(-np.abs(x[:, None] - x[None, :]) / xi)


def noise(K, L, xi): return float(K @ R(L, xi) @ K)          # N(xi) = K^T R(xi) K


# ---- Majorana-pair view (diagnostic only: widths, cross-check of K) -----------
def localize(G, L, n):
    S = G.conj().T @ G; S = 0.5 * (S + S.conj().T)
    ew, ev = la.eigh(S); G = G @ (ev @ np.diag(1 / np.sqrt(np.maximum(ew, 1e-14))) @ ev.conj().T)
    x = np.arange(L, dtype=float); X = np.real(G.conj().T @ (np.repeat(x, n)[:, None] * G))
    _, V = la.eigh(0.5 * (X + X.T)); G = G @ V
    c = x @ np.sum(np.abs(G.reshape(L, n, 2)) ** 2, axis=1)
    if c[0] > c[1]: G = G[:, ::-1]
    return G


def majorana_pair(m, psi, L):
    n = m.n; p = psi.reshape(L, n)
    ph = (p.conj() @ m.UC.T).reshape(-1); ph = ph / max(la.norm(ph), 1e-30)
    G = localize(np.column_stack([(psi + ph) / np.sqrt(2), -1j * (psi - ph) / np.sqrt(2)]), L, n)
    x = np.arange(L); pr = np.sum(np.abs(G.reshape(L, n, 2)) ** 2, axis=1); c = x @ pr
    width = float(np.mean(np.sqrt(np.sum((x[:, None] - c[None, :]) ** 2 * pr, axis=0))))
    gL, gR = G[:, 0].reshape(L, n), G[:, 1].reshape(L, n)
    Kmaj = np.imag(np.einsum('xa,ab,xb->x', gL.conj(), -m.T, gR))     # Im<gL|-T|gR>  (fixed einsum)
    return c, width, Kmaj


# ---- helpers ----------------------------------------------------------------
def q_peak(K):
    """continuous-q peak of the (Hann-windowed) DTFT of K(x)-mean, refined off-grid"""
    L = len(K); x = np.arange(L); y = (K - K.mean()) * np.hanning(L)
    qs = np.linspace(0, np.pi, max(16 * L, 1024)); P = np.abs(np.exp(-1j * np.outer(qs, x)) @ y) ** 2
    idx = np.where(qs > 4 * np.pi / L)[0]; j = idx[np.argmax(P[idx])]
    f = lambda q: -abs(np.exp(-1j * q * x) @ y) ** 2
    res = minimize_scalar(f, bounds=(qs[max(j - 1, 0)], qs[min(j + 1, len(qs) - 1)]),
                          method='bounded', options={'xatol': 1e-10})
    return float(res.x)


def err(a, b): return float(la.norm(a - b) / max(la.norm(b), 1e-300))
def corr(a, b): return float(abs(np.corrcoef(a, b)[0, 1]))


def clean(x):
    if isinstance(x, dict): return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [clean(v) for v in x]
    if isinstance(x, np.ndarray): return clean(x.tolist())
    if isinstance(x, (np.floating, np.integer)): return clean(x.item())
    if isinstance(x, (np.bool_,)): return bool(x)
    if isinstance(x, float) and not math.isfinite(x): return None
    return x


def stat(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return [float(np.median(v)), float(np.percentile(v, 95)), float(np.max(v))] if len(v) else [np.nan] * 3


# =============================================================================
# Point location in mu (node = zero crossing of eps, anti = extremum of eps)
# =============================================================================
def refine_mu(m, L, H0, Tf, mu, step, sign, lo, hi):
    for _ in range(6):
        g = np.clip(np.linspace(mu - step, mu + step, 11), lo, hi)
        E = np.array([eplus(m, L, H0, Tf, x) for x in g]); mu = float(g[np.argmin(sign * E)]); step *= 0.3
    return mu


def locate(m, L, lo, hi, rng):
    mm = .5 * (lo + hi); h = 1e-3; kp, km = kstar(m, mm + h), kstar(m, mm - h)
    if kp is None or km is None: return None
    dk = (kp - km) / (2 * h)
    if not np.isfinite(dk) or abs(dk) < 1e-6: return None
    W = min(10 * np.pi / (L * abs(dk)), .9 * (hi - lo)); st = lo + rng.uniform() * (hi - lo - W)
    g = np.linspace(st, st + W, 81); H0, Tf = m.parts(L)
    E = np.array([eplus(m, L, H0, Tf, x) for x in g])
    mins = [i for i in range(1, 80) if E[i] < E[i - 1] and E[i] <= E[i + 1]]
    if len(mins) < 2: return None
    j = (len(mins) - 2) // 2; i0, i1 = mins[j], mins[j + 1]
    ia = i0 + int(np.argmax(E[i0:i1 + 1])); step = g[1] - g[0]
    return (refine_mu(m, L, H0, Tf, float(g[i0]), step, +1, lo, hi),
            refine_mu(m, L, H0, Tf, float(g[ia]), step, -1, lo, hi))


# =============================================================================
# Validation
# =============================================================================
def fd_derivative(m, L, H0, Tf, mu, E0, su):
    """finite-difference dE_+/dmu.  Central at generic/anti points; one-sided (|.|) at nodes,
    where E_+ = |eps| has a kink and h must satisfy |eps'| h >> E_+."""
    h = 2e-5
    if E0 > 20 * abs(su) * h:
        return (eplus(m, L, H0, Tf, mu + h) - eplus(m, L, H0, Tf, mu - h)) / (2 * h), False
    h = min(max(h, 20 * E0 / max(abs(su), 1e-12)), 1e-3)
    Ep, Em = eplus(m, L, H0, Tf, mu + h), eplus(m, L, H0, Tf, mu - h)
    return (max(Ep, Em) - E0) / h, True


def mc_point(m, L, mu, K, E0, xis, nreal, scales, rng):
    """exact diagonalisation with correlated random mu(x) vs first-order |E0 + K.dmu|"""
    H0, Tf = m.parts(L); H = H0 - mu * Tf; tdiag = np.real(np.diag(Tf)); nn = m.n * L; hidx = nn // 2
    sig0 = 2e-6        # rms of mu(x) noise; small so that first order dominates (err should scale ~ sigma)
    out = []
    for xi in xis:
        if xi == 0: Sq = np.eye(L)
        elif math.isinf(xi): Sq = np.ones((L, 1))
        else:
            w, U = la.eigh(R(L, xi)); Sq = U * np.sqrt(np.maximum(w, 0))
        z = rng.standard_normal((Sq.shape[1], nreal))
        for sc in scales:
            dmu = sig0 * sc * (Sq @ z); resp = K @ dmu; pred = np.abs(E0 + resp); num = np.empty(nreal)
            for i in range(nreal):
                Hn = H.copy(); Hn[np.diag_indices(nn)] -= tdiag * np.repeat(dmu[:, i], m.n)
                num[i] = la.eigvalsh(Hn, subset_by_index=[hidx, hidx])[0]
            out.append(dict(xi=xi, scale=sc, sigma=sig0 * sc, resp_rms=float(la.norm(resp) / math.sqrt(nreal)),
                            err=float(la.norm(num - pred) / max(la.norm(resp), 1e-300))))
    return out


# =============================================================================
# One (model, L, mu) point
# =============================================================================
def analyze_point(m, L, mu, phase, cfg, xis, a):
    H0, Tf = m.parts(L); H = H0 - mu * Tf
    E0, psi = exact_state(m, H, L); K = kernel_density(m, psi, L)
    c, width, Kmaj = majorana_pair(m, psi, L)
    gap = bulk_gap(m, mu, a.gap_nk); topo = z2(m, mu); sub = bool(E0 < gap)
    near_edge = bool(E0 >= a.max_E_ratio * gap)
    dom = bool(topo == -1 and sub and not near_edge)
    N0 = float(K @ K); su = float(K.sum())
    fd, kink = fd_derivative(m, L, H0, Tf, mu, E0, su)
    fd_err = abs(abs(fd) - abs(su)) if kink else abs(fd - su)
    r = dict(cfg=cfg, phase=phase, model=m.name, L=L, mu=mu, **m.params, E_low=E0, E_gap=gap, z2=topo,
             subgap=sub, near_edge=near_edge, domain=dom, K=K, N0=N0, sumK=su, p0=su ** 2 / max(L * N0, 1e-300),
             width=width, center_left=c[0], center_right=c[1], fd=fd, fd_kink=kink, fd_err=fd_err,
             Kmaj_dev=float(min(np.abs(Kmaj - K).max(), np.abs(Kmaj + K).max()) / max(np.abs(K).max(), 1e-300)),
             q_fft=q_peak(K))
    if phase == 'anti':
        d = 2e-3; r['eps2'] = (eplus(m, L, H0, Tf, mu + d) + eplus(m, L, H0, Tf, mu - d) - 2 * E0) / d ** 2
    if dom:
        r0 = zero_root(m, mu); pred = predict_E(m, L, mu, gap); rs = None
        if r0 is not None and np.isfinite(pred['E']): rs = root_state(m, L, mu, pred['E'])
        r.update(E_pred=pred['E'], pred_status=pred['status'], q_root=r0['qK'] if r0 else np.nan,
                 gamma_root=r0['gamma'] if r0 else np.nan)
        if rs is not None:
            KR = kernel_density(m, rs['psi'], L); zi = rs['roots']['zi']
            r.update(has_root=True, KR=KR, secular=rs['secular'], E_relerr=abs(pred['E'] - E0) / max(E0, 1e-300),
                     E_abserr=abs(pred['E'] - E0), K_relerr=err(KR, K), K_corr=corr(KR, K), N0_ratio=float(KR @ KR) / max(N0, 1e-300),
                     q_energy=fold(2 * np.angle(zi[0])))
            r['q_diff_fft'] = fold(r['q_energy'] - r['q_fft']); r['q0_diff_fft'] = fold(r['q_root'] - r['q_fft'])
            # the DTFT peak is only resolvable when q_K is >4pi/L away from 0 and pi (+/-q peaks overlap otherwise)
            r['q_resolved'] = bool(4 * np.pi / L < r['q_energy'] < np.pi - 4 * np.pi / L)
            Nex = {xi: noise(K, L, xi) for xi in xis}; Nro = {xi: noise(KR, L, xi) for xi in xis}
            r['Nex'], r['Nro'] = Nex, Nro
            r['N_err'] = max(abs(Nro[x] - Nex[x]) for x in xis) / max(max(Nex.values()), 1e-300)
    return r


# =============================================================================
# Scan
# =============================================================================
def sample_rashba(rng, a):
    L = int(rng.integers(int(a.L_range[0]), int(a.L_range[1]) + 1))
    alpha, Ez, D, t = (rng.uniform(*q) for q in (a.alpha_range, a.Ez_range, a.Delta_range, a.t0_range))
    muc = math.sqrt(max(Ez ** 2 - D ** 2, 0)); lo = .15 * muc; hi = min(.9 * muc, 4 * t - Ez - .1)
    return rashba(alpha, Ez, D, t), L, lo, hi


def sample_kitaev(rng, a):
    L = int(rng.integers(int(a.L_range[0]), int(a.L_range[1]) + 1))
    D = rng.uniform(*a.Delta_range_kitaev); t = rng.uniform(*a.t0_range)
    r = math.sqrt(max(4 * t * t - D * D, 0))
    return kitaev(D, t), L, 2 * t - .6 * r, 2 * t + .6 * r


SAMPLERS = dict(rashba=sample_rashba, kitaev=sample_kitaev)


def scan_model(name, a, xis, out):
    rng = np.random.default_rng(a.seed + (0 if name == 'rashba' else 1000))
    rows = []; cfg = 0; tried = 0; t0 = time.time()
    print(f'\n=== model={name}: configs={a.n_configs}, seed={a.seed} ===', flush=True)
    while cfg < a.n_configs and tried < 30 * a.n_configs:
        tried += 1; m, L, lo, hi = SAMPLERS[name](rng, a)
        if hi - lo < .3: continue
        pts = locate(m, L, lo, hi, rng)
        if pts is None: continue
        cfg += 1
        for phase, mu in zip(('node', 'anti'), pts): rows.append(analyze_point(m, L, mu, phase, cfg, xis, a))
        if cfg % 10 == 0 or cfg == a.n_configs:
            print(f'  sampled {cfg}/{a.n_configs} configs ({tried} tried, {len(rows)} pts, {time.time() - t0:.0f}s)', flush=True)
    before = len(rows); rows = [r for r in rows if (r['L'] - 1) / max(r['width'], 1e-30) >= a.min_Lx]
    print(f'asymptotic cut: {before} -> {len(rows)} points with (L-1)/width >= {a.min_Lx:g}')
    dom = [r for r in rows if r['domain']]

    # ---- end-to-end disorder test on an evenly spaced subset ----
    mc = []
    if a.mc_points > 0:
        rngmc = np.random.default_rng(a.seed + 7)
        for phs in ('node', 'anti'):
            pool = [r for r in dom if r['phase'] == phs and r.get('has_root')]
            if not pool: continue
            for i in np.unique(np.linspace(0, len(pool) - 1, min(a.mc_points, len(pool))).astype(int)):
                r = pool[i]; m = rashba(r['alpha'], r['Ez'], r['Delta'], r['t0']) if name == 'rashba' \
                    else kitaev(r['Delta'], r['t0'])
                res = mc_point(m, r['L'], r['mu'], r['K'], r['E_low'], xis, a.mc_real, (1.0, 4.0), rngmc)
                for q in res: q.update(cfg=r['cfg'], phase=phs)
                mc += res
        print(f'MC disorder test: {len(mc)} (point, xi, sigma) cases done')
    return rows, dom, mc


def summarize(name, rows, dom, mc):
    S = dict(total=len(rows), subgap=sum(r['subgap'] for r in rows), above_gap=sum(not r['subgap'] for r in rows),
             domain=len(dom), construction_failures=sum(not r.get('has_root', False) for r in dom), phase={})
    for phs in ('node', 'anti'):
        rr = [r for r in dom if r['phase'] == phs]; g = lambda k: [r.get(k, np.nan) for r in rr]
        rq = [r for r in rr if r.get('q_resolved')]; gq = lambda k: [r.get(k, np.nan) for r in rq]
        d = dict(n=len(rr), E_abserr=stat(g('E_abserr')),
                 E_relerr_Elow_ge_1e9=stat([r['E_relerr'] for r in rr if r.get('has_root') and r['E_low'] >= 1e-9]),
                 n_Elow_lt_1e9=sum(1 for r in rr if r['E_low'] < 1e-9), K_relerr=stat(g('K_relerr')),
                 K_corr_min=float(np.nanmin(g('K_corr'))) if rr else np.nan, secular=stat(g('secular')),
                 N_err=stat(g('N_err')), N0_absdev=stat(np.abs(np.asarray(g('N0_ratio'), float) - 1)),
                 fd_err=stat(g('fd_err')), Kmaj_dev=stat(g('Kmaj_dev')),
                 n_q_resolved=len(rq), q_energy_vs_fft=stat(gq('q_diff_fft')), q_root_vs_fft=stat(gq('q0_diff_fft')),
                 q_energy_vs_fft_units_of_2pi_over_L=stat([r['q_diff_fft'] / (2 * np.pi / r['L']) for r in rq]),
                 p0_median=float(np.nanmedian(g('p0'))) if rr else np.nan,
                 absK_sum_over_sqrtN0_median=float(np.nanmedian([abs(r['sumK']) / math.sqrt(max(r['N0'], 1e-300)) for r in rr])) if rr else np.nan)
        if phs == 'anti': d['eps2_abs_median'] = float(np.nanmedian(np.abs(g('eps2')))) if rr else np.nan
        if phs == 'node': d['dEdmu_abs_median'] = float(np.nanmedian(np.abs([r['sumK'] for r in rr]))) if rr else np.nan
        # MC: only (point, sigma) entries where the response is at least 5% of the largest xi for that point
        mm = []
        for cfg_ in set(q['cfg'] for q in mc if q['phase'] == phs):
            for sc in (1.0, 4.0):
                qs = [q for q in mc if q['phase'] == phs and q['cfg'] == cfg_ and q['scale'] == sc]
                if not qs: continue
                mx = max(q['resp_rms'] for q in qs)
                mm += [(q['scale'], q['err'], q['xi']) for q in qs if q['resp_rms'] >= .05 * mx]
        d['mc_err_sigma1'] = stat([e for s, e, _ in mm if s == 1.0]); d['mc_err_sigma4'] = stat([e for s, e, _ in mm if s == 4.0])
        S['phase'][phs] = d
    return S


def print_summary(name, S):
    f = lambda v: '/'.join(f'{x:.2e}' for x in v)
    print(f'\nSUMMARY [{name}]  (median/p95/max)')
    print(f"total={S['total']} subgap={S['subgap']} above_gap={S['above_gap']} domain={S['domain']} "
          f"construction_failures={S['construction_failures']}")
    for phs, d in S['phase'].items():
        print(f" {phs}: n={d['n']}")
        print(f"   [root closure] E_abserr={f(d['E_abserr'])}  E_relerr(E_low>=1e-9, {d['n'] - d['n_Elow_lt_1e9']} pts)={f(d['E_relerr_Elow_ge_1e9'])}  K_relerr={f(d['K_relerr'])}  K_corr_min={d['K_corr_min']:.8f}"
              f"  secular={f(d['secular'])}")
        print(f"   [noise closure] N_err(norm. by max_xi N)={f(d['N_err'])}  N0 absdev={f(d['N0_absdev'])}")
        print(f"   [independent]  |dE/dmu FD - sumK|={f(d['fd_err'])}   K vs Im<gL|-T|gR> dev={f(d['Kmaj_dev'])}")
        print(f"   [q_K rule]     ({d['n_q_resolved']}/{d['n']} resolved) |q_energy-q_fft|={f(d['q_energy_vs_fft'])} rad; in units of 2pi/L={f(d['q_energy_vs_fft_units_of_2pi_over_L'])}"
              f";  |q_root(E=0)-q_fft|={f(d['q_root_vs_fft'])}")
        print(f"   [cancellation] median p0=(sumK)^2/(L N0)={d['p0_median']:.3e}; median |sumK|/sqrt(N0)={d['absK_sum_over_sqrtN0_median']:.3e}")
        print(f"   [disorder MC]  err(sigma)={f(d['mc_err_sigma1'])}   err(4 sigma)={f(d['mc_err_sigma4'])}"
              f"   (first-order => err ~ sigma)")
        if 'eps2_abs_median' in d: print(f"   [anti] median |d2E/dmu2|={d['eps2_abs_median']:.3e} (second-order coefficient; uniform noise is O(dmu^2) here)")
        if 'dEdmu_abs_median' in d: print(f"   [node] median |dE/dmu|={d['dEdmu_abs_median']:.3e}")


def write_outputs(name, rows, S, mc, xis, out, a):
    xk = [str(x) for x in xis]
    fields = ['cfg', 'phase', 'model', 'L', 'mu', 'alpha', 'Ez', 'Delta', 't0', 'E_low', 'E_gap', 'z2', 'subgap', 'near_edge', 'domain',
              'width', 'center_left', 'center_right', 'E_pred', 'E_abserr', 'E_relerr', 'q_resolved', 'secular', 'K_relerr', 'K_corr', 'N0_ratio',
              'N_err', 'q_root', 'q_energy', 'q_fft', 'q_diff_fft', 'q0_diff_fft', 'sumK', 'fd', 'fd_kink', 'fd_err',
              'Kmaj_dev', 'p0', 'N0', 'eps2'] + [f'Nex_xi{x}' for x in xk] + [f'Nro_xi{x}' for x in xk]
    with open(out / f'v5_{name}_points.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore'); w.writeheader()
        for r in rows:
            d = dict(r)
            for x in xis:
                d[f'Nex_xi{x}'] = r.get('Nex', {}).get(x, np.nan); d[f'Nro_xi{x}'] = r.get('Nro', {}).get(x, np.nan)
            w.writerow(clean(d))
    ks = {f'K_exact_{i}': r['K'] for i, r in enumerate(rows)}
    ks.update({f'K_root_{i}': r['KR'] for i, r in enumerate(rows) if 'KR' in r})
    np.savez_compressed(out / f'v5_{name}_kernels.npz', **ks)
    with open(out / f'v5_{name}_mc.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['cfg', 'phase', 'xi', 'scale', 'sigma', 'resp_rms', 'err']); w.writeheader()
        for q in mc: w.writerow(clean(q))
    return dict(summary=S, scan=dict(configs=a.n_configs, seed=a.seed, xi=[str(x) for x in xis], min_Lx=a.min_Lx))


# =============================================================================
def main():
    pair = lambda s: tuple(float(x) for x in s.split(','))
    ap = argparse.ArgumentParser(description='Majorana noise-kernel law v5')
    ap.add_argument('--models', default='rashba,kitaev'); ap.add_argument('--n-configs', type=int, default=200)
    ap.add_argument('--seed', type=int, default=2028)
    ap.add_argument('--L-range', type=pair, default=(40, 100)); ap.add_argument('--alpha-range', type=pair, default=(.10, .30))
    ap.add_argument('--Ez-range', type=pair, default=(1.8, 3.0)); ap.add_argument('--Delta-range', type=pair, default=(.7, 1.3))
    ap.add_argument('--Delta-range-kitaev', type=pair, default=(.15, .35)); ap.add_argument('--t0-range', type=pair, default=(.8, 1.4))
    ap.add_argument('--xi-noise', default='0,1,2,5,10,20,50,inf'); ap.add_argument('--gap-nk', type=int, default=1201)
    ap.add_argument('--min-Lx', type=float, default=4)
    ap.add_argument('--max-E-ratio', type=float, default=0.85,
                 help='physical validity cut: only points with E_low < ratio * E_gap enter domain')
    ap.add_argument('--mc-points', type=int, default=8, help='points per phase for the disorder test (0 = skip)')
    ap.add_argument('--mc-real', type=int, default=16, help='disorder realisations per (point, xi, sigma)')
    ap.add_argument('--outdir', default='outputs_v5')
    a = ap.parse_args(); out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    xis = sorted({float(s) if s != 'inf' else np.inf for s in a.xi_noise.split(',')})
    payload = {'theory': {'roots': 'det[A z^2 + B_E z + C] = 0, z = e^{ik}', 'energy': 'open-boundary secular minimum',
                          'kernel': 'K(x) = <psi|-T_x|psi> = Im<gL(x)|-T|gR(x)>', 'first_order': 'd eps = sum_x K(x) d mu(x)',
                          'noise': 'N(xi) = K^T R(xi) K', 'wavevector': 'q_K = fold(2 Re k_*)',
                          'HF': 'sum_x K(x) = dE_+/dmu', 'uniform': 'N(inf) = (sum K)^2'}, 'models': {}}
    for name in [s.strip() for s in a.models.split(',') if s.strip()]:
        rows, dom, mc = scan_model(name, a, xis, out); S = summarize(name, rows, dom, mc); print_summary(name, S)
        payload['models'][name] = write_outputs(name, rows, S, mc, xis, out, a)
    with open(out / 'v5_results.json', 'w', encoding='utf-8') as f: json.dump(clean(payload), f, indent=1, allow_nan=False)
    with open(out / 'law_card_v5.md', 'w', encoding='utf-8') as f:
        f.write('# Majorana noise-kernel law v5\n\n'
                '$H\\rightarrow z_j(E)\\rightarrow\\psi_E(x)\\rightarrow K(x)=\\langle\\psi|-T_x|\\psi\\rangle'
                '\\rightarrow N=K^TRK$, $\\sum K=\\partial_\\mu E_+$.\n\n'
                'Root construction = exact open-chain solution (consistency check, not a fit). '
                'Independent checks: finite-difference $\\partial_\\mu E$, exact-diagonalisation disorder test, DTFT peak vs $2\\,\\mathrm{Re}\\,k_*$.\n')
    print(f'\noutputs -> {out}/')


if __name__ == '__main__':
    main()
# (quantumcomputing) marcelo-group@marcelo-group-2:~/Desktop/Miscellaneous$ python majorana_noise_law_v5.py --models rashba,kitaev --n-configs 200 --seed 2028 --max-E-ratio 0.85 --outdir outputs_v5

# === model=rashba: configs=200, seed=2028 ===
#   sampled 10/200 configs (10 tried, 20 pts, 30s)
#   sampled 20/200 configs (20 tried, 40 pts, 53s)
#   sampled 30/200 configs (30 tried, 60 pts, 82s)
#   sampled 40/200 configs (42 tried, 80 pts, 108s)
#   sampled 50/200 configs (53 tried, 100 pts, 137s)
#   sampled 60/200 configs (64 tried, 120 pts, 164s)
#   sampled 70/200 configs (76 tried, 140 pts, 193s)
#   sampled 80/200 configs (86 tried, 160 pts, 220s)
#   sampled 90/200 configs (97 tried, 180 pts, 248s)
#   sampled 100/200 configs (108 tried, 200 pts, 276s)
#   sampled 110/200 configs (121 tried, 220 pts, 300s)
#   sampled 120/200 configs (131 tried, 240 pts, 325s)
#   sampled 130/200 configs (142 tried, 260 pts, 356s)
#   sampled 140/200 configs (152 tried, 280 pts, 385s)
#   sampled 150/200 configs (164 tried, 300 pts, 408s)
#   sampled 160/200 configs (174 tried, 320 pts, 433s)
#   sampled 170/200 configs (184 tried, 340 pts, 463s)
#   sampled 180/200 configs (196 tried, 360 pts, 489s)
#   sampled 190/200 configs (207 tried, 380 pts, 510s)
#   sampled 200/200 configs (218 tried, 400 pts, 533s)
# asymptotic cut: 400 -> 383 points with (L-1)/width >= 4
# MC disorder test: 256 (point, xi, sigma) cases done

# SUMMARY [rashba]  (median/p95/max)
# total=383 subgap=378 above_gap=5 domain=376 construction_failures=0
#  node: n=183
#    [root closure] E_abserr=1.25e-14/8.77e-14/1.68e-13  E_relerr(E_low>=1e-9, 178 pts)=5.52e-08/1.36e-07/5.28e-07  K_relerr=5.14e-13/1.95e-12/4.10e-12  K_corr_min=1.00000000  secular=4.42e-14/3.79e-13/7.65e-13
#    [noise closure] N_err(norm. by max_xi N)=6.32e-14/3.81e-13/7.97e-12  N0 absdev=8.14e-14/4.13e-13/7.97e-12
#    [independent]  |dE/dmu FD - sumK|=8.40e-07/2.97e-06/5.05e-06   K vs Im<gL|-T|gR> dev=8.50e-16/2.31e-14/8.51e-12
#    [q_K rule]     (164/183 resolved) |q_energy-q_fft|=2.21e-06/2.35e-04/9.74e-04 rad; in units of 2pi/L=2.47e-05/2.37e-03/7.07e-03;  |q_root(E=0)-q_fft|=2.21e-06/2.35e-04/9.74e-04
#    [cancellation] median p0=(sumK)^2/(L N0)=6.782e-01; median |sumK|/sqrt(N0)=6.952e+00
#    [disorder MC]  err(sigma)=1.33e-06/6.27e-06/9.07e-06   err(4 sigma)=5.33e-06/2.51e-05/3.63e-05   (first-order => err ~ sigma)
#    [node] median |dE/dmu|=2.872e-01
#  anti: n=193
#    [root closure] E_abserr=1.11e-10/5.90e-10/1.12e-09  E_relerr(E_low>=1e-9, 193 pts)=1.95e-08/5.15e-08/6.12e-08  K_relerr=9.84e-09/2.43e-08/3.36e-08  K_corr_min=1.00000000  secular=3.86e-10/2.07e-09/3.02e-09
#    [noise closure] N_err(norm. by max_xi N)=9.07e-09/2.49e-08/3.47e-08  N0 absdev=9.07e-09/2.49e-08/3.47e-08
#    [independent]  |dE/dmu FD - sumK|=2.19e-10/1.05e-09/2.73e-09   K vs Im<gL|-T|gR> dev=1.07e-15/2.96e-14/9.89e-13
#    [q_K rule]     (174/193 resolved) |q_energy-q_fft|=2.73e-04/8.88e-04/1.38e-03 rad; in units of 2pi/L=3.28e-03/6.60e-03/9.89e-03;  |q_root(E=0)-q_fft|=2.91e-04/1.01e-03/1.58e-03
#    [cancellation] median p0=(sumK)^2/(L N0)=9.559e-09; median |sumK|/sqrt(N0)=8.501e-04
#    [disorder MC]  err(sigma)=3.49e-04/5.42e-03/9.23e-03   err(4 sigma)=1.40e-03/2.17e-02/3.69e-02   (first-order => err ~ sigma)
#    [anti] median |d2E/dmu2|=1.125e+01 (second-order coefficient; uniform noise is O(dmu^2) here)

# === model=kitaev: configs=200, seed=2028 ===
#   sampled 10/200 configs (10 tried, 20 pts, 9s)
#   sampled 20/200 configs (20 tried, 40 pts, 19s)
#   sampled 30/200 configs (30 tried, 60 pts, 30s)
#   sampled 40/200 configs (40 tried, 80 pts, 39s)
#   sampled 50/200 configs (50 tried, 100 pts, 48s)
#   sampled 60/200 configs (60 tried, 120 pts, 58s)
#   sampled 70/200 configs (70 tried, 140 pts, 67s)
#   sampled 80/200 configs (80 tried, 160 pts, 75s)
#   sampled 90/200 configs (90 tried, 180 pts, 85s)
#   sampled 100/200 configs (100 tried, 200 pts, 95s)
#   sampled 110/200 configs (110 tried, 220 pts, 105s)
#   sampled 120/200 configs (120 tried, 240 pts, 114s)
#   sampled 130/200 configs (130 tried, 260 pts, 124s)
#   sampled 140/200 configs (140 tried, 280 pts, 137s)
#   sampled 150/200 configs (150 tried, 300 pts, 144s)
#   sampled 160/200 configs (160 tried, 320 pts, 156s)
#   sampled 170/200 configs (170 tried, 340 pts, 167s)
#   sampled 180/200 configs (180 tried, 360 pts, 178s)
#   sampled 190/200 configs (190 tried, 380 pts, 189s)
#   sampled 200/200 configs (200 tried, 400 pts, 198s)
# asymptotic cut: 400 -> 400 points with (L-1)/width >= 4
# MC disorder test: 256 (point, xi, sigma) cases done

# SUMMARY [kitaev]  (median/p95/max)
# total=400 subgap=400 above_gap=0 domain=400 construction_failures=0
#  node: n=200
#    [root closure] E_abserr=5.61e-16/5.02e-14/2.88e-13  E_relerr(E_low>=1e-9, 130 pts)=7.76e-08/2.01e-07/4.42e-07  K_relerr=1.41e-12/3.50e-08/1.31e-06  K_corr_min=1.00000000  secular=1.36e-15/1.44e-13/6.72e-13
#    [noise closure] N_err(norm. by max_xi N)=2.35e-13/2.15e-09/2.62e-06  N0 absdev=3.40e-13/7.91e-09/2.62e-06
#    [independent]  |dE/dmu FD - sumK|=3.97e-09/2.86e-07/5.31e-07   K vs Im<gL|-T|gR> dev=6.05e-14/8.69e-11/1.38e-06
#    [q_K rule]     (140/200 resolved) |q_energy-q_fft|=5.30e-05/5.92e-04/1.04e-03 rad; in units of 2pi/L=5.79e-04/6.99e-03/7.08e-03;  |q_root(E=0)-q_fft|=5.30e-05/5.92e-04/1.04e-03
#    [cancellation] median p0=(sumK)^2/(L N0)=6.763e-01; median |sumK|/sqrt(N0)=6.782e+00
#    [disorder MC]  err(sigma)=1.61e-06/7.62e-06/1.96e-05   err(4 sigma)=4.24e-06/2.11e-05/3.06e-05   (first-order => err ~ sigma)
#    [node] median |dE/dmu|=3.601e-03
#  anti: n=200
#    [root closure] E_abserr=2.79e-12/1.66e-10/7.66e-10  E_relerr(E_low>=1e-9, 199 pts)=3.11e-08/9.71e-08/1.66e-07  K_relerr=6.64e-09/2.44e-08/8.90e-06  K_corr_min=1.00000000  secular=7.09e-12/4.17e-10/2.50e-09
#    [noise closure] N_err(norm. by max_xi N)=4.66e-09/1.55e-08/2.19e-06  N0 absdev=4.66e-09/1.55e-08/3.19e-06
#    [independent]  |dE/dmu FD - sumK|=4.84e-12/1.90e-11/3.30e-11   K vs Im<gL|-T|gR> dev=1.41e-13/1.30e-10/1.65e-06
#    [q_K rule]     (139/200 resolved) |q_energy-q_fft|=6.65e-05/5.67e-04/7.56e-04 rad; in units of 2pi/L=7.85e-04/5.03e-03/7.98e-03;  |q_root(E=0)-q_fft|=6.65e-05/5.67e-04/7.56e-04
#    [cancellation] median p0=(sumK)^2/(L N0)=8.362e-09; median |sumK|/sqrt(N0)=7.421e-04
#    [disorder MC]  err(sigma)=2.63e-04/3.81e-03/7.61e-03   err(4 sigma)=1.06e-03/1.51e-02/3.04e-02   (first-order => err ~ sigma)
#    [anti] median |d2E/dmu2|=1.210e-01 (second-order coefficient; uniform noise is O(dmu^2) here)

# outputs -> outputs_v5/
# (quantumcomputing) marcelo-group@marcelo-group-2:~/Desktop/Miscellaneous$ 