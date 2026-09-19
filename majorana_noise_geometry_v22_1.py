#!/usr/bin/env python3
"""Majorana noise-kernel law v22.1 (theory-first; CuPy-batched mu scan, CPU/GPU dual track).

Chain:  microscopic wire -> K(x) = d eps/d mu(x) -> N(xi_n) = K^T R(xi_n) K -> law.

Goals / hypotheses (theory-first and falsifiable):
 H1 envelope law  : N(0) = kappa * L * xi^-2 * exp[-(L-1)/xi]      (xi = RMS width of a Majorana)
                    K(x) is FLAT in the bulk (product of two opposite exponentials), so
                    N = sum K^2 carries a prefactor L/xi^2, not just e^{-L/xi}.
 H2 two-channel   : N(xi_n)/N(0) = p0*u(xi_n,L) + (1-p0)*lam(xi_n,q0),   q0 = 2 kF (mod 2pi)
                    p0 = (sum K)^2 / (L sum K^2) = (d eps/d mu)^2/(L N0);   exact at xi_n = 0 and inf.
 H3 pair rule     : N_geom = nbar*[1 + sqrt(delta^2 + (1-delta^2) rho_eff^2)], delta = tanh(Delta/2),
                    Delta = ln(n1/n2)  ->  correlated noise is suppressed by exponent mismatch.
Scan varies L, alpha, Ez, Delta, t0 AND mu across a six-parameter microscopic space, at two operating phases:
 'node' (eps = 0 sweet spot) and 'anti' (eps extremum = d eps/d mu ~ 0).
Only the mu-scan (batched eigvalsh) is heavy -> runs on GPU; everything else is cheap NumPy.
"""
from __future__ import annotations
import argparse, csv, json, math, os, time
import numpy as np
import scipy.linalg as la
from scipy.optimize import brentq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    import cupy as cp
except Exception:
    cp = None

# ------------------------------------------------------------------ model ----
I2 = np.eye(2); SX = np.array([[0., 1.], [1., 0.]]); SZ = np.diag([1., -1.])
SY = np.array([[0, -1j], [1j, 0]])
TZI, TXI, ISX = np.kron(SZ, I2), np.kron(SX, I2), np.kron(I2, SX)
TZSY, UC = np.kron(SZ, SY), np.kron(SY, SY)          # UC: particle-hole operator (C = UC * K)


def build_H0(L, alpha, Ez, Delta, t0):
    """Finite Rashba BdG wire, H(mu) = H0 - mu*T  (basis: 4 = Nambu x spin per site)."""
    site = 2*t0*TZI + Ez*ISX + Delta*TXI
    hop = -t0*TZI - 0.5j*alpha*TZSY
    H0 = np.kron(np.eye(L), site).astype(complex)
    H0 += np.kron(np.eye(L, k=1), hop) + np.kron(np.eye(L, k=-1), hop.conj().T)
    return H0, np.kron(np.eye(L), TZI)


def bloch(k, alpha, Ez, Delta, t0, mu):
    k = np.atleast_1d(k).astype(float)
    xi = (2*t0 - 2*t0*np.cos(k) - mu)[:, None, None]
    return xi*TZI + (alpha*np.sin(k))[:, None, None]*TZSY + Ez*ISX + Delta*TXI


def bulk_gap(*p, nk=2001):                            # p = alpha, Ez, Delta, t0, mu
    return float(np.abs(np.linalg.eigvalsh(bloch(np.linspace(-np.pi, np.pi, nk), *p))).min())


def pf4(A):
    return A[0, 1]*A[2, 3] - A[0, 2]*A[1, 3] + A[0, 3]*A[1, 2]


def z2(*p):
    return -1 if np.real(pf4(bloch(0.0, *p)[0] @ UC)*pf4(bloch(np.pi, *p)[0] @ UC)) < 0 else 1


def k_fermi(mu, alpha, Ez, t0):
    """Smallest positive Fermi root of the lower normal-state helical band (nan if none)."""
    f = lambda k: 2*t0 - 2*t0*np.cos(k) - mu - np.sqrt(Ez**2 + (alpha*np.sin(k))**2)
    ks = np.linspace(1e-6, np.pi - 1e-6, 400); v = f(ks)
    i = np.where(v[:-1]*v[1:] < 0)[0]
    return float(brentq(f, ks[i[0]], ks[i[0] + 1])) if i.size else float("nan")


def majorana(H, L):
    """Two end Majoranas of the lowest positive BdG state; returns K(x) = -Im<g1|tau_z(x)|g2>."""
    w, V = la.eigh(H, subset_by_index=[2*L - 1, 2*L])
    psi = V[:, 1]
    psih = (psi.conj().reshape(L, 4) @ UC.T).reshape(-1)
    psih /= la.norm(psih)
    G = np.column_stack([(psi + psih)/np.sqrt(2), -1j*(psi - psih)/np.sqrt(2)])
    xs = np.arange(L, dtype=float)                    # physical site positions: 0,1,...,L-1
    X = np.real(G.conj().T @ (np.repeat(xs, 4)[:, None]*G))
    M = G @ la.eigh(0.5*(X + X.T))[1]
    pr = (np.abs(M)**2).reshape(L, 4, 2).sum(1)
    c = xs @ pr
    if c[0] > c[1]:
        M, pr, c = M[:, ::-1], pr[:, ::-1], c[::-1]
    width = np.sqrt(((xs[:, None] - c[None, :])**2*pr).sum(0))
    Ms = M.reshape(L, 4, 2)
    K = np.imag(np.einsum("lia,ij,ljb->lab", Ms.conj(), -TZI, Ms)[:, 0, 1])
    return dict(M=M, E=float(w[1]), xi=float(width.mean()), d=float(c[1] - c[0]), K=K)


def fd_uniform(H0, T, L, mu, m, h=1e-4):
    """|sum K| vs finite-difference d eps/d mu (uniform shift), gauge-aligned."""
    def eps(mv):
        H = H0 - mv*T
        Mm = majorana(H, L)["M"]
        U, _, Vh = la.svd(np.real(m["M"].conj().T @ Mm))
        Mm = Mm @ (U @ Vh)
        return float(np.imag(Mm.conj().T @ H @ Mm)[0, 1])
    fd = (eps(mu + h) - eps(mu - h))/(2*h)
    return abs(fd - m["K"].sum())/max(abs(fd), 1e-30)


# ------------------------------------------------------------ GPU mu-scan ----
def pick_backend(device):
    if device != "cpu" and cp is not None:
        try:
            if cp.cuda.runtime.getDeviceCount() > 0:
                return cp, "gpu"
        except Exception:
            pass
    if device == "gpu":
        raise RuntimeError("--device gpu requested but CuPy/CUDA is unavailable")
    return np, "cpu"


def low_energy(H0, T, mus, xp, chunk):
    """Lowest positive BdG eigenvalue for each mu (batched eigvalsh; GPU if xp is cupy)."""
    H0x, Tx, half, out = xp.asarray(H0), xp.asarray(T), H0.shape[0]//2, []
    for i in range(0, len(mus), chunk):
        m = xp.asarray(np.asarray(mus[i:i + chunk], float))
        ev = xp.linalg.eigvalsh(H0x[None] - m[:, None, None]*Tx[None])
        out.append(ev[:, half])
    r = xp.concatenate(out)
    return cp.asnumpy(r) if xp is not np else r


def refine(H0, T, mu0, step, sign, lim, xp, chunk, stages=3, npts=11):
    for _ in range(stages):
        mus = np.clip(np.linspace(mu0 - step, mu0 + step, npts), *lim)
        E = low_energy(H0, T, mus, xp, chunk)
        mu0 = float(mus[int(np.argmin(sign*E))]); step = 1.5*(2*step/(npts - 1))
    return mu0


def locate_points(H0, T, L, p, mu_lo, mu_hi, u, xp, chunk):
    """Find a node (E_low minimum) and the next antinode (E_low maximum) of the eps(mu) oscillation."""
    alpha, Ez, Delta, t0 = p
    mid, h = 0.5*(mu_lo + mu_hi), 1e-3
    dk = (k_fermi(mid + h, alpha, Ez, t0) - k_fermi(mid - h, alpha, Ez, t0))/(2*h)
    if not np.isfinite(dk) or abs(dk) < 1e-6:
        return None
    P = math.pi/(L*abs(dk))                           # expected mu-period of eps ~ cos(kF L)
    for scale in (1.0, 2.5):
        W = min(4*P*scale, 0.95*(mu_hi - mu_lo))
        mus = np.linspace(mu_lo + u*(mu_hi - mu_lo - W), mu_lo + u*(mu_hi - mu_lo - W) + W, 81)
        E = low_energy(H0, T, mus, xp, chunk)
        mins = [i for i in range(1, 80) if E[i] < E[i - 1] and E[i] <= E[i + 1]]
        if len(mins) >= 2:
            break
    else:
        return None
    j = (len(mins) - 2)//2
    i0, i1 = mins[j], mins[j + 1]
    ia = i0 + int(np.argmax(E[i0:i1 + 1]))
    step, lim = float(mus[1] - mus[0]), (mu_lo, mu_hi)
    return {"node": refine(H0, T, mus[i0], step, +1, lim, xp, chunk),
            "anti": refine(H0, T, mus[ia], step, -1, lim, xp, chunk)}


# ------------------------------------------------------------ noise algebra --
def corr_matrix(n, xi):
    if xi == 0:
        return np.eye(n)
    if math.isinf(xi):
        return np.ones((n, n))
    i = np.arange(n)
    return np.exp(-np.abs(i[:, None] - i[None, :])/xi)


def fold(q):
    return abs(((q + math.pi) % (2*math.pi)) - math.pi)


def osc_q(K):
    """Dominant oscillation wavevector of K(x) (uniform part removed, Hann window)."""
    L = K.size; n = 16*L
    P = np.abs(np.fft.rfft((K - K.mean())*np.hanning(L), n=n))**2
    q = 2*np.pi*np.arange(P.size)/n
    m = q > 4*np.pi/L
    return float(q[m][np.argmax(P[m])])


def two_channel(N0, p0, q0, L, xi, Rs):
    """N(xi_n) = N0 [p0 u + (1-p0) lam]: uniform channel u=1^T R 1/L, oscillating channel lam(q0)."""
    if xi == 0:
        return N0
    if math.isinf(xi):
        return p0*L*N0
    e = math.exp(-1.0/xi)
    lam = (1 - e*e)/(1 - 2*e*np.cos(q0) + e*e)
    u = np.array([Rs[xi][:l, :l].sum()/l for l in np.asarray(L).astype(int)])
    return N0*(p0*u + (1 - p0)*lam)


def analyse_point(H0, T, L, p, mu, phase, xis, Rs, check):
    alpha, Ez, Delta, t0 = p
    if z2(alpha, Ez, Delta, t0, mu) != -1:
        return None
    kF = k_fermi(mu, alpha, Ez, t0)
    m = majorana(H0 - mu*T, L)
    K = m["K"]
    NX = [float(K @ Rs[x][:L, :L] @ K) for x in xis]
    if not (np.isfinite(kF) and NX[0] > 0 and m["xi"] > 0):
        return None
    return dict(phase=phase, L=L, alpha=alpha, Ez=Ez, Delta=Delta, t0=t0, mu=mu, E_low=m["E"],
                E_gap=bulk_gap(alpha, Ez, Delta, t0, mu), xi=m["xi"], d=m["d"], kF=kF,
                q0=osc_q(K), q2kF=fold(2*kF), N0=NX[0], p0=float(K.sum()**2/(L*NX[0])), NX=NX, K=K,
                fd_err=fd_uniform(H0, T, L, mu, m) if check else float("nan"))


def run_scan(args, xp, xis, Rs, rng):
    recs, done, tried, t_start = [], 0, 0, time.time()
    while done < args.n_configs and tried < 30*args.n_configs:
        tried += 1
        L = int(rng.integers(int(args.L_range[0]), int(args.L_range[1]) + 1))
        alpha, Ez, Delta, t0 = (rng.uniform(*r) for r in (args.alpha_range, args.Ez_range, args.Delta_range, args.t0_range))
        muc = math.sqrt(Ez**2 - Delta**2)             # k=0 gap closing; k=pi closing at 4 t0 -/+ muc
        mu_lo, mu_hi = 0.15*muc, min(0.9*muc, 4*t0 - Ez - 0.1)   # topological, lower band has a Fermi root
        if mu_hi - mu_lo < 0.3:
            continue
        H0, T = build_H0(L, alpha, Ez, Delta, t0)
        loc = locate_points(H0, T, L, (alpha, Ez, Delta, t0), mu_lo, mu_hi, rng.uniform(), xp, args.chunk)
        if loc is None:
            continue
        for ph, mu in loc.items():
            r = analyse_point(H0, T, L, (alpha, Ez, Delta, t0), mu, ph, xis, Rs, check=(done < 3 and ph == "node"))
            if r:
                recs.append(r)
        done += 1
        if xp is not np:
            cp.get_default_memory_pool().free_all_blocks()
        if done % 10 == 0 or done == args.n_configs:
            print(f"  scan {done}/{args.n_configs} configs ({tried} tried, {len(recs)} points, {time.time() - t_start:.0f}s)", flush=True)
    return recs


# ------------------------------------------------------------ law fitting ----
FEATS = {"Lx": lambda a: (a["L"] - 1)/a["xi"], "logL": lambda a: np.log(a["L"]), "logxi": lambda a: np.log(a["xi"]),
         "dx": lambda a: a["d"]/a["xi"], "xi": lambda a: a["xi"], "gap": lambda a: a["E_gap"]/a["Delta"],
         "loggap": lambda a: np.log(a["E_gap"]/a["Delta"]), "dez": lambda a: a["Delta"]/a["Ez"],
         "mfrac": lambda a: a["mu"]/np.sqrt(a["Ez"]**2 - a["Delta"]**2)}
FREE = ("Lx", "logL", "logxi"); THEORY = (-1.0, 1.0, -2.0)      # frozen exponents for (Lx, logL, logxi)
LADDER = [("constant", ()), ("E_gap/Delta", ("gap",)), ("(L-1)/xi", ("Lx",)), ("(L-1)/xi + log xi", ("Lx", "logxi")),
          ("d/xi + xi  [v21.4 law]", ("dx", "xi")), ("d/xi + log xi", ("dx", "logxi")),
          ("(L-1)/xi + log L + log xi [free]", FREE), ("free + log(E_gap/Delta)", FREE + ("loggap",)),
          ("free + Delta/Ez + mu/mu_c", FREE + ("dez", "mfrac")), ("THEORY frozen (-1,+1,-2)", "theory")]
HOLD_VARS = ("L", "alpha", "Ez", "Delta", "t0", "xi")
RES_VARS = HOLD_VARS + ("E_gap", "kF", "mu")


def arrays(recs):
    a = {k: np.array([r[k] for r in recs], float) for k in
         ("L", "alpha", "Ez", "Delta", "t0", "mu", "E_low", "E_gap", "xi", "d", "kF", "N0", "p0", "q0", "q2kF")}
    a["NX"] = np.array([r["NX"] for r in recs], float)
    a["y"] = np.log(a["N0"])
    return a


def theory_shape(a):
    return THEORY[1]*np.log(a["L"]) + THEORY[2]*np.log(a["xi"]) + THEORY[0]*(a["L"] - 1)/a["xi"]


def design(a, names):
    return np.column_stack([np.ones(len(a["L"]))] + [FEATS[n](a) for n in names])


def fit_predict(a, y, model, tr, te):
    if model == "theory":
        sh = theory_shape(a)
        return np.mean(y[tr] - sh[tr]) + sh[te]
    X = design(a, model)
    return X[te] @ np.linalg.lstsq(X[tr], y[tr], rcond=None)[0]


def metrics(y, p):
    return dict(rmse=float(np.sqrt(np.mean((y - p)**2))),
                r2=float(1 - np.sum((y - p)**2)/max(np.sum((y - y.mean())**2), 1e-30)),
                relerr=float(np.median(np.abs(np.exp(p - y) - 1))))


def cv_eval(a, y, model, rng, folds=5, reps=10):
    n, out = len(y), []
    for _ in range(reps):
        pred = np.empty(n)
        for te in np.array_split(rng.permutation(n), folds):
            pred[te] = fit_predict(a, y, model, np.setdiff1d(np.arange(n), te), te)
        out.append(metrics(y, pred))
    return {k: float(np.mean([o[k] for o in out])) for k in out[0]}


def holdout_eval(a, y, model, min_n=8):
    res = []
    for v in HOLD_VARS:                                # train low half -> test high half, and reverse
        lo = a[v] <= np.median(a[v]); hi = ~lo
        for tr, te in ((lo, hi), (hi, lo)):
            if tr.sum() >= min_n and te.sum() >= min_n:
                res.append(metrics(y[te], fit_predict(a, y, model, np.where(tr)[0], np.where(te)[0])))
    return res


def audit_phase(ph, recs, xis, Rs, rng):
    a = arrays(recs); y = a["y"]; n = len(y); sh = theory_shape(a); out = {"n": n}
    print(f"\n{'=' * 24} phase = {ph}  (n = {n}) {'=' * 24}")
    print(f"sanity: median d/(L-1-2xi) = {np.median(a['d']/(a['L'] - 1 - 2*a['xi'])):.3f} (expect ~1);  "
          f"xi in [{a['xi'].min():.2f},{a['xi'].max():.2f}];  (L-1)/xi in [{FEATS['Lx'](a).min():.1f},{FEATS['Lx'](a).max():.1f}]")
    # 1) model ladder: random CV + 12 region holdouts over (L, alpha, Ez, Delta, t0, xi)
    print("\n[H1] envelope law -- ladder (CV = 5-fold x10; holdout = train one half / test other half of each variable)")
    print(f"  {'model':<36s}{'CV relerr':>10s}{'CV R2':>9s}{'hold worst':>12s}{'hold med R2':>13s}")
    out["ladder"] = {}
    for name, model in LADDER:
        cv = cv_eval(a, y, model, rng); ho = holdout_eval(a, y, model)
        row = dict(cv_relerr=cv["relerr"], cv_r2=cv["r2"], hold_worst_relerr=max(h["relerr"] for h in ho) if ho else float("nan"),
                   hold_med_r2=float(np.median([h["r2"] for h in ho])) if ho else float("nan"))
        out["ladder"][name] = row
        print(f"  {name:<36s}{100*row['cv_relerr']:>9.2f}%{row['cv_r2']:>9.4f}{100*row['hold_worst_relerr']:>11.2f}%{row['hold_med_r2']:>13.4f}")
    # 2) coefficients: free fit + bootstrap vs frozen theory
    X = design(a, FREE); beta = np.linalg.lstsq(X, y, rcond=None)[0]
    bs = np.array([np.linalg.lstsq(X[i], y[i], rcond=None)[0] for i in (rng.integers(0, n, n) for _ in range(300))])
    lo, hi = np.percentile(bs, 2.5, axis=0), np.percentile(bs, 97.5, axis=0)
    raw = y - sh; A = float(raw.mean())
    out["free_fit"] = {"beta": beta.tolist(), "ci_lo": lo.tolist(), "ci_hi": hi.tolist()}
    out["theory"] = {"A": A, "kappa": math.exp(A), "resid_std": float(raw.std(ddof=1))}
    print("\n  free fit  log N0 = A + B (L-1)/xi + C1 log L + C2 log xi:")
    for nm, i, t in (("B ", 1, THEORY[0]), ("C1", 2, THEORY[1]), ("C2", 3, THEORY[2])):
        print(f"    {nm} = {beta[i]:+.3f}  95% CI [{lo[i]:+.3f},{hi[i]:+.3f}]   theory {t:+.1f}  {'inside' if lo[i] <= t <= hi[i] else 'OUTSIDE'}")
    print(f"  frozen theory: kappa = exp(A) = {math.exp(A):.4g};  residual std of log N0 = {raw.std(ddof=1):.4f}")
    res = raw - A
    out["resid_corr"] = {v: float(np.corrcoef(res, a[v])[0, 1]) for v in RES_VARS}
    print("  residual correlation with parameters (hidden dependence would show here): " +
          ", ".join(f"{v}={out['resid_corr'][v]:+.2f}" for v in RES_VARS))
    # 3) xi_noise scan (spectral two-channel law)
    p_bar = float(np.median(a["p0"])); out["p0_median"] = p_bar
    out["p0_quartiles"] = np.percentile(a["p0"], [25, 75]).tolist()
    print(f"\n[H2] noise-correlation scan.  p0: median {p_bar:.4g}, IQR [{out['p0_quartiles'][0]:.4g},{out['p0_quartiles'][1]:.4g}];  "
          f"median |q0 - 2kF| = {np.median(np.abs(a['q0'] - a['q2kF'])):.3f} rad")
    print(f"  {'xi_n':>6s}{'med N/N0':>10s} | two-channel median rel.err: {'(i) p0,q0':>10s}{'(ii) p0,2kF':>12s}{'(iii) pbar,2kF':>15s} | law CV err: {'theory':>7s}{'free':>7s} | free (B,C1,C2)")
    out["xi_scan"] = []
    for j, xi in enumerate(xis):
        Nx = a["NX"][:, j]; yx = np.log(Nx)
        preds = (two_channel(a["N0"], a["p0"], a["q0"], a["L"], xi, Rs), two_channel(a["N0"], a["p0"], a["q2kF"], a["L"], xi, Rs),
                 two_channel(a["N0"], np.full(n, p_bar), a["q2kF"], a["L"], xi, Rs))
        err = [float(np.median(np.abs(p/Nx - 1))) for p in preds]
        cvt, cvf = cv_eval(a, yx, "theory", rng)["relerr"], cv_eval(a, yx, FREE, rng)["relerr"]
        bf = np.linalg.lstsq(X, yx, rcond=None)[0][1:]
        out["xi_scan"].append(dict(xi=xi, S_med=float(np.median(Nx/a["N0"])), S_pred_iii=float(np.median(preds[2]/a["N0"])),
                                   err_i=err[0], err_ii=err[1], err_iii=err[2], cv_theory=cvt, cv_free=cvf, free_coef=bf.tolist()))
        print(f"  {xi:>6g}{np.median(Nx/a['N0']):>10.3f} | {'':>27s}{100*err[0]:>6.1f}%{100*err[1]:>10.1f}%{100*err[2]:>13.1f}% | {'':>12s}{100*cvt:>6.1f}%{100*cvf:>6.1f}% | "
              f"({bf[0]:+.2f},{bf[1]:+.2f},{bf[2]:+.2f})")
    out["finite_projection"] = finite_projection_audit(recs, xis, Rs)
    print_finite_projection(out["finite_projection"])
    fd = [r["fd_err"] for r in recs if np.isfinite(r["fd_err"])]
    if fd:
        out["fd_err"] = fd
        print(f"  FD check |sum K| vs d eps/d mu (uniform shift): rel.err = {', '.join(f'{x:.1e}' for x in fd)}")
    # 4) predeclared verdicts
    th = out["ladder"]["THEORY frozen (-1,+1,-2)"]
    v = {"H1 frozen-theory worst holdout err <= 10%": th["hold_worst_relerr"] <= 0.10,
         "H1 free-fit 95% CI contains (-1,+1,-2)": all(lo[i] <= t <= hi[i] for i, t in zip((1, 2, 3), THEORY)),
         "H1 log xi beats linear xi (CV err)": out["ladder"]["d/xi + log xi"]["cv_relerr"] < out["ladder"]["d/xi + xi  [v21.4 law]"]["cv_relerr"],
         "H1 residual |corr| < 0.3 with all parameters": max(abs(x) for x in out["resid_corr"].values()) < 0.3,
         "H2 two-channel (ii: measured p0, q0=2kF) err <= 15% at every xi_n": all(r["err_ii"] <= 0.15 for r in out["xi_scan"]),
         "H2' parameter-free (iii: median p0, q0=2kF) err <= 15% at every xi_n": all(r["err_iii"] <= 0.15 for r in out["xi_scan"]),
         "H2 finite 2kF channel median error <= 15% at every xi_n": all(r["median_channel_relerr_2kF"] <= 0.15 for r in out["finite_projection"])}
    out["verdict"] = {k: bool(x) for k, x in v.items()}
    print("\n  predeclared verdicts:")
    for k, x in v.items():
        print(f"    [{'PASS' if x else 'FAIL'}] {k}")
    return out


def pair_audit(recs, xis, Rs, rho, npairs, rng):
    """Exact two-wire N_geom = lambda_max([[n1,c],[c,n2]]) for random wire pairs; test the mismatch rule."""
    idx = rng.integers(0, len(recs), size=(npairs, 2)); idx = idx[idx[:, 0] != idx[:, 1]]
    out = {}
    for xi in xis:
        R, rows = Rs[xi], []
        for i, j in idx:
            K1, K2 = recs[i]["K"], recs[j]["K"]; l1, l2 = K1.size, K2.size
            rows.append((K1 @ R[:l1, :l1] @ K1, K2 @ R[:l2, :l2] @ K2, rho*(K1 @ R[:l1, :l2] @ K2)))
        n1, n2, c = np.array(rows).T
        lam = (n1 + n2)/2 + np.sqrt(((n1 - n2)/2)**2 + c**2)
        chk = max(abs(np.linalg.eigvalsh([[n1[k], c[k]], [c[k], n2[k]]])[-1]/lam[k] - 1) for k in range(min(200, len(c))))
        D = np.abs(np.log(n1/n2)); dl = np.tanh(D/2); re = c/np.sqrt(n1*n2); Pi = lam/np.maximum(n1, n2)
        pred = (1 + np.sqrt(dl**2 + (1 - dl**2)*re**2))/(1 + dl)
        out[xi] = dict(D=D, rho_eff=re, Pi=Pi, identity_err=float(np.max(np.abs(pred/Pi - 1))), eig_err=float(chk))
    return out


def print_pairs(ph, pairs, rho):
    edges = [0, 0.5, 1, 2, 4, np.inf]
    print(f"\n[H3] pair rule, phase = {ph}, rho_site = {rho}: penalty Pi = N_geom/max(n1,n2) by exponent mismatch Delta = |ln n1/n2|")
    print(f"  {'xi_n':>6s}{'med|rho_eff|':>13s} | " + " ".join(f"D in [{lo:g},{hi:g})".rjust(15) for lo, hi in zip(edges[:-1], edges[1:])) + " | identity err")
    for xi, d in pairs.items():
        cells = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (d["D"] >= lo) & (d["D"] < hi)
            cells.append((f"{np.median(d['Pi'][m]):.3f} (n={m.sum()})" if m.any() else "-").rjust(15))
        print(f"  {xi:>6g}{np.median(np.abs(d['rho_eff'])):>13.3f} | " + " ".join(cells) + f" | {d['identity_err']:.1e}")




# ------------------------------------------------ finite-chain H2 projection ----
def _orthonormal_columns(B: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """Stable real QR basis for the column space of B."""
    Q, R = np.linalg.qr(np.asarray(B, dtype=float), mode="reduced")
    if R.size == 0:
        return np.zeros((B.shape[0], 0), float)
    diag = np.abs(np.diag(R))
    rank = int(np.sum(diag > tol * max(1.0, float(diag.max()))))
    return Q[:, :rank]


def finite_channel_projection(K: np.ndarray, q: float, R: np.ndarray) -> dict:
    """Project the exact finite-chain K onto uniform + oscillatory q subspaces.

    The oscillatory subspace is span{cos(qx), sin(qx)}, explicitly orthogonalized
    against the uniform channel.  This gives a finite-chain, boundary-aware
    Rayleigh quotient for the oscillatory component rather than the infinite-chain
    Toeplitz eigenvalue used by the simple analytic two-channel model.
    """
    K = np.asarray(K, float)
    R = np.asarray(R, float)
    L = K.size
    x = np.arange(L, dtype=float)
    u = np.ones((L, 1), float) / math.sqrt(L)
    Q0 = _orthonormal_columns(u)
    Bq = np.column_stack((np.cos(q * x), np.sin(q * x)))
    if Q0.size:
        Bq = Bq - Q0 @ (Q0.T @ Bq)
    Qq = _orthonormal_columns(Bq)

    P_u_K = Q0 @ (Q0.T @ K) if Q0.size else np.zeros_like(K)
    P_q_K = Qq @ (Qq.T @ K) if Qq.size else np.zeros_like(K)
    K2 = P_u_K + P_q_K
    N0 = float(K @ K)
    Eu = float(P_u_K @ P_u_K)
    Eq = float(P_q_K @ P_q_K)
    Er = float(max(0.0, N0 - Eu - Eq))

    lam_u = float((Q0[:, 0] @ R @ Q0[:, 0]) if Q0.size else 0.0)
    if Eq > 1e-30:
        lam_q = float(P_q_K @ R @ P_q_K / Eq)
    else:
        lam_q = float('nan')
    N_proj = float(K2 @ R @ K2)
    N_exact = float(K @ R @ K)
    capture = float((Eu + Eq) / max(N0, 1e-30))
    proj_relerr = float(abs(N_proj - N_exact) / max(abs(N_exact), 1e-30))
    channel_pred = float(Eu * lam_u + Eq * lam_q) if np.isfinite(lam_q) else float('nan')
    channel_relerr = float(abs(channel_pred - N_exact) / max(abs(N_exact), 1e-30)) if np.isfinite(lam_q) else float('nan')
    return {
        "N0": N0, "Eu": Eu, "Eq": Eq, "Er": Er, "capture": capture,
        "lambda_uniform": lam_u, "lambda_osc_finite": lam_q,
        "N_projection": N_proj, "N_channel": channel_pred,
        "projection_relerr": proj_relerr, "channel_relerr": channel_relerr,
        "q": float(q), "K_residual_norm_fraction": float(math.sqrt(Er / max(N0, 1e-30))),
    }


def finite_projection_audit(recs, xis, Rs):
    """Audit finite-chain H2 projection using q=2 kF and q=FFT(K)."""
    a = arrays(recs)
    out = []
    for j, xi in enumerate(xis):
        R = Rs[xi]
        rows_2kf, rows_fft = [], []
        for r in recs:
            s1 = finite_channel_projection(r["K"], r["q2kF"], R[:r["L"], :r["L"]])
            s2 = finite_channel_projection(r["K"], r["q0"], R[:r["L"], :r["L"]])
            rows_2kf.append(s1); rows_fft.append(s2)
        def med(key, rows):
            vals = np.array([x[key] for x in rows], float)
            return float(np.nanmedian(vals)) if np.isfinite(vals).any() else float('nan')
        out.append({
            "xi": xi,
            "median_capture_2kF": med("capture", rows_2kf),
            "median_channel_relerr_2kF": med("channel_relerr", rows_2kf),
            "worst_channel_relerr_2kF": float(np.nanmax([x["channel_relerr"] for x in rows_2kf])),
            "median_projection_relerr_2kF": med("projection_relerr", rows_2kf),
            "median_lambda_osc_finite_2kF": med("lambda_osc_finite", rows_2kf),
            "median_capture_fft": med("capture", rows_fft),
            "median_channel_relerr_fft": med("channel_relerr", rows_fft),
            "worst_channel_relerr_fft": float(np.nanmax([x["channel_relerr"] for x in rows_fft])),
            "median_projection_relerr_fft": med("projection_relerr", rows_fft),
            "median_lambda_osc_finite_fft": med("lambda_osc_finite", rows_fft),
        })
    return out


def print_finite_projection(aud):
    print("\n[H2-finite] finite-chain channel projection audit (boundary-aware Rayleigh quotient)")
    print(f"  {'xi_n':>7s} | {'2kF capture':>11s}{'2kF err':>11s}{'2kF worst':>12s} | {'FFT capture':>11s}{'FFT err':>10s}{'FFT worst':>11s}")
    for r in aud:
        print(f"  {r['xi']:>7g} | {100*r['median_capture_2kF']:>10.1f}% {100*r['median_channel_relerr_2kF']:>10.1f}% {100*r['worst_channel_relerr_2kF']:>11.1f}% | "
              f"{100*r['median_capture_fft']:>10.1f}% {100*r['median_channel_relerr_fft']:>9.1f}% {100*r['worst_channel_relerr_fft']:>10.1f}%")


# --------------------------------------------------------------- outputs -----
def make_figures(outdir, aud, arrs, pairs, xis):
    cols = {"node": "tab:blue", "anti": "tab:orange"}
    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    for ph, a in arrs.items():
        x = FEATS["Lx"](a); yy = a["y"] - np.log(a["L"]) + 2*np.log(a["xi"])
        ax[0, 0].scatter(x, yy, s=14, alpha=0.6, c=cols[ph], label=ph)
        xl = np.linspace(x.min(), x.max(), 50); ax[0, 0].plot(xl, aud[ph]["theory"]["A"] - xl, c=cols[ph], ls="--")
        ax[0, 1].scatter(theory_shape(a) + aud[ph]["theory"]["A"], a["y"], s=14, alpha=0.6, c=cols[ph], label=ph)
    ax[0, 0].set(xlabel="(L-1)/xi", ylabel="log N0 - log L + 2 log xi", title="H1 collapse (dashed: slope -1, frozen)"); ax[0, 0].legend()
    lim = ax[0, 1].get_xlim(); ax[0, 1].plot(lim, lim, "k--", lw=1)
    ax[0, 1].set(xlabel="frozen theory + A", ylabel="log N0", title="H1 parameter-free prediction"); ax[0, 1].legend()
    xp_ = [0.3 if x == 0 else (300 if math.isinf(x) else x) for x in xis]
    for ph in arrs:
        ax[1, 0].plot(xp_, [r["S_med"] for r in aud[ph]["xi_scan"]], "o-", c=cols[ph], label=f"{ph} data")
        ax[1, 0].plot(xp_, [r["S_pred_iii"] for r in aud[ph]["xi_scan"]], "x--", c=cols[ph], label=f"{ph} two-channel (iii)")
    ax[1, 0].set(xscale="log", yscale="log", xlabel="xi_noise (0 -> 0.3, inf -> 300)", ylabel="median N(xi_n)/N(0)", title="H2 noise-correlation law"); ax[1, 0].legend(fontsize=8)
    xs_show = min(xis[1:-1], key=lambda x: abs(x - 5))
    for ph, pr in pairs.items():
        d = pr[xs_show]; ax[1, 1].scatter(d["D"], d["Pi"], s=6, alpha=0.35, c=cols[ph], label=ph)
    Dg = np.linspace(0, 6, 100); dl = np.tanh(Dg/2)
    for r in (0.2, 0.5, 0.8):
        ax[1, 1].plot(Dg, (1 + np.sqrt(dl**2 + (1 - dl**2)*r**2))/(1 + dl), "k:", lw=1)
    ax[1, 1].set(xlabel="Delta = |ln n1/n2|", ylabel="Pi = N_geom / max(n1,n2)", title=f"H3 pair penalty (xi_n={xs_show:g}; dotted: rho_eff=0.2,0.5,0.8)"); ax[1, 1].legend()
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig1_laws_v22_1.png"), dpi=200); plt.close(fig)
    fig, ax = plt.subplots(2, 3, figsize=(13, 7))
    for k, v in enumerate(HOLD_VARS):
        for ph, a in arrs.items():
            ax.flat[k].scatter(a[v], a["y"] - theory_shape(a) - aud[ph]["theory"]["A"], s=10, alpha=0.5, c=cols[ph], label=ph)
        ax.flat[k].axhline(0, c="k", lw=0.8); ax.flat[k].set(xlabel=v, ylabel="residual of frozen H1")
    ax.flat[0].legend(); fig.suptitle("Residual of the parameter-free envelope law vs every microscopic parameter")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig2_residuals_v22_1.png"), dpi=200); plt.close(fig)




def make_h3_phase_diagram(outdir, pairs, xis):
    """Scatter phase diagram of correlation penalty vs exponent mismatch and |rho_eff|."""
    finite = [x for x in xis if np.isfinite(x) and x > 0]
    if not finite:
        return
    xs = min(finite, key=lambda x: abs(x - 5.0))
    fig, ax = plt.subplots(figsize=(7.4, 5.5))
    for ph, pr in pairs.items():
        d = pr[xs]
        sc = ax.scatter(d["D"], np.abs(d["rho_eff"]), c=d["Pi"], s=18, alpha=0.6, label=ph)
    Dg = np.linspace(0, 6, 180)
    rg = np.linspace(0, 1, 120)
    DD, RR = np.meshgrid(Dg, rg)
    delta = np.tanh(DD / 2)
    P = (1 + np.sqrt(delta**2 + (1-delta**2)*RR**2))/(1 + delta)
    # Contours give a simple map of the exact algebraic identity.
    cs = ax.contour(DD, RR, P, levels=[1.01, 1.05, 1.10, 1.20, 1.40], linewidths=0.8)
    ax.clabel(cs, inline=True, fontsize=8, fmt="%.2f")
    ax.set_xlabel(r"$|\ln(n_1/n_2)|$")
    ax.set_ylabel(r"$|\rho_{\rm eff}|$")
    ax.set_title(f"H3 correlation-penalty phase diagram (xi_noise={xs:g})")
    ax.grid(True, ls=":", alpha=0.3)
    ax.legend()
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label(r"$\Pi=N_{\rm geom}/\max(n_1,n_2)$")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig3_h3_phase_diagram_v22_1.png"), dpi=220); plt.close(fig)


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items() if not str(k).startswith("_")}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (np.floating, np.integer)):
        return clean(o.item())
    if isinstance(o, float) and not math.isfinite(o):
        return str(o)
    return o


def law_card(aud, xis):
    n = aud.get("node")
    if not n:
        return
    f = n["free_fit"]["beta"]; t = n["theory"]; vd = n["verdict"]
    tag = lambda key: "PASS" if any(v for k, v in vd.items() if k.startswith(key)) and all(v for k, v in vd.items() if k.startswith(key)) else "FAIL"
    print("\n" + "=" * 78 + "\nLAW CARD (node phase; numbers from this run)\n" + "=" * 78)
    print(" (1) d eps = sum_x K(x) d mu(x),   K(x) = -Im <g1| tau_z(x) |g2>")
    print(" (2) Cbar = K R K^T ,   N(xi_n) = K^T R(xi_n) K ,   N_geom = lambda_max(Cbar)")
    print(f" (3) [H1 frozen exponents: {tag('H1 frozen')}] N(0) = kappa L xi^-2 exp[-(L-1)/xi],   kappa = {t['kappa']:.4g}  (resid std {t['resid_std']:.3f});  free fit exponents ({f[1]:+.2f},{f[2]:+.2f},{f[3]:+.2f}) vs (-1,+1,-2)")
    print(f" (4-6) [H2: {tag('H2 ')}, H2': {tag(chr(72)+'2'+chr(39))}]")
    print(" (4) N(xi_n)/N(0) = p0 u(xi_n,L) + (1-p0) lam(xi_n,q0),   u = 1^T R 1/L")
    print(" (5) lam = (1-e^{-2/xi_n}) / (1 - 2 e^{-1/xi_n} cos q0 + e^{-2/xi_n}),   q0 = 2 kF (mod 2 pi)")
    print(f" (6) N(inf) = (sum K)^2 = p0 L N(0) = (d eps/d mu)^2 ,   p0 median = {n['p0_median']:.3f}")
    if n.get("finite_projection"):
        fp = min(n["finite_projection"], key=lambda r: abs(r["xi"] - 5.0))
        print(f"     finite-chain 2kF projection: xi_n={fp['xi']:g}, capture={fp['median_capture_2kF']:.3f}, median channel err={100*fp['median_channel_relerr_2kF']:.2f}%")
    print(" (7) N_geom = nbar [1 + sqrt(delta^2 + (1-delta^2) rho_eff^2)],  nbar=(n1+n2)/2, rho_eff = c/sqrt(n1 n2)")
    print(" (8) delta = tanh(Delta/2),  Delta = ln(n1/n2) = ln(L1/xi1^2) - ln(L2/xi2^2) - [(L1-1)/xi1 - (L2-1)/xi2]")
    print(" (9) penalty Pi = N_geom/max(n1,n2) = [1+sqrt(delta^2+(1-delta^2) rho_eff^2)]/(1+|delta|)  -> 1 for |Delta| >> 1")
    print("(10) design rules: tune d eps/d mu -> 0 to suppress long-range noise; at fixed max(n_i), asymmetry reduces correlation amplification; keep E_gap as a protection constraint")
    if "anti" in aud:
        r = aud["anti"]["p0_median"]/max(n["p0_median"], 1e-30)
        print(f"     p0(anti)/p0(node) = {r:.3g}  ->  long-correlation noise N(inf) reduced by that factor at the eps-extremum")


def main():
    pr = lambda s: tuple(float(x) for x in s.split(","))
    ap = argparse.ArgumentParser(description="Majorana noise-kernel law v22.1")
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "gpu")); ap.add_argument("--outdir", default="outputs_v22_1")
    ap.add_argument("--n-configs", type=int, default=200); ap.add_argument("--seed", type=int, default=2028)
    ap.add_argument("--L-range", type=pr, default=(40, 100)); ap.add_argument("--alpha-range", type=pr, default=(0.10, 0.30))
    ap.add_argument("--Ez-range", type=pr, default=(1.8, 3.0)); ap.add_argument("--Delta-range", type=pr, default=(0.7, 1.3))
    ap.add_argument("--t0-range", type=pr, default=(0.8, 1.4))
    ap.add_argument("--xi-noise", default="0,1,2,5,10,20,50,inf"); ap.add_argument("--rho", type=float, default=0.6)
    ap.add_argument("--min-Lx", type=float, default=4.0, help="keep wires with (L-1)/xi >= this (asymptotic regime)")
    ap.add_argument("--pairs", type=int, default=3000); ap.add_argument("--chunk", type=int, default=0)
    args, _ = ap.parse_known_args()
    os.makedirs(args.outdir, exist_ok=True)
    xp, dev = pick_backend(args.device)
    args.chunk = args.chunk or (128 if dev == "gpu" else 16)
    rng = np.random.default_rng(args.seed)
    xis = sorted(set(float(s) for s in args.xi_noise.split(",")) | {0.0, math.inf})
    Rs = {x: corr_matrix(int(args.L_range[1]), x) for x in xis}
    print(f"Majorana noise-kernel law v22.1 | device = {dev}" + (f" ({cp.cuda.runtime.getDeviceProperties(0)['name'].decode()})" if dev == "gpu" else ""))
    t0 = time.time()
    recs = run_scan(args, xp, xis, Rs, rng)
    n_all = len(recs)
    recs = [r for r in recs if (r["L"] - 1)/r["xi"] >= args.min_Lx]
    print(f"scan done in {time.time() - t0:.0f}s: {n_all} points, {len(recs)} kept with (L-1)/xi >= {args.min_Lx:g}")
    by = {ph: [r for r in recs if r["phase"] == ph] for ph in ("node", "anti")}
    aud, arrs, pairs = {}, {}, {}
    for ph, rr in by.items():
        if len(rr) < 25:
            print(f"phase {ph}: only {len(rr)} points, skipped (increase --n-configs)")
            continue
        aud[ph] = audit_phase(ph, rr, xis, Rs, rng); arrs[ph] = arrays(rr)
        pairs[ph] = pair_audit(rr, xis, Rs, args.rho, args.pairs, rng); print_pairs(ph, pairs[ph], args.rho)
        aud[ph]["pairs"] = {str(x): dict(med_abs_rho_eff=float(np.median(np.abs(d["rho_eff"]))), identity_err=d["identity_err"],
                                         Pi_median_by_Delta={f"{lo:g}-{hi:g}": float(np.median(d["Pi"][(d["D"] >= lo) & (d["D"] < hi)]))
                                                             for lo, hi in ((0, .5), (.5, 1), (1, 2), (2, 4), (4, 1e9)) if ((d["D"] >= lo) & (d["D"] < hi)).any()})
                            for x, d in pairs[ph].items()}
    if not aud:
        raise SystemExit("no phase had enough points")
    make_figures(args.outdir, aud, arrs, pairs, xis)
    make_h3_phase_diagram(args.outdir, pairs, xis)
    law_card(aud, xis)
    flat = [{k: v for k, v in r.items() if k not in ("K", "NX")} | {f"N_xi{x:g}": r["NX"][j] for j, x in enumerate(xis)} for r in recs]
    with open(os.path.join(args.outdir, "points_v22_1.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(flat[0].keys())); w.writeheader(); w.writerows(flat)
    np.savez_compressed(os.path.join(args.outdir, "kernels_v22_1.npz"), **{f"K_{i}_{r['phase']}": r["K"] for i, r in enumerate(recs)})
    with open(os.path.join(args.outdir, "results_v22_1.json"), "w") as fh:
        json.dump(clean({"version": "v22.1", "args": vars(args), "device": dev, "audit": aud}), fh, indent=1)

    node = aud.get("node", {})
    with open(os.path.join(args.outdir, "law_card_v22_1.md"), "w", encoding="utf-8") as fh:
        fh.write("# Majorana noise-kernel law v22.1\n\n")
        fh.write("## Core equations\n\n")
        fh.write("1. $\\delta\\epsilon=K\\,\\delta\\mu$\n2. $N(\\xi_n)=K^T R(\\xi_n)K$\n3. $N(\\infty)=(\\sum_x K_x)^2=(d\\epsilon/d\\mu)^2$\n4. $N_{\\rm geom}=\\lambda_{\\max}(\\bar C)$\n\n")
        fh.write("## H1\n\nThe finite-parameter scan is used as a falsification test for a universal envelope law; no universal H1 claim is made unless the frozen-theory and residual tests pass.\n\n")
        fh.write("## H2\n\nThe primary mechanism is channel selection by the noise correlation length. The finite-chain projection audit compares a boundary-aware 2kF Rayleigh quotient against the analytic Toeplitz approximation.\n\n")
        fh.write("## H3\n\nThe exact two-wire covariance eigenvalue is used to expose correlation amplification and its suppression by exponent mismatch.\n\n")
        fh.write("## Engineering rule\n\nTune the operating point toward $d\\epsilon/d\\mu=0$ when long-wavelength noise dominates; for fixed single-wire exposures, geometric asymmetry reduces the amplification caused by correlated cross-wire noise.\n")
    print(f"\noutputs -> {args.outdir}/  (results_v22_1.json, points_v22_1.csv, kernels_v22_1.npz, fig1_laws_v22_1.png, fig2_residuals_v22_1.png, fig3_h3_phase_diagram_v22_1.png)")


if __name__ == "__main__":
    main()
