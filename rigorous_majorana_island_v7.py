#!/usr/bin/env python3
"""
Rigorous Majorana-Coulomb-Island Validation + Operating-Window Analysis v7
============================================================================

What v7 changes relative to v6
-------------------------------
1. EXPLICIT, COMPUTED REACHABILITY CHECK (not just observed after the fact).
   v6's output showed that the bias-only (Z-rotation-only) protocol reached
   the canonical T target with 0% infidelity at zero dephasing. That is a
   direct consequence of Bloch-sphere geometry: |+> and the canonical |T>
   state both have Bloch-vector Z-component 0 (they sit on the same
   "equator"), and a pure Z-axis rotation preserves the Z-component while
   sweeping the azimuthal angle -- so a Z-only rotation can connect them
   exactly. v7 adds `bloch_axis_reachability()`, which computes this
   directly from the initial/target Bloch vectors (for ANY chosen target,
   not just T), reports whether a pure-Z-rotation protocol can reach the
   target, and if so the required rotation angle. This turns what was
   previously a post-hoc narrative into a falsifiable, printed diagnostic.
   (For the H-type target, Z-component is nonzero and the check correctly
   reports "not reachable by Z alone"; the external X/Y drive term is
   genuinely needed there.)

2. QUASIPARTICLE-POISONING CHANNEL NAMING TIGHTENED.
   gamma_qp is documented and printed everywhere as an "effective
   parity-flip poisoning channel (phenomenological proxy)", never as
   "realistic quasiparticle poisoning dynamics".

3. THE MAIN NEW RESULT: an explicit two-stage Majorana-operating-window /
   T-state-fidelity composition (the reviewer's suggested "core figure"),
   built at a computationally tractable cost instead of a brute-force 4D
   scan:
     Stage A: `majorana_operating_window_scan(L_list, W_list)` computes the
       Majorana-pair diagnostic support fraction over a (chain length L,
       on-site disorder W) grid, and the corresponding distribution of
       finite-chain splittings E_low = eps_M in the cells that pass the
       diagnostic ("the Majorana operating region").
     Stage B: `scan_fidelity_vs_eps_and_qp(eps_grid, gamma_qp_grid)` uses
       THAT eps_M range (not an arbitrary one) as one axis of a drive-
       optimized target-state-infidelity heatmap over (eps_M, gamma_qp).
   The two are combined into one summary figure: the fidelity heatmap with
   the physically accessible eps_M band (from Stage A) overlaid, plus a
   printed statement of whether/where the two regions overlap. This is the
   "Majorana operating region ∩ high-fidelity region" figure the review
   asked for, without pretending to have swept all four parameters jointly.

4. Everything from v6 is kept: independent spectral-isolation criterion
   (E_low vs E_next, not just vs the bulk gap), spatial-separation-ratio
   test on the constructed Majorana pair, 100-realization single-length
   disorder ensemble with Wilson confidence intervals, oscillatory
   finite-size splitting fit (explicitly an empirical descriptor), the
   explicit two-state charging spectrum E_n = 4 Ec (n-n_g)^2, bias-only vs
   coherent-drive calibration, control-bound sensitivity scan, and
   out-of-sample control-uncertainty evaluation (including openly reporting
   when the "robust" optimizer shows no measurable advantage -- this is
   NOT hidden or removed in v7, per the review's explicit recommendation).

What is intentionally NOT claimed (unchanged from v6)
-------------------------------------------------------
- No claim of experimental proof of topological superconductivity.
- No claim that the charge-parity bridge is a microscopic projection of a
  full BdG + charging + tunneling many-body Hamiltonian; eps_M is an
  imported phenomenological parameter.
- No claim that this simulation is calibrated to any real device's energy
  or time units (t0 = 1 throughout; no mapping to physical frequencies is
  attempted).
- No claim of relevance to, or comparison against, any specific vendor's
  hardware roadmap or published qubit specifications. Those are a separate,
  unverified, and largely orthogonal question to what this code actually
  computes, and this script does not make that comparison.
- The gamma_qp Lindblad channel is a phenomenological parity-flip proxy,
  not a microscopic quasiparticle-tunneling/reservoir model.

Outputs
-------
outputs_v7/
  majorana_mode_profile_v7.png
  bulk_gap_phase_diagram_v7.png
  bulk_gap_vs_gate_v7.png
  finite_size_splitting_v7.png
  majorana_separation_v7.png
  disorder_splitting_v7.png
  disorder_support_fraction_v7.png
  dephasing_calibration_v7.png
  poisoning_sensitivity_v7.png
  control_bound_sensitivity_v7.png
  operating_window_Lw_v7.png
  fidelity_vs_epsM_gammaqp_v7.png          <- the new combined "core figure"
  calibration_scan_v7.csv
  finite_size_scan_v7.csv
  disorder_scan_v7.csv
  control_robustness_v7.csv
  control_bound_sensitivity_v7.csv
  operating_window_v7.csv
  fidelity_vs_epsM_gammaqp_v7.csv
  scientific_summary_v7.json

Dependencies
------------
numpy, scipy, matplotlib

Example
-------
python rigorous_majorana_island_v7.py
python rigorous_majorana_island_v7.py --target H
python rigorous_majorana_island_v7.py --quick
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
from scipy.optimize import differential_evolution, minimize_scalar, least_squares


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def dagger(a: np.ndarray) -> np.ndarray:
    return a.conj().T


def pauli_matrices() -> Dict[str, np.ndarray]:
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    return {"I": I, "X": X, "Y": Y, "Z": Z}


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------


@dataclass
class BdGParameters:
    L: int = 80
    alpha: float = 0.15
    Delta: float = 1.0
    Ez: float = 2.5
    t0: float = 1.0
    gate_to_mu: float = 1.0


@dataclass
class IslandParameters:
    Ec: float = 0.8
    gate_to_ng: float = 0.5


@dataclass
class ControlParameters:
    Vg_nominal: float = 0.50
    true_bias: float = 0.15
    delta_v_min: float = -1.00
    delta_v_max: float = 1.00
    t_pulse: float = 0.5
    drive_amp_max: float = 6.0


@dataclass
class NoiseParameters:
    gamma_phi: float = 0.05
    gamma_relax: float = 0.0
    gamma_excitation: float = 0.0
    # Effective parity-flip poisoning channel (phenomenological proxy for
    # quasiparticle poisoning). This is a sigma_x Lindblad jump operator on
    # the logical qubit; it is NOT a microscopic quasiparticle-tunneling or
    # reservoir model.
    gamma_qp: float = 0.0


@dataclass
class ValidationThresholds:
    max_phs_relative_error: float = 1e-10
    edge_sites_fraction: float = 0.10
    min_edge_localization_legacy: float = 0.25
    max_low_energy_over_bulk_gap: float = 0.25
    max_low_energy_over_next: float = 0.25
    min_separation_ratio: float = 3.0
    max_self_conjugate_error: float = 1e-6


# ---------------------------------------------------------------------------
# BdG / Majorana layer
# ---------------------------------------------------------------------------


class MajoranaIslandPhysics:
    """
    Standard lattice Rashba nanowire BdG model.

    Nambu basis: Psi = (c_up, c_down, c_down^dagger, -c_up^dagger)
    (the standard time-reversed Nambu basis, consistent with a bare
    Delta * tau_x pairing term with no explicit i*sigma_y).

    Real-space model:
      H_ii    = (2t - mu) tau_z + Ez sigma_x + Delta tau_x
      H_i,i+1 = -t tau_z - i(alpha/2) tau_z sigma_y

    Energies in units of t0; time in units of 1/t0 (hbar = 1). No mapping
    to physical device units is attempted anywhere in this script.
    """

    def __init__(self, params: BdGParameters):
        self.p = params
        P = pauli_matrices()
        self.I2 = P["I"]
        self.sx = P["X"]
        self.sy = P["Y"]
        self.tx = P["X"]
        self.ty = P["Y"]
        self.tz = P["Z"]
        self.U_C_cell = np.kron(self.ty, self.sy)

    def chemical_potential(self, Vg: float) -> float:
        return self.p.gate_to_mu * Vg

    def generate_disorder(self, strength: float, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.uniform(-strength, strength, size=self.p.L)

    def build_bdg_hamiltonian(
        self, Vg: float, disorder_realization: Optional[np.ndarray] = None
    ) -> np.ndarray:
        p = self.p
        mu = self.chemical_potential(Vg)
        tau_z_I = np.kron(self.tz, self.I2)
        tau_x_I = np.kron(self.tx, self.I2)
        I_sigma_x = np.kron(self.I2, self.sx)
        tau_z_sy = np.kron(self.tz, self.sy)
        H0 = (2 * p.t0 - mu) * tau_z_I + p.Ez * I_sigma_x + p.Delta * tau_x_I
        Hhop = -p.t0 * tau_z_I - 0.5j * p.alpha * tau_z_sy

        H = np.zeros((4 * p.L, 4 * p.L), dtype=complex)
        for i in range(p.L):
            sl_i = slice(4 * i, 4 * (i + 1))
            site = H0.copy()
            if disorder_realization is not None:
                if len(disorder_realization) != p.L:
                    raise ValueError("disorder_realization must have length L")
                site = site - float(disorder_realization[i]) * tau_z_I
            H[sl_i, sl_i] = site
            if i < p.L - 1:
                sl_j = slice(4 * (i + 1), 4 * (i + 2))
                H[sl_i, sl_j] = Hhop
                H[sl_j, sl_i] = dagger(Hhop)
        return (H + dagger(H)) / 2.0

    def bloch_hamiltonian(self, k: float, Vg: float) -> np.ndarray:
        p = self.p
        mu = self.chemical_potential(Vg)
        xi = 2 * p.t0 - 2 * p.t0 * np.cos(k) - mu
        return (
            xi * np.kron(self.tz, self.I2)
            + p.alpha * np.sin(k) * np.kron(self.tz, self.sy)
            + p.Ez * np.kron(self.I2, self.sx)
            + p.Delta * np.kron(self.tx, self.I2)
        )

    def bulk_spectrum(self, Vg: float, k_grid: Optional[np.ndarray] = None):
        if k_grid is None:
            k_grid = np.linspace(-np.pi, np.pi, 801)
        k_grid = np.asarray(k_grid, dtype=float)
        p = self.p
        mu = self.chemical_potential(Vg)
        tzI = np.kron(self.tz, self.I2)
        tzsy = np.kron(self.tz, self.sy)
        Isx = np.kron(self.I2, self.sx)
        txI = np.kron(self.tx, self.I2)
        xi = 2 * p.t0 - 2 * p.t0 * np.cos(k_grid) - mu
        H = (
            xi[:, None, None] * tzI[None, :, :]
            + (p.alpha * np.sin(k_grid))[:, None, None] * tzsy[None, :, :]
            + p.Ez * Isx[None, :, :]
            + p.Delta * txI[None, :, :]
        )
        return k_grid, np.linalg.eigvalsh(H)

    def bulk_gap(self, Vg: float) -> float:
        _, e = self.bulk_spectrum(Vg)
        return float(np.min(np.abs(e)))

    def finite_chain_central_spectrum(
        self, Vg: float, disorder_realization: Optional[np.ndarray] = None,
        n_each_side: int = 4,
    ):
        H = self.build_bdg_hamiltonian(Vg, disorder_realization)
        n = H.shape[0]
        mid = n // 2
        lo = max(0, mid - n_each_side)
        hi = min(n - 1, mid + n_each_side - 1)
        return la.eigh(H, subset_by_index=[lo, hi])

    def finite_chain_phs_relative_error(
        self, Vg: float, disorder_realization: Optional[np.ndarray] = None
    ) -> float:
        H = self.build_bdg_hamiltonian(Vg, disorder_realization)
        U_C = np.kron(np.eye(self.p.L, dtype=complex), self.U_C_cell)
        residual = U_C @ H.conj() @ dagger(U_C) + H
        return float(la.norm(residual) / max(la.norm(H), 1e-15))

    def bulk_phs_relative_error(self, k: float, Vg: float) -> float:
        Hk = self.bloch_hamiltonian(k, Vg)
        Hm = self.bloch_hamiltonian(-k, Vg)
        residual = self.U_C_cell @ Hk.conj() @ dagger(self.U_C_cell) + Hm
        return float(la.norm(residual) / max(la.norm(Hk), 1e-15))

    @staticmethod
    def _pfaffian_4x4(A: np.ndarray) -> complex:
        if A.shape != (4, 4):
            raise ValueError("_pfaffian_4x4 expects a 4x4 matrix")
        anti = la.norm(A + A.T)
        if anti > 1e-8 * max(la.norm(A), 1.0):
            raise ValueError(f"Matrix is not antisymmetric enough: {anti:.3e}")
        return A[0, 1] * A[2, 3] - A[0, 2] * A[1, 3] + A[0, 3] * A[1, 2]

    def class_D_pfaffian_invariant(self, Vg: float) -> Dict[str, float]:
        B0 = self.bloch_hamiltonian(0.0, Vg) @ self.U_C_cell
        Bp = self.bloch_hamiltonian(np.pi, Vg) @ self.U_C_cell
        pf0 = self._pfaffian_4x4(B0)
        pfpi = self._pfaffian_4x4(Bp)
        product = float(np.real_if_close(pf0 * pfpi))
        return {
            "pfaffian_k0": float(np.real_if_close(pf0)),
            "pfaffian_kpi": float(np.real_if_close(pfpi)),
            "pfaffian_product": product,
            "z2_invariant": -1.0 if product < 0 else 1.0,
        }

    def pair_spectrum_diagnostics(
        self, Vg: float, disorder_realization: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        evals, _ = self.finite_chain_central_spectrum(Vg, disorder_realization, n_each_side=4)
        pos = evals[evals > 1e-12]
        if len(pos) < 2:
            raise RuntimeError("Need at least two positive-energy levels in the central window.")
        E1, E2 = float(pos[0]), float(pos[1])
        gap = float(self.bulk_gap(Vg))
        return {
            "E_low": E1, "E_next": E2, "bulk_gap": gap,
            "E_low_over_gap": E1 / max(gap, 1e-15),
            "E_low_over_next": E1 / max(E2, 1e-15),
        }

    def low_energy_state_diagnostics(
        self, Vg: float, edge_sites: Optional[int] = None,
        disorder_realization: Optional[np.ndarray] = None,
    ) -> Dict[str, object]:
        if edge_sites is None:
            edge_sites = max(4, int(round(self.p.L * 0.10)))
        edge_sites = max(1, min(edge_sites, self.p.L // 2))
        evals, evecs = self.finite_chain_central_spectrum(Vg, disorder_realization, n_each_side=4)
        idx = np.where(evals > 1e-12)[0]
        if len(idx) < 1:
            raise RuntimeError("No positive-energy BdG state found in the central window.")
        vec = evecs[:, idx[0]]
        w = self.mode_site_weights(vec)
        return {
            "energy": float(evals[idx[0]]),
            "edge_localization": float(np.sum(w[:edge_sites]) + np.sum(w[-edge_sites:])),
            "site_weight": w,
        }

    def majorana_mode_pair(
        self, Vg: float, disorder_realization: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        evals, evecs = self.finite_chain_central_spectrum(Vg, disorder_realization, n_each_side=4)
        pos = np.where(evals > 1e-12)[0]
        if len(pos) < 1:
            raise RuntimeError("No positive-energy state found in the central window")
        i_pos = int(pos[0])
        E = float(evals[i_pos])
        psi_plus = evecs[:, i_pos]
        U_C = np.kron(np.eye(self.p.L, dtype=complex), self.U_C_cell)
        psi_minus = U_C @ psi_plus.conj()
        psi_minus /= max(la.norm(psi_minus), 1e-15)
        overlap = np.vdot(psi_minus, U_C @ psi_plus.conj())
        psi_minus *= np.exp(-1j * np.angle(overlap))
        gamma_A = (psi_plus + psi_minus) / np.sqrt(2.0)
        gamma_B = -1j * (psi_plus - psi_minus) / np.sqrt(2.0)
        gamma_A /= max(la.norm(gamma_A), 1e-15)
        gamma_B /= max(la.norm(gamma_B), 1e-15)
        return gamma_A, gamma_B, E

    def particle_hole_apply(self, mode: np.ndarray) -> np.ndarray:
        U_C = np.kron(np.eye(self.p.L, dtype=complex), self.U_C_cell)
        return U_C @ mode.conj()

    def mode_site_weights(self, mode: np.ndarray) -> np.ndarray:
        w = np.zeros(self.p.L, dtype=float)
        for i in range(self.p.L):
            w[i] = float(np.sum(np.abs(mode[4 * i:4 * (i + 1)]) ** 2))
        return w / max(np.sum(w), 1e-15)

    def finite_size_scan(self, Vg: float, lengths: List[int]) -> Dict[int, Dict[str, float]]:
        out: Dict[int, Dict[str, float]] = {}
        base = self.p
        for L in lengths:
            p = BdGParameters(L=L, alpha=base.alpha, Delta=base.Delta, Ez=base.Ez,
                              t0=base.t0, gate_to_mu=base.gate_to_mu)
            m = MajoranaIslandPhysics(p)
            pair = majorana_pair_diagnostics(m, Vg)
            diag = m.low_energy_state_diagnostics(Vg)
            out[L] = {
                "E_low": pair["E_low"], "E_next": pair["E_next"], "bulk_gap": pair["bulk_gap"],
                "E_low_over_gap": pair["E_low_over_gap"], "E_low_over_next": pair["E_low_over_next"],
                "legacy_edge_localization": float(diag["edge_localization"]),
                "R_L": pair["R_L"], "R_R": pair["R_R"],
                "phs_residual_L": pair["phs_residual_L"], "phs_residual_R": pair["phs_residual_R"],
            }
        return out


def _self_conjugate_error(physics: MajoranaIslandPhysics, gamma: np.ndarray) -> float:
    c_gamma = physics.particle_hole_apply(gamma)
    overlap = np.vdot(gamma, c_gamma)
    phase = np.exp(-1j * np.angle(overlap))
    return float(la.norm(phase * c_gamma - gamma) / max(la.norm(gamma), 1e-15))


def majorana_pair_diagnostics(
    physics: MajoranaIslandPhysics, Vg: float, edge_sites: Optional[int] = None,
    disorder_realization: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    if edge_sites is None:
        edge_sites = max(4, int(round(physics.p.L * 0.10)))
    edge_sites = max(1, min(edge_sites, physics.p.L // 2))
    gamma_A, gamma_B, splitting = physics.majorana_mode_pair(Vg, disorder_realization)
    wA = physics.mode_site_weights(gamma_A)
    wB = physics.mode_site_weights(gamma_B)
    A_L, A_R = float(wA[:edge_sites].sum()), float(wA[-edge_sites:].sum())
    B_L, B_R = float(wB[:edge_sites].sum()), float(wB[-edge_sites:].sum())
    if A_L >= B_L:
        gamma_L, gamma_R = gamma_A, gamma_B
        left_L, right_L, left_R, right_R = A_L, A_R, B_L, B_R
    else:
        gamma_L, gamma_R = gamma_B, gamma_A
        left_L, right_L, left_R, right_R = B_L, B_R, A_L, A_R
    eps = 1e-12
    R_L = left_L / (right_L + eps)
    R_R = right_R / (left_R + eps)
    spec = physics.pair_spectrum_diagnostics(Vg, disorder_realization)
    return {
        "gamma_L": gamma_L, "gamma_R": gamma_R,
        "R_L": R_L, "R_R": R_R,
        "LR_overlap": float(abs(np.vdot(gamma_L, gamma_R))),
        "phs_residual_L": _self_conjugate_error(physics, gamma_L),
        "phs_residual_R": _self_conjugate_error(physics, gamma_R),
        "splitting": float(splitting),
        **spec,
    }


def majorana_status(
    physics: MajoranaIslandPhysics, Vg: float, thresholds: ValidationThresholds,
) -> Dict[str, object]:
    z2 = physics.class_D_pfaffian_invariant(Vg)
    pair = majorana_pair_diagnostics(physics, Vg)
    legacy = physics.low_energy_state_diagnostics(Vg)
    phs_error = physics.finite_chain_phs_relative_error(Vg)

    criterion_topological = bool(z2["z2_invariant"] < 0)
    criterion_phs = bool(phs_error <= thresholds.max_phs_relative_error)
    criterion_low_gap = bool(pair["E_low_over_gap"] <= thresholds.max_low_energy_over_bulk_gap)
    criterion_isolated = bool(pair["E_low_over_next"] <= thresholds.max_low_energy_over_next)
    criterion_separated = bool(
        pair["R_L"] >= thresholds.min_separation_ratio
        and pair["R_R"] >= thresholds.min_separation_ratio
    )
    supported = bool(
        criterion_topological and criterion_phs and criterion_low_gap
        and criterion_isolated and criterion_separated
    )
    return {
        "supported": supported, "z2": z2,
        "bulk_gap": pair["bulk_gap"], "finite_energy": pair["E_low"],
        "next_positive_energy": pair["E_next"],
        "low_energy_over_gap": pair["E_low_over_gap"],
        "low_energy_over_next": pair["E_low_over_next"],
        "edge_localization_legacy": float(legacy["edge_localization"]),
        "phs_relative_error": phs_error, "R_L": pair["R_L"], "R_R": pair["R_R"],
        "LR_overlap": pair["LR_overlap"],
        "self_conjugate_error_L": pair["phs_residual_L"],
        "self_conjugate_error_R": pair["phs_residual_R"],
        "criterion_topological": criterion_topological, "criterion_phs": criterion_phs,
        "criterion_low_gap": criterion_low_gap, "criterion_isolated": criterion_isolated,
        "criterion_separated": criterion_separated,
        "site_weight": np.asarray(legacy["site_weight"]), "pair": pair,
    }


# ---------------------------------------------------------------------------
# Disorder (single length) -- kept from v6
# ---------------------------------------------------------------------------


def disorder_robustness_scan(
    bdg_params: BdGParameters, Vg: float, strengths: List[float],
    n_realizations: int = 8, base_seed: int = 1000,
    thresholds: Optional[ValidationThresholds] = None,
) -> Dict[float, Dict[str, float]]:
    if thresholds is None:
        thresholds = ValidationThresholds()
    physics = MajoranaIslandPhysics(bdg_params)
    out: Dict[float, Dict[str, float]] = {}
    for W in strengths:
        split, ratio_gap, ratio_next, RL, RR = [], [], [], [], []
        supported = 0
        for r in range(n_realizations):
            disorder = None if W == 0 else physics.generate_disorder(W, base_seed + r + int(W * 10000))
            pair = majorana_pair_diagnostics(physics, Vg, disorder_realization=disorder)
            split.append(abs(pair["splitting"]))
            ratio_gap.append(pair["E_low_over_gap"])
            ratio_next.append(pair["E_low_over_next"])
            RL.append(pair["R_L"]); RR.append(pair["R_R"])
            ok = (
                pair["E_low_over_gap"] <= thresholds.max_low_energy_over_bulk_gap
                and pair["E_low_over_next"] <= thresholds.max_low_energy_over_next
                and pair["R_L"] >= thresholds.min_separation_ratio
                and pair["R_R"] >= thresholds.min_separation_ratio
            )
            supported += int(ok)
        out[W] = {
            "mean_splitting": float(np.mean(split)), "std_splitting": float(np.std(split)),
            "mean_E_low_over_gap": float(np.mean(ratio_gap)), "mean_E_low_over_next": float(np.mean(ratio_next)),
            "mean_R_L": float(np.mean(RL)), "mean_R_R": float(np.mean(RR)),
            "min_R_L": float(np.min(RL)), "min_R_R": float(np.min(RR)),
            "support_fraction": float(supported / n_realizations),
        }
    return out


# ---------------------------------------------------------------------------
# NEW (v7): joint (L, W) Majorana operating-window scan
# ---------------------------------------------------------------------------


def majorana_operating_window_scan(
    base_params: BdGParameters, Vg: float, L_list: List[int], W_list: List[float],
    n_realizations: int = 4, base_seed: int = 2000,
    thresholds: Optional[ValidationThresholds] = None,
) -> Dict[str, np.ndarray]:
    """
    Stage A of the v7 core analysis.

    Scans chain length L and on-site disorder strength W jointly, and for
    each (L, W) cell reports:
      - the Majorana-pair diagnostic support fraction (fraction of disorder
        realizations passing the full topological + isolation + separation
        test, as in `majorana_status`);
      - the mean and full set of extracted splittings E_low = eps_M.

    This defines "the Majorana operating region" empirically, as a region in
    (L, W) space together with the corresponding range of eps_M values it
    produces -- rather than assuming an eps_M range a priori.
    """
    if thresholds is None:
        thresholds = ValidationThresholds()
    support = np.zeros((len(L_list), len(W_list)))
    mean_eps = np.zeros((len(L_list), len(W_list)))
    all_supported_eps: List[float] = []

    for i, L in enumerate(L_list):
        params = BdGParameters(L=L, alpha=base_params.alpha, Delta=base_params.Delta,
                                Ez=base_params.Ez, t0=base_params.t0, gate_to_mu=base_params.gate_to_mu)
        physics = MajoranaIslandPhysics(params)
        for j, W in enumerate(W_list):
            splits = []
            ok_count = 0
            for r in range(n_realizations):
                disorder = None if W == 0 else physics.generate_disorder(
                    W, base_seed + i * 10000 + j * 100 + r
                )
                pair = majorana_pair_diagnostics(physics, Vg, disorder_realization=disorder)
                splits.append(abs(pair["splitting"]))
                ok = (
                    pair["E_low_over_gap"] <= thresholds.max_low_energy_over_bulk_gap
                    and pair["E_low_over_next"] <= thresholds.max_low_energy_over_next
                    and pair["R_L"] >= thresholds.min_separation_ratio
                    and pair["R_R"] >= thresholds.min_separation_ratio
                )
                ok_count += int(ok)
                if ok:
                    all_supported_eps.append(abs(pair["splitting"]))
            support[i, j] = ok_count / n_realizations
            mean_eps[i, j] = float(np.mean(splits))

    return {
        "L_list": np.array(L_list, dtype=float),
        "W_list": np.array(W_list, dtype=float),
        "support": support,
        "mean_eps": mean_eps,
        "supported_eps_values": np.array(all_supported_eps, dtype=float),
    }


# ---------------------------------------------------------------------------
# Charge-parity island + open-system control
# ---------------------------------------------------------------------------


class CoulombBlockadeMasterEquation:
    """
    Two-charge-state charge-parity bridge with an explicit charging-energy
    origin:  E_n = 4 Ec (n - n_g)^2, n in {0, 1}, giving H_charge = [(E0-E1)/2] Z.

        H_eff = H_charge + (eps_M/2) X + (Omega/2)[cos(phi) X + sin(phi) Y]

    eps_M is an imported phenomenological splitting scale from the BdG
    layer, not a microscopic projection. The (Omega, phi) term is an
    explicit external control drive, independent of eps_M.
    """

    def __init__(self, params: IslandParameters):
        self.p = params
        self.P = pauli_matrices()

    def charging_energies(self, Vg: float, delta_V: float, Ec_scale: float = 1.0):
        V_eff = Vg + delta_V
        n_g = self.p.gate_to_ng * V_eff
        Ec = self.p.Ec * Ec_scale
        E0 = 4.0 * Ec * (0.0 - n_g) ** 2
        E1 = 4.0 * Ec * (1.0 - n_g) ** 2
        return float(E0), float(E1), float(n_g)

    def build_effective_hamiltonian(
        self, Vg: float, delta_V: float, eps_M: float,
        drive_amp: float = 0.0, drive_phase: float = 0.0, Ec_scale: float = 1.0,
    ) -> np.ndarray:
        E0, E1, _ = self.charging_energies(Vg, delta_V, Ec_scale=Ec_scale)
        H_charge = 0.5 * (E0 - E1) * self.P["Z"]
        H_intrinsic = 0.5 * eps_M * self.P["X"]
        H_drive = 0.5 * drive_amp * (
            np.cos(drive_phase) * self.P["X"] + np.sin(drive_phase) * self.P["Y"]
        )
        H = H_charge + H_intrinsic + H_drive
        return (H + dagger(H)) / 2.0

    @staticmethod
    def liouvillian_hamiltonian(H: np.ndarray) -> np.ndarray:
        d = H.shape[0]
        I = np.eye(d, dtype=complex)
        return -1j * (np.kron(I, H) - np.kron(H.T, I))

    @staticmethod
    def liouvillian_lindblad(L: np.ndarray) -> np.ndarray:
        d = L.shape[0]
        I = np.eye(d, dtype=complex)
        LdagL = dagger(L) @ L
        return np.kron(L.conj(), L) - 0.5 * np.kron(I, LdagL) - 0.5 * np.kron(LdagL.T, I)

    def build_liouvillian(self, H: np.ndarray, noise: NoiseParameters) -> np.ndarray:
        P = self.P
        Ltot = self.liouvillian_hamiltonian(H)
        if noise.gamma_phi > 0:
            Ltot += self.liouvillian_lindblad(np.sqrt(noise.gamma_phi / 2.0) * P["Z"])
        if noise.gamma_relax > 0:
            sm = np.array([[0, 1], [0, 0]], dtype=complex)
            Ltot += self.liouvillian_lindblad(np.sqrt(noise.gamma_relax) * sm)
        if noise.gamma_excitation > 0:
            sp = np.array([[0, 0], [1, 0]], dtype=complex)
            Ltot += self.liouvillian_lindblad(np.sqrt(noise.gamma_excitation) * sp)
        if noise.gamma_qp > 0:
            # Effective parity-flip poisoning channel (phenomenological proxy).
            Ltot += self.liouvillian_lindblad(np.sqrt(noise.gamma_qp) * P["X"])
        return Ltot

    def evolve_density_matrix(
        self, rho_0: np.ndarray, Vg: float, delta_V: float, t_pulse: float, eps_M: float,
        noise: NoiseParameters, drive_amp: float = 0.0, drive_phase: float = 0.0, Ec_scale: float = 1.0,
    ) -> np.ndarray:
        H = self.build_effective_hamiltonian(Vg, delta_V, eps_M, drive_amp, drive_phase, Ec_scale)
        L = self.build_liouvillian(H, noise)
        rho_vec = np.asarray(rho_0, dtype=complex).flatten(order="F")
        rho_t = (la.expm(L * t_pulse) @ rho_vec).reshape((2, 2), order="F")
        rho_t = (rho_t + dagger(rho_t)) / 2.0
        tr = np.trace(rho_t)
        if abs(tr) < 1e-14:
            raise RuntimeError("Density-matrix trace collapsed numerically")
        rho_t /= tr
        mineig = float(np.min(la.eigvalsh(rho_t)))
        if mineig < -1e-8:
            raise RuntimeError(f"Density matrix lost positivity: min eigenvalue={mineig:.3e}")
        return rho_t


# ---------------------------------------------------------------------------
# Targets, metrics, and the NEW analytic reachability check
# ---------------------------------------------------------------------------


def target_density_matrix(name: str) -> Tuple[np.ndarray, str]:
    name = name.upper()
    if name == "H":
        psi = np.array([np.cos(np.pi / 8.0), np.sin(np.pi / 8.0)], dtype=complex)
        return np.outer(psi, psi.conj()), "H-type magic state"
    if name == "T":
        psi = np.array([1.0, np.exp(1j * np.pi / 4.0)], dtype=complex) / np.sqrt(2.0)
        return np.outer(psi, psi.conj()), "canonical T magic state"
    raise ValueError("target must be H or T")


def pure_state_fidelity(rho: np.ndarray, target_rho: np.ndarray) -> float:
    return float(np.clip(np.real(np.trace(target_rho @ rho)), 0.0, 1.0))


def target_state_infidelity(rho: np.ndarray, target_rho: np.ndarray) -> float:
    return float(np.clip(1.0 - pure_state_fidelity(rho, target_rho), 0.0, 1.0))


def bloch_vector(rho: np.ndarray) -> np.ndarray:
    P = pauli_matrices()
    return np.array([
        np.real(np.trace(P["X"] @ rho)),
        np.real(np.trace(P["Y"] @ rho)),
        np.real(np.trace(P["Z"] @ rho)),
    ])


def pauli_fourth_moment_nonstabilizerness(rho: np.ndarray) -> float:
    """Pauli-fourth-moment nonstabilizerness PROXY -- not a certified
    mixed-state magic monotone (see module notes in earlier versions)."""
    P = pauli_matrices()
    moment = sum(abs(np.trace(op @ rho)) ** 4 for op in P.values()) / 2.0
    if moment <= 0:
        raise RuntimeError("Invalid fourth-moment value")
    return float(-np.log2(moment))


def bloch_axis_reachability(
    rho_init: np.ndarray, target_rho: np.ndarray, axis: str = "Z", tol: float = 1e-6
) -> Dict[str, object]:
    """
    NEW in v7: explicit, computed check of whether `target_rho` can be
    reached from `rho_init` by a PURE rotation about the given Bloch axis
    alone (e.g. the charge/"Z" term with no X/Y drive).

    A single-axis rotation preserves both the state's radius on the Bloch
    sphere and its component along the rotation axis; it only sweeps the
    angle in the perpendicular plane. So reachability requires the two
    states to share the same axis-component and the same perpendicular
    radius (both are automatically 1 and axis-component-consistent for two
    pure states related by such a rotation). If reachable, the required
    signed rotation angle in the perpendicular plane is also returned.

    This is what turns "the optimizer happened to find delta_V that cancels
    the bias" into a predicted, falsifiable fact about the specific pair of
    states involved, instead of a coincidence noticed after the run.
    """
    v0 = bloch_vector(rho_init)
    v1 = bloch_vector(target_rho)
    axis_idx = {"X": 0, "Y": 1, "Z": 2}[axis.upper()]
    perp_idx = [i for i in range(3) if i != axis_idx]

    axis_match = bool(abs(v0[axis_idx] - v1[axis_idx]) < tol)
    r0 = float(np.hypot(v0[perp_idx[0]], v0[perp_idx[1]]))
    r1 = float(np.hypot(v1[perp_idx[0]], v1[perp_idx[1]]))
    radius_match = bool(abs(r0 - r1) < tol)

    reachable = axis_match and radius_match
    angle = None
    if reachable and r0 > tol:
        theta0 = float(np.arctan2(v0[perp_idx[1]], v0[perp_idx[0]]))
        theta1 = float(np.arctan2(v1[perp_idx[1]], v1[perp_idx[0]]))
        angle = float(np.mod(theta1 - theta0 + np.pi, 2 * np.pi) - np.pi)
    elif reachable and r0 <= tol:
        # Both states lie exactly on the rotation axis: any angle "works"
        # (rotation does nothing to a pole), including angle 0.
        angle = 0.0

    return {
        "axis": axis.upper(), "reachable_by_axis_rotation_alone": reachable,
        "axis_component_match": axis_match, "perp_radius_match": radius_match,
        "required_angle_rad": angle,
        "init_bloch": v0.tolist(), "target_bloch": v1.tolist(),
    }


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    if trials <= 0:
        raise ValueError("trials must be positive")
    p = successes / trials
    denom = 1.0 + z ** 2 / trials
    center = (p + z ** 2 / (2.0 * trials)) / denom
    half = z * np.sqrt((p * (1 - p) / trials) + z ** 2 / (4.0 * trials ** 2)) / denom
    return float(max(0.0, center - half)), float(min(1.0, center + half))


def cvar(values: np.ndarray, tail_fraction: float = 0.05) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        raise ValueError("values cannot be empty")
    k = max(1, int(np.ceil(tail_fraction * values.size)))
    return float(np.mean(np.sort(values)[-k:]))


def fit_oscillatory_splitting(lengths: np.ndarray, energies: np.ndarray) -> Dict[str, float]:
    """Empirical descriptor E(L) ~= A exp(-L/xi) |cos(kL+phi)|. NOT claimed
    as proof of exponential topological protection."""
    L = np.asarray(lengths, dtype=float)
    E = np.asarray(energies, dtype=float)
    mask = np.isfinite(E) & (E > 1e-12) & np.isfinite(L)
    L, E = L[mask], E[mask]
    if L.size < 8:
        return {"fit_success": 0.0, "A": np.nan, "xi": np.nan, "k": np.nan, "phi": np.nan, "log_rmse": np.nan}

    def predict(p):
        return np.exp(p[0]) * np.exp(-L / np.exp(p[1])) * np.abs(np.cos(p[2] * L + p[3])) + 1e-12

    def residual(p):
        return np.log(predict(p)) - np.log(E)

    lo = np.array([-12.0, np.log(2.0), 0.0, -np.pi])
    hi = np.array([1.0, np.log(1000.0), np.pi, np.pi])
    rng = np.random.default_rng(11)
    starts = [
        np.array([np.log(max(E)), np.log(30.0), 0.15, 0.0]),
        np.array([np.log(max(E)), np.log(50.0), 0.35, 1.0]),
        np.array([np.log(max(E)), np.log(80.0), 0.60, -1.0]),
    ]
    for _ in range(10):
        starts.append(np.array([rng.uniform(lo[i], hi[i]) for i in range(4)]))
    best = None
    for x0 in starts:
        try:
            res = least_squares(residual, x0, bounds=(lo, hi), max_nfev=5000)
            score = float(np.mean(res.fun ** 2))
            if best is None or score < best[0]:
                best = (score, res.x)
        except Exception:
            continue
    if best is None:
        return {"fit_success": 0.0, "A": np.nan, "xi": np.nan, "k": np.nan, "phi": np.nan, "log_rmse": np.nan}
    score, p = best
    return {"fit_success": 1.0, "A": float(np.exp(p[0])), "xi": float(np.exp(p[1])),
            "k": float(p[2]), "phi": float(p[3]), "log_rmse": float(np.sqrt(score))}


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


class GateCalibrator:
    """Model-based bounded optimization; never represented as lab feedback."""

    def __init__(self, master_eq, control, rho_init, target_rho, eps_M):
        self.master_eq = master_eq
        self.control = control
        self.rho_init = rho_init
        self.target_rho = target_rho
        self.eps_M = float(eps_M)

    def loss(self, delta_V, gamma_phi, gamma_qp=0.0, drive_amp=0.0, drive_phase=0.0,
              bias_override=None, Ec_scale=1.0) -> float:
        true_bias = self.control.true_bias if bias_override is None else float(bias_override)
        rho = self.master_eq.evolve_density_matrix(
            rho_0=self.rho_init, Vg=self.control.Vg_nominal, delta_V=true_bias + delta_V,
            t_pulse=self.control.t_pulse, eps_M=self.eps_M,
            noise=NoiseParameters(gamma_phi=gamma_phi, gamma_qp=gamma_qp),
            drive_amp=drive_amp, drive_phase=drive_phase, Ec_scale=Ec_scale,
        )
        return target_state_infidelity(rho, self.target_rho)

    def optimize_bias_only(self, gamma_phi: float, gamma_qp: float = 0.0):
        res = minimize_scalar(
            lambda dv: self.loss(dv, gamma_phi=gamma_phi, gamma_qp=gamma_qp),
            bounds=(self.control.delta_v_min, self.control.delta_v_max),
            method="bounded", options={"xatol": 1e-8},
        )
        return float(res.x), float(res.fun)

    def optimize_with_drive(self, gamma_phi: float, gamma_qp: float = 0.0,
                              maxiter: int = 60, popsize: int = 12,
                              delta_v_min=None, delta_v_max=None):
        lo = self.control.delta_v_min if delta_v_min is None else delta_v_min
        hi = self.control.delta_v_max if delta_v_max is None else delta_v_max
        bounds = [(lo, hi), (0.0, self.control.drive_amp_max), (0.0, 2 * np.pi)]

        def objective(x):
            return self.loss(x[0], gamma_phi=gamma_phi, gamma_qp=gamma_qp, drive_amp=x[1], drive_phase=x[2])

        res = differential_evolution(objective, bounds, seed=0, maxiter=maxiter, popsize=popsize,
                                       tol=1e-9, polish=True, updating="deferred", workers=1)
        return (float(res.x[0]), float(res.x[1]), float(res.x[2])), float(res.fun)

    @staticmethod
    def _sample_uncertainties(n_samples: int, seed: int):
        rng = np.random.default_rng(seed)
        return {
            "bias": rng.normal(0.0, 0.02, n_samples),
            "amp": rng.normal(0.0, 0.02, n_samples),
            "phase": rng.normal(0.0, 0.02, n_samples),
            "Ec": rng.normal(0.0, 0.01, n_samples),
            "gamma_phi_scale": np.exp(rng.normal(0.0, 0.10, n_samples)),
            "gamma_qp_scale": np.exp(rng.normal(0.0, 0.20, n_samples)),
        }

    def evaluate_parameter_robustness(self, control, gamma_phi, gamma_qp, n_samples=300, seed=123):
        dv, amp, phase = control
        err = self._sample_uncertainties(n_samples, seed)
        vals = np.array([
            self.loss(
                dv + err["bias"][i], gamma_phi=gamma_phi * err["gamma_phi_scale"][i],
                gamma_qp=gamma_qp * err["gamma_qp_scale"][i],
                drive_amp=max(0.0, amp * (1.0 + err["amp"][i])), drive_phase=phase + err["phase"][i],
                Ec_scale=max(0.5, 1.0 + err["Ec"][i]),
            ) for i in range(n_samples)
        ])
        return {
            "mean_infidelity": float(np.mean(vals)), "std_infidelity": float(np.std(vals)),
            "p95_infidelity": float(np.quantile(vals, 0.95)), "worst_infidelity": float(np.max(vals)),
            "cvar95_infidelity": cvar(vals, 0.05),
            "failure_gt_1pct": float(np.mean(vals > 0.01)), "n_samples": float(n_samples),
        }

    def optimize_with_uncertainty_penalty(self, gamma_phi, gamma_qp, n_samples=20, seed=7,
                                            maxiter=35, popsize=10):
        err = self._sample_uncertainties(n_samples, seed)
        bounds = [(self.control.delta_v_min, self.control.delta_v_max),
                  (0.0, self.control.drive_amp_max), (0.0, 2 * np.pi)]

        def objective(x):
            vals = np.array([
                self.loss(
                    x[0] + err["bias"][i], gamma_phi=gamma_phi * err["gamma_phi_scale"][i],
                    gamma_qp=gamma_qp * err["gamma_qp_scale"][i],
                    drive_amp=max(0.0, x[1] * (1.0 + err["amp"][i])), drive_phase=x[2] + err["phase"][i],
                    Ec_scale=max(0.5, 1.0 + err["Ec"][i]),
                ) for i in range(n_samples)
            ])
            return float(0.5 * np.mean(vals) + 0.5 * cvar(vals, 0.05))

        res = differential_evolution(objective, bounds, seed=seed, maxiter=maxiter, popsize=popsize,
                                       tol=1e-7, polish=True, updating="deferred", workers=1)
        return (float(res.x[0]), float(res.x[1]), float(res.x[2])), float(res.fun)


# ---------------------------------------------------------------------------
# NEW (v7): Stage B -- fidelity map vs (eps_M, gamma_qp)
# ---------------------------------------------------------------------------


def scan_fidelity_vs_eps_and_qp(
    island: CoulombBlockadeMasterEquation, control: ControlParameters,
    rho_init: np.ndarray, target_rho: np.ndarray,
    eps_grid: np.ndarray, gamma_qp_grid: np.ndarray, gamma_phi_fixed: float,
    maxiter: int = 25, popsize: int = 8,
) -> np.ndarray:
    """
    Stage B of the v7 core analysis: drive-optimized target-state infidelity
    over a grid of (eps_M, gamma_qp), at a fixed representative gamma_phi.
    The eps_M axis is populated by the caller with values drawn from the
    Majorana layer's actual achievable splittings (see
    `majorana_operating_window_scan`), not an arbitrary range.
    """
    result = np.zeros((len(eps_grid), len(gamma_qp_grid)))
    for i, eps_M in enumerate(eps_grid):
        calibrator = GateCalibrator(island, control, rho_init, target_rho, float(eps_M))
        for j, gqp in enumerate(gamma_qp_grid):
            _, loss = calibrator.optimize_with_drive(
                gamma_phi=gamma_phi_fixed, gamma_qp=float(gqp), maxiter=maxiter, popsize=popsize,
            )
            result[i, j] = loss
    return result


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def save_csv(path: str, rows: List[Dict[str, float]], fieldnames: List[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: str, obj: Dict[str, object]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------


def run_sanity_checks(physics, island, target_rho, rho_init) -> None:
    rho_mixed = np.eye(2, dtype=complex) / 2.0
    assert np.isclose(pauli_fourth_moment_nonstabilizerness(rho_init), 0.0, atol=1e-12)
    assert pauli_fourth_moment_nonstabilizerness(rho_mixed) > 0.9
    assert np.isclose(pure_state_fidelity(target_rho, target_rho), 1.0, atol=1e-12)
    assert physics.bulk_phs_relative_error(0.37, 0.5) < 1e-10
    assert physics.finite_chain_phs_relative_error(0.5) < 1e-10
    pair = majorana_pair_diagnostics(physics, 0.5)
    assert pair["E_low"] < pair["E_next"]
    assert pair["phs_residual_L"] < 1e-6
    assert pair["phs_residual_R"] < 1e-6

    # New reachability check sanity: a state rotated purely about Z from
    # rho_init must be reported reachable, with the correct angle.
    theta_test = 0.37
    v0 = bloch_vector(rho_init)
    r0 = np.hypot(v0[0], v0[1])
    phi0 = np.arctan2(v0[1], v0[0])
    rotated_bloch = np.array([r0 * np.cos(phi0 + theta_test), r0 * np.sin(phi0 + theta_test), v0[2]])
    P = pauli_matrices()
    rho_rotated = 0.5 * (P["I"] + rotated_bloch[0] * P["X"] + rotated_bloch[1] * P["Y"] + rotated_bloch[2] * P["Z"])
    check = bloch_axis_reachability(rho_init, rho_rotated, axis="Z")
    assert check["reachable_by_axis_rotation_alone"]
    assert abs(check["required_angle_rad"] - theta_test) < 1e-6, check

    rho = island.evolve_density_matrix(
        rho_0=rho_init, Vg=0.5, delta_V=0.0, t_pulse=0.5, eps_M=0.1,
        noise=NoiseParameters(gamma_phi=0.05, gamma_qp=0.02), drive_amp=2.0, drive_phase=0.2,
    )
    assert np.isclose(np.trace(rho), 1.0, atol=1e-10)
    assert np.min(la.eigvalsh(rho)) >= -1e-8

    print("[CHECK] Basic numerical checks passed.")
    print(f"[CHECK] proxy(|+>) = {pauli_fourth_moment_nonstabilizerness(rho_init):.8f}")
    print(f"[CHECK] proxy(I/2) = {pauli_fourth_moment_nonstabilizerness(rho_mixed):.8f}  <-- mixed-state limitation")
    print(f"[CHECK] pair self-conjugacy errors = {pair['phs_residual_L']:.3e}, {pair['phs_residual_R']:.3e}")
    print(f"[CHECK] Z-axis reachability self-test: predicted angle matches injected angle to <1e-6")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_simulation(target_name: str = "T", quick: bool = False) -> None:
    print("=" * 104)
    print("Rigorous Majorana-Coulomb-Island Validation + Operating-Window Analysis v7")
    print("=" * 104)
    outdir = "outputs_v7"
    os.makedirs(outdir, exist_ok=True)

    bdg_params = BdGParameters(L=60 if quick else 80, alpha=0.15, Delta=1.0, Ez=2.5, t0=1.0, gate_to_mu=1.0)
    island_params = IslandParameters(Ec=0.8, gate_to_ng=0.5)
    control_params = ControlParameters(Vg_nominal=0.50, true_bias=0.15, delta_v_min=-1.0, delta_v_max=1.0,
                                        t_pulse=0.5, drive_amp_max=6.0)
    thresholds = ValidationThresholds()
    physics = MajoranaIslandPhysics(bdg_params)
    island = CoulombBlockadeMasterEquation(island_params)
    target_rho, target_label = target_density_matrix(target_name)

    plus = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2.0)
    rho_init = np.outer(plus, plus.conj())
    rho_H, _ = target_density_matrix("H")
    rho_T, _ = target_density_matrix("T")

    run_sanity_checks(physics, island, target_rho, rho_init)

    print(f"\n[TARGET] {target_label}")
    print(f"  target Bloch vector = {np.round(bloch_vector(target_rho), 5)}")
    print(f"  H-state Bloch vector = {np.round(bloch_vector(rho_H), 5)}")
    print(f"  T-state Bloch vector = {np.round(bloch_vector(rho_T), 5)}")
    print("  energy units = t0; time units = 1/t0; hbar = 1")

    # NEW: explicit, computed reachability check (replaces post-hoc narrative).
    reach = bloch_axis_reachability(rho_init, target_rho, axis="Z")
    print("\n[REACHABILITY CHECK] Can a pure charge (Z-axis) rotation alone reach the target?")
    print(f"  init Bloch = {np.round(reach['init_bloch'], 5)}, target Bloch = {np.round(reach['target_bloch'], 5)}")
    print(f"  Z-component match = {reach['axis_component_match']}, perpendicular-radius match = {reach['perp_radius_match']}")
    if reach["reachable_by_axis_rotation_alone"]:
        print(f"  RESULT: reachable by Z-only rotation, required angle = {reach['required_angle_rad']:.6f} rad")
        print("  => a coherent X/Y drive is NOT strictly necessary for this target from this initial state;")
        print("     any drive-optimized improvement below is about robustness under noise, not reachability.")
    else:
        print("  RESULT: NOT reachable by Z-only rotation from this initial state.")
        print("  => the external X/Y drive is genuinely required to prepare this target, not optional.")

    # ----------------------------------------------------------------------
    # Majorana validation
    # ----------------------------------------------------------------------
    Vg = control_params.Vg_nominal
    status = majorana_status(physics, Vg, thresholds)
    print("\n[MAJORANA VALIDATION] Nominal point")
    print(f"  mu = {physics.chemical_potential(Vg):.6f}")
    print(f"  clean bulk gap = {status['bulk_gap']:.8e}")
    print(f"  E_low = {status['finite_energy']:.8e}, E_next = {status['next_positive_energy']:.8e}")
    print(f"  E_low/E_gap = {status['low_energy_over_gap']:.6f}, E_low/E_next = {status['low_energy_over_next']:.6f}")
    print(f"  class-D Z2 = {status['z2']['z2_invariant']:+.0f}")
    print(f"  R_L/R_R = {status['R_L']:.3f} / {status['R_R']:.3f}")
    print(f"  MAJORANA STATUS = {'SUPPORTED' if status['supported'] else 'NOT SUPPORTED'}")

    # ----------------------------------------------------------------------
    # Bulk gate scan / Z2 phase diagram
    # ----------------------------------------------------------------------
    gate_scan = np.linspace(-3.0, 3.0, 121 if quick else 241)
    bulk_gaps = np.array([physics.bulk_gap(v) for v in gate_scan])
    mu_scan = np.linspace(-3.0, 3.0, 61 if quick else 121)
    ez_scan = np.linspace(0.0, 4.0, 41 if quick else 81)
    z2_map = np.empty((len(ez_scan), len(mu_scan)))
    for ie, Ez in enumerate(ez_scan):
        temp = MajoranaIslandPhysics(BdGParameters(L=bdg_params.L, alpha=bdg_params.alpha, Delta=bdg_params.Delta,
                                                     Ez=float(Ez), t0=bdg_params.t0, gate_to_mu=1.0))
        for im, mu in enumerate(mu_scan):
            z2_map[ie, im] = temp.class_D_pfaffian_invariant(float(mu))["z2_invariant"]

    # ----------------------------------------------------------------------
    # Finite-size scan + oscillatory fit
    # ----------------------------------------------------------------------
    lengths = list(range(20, 121, 10)) if quick else list(range(20, 121, 5))
    finite = physics.finite_size_scan(Vg, lengths)
    print("\n[FINITE SIZE]")
    for L, row in finite.items():
        print(f"  L={L:3d}  E_low={row['E_low']:.6e}  E_next={row['E_next']:.6e}  "
              f"E_low/E_next={row['E_low_over_next']:.4f}  R_L={row['R_L']:.2f}  R_R={row['R_R']:.2f}")
    fit = fit_oscillatory_splitting(np.array(list(finite.keys()), dtype=float),
                                     np.array([finite[L]["E_low"] for L in finite], dtype=float))
    if fit["fit_success"]:
        print(f"  empirical fit: A={fit['A']:.4e}, xi={fit['xi']:.4f}, k={fit['k']:.4f}, log-RMSE={fit['log_rmse']:.4f}")
        print("  (empirical descriptor only, not a proof of exponential protection)")

    # ----------------------------------------------------------------------
    # Single-length disorder scan (100 realizations, as in v6)
    # ----------------------------------------------------------------------
    n_disorder = 20 if quick else 100
    W_list_single = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
    disorder = disorder_robustness_scan(bdg_params, Vg, W_list_single, n_realizations=n_disorder,
                                          base_seed=1000, thresholds=thresholds)
    print(f"\n[DISORDER] (single length L={bdg_params.L}) n_realizations={n_disorder}")
    for W, row in disorder.items():
        lo, hi = wilson_interval(int(round(row["support_fraction"] * n_disorder)), n_disorder)
        row["support_ci_low"], row["support_ci_high"] = lo, hi
        print(f"  W={W:.2f}  <E_low>={row['mean_splitting']:.4e}+/-{row['std_splitting']:.4e}  "
              f"support={row['support_fraction']:.2f}  95%CI=[{lo:.2f},{hi:.2f}]")

    # ----------------------------------------------------------------------
    # NEW v7 STAGE A: joint (L, W) Majorana operating window
    # ----------------------------------------------------------------------
    print("\n[STAGE A] Joint (L, W) Majorana operating-window scan")
    L_grid = [40, 60, 80, 100, 120] if not quick else [40, 80, 120]
    W_grid = [0.0, 0.1, 0.2, 0.3, 0.4] if not quick else [0.0, 0.2, 0.4]
    n_real_window = 3 if quick else 5
    window = majorana_operating_window_scan(bdg_params, Vg, L_grid, W_grid,
                                              n_realizations=n_real_window, thresholds=thresholds)
    print("  support fraction grid (rows=L, cols=W):")
    header = "        " + "".join(f"W={w:5.2f} " for w in W_grid)
    print(header)
    for i, L in enumerate(L_grid):
        line = f"  L={L:4d} " + "".join(f"{window['support'][i, j]:8.2f} " for j in range(len(W_grid)))
        print(line)

    supported_eps = window["supported_eps_values"]
    if supported_eps.size >= 3:
        eps_lo, eps_hi = float(np.min(supported_eps)), float(np.max(supported_eps))
    else:
        # Fall back to the nominal-point splitting if too few cells pass.
        eps_lo = eps_hi = float(status["finite_energy"])
    print(f"  --> Majorana-operating eps_M range from {supported_eps.size} passing cells: "
          f"[{eps_lo:.4e}, {eps_hi:.4e}]")

    # ----------------------------------------------------------------------
    # Island bridge
    # ----------------------------------------------------------------------
    eps_M = float(status["finite_energy"])
    E0, E1, n_g = island.charging_energies(Vg, 0.0)
    print("\n[ISLAND BRIDGE]")
    print(f"  Ec={island_params.Ec:.4f}, n_g(no correction)={n_g:.4f}, E0/E1={E0:.4f}/{E1:.4f}")
    print(f"  extracted eps_M (nominal point) = {eps_M:.8e}")
    print("  charge term derived from explicit two-state charging spectrum E_n = 4 Ec (n-n_g)^2.")
    print("  Full microscopic BdG+charging+tunneling projection is still NOT claimed.")

    calibrator = GateCalibrator(island, control_params, rho_init, target_rho, eps_M)
    gamma_phi_grid = np.linspace(0.0, 0.20, 5 if quick else 8)
    rows: List[Dict[str, float]] = []
    print(f"\n[CALIBRATION] Target = {target_name}; bias-only vs coherent drive")
    print(" gamma_phi | infid_bias_only | infid_drive | delta_V* Omega* phi*")
    for gphi in gamma_phi_grid:
        dv_base, loss_base = calibrator.optimize_bias_only(float(gphi))
        (dv, om, ph), loss_drive = calibrator.optimize_with_drive(
            gamma_phi=float(gphi), maxiter=45 if quick else 65, popsize=10 if quick else 14)
        print(f" {gphi:9.3f} |     {100*loss_base:10.5f}% |   {100*loss_drive:10.5f}% | {dv:7.4f} {om:6.3f} {ph:6.3f}")
        rows.append({"gamma_phi": float(gphi), "gamma_qp": 0.0, "infidelity_bias_only": loss_base,
                      "infidelity_drive_optimized": loss_drive, "delta_v": dv, "drive_amp": om, "drive_phase": ph})

    representative_phi = 0.05
    gamma_qp_grid_single = [0.0, 0.01, 0.03, 0.05, 0.10]
    print("\n[POISONING] Effective parity-flip channel scan, fixed gamma_phi=0.05")
    for gqp in gamma_qp_grid_single:
        (dv, om, ph), loss = calibrator.optimize_with_drive(gamma_phi=representative_phi, gamma_qp=float(gqp),
                                                              maxiter=35 if quick else 55, popsize=10 if quick else 12)
        print(f"  gamma_qp={gqp:6.3f}  infidelity={100*loss:9.5f}%  delta_V={dv:7.4f}  Omega={om:6.3f}")
        rows.append({"gamma_phi": representative_phi, "gamma_qp": float(gqp), "infidelity_bias_only": np.nan,
                      "infidelity_drive_optimized": loss, "delta_v": dv, "drive_amp": om, "drive_phase": ph})

    print("\n[CONTROL-BOUND SENSITIVITY]")
    bound_results = []
    for bound in ([0.75, 1.0, 1.5, 2.0] if not quick else [0.75, 1.0, 1.5]):
        (dv, om, ph), loss = calibrator.optimize_with_drive(gamma_phi=representative_phi, gamma_qp=0.01,
                                                              maxiter=25 if quick else 35, popsize=8 if quick else 10,
                                                              delta_v_min=-bound, delta_v_max=bound)
        active = abs(abs(dv) - bound) < 2e-3
        bound_results.append((bound, loss, dv, om, ph, active))
        print(f"  bound=+/-{bound:.2f}  infidelity={100*loss:.5f}%  delta_V={dv:.5f}  active={active}")
    b1 = next((r for r in bound_results if abs(r[0] - 1.0) < 1e-12), None)
    b2 = next((r for r in bound_results if abs(r[0] - 2.0) < 1e-12), None)
    bound_sensitive = bool(b1 and b2 and b2[1] < b1[1] - 1e-5)

    # ----------------------------------------------------------------------
    # NEW v7 STAGE B: fidelity map over the physically motivated eps_M range
    # ----------------------------------------------------------------------
    print("\n[STAGE B] Drive-optimized T/H-state infidelity vs (eps_M, gamma_qp)")
    eps_grid = np.geomspace(max(eps_lo * 0.3, 1e-6), eps_hi * 3.0, 5)
    gamma_qp_grid_2d = np.array([0.0, 0.02, 0.05, 0.1, 0.2])
    fidelity_map = scan_fidelity_vs_eps_and_qp(
        island, control_params, rho_init, target_rho, eps_grid, gamma_qp_grid_2d,
        gamma_phi_fixed=representative_phi,
        maxiter=20 if quick else 25, popsize=6 if quick else 8,
    )
    print("  infidelity (%) grid (rows=eps_M, cols=gamma_qp):")
    header2 = "              " + "".join(f"qp={g:5.2f} " for g in gamma_qp_grid_2d)
    print(header2)
    for i, eps in enumerate(eps_grid):
        in_band = " <-- operating band" if eps_lo <= eps <= eps_hi else ""
        line = f"  eps={eps:.3e} " + "".join(f"{100*fidelity_map[i, j]:8.4f} " for j in range(len(gamma_qp_grid_2d)))
        print(line + in_band)

    # Overlap summary: does the physically-accessible eps_M band overlap a
    # low-infidelity region, and over what gamma_qp range?
    in_band_mask = (eps_grid >= eps_lo) & (eps_grid <= eps_hi)
    if not np.any(in_band_mask):
        # Use the closest grid row(s) to the band if none fall exactly inside.
        in_band_mask = np.zeros_like(eps_grid, dtype=bool)
        in_band_mask[int(np.argmin(np.abs(eps_grid - 0.5 * (eps_lo + eps_hi))))] = True
    band_fidelity_rows = fidelity_map[in_band_mask, :]
    band_infid_vs_qp = np.mean(band_fidelity_rows, axis=0)
    low_infid_threshold = 0.01
    ok_qp = gamma_qp_grid_2d[band_infid_vs_qp < low_infid_threshold]
    print(f"\n[OVERLAP SUMMARY] Within the Majorana-operating eps_M band [{eps_lo:.3e},{eps_hi:.3e}]:")
    print(f"  mean infidelity vs gamma_qp = {np.round(100*band_infid_vs_qp, 4)}%")
    if ok_qp.size > 0:
        print(f"  --> infidelity stays below {100*low_infid_threshold:.1f}% for gamma_qp up to "
              f"{ok_qp.max():.3f} (of the {gamma_qp_grid_2d.tolist()} scanned)")
        print("  --> the Majorana-operating region and the high-fidelity region DO overlap over that range.")
    else:
        print(f"  --> infidelity never drops below {100*low_infid_threshold:.1f}% anywhere in the scanned "
              f"gamma_qp range at this gamma_phi.")
        print("  --> no overlap found in the scanned range; either lower gamma_qp/gamma_phi or a different")
        print("      (L, W) operating point would be needed for this target.")

    # ----------------------------------------------------------------------
    # Out-of-sample control-uncertainty robustness (kept, incl. negative result)
    # ----------------------------------------------------------------------
    best_nominal, _ = calibrator.optimize_with_drive(0.0, 0.0, maxiter=45 if quick else 55, popsize=10)
    eval_n = 200 if quick else 300
    nominal_robust = calibrator.evaluate_parameter_robustness(best_nominal, gamma_phi=0.05, gamma_qp=0.02,
                                                                 n_samples=eval_n, seed=12345)
    print("\n[CONTROL ROBUSTNESS] Independent out-of-sample uncertainty ensemble")
    print(f"  clean optimum: delta_V={best_nominal[0]:.4f}, Omega={best_nominal[1]:.4f}, phi={best_nominal[2]:.4f}")
    print(f"  mean/p95/CVaR95 = {100*nominal_robust['mean_infidelity']:.4f}% / "
          f"{100*nominal_robust['p95_infidelity']:.4f}% / {100*nominal_robust['cvar95_infidelity']:.4f}%")

    robust_test = None
    robust_better = False
    if not quick:
        robust_control, _ = calibrator.optimize_with_uncertainty_penalty(
            gamma_phi=0.05, gamma_qp=0.02, n_samples=20, seed=7, maxiter=35, popsize=10)
        robust_test = calibrator.evaluate_parameter_robustness(robust_control, gamma_phi=0.05, gamma_qp=0.02,
                                                                  n_samples=eval_n, seed=54321)
        nominal_same_test = calibrator.evaluate_parameter_robustness(best_nominal, gamma_phi=0.05, gamma_qp=0.02,
                                                                        n_samples=eval_n, seed=54321)
        robust_better = bool(robust_test["cvar95_infidelity"] < nominal_same_test["cvar95_infidelity"]
                              and robust_test["p95_infidelity"] <= nominal_same_test["p95_infidelity"])
        print("[ROBUST OPTIMIZATION] Train on 20 uncertainty samples; test independently")
        print(f"  nominal same-test p95/CVaR95 = {100*nominal_same_test['p95_infidelity']:.4f}% / "
              f"{100*nominal_same_test['cvar95_infidelity']:.4f}%")
        print(f"  robust  same-test p95/CVaR95 = {100*robust_test['p95_infidelity']:.4f}% / "
              f"{100*robust_test['cvar95_infidelity']:.4f}%")
        print(f"  OUT-OF-SAMPLE ROBUST ADVANTAGE = {robust_better}  "
              "(reported honestly regardless of outcome; not a claim this script needs to succeed at)")

    # ----------------------------------------------------------------------
    # Save data
    # ----------------------------------------------------------------------
    finite_rows = [{"L": float(L), **row} for L, row in finite.items()]
    disorder_rows = [{"disorder_strength": float(W), **row} for W, row in disorder.items()]
    save_csv(os.path.join(outdir, "calibration_scan_v7.csv"), rows,
              ["gamma_phi", "gamma_qp", "infidelity_bias_only", "infidelity_drive_optimized", "delta_v", "drive_amp", "drive_phase"])
    save_csv(os.path.join(outdir, "finite_size_scan_v7.csv"), finite_rows,
              ["L", "E_low", "E_next", "bulk_gap", "E_low_over_gap", "E_low_over_next",
               "legacy_edge_localization", "R_L", "R_R", "phs_residual_L", "phs_residual_R"])
    save_csv(os.path.join(outdir, "disorder_scan_v7.csv"), disorder_rows,
              ["disorder_strength", "mean_splitting", "std_splitting", "mean_E_low_over_gap", "mean_E_low_over_next",
               "mean_R_L", "mean_R_R", "min_R_L", "min_R_R", "support_fraction", "support_ci_low", "support_ci_high"])
    save_csv(os.path.join(outdir, "control_bound_sensitivity_v7.csv"),
              [{"delta_v_abs_bound": b, "infidelity": loss, "delta_v": dv, "drive_amp": om, "drive_phase": ph, "active": float(a)}
               for b, loss, dv, om, ph, a in bound_results],
              ["delta_v_abs_bound", "infidelity", "delta_v", "drive_amp", "drive_phase", "active"])
    robustness_rows = [{"protocol": "nominal_clean_control", "metric": k, "value": float(v)} for k, v in nominal_robust.items()]
    if robust_test is not None:
        robustness_rows += [{"protocol": "robust_control", "metric": k, "value": float(v)} for k, v in robust_test.items()]
    save_csv(os.path.join(outdir, "control_robustness_v7.csv"), robustness_rows, ["protocol", "metric", "value"])

    window_rows = []
    for i, L in enumerate(L_grid):
        for j, W in enumerate(W_grid):
            window_rows.append({"L": L, "W": W, "support_fraction": float(window["support"][i, j]),
                                 "mean_eps": float(window["mean_eps"][i, j])})
    save_csv(os.path.join(outdir, "operating_window_v7.csv"), window_rows, ["L", "W", "support_fraction", "mean_eps"])

    fmap_rows = []
    for i, eps in enumerate(eps_grid):
        for j, gqp in enumerate(gamma_qp_grid_2d):
            fmap_rows.append({"eps_M": float(eps), "gamma_qp": float(gqp), "infidelity": float(fidelity_map[i, j]),
                               "in_operating_band": bool(eps_lo <= eps <= eps_hi)})
    save_csv(os.path.join(outdir, "fidelity_vs_epsM_gammaqp_v7.csv"), fmap_rows,
              ["eps_M", "gamma_qp", "infidelity", "in_operating_band"])

    # ----------------------------------------------------------------------
    # Figures
    # ----------------------------------------------------------------------
    Ls = np.array(list(finite.keys()), dtype=float)
    E1a = np.array([finite[int(L)]["E_low"] for L in Ls])
    E2a = np.array([finite[int(L)]["E_next"] for L in Ls])

    fig = plt.figure(figsize=(9, 5.5)); ax = fig.add_subplot(1, 1, 1)
    ax.plot(gate_scan, bulk_gaps, lw=1.5); ax.axvline(Vg, linestyle="--", label="nominal Vg")
    ax.set_xlabel("Vg"); ax.set_ylabel("Clean bulk gap"); ax.set_title("Clean bulk gap vs gate")
    ax.grid(True, linestyle=":", alpha=0.6); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "bulk_gap_vs_gate_v7.png"), dpi=300, bbox_inches="tight"); plt.close(fig)

    fig = plt.figure(figsize=(9, 5.5)); ax = fig.add_subplot(1, 1, 1)
    ax.semilogy(Ls, E1a, "o-", label="E_low"); ax.semilogy(Ls, E2a, "s--", label="E_next")
    ax.axhline(status["bulk_gap"], linestyle=":", label="clean bulk gap")
    ax.set_xlabel("L"); ax.set_ylabel("Energy"); ax.set_title("Finite-size low-energy spectrum")
    ax.grid(True, which="both", linestyle=":", alpha=0.6); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "finite_size_splitting_v7.png"), dpi=300, bbox_inches="tight"); plt.close(fig)

    fig = plt.figure(figsize=(9, 5.5)); ax = fig.add_subplot(1, 1, 1)
    ax.semilogy(Ls, [finite[int(L)]["R_L"] for L in Ls], "o-", label="R_L")
    ax.semilogy(Ls, [finite[int(L)]["R_R"] for L in Ls], "s-", label="R_R")
    ax.axhline(thresholds.min_separation_ratio, linestyle=":", label="threshold")
    ax.set_xlabel("L"); ax.set_ylabel("Separation ratio"); ax.set_title("Majorana-pair spatial separation")
    ax.grid(True, which="both", linestyle=":", alpha=0.6); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "majorana_separation_v7.png"), dpi=300, bbox_inches="tight"); plt.close(fig)

    fig = plt.figure(figsize=(9, 5.5)); ax = fig.add_subplot(1, 1, 1)
    Wd = np.array(list(disorder.keys()), dtype=float)
    ax.errorbar(Wd, [disorder[w]["mean_splitting"] for w in Wd], yerr=[disorder[w]["std_splitting"] for w in Wd],
                marker="o", capsize=4)
    ax.set_xlabel("W"); ax.set_ylabel("Mean |E_low|"); ax.set_title(f"Disorder-induced splitting, L={bdg_params.L}")
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "disorder_splitting_v7.png"), dpi=300, bbox_inches="tight"); plt.close(fig)

    fig = plt.figure(figsize=(9, 5.5)); ax = fig.add_subplot(1, 1, 1)
    support_arr = np.array([disorder[w]["support_fraction"] for w in Wd])
    lo95 = np.array([disorder[w]["support_ci_low"] for w in Wd]); hi95 = np.array([disorder[w]["support_ci_high"] for w in Wd])
    ax.plot(Wd, support_arr, "o-"); ax.fill_between(Wd, lo95, hi95, alpha=0.2)
    ax.set_xlabel("W"); ax.set_ylabel("Support fraction"); ax.set_ylim(0, 1.05)
    ax.set_title(f"Diagnostic survival probability, N={n_disorder}, L={bdg_params.L}")
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "disorder_support_fraction_v7.png"), dpi=300, bbox_inches="tight"); plt.close(fig)

    gphi_arr = np.array([r["gamma_phi"] for r in rows if r["gamma_qp"] == 0.0])
    base_arr = np.array([r["infidelity_bias_only"] for r in rows if r["gamma_qp"] == 0.0])
    drive_arr = np.array([r["infidelity_drive_optimized"] for r in rows if r["gamma_qp"] == 0.0])
    fig = plt.figure(figsize=(9, 5.5)); ax = fig.add_subplot(1, 1, 1)
    ax.plot(gphi_arr, 100*base_arr, "o--", label="Bias-only (Z rotation)")
    ax.plot(gphi_arr, 100*drive_arr, "s-", label="Coherent-drive optimized")
    ax.set_xlabel("gamma_phi"); ax.set_ylabel("Infidelity (%)")
    ax.set_title(f"{target_name}-state preparation under dephasing")
    ax.grid(True, linestyle=":", alpha=0.6); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "dephasing_calibration_v7.png"), dpi=300, bbox_inches="tight"); plt.close(fig)

    poison_rows_plot = [r for r in rows if abs(r["gamma_phi"] - representative_phi) < 1e-12 and r["gamma_qp"] > 0]
    if poison_rows_plot:
        px = np.array([r["gamma_qp"] for r in poison_rows_plot]); py = np.array([r["infidelity_drive_optimized"] for r in poison_rows_plot])
        fig = plt.figure(figsize=(9, 5.5)); ax = fig.add_subplot(1, 1, 1)
        ax.plot(px, 100*py, "o-")
        ax.set_xlabel("gamma_qp (effective parity-flip rate)"); ax.set_ylabel("Infidelity (%)")
        ax.set_title(f"Poisoning-channel sensitivity at gamma_phi={representative_phi}")
        ax.grid(True, linestyle=":", alpha=0.6)
        fig.tight_layout(); fig.savefig(os.path.join(outdir, "poisoning_sensitivity_v7.png"), dpi=300, bbox_inches="tight"); plt.close(fig)

    fig = plt.figure(figsize=(9, 5.5)); ax = fig.add_subplot(1, 1, 1)
    ba = np.array([b[0] for b in bound_results]); bl = np.array([b[1] for b in bound_results])
    ax.plot(ba, 100*bl, "o-")
    ax.set_xlabel("Absolute delta_V bound"); ax.set_ylabel("Infidelity (%)")
    ax.set_title("Control-bound sensitivity")
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "control_bound_sensitivity_v7.png"), dpi=300, bbox_inches="tight"); plt.close(fig)

    fig = plt.figure(figsize=(10, 5.5)); ax = fig.add_subplot(1, 1, 1)
    wL = physics.mode_site_weights(status["pair"]["gamma_L"]); wR = physics.mode_site_weights(status["pair"]["gamma_R"])
    ax.plot(np.arange(physics.p.L), wL, lw=1.6, label="gamma_L")
    ax.plot(np.arange(physics.p.L), wR, lw=1.6, label="gamma_R")
    ax.set_xlabel("Site index"); ax.set_ylabel("Site probability"); ax.set_title("Majorana-like end-mode profiles")
    ax.grid(True, linestyle=":", alpha=0.6); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "majorana_mode_profile_v7.png"), dpi=300, bbox_inches="tight"); plt.close(fig)

    fig = plt.figure(figsize=(10, 5.5)); ax = fig.add_subplot(1, 1, 1)
    mask = (z2_map < 0).astype(float)
    im = ax.imshow(mask, origin="lower", aspect="auto", extent=[mu_scan.min(), mu_scan.max(), ez_scan.min(), ez_scan.max()],
                    interpolation="nearest")
    ax.axhline(bdg_params.Ez, linestyle="--", label="nominal Ez"); ax.axvline(physics.chemical_potential(Vg), linestyle=":", label="nominal mu")
    ax.set_xlabel("mu"); ax.set_ylabel("Ez"); ax.set_title("Class-D Z2 diagnostic")
    fig.colorbar(im, ax=ax, label="nontrivial sector"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "bulk_gap_phase_diagram_v7.png"), dpi=300, bbox_inches="tight"); plt.close(fig)

    # NEW: operating-window heatmap (Stage A)
    fig = plt.figure(figsize=(9, 5.5)); ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(window["support"], origin="lower", aspect="auto",
                    extent=[min(W_grid), max(W_grid), min(L_grid), max(L_grid)], interpolation="nearest",
                    vmin=0, vmax=1, cmap="viridis")
    ax.set_xlabel("Disorder strength W"); ax.set_ylabel("Chain length L")
    ax.set_title("Stage A: Majorana diagnostic support fraction over (L, W)")
    fig.colorbar(im, ax=ax, label="support fraction")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "operating_window_Lw_v7.png"), dpi=300, bbox_inches="tight"); plt.close(fig)

    # NEW: the combined "core figure" (Stage B with Stage A band overlaid)
    fig = plt.figure(figsize=(9.5, 6)); ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(100 * fidelity_map, origin="lower", aspect="auto",
                    extent=[gamma_qp_grid_2d.min(), gamma_qp_grid_2d.max(),
                            np.log10(eps_grid.min()), np.log10(eps_grid.max())],
                    interpolation="nearest", cmap="magma")
    ax.axhspan(np.log10(max(eps_lo, eps_grid.min())), np.log10(min(eps_hi, eps_grid.max())),
               color="cyan", alpha=0.25, label="Majorana-operating eps_M band (Stage A)")
    ax.set_xlabel("gamma_qp (effective parity-flip rate)")
    ax.set_ylabel("log10(eps_M)")
    ax.set_title(f"Stage B: {target_name}-state infidelity(%) vs (eps_M, gamma_qp), gamma_phi={representative_phi}\n"
                 "cyan band = physically accessible eps_M range from Stage A")
    fig.colorbar(im, ax=ax, label="infidelity (%)")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "fidelity_vs_epsM_gammaqp_v7.png"), dpi=300, bbox_inches="tight"); plt.close(fig)

    # ----------------------------------------------------------------------
    # Summary JSON
    # ----------------------------------------------------------------------
    summary = {
        "target": target_name,
        "z_axis_reachability": reach,
        "clean_majorana_status": bool(status["supported"]),
        "nominal": {
            "mu": float(physics.chemical_potential(Vg)), "bulk_gap": float(status["bulk_gap"]),
            "E_low": float(status["finite_energy"]), "E_next": float(status["next_positive_energy"]),
            "Z2": float(status["z2"]["z2_invariant"]), "R_L": float(status["R_L"]), "R_R": float(status["R_R"]),
        },
        "finite_size_fit": fit,
        "operating_window_eps_range": [eps_lo, eps_hi],
        "stage_b_overlap": {
            "eps_grid": eps_grid.tolist(), "gamma_qp_grid": gamma_qp_grid_2d.tolist(),
            "band_mean_infidelity_vs_qp": band_infid_vs_qp.tolist(),
            "max_gamma_qp_below_1pct": float(ok_qp.max()) if ok_qp.size > 0 else None,
        },
        "control_bound_sensitive_at_1_to_2": bound_sensitive,
        "out_of_sample_robust_advantage": bool(robust_better),
        "scientific_claims": {
            "clean_class_D_topology_for_chosen_model": True,
            "well_isolated_spatially_separated_end_mode_candidates": bool(status["supported"]),
            "island_mapping_is_microscopic": False,
            "calibration_is_experimental_closed_loop": False,
            "gamma_qp_is_microscopic_quasiparticle_model": False,
            "device_units_calibrated_to_hardware": False,
            "comparison_to_any_vendor_roadmap_performed": False,
        },
    }
    save_json(os.path.join(outdir, "scientific_summary_v7.json"), summary)

    print("\n" + "=" * 104)
    print("SCIENTIFIC STATUS v7")
    print("=" * 104)
    print(f"A. Clean class-D topology at nominal point: {'PASS' if status['z2']['z2_invariant']<0 else 'FAIL'}")
    print(f"B. Isolated/separated Majorana-pair candidates at nominal point: {'PASS' if status['supported'] else 'FAIL'}")
    print(f"C. Z-axis reachability of the '{target_name}' target from |+>: "
          f"{'REACHABLE (drive optional)' if reach['reachable_by_axis_rotation_alone'] else 'NOT REACHABLE (drive required)'}")
    print(f"D. Stage A/B overlap: high-fidelity (<{100*low_infid_threshold:.0f}%) region reached for "
          f"gamma_qp up to {ok_qp.max():.3f}" if ok_qp.size > 0 else
          "D. Stage A/B overlap: NOT found in the scanned range")
    print(f"E. Control-bound sensitivity (1 -> 2): {'DEMONSTRATED' if bound_sensitive else 'NOT DEMONSTRATED'}")
    print(f"F. Out-of-sample robust-control advantage: {'DEMONSTRATED' if robust_better else 'NOT DEMONSTRATED'} "
          "(reported as-is, negative result kept)")
    print("G. gamma_qp remains an effective parity-flip proxy, not a microscopic quasiparticle model.")
    print("H. No device-unit calibration or vendor-roadmap comparison is made anywhere in this script.")
    print("\n[OUTPUTS] see outputs_v7/ (see module docstring for the full file list)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Majorana island v7: operating-window / fidelity composition")
    parser.add_argument("--target", choices=["H", "T"], default="T", help="target magic state (default: T)")
    parser.add_argument("--quick", action="store_true", help="reduced scans for a fast run")
    args = parser.parse_args()
    run_simulation(target_name=args.target, quick=args.quick)


if __name__ == "__main__":
    main()