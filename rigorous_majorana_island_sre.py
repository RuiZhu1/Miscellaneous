"""
rigorous_majorana_island_comprehensive.py
=========================================
综合版：马约拉纳库仑岛魔态注入校准与控制景观综合分析框架
功能：
1. 消除初态本征态锁定，打通控制通道；
2. 扫描并展示一维 Loss 景观剖面（Loss Curve vs Delta_V）；
3. 模拟不同退相干率下的 SRE 资源量与魔态注入错误率（未校准 vs 优化校准）；
4. 一键生成并保存三拼版综合分析图表。
"""

import numpy as np
import scipy.linalg as la
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import os

os.makedirs("outputs", exist_ok=True)

class MajoranaIslandPhysics:
    """
    1. 微观拓扑物理层：计算 BdG 紧束缚模型在栅压 V_g 下的分裂能
    """
    def __init__(self, L=40, alpha=0.15, Delta=1.0, E_z=2.5, t_0=1.0):
        self.L = L
        self.alpha = alpha
        self.Delta = Delta
        self.E_z = E_z
        self.t_0 = t_0

    def compute_bdg_splitting(self, V_g):
        mu = V_g
        H_0 = (2 * self.t_0 - mu) * np.kron(np.array([[1, 0], [0, -1]]), np.eye(2)) \
              + self.E_z * np.kron(np.eye(2), np.array([[0, 1], [1, 0]])) \
              + self.Delta * np.kron(np.array([[0, 1], [1, 0]]), np.eye(2))
        H_hop = -self.t_0 * np.kron(np.array([[1, 0], [0, -1]]), np.eye(2)) \
                - 1j * self.alpha * np.kron(np.array([[1, 0], [0, -1]]), np.array([[0, -1j], [1j, 0]]))
        
        H_tot = np.zeros((4 * self.L, 4 * self.L), dtype=complex)
        for i in range(self.L):
            H_tot[4*i:4*(i+1), 4*i:4*(i+1)] = H_0
            if i < self.L - 1:
                H_tot[4*i:4*(i+1), 4*(i+1):4*(i+2)] = H_hop
                H_tot[4*(i+1):4*(i+2), 4*i:4*(i+1)] = H_hop.conj().T
                
        evals = la.eigvalsh(H_tot)
        pos_evals = np.sort(evals[evals >= 0])
        eps_M = pos_evals[0] if len(pos_evals) > 0 else 0.001
        return eps_M

class CoulombBlockadeMasterEquation:
    """
    2. 动力学演化层：库仑阻塞开放系统主方程
    """
    def __init__(self, E_C=0.8):
        self.E_C = E_C

    def build_effective_hamiltonian(self, V_g, delta_V, eps_M):
        V_eff = V_g + delta_V
        n_gate = 0.5 * V_eff
        H_z = 0.5 * self.E_C * np.sin(np.pi * n_gate) * np.array([[1, 0], [0, -1]])
        H_x = 0.5 * eps_M * np.array([[0, 1], [1, 0]])
        return H_z + H_x

    def evolve_density_matrix(self, rho_0, V_g, delta_V, t_pulse, gamma_phi, eps_M):
        H = self.build_effective_hamiltonian(V_g, delta_V, eps_M)
        sigma_z = np.array([[1, 0], [0, -1]])
        L_H = -1j * (np.kron(np.eye(2), H) - np.kron(H.T, np.eye(2)))
        L_dephasing = gamma_phi * (np.kron(sigma_z.conj(), sigma_z) - np.eye(4))
        L_tot = L_H + L_dephasing
        
        rho_vec = rho_0.flatten()
        rho_vec_t = la.expm(L_tot * t_pulse) @ rho_vec
        rho_t = rho_vec_t.reshape((2, 2))
        rho_t = (rho_t + rho_t.conj().T) / 2.0
        rho_t /= np.trace(rho_t)
        return rho_t

def compute_stabilizer_renyi_entropy(rho):
    """
    3. 量子资源层：单比特 SRE (Stabilizer Renyi Entropy)
    """
    I = np.eye(2)
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])
    paulis = [I, X, Y, Z]
    p_sum = 0.0
    for P in paulis:
        tr_val = np.real(np.trace(P @ rho))
        prob = (tr_val / np.sqrt(2.0)) ** 4
        p_sum += prob
    sre = -np.log2(p_sum)
    return max(0.0, float(sre))

def compute_parity_injection_error(rho, target_state):
    fid = np.real(np.trace(target_state @ rho))
    return max(0.0, 1.0 - fid)

def run_comprehensive_simulation():
    print("================== 综合分析版：马约拉纳库仑岛魔态校准仿真 ==================")
    physics_engine = MajoranaIslandPhysics(L=40, alpha=0.15, Delta=1.0, E_z=2.5)
    master_eq = CoulombBlockadeMasterEquation(E_C=0.8)
    
    target_psi = np.array([np.cos(np.pi/8), np.sin(np.pi/8)])
    target_rho = np.outer(target_psi, target_psi.conj())
    
    # 解除本征态锁定：采用叠加态初态
    psi_init = np.array([1.0, 1.0]) / np.sqrt(2.0)
    rho_init = np.outer(psi_init, psi_init.conj())
    
    true_v_bias = 0.15
    v_pulse_nominal = 0.50
    t_pulse = 0.5
    sample_gamma_phi = 0.05
    
    eps_M_sample = physics_engine.compute_bdg_splitting(v_pulse_nominal)
    
    # 1. 计算 1D Loss Landscape
    print("\n[计算] 正在生成 1D Loss 景观剖面...")
    delta_v_scan = np.linspace(-0.6, 0.6, 30)
    loss_scan = []
    for dv in delta_v_scan:
        rho_eval = master_eq.evolve_density_matrix(
            rho_init, v_pulse_nominal, true_v_bias + dv, t_pulse, sample_gamma_phi, eps_M_sample
        )
        loss_val = compute_parity_injection_error(rho_eval, target_rho)
        loss_scan.append(loss_val)
        
    # 2. 运行多噪声强度下的闭环寻优仿真
    print("\n--- 正在执行多噪声强度闭环校准扫描 ---")
    print(" gamma_phi | SRE_uncorr  SRE_opt | p_inj_uncorr  p_inj_opt | delta_V* (V)")
    print("---------------------------------------------------------------------------")
    
    gamma_phi_list = np.linspace(0.000, 0.200, 8)
    sre_uncorr_vec, sre_opt_vec = [], []
    p_inj_uncorr_vec, p_inj_opt_vec = [], []
    delta_v_opt_vec = []
    
    for gamma_phi_val in gamma_phi_list:
        eps_M_val = physics_engine.compute_bdg_splitting(v_pulse_nominal)
        
        rho_uncorr = master_eq.evolve_density_matrix(
            rho_init, v_pulse_nominal, true_v_bias, t_pulse, gamma_phi_val, eps_M_val
        )
        sre_uncorr = compute_stabilizer_renyi_entropy(rho_uncorr)
        p_inj_uncorr = compute_parity_injection_error(rho_uncorr, target_rho)
        
        def loss_function(delta_V_arr):
            delta_V = delta_V_arr[0]
            rho_eval = master_eq.evolve_density_matrix(
                rho_init, v_pulse_nominal, true_v_bias + delta_V, t_pulse, gamma_phi_val, eps_M_val
            )
            return compute_parity_injection_error(rho_eval, target_rho)
            
        res = minimize(loss_function, x0=[0.0], method='Nelder-Mead', options={'xatol': 1e-5, 'fatol': 1e-5})
        best_delta_V = res.x[0]
        
        rho_opt = master_eq.evolve_density_matrix(
            rho_init, v_pulse_nominal, true_v_bias + best_delta_V, t_pulse, gamma_phi_val, eps_M_val
        )
        sre_opt = compute_stabilizer_renyi_entropy(rho_opt)
        p_inj_opt = compute_parity_injection_error(rho_opt, target_rho)
        
        sre_uncorr_vec.append(sre_uncorr)
        sre_opt_vec.append(sre_opt)
        p_inj_uncorr_vec.append(p_inj_uncorr)
        p_inj_opt_vec.append(p_inj_opt)
        delta_v_opt_vec.append(best_delta_V)
        
        print(f"   {gamma_phi_val:6.3f} |     {sre_uncorr:6.4f}  {sre_opt:6.4f} |     {p_inj_uncorr*100:6.4f}%   {p_inj_opt*100:6.4f}% |     {best_delta_V:8.4f}")
        
    avg_error_uncorr = np.mean(p_inj_uncorr_vec) * 100
    avg_error_opt = np.mean(p_inj_opt_vec) * 100
    mean_delta_v = np.mean(delta_v_opt_vec)
    
    print("---------------------------------------------------------------------------")
    print(f"[完成] 闭环校准仿真结束。")
    print(f"未校准平均魔态注入错误率: {avg_error_uncorr:.2f}%")
    print(f"校准后平均魔态注入错误率: {avg_error_opt:.2f}%")
    
    # 3. 绘制三拼版综合分析大图
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
    
    # 子图 (a): 1D Loss Landscape
    ax1.plot(delta_v_scan, loss_scan, 'b.-', lw=1.5, label=f'Loss Landscape ($\gamma_\phi$={sample_gamma_phi})')
    ax1.axvline(x=0.0, color='r', linestyle='--', label='Target Offset')
    ax1.set_xlabel(r'Compensation Voltage $\delta V$ (V)')
    ax1.set_ylabel('Infidelity (Loss)')
    ax1.set_title('(a) 1D Loss Landscape')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(fontsize=9)
    
    # 子图 (b): SRE vs Dephasing
    ax2.plot(gamma_phi_list, sre_uncorr_vec, 'o--', color='#d95f02', label='Uncalibrated SRE')
    ax2.plot(gamma_phi_list, sre_opt_vec, 's-', color='#1b9e77', label='Optimized SRE')
    ax2.set_xlabel(r'Dephasing Rate $\gamma_{\phi}$')
    ax2.set_ylabel('Stabilizer Renyi Entropy (SRE)')
    ax2.set_title('(b) Magic Resource (SRE)')
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(fontsize=9)
    
    # 子图 (c): Infidelity vs Dephasing
    ax3.plot(gamma_phi_list, np.array(p_inj_uncorr_vec)*100, 'o--', color='#d95f02', label='Uncalibrated Infidelity')
    ax3.plot(gamma_phi_list, np.array(p_inj_opt_vec)*100, 's-', color='#1b9e77', label='Optimized Infidelity')
    ax3.axhline(1.0, color='red', linestyle=':', label='1% Distillation Threshold')
    ax3.set_xlabel(r'Dephasing Rate $\gamma_{\phi}$')
    ax3.set_ylabel('Physical Injection Error Rate (%)')
    ax3.set_title('(c) Injection Infidelity')
    ax3.grid(True, linestyle=':', alpha=0.6)
    ax3.legend(fontsize=9)
    
    plt.tight_layout()
    output_path = 'outputs/comprehensive_calibration_analysis.png'
    plt.savefig(output_path, dpi=300)
    print(f"[图表] 综合分析大图已成功生成并保存至: {output_path}")

if __name__ == '__main__':
    run_comprehensive_simulation()