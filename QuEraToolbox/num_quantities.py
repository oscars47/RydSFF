## computes everything given numerical density matrix
import numpy as np
import qutip as qt
from tqdm import trange, tqdm

def purity_num(psi_ls, qubits_A):
    purity_ls = np.zeros(len(psi_ls))
    for i, psi in enumerate(psi_ls):
        psi = qt.ket2dm(psi)
        psi_A = psi.ptrace(qubits_A)
        purity_ls[i]=(psi_A**2).tr()
    return purity_ls

def survival_probability_num(psi_ls, n_qubits, rho0=None):
    if rho0 is None:
        rho0 = qt.ket2dm(qt.tensor([qt.basis(2, 0) for _ in range(n_qubits)]))
    else:
        rho0 = qt.ket2dm(rho0)
        
    survival_prob_ls = np.zeros(len(psi_ls))
    for i, psi in enumerate(psi_ls):
        psi = qt.ket2dm(psi)
        survival_prob_ls[i] = (rho0 * psi).tr().real
    return survival_prob_ls

def renyi_t_indep(H, N, t_ls, qubits_A=None):
    """
    compute von neumann entropy for given bipartition (trace out complement of qubits_A), evolving the |0>^{otimes N} state
    """
    if qubits_A is None:
        qubits_A = list(range(N//2))
    rho0 = qt.ket2dm(qt.tensor([qt.basis(2, 0) for _ in range(N)]))
    rho_ls = qt.sesolve(H, rho0, t_ls, options={'nsteps':100000}).states
    return [-np.log((qt.ptrace(rho, qubits_A)**2).tr()) for rho in tqdm(rho_ls)]
    

def eigen_quantities(eigenvals, N, t_ls):

    # order the eigenvalues
    eigenvals = np.sort(eigenvals)
    level_spacing = np.diff(eigenvals)
    level_spacing_ratios = level_spacing[1:] / level_spacing[:-1]

    # sff
    d = 2**N
    phases = np.exp(1j * np.outer(eigenvals, t_ls))   
    Z = phases.sum(axis=0)                           
    # Spectral form factor (un-normalized)
    sff = np.abs(Z)**2  / d**2
    return eigenvals, level_spacing_ratios, sff                             