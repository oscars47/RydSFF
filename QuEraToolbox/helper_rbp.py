## helper functions for the process_rbp.py file
## merger of report_processing.py and rmt.py from old code
import numpy as np
# from QuEraToolbox.hamiltonian import get_H_ramp, get_H_indep
from QuEraToolbox.hamiltonian import drive_main
from QuEraToolbox.num_quantities import purity_num, survival_probability_num
import qutip as qt
import math
from tqdm import trange
import uncertainties.unumpy as unp
from uncertainties import ufloat

## get the hamiltonian functions from drive_main
get_H_indep, get_H_ramp = drive_main(neg_phi=True)

## ---- bitstrings
def bins_to_probs(bins, n_qubits,ret_shots=False):
    # print(bins)
    counts = np.bincount(bins, minlength=2**n_qubits)
    return counts / np.sum(counts) if not ret_shots else (counts / np.sum(counts), np.sum(counts))

def report_to_bins(report):
    """
    input is assumed to be list of key, value pairs
    where key is the bitstring and value is the count
    """
    bins =[]
    for r in report:
        bins += list(np.repeat(r[0], r[1]))
    return bins

def restrict_to_subsys(indices, qubits_A):
    """
    indices    : iterable of ints in [0, 2^n_qubits)
    n_qubits   : total number of qubits n_q
    qubits_A   : list of positions (each in [0, n_q)), length = n_A
                 positions are counted from LSB=0 up to MSB=n_q-1
    returns    : list of ints in [0, 2^n_A), each the restricted index i_A
    """
    out = []
    for i in indices:
        i_A = 0
        for j, pos in enumerate(qubits_A):
            bit = (i >> pos) & 1
            i_A |= (bit << j)
        out.append(i_A)
    return out

def restrict_probabilities(p, qubits_A):
    """
    Restrict a probability vector p over n_qubits to a subsystem A.

    Parameters
    ----------
    p : array-like of length 2^n_qubits
        Full probability distribution over computational basis states.
    qubits_A : list of ints
        Subsystem qubits to keep (positions counted from LSB=0).
    
    Returns
    -------
    p_A : np.ndarray of length 2^len(qubits_A)
        Marginal probability distribution over subsystem A.
    """
    p = np.asarray(p)
    n_A = len(qubits_A)
    p_A = np.zeros(2**n_A)

    for i, prob in enumerate(p):
        i_A = 0
        for j, pos in enumerate(qubits_A):
            bit = (i >> pos) & 1
            i_A |= (bit << j)
        p_A[i_A] += prob
    
    return p_A

def hamming_dist(i, j):
    return bin(i ^ j).count('1')

def get_hamming_matrix(n_qubits_A):
    hamming_matrix = np.zeros((2**n_qubits_A, 2**n_qubits_A))
    for i in range(2**n_qubits_A):
        for j in range(2**n_qubits_A):
            hamming_matrix[i,j] = hamming_dist(i, j)
    return hamming_matrix

def get_hamming_global(n_qubits_A):
    return np.ones((2**n_qubits_A, 2**n_qubits_A)) - np.eye(2**n_qubits_A)

def _sum_nominal(x):
    """Sum nominal values whether x contains floats or UFloats."""
    try:
        # Works for numpy arrays and lists
        return float(np.sum(unp.nominal_values(x)))
    except Exception:
        # Fallback: assume plain floats
        return float(np.sum(x))



def _to_nominal(arr):
    # Works for both numpy arrays and unumpy arrays
    return np.asarray(unp.nominal_values(arr), dtype=float)

def _normalize_prob(p, tol=1e-10):
    s = float(np.sum(p))
    if not np.isfinite(s) or s <= 0:
        print(f"Invalid probability vector; sum={s}")
        return None
    if not np.isclose(s, 1.0, atol=tol, rtol=0):
        p = p / s
    return p

def _cov_multinomial(p, n_shots):
    # Cov(p) = (diag(p) - p p^T)/n
    D = np.diag(p)
    return (D - np.outer(p, p)) / float(n_shots)

def _cov_dirichlet(alpha):
    # For Dirichlet(alpha): mean mu_i = alpha_i/alpha0
    # Cov(p) = [diag(mu) - mu mu^T]/(alpha0+1)
    alpha0 = np.sum(alpha)
    mu = alpha / alpha0
    return (np.diag(mu) - np.outer(mu, mu)) / (alpha0 + 1.0)

def _var_quadratic_form(A, p, Cov):
    # Var(p^T A p) with symmetric A
    Ap = A @ p
    g = 2.0 * Ap
    return float(g @ (Cov @ g))

def est_purity(report,
               num_qubits,
               qubits_A,
               epsilon_r,
               epsilon_g,
               is_bloqade,
               n_shots=None,
               hamming=None,
               shot_noise_model="multinomial",
               n_boot=500,
               random_state=None,
               verbose=False):
    """
    shot_noise_model in {"none","binomial","multinomial","dirichlet-jeffreys","dirichlet-laplace","bootstrap"}
    - When incl_shot_noise==False or shot_noise_model=="none": returns float
    - Otherwise returns ufloat with 1-sigma uncertainty on X*(2**n_A)
    """
    rng = np.random.default_rng(random_state)

    n_A = len(qubits_A)
    if hamming is None:
        hamming = get_hamming_matrix(n_A)

    # Symmetric A for the quadratic form
    A = ((-2.0) ** (-hamming))

    # --- Get probabilities (and shots if needed)
    if is_bloqade:
        bins = report_to_bins(report)
        if shot_noise_model != "none":
            probs, n_shots = bins_to_probs(bins, num_qubits, ret_shots=True)
            if verbose: print("NUM SHOTS:", n_shots)
        else:
            probs = bins_to_probs(bins, num_qubits)
            # n_shots = None
    else:
        probs = report
        # n_shots = None  # unless user passes counts in report, we don't know shots

    # --- Readout correction and subsystem restriction
    probs_corrected = apply_readout_noise(probs, epsilon_g, epsilon_r)
    P_jk = restrict_probabilities(probs_corrected, qubits_A)

    # Work with nominal probabilities for checks/analytics
    p = _to_nominal(P_jk)
    p = _normalize_prob(p, tol=1e-10)
    if p is None:
        print("Invalid probability vector after correction/restriction.")
        return None

    # --- Base purity (no uncertainty yet)
    X = float(p @ (A @ p))             # p^T A p
    X_scaled = X * (2.0 ** n_A)        # include 2^|A|

    if shot_noise_model in ("none", None):
        return X_scaled

    # --- Build Cov(p) or estimate via bootstrap
    model = shot_noise_model.lower()

    if model == "binomial":
        if n_shots is None:
            raise ValueError("Binomial model requires n_shots (use is_bloqade=True or provide shots).")
        # Independent binomial approx: Var(p_i) = p_i(1-p_i)/n, Cov=0 (rough)
        var = p * (1.0 - p) / float(n_shots)
        Cov = np.diag(var)

    elif model == "multinomial":
        if n_shots is None:
            raise ValueError("Multinomial model requires n_shots (use is_bloqade=True or provide shots).")
        Cov = _cov_multinomial(p, n_shots)

    elif model in ("dirichlet-jeffreys", "dirichlet-laplace"):
        if n_shots is None:
            raise ValueError("Dirichlet models need counts; call via Bloqade path to get shots.")
        # Reconstruct counts from p and n_shots (after correction and restriction we only have p;
        # this is an approximation—prefer passing raw counts if available).
        counts = np.round(p * float(n_shots)).astype(int)
        if model == "dirichlet-jeffreys":
            alpha = counts.astype(float) + 0.5
        else:  # laplace
            alpha = counts.astype(float) + 1.0
        Cov = _cov_dirichlet(alpha)

    elif model == "bootstrap":
        if n_shots is None:
            raise ValueError("Bootstrap model requires n_shots (use is_bloqade=True or provide shots).")
        # Fast multinomial resampling around p
        X_boot = np.empty(n_boot, dtype=float)
        for b in range(n_boot):
            draw = rng.multinomial(n_shots, p)
            pb = draw / float(n_shots)
            X_boot[b] = float(pb @ (A @ pb))
        mu = float(np.mean(X_boot))
        sig = float(np.std(X_boot, ddof=1))
        # Scale after variance
        return ufloat(mu * (2.0 ** n_A), sig * (2.0 ** n_A))

    else:
        raise ValueError(f"Unknown shot_noise_model='{shot_noise_model}'")

    # --- Analytic variance for quadratic form
    var_X = _var_quadratic_form(A, p, Cov)         # variance of p^T A p
    sig_X = float(np.sqrt(max(var_X, 0.0))) # for num stability
    # print("sig_X:", sig_X)
    return ufloat(X_scaled, sig_X * (2.0 ** n_A))    # scaling already in X_scaled

def est_fidelity(report1, report2, n_qubits, epsilon_r, epsilon_g, is_bloqade, hamming=None, incl_shot_noise=False):
    if hamming is None:
        hamming = get_hamming_matrix(n_qubits)
    
    A = ((-2)**(-hamming))  

    if is_bloqade:
        # convert to probability vector
        bins1 = report_to_bins(report1)
        bins2 = report_to_bins(report2)
        
        if incl_shot_noise:
            probs1, n_shots1 = bins_to_probs(bins1, n_qubits, ret_shots=True)
            probs2, n_shots2 = bins_to_probs(bins2, n_qubits, ret_shots=True)
        else:
            probs1 = bins_to_probs(bins1, n_qubits)
            probs2 = bins_to_probs(bins2, n_qubits)
    else:
        probs1 = report1
        probs2 = report2

    P_jk_1 = apply_readout_noise(probs1, epsilon_g, epsilon_r)
    P_jk_2 = apply_readout_noise(probs2, epsilon_g, epsilon_r)

    if incl_shot_noise and is_bloqade:
        sig_P1 = np.sqrt(P_jk_1 / float(n_shots1))
        sig_P2 = np.sqrt(P_jk_2 / float(n_shots2))
        P_jk_1 = unp.uarray(P_jk_1, sig_P1)  # unumpy array of ufloat elements
        P_jk_2 = unp.uarray(P_jk_2, sig_P2)  # unumpy array of ufloat elements

        

    if np.isclose(_sum_nominal(P_jk_1), 1, 1e-10) and np.isclose(_sum_nominal(P_jk_2), 1, 1e-10):
        # X = 0
        # for i in range(2**n_qubits):
        #     for j in range(2**n_qubits):
        #         X += P_jk_1[i] * P_jk_2[j] * A[i, j] 
        # return X * (2**n_qubits)  # include the 2^NA prefactor
        X = P_jk_1 @ (A @ P_jk_2)
        return X * (2 ** n_qubits)
    else:
        print(f"Probabilities do not sum to 1, sum1: {sum(P_jk_1)}, sum2: {sum(P_jk_2)}")
        return None


## ---- probability testing

def get_evolved_trapezoid_kink(h_ls, x, t_plateau_ls, base_params, Delta_mean, Delta_local, min_dt=0.05, J_arr=None, start_Delta_from0=False, Delta_max_slope = 2500, uniform_Omega_Delta_ramp=False, Delta_local_ramp_time=0.05, Omega_delay_time=0.0):
    Omega = base_params['ev_params']['Omega']
    phi = base_params['ev_params']['phi']

    ev_params = base_params['ev_params']
    t_ramp = ev_params['t_ramp']

    Delta_global = Delta_mean - 1/2 * Delta_local

    # frac = min_dt / t_ramp 
    # Omega_slope = Omega / t_ramp
    # assert Omega_delay_time == 0.0, "Omega_delay_time not implemented yet"
   

    if not uniform_Omega_Delta_ramp:
        # Delta_local_val = Delta_max_slope*min_dt if np.abs(Delta_local) > Delta_max_slope*min_dt else Delta_local
        if start_Delta_from0:
            # p0_up = (0.0, 0, 0.0, 0.0) # phi must start at 0!!
            # Delta_global_val = Delta_max_slope*min_dt if np.abs(Delta_global) > Delta_max_slope*min_dt else Delta_global
            
            # p1_up = (Omega_slope*min_dt, phi, Delta_global_val, Delta_local_val)
            raise NotImplementedError("start_Delta_from0=True not implemented")
        else:
            p0_up = (0.0, 0.0, Delta_global, 0.0) # Delta_local constrained to start at 0
            p1_up = (0, phi, Delta_global, Delta_local)
            p2_up = (Omega, phi, Delta_global, Delta_local)
    else:
        # Delta_global_slope = Delta_global / t_ramp
        # Delta_local_slope = Delta_local / t_ramp
        # if start_Delta_from0:
        #     p0_up = (0.0, 0, 0.0, 0.0) # phi must start at 0!!
        #     p1_up = (Omega_slope*min_dt, phi, Delta_global_slope * min_dt, Delta_local_slope * min_dt)
        # else:
        #     p0_up = (0.0, 0.0, Delta_global, 0.0) 
        #     p1_up = (Omega_slope*min_dt, phi, Delta_global, Delta_local_slope * min_dt)
        raise NotImplementedError("uniform_Omega_Delta_ramp=True not implemented")
            

    psi0 = qt.tensor([qt.basis(2, 0) for _ in range(len(h_ls))])  # initial state |0...0>

    H_up_phi_init = get_H_ramp(p0_up, p1_up, x, h_ls, Delta_local_ramp_time, Delta_local_ramp_time)
    psi_up_phi_init = qt.sesolve(H_up_phi_init, psi0, [0, Delta_local_ramp_time], options={'nsteps':100000}).states[-1]

    H_up_rest = get_H_ramp(p1_up, p2_up, x, h_ls, t_ramp, t_ramp)
    psi_up = qt.sesolve(H_up_rest, psi_up_phi_init, [0, t_ramp], options={'nsteps':100000}).states[-1]
    # now the plateau
    H_plateau = get_H_indep(Omega, phi, Delta_global, Delta_local, h_ls, x=x)
    
    psi_t_ls = qt.sesolve(
        H_plateau, psi_up, t_plateau_ls,options={'nsteps':100000}
    ).states

    # exact reverse of ramp up with kink  
    H_ramp_down_first = get_H_ramp(p2_up, p1_up, x, h_ls, t_ramp, t_ramp)
    H_ramp_down = get_H_ramp(p1_up, p0_up, x, h_ls, Delta_local_ramp_time, Delta_local_ramp_time) 

    U_down_first = qt.propagator(H_ramp_down_first, [0, t_ramp])[-1]
    U_down = qt.propagator(H_ramp_down, [0, Delta_local_ramp_time])[-1]
    U_down = U_down * U_down_first

    for psi in psi_t_ls:
        psi = U_down * psi
    return psi_t_ls
    

def get_psi_t_ls(h_ls, x, t_plateau_ls, base_params, Delta_mean, Delta_local, time_dep=True, show_progress=False, start_Delta_from0=False, uniform_Omega_Delta_ramp=False, Delta_local_ramp_time=0.05, Omega_delay_time=0.0):
    if len(t_plateau_ls) ==1:
        # print("ONLY ONE TIME POINT")
        t_plateau_ls = [0, t_plateau_ls[0]]
    if time_dep:
        psi_t_ls = get_evolved_trapezoid_kink(h_ls, x, t_plateau_ls, base_params, Delta_mean, Delta_local, start_Delta_from0=start_Delta_from0, uniform_Omega_Delta_ramp=uniform_Omega_Delta_ramp, Delta_local_ramp_time=Delta_local_ramp_time, Omega_delay_time=Omega_delay_time)
    else:
        # print("solving non time dependent")
        if not show_progress:
            psi_t_ls = qt.sesolve(
                get_H_indep(base_params['ev_params']['Omega'], base_params['ev_params']['phi'], Delta_mean - 1/2 * Delta_local, Delta_local, h_ls, x=x),
                qt.tensor([qt.basis(2, 0) for _ in range(len(h_ls))]), t_plateau_ls, options={'nsteps':100000}
            ).states
        else:
            psi_t_ls = qt.sesolve(
                get_H_indep(base_params['ev_params']['Omega'], base_params['ev_params']['phi'], Delta_mean - 1/2 * Delta_local, Delta_local, h_ls, x=x),
                qt.tensor([qt.basis(2, 0) for _ in range(len(h_ls))]), t_plateau_ls, options={'nsteps':100000}, progress_bar='tqdm'
            ).states
    return psi_t_ls

def get_ee(h_ls, x, t_plateau_ls, base_params, Delta_mean, Delta_local, time_dep=True, start_Delta_from0=False, uniform_Omega_Delta_ramp=False, Delta_local_ramp_time=0.05, Omega_delay_time=0.0):

    psi_t_ls = get_psi_t_ls(h_ls, x, t_plateau_ls, base_params, Delta_mean, Delta_local, time_dep=time_dep, start_Delta_from0=start_Delta_from0, uniform_Omega_Delta_ramp=uniform_Omega_Delta_ramp, Delta_local_ramp_time=Delta_local_ramp_time, Omega_delay_time=Omega_delay_time)
    # print(psi_t_ls)
    if len(t_plateau_ls) ==1:
        psi_t_ls = [psi_t_ls[-1]]

    # use default partition of half system size
    n_qubits = len(h_ls)
    n_A = n_qubits // 2
    # n_A = n_qubits
    qubits_A = list(range(n_A))  # all qubits
    return -np.log(purity_num(psi_t_ls, qubits_A))

def get_sp(h_ls, x, t_plateau_ls, base_params, Delta_mean, Delta_local, time_dep=True, start_Delta_from0=False, uniform_Omega_Delta_ramp=False, Delta_local_ramp_time=0.05, Omega_delay_time=0.0):
    psi_t_ls = get_psi_t_ls(h_ls, x, t_plateau_ls, base_params, Delta_mean, Delta_local, time_dep=time_dep, start_Delta_from0=start_Delta_from0, uniform_Omega_Delta_ramp=uniform_Omega_Delta_ramp, Delta_local_ramp_time=Delta_local_ramp_time, Omega_delay_time=Omega_delay_time)
    
    if time_dep:
        # print("time dependent inside get_sp")
        psi0 = get_evolved_trapezoid_kink(h_ls, x, [0], base_params, Delta_mean, Delta_local, uniform_Omega_Delta_ramp=uniform_Omega_Delta_ramp)[-1] # initial state of plateau time 0
        # print("psi0:", psi0)
        
    else:
        psi0 = qt.tensor([qt.basis(2, 0) for _ in range(len(h_ls))])  # initial state |0...0>

    # compute the fidelity to unevolved state
    return survival_probability_num(psi_t_ls, len(h_ls), rho0=psi0)


## ---- correct observed probabilities
def apply_readout_channel(p_true, epsilon_r=0.1, epsilon_g=0.05):
    """
    Forward map (true -> observed) under independent single-qubit readout errors.
    epsilon_r: P(report 0 | true 1)
    epsilon_g: P(report 1 | true 0)
    Accepts counts or probabilities (normalizes internally). Returns probabilities.
    """
    p_true = np.asarray(p_true, dtype=float)
    p_true = p_true / p_true.sum()
    N = p_true.size
    n = int(np.log2(N))
    assert (1 << n) == N, "Length of distribution must be a power of 2."

    A = np.array([[1 - epsilon_g, epsilon_r],
                  [epsilon_g,     1 - epsilon_r]], dtype=float)

    P = p_true.reshape([2]*n)
    for axis in range(n):
        # A acts on that axis
        P = np.tensordot(A, P, axes=([1], [axis]))
        P = np.moveaxis(P, 0, axis)
    return P.reshape(N)

def correct_readout_probs(p_obs, epsilon_r=0.1, epsilon_g=0.05, clip=True):
    """
    Inverse map (observed -> estimated true) by applying (A^{-1})^{⊗ n}.
    epsilon_r: P(report 0 | true 1)
    epsilon_g: P(report 1 | true 0)
    Accepts counts or probabilities (normalizes internally). Returns probabilities.
    If clip=True, tiny negative values from numerical noise are clipped to 0 then renormalized.
    """
    p_obs = np.asarray(p_obs, dtype=float)
    p_obs = p_obs / p_obs.sum()
    N = p_obs.size
    n = int(np.log2(N))
    assert (1 << n) == N, "Length of distribution must be a power of 2."

    det = 1.0 - epsilon_g - epsilon_r
    if abs(det) < 1e-12:
        raise ValueError("Readout matrix is singular/ill-conditioned (ε_g + ε_r ≈ 1).")

    Ainv = (1.0/det) * np.array([[1 - epsilon_r, -epsilon_r],
                                 [-epsilon_g,     1 - epsilon_g]], dtype=float)

    P = p_obs.reshape([2]*n)
    for axis in range(n):
        P = np.tensordot(Ainv, P, axes=([1], [axis]))
        P = np.moveaxis(P, 0, axis)

    p_true = P.reshape(N)

    if clip:
        p_true = np.maximum(p_true, 0.0)
        s = p_true.sum()
        if s > 0:
            p_true /= s
    return p_true

if __name__ == "__main__":
    pass
    
    # get_psi_t_ls(h_ls, x, t_plateau_ls, base_params, Delta_mean, Delta_local, time_dep=True)