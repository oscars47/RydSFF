## base file containing the quera aquila hamiltonian

import numpy as np
import qutip as qt

def get_h_ls(n_q):
    return np.random.uniform(0, 1, n_q)

def get_rand_x(a, eps_std, n_q):
    return [(i * a + np.random.normal(0, eps_std), np.random.normal(0, eps_std)) for i in range(n_q)]

I = qt.qeye(2)
nhat = (I - qt.sigmaz()) / 2

def get_J_arr(arr, n_qubits, C6=862690*2*np.pi):
    # print("arr", arr)
    J_ij = np.zeros((n_qubits, n_qubits))
    for i in range(n_qubits):
        for j in range(i+1, n_qubits):
            xi, yi = arr[i]
            xj, yj = arr[j]
            r = np.sqrt((xi - xj)**2 + (yi - yj)**2)
            # print("R", r)
            J_ij[i,j] = C6 / r**6
    return J_ij + J_ij.T

def H_int(J_arr, n_qubits):
    '''interaction Hamiltonian'''
    H = qt.tensor([I*0 for _ in range(n_qubits)])
    for i in range(n_qubits):
        for j in range(i+1, n_qubits):
            H += J_arr[i,j] * qt.tensor([nhat if k == i else I for k in range(n_qubits)]) * qt.tensor([nhat if k == j else I for k in range(n_qubits)])
    return H


def drive_main(neg_phi=True, ret_H_d=False):
    if neg_phi:
        xy_rabi = lambda phi: qt.Qobj([[0, np.exp(-1j*phi)], [np.exp(1j*phi), 0]]) ## NOTE NEW DEFINITION SWITCHING PHI -> -PHI RELATIVE TO THE MANUAL
    else: ## original definition
        xy_rabi = lambda phi: qt.Qobj([[0, np.exp(1j*phi)], [np.exp(-1j*phi), 0]])

    def _arrN(val, N):
        arr = np.asarray(val)
        if arr.ndim == 0:
            return np.full(N, float(arr))
        return arr

    def H_d_i(Omega, phi, Delta):
        '''drive Hamiltonian for single qubit'''
        return (Omega/2) * xy_rabi(phi) - Delta * nhat

    def H_d(Omega, phi, Delta_global, Delta_local, h_ls):
        '''drive Hamiltonian'''
        # print("h_ls", h_ls)
        N = len(h_ls)
        Omega = _arrN(Omega, N)
        phi = _arrN(phi, N)
        Delta_global = _arrN(Delta_global, N)
        Delta_local = _arrN(Delta_local, N)
        H = qt.tensor([I*0 for _ in range(N)])
        for i, h in enumerate(h_ls):
            Delta_i = Delta_global[i] + h * Delta_local[i]
            H += qt.tensor([H_d_i(Omega[i], phi[i], Delta_i) if k == i else I for k in range(N)])
        return H

    def H_pieces(Omega, phi, Delta_global, Delta_local, h_ls, J_arr):
        '''splits the Hamiltonian into drive and interaction parts given some coupling strengths.

        h_ls: list of local field strengths
        J_arr: array of coupling strengths where J_arr[i,j] is the coupling strength between qubits i and j. will be symmetric.
        
        '''
        return H_d(Omega, phi, Delta_global, Delta_local, h_ls), H_int(J_arr, len(h_ls))

    def get_H_indep(Omega, phi, Delta_global, Delta_local, h_ls, x=None, J_arr=None):
        assert not (x is None and J_arr is None), "Either x or J_arr must be provided"
        if J_arr is None:
            J_arr = get_J_arr(x, len(h_ls)) 
        H_d_part, H_int_part = H_pieces(Omega, phi, Delta_global, Delta_local, h_ls, J_arr)
        return H_d_part + H_int_part

    def get_H_ramp(p0, p1, x, h_ls, t_ramp_Omega, t_ramp_Delta, Omega_max_slope=250, Delta_slope_max=2500):
        """
        ramp from Hamiltonian defined by p0 to Hamiltonian defined by p1, where Omega and Delta are ramped in time t_tramp_Omega, t_ramp_Delta.

        NOTE: WHEN RETURNING HAMILTONIAN IT ASSUMES t STARTS AT 0 AND GOES TO t_ramp_Omega, t_ramp_Delta.

        """
        Omega0, phi0, Delta_global0, Delta_local0 = p0
        Omega1, phi1, Delta_global1, Delta_local1 = p1
        N = len(h_ls)
        Omega0 = _arrN(Omega0, N)
        Omega1 = _arrN(Omega1, N)
        Delta_global0 = _arrN(Delta_global0, N)
        Delta_global1 = _arrN(Delta_global1, N)
        Delta_local0 = _arrN(Delta_local0, N)
        Delta_local1 = _arrN(Delta_local1, N)
        phi0 = _arrN(phi0, N)

        Omega_slope = (Omega1 - Omega0) / t_ramp_Omega
        Delta_global_slope = (Delta_global1 - Delta_global0) / t_ramp_Delta
        Delta_local_slope = (Delta_local1 - Delta_local0) / t_ramp_Delta

        # assert np.abs(Omega_slope) <= Omega_max_slope, f"Omega slope {Omega_slope} exceeds maximum {Omega_max_slope}"
        # assert np.abs(Delta_global_slope) <= Delta_slope_max, f"Delta global slope {Delta_global_slope} exceeds maximum {Delta_slope_max}"
        # assert np.abs(Delta_local_slope) <= Delta_slope_max, f"Delta local slope {Delta_local_slope} exceeds maximum {Delta_slope_max}"

        # precompute J_arr
        J_arr = get_J_arr(x, len(h_ls)) 

        def H_ramp(t):
            if t > t_ramp_Omega:
                Omega_t = Omega1
            else:
                Omega_t = Omega0 + Omega_slope * t
            if t > t_ramp_Delta:
                Delta_global_t = Delta_global1
                Delta_local_t = Delta_local1
            else:
                Delta_global_t = Delta_global0 + Delta_global_slope * t
                Delta_local_t = Delta_local0 + Delta_local_slope * t
            return get_H_indep(Omega_t, phi0, Delta_global_t, Delta_local_t, h_ls, J_arr=J_arr) ## imediately switch to phi1

        return H_ramp
    if not ret_H_d:
        return get_H_indep, get_H_ramp
    else:
        return get_H_indep, get_H_ramp, H_d


if __name__ == "__main__":
    pass

