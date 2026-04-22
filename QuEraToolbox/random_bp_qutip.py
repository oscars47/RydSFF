## qutip version of random_bp_prep.py; drawing from quera_time_check.py
## first do positive time evolution, -phi
import numpy as np
import qutip as qt
from QuEraToolbox.hamiltonian import drive_main
from copy import deepcopy
from tqdm import tqdm

min_dt = 0.05
_Omega_slope = 250.0
_Delta_slope = 2500.0

# \left(\frac{1}{3.8}+\frac{x}{2\pi\left(57\right)}\right)^{-1}

_T2 = lambda Delta_local: (1/3.8 + Delta_local/(2*np.pi*57))**(-1)


def get_probs_seq_ls(h_ls, x, t_plateau_ls, seq_ls_pre, base_params, Delta_mean, Delta_local, in_gate_params, neg_phi=True, preset_opt=None, override_local=False, start_Delta_from0=False, uniform_Omega_Delta_ramp=False, phi_mode='binary', ret_probs=True, ret_continuous=False, n_continuous_steps=100, return_last_state=False, skip_to_this_state=None, local_haar=False, indep_haar=True, include_T2=False, Delta_local_ramp_time=0.05, Omega_delay_time=0, continuous_gates_only=False):
    """
    Helper function to create collapse operators for T2 dephasing
    """
    def get_c_ops(Delta_local_val, h_ls):
        if not include_T2:
            return []
        N = len(h_ls)
        c_ops = []
        for i in range(N):
            T2 = _T2(abs(Delta_local_val*h_ls[i]))
            gamma_i = 1.0 / T2
            sz_i = qt.tensor([qt.sigmaz() if k==i else qt.qeye(2) for k in range(N)])
            c_ops.append(np.sqrt(gamma_i) * sz_i)
        return c_ops
    """
    returns list of probabilities after all seq in seq_ls
    
    Parameters:
    -----------
    ret_continuous : bool, default False
        If True, returns (probs_all_t, times_list, probs_list, states_list) where the last 3 are continuous evolution
    n_continuous_steps : int, default 100
        Number of time steps to capture during continuous evolution (only used if ret_continuous=True)
    return_last_state : bool, default False
        If True, returns the final state even when ret_continuous=False
    continuous_gates_only : bool, default False
        If True (and ret_continuous=True), only records states during the gate sequence itself,
        skipping the ramp-up and plateau evolution. Time is reset to 0 at the start of the gates.
    
    Returns:
    --------
    If ret_continuous=False and return_last_state=False: probs_all_t
    If ret_continuous=False and return_last_state=True: (probs_all_t, final_states)
    If ret_continuous=True: (probs_all_t, times_list, probs_list, states_list)
        - times_list: chronological list of all time points
        - probs_list: list of probability arrays at each time point  
        - states_list: list of QuTip states at each time point
        - final_states: list of final states for each sequence (when return_last_state=True)
    """

    # assert local_haar == True

    # print("inside qutip random", seq_ls_pre)
    print("neg_phi:", neg_phi)

    get_H_indep, get_H_ramp = drive_main(neg_phi=neg_phi)

    ev_params = base_params['ev_params']
    Omega = ev_params['Omega']
    phi = ev_params['phi']
    try:
        Delta_global = Delta_mean - 1/2 * Delta_local
    except:
        Delta_global = Delta_mean - 1/2 * np.array(Delta_local)
        
    t_ramp = ev_params['t_ramp']

    print("Omega:", Omega, "phi:", phi, "Delta_global:", Delta_global, "Delta_local:", Delta_local)

    if override_local:
        Delta_local = 0

    H_plateau = get_H_indep(Omega, phi, Delta_global, Delta_local, h_ls, x=x)

    if not uniform_Omega_Delta_ramp:
        try:
            t_ramp_Delta0 = max([min_dt, abs(Delta_global) / _Delta_slope, abs(Delta_local) / _Delta_slope, Delta_local_ramp_time])
        except:
            t_ramp_Delta0 = 0.05
    else:
        t_ramp_Delta0 = t_ramp

    if start_Delta_from0:
        p0_up = (0.0, 0.0, 0.0, 0.0)
    else:
        p0_up = (0.0, 0.0, Delta_global, 0.0)

    p_up_after_delta = (0.0, phi, Delta_global, Delta_local)
    p1_up_afterphi = (Omega, phi, Delta_global, Delta_local)
    print("t_Delta0:", t_ramp_Delta0)

     # Store continuous evolution data if requested
    times_list = [] if ret_continuous else None
    probs_list = [] if ret_continuous else None  
    states_list = [] if ret_continuous else None
    current_time = 0.0 if ret_continuous else None
    _record_continuous = ret_continuous and not continuous_gates_only  # False until gates start when continuous_gates_only

    if skip_to_this_state is None:

        psi0 = qt.tensor([qt.basis(2, 0) for _ in range(len(h_ls))])  # initial state |0...0>
        
        H_up_delta = get_H_ramp(p0_up, p_up_after_delta, x, h_ls, t_ramp_Delta0, t_ramp_Delta0)
        c_ops_plateau = get_c_ops(Delta_local, h_ls)  # Use plateau Delta_local for ramp up

        if ret_continuous:
            t_list_delta = np.linspace(0, t_ramp_Delta0, n_continuous_steps)
            result_delta = qt.mesolve(H_up_delta, psi0, t_list_delta, c_ops_plateau, options={'nsteps':100000}) if include_T2 else qt.sesolve(H_up_delta, psi0, t_list_delta, options={'nsteps':100000})
            if _record_continuous:
                for t, state in zip(t_list_delta, result_delta.states):
                    times_list.append(current_time + t)
                    states_list.append(state)
                    probs = np.abs(state.full().flatten())**2
                    probs /= np.sum(probs)
                    probs_list.append(probs)
            current_time += t_ramp_Delta0
            psi_after_delta = result_delta.states[-1]
        else:
            result = qt.mesolve(H_up_delta, psi0, [0, t_ramp_Delta0], c_ops_plateau, options={'nsteps':100000}) if include_T2 else qt.sesolve(H_up_delta, psi0, [0, t_ramp_Delta0], options={'nsteps':100000})
            psi_after_delta = result.states[-1]

        if Omega_delay_time is not None and Omega_delay_time >= 0:
            H_delay = get_H_indep(0.0, phi, Delta_global, Delta_local, h_ls, x=x)
            if ret_continuous:
                t_list_delay = np.linspace(0, Omega_delay_time, n_continuous_steps)
                result_delay = qt.mesolve(H_delay, psi_after_delta, t_list_delay, c_ops_plateau, options={'nsteps':100000}) if include_T2 else qt.sesolve(H_delay, psi_after_delta, t_list_delay, options={'nsteps':100000})
                if _record_continuous:
                    for t, state in zip(t_list_delay, result_delay.states):
                        times_list.append(current_time + t)
                        states_list.append(state)
                        probs = np.abs(state.full().flatten())**2
                        probs /= np.sum(probs)
                        probs_list.append(probs)
                current_time += Omega_delay_time
                psi_after_delay = result_delay.states[-1]
            else:
                if include_T2:
                    result = qt.mesolve(H_delay, psi_after_delta, [0, Omega_delay_time], c_ops_plateau, options={'nsteps':100000})
                    psi_after_delay = result.states[-1]
                else:
                    U_delay = (-1j * H_delay * Omega_delay_time).expm()
                    psi_after_delay = U_delay * psi_after_delta
        else:
            psi_after_delay = psi_after_delta

        H_up_omega = get_H_ramp(p_up_after_delta, p1_up_afterphi, x, h_ls, t_ramp, t_ramp)

        if ret_continuous:
            t_list_omega = np.linspace(0, t_ramp, n_continuous_steps)
            result_omega = qt.mesolve(H_up_omega, psi_after_delay, t_list_omega, c_ops_plateau, options={'nsteps':100000}) if include_T2 else qt.sesolve(H_up_omega, psi_after_delay, t_list_omega, options={'nsteps':100000})
            if _record_continuous:
                for t, state in zip(t_list_omega, result_omega.states):
                    times_list.append(current_time + t)
                    states_list.append(state)
                    probs = np.abs(state.full().flatten())**2
                    probs /= np.sum(probs)
                    probs_list.append(probs)
            current_time += t_ramp
            psi_up = result_omega.states[-1]
        else:
            result = qt.mesolve(H_up_omega, psi_after_delay, [0, t_ramp], c_ops_plateau, options={'nsteps':100000}) if include_T2 else qt.sesolve(H_up_omega, psi_after_delay, [0, t_ramp], options={'nsteps':100000})
            psi_up = result.states[-1]
    else:
        psi_up = skip_to_this_state

    def before_gates(t_plateau, start_time):
        # now the plateau
        if ret_continuous:
            t_list_plateau = np.linspace(0, t_plateau, n_continuous_steps)
            result_plateau = qt.mesolve(H_plateau, psi_up, t_list_plateau, c_ops_plateau, options={'nsteps':100000}) if include_T2 else qt.sesolve(H_plateau, psi_up, t_list_plateau, options={'nsteps':100000})
            # Add to continuous lists only if not gates-only mode
            if _record_continuous:
                for t, state in zip(t_list_plateau, result_plateau.states):
                    times_list.append(start_time + t)
                    states_list.append(state)
                    probs = np.abs(state.full().flatten())**2
                    probs /= np.sum(probs)
                    probs_list.append(probs)
            return result_plateau.states[-1]
        else:
            if include_T2:
                result = qt.mesolve(H_plateau, psi_up, [0, t_plateau], c_ops_plateau, options={'nsteps':100000})
                return result.states[-1]
            else:
                U_plateau = (-1j * H_plateau * t_plateau).expm()
                return U_plateau * psi_up
    

    def get_U_gate(phi, gate_params, gate_time, h_ls, x):
        return (-1j* gate_time * get_H_indep(gate_params[1]['Omega'], phi, gate_params[1]['Delta_global'], gate_params[1]['Delta_local'], h_ls, x=x)).expm()
        
    
    def gates_prep():
        # populate the gate parameters
        gate_params = {
            1: {
                'phi': 0,
                'Omega': 15.8,
            },
            2: {
                'phi': np.pi/2,
                'Omega': 15.8,
            },
        }

        gate_params[1]['Delta_global'] = in_gate_params['Delta_global']
        gate_params[1]['Delta_local'] = in_gate_params['Delta_local']
        gate_params[2]['Delta_global'] = in_gate_params['Delta_global']
        gate_params[2]['Delta_local'] = in_gate_params['Delta_local']

        # gate time
        gate_time = in_gate_params['gate_duration']

        # get the gate unitaries
        if phi_mode == 'binary':
            U_1 = get_U_gate(0, gate_params, gate_time, h_ls, x)
            U_2 = get_U_gate(np.pi/2, gate_params, gate_time, h_ls, x)
        else:
            # return lambda functions that depend on phi
            U_1 = lambda phi: get_U_gate(phi, gate_params, gate_time, h_ls, x)
            U_2 = None
            
        
        # gotta ramp to the gate Delta_global and Delta_local
        p0 = (Omega, phi, Delta_global, Delta_local)
        p1 = (Omega, phi, gate_params[1]['Delta_global'], gate_params[1]['Delta_local'])

        try:
            Delta_global_delta = Delta_global - gate_params[1]['Delta_global']
            Delta_local_delta = Delta_local - gate_params[1]['Delta_local']
            t_ramp_Delta = max(min_dt, abs(Delta_global_delta) / _Delta_slope, abs(Delta_local_delta) / _Delta_slope, Delta_local_ramp_time)
        except:
            t_ramp_Delta = 0.05

        H_ramp_to_init_gate = get_H_ramp(p0, p1, x, h_ls, t_ramp_Delta, t_ramp_Delta)

        U_ramp_to_init_gate = qt.propagator(H_ramp_to_init_gate, [0, t_ramp_Delta])[-1]

        return U_ramp_to_init_gate, U_1, U_2
    
    # now we ramp down the detunings in time t_ramp, reverse of the beginning
    # while ramping down, put phi back to original value
    p1_after_phi_gates = (Omega, phi, in_gate_params['Delta_global'], in_gate_params['Delta_local'])


    # H_ramp_down_first = get_H_ramp(p1_after_phi_gates, p1_up, x, h_ls, t_ramp, t_ramp_Delta0)
    # going from gate detunings back to p0, 
    if not uniform_Omega_Delta_ramp:
        
        # Delta_global_diff = in_gate_params['Delta_global'] - p0_up[2]
        # Delta_local_diff = in_gate_params['Delta_local'] - p0_up[3]
        # t_ramp_Delta = max(min_dt, abs(Delta_global_diff) / _Delta_slope, abs(Delta_local_diff) / _Delta_slope, Delta_local_ramp_time)
        t_ramp_Delta = 0.05
    else:
        t_ramp_Delta = t_ramp
    
    p_after_omega_down = (0.0, phi, in_gate_params['Delta_global'], in_gate_params['Delta_local'])
    p_after_delta_down = (0.0, p0_up[1], p0_up[2], p0_up[3])

    H_ramp_down_omega = get_H_ramp(p1_after_phi_gates, p_after_omega_down, x, h_ls, t_ramp, t_ramp)
    H_ramp_down_delta = get_H_ramp(p_after_omega_down, p_after_delta_down, x, h_ls, t_ramp_Delta, t_ramp_Delta)

    U_ramp_down_omega = qt.propagator(H_ramp_down_omega, [0, t_ramp])[-1]
    U_ramp_down_delta = qt.propagator(H_ramp_down_delta, [0, t_ramp_Delta])[-1]
    U_ramp_down = U_ramp_down_delta * U_ramp_down_omega

    if phi_mode == 'binary':
        def gates(psi_gate, U_1, U_2, seq, start_time):
            # now we compose the sequence; only changing phi
            # print("seq:", seq)
            psi_gate = deepcopy(psi_gate)
            gate_start_time = start_time
            
            if ret_probs and not ret_continuous:
                if include_T2:
                    gate_time = in_gate_params['gate_duration']
                    for s in seq:
                        if s == 1:
                            H_gate = get_H_indep(15.8, 0, in_gate_params['Delta_global'], in_gate_params['Delta_local'], h_ls, x=x)
                        elif s == 2:
                            H_gate = get_H_indep(15.8, np.pi/2, in_gate_params['Delta_global'], in_gate_params['Delta_local'], h_ls, x=x)
                        else:
                            raise ValueError("Unknown gate in sequence:", s)
                        result_gate = qt.mesolve(H_gate, psi_gate, [0, gate_time], c_ops_gate, options={'nsteps':100000})
                        psi_gate = result_gate.states[-1]
                    # finally ramp down with T2
                    result_omega = qt.mesolve(H_ramp_down_omega, psi_gate, [0, t_ramp], c_ops_gate, options={'nsteps':100000})
                    psi_gate = result_omega.states[-1]
                    result_delta = qt.mesolve(H_ramp_down_delta, psi_gate, [0, t_ramp_Delta], c_ops_gate, options={'nsteps':100000})
                    psi_gate = result_delta.states[-1]
                else:
                    for s in seq:
                        if s == 1:
                            psi_gate = U_1 * psi_gate
                        elif s == 2:
                            psi_gate = U_2 * psi_gate
                        else:
                            raise ValueError("Unknown gate in sequence:", s)
                    # finally ramp down
                    psi_gate = U_ramp_down * psi_gate
                return psi_gate
            elif ret_continuous:
                # For continuous evolution, we need to use sesolve instead of unitary evolution
                gate_time = in_gate_params['gate_duration']
                
                for i, s in enumerate(seq):
                    if s == 1:
                        H_gate = get_H_indep(15.8, 0, in_gate_params['Delta_global'], in_gate_params['Delta_local'], h_ls, x=x)
                    elif s == 2:
                        H_gate = get_H_indep(15.8, np.pi/2, in_gate_params['Delta_global'], in_gate_params['Delta_local'], h_ls, x=x)
                    else:
                        raise ValueError("Unknown gate in sequence:", s)
                    
                    t_list_gate = np.linspace(0, gate_time, n_continuous_steps)
                    result_gate = qt.mesolve(H_gate, psi_gate, t_list_gate, c_ops_gate, options={'nsteps':100000}) if include_T2 else qt.sesolve(H_gate, psi_gate, t_list_gate, options={'nsteps':100000})
                    # Add to continuous lists
                    for t, state in zip(t_list_gate, result_gate.states):
                        times_list.append(gate_start_time + t)
                        states_list.append(state)
                        probs = np.abs(state.full().flatten())**2
                        probs /= np.sum(probs)
                        probs_list.append(probs)
                    gate_start_time += gate_time
                    psi_gate = result_gate.states[-1]
                
                t_list_ramp_down_omega = np.linspace(0, t_ramp, n_continuous_steps)
                result_ramp_down_omega = qt.mesolve(H_ramp_down_omega, psi_gate, t_list_ramp_down_omega, c_ops_gate, options={'nsteps':100000}) if include_T2 else qt.sesolve(H_ramp_down_omega, psi_gate, t_list_ramp_down_omega, options={'nsteps':100000})
                for t, state in zip(t_list_ramp_down_omega, result_ramp_down_omega.states):
                    times_list.append(gate_start_time + t)
                    states_list.append(state)
                    probs = np.abs(state.full().flatten())**2
                    probs /= np.sum(probs)
                    probs_list.append(probs)
                gate_start_time += t_ramp
                psi_gate = result_ramp_down_omega.states[-1]

                t_list_ramp_down_delta = np.linspace(0, t_ramp_Delta, n_continuous_steps)
                result_ramp_down_delta = qt.mesolve(H_ramp_down_delta, psi_gate, t_list_ramp_down_delta, c_ops_gate, options={'nsteps':100000}) if include_T2 else qt.sesolve(H_ramp_down_delta, psi_gate, t_list_ramp_down_delta, options={'nsteps':100000})
                for t, state in zip(t_list_ramp_down_delta, result_ramp_down_delta.states):
                    times_list.append(gate_start_time + t)
                    states_list.append(state)
                    probs = np.abs(state.full().flatten())**2
                    probs /= np.sum(probs)
                    probs_list.append(probs)
                psi_gate = result_ramp_down_delta.states[-1]
                
                return psi_gate
            else:
                state_ls = []
                print("BEFORE STATE", psi_gate)
                print("seq-----//:")
                for s in seq:
                    print(s)
                    if s == 1:
                        psi_gate = U_1 * psi_gate
                    elif s == 2:
                        psi_gate = U_2 * psi_gate
                    else:
                        raise ValueError("Unknown gate in sequence:", s)
                    state_ls.append(psi_gate)

                
                # finally ramp down
                psi_gate = U_ramp_down * psi_gate
                state_ls.append(psi_gate)
                print("AFTER STATE", psi_gate)
                return state_ls
    else:
        def gates(psi_gate, U_1_func, U_2, seq, start_time):
            # now we compose the sequence; only changing phi
            # print("seq:", seq)
            gate_start_time = start_time
            
            if ret_probs and not ret_continuous:
                if include_T2:
                    gate_time = in_gate_params['gate_duration']
                    for phi_val in seq:
                        H_gate = get_H_indep(15.8, phi_val, in_gate_params['Delta_global'], in_gate_params['Delta_local'], h_ls, x=x)
                        result_gate = qt.mesolve(H_gate, psi_gate, [0, gate_time], c_ops_gate, options={'nsteps':100000})
                        psi_gate = result_gate.states[-1]
                    # finally ramp down with T2
                    result_omega = qt.mesolve(H_ramp_down_omega, psi_gate, [0, t_ramp], c_ops_gate, options={'nsteps':100000})
                    psi_gate = result_omega.states[-1]
                    result_delta = qt.mesolve(H_ramp_down_delta, psi_gate, [0, t_ramp_Delta], c_ops_gate, options={'nsteps':100000})
                    psi_gate = result_delta.states[-1]
                else:
                    for s in seq:
                        U_1 = U_1_func(s)
                        psi_gate = U_1 * psi_gate
                    
                    # finally ramp down
                    psi_gate = U_ramp_down * psi_gate
                return psi_gate
            elif ret_continuous:
                # For continuous evolution with variable phi
                gate_time = in_gate_params['gate_duration']
                
                for i, phi_val in enumerate(seq):
                    H_gate = get_H_indep(15.8, phi_val, in_gate_params['Delta_global'], in_gate_params['Delta_local'], h_ls, x=x)
                    
                    t_list_gate = np.linspace(0, gate_time, n_continuous_steps)
                    result_gate = qt.mesolve(H_gate, psi_gate, t_list_gate, c_ops_gate, options={'nsteps':100000}) if include_T2 else qt.sesolve(H_gate, psi_gate, t_list_gate, options={'nsteps':100000})
                    # Add to continuous lists
                    for t, state in zip(t_list_gate, result_gate.states):
                        times_list.append(gate_start_time + t)
                        states_list.append(state)
                        probs = np.abs(state.full().flatten())**2
                        probs /= np.sum(probs)
                        probs_list.append(probs)
                    gate_start_time += gate_time
                    psi_gate = result_gate.states[-1]
                
                t_list_ramp_down_omega = np.linspace(0, t_ramp, n_continuous_steps)
                result_ramp_down_omega = qt.mesolve(H_ramp_down_omega, psi_gate, t_list_ramp_down_omega, c_ops_gate, options={'nsteps':100000}) if include_T2 else qt.sesolve(H_ramp_down_omega, psi_gate, t_list_ramp_down_omega, options={'nsteps':100000})
                for t, state in zip(t_list_ramp_down_omega, result_ramp_down_omega.states):
                    times_list.append(gate_start_time + t)
                    states_list.append(state)
                    probs = np.abs(state.full().flatten())**2
                    probs /= np.sum(probs)
                    probs_list.append(probs)
                gate_start_time += t_ramp
                psi_gate = result_ramp_down_omega.states[-1]

                t_list_ramp_down_delta = np.linspace(0, t_ramp_Delta, n_continuous_steps)
                result_ramp_down_delta = qt.mesolve(H_ramp_down_delta, psi_gate, t_list_ramp_down_delta, c_ops_gate, options={'nsteps':100000}) if include_T2 else qt.sesolve(H_ramp_down_delta, psi_gate, t_list_ramp_down_delta, options={'nsteps':100000})
                for t, state in zip(t_list_ramp_down_delta, result_ramp_down_delta.states):
                    times_list.append(gate_start_time + t)
                    states_list.append(state)
                    probs = np.abs(state.full().flatten())**2
                    probs /= np.sum(probs)
                    probs_list.append(probs)
                psi_gate = result_ramp_down_delta.states[-1]
                
                return psi_gate
            else:
                state_ls = []
                for s in seq:
                    U_1 = U_1_func(s)
                    psi_gate = U_1 * psi_gate
                    state_ls.append(psi_gate)
                
                # finally ramp down
                psi_gate = U_ramp_down * psi_gate
                state_ls.append(psi_gate)
                return state_ls

    U_ramp_to_init_gate, U_1, U_2 = gates_prep()
    
    # Collapse operators for gate evolution (using gate Delta_local)
    c_ops_gate = get_c_ops(in_gate_params['Delta_local'], h_ls)
    
    probs_all_t = []
    final_states = [] if return_last_state else None
    
    for i, t_plateau in enumerate(t_plateau_ls):
        # print("Processing sequence", i+1, "of", len(t_plateau_ls))
        
        if ret_continuous:
            psi_before = before_gates(t_plateau, current_time)
            current_time += t_plateau
        else:
            psi_before = before_gates(t_plateau, 0)
            
        psi_gate_init = U_ramp_to_init_gate * psi_before

        probs_seq_ls = []
        final_states_seq = [] if return_last_state else None
        # print(seq_ls_pre)
        seq_ls = seq_ls_pre[i]
        # print("seq_ls!!!!:", seq_ls)

        if not type(seq_ls) is list:
            seq_ls = []

        if preset_opt is None:
            if len(seq_ls) >= 1:
                # If only recording from gate start, reset time to 0 now
                if ret_continuous and continuous_gates_only:
                    current_time = 0.0
                for seq in tqdm(seq_ls, desc="Applying gate sequences"):
                    if not local_haar:
                    # print("Applying sequence:", seq)
                        if ret_continuous:
                            psi_gate = gates(psi_gate_init, U_1, U_2, seq, current_time)
                            current_time += len(seq) * in_gate_params['gate_duration'] + t_ramp + t_ramp_Delta
                        else:
                            psi_gate = gates(psi_gate_init, U_1, U_2, seq, 0)
                        
                        # Store final state if requested
                        if return_last_state:
                            final_states_seq.append(psi_gate)
                        
                        # compute the probabilities
                        # psi_gate = qt.ket2dm(psi_gate)
                        # probs = np.array([psi_gate[i, i].real for i in range(2**len(h_ls))])
                        if ret_probs or ret_continuous:
                            probs = np.abs(psi_gate.full().flatten())**2
                            probs /= np.sum(probs)  # normalize
                            probs_seq_ls.append(probs)
                        else:
                            print("appending state, the last", psi_gate[-2])
                            probs_seq_ls.append(psi_gate)
                    else:
                        # print("LOCAL HAARR")
                        # local haar random unitaries
                        psi_gate = deepcopy(psi_gate_init)
                        gate_start_time = current_time
                        
                        if ret_probs and not ret_continuous:
                            # first ramp down before applying random unitaries
                            if include_T2:
                                result_omega = qt.mesolve(H_ramp_down_omega, psi_gate, [0, t_ramp], c_ops_gate, options={'nsteps':100000})
                                psi_gate = result_omega.states[-1]
                                result_delta = qt.mesolve(H_ramp_down_delta, psi_gate, [0, t_ramp_Delta], c_ops_gate, options={'nsteps':100000})
                                psi_gate = result_delta.states[-1]
                            else:
                                psi_gate = U_ramp_down * psi_gate
                            
                            # for s in seq:
                            # generate a random local unitary for each qubit
                            if indep_haar:
                                U_local = qt.tensor([qt.rand_unitary(2) for _ in range(len(h_ls))])
                            else:
                                rand_U = qt.rand_unitary(2) 
                                U_local = qt.tensor([rand_U for _ in range(len(h_ls))])

                            psi_gate = U_local * psi_gate
                            # Store final state if requested
                            if return_last_state:
                                final_states_seq.append(psi_gate)
                            probs = np.abs(psi_gate.full().flatten())**2
                            probs /= np.sum(probs)  # normalize
                            probs_seq_ls.append(probs)
                        elif ret_continuous:
                            t_list_ramp_down_omega = np.linspace(0, t_ramp, n_continuous_steps)
                            result_ramp_down_omega = qt.mesolve(H_ramp_down_omega, psi_gate, t_list_ramp_down_omega, c_ops_gate, options={'nsteps':100000}) if include_T2 else qt.sesolve(H_ramp_down_omega, psi_gate, t_list_ramp_down_omega, options={'nsteps':100000})
                            for t, state in zip(t_list_ramp_down_omega, result_ramp_down_omega.states):
                                times_list.append(gate_start_time + t)
                                states_list.append(state)
                                probs = np.abs(state.full().flatten())**2
                                probs /= np.sum(probs)
                                probs_list.append(probs)
                            psi_gate = result_ramp_down_omega.states[-1]
                            gate_start_time += t_ramp

                            t_list_ramp_down_delta = np.linspace(0, t_ramp_Delta, n_continuous_steps)
                            result_ramp_down_delta = qt.mesolve(H_ramp_down_delta, psi_gate, t_list_ramp_down_delta, c_ops_gate, options={'nsteps':100000}) if include_T2 else qt.sesolve(H_ramp_down_delta, psi_gate, t_list_ramp_down_delta, options={'nsteps':100000})
                            for t, state in zip(t_list_ramp_down_delta, result_ramp_down_delta.states):
                                times_list.append(gate_start_time + t)
                                states_list.append(state)
                                probs = np.abs(state.full().flatten())**2
                                probs /= np.sum(probs)
                                probs_list.append(probs)
                            psi_gate = result_ramp_down_delta.states[-1]
                            gate_start_time += t_ramp_Delta
                            
                            # for s in seq:
                            U_local = qt.tensor([qt.rand_unitary(2) for _ in range(len(h_ls))])
                            psi_gate = U_local * psi_gate
                            # Store continuous evolution after each local unitary
                            times_list.append(gate_start_time)
                            states_list.append(psi_gate)
                            probs = np.abs(psi_gate.full().flatten())**2
                            probs /= np.sum(probs)
                            probs_list.append(probs)
                            gate_start_time += 0  # instantaneous local unitary
                            
                            current_time = gate_start_time
                            
                            # Store final state if requested
                            if return_last_state:
                                final_states_seq.append(psi_gate)
                        else:
                            raise ValueError("local_haar must be used with ret_probs or ret_continuous")
            else: # ramp down from plateau then 
                print("no gates in sequence, just ramp down")
                # p1 = (0, phi, 0, 0)

                if ret_continuous:
                    t_list_ramp_down_omega = np.linspace(0, t_ramp, n_continuous_steps)
                    result_ramp_down_omega = qt.mesolve(H_ramp_down_omega, psi_before, t_list_ramp_down_omega, c_ops_plateau, options={'nsteps':100000}) if include_T2 else qt.sesolve(H_ramp_down_omega, psi_before, t_list_ramp_down_omega, options={'nsteps':100000})
                    for t, state in zip(t_list_ramp_down_omega, result_ramp_down_omega.states):
                        times_list.append(current_time + t)
                        states_list.append(state)
                        probs = np.abs(state.full().flatten())**2
                        probs /= np.sum(probs)
                        probs_list.append(probs)
                    current_time += t_ramp
                    psi_after_omega_down = result_ramp_down_omega.states[-1]

                    t_list_ramp_down_delta = np.linspace(0, t_ramp_Delta, n_continuous_steps)
                    result_ramp_down_delta = qt.mesolve(H_ramp_down_delta, psi_after_omega_down, t_list_ramp_down_delta, c_ops_plateau, options={'nsteps':100000}) if include_T2 else qt.sesolve(H_ramp_down_delta, psi_after_omega_down, t_list_ramp_down_delta, options={'nsteps':100000})
                    for t, state in zip(t_list_ramp_down_delta, result_ramp_down_delta.states):
                        times_list.append(current_time + t)
                        states_list.append(state)
                        probs = np.abs(state.full().flatten())**2
                        probs /= np.sum(probs)
                        probs_list.append(probs)
                    current_time += t_ramp_Delta
                    psi_no_gate = result_ramp_down_delta.states[-1]
                else:
                    if include_T2:
                        result_omega = qt.mesolve(H_ramp_down_omega, psi_before, [0, t_ramp], c_ops_plateau, options={'nsteps':100000})
                        psi_after_omega = result_omega.states[-1]
                        result_delta = qt.mesolve(H_ramp_down_delta, psi_after_omega, [0, t_ramp_Delta], c_ops_plateau, options={'nsteps':100000})
                        psi_no_gate = result_delta.states[-1]
                    else:
                        psi_no_gate = U_ramp_down * psi_before
                
                # Store final state if requested
                if return_last_state:
                    final_states_seq.append(psi_no_gate)
                    
                # psi_no_gate = qt.ket2dm(psi_no_gate)
                # probs = np.array([psi_no_gate[i, i].real for i in range(2**len(h_ls))])
                if ret_probs or ret_continuous:
                    probs = np.abs(psi_no_gate.full().flatten())**2
                    probs /= np.sum(probs)  # normalize
                else:
                    probs = psi_no_gate
                probs_seq_ls.append(probs)

        elif preset_opt == 'ramsey':
            # print("using ramsey preset opt in qutip, Delta_global, Delta_local:", Delta_global, Delta_local)

            t_plateau_ramsey = (np.pi / 2) / Omega - t_ramp
            # print("t_plateau_ramsey:", t_plateau_ramsey)
            t_plateau_ramsey = max(t_plateau_ramsey, 0.05)  # ensure non-negative
            t_ramp_Delta = max(min_dt, abs(Delta_global) / _Delta_slope, abs(Delta_local) / _Delta_slope)

            if ret_continuous:
                psi_after_ramsey_plat = before_gates(t_plateau_ramsey, current_time)
                current_time += t_plateau_ramsey
            else:
                psi_after_ramsey_plat = before_gates(t_plateau_ramsey, 0)
                
            H_ramp_down = get_H_ramp((Omega, phi, Delta_global, Delta_local), (0, phi, Delta_global, Delta_local), x, h_ls, t_ramp, t_ramp_Delta)
            
            if ret_continuous:
                # Continuous evolution for Ramsey sequence
                t_list_ramp_down = np.linspace(0, t_ramp, n_continuous_steps)
                result_ramp_down = qt.mesolve(H_ramp_down, psi_after_ramsey_plat, t_list_ramp_down, c_ops_plateau, options={'nsteps':100000}) if include_T2 else qt.sesolve(H_ramp_down, psi_after_ramsey_plat, t_list_ramp_down, options={'nsteps':100000})
                # Add to continuous lists
                for t, state in zip(t_list_ramp_down, result_ramp_down.states):
                    times_list.append(current_time + t)
                    states_list.append(state)
                    probs = np.abs(state.full().flatten())**2
                    probs /= np.sum(probs)
                    probs_list.append(probs)
                current_time += t_ramp
                psi_ramped_down = result_ramp_down.states[-1]
                
                # now delay for time t_plateau
                H_plateau_delay = get_H_indep(0, phi, Delta_global, Delta_local, h_ls, x=x)
                t_list_delay = np.linspace(0, t_plateau, n_continuous_steps)
                result_delay = qt.mesolve(H_plateau_delay, psi_ramped_down, t_list_delay, c_ops_plateau, options={'nsteps':100000}) if include_T2 else qt.sesolve(H_plateau_delay, psi_ramped_down, t_list_delay, options={'nsteps':100000})
                # Add to continuous lists
                for t, state in zip(t_list_delay, result_delay.states):
                    times_list.append(current_time + t)
                    states_list.append(state)
                    probs = np.abs(state.full().flatten())**2
                    probs /= np.sum(probs)
                    probs_list.append(probs)
                current_time += t_plateau
                psi_plateau_delay = result_delay.states[-1]

                # now ramp up again
                H_ramp_up = get_H_ramp((0, phi, Delta_global, Delta_local), (Omega, phi, Delta_global, Delta_local), x, h_ls, t_ramp, t_ramp_Delta)
                H_plateau_ramsey = get_H_indep(Omega, phi, Delta_global, Delta_local, h_ls, x=x)

                t_list_ramp_up = np.linspace(0, t_ramp, n_continuous_steps)
                result_ramp_up = qt.mesolve(H_ramp_up, psi_plateau_delay, t_list_ramp_up, c_ops_plateau, options={'nsteps':100000}) if include_T2 else qt.sesolve(H_ramp_up, psi_plateau_delay, t_list_ramp_up, options={'nsteps':100000})
                # Add to continuous lists
                for t, state in zip(t_list_ramp_up, result_ramp_up.states):
                    times_list.append(current_time + t)
                    states_list.append(state)
                    probs = np.abs(state.full().flatten())**2
                    probs /= np.sum(probs)
                    probs_list.append(probs)
                current_time += t_ramp
                psi_ramsey_up = result_ramp_up.states[-1]
                
                t_list_ramsey_plat = np.linspace(0, t_plateau_ramsey, n_continuous_steps)
                result_ramsey_plat = qt.mesolve(H_plateau_ramsey, psi_ramsey_up, t_list_ramsey_plat, c_ops_plateau, options={'nsteps':100000}) if include_T2 else qt.sesolve(H_plateau_ramsey, psi_ramsey_up, t_list_ramsey_plat, options={'nsteps':100000})
                # Add to continuous lists
                for t, state in zip(t_list_ramsey_plat, result_ramsey_plat.states):
                    times_list.append(current_time + t)
                    states_list.append(state)
                    probs = np.abs(state.full().flatten())**2
                    probs /= np.sum(probs)
                    probs_list.append(probs)
                current_time += t_plateau_ramsey
                psi_ramsey_plat = result_ramsey_plat.states[-1]
                
                t_list_final_down = np.linspace(0, t_ramp, n_continuous_steps)
                result_final_down = qt.mesolve(H_ramp_down, psi_ramsey_plat, t_list_final_down, c_ops_plateau, options={'nsteps':100000}) if include_T2 else qt.sesolve(H_ramp_down, psi_ramsey_plat, t_list_final_down, options={'nsteps':100000})
                # Add to continuous lists
                for t, state in zip(t_list_final_down, result_final_down.states):
                    times_list.append(current_time + t)
                    states_list.append(state)
                    probs = np.abs(state.full().flatten())**2
                    probs /= np.sum(probs)
                    probs_list.append(probs)
                psi_ramsey_down = result_final_down.states[-1]
                
            else:
                if include_T2:
                    psi_ramped_down = qt.mesolve(H_ramp_down, psi_after_ramsey_plat, [0, t_ramp], c_ops_plateau, options={'nsteps':100000}).states[-1]
                else:
                    psi_ramped_down = qt.sesolve(H_ramp_down, psi_after_ramsey_plat, [0, t_ramp], options={'nsteps':100000}).states[-1]

                # now delay for time t_plateau
                H_plateau_delay = get_H_indep(0, phi, Delta_global, Delta_local, h_ls, x=x)

                if include_T2:
                    psi_plateau_delay = qt.mesolve(H_plateau_delay, psi_ramped_down, [0, t_plateau], c_ops_plateau, options={'nsteps':100000}).states[-1]
                else:
                    U_plateau_delay = (-1j * H_plateau_delay * t_plateau).expm()
                    psi_plateau_delay = U_plateau_delay * psi_ramped_down

                # now ramp up again
                H_ramp_up = get_H_ramp((0, phi, Delta_global, Delta_local), (Omega, phi, Delta_global, Delta_local), x, h_ls, t_ramp, t_ramp_Delta)
                H_plateau_ramsey = get_H_indep(Omega, phi, Delta_global, Delta_local, h_ls, x=x)

                # print("t_plateau_ramsey:", t_plateau_ramsey)

                if include_T2:
                    psi_ramsey_up = qt.mesolve(H_ramp_up, psi_plateau_delay, [0, t_ramp], c_ops_plateau, options={'nsteps':100000}).states[-1]
                    psi_ramsey_plat = qt.mesolve(H_plateau_ramsey, psi_ramsey_up, [0, t_plateau_ramsey], c_ops_plateau, options={'nsteps':100000}).states[-1]
                    psi_ramsey_down = qt.mesolve(H_ramp_down, psi_ramsey_plat, [0, t_ramp], c_ops_plateau, options={'nsteps':100000}).states[-1]
                else:
                    U_plateau_ramsey = (-1j * H_plateau_ramsey * t_plateau_ramsey).expm()
                    psi_ramsey_up = qt.sesolve(H_ramp_up, psi_plateau_delay, [0, t_ramp], options={'nsteps':100000}).states[-1]
                    psi_ramsey_plat = U_plateau_ramsey * psi_ramsey_up
                    psi_ramsey_down = qt.sesolve(H_ramp_down, psi_ramsey_plat, [0, t_ramp], options={'nsteps':100000}).states[-1]

            # Store final state if requested (before converting to density matrix)
            if return_last_state:
                final_states_seq.append(psi_ramsey_down)
                
            psi_ramsey_down = qt.ket2dm(psi_ramsey_down)

            probs = np.array([psi_ramsey_down[i, i].real
             for i in range(2**len(h_ls))])
            probs /= np.sum(probs)  # normalize
            probs_seq_ls.append(probs)
            # print(probs)

        else:
            raise ValueError("Unknown preset_opt:", preset_opt)

        probs_all_t.append(probs_seq_ls)
        if return_last_state:
            final_states.append(final_states_seq)

    if ret_continuous:
        return probs_all_t, times_list, probs_list, states_list
    elif return_last_state:
        return probs_all_t, final_states
    else:
        return probs_all_t




