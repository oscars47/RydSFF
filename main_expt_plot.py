## single time point, sweep Delta_local and N
from QuEraToolbox.hamiltonian import drive_main, get_h_ls, get_J_arr
from QuEraToolbox.expt_file_manager import ExptStore
from master_params_rbp import read_expt_task, gen_tasks, det_cost
from process_rbp import process_bitstrings, numerical
import numpy as np
import qutip as qt
import os, time, json, shutil, shlex, subprocess, sys
import matplotlib.pyplot as plt
import matplotlib as mpl
from joblib import Parallel, delayed
from tqdm import trange
from uncertainties import unumpy as unp
from fig_styling import style_axis
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm, Normalize
from scipy.interpolate import PchipInterpolator
from task_submission_main import extract_readouterror_rabi



def schmidt_entropy_from_ket(psi_qobj, N_A, base=2):
    psi = psi_qobj.full().ravel(order='F')
    dimA = 1 << N_A
    dimB = psi.size // dimA
    mat = psi.reshape(dimA, dimB, order='F')
    s = np.linalg.svd(mat, compute_uv=False)
    p = np.clip(s * s, 0.0, 1.0)
    p /= p.sum()
    logf = np.log2 if base == 2 else np.log
    nz = p > 0
    return float(-(p[nz] * logf(p[nz])).sum())

def schmidt_renyi2_from_ket(psi_qobj, N_A, base=2):
    """
    Renyi-2 entropy S2(rho_A) for a bipartition of a pure state |psi⟩.
    Uses SVD of the reshaped coefficient matrix; no density matrix or ptrace.

    Args:
        psi_qobj : QuTiP ket (Qobj) of dimension 2^N.
        N_A      : number of qubits in subsystem A.
        base     : log base (2 for bits, np.e for nats).

    Returns:
        float S2 = -log_base( Tr[rho_A^2] ).
    """
    # QuTiP uses Kronecker (column-major) ordering
    psi = psi_qobj.full().ravel(order='F')
    dimA = 1 << N_A
    dimB = psi.size // dimA
    mat = psi.reshape(dimA, dimB, order='F')

    # singular values s are Schmidt coeffs; p = s^2 are eigenvalues of rho_A
    s = np.linalg.svd(mat, compute_uv=False)
    p = np.clip(s * s, 0.0, 1.0)
    # normalize (guards tiny FP drift)
    ssum = p.sum()
    if ssum <= 0:
        return 0.0
    p /= ssum

    purity = float(np.sum(p * p))          # Tr[rho_A^2] = Σ p_i^2
    purity = min(max(purity, 1e-16), 1.0)  # clamp for numerical safety

    if base == 2:
        return -np.log2(purity)
    elif base == np.e:
        return -np.log(purity)
    else:
        return -np.log(purity) / np.log(base)

# ---------- single-N job ----------

def run_for_N_num(iN, N_, Delta_local_ls, t_fixed, num_ens,Delta_mean, a, Omega, phi, force_recompute=False, dir_root="."):
    """Compute S(N, Δ_local) for one system size."""

    payload = {
        "N": N_,
        "t_fixed": t_fixed,
        "num_ens": num_ens,
        "Delta_mean": Delta_mean,
        "Delta_local_ls": Delta_local_ls.tolist() if isinstance(Delta_local_ls, np.ndarray) else Delta_local_ls,
        "a": a,
        "Omega": Omega,
        "phi": phi
    }

    manager = ExptStore(dir_root)
    puid, added = manager.add(payload, timestamp=0)
    os.makedirs(os.path.join(dir_root, "data"), exist_ok=True)

    results_filename = os.path.join(dir_root, "data", f"result_{puid}.npy")
    avgs_filename = os.path.join(dir_root, "data", f"avgs_{puid}.npy")

    if not os.path.exists(results_filename) or force_recompute:

        psi0 = qt.tensor([qt.basis(2, 0) for _ in range(N_)])
        N_A = N_ // 2
        J_arr = get_J_arr([(i * a, 0) for i in range(N_)], N_)
        get_H_indep, _ = drive_main(neg_phi=True)

        results_N = np.empty((len(Delta_local_ls), num_ens), dtype=float)
        avgs_N = np.empty(len(Delta_local_ls), dtype=float)

        for d, Delta_local in enumerate(Delta_local_ls):
            Delta_global = Delta_mean - 0.5 * Delta_local
            S_ls = np.empty(num_ens, dtype=float)

            for n in trange(num_ens, desc=f"N={N_}, Delta_local={Delta_local:.2f}", leave=False):
                h_ls = get_h_ls(N_)
                H = get_H_indep(Omega=Omega, phi=phi,
                                Delta_global=Delta_global,
                                Delta_local=Delta_local,
                                h_ls=h_ls, J_arr=J_arr)
                
                U = (-1j * H * t_fixed).expm()
                psi_t = U * psi0    
                S_ls[n] = schmidt_entropy_from_ket(psi_t, N_A)
            results_N[d, :] = S_ls
            avgs_N[d] = S_ls.mean()
            # else:
            #     avg_num, no_avg_num = numerical(h_ls_pre, x_pre, [t_fixed],  base_params, [Delta_mean], Delta_local_ls, dir_root, force_recompute=force_recompute, process_opt=process_opt, time_dep=True)
            #     results_N[d, :] = no_avg_num
            #     avgs_N[d] = avg_num

        np.save(results_filename, results_N)
        np.save(avgs_filename, avgs_N)
        print(f"Saved results to {results_filename} and {avgs_filename}")
    else:
        results_N = np.load(results_filename)
        avgs_N = np.load(avgs_filename)
        print(f"Loaded results from {results_filename} and {avgs_filename}")

    return iN, results_N, avgs_N

def entropy_sweep_numerics(t_fixed, num_ens, Delta_mean, Delta_local_ls, N_ls, dir_root, force_recompute=False, fontsize=30, a=10, Omega=15.8, phi=0, n_jobs=1):

   
    t0 = int(time.time())
    print(f"starting at {t0}")
    # run all N in parallel
    outputs = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(run_for_N_num)(iN, N_, Delta_local_ls, t_fixed, num_ens,
                            Delta_mean, a, Omega, phi, force_recompute=force_recompute, dir_root=dir_root)
        for iN, N_ in enumerate(N_ls)
    )

    # aggregate
    results = np.empty((len(N_ls), len(Delta_local_ls), num_ens))
    avgs = np.empty((len(N_ls), len(Delta_local_ls)))
    for iN, resN, avgN in outputs:
        results[iN] = resN
        avgs[iN] = avgN

    tf = int(time.time())
    print(f"finished at {tf}, total time {tf - t0} s, approx {((tf - t0)/60):.1f} min")

       
    mpl.rcParams.update({'font.size': fontsize})
    plt.rc('text', usetex=True)
    mpl.rc('text.latex', preamble=r"\usepackage{amsmath}\usepackage{newtxtext,newtxmath}")

    fig, ax = plt.subplots(figsize=(8,8))
    for i, N_ in enumerate(N_ls):
        # Delta_local_ls_restr = [dl for dl in Delta_local_ls if dl<=-0.01 and dl>=-15.0]
        # avgs_restr = avgs[i, (Delta_local_ls<=-0.01) & (Delta_local_ls>=-15.0)]

        # ax.plot(np.abs(Delta_local_ls_restr), avgs_restr, marker='o', label=fr"$N={N_}$")
        ax.scatter(np.abs(Delta_local_ls)/J, avgs[i], marker='o', label=fr"$N={N_}$", color='black')
    ax.set_xlabel(r'$|\Delta_{\mathrm{local}}|/J$')
    ax.set_ylabel(rf'$\langle S(\rho_A)\rangle(t={t_fixed}\ \mu\mathrm{{s}})$')
    ax.set_xscale('log')

    for spine in ax.spines.values():
        spine.set_visible(True)
    ax.tick_params(axis='both', which='both', top=True, right=True)
    ax.minorticks_on()
    ax.tick_params(axis='both', which='both', length=4, color='gray')
    ax.legend(fontsize=fontsize*0.8)
    plt.tight_layout()
    

    # get uid for plot
    payload = {
        "t_fixed": t_fixed,
        "num_ens": num_ens,
        "Delta_mean": Delta_mean,
        "Delta_local_ls": Delta_local_ls.tolist() if isinstance(Delta_local_ls, np.ndarray) else Delta_local_ls,
        "N_ls": N_ls,
        "a": a,
        "Omega": Omega,
        "phi": phi
    }
    manager = ExptStore(dir_root)
    puid, added = manager.add(payload, timestamp=0)

    os.makedirs(os.path.join(dir_root, "results"), exist_ok=True)
    plot_filepath = os.path.join(dir_root, "results", f"entropy_sweep_num_{puid}.pdf")
    plt.savefig(plot_filepath)
    print(f"Saved plot to {plot_filepath}")
    plt.show()


##------ our protocol ------ ##
def _get_task(N, Delta_local_ls, t_fixed, num_ens ,Delta_mean, a, Omega, phi, n_shots, n_gates, n_U, t_ramp = 0.0632, Delta_local_ramp_time = 0.05, Omega_delay_time=0.0, f=1, dir_root=".", same_h_ls_task=None, Delta_global_gate = 26.733840091053594, Delta_local_gate = -102.72161226237358, gate_duration = 0.06220645598688665):

    Delta_local_ls_scaled = np.array(Delta_local_ls) * f
    Delta_mean_scaled = Delta_mean * f
    a_scaled = a * (1/f)**(1/6)
    Omega_scaled = Omega * f

    base_params = {
            "ev_params": {
                "Omega": Omega_scaled,
                "Delta_local": 0, # placeholder, overwritten later
                "Delta_global": 0,  # placeholder, overwritten later
                "phi": phi,
                "t_ramp": t_ramp,
                "a": a_scaled
            },
            "n_ens": num_ens
        }

    gate_params_all = [[
            {"n_U":n_U, "n_gates":n_gates, "Delta_global":  Delta_global_gate* f, "Delta_local":  Delta_local_gate * f, "gate_duration": gate_duration, "n_shots":n_shots}
            for _ in range(len(Delta_local_ls_scaled))]]

    if same_h_ls_task is None:
        h_ls_pre=None
    else:
        try:
            h_ls_pre, x_pre, t_plateau_ls, seq_ls_pre_all, base_params, Delta_mean_ls, Delta_local_ls, gate_params_all, cluster_spacing, manual_parallelization, override_local, x0_y0_offset, t_delay, start_Delta_from0, uniform_Omega_Delta_ramp, phi_mode, Delta_local_ramp_time, Omega_delay_time = read_expt_task(same_h_ls_task, dir_root)
            print("Success! Using h_ls_pre from file")
        except:
            print("can't load same_h_ls_task file, using None as default")
            h_ls_pre = None

    Delta_mean_ls = [Delta_mean_scaled]
    Delta_local_ls_scaled = list(Delta_local_ls_scaled)
    t_plateau_ls = list(t_fixed)

    print("Delta_mean_ls:", Delta_mean_ls)
    print("Delta_local_ls_scaled:", Delta_local_ls_scaled)
    print("t_plateau_ls:", t_plateau_ls)


    task_name = gen_tasks(N, Delta_mean_ls, Delta_local_ls_scaled, base_params, gate_params_all, cluster_spacing=None, t_plateau_ls=t_plateau_ls, dir_root=dir_root, same_U_all_time=True, h_ls_pre=h_ls_pre, seq_ls_pre_all = None, manual_parallelization=False, x_pre=None, override_local=False,start_Delta_from0=False, uniform_Omega_Delta_ramp=False, phi_mode='binary', Delta_local_ramp_time=Delta_local_ramp_time, Omega_delay_time=Omega_delay_time)

    cost = det_cost(task_name, dir_root=dir_root)
    print(f"Delta_mean = {Delta_mean_scaled:.3g}. Cost = {cost:.3f}")
    print("task_name:", task_name)
    print('-'*20)
    return task_name


def run_for_N_protocol(task_name, timestamp, opt, epsilon_r_ens = 0.0, epsilon_g_ens = 0.0,  dir_root=".",force_recompute=False, force_recompute_processing=False, shot_noise_model = 'multinomial', specify_ensemble=None, force_recompute_expt=False, T2star_workers=1, save_one_at_a_time=False, haar_opt=None):

    assert haar_opt in [None, 'local-indep', 'local-same'] # haar_opt is for no-rc only, for testing purposes comparing my gates to actual haar
    
    assert opt in ['num','bloqade-sim-no-rc', 'bloqade-sim-rc', 'bloqade-expt', 'T2star-sim-rc', 'T2-sim-rc', 'T2-T2star-sim-rc'], f"Invalid opt {opt}"

    def do(specify_ensemble, force_recompute, force_recompute_processing):
        print("SPECIFY ENSEMBLE:", specify_ensemble)
   
        payload = {
            "task_name": task_name,
            "timestamp": timestamp,
            "shot_noise_model": shot_noise_model,
            "opt": opt,
            "note":"revised 4/10/26 T2star in us"
        }
        if specify_ensemble is not None:
            payload["specify_ensemble"] = specify_ensemble
        if opt not in ['num','bloqade-sim-no-rc', 'bloqade-expt']: 
            payload["epsilon_r_ens"] = epsilon_r_ens
            payload["epsilon_g_ens"] = epsilon_g_ens

        if haar_opt is not None and opt=='bloqade-sim-no-rc':
            payload["haar_opt"] = haar_opt


        print("Payload for run_for_N_protocol:", payload)

        manager = ExptStore(dir_root)
        puid, added = manager.add(payload, timestamp=0)
        print("PUID:", puid, timestamp, "added to manager:", added)

        result_filename = os.path.join(dir_root, "data", f"result_{puid}.npy")
        sem_filename = os.path.join(dir_root, "data", f"sem_{puid}.npy")
        avg_filename = os.path.join(dir_root, "data", f"avg_{puid}.npy")

        if opt == 'bloqade-expt':
            name = f"full_expt_{task_name.split('task_')[1]}_{timestamp}"
        else:
            name = f"{opt}_{task_name.split('task_')[1]}_{timestamp}"

        if not os.path.exists(result_filename) or force_recompute:

            ## then compute vals for all tasks
            h_ls_pre, x_pre, t_plateau_ls, seq_ls_pre_all, base_params, Delta_mean_ls, Delta_local_ls, gate_params_all, cluster_spacing, manual_parallelization, override_local, x0_y0_offset, t_delay, start_Delta_from0, uniform_Omega_Delta_ramp, phi_mode, Delta_local_ramp_time, Omega_delay_time = read_expt_task(task_name, dir_root)

            if opt=='num':
                avg, no_avg = numerical(h_ls_pre, x_pre, t_plateau_ls,  base_params, Delta_mean_ls, Delta_local_ls, dir_root, force_recompute=force_recompute, process_opt='ee', time_dep=True, start_Delta_from0=start_Delta_from0, uniform_Omega_Delta_ramp=uniform_Omega_Delta_ramp, Delta_local_ramp_time=Delta_local_ramp_time, Omega_delay_time=Omega_delay_time)
                sem = None

            elif opt=='bloqade-sim-no-rc':
                avg, sem, no_avg = process_bitstrings(h_ls_pre, x_pre, t_plateau_ls, seq_ls_pre_all, base_params, Delta_mean_ls, Delta_local_ls, gate_params_all, cluster_spacing, manual_parallelization , name=name, is_expt_data=False, timestamp=0, dir_root=dir_root, process_opt='ee', force_recompute=force_recompute, same_U_all_time = True, is_bloqade=True, shot_noise_model = shot_noise_model, force_recompute_processing=force_recompute_processing, start_Delta_from0=start_Delta_from0, uniform_Omega_Delta_ramp=uniform_Omega_Delta_ramp, apply_correction=False, Delta_local_ramp_time=Delta_local_ramp_time, Omega_delay_time=Omega_delay_time, specify_ensemble=specify_ensemble, haar_opt=haar_opt)

            elif opt=='bloqade-sim-rc':
                avg, sem, no_avg = process_bitstrings(h_ls_pre, x_pre, t_plateau_ls, seq_ls_pre_all, base_params, Delta_mean_ls, Delta_local_ls, gate_params_all, cluster_spacing, manual_parallelization , name=name, is_expt_data=False, timestamp=0, dir_root=dir_root, process_opt='ee', force_recompute=force_recompute, same_U_all_time = True, is_bloqade=True, shot_noise_model = shot_noise_model, force_recompute_processing=force_recompute_processing, start_Delta_from0=start_Delta_from0, uniform_Omega_Delta_ramp=uniform_Omega_Delta_ramp, apply_correction=True, Delta_local_ramp_time=Delta_local_ramp_time, Omega_delay_time=Omega_delay_time, epsilon_r_ens=epsilon_r_ens, epsilon_g_ens=epsilon_g_ens, include_T2star=False, include_T2=False, specify_ensemble=specify_ensemble)

            elif opt=='T2star-sim-rc':
                avg, sem, no_avg = process_bitstrings(h_ls_pre, x_pre, t_plateau_ls, seq_ls_pre_all, base_params, Delta_mean_ls, Delta_local_ls, gate_params_all, cluster_spacing, manual_parallelization , name=name, is_expt_data=False, timestamp=0, dir_root=dir_root, process_opt='ee', force_recompute=force_recompute, same_U_all_time = True, is_bloqade=False, shot_noise_model = shot_noise_model, force_recompute_processing=force_recompute_processing, start_Delta_from0=start_Delta_from0, uniform_Omega_Delta_ramp=uniform_Omega_Delta_ramp, apply_correction=True, Delta_local_ramp_time=Delta_local_ramp_time, Omega_delay_time=Omega_delay_time, epsilon_r_ens=epsilon_r_ens, epsilon_g_ens=epsilon_g_ens, include_T2star=True, include_T2=False, specify_ensemble=specify_ensemble, T2star_workers=T2star_workers)

            elif opt=='T2-sim-rc':
                print("INPUT EPSILON_R_ENS, EPSILON_G_ENS:", epsilon_r_ens, epsilon_g_ens)
                avg, sem, no_avg = process_bitstrings(h_ls_pre, x_pre, t_plateau_ls, seq_ls_pre_all, base_params, Delta_mean_ls, Delta_local_ls, gate_params_all, cluster_spacing, manual_parallelization , name=name, is_expt_data=False, timestamp=0, dir_root=dir_root, process_opt='ee', force_recompute=force_recompute, same_U_all_time = True, is_bloqade=False, shot_noise_model = shot_noise_model, force_recompute_processing=force_recompute_processing, start_Delta_from0=start_Delta_from0, uniform_Omega_Delta_ramp=uniform_Omega_Delta_ramp, apply_correction=True, Delta_local_ramp_time=Delta_local_ramp_time, Omega_delay_time=Omega_delay_time, epsilon_r_ens=epsilon_r_ens, epsilon_g_ens=epsilon_g_ens, include_T2star=False, include_T2=True, specify_ensemble=specify_ensemble)

            elif opt=='T2-T2star-sim-rc':
                avg, sem, no_avg = process_bitstrings(h_ls_pre, x_pre, t_plateau_ls, seq_ls_pre_all, base_params, Delta_mean_ls, Delta_local_ls, gate_params_all, cluster_spacing, manual_parallelization , name=name, is_expt_data=False, timestamp=0, dir_root=dir_root, process_opt='ee', force_recompute=force_recompute, same_U_all_time = True, is_bloqade=False, shot_noise_model = shot_noise_model, force_recompute_processing=force_recompute_processing, start_Delta_from0=start_Delta_from0, uniform_Omega_Delta_ramp=uniform_Omega_Delta_ramp, apply_correction=True, Delta_local_ramp_time=Delta_local_ramp_time, Omega_delay_time=Omega_delay_time, epsilon_r_ens=epsilon_r_ens, epsilon_g_ens=epsilon_g_ens, include_T2star=True, include_T2=True, specify_ensemble=specify_ensemble, T2star_workers=T2star_workers)
                
            elif opt=='bloqade-expt':
                avg, sem, no_avg = process_bitstrings(h_ls_pre, x_pre, t_plateau_ls, seq_ls_pre_all, base_params, Delta_mean_ls, Delta_local_ls, gate_params_all, cluster_spacing, manual_parallelization , name=name, is_expt_data=True, timestamp=timestamp, dir_root=dir_root, process_opt='ee', force_recompute=force_recompute_expt, same_U_all_time = True, is_bloqade=True, shot_noise_model = shot_noise_model, force_recompute_processing=force_recompute_expt, start_Delta_from0=start_Delta_from0, uniform_Omega_Delta_ramp=uniform_Omega_Delta_ramp, apply_correction=False, Delta_local_ramp_time=Delta_local_ramp_time, Omega_delay_time=Omega_delay_time, specify_ensemble=specify_ensemble)
                
            
            np.save(result_filename, no_avg)
            np.save(avg_filename, avg)
            if sem is not None:
                np.save(sem_filename, sem)

        else:
            no_avg = np.load(result_filename, allow_pickle= True)
            avg = np.load(avg_filename, allow_pickle=   True)
            if opt !='num':
                sem = np.load(sem_filename, allow_pickle=   True)
            else:
                sem = None
            
        return avg, sem, no_avg

    if save_one_at_a_time and specify_ensemble is not None:
        for ens in specify_ensemble:
            do(ens, force_recompute, force_recompute_processing)
        return None
    else:        
        return do(specify_ensemble, force_recompute, force_recompute_processing)


    

def entropy_sweep_protocol(task_ls, timestamp_ls, opt_ls, fontsize=20, force_recompute=False, force_recompute_processing=False, shot_noise_model = 'multinomial', epsilon_r_ens_ls = 0.0, epsilon_g_ens_ls = 0.0, dir_root=".", fixed_time_protocol=2.125, specify_ensemble_ls=None, alpha=0.3, force_recompute_expt=False):

    mpl.rcParams.update({'font.size': fontsize})
    plt.rc('text', usetex=True)
    mpl.rc('text.latex', preamble=r"\usepackage{amsmath}\usepackage{newtxtext,newtxmath}")

    fig, axs = plt.subplots(2,1, figsize=(10, 10*2))
    # Collect points across tasks to draw one connected band per opt type.
    expt_delta_all = []
    expt_avg_all = []
    expt_sem_all = []

    t2star_delta_all = []
    t2star_avg_all = []
    t2star_sem_all = []

    t2_delta_all = []
    t2_avg_all = []
    t2_sem_all = []

    t2_t2star_delta_all = []
    t2_t2star_avg_all = []
    t2_t2star_sem_all = []

    idx = 0
    
    for task_name, timestamp, opt in zip(task_ls, timestamp_ls, opt_ls):
    
        try:
            epsilon_r_ens = epsilon_r_ens_ls[idx]
            epsilon_g_ens = epsilon_g_ens_ls[idx]
            print(f"Using epsilon_r_ens={epsilon_r_ens}, epsilon_g_ens={epsilon_g_ens} for task {task_name}!!!")
        except:
            print(f"Error: index {idx} out of range for epsilon_ens_ls. Using 0.0 for both.")
            epsilon_r_ens = 0.0
            epsilon_g_ens = 0.0

        Delta_local_ls = np.abs(read_expt_task(task_name, dir_root)[6])
        if specify_ensemble_ls is not None:
            ensembles = specify_ensemble_ls[idx]
        else:
            ensembles = None
            
        avg, sem, no_avg = run_for_N_protocol(task_name, timestamp, opt, force_recompute=force_recompute, force_recompute_processing=force_recompute_processing, shot_noise_model=shot_noise_model, epsilon_r_ens=epsilon_r_ens, epsilon_g_ens=epsilon_g_ens, dir_root=dir_root, specify_ensemble=ensembles, force_recompute_expt=force_recompute_expt)

        if opt == 'bloqade-expt':
            print("Delta_local_ls:", Delta_local_ls)

            if len(Delta_local_ls) == 1:
                print("LEN 1, using fixed_time_protocol to find index")
                t_plateau_ls = np.array(read_expt_task(task_name, dir_root)[2])
                t_idx = np.argmin(np.abs(t_plateau_ls - fixed_time_protocol))
                print(t_idx)
                print("avg before:", avg)
                avg_slice = [avg[0][0][t_idx]]
                sem_slice = [sem[0][0][t_idx]]
                print("avg:", avg_slice)
                print("shape in no_avg", np.array(no_avg[0][0]).shape)
                print("no_avg", np.array(no_avg[0][0]))
            else:
                t_plateau_ls = np.array(read_expt_task(task_name, dir_root)[2])
                t_idx = np.argmin(np.abs(t_plateau_ls - fixed_time_protocol))
                avg_slice = [avg[0][d][t_idx] for d in range(len(Delta_local_ls))]
                sem_slice = [sem[0][d][t_idx] for d in range(len(Delta_local_ls))]

            expt_delta_all.extend(np.asarray(Delta_local_ls, dtype=float).tolist())
            expt_avg_all.extend(np.asarray(avg_slice, dtype=float).tolist())
            expt_sem_all.extend(np.asarray(sem_slice, dtype=float).tolist())

        elif opt == 'bloqade-sim-rc':
            print("sim-rc")
            print("avg:", avg)
            if len(Delta_local_ls) == 1:
                print("LEN 1, using fixed_time_protocol to find index")
            t_plateau_ls = np.array(read_expt_task(task_name, dir_root)[2])
            t_idx = np.argmin(np.abs(t_plateau_ls - fixed_time_protocol))
            avg = [avg[0][d][t_idx] for d in range(len(Delta_local_ls))]
            sem= [sem[0][d][t_idx] for d in range(len(Delta_local_ls))]

            upper_bound = np.array(avg) + np.array(sem)
            lower_bound = np.array(avg) - np.array(sem)
            axs[0].fill_between(Delta_local_ls, lower_bound, upper_bound, color='black', alpha=alpha, label=r'$\mathrm{Emul.\, r.c.}$')

        elif opt == 'T2star-sim-rc':
            print("sim-rc")
            print("avg:", avg)
            if len(Delta_local_ls) == 1:
                print("LEN 1, using fixed_time_protocol to find index")
            t_plateau_ls = np.array(read_expt_task(task_name, dir_root)[2])
            t_idx = np.argmin(np.abs(t_plateau_ls - fixed_time_protocol))
            avg = [avg[0][d][t_idx] for d in range(len(Delta_local_ls))]
            sem= [sem[0][d][t_idx] for d in range(len(Delta_local_ls))]

            t2star_delta_all.extend(np.asarray(Delta_local_ls, dtype=float).tolist())
            t2star_avg_all.extend(np.asarray(avg, dtype=float).tolist())
            t2star_sem_all.extend(np.asarray(sem, dtype=float).tolist())

        elif opt == 'T2-sim-rc':
            print("sim-rc")
            print("avg:", avg)
            if len(Delta_local_ls) == 1:
                print("LEN 1, using fixed_time_protocol to find index")
            t_plateau_ls = np.array(read_expt_task(task_name, dir_root)[2])
            t_idx = np.argmin(np.abs(t_plateau_ls - fixed_time_protocol))
            avg = [avg[0][d][t_idx] for d in range(len(Delta_local_ls))]
            sem= [sem[0][d][t_idx] for d in range(len(Delta_local_ls))]

            t2_delta_all.extend(np.asarray(Delta_local_ls, dtype=float).tolist())
            t2_avg_all.extend(np.asarray(avg, dtype=float).tolist())
            t2_sem_all.extend(np.asarray(sem, dtype=float).tolist())

        elif opt == 'T2-T2star-sim-rc':
            print("sim-rc")
            print("avg:", avg)
            if len(Delta_local_ls) == 1:
                print("LEN 1, using fixed_time_protocol to find index")
            t_plateau_ls = np.array(read_expt_task(task_name, dir_root)[2])
            t_idx = np.argmin(np.abs(t_plateau_ls - fixed_time_protocol))
            avg = [avg[0][d][t_idx] for d in range(len(Delta_local_ls))]
            sem= [sem[0][d][t_idx] for d in range(len(Delta_local_ls))]

            t2_t2star_delta_all.extend(np.asarray(Delta_local_ls, dtype=float).tolist())
            t2_t2star_avg_all.extend(np.asarray(avg, dtype=float).tolist())
            t2_t2star_sem_all.extend(np.asarray(sem, dtype=float).tolist())

        elif opt == 'bloqade-sim-no-rc':
            print("sim-rc")
            print("avg:", avg)
            if len(Delta_local_ls) == 1:
                print("LEN 1, using fixed_time_protocol to find index")
            t_plateau_ls = np.array(read_expt_task(task_name, dir_root)[2])
            t_idx = np.argmin(np.abs(t_plateau_ls - fixed_time_protocol))
            avg = [avg[0][d][t_idx] for d in range(len(Delta_local_ls))]
            sem= [sem[0][d][t_idx] for d in range(len(Delta_local_ls))]

            upper_bound = np.array(avg) + np.array(sem)
            lower_bound = np.array(avg) - np.array(sem)
            axs[1].fill_between(Delta_local_ls, lower_bound, upper_bound, color='black', alpha=alpha, label=r'$\mathrm{Emul.}$')

        elif opt == 'num':
            avg = avg[0]
            axs[1].plot(Delta_local_ls, avg, label=r'$\mathrm{Num.}$', color='black', linewidth=4)

        idx += 1

    # Plot collected expt points
    if len(expt_delta_all) > 0:
        expt_delta_all = np.asarray(expt_delta_all, dtype=float)
        expt_avg_all = np.asarray(expt_avg_all, dtype=float)
        expt_sem_all = np.asarray(expt_sem_all, dtype=float)
        sort_idx = np.argsort(expt_delta_all)
        axs[0].errorbar(expt_delta_all[sort_idx], expt_avg_all[sort_idx], yerr=expt_sem_all[sort_idx], marker='s', label=r'$\mathrm{Aquila}$', color='black', markerfacecolor='black', markeredgecolor='black', markeredgewidth=2, markersize=10, ecolor='black', elinewidth=3, capsize=6, linestyle='none')

    # Plot collected T2* band
    if len(t2star_delta_all) > 0:
        t2star_delta_all = np.asarray(t2star_delta_all, dtype=float)
        t2star_avg_all = np.asarray(t2star_avg_all, dtype=float)
        t2star_sem_all = np.asarray(t2star_sem_all, dtype=float)
        sort_idx = np.argsort(t2star_delta_all)
        delta_sorted = t2star_delta_all[sort_idx]
        avg_sorted = t2star_avg_all[sort_idx]
        sem_sorted = t2star_sem_all[sort_idx]
        axs[0].fill_between(delta_sorted, avg_sorted - sem_sorted, avg_sorted + sem_sorted, color='black', alpha=alpha, label=r'$\mathrm{Emul.\, r.c.} \; T_2^*$')

    # Plot collected T2 band
    if len(t2_delta_all) > 0:
        t2_delta_all = np.asarray(t2_delta_all, dtype=float)
        t2_avg_all = np.asarray(t2_avg_all, dtype=float)
        t2_sem_all = np.asarray(t2_sem_all, dtype=float)
        sort_idx = np.argsort(t2_delta_all)
        delta_sorted = t2_delta_all[sort_idx]
        avg_sorted = t2_avg_all[sort_idx]
        sem_sorted = t2_sem_all[sort_idx]
        axs[0].fill_between(delta_sorted, avg_sorted - sem_sorted, avg_sorted + sem_sorted, color='black', alpha=alpha, label=r'$\mathrm{Emul.\, r.c.}\; T_2$')

    # Plot collected T2+T2* band
    if len(t2_t2star_delta_all) > 0:
        t2_t2star_delta_all = np.asarray(t2_t2star_delta_all, dtype=float)
        t2_t2star_avg_all = np.asarray(t2_t2star_avg_all, dtype=float)
        t2_t2star_sem_all = np.asarray(t2_t2star_sem_all, dtype=float)
        sort_idx = np.argsort(t2_t2star_delta_all)
        delta_sorted = t2_t2star_delta_all[sort_idx]
        avg_sorted = t2_t2star_avg_all[sort_idx]
        sem_sorted = t2_t2star_sem_all[sort_idx]
        axs[0].fill_between(delta_sorted, avg_sorted - sem_sorted, avg_sorted + sem_sorted, color='black', alpha=alpha, label=r'$\mathrm{Emul.\, r.c.}\; T_2\, \mathrm{with }\, T_2^*$')

    axs[1].set_xscale('log')
    axs[0].set_xlabel(r'$|\Delta_{\mathrm{local}}|\ (\mu\mathrm{s}^{-1})$')
    axs[1].set_xlabel(r'$|\Delta_{\mathrm{local}}|\ (\mu\mathrm{s}^{-1})$')
    axs[1].set_ylabel(rf'$S(\rho_A)(t_{{\mathrm{{evol}}}}={fixed_time_protocol:.5g}\ \mu\mathrm{{s}})$')

    axs[0].set_xscale('log')
    axs[0].set_ylabel(rf'$\langle S(\rho_A)\rangle(t_{{\mathrm{{evol}}}}={fixed_time_protocol:.5g}\ \mu\mathrm{{s}})$')

    axs[0].text(-0.15, 1.05, r'$\sf{\textbf{b}}$', transform=axs[0].transAxes, fontsize=fontsize*1.2)
    axs[1].text(-0.15, 1.05, r'$\sf{\textbf{b}}$', transform=axs[1].transAxes, fontsize=fontsize*1.2)

    # axs[0].legend(fontsize=fontsize*0.8, loc='lower left')
    # axs[1].legend(fontsize=fontsize*0.8, loc='upper right')

    for ax in axs:
        style_axis(ax, fontsize=fontsize)
        ax.set_box_aspect(1)
    plt.tight_layout()

    payload = {
        "task_ls": task_ls,
        "timestamp_ls": timestamp_ls,
        "opt_ls": opt_ls,
        "shot_noise_model": shot_noise_model,
        "epsilon_r_ens": epsilon_r_ens,
        "epsilon_g_ens": epsilon_g_ens
    }

    manager = ExptStore(dir_root)
    puid, added = manager.add(payload, timestamp=0)

    os.makedirs(os.path.join(dir_root, "results"), exist_ok=True)
    plot_filepath = os.path.join(dir_root, "results", f"disorder_sweep_{puid}.pdf")
    plt.savefig(plot_filepath)
    print(f"Saved plot to {plot_filepath}")


def time_sweep_protocol(task_ls, timestamp_ls, opt_ls, fontsize=20, force_recompute=False, force_recompute_processing=False, shot_noise_model = 'multinomial', epsilon_r_ens_ls = 0.0, epsilon_g_ens_ls = 0.0, dir_root=".", specify_ensemble_ls=None, alpha=0.3, force_recompute_expt=False, T2star_workers=1, haar_opt=None):

    mpl.rcParams.update({'font.size': fontsize})
    plt.rc('text', usetex=True)
    mpl.rc('text.latex', preamble=r"\usepackage{amsmath}\usepackage{newtxtext,newtxmath}")

    print(task_ls[-1], timestamp_ls[-1], opt_ls[-1], specify_ensemble_ls[-1])
    # raise ValueError

    # top plot: experiment with bloqade rc
    # bottom plot: numerical and bloqade sim with and without rc, with Delta_local as x axis
    fig, axs = plt.subplots(2,1, figsize=(10, 10*2))
    Delta_local_ls_all = []
    idx = 0

    # Collect results per Delta_local to merge/connect after the loop.
    # Maps Delta_local -> list of (t_plateau_arr, avg_arr, sem_arr)
    expt_collected = {}
    t2star_collected = {}
    t2_collected = {}
    t2_t2star_collected = {}
    
    for task_name, timestamp, opt in zip(task_ls, timestamp_ls, opt_ls):
        try:
            epsilon_r_ens = epsilon_r_ens_ls[idx]
            epsilon_g_ens = epsilon_g_ens_ls[idx]
        except:
            print(f"Error: index {idx} out of range for epsilon_ens_ls. Using 0.0 for both.")
            epsilon_r_ens = 0.0
            epsilon_g_ens = 0.0

        t_plateau_ls = np.array(read_expt_task(task_name, dir_root)[2])

        if specify_ensemble_ls is not None:
            
            ensembles = specify_ensemble_ls[idx]
        else:
            ensembles = None

        print("task_name:", task_name, "timestamp", timestamp, "opt:", opt, "epsilon_r_ens:", epsilon_r_ens, "epsilon_g_ens:", epsilon_g_ens, "ensembles:", ensembles)    
        avg, sem, no_avg = run_for_N_protocol(task_name, timestamp, opt, force_recompute=force_recompute, force_recompute_processing=force_recompute_processing, shot_noise_model=shot_noise_model, epsilon_r_ens=epsilon_r_ens, epsilon_g_ens=epsilon_g_ens, dir_root=dir_root, specify_ensemble=ensembles, force_recompute_expt=force_recompute_expt, T2star_workers=T2star_workers, haar_opt=haar_opt)

        idx += 1

    #     chaotic_expt_20timepts = "task_dc5c98f11ff4d89e87af2917" # demo12

    # D2d71_no_err = "task_556886b14a2b72fed9c5e05b"
    # D125d0_no_err = "task_674ce88dad3fbfd139509d1e"
    # D54d2_no_err = "task_67d6229bc32f0413e3ef6e1e"

        # color_dict = {
        # "task_ddebfeaea3ca63079a4c10c4": 'black', # chaotic, 6 qubits
        # "task_15eaace71ebd595814b7bd17": 'red' ,# localized, 6 qubits
        # "task_1f671c4d079b68342b55451a": 'blue', 
        # "task_d0c2eadd5b392ec6c313c9a3": 'blue',
        # "task_5fb40b8e1eb9f1cc8956bd30": 'blue',
        # "task_ed9d7eee9317b5a644e4201f": 'red',
        # "task_a7d98fcffb940592e5f38f77": 'black',
        # 'task_e1863e3d088dfe12500a4d77': 'black',
        # 'task_5cd0bfd9c4fc13d02734571e':'red',
        # 'task_c96b150f2fd48db682f65b4b':'blue',
        # 'task_dc5c98f11ff4d89e87af2917': 'black',
        # 'task_1d092a93d800af23a95f1c7e': 'black',
        # "task_6170712a74d8f7fa845f8a47":'red',
        # 'task_556886b14a2b72fed9c5e05b': 'black',
        # 'task_674ce88dad3fbfd139509d1e': 'blue',
        # 'task_67d6229bc32f0413e3ef6e1e':'red'
        # }

        color_dict = {
            -2.71: 'black',
            -54.2: 'red',
            -125.0: 'blue',
        }

        marker_dict = {
            -2.71: 'o',
            -54.2: 's',
            -125.0: '^',
        }

        linestyle_dict = {
            'black':'--',
            'red': '-',
            'blue': '-.'
        }

        # read expt task to get Delta_local label

        h_ls_pre, x_pre, t_plateau_ls, seq_ls_pre_all, base_params, Delta_mean_ls, Delta_local_ls, gate_params_all, cluster_spacing, manual_parallelization, override_local, x0_y0_offset, t_delay, start_Delta_from0, uniform_Omega_Delta_ramp, phi_mode, Delta_local_ramp_time, Omega_delay_time = read_expt_task(task_name, dir_root)

        Delta_local = Delta_local_ls[0]
        
        Delta_local_in_Delta_local_ls_all = any(np.isclose(Delta_local, dl) for dl in Delta_local_ls_all)


        if opt == 'bloqade-expt':
            key = Delta_local_ls[0]
            expt_collected.setdefault(key, []).append((t_plateau_ls, np.array(avg[0][0]), np.array(sem[0][0])))
            

        elif opt == 'bloqade-sim-rc':
            upper_bound = np.array(avg[0][0]) + np.array(sem[0][0])
            lower_bound = np.array(avg[0][0]) - np.array(sem[0][0])
            axs[0].fill_between(t_plateau_ls, lower_bound, upper_bound, color=color_dict[Delta_local_ls[0]], alpha=alpha)
            # label=r'$\mathrm{Emul.\, r.c.}$'

        elif opt == 'T2star-sim-rc':
            key = Delta_local_ls[0]
            t2star_collected.setdefault(key, []).append((t_plateau_ls, np.array(avg[0][0]), np.array(sem[0][0])))

        elif opt == 'T2-sim-rc':
            key = Delta_local_ls[0]
            t2_collected.setdefault(key, []).append((t_plateau_ls, np.array(avg[0][0]), np.array(sem[0][0])))

        elif opt == 'T2-T2star-sim-rc':
            key = Delta_local_ls[0]
            t2_t2star_collected.setdefault(key, []).append((t_plateau_ls, np.array(avg[0][0]), np.array(sem[0][0])))


        elif opt == 'bloqade-sim-no-rc':
            upper_bound = np.array(avg[0][0]) + np.array(sem[0][0])
            lower_bound = np.array(avg[0][0]) - np.array(sem[0][0])
            axs[1].fill_between(t_plateau_ls, lower_bound, upper_bound, color=color_dict[Delta_local_ls[0]], alpha=alpha)
            # label=r'$\mathrm{Emul.}$'


        elif opt == 'num':
            avg = avg[0][0]
            axs[1].plot(t_plateau_ls, avg, label=rf'$\mathrm{{Num.}}, \, \Delta_{{\mathrm{{local}}}} = {Delta_local} \mu \mathrm{{s}}^{{-1}}$', color=color_dict[Delta_local_ls[0]], linewidth=4, linestyle=linestyle_dict[color_dict[Delta_local_ls[0]]])

        Delta_local_ls_all.append(Delta_local)

    # --- Plot merged expt points (one connected errorbar series per unique Delta_local) ---
    for Delta_local_key, entries in expt_collected.items():
        t_all = np.concatenate([e[0] for e in entries])
        avg_all = np.concatenate([e[1] for e in entries])
        sem_all = np.concatenate([e[2] for e in entries])
        sort_idx = np.argsort(t_all)
        t_sorted = t_all[sort_idx]
        avg_sorted = avg_all[sort_idx]
        sem_sorted = sem_all[sort_idx]
        color = color_dict[Delta_local_key]
        marker = marker_dict.get(Delta_local_key, 'o')
        axs[0].errorbar(t_sorted, avg_sorted, yerr=sem_sorted, marker=marker,
                        label=rf'$\mathrm{{Aquila}}, \, \Delta_{{\mathrm{{local}}}} = {Delta_local_key} \mu \mathrm{{s}}^{{-1}}$',
                        color=color, markerfacecolor=color, markeredgecolor=color,
                markeredgewidth=2, markersize=10, linestyle='none', elinewidth=3, capsize=0)  

    # --- Plot merged T2star bands (one per unique Delta_local) ---
    for Delta_local_key, entries in t2star_collected.items():
        t_all = np.concatenate([e[0] for e in entries])
        avg_all = np.concatenate([e[1] for e in entries])
        sem_all = np.concatenate([e[2] for e in entries])
        sort_idx = np.argsort(t_all)
        t_sorted, avg_sorted, sem_sorted = t_all[sort_idx], avg_all[sort_idx], sem_all[sort_idx]
        axs[0].fill_between(t_sorted, avg_sorted - sem_sorted, avg_sorted + sem_sorted,
                            color=color_dict[Delta_local_key], alpha=alpha,
                            label=rf'$T_2^*, \, \Delta_{{\mathrm{{local}}}} = {Delta_local_key} \mu \mathrm{{s}}^{{-1}}$')

    # --- Plot merged T2 bands ---
    for Delta_local_key, entries in t2_collected.items():
        t_all = np.concatenate([e[0] for e in entries])
        avg_all = np.concatenate([e[1] for e in entries])
        sem_all = np.concatenate([e[2] for e in entries])
        sort_idx = np.argsort(t_all)
        t_sorted, avg_sorted, sem_sorted = t_all[sort_idx], avg_all[sort_idx], sem_all[sort_idx]
        axs[0].fill_between(t_sorted, avg_sorted - sem_sorted, avg_sorted + sem_sorted,
                            color=color_dict[Delta_local_key], alpha=alpha,
                            label=rf'$T_2, \, \Delta_{{\mathrm{{local}}}} = {Delta_local_key} \mu \mathrm{{s}}^{{-1}}$')

    # --- Plot merged T2+T2star bands ---
    for Delta_local_key, entries in t2_t2star_collected.items():
        t_all = np.concatenate([e[0] for e in entries])
        avg_all = np.concatenate([e[1] for e in entries])
        sem_all = np.concatenate([e[2] for e in entries])
        sort_idx = np.argsort(t_all)
        t_sorted, avg_sorted, sem_sorted = t_all[sort_idx], avg_all[sort_idx], sem_all[sort_idx]
        axs[0].fill_between(t_sorted, avg_sorted - sem_sorted, avg_sorted + sem_sorted,
                            color=color_dict[Delta_local_key], alpha=alpha,
                            label=rf'$T_2\, \mathrm{{with}}\, T_2^*, \, \Delta_{{\mathrm{{local}}}} = {Delta_local_key} \mu \mathrm{{s}}^{{-1}}$')

    # axs[1].set_xscale('log')
    axs[0].set_xlabel(r'$t_{{\mathrm{{evol}}}}\, (\mu \mathrm{{s}})$')
    axs[1].set_xlabel(r'$t_{{\mathrm{{evol}}}}\, (\mu \mathrm{{s}})$')
    axs[1].set_ylabel(rf'$S(\rho_A)$')

    # axs[0].set_xscale('log')

    # axs[0].legend(loc='lower right', fontsize=fontsize*0.8)
    # axs[1].legend(loc='upper left', fontsize=fontsize*0.8)

    axs[0].set_ylabel(rf'$\langle S(\rho_A)\rangle$')

    ## add a, b labels
    axs[0].text(-0.15, 1.05, r'$\sf{\textbf{a}}$', transform=axs[0].transAxes, fontsize=fontsize*1.2)
    axs[1].text(-0.15, 1.05, r'$\sf{\textbf{a}}$', transform=axs[1].transAxes, fontsize=fontsize*1.2)

    ### REMOVE THIS !!! !!
    # t_ls = [ 0.525]
    # for t in t_ls:
    #     axs[0].axvline(t, color='gray', linestyle='--', alpha=1)
    #     axs[1].axvline(t, color='gray', linestyle='--', alpha=1)

    # axs[0].axvline(3.25, color='gray', linestyle='--', alpha=0.6)
    # axs[1].axvline(3.25, color='gray', linestyle='--', alpha=0.6)
    
    for ax in axs:
        style_axis(ax, fontsize=fontsize)
        # force ax to be square
        ax.set_box_aspect(1)
    
    plt.tight_layout()

    # get uid for plot
    payload = {
        "task_ls": task_ls,
        "timestamp_ls": timestamp_ls,
        "opt_ls": opt_ls,
        "shot_noise_model": shot_noise_model,
        "epsilon_r_ens": epsilon_r_ens,
        "epsilon_g_ens": epsilon_g_ens,
    }

    manager = ExptStore(dir_root)
    puid, added = manager.add(payload, timestamp=0)

    os.makedirs(os.path.join(dir_root, "results"), exist_ok=True)
    plot_filepath = os.path.join(dir_root, "results", f"time_sweep_{puid}.pdf")
    plt.savefig(plot_filepath)
    print(f"Saved plot to {plot_filepath}")
    # plt.show()

    return axs


def time_disorder_protocol(
    time_task_ls,
    time_timestamp_ls,
    time_opt_ls,
    disorder_task_ls,
    disorder_timestamp_ls,
    disorder_opt_ls,
    fontsize=20,
    force_recompute=False,
    force_recompute_processing=False,
    shot_noise_model='multinomial',
    epsilon_r_ens_time_ls=0.0,
    epsilon_g_ens_time_ls=0.0,
    epsilon_r_ens_disorder_ls=0.0,
    epsilon_g_ens_disorder_ls=0.0,
    dir_root='.',
    specify_ensemble_time_ls=None,
    specify_ensemble_disorder_ls=None,
    fixed_time_protocol=2.125,
    alpha=0.3,
    force_recompute_expt=False,
    T2star_workers=1,
    haar_opt=None,
    ar = 1.618,
    use_bc=False
):
    """
    Combined figure with:
    - panel a: time sweep (experiment + theory)
    - panel b: disorder sweep at a fixed time (experiment + theory)
    """

    def _safe_eps(eps_vals, idx):
        try:
            return eps_vals[idx]
        except Exception:
            return 0.0

    def _safe_ensemble(ensemble_vals, idx):
        if ensemble_vals is None:
            return None
        try:
            return ensemble_vals[idx]
        except Exception:
            return None

    def _merge_entries(entries):
        x_all = np.concatenate([e[0] for e in entries])
        y_all = np.concatenate([e[1] for e in entries])
        has_sem = all(e[2] is not None for e in entries)
        if has_sem:
            sem_all = np.concatenate([e[2] for e in entries])
        else:
            sem_all = None
        sort_idx = np.argsort(x_all)
        x_sorted = x_all[sort_idx]
        y_sorted = y_all[sort_idx]
        if sem_all is not None:
            sem_sorted = sem_all[sort_idx]
        else:
            sem_sorted = None
        return x_sorted, y_sorted, sem_sorted

    def _higher_order_spline_band(x, y, sem=None, order=5, n_dense=500, log_x=False):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        sem_arr = None if sem is None else np.asarray(sem, dtype=float)

        sort_idx = np.argsort(x)
        x_sorted = x[sort_idx]
        y_sorted = y[sort_idx]
        sem_sorted = None if sem_arr is None else sem_arr[sort_idx]

        unique_x, inv = np.unique(x_sorted, return_inverse=True)
        counts = np.bincount(inv)
        y_unique = np.bincount(inv, weights=y_sorted) / counts
        if sem_sorted is not None:
            sem_unique = np.bincount(inv, weights=sem_sorted) / counts
        else:
            sem_unique = None

        if unique_x.size < 2:
            return unique_x, y_unique, sem_unique

        x_base = np.log10(unique_x) if log_x else unique_x
        x_dense_base = np.linspace(x_base.min(), x_base.max(), n_dense)

        try:
            # PCHIP avoids spline overshoot/wiggles while keeping smooth bands.
            y_dense = PchipInterpolator(x_base, y_unique)(x_dense_base)
            sem_dense = None
            if sem_unique is not None:
                sem_dense = PchipInterpolator(x_base, sem_unique)(x_dense_base)
                sem_dense = np.clip(sem_dense, 0.0, None)
            x_dense = (10 ** x_dense_base) if log_x else x_dense_base
            return x_dense, y_dense, sem_dense
        except Exception:
            return unique_x, y_unique, sem_unique

    mpl.rcParams.update({'font.size': fontsize})
    plt.rc('text', usetex=True)
    mpl.rc('text.latex', preamble=r"\usepackage{amsmath}\usepackage{newtxtext,newtxmath}")

    fig, axs = plt.subplots(2, 1, figsize=(15*ar, 15*2))
    ax_time, ax_dis = axs

    color_dict = {
        -2.71: 'black',
        -54.2: 'red',
        -125.0: 'blue',
    }

    linestyle_dict = {
        'black':'--',
        'red': '-',
        'blue': '-.'
    }

    marker_dict = {
            -2.71: 'o',
            -54.2: 's',
            -125.0: '^',
        }

    def _marker_for_delta(delta_val):
        for delta_key, marker in marker_dict.items():
            if np.isclose(np.abs(delta_val), np.abs(delta_key)):
                return marker
        return 'o'

    # ------------ panel a: time sweep ------------
    time_expt = {}
    time_theory = {}

    for idx, (task_name, timestamp, opt) in enumerate(zip(time_task_ls, time_timestamp_ls, time_opt_ls)):
        epsilon_r_ens = _safe_eps(epsilon_r_ens_time_ls, idx)
        epsilon_g_ens = _safe_eps(epsilon_g_ens_time_ls, idx)
        ensembles = _safe_ensemble(specify_ensemble_time_ls, idx)

        avg, sem, _ = run_for_N_protocol(
            task_name,
            timestamp,
            opt,
            force_recompute=force_recompute,
            force_recompute_processing=force_recompute_processing,
            shot_noise_model=shot_noise_model,
            epsilon_r_ens=epsilon_r_ens,
            epsilon_g_ens=epsilon_g_ens,
            dir_root=dir_root,
            specify_ensemble=ensembles,
            force_recompute_expt=force_recompute_expt,
            T2star_workers=T2star_workers,
            haar_opt=haar_opt,
        )

        task_payload = read_expt_task(task_name, dir_root)
        t_plateau_ls = np.asarray(task_payload[2], dtype=float)
        Delta_local = float(task_payload[6][0])

        y_avg = np.asarray(avg[0][0], dtype=float)
        y_sem = None if sem is None else np.asarray(sem[0][0], dtype=float)

        if opt == 'bloqade-expt':
            time_expt.setdefault(Delta_local, []).append((t_plateau_ls, y_avg, y_sem))
        else:
            time_theory.setdefault((opt, Delta_local), []).append((t_plateau_ls, y_avg, y_sem))

    for Delta_local, entries in time_expt.items():
        t_sorted, avg_sorted, sem_sorted = _merge_entries(entries)
        color = color_dict.get(Delta_local, 'gray')
        ax_time.errorbar(
            t_sorted,
            avg_sorted,
            yerr=sem_sorted,
            marker=marker_dict.get(Delta_local, 'o'),
            linestyle='none',
            color=color,
            markerfacecolor=color,
            markeredgecolor=color,
            markeredgewidth=2, markersize=20, ecolor=color, elinewidth=3, capsize=0,
            label=rf'$\Delta_{{\mathrm{{local}}}}={Delta_local:g}\mu \mathrm{{s}}^{{-1}}$',
        )

    for (opt, Delta_local), entries in time_theory.items():
        t_sorted, avg_sorted, sem_sorted = _merge_entries(entries)
        color = color_dict.get(Delta_local, 'gray')
        theory_tag = opt.replace('-', r'\,')
        if sem_sorted is not None:
            t_spline, avg_spline, sem_spline = _higher_order_spline_band(
                t_sorted, avg_sorted, sem_sorted, order=5, n_dense=500, log_x=False
            )
            ax_time.fill_between(
                t_spline,
                avg_spline - sem_spline,
                avg_spline + sem_spline,
                color=color,
                alpha=alpha,
                label=None,
            )
        else:
            ax_time.plot(
                t_sorted,
                avg_sorted,
                color=color,
                linewidth=3,
                label=rf'$\mathrm{{{theory_tag}}},\ \Delta_{{\mathrm{{local}}}}={Delta_local:g}\mu \mathrm{{s}}^{{-1}}$',
                linestyle=linestyle_dict.get(color, '-'),
            )

            #  axs[0].fill_between(delta_sorted, avg_sorted - sem_sorted, avg_sorted + sem_sorted, color='black', alpha=alpha, label=r'$\mathrm{Emul.\, r.c.}\; T_2\, \mathrm{with }\, T_2^*$')

    ax_time.set_xlabel(r'$t_{\mathrm{evol}}\ (\mu\mathrm{s})$')
    ax_time.set_ylabel(r'$S_{{2,A}}(t_{\mathrm{evol}})$')

    # ------------ panel b: disorder sweep ------------
    disorder_expt = {}
    disorder_theory = {}

    for idx, (task_name, timestamp, opt) in enumerate(zip(disorder_task_ls, disorder_timestamp_ls, disorder_opt_ls)):
        epsilon_r_ens = _safe_eps(epsilon_r_ens_disorder_ls, idx)
        epsilon_g_ens = _safe_eps(epsilon_g_ens_disorder_ls, idx)
        ensembles = _safe_ensemble(specify_ensemble_disorder_ls, idx)

        avg, sem, _ = run_for_N_protocol(
            task_name,
            timestamp,
            opt,
            force_recompute=force_recompute,
            force_recompute_processing=force_recompute_processing,
            shot_noise_model=shot_noise_model,
            epsilon_r_ens=epsilon_r_ens,
            epsilon_g_ens=epsilon_g_ens,
            dir_root=dir_root,
            specify_ensemble=ensembles,
            force_recompute_expt=force_recompute_expt,
            T2star_workers=T2star_workers,
            haar_opt=haar_opt,
        )

        task_payload = read_expt_task(task_name, dir_root)
        t_plateau_ls = np.asarray(task_payload[2], dtype=float)
        Delta_local_ls = np.abs(np.asarray(task_payload[6], dtype=float))
        t_idx = int(np.argmin(np.abs(t_plateau_ls - fixed_time_protocol)))

        y_avg = np.asarray([avg[0][d][t_idx] for d in range(len(Delta_local_ls))], dtype=float)
        y_sem = None
        if sem is not None:
            y_sem = np.asarray([sem[0][d][t_idx] for d in range(len(Delta_local_ls))], dtype=float)

        if opt == 'bloqade-expt':
            disorder_expt.setdefault(opt, []).append((Delta_local_ls, y_avg, y_sem))
        else:
            disorder_theory.setdefault(opt, []).append((Delta_local_ls, y_avg, y_sem))

    for opt, entries in disorder_expt.items():
        x_sorted, avg_sorted, sem_sorted = _merge_entries(entries)
        for i, (x_i, y_i) in enumerate(zip(x_sorted, avg_sorted)):
            marker_i = _marker_for_delta(x_i)
            if sem_sorted is not None:
                yerr_i = [sem_sorted[i]]
            else:
                yerr_i = None
            ax_dis.errorbar(
                [x_i],
                [y_i],
                yerr=yerr_i,
                marker='o',
                linestyle='none',
                color='black',
                markerfacecolor='black',
                markeredgecolor='black',
                markeredgewidth=2, markersize=20, ecolor='black', elinewidth=3, capsize=0,
                label=r'$\mathrm{Aquila}$' if i == 0 else None,
            )

    for opt, entries in disorder_theory.items():
        x_sorted, avg_sorted, sem_sorted = _merge_entries(entries)
        theory_tag = opt.replace('-', r'\,')
        if sem_sorted is not None:
            x_spline, avg_spline, sem_spline = _higher_order_spline_band(
                x_sorted, avg_sorted, sem_sorted, order=5, n_dense=500, log_x=True
            )
            ax_dis.fill_between(
                x_spline,
                avg_spline - sem_spline,
                avg_spline + sem_spline,
                color='black',
                alpha=alpha,
                label=rf'$\mathrm{{{theory_tag}}}$',
            )
        else:
            ax_dis.plot(
                x_sorted,
                avg_sorted,
                color='black',
                linewidth=3,
                label=rf'$\mathrm{{{theory_tag}}}$',
            )

    ax_dis.set_xscale('log')
    ax_dis.set_xlabel(r'$|\Delta_{\mathrm{local}}|\ (\mu\mathrm{s}^{-1})$')
    ax_dis.set_ylabel(rf'$S_{{2,A}}(t_{{\mathrm{{evol}}}}={fixed_time_protocol:.5g}\ \mu\mathrm{{s}})$')

    if use_bc:
        ax_time.text(-0.15, 1.05, r'$\sf{\textbf{b}}$', transform=ax_time.transAxes, fontsize=fontsize * 1.2)
        ax_dis.text(-0.15, 1.05, r'$\sf{\textbf{c}}$', transform=ax_dis.transAxes, fontsize=fontsize * 1.2)
    else:
        ax_time.text(-0.15, 1.05, r'$\sf{\textbf{a}}$', transform=ax_time.transAxes, fontsize=fontsize * 1.2)
        ax_dis.text(-0.15, 1.05, r'$\sf{\textbf{b}}$', transform=ax_dis.transAxes, fontsize=fontsize * 1.2)

    ax_time.legend(loc='lower right', bbox_to_anchor=(1.02, -0.045), fontsize=fontsize * 0.8, frameon=False)


   

    for ax in axs:
        style_axis(ax, fontsize=fontsize)
        # ax.set_box_aspect(1)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.25)

    payload = {
        'time_task_ls': time_task_ls,
        'time_timestamp_ls': time_timestamp_ls,
        'time_opt_ls': time_opt_ls,
        'disorder_task_ls': disorder_task_ls,
        'disorder_timestamp_ls': disorder_timestamp_ls,
        'disorder_opt_ls': disorder_opt_ls,
        'fixed_time_protocol': fixed_time_protocol,
        'shot_noise_model': shot_noise_model,
    }

    manager = ExptStore(dir_root)
    puid, _ = manager.add(payload, timestamp=0)

    os.makedirs(os.path.join(dir_root, 'results'), exist_ok=True)
    plot_filepath = os.path.join(dir_root, 'results', f'time_disorder_{puid}.pdf')
    plt.savefig(plot_filepath)
    print(f"Saved plot to {plot_filepath}")

    return axs


def get_avg_readouterror_helper(tasks, dir_root, N=6):
    epsilon_r_ls_ls = []
    epsilon_g_ls_ls = []
    for task in tasks:
       
        h_ls_pre, x_pre, t_plateau_ls, seq_ls_pre_all, base_params, Delta_mean_ls, Delta_local_ls, gate_params_all, cluster_spacing, manual_parallelization, override_local, x0_y0_offset, t_delay, start_Delta_from0, uniform_Omega_Delta_ramp ,phi_opt, Delta_local_ramp_time, Omega_delay_time = read_expt_task(task, dir_root)
        num_ens = len(h_ls_pre)
        epsilon_r_ls, epsilon_r_unc_ls, epsilon_g_ls, epsilon_g_unc_ls, Omega_ls, Omega_unc_ls = extract_readouterror_rabi(dir_root, task, num_ens, N=N, force_recompute=False)

        epsilon_r_ls_ls.append(epsilon_r_ls)
        epsilon_g_ls_ls.append(epsilon_g_ls)

    # compute avg across tasks and ensambles
    # print(epsilon_r_ls_ls)
    epsilon_r_ls_ls = np.array(epsilon_r_ls_ls)
    epsilon_g_ls_ls = np.array(epsilon_g_ls_ls)
    
    epsilon_r_ls_avg = np.mean(epsilon_r_ls_ls, axis=(0,1))
    epsilon_g_ls_avg = np.mean(epsilon_g_ls_ls, axis=(0,1))
    return epsilon_r_ls_avg, epsilon_g_ls_avg


def _resolve_tasks_dir(tasks_root):
    nested_tasks_dir = os.path.join(tasks_root, "tasks")
    if os.path.isdir(nested_tasks_dir):
        return nested_tasks_dir
    return tasks_root


def _stage_task_bundle_for_dir_root(task_name, all_tasks_dir, dir_root):
    src_tasks_dir = _resolve_tasks_dir(all_tasks_dir)
    dst_tasks_dir = os.path.join(dir_root, "tasks")
    os.makedirs(dst_tasks_dir, exist_ok=True)

    src_task_path = os.path.join(src_tasks_dir, f"{task_name}.json")
    if not os.path.exists(src_task_path):
        raise FileNotFoundError(f"Task file not found: {src_task_path}")

    shutil.copy2(src_task_path, os.path.join(dst_tasks_dir, f"{task_name}.json"))

    with open(src_task_path, "r") as f:
        task_payload = json.load(f)

    stem_name = task_payload.get("stem")
    if stem_name is None:
        raise KeyError(f"Task {task_name} is missing a 'stem' field")

    stem_filename = stem_name if stem_name.endswith(".json") else f"{stem_name}.json"
    src_stem_path = os.path.join(src_tasks_dir, stem_filename)
    if not os.path.exists(src_stem_path):
        raise FileNotFoundError(f"Stem file not found: {src_stem_path}")

    shutil.copy2(src_stem_path, os.path.join(dst_tasks_dir, stem_filename))


def _cpu_ok_for_submit(max_load_fraction=0.75):
    n_cpu = os.cpu_count() or 1
    threshold = max_load_fraction * n_cpu

    try:
        load_1min, _, _ = os.getloadavg()
    except (AttributeError, OSError):
        # getloadavg is unavailable on Windows.
        return True, None, threshold, n_cpu

    return load_1min <= threshold, load_1min, threshold, n_cpu

def _launch_run_for_N_protocol_in_new_terminal(
    task_name,
    submit_timestamp,
    submit_opt,
    submit_epsilon_r_ens,
    submit_epsilon_g_ens,
    task_dir_root,
    specify_ensemble,
    force_recompute=False,
    force_recompute_processing=False,
    T2star_workers=2,
    conda_env_name=None,
):
    module_name = os.path.splitext(os.path.basename(__file__))[0]
    script_dir = os.path.dirname(os.path.abspath(__file__))

    py_inline = (
        f"from {module_name} import run_for_N_protocol; "
        f"run_for_N_protocol({task_name!r}, {submit_timestamp!r}, {submit_opt!r}, "
        f"epsilon_r_ens={submit_epsilon_r_ens!r}, "
        f"epsilon_g_ens={submit_epsilon_g_ens!r}, "
        f"dir_root={task_dir_root!r}, "
        f"specify_ensemble={specify_ensemble!r}, "
        f"force_recompute={force_recompute!r}, "
        f"force_recompute_processing={force_recompute_processing!r}, "
        f"T2star_workers={T2star_workers!r}, "
        f"save_one_at_a_time=True)"
    )

    if sys.platform.startswith("win"):
        conda_env = conda_env_name or os.environ.get("CONDA_DEFAULT_ENV", "base")
        conda_exe = os.environ.get("CONDA_EXE")
        conda_bat = None

        if conda_exe:
            conda_exe_norm = os.path.normpath(conda_exe)
            if conda_exe_norm.lower().endswith("conda.bat") and os.path.exists(conda_exe_norm):
                conda_bat = conda_exe_norm
            else:
                conda_root = os.path.dirname(os.path.dirname(conda_exe_norm))
                conda_bat_candidate = os.path.join(conda_root, "condabin", "conda.bat")
                if os.path.exists(conda_bat_candidate):
                    conda_bat = conda_bat_candidate

        # Write a temporary batch file so there's no quoting ambiguity
        bat_lines = ["@echo off"]
        if conda_bat is not None:
            bat_lines.append(f'call "{conda_bat}" activate {conda_env}')
        else:
            bat_lines.append(f"conda activate {conda_env} 2>nul")

        bat_lines.append(f'cd /d "{script_dir}"')
        bat_lines.append(f"python -c {json.dumps(py_inline)}")
        bat_lines.append("pause")

        bat_path = os.path.join(task_dir_root, f"_run_{task_name}.bat")
        with open(bat_path, "w") as f:
            f.write("\r\n".join(bat_lines))

        creationflags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
        subprocess.Popen(
            [os.environ.get("COMSPEC", "cmd.exe"), "/K", bat_path],
            creationflags=creationflags,
        )
        return

def main_disorder_sweep(err_show=None):

    ## opt for err_show: None, T2, T2star, T2+T2star

    dir_root = "alexei_new_demo"
    J = 5.42
    Delta_mean = 0.5 * J
    Omega = 15.8
    a = 10
    phi = 0

    # Delta_local_ls_expt = -np.logspace(np.log10(0.2*J), np.log10(50), 10) 
    # Delta_local_ls_expt = -np.logspace(np.log10(2.6), np.log10(90), 15)
    # Delta_local_ls_expt= -np.array([ 2.71      ,  4.46485772,  7.35607174, 12.11948844, 19.96745072,32.89735292, 54.2, 89.6       ])
    # Delta_local_ls_expt = np.array([-125.0])
    # Delta_local_ls_sim = -np.logspace(np.log10(2), np.log10(130), 100)
    N = 6
    n_shots=200 
    n_gates = 16
    
    t_ramp = 0.0632
    Delta_local_ramp_time = 0.05
    Omega_delay_time=0.0 
    f=1

    # theory_task = "task_5553e380b975f263a28c0489" # 20 ens, 20 n_U
    theory_task = "task_523aeb4b3ddedda027454536" # 25 ens, 20 n_U

    theory_expt1 = "task_54bfa61a1b8dee273dc7c6b0" # 15 ens, 15 n_U. first 5 D_l
    theory_expt2 = "task_58c44610f6f1ffbb74b7617e" # 20 ens, 20 n_U. second 5 D_l
    expt_actual_complete = "task_ad35cfe78828005ff7447358" # all the Delta_local vals used across the expts, 15 ens and 20 n_U
    expt_sim_15_20 = "task_f420fa788f5735e90edde8bd"
    sim_sim_15_20 = "task_09af5353849158b99828cd69"
    # last_point_20 = "task_94063ee7805e9261215157d7"
    # last_point_25 = "task_49b401a645289c98d7cb31a4"
    
    sim_2_130 = "task_bb341ece5151bf4f0ab41477"
    sim_2_130_15_20 = "task_851e8f5e0ebb5bcfab2aca74"

    expt_task_name_005 = "task_ddebfeaea3ca63079a4c10c4" # chaotic, 6 qubits
    expt_task_name_10 = "task_15eaace71ebd595814b7bd17" # localized, 6 qubits
    expt_task_125_wo2125 = "task_1f671c4d079b68342b55451a"

    middle_points_tasks = ["task_1a1b10d5afb4687600e767ba", "task_7d1f0dceecf7ebcce140df99", "task_d6f35628b84d6df294fdd9a5", "task_1b3d62cb0a9c01c1b9d9e45b", "task_3e46efcb6c6022b9a937b744", "task_01cf9244e6858929c19ebffa"] 
    middle_points_timestamps = [1771503922] * len(middle_points_tasks)

    expt_125_2125_plateau = "task_d0c2eadd5b392ec6c313c9a3"
    expt_125_2125_plateau_timestamp = 1772151351


    timestamp_005 = 1766200542 ## CHANGE FOR EXPT
    timestamp_10 = 1768489892

    D2d71_T2d125 = "task_054dbed9870a05386fd07848" # peak
    D54d2_T2d125 = "task_6ddfe8a7f8b504ed1700e53d"

    Dsweep_noerr = "task_5d19e26015087bf320a09546"




    t_fixed = [2.125]

    num_ens = 20
    # n_U = 25

    get_task = False
    # same_h_ls_task_ls=[expt_task_name_005, expt_task_name_10]
    J = 5.42
    # Delta_local_ls_ls = [-J/2, -10*J]
    # n_U_ls = [15, 20]
    # t_ls = [0.05, 0.25, 0.625, 1.0, 1.375, 1.75, 2.125, 2.5, 4.0]
    same_h_ls_task_ls=[expt_task_125_wo2125]
    Delta_local_ls_ls = [-125.0]
    n_U_ls = [25]
    t_ls = [0.05, 0.25, 0.625, 1.0, 1.375, 1.75, 2.5, 4.0]

    if get_task:
        for same_h_ls_task, n_U, Delta_local_ls in zip(same_h_ls_task_ls, n_U_ls, Delta_local_ls_ls):
            for t in t_ls:
                print("n_U:", n_U, "Delta_local_ls:", Delta_local_ls, "t:", t)
                _get_task(N, [Delta_local_ls,], [t], num_ens ,Delta_mean, a, Omega, phi, n_shots, n_gates, n_U, t_ramp = t_ramp, Delta_local_ramp_time = Delta_local_ramp_time, Omega_delay_time=Omega_delay_time, f=f, dir_root=dir_root, same_h_ls_task=same_h_ls_task)

    D2d71_T2d125 = "task_054dbed9870a05386fd07848" # peak
    D54d2_T2d125 = "task_6ddfe8a7f8b504ed1700e53d"

    expt_tasks = middle_points_tasks + [expt_125_2125_plateau] + [expt_task_name_005, expt_task_name_10]
    single_time_pts=[D2d71_T2d125, D54d2_T2d125]

    timestamp_ls = middle_points_timestamps + [expt_125_2125_plateau_timestamp] + [timestamp_005, timestamp_10] 

    # if err_show == 'T2+T2star':
    #     task_ls = expt_tasks + middle_points_tasks + [expt_125_2125_plateau] + single_time_pts + [Dsweep_noerr]*2
    #     timestamp_ls += [0] * (len(single_time_pts)  + len(middle_points_tasks) + 3)
    # elif err_show is None:
    #     task_ls = expt_tasks + [Dsweep_noerr]*2
    #     timestamp_ls += [0] * 2
    # else:
    #     raise NotImplementedError(f"err_show {err_show} not implemented")

    T2star_ls = middle_points_tasks  + [expt_125_2125_plateau, ] + single_time_pts
    # 
    task_ls = expt_tasks + T2star_ls + [Dsweep_noerr]*2
    timestamp_ls += [0] * (len(T2star_ls) + 2)


    # + [expt_125_2125_plateau]
    # [sim_2_130_15_20, sim_2_130]
                # sim_2_130]
    # NEED TO RUN sim_2_130
    # +[0]*(len(middle_points_timestamps)+3)
                                                                                # +[0, 0] 

    opt_ls = ['bloqade-expt'] * len(expt_tasks) + ['T2star-sim-rc'] * len(T2star_ls) + [ 'num', 'bloqade-sim-no-rc'] 
    # if err_show == 'T2star':
    #     opt_ls += ['T2star-sim-rc'] * len(expt_tasks)
    # elif err_show == 'T2':
    #     opt_ls += ['T2-sim-rc'] * len(expt_tasks)
    # elif err_show == 'T2+T2star':
    #     opt_ls += ['T2-T2star-sim-rc'] * (len(single_time_pts)  + len(middle_points_tasks) + 1)




    print("task_ls:", task_ls)
    print("timestamp_ls:", timestamp_ls)
    print("opt_ls:", opt_ls)


    epsilon_r_ens_ls = [0.1] * len(task_ls)
    epsilon_g_ens_ls = [0.05] * len(task_ls)

    # epsilon_r_ls_avg, epsilon_g_ls_avg = get_avg_readouterror_helper(expt_tasks, dir_root, N=N)
    specify_ensemble_ls = [list(range(15)) for _ in range(len(task_ls))]
    # specify_ensemble_ls[-1] = list(range(20))

    # for i in range(1, len(middle_points_timestamps)   +1, 1):
    #     print(f"Setting specify_ensemble_ls[{-i}] to [0], has options {specify_ensemble_ls[-i]}")
    #     specify_ensemble_ls[-i-3] = list(range(5))

    # for i in range(1, 4):
    #     print(f"Setting specify_ensemble_ls[{-i}] to [0], has options {specify_ensemble_ls[-i]}")

    # specify_ensemble_ls[-1] = [0]
    # specify_ensemble_ls[-2] = [0]

    print(specify_ensemble_ls)
   
    t0 = time.time()
    print(f"Starting entropy sweep protocol at {t0}")
    entropy_sweep_protocol(task_ls, timestamp_ls, opt_ls, fontsize=20, force_recompute=False, force_recompute_processing=False, epsilon_r_ens_ls = epsilon_r_ens_ls, epsilon_g_ens_ls = epsilon_g_ens_ls, dir_root=dir_root, specify_ensemble_ls = specify_ensemble_ls, alpha=0.3, force_recompute_expt=False)
    tf = time.time()
    print(f"Finished entropy sweep protocol at {tf}, total time {(tf-t0)/60:.1f} min")  


def main_time_sweep(err_show=None):
    ## opt for err_show: None, T2, T2star, T2+T2star

    dir_root = "alexei_new_demo"
    J = 5.42
    Delta_mean = 0.5 * J
    Omega = 15.8
    a = 10
    phi = 0

    # Delta_local_ls_expt = -np.logspace(np.log10(0.2*J), np.log10(50), 10) 
    # Delta_local_ls_expt = -np.logspace(np.log10(2.6), np.log10(90), 15)
    # Delta_local_ls_expt= -np.array([ 2.71      ,  4.46485772,  7.35607174, 12.11948844, 19.96745072,32.89735292, 54.2, 89.6       ])
    
    # Delta_local_ls_sim = -np.logspace(np.log10(2), np.log10(130), 100)
    N = 6
    n_shots=200 
   
    n_gates = 16
    
    t_ramp = 0.0632
    Delta_local_ramp_time = 0.05
    Omega_delay_time=0.0 
    f=1

    # theory_task = "task_5553e380b975f263a28c0489" # 20 ens, 20 n_U
    theory_task = "task_523aeb4b3ddedda027454536" # 25 ens, 20 n_U

    theory_expt1 = "task_54bfa61a1b8dee273dc7c6b0" # 15 ens, 15 n_U. first 5 D_l
    theory_expt2 = "task_58c44610f6f1ffbb74b7617e" # 20 ens, 20 n_U. second 5 D_l
    expt_actual_complete = "task_ad35cfe78828005ff7447358" # all the Delta_local vals used across the expts, 15 ens and 20 n_U
    expt_sim_15_20 = "task_f420fa788f5735e90edde8bd"
    sim_sim_15_20 = "task_09af5353849158b99828cd69"
    # last_point_20 = "task_94063ee7805e9261215157d7"
    # last_point_25 = "task_49b401a645289c98d7cb31a4"
    sim_2_130 = "task_bb341ece5151bf4f0ab41477"

    expt_task_name_005 = "task_ddebfeaea3ca63079a4c10c4" # chaotic, 6 qubits
    expt_task_name_10 = "task_15eaace71ebd595814b7bd17" # localized, 6 qubits
    expt_task_125_wo2125 = "task_1f671c4d079b68342b55451a"

    middle_points_tasks = ["task_1a1b10d5afb4687600e767ba", "task_7d1f0dceecf7ebcce140df99", "task_d6f35628b84d6df294fdd9a5", "task_1b3d62cb0a9c01c1b9d9e45b", "task_3e46efcb6c6022b9a937b744", "task_01cf9244e6858929c19ebffa"] 
    middle_points_timestamps = [1771503922] * len(middle_points_tasks)

    expt_125_2125_plateau = "task_d0c2eadd5b392ec6c313c9a3"
    expt_125_2125_plateau_timestamp = 1772151351
    timestamp_125_wo2125 = 1772408252


    D2d71_T0d05 = "task_5692cdce3a76e63ae3a772f2"
    D2d71_T0d525 = "task_40b2edc8e4622b0d21eaf6b9" ## NEW EXPT
    D2d71_T1d0 = "task_42c55ad726b9f94bdc158967"
    D2d71_T1d375 = "task_0f1bf50951dfaf5fe6bf382c"
    D2d71_T1d75 = "task_bf69383ab41fca45292efca4"
    D2d71_T2d125 = "task_054dbed9870a05386fd07848" # peak
    D2d71_T2d5 = "task_bdca1f5f9559b38d6ac220b2"
    D2d71_T4d0 = "task_2b83a0503df2abfe4e8accdd"
    D2d71_ls = [D2d71_T0d05, D2d71_T0d525, D2d71_T1d0, D2d71_T1d375, D2d71_T1d75, D2d71_T2d125, D2d71_T2d5, D2d71_T4d0]

    D54d2_T0d05 = "task_59e554b7e98b1e23555a8a00"
    D54d2_T0d525 = "task_28501f55cc7b14b219747910" ## NEW EXPT
    D54d2_T1d0 = "task_16733ecbd4817de7c7e3f95b"
    D54d2_T1d375 = "task_e540de6a491931e662c259ce"
    D54d2_T1d75 = "task_45e873589dcb074adcc41fd1"
    D54d2_T2d125 = "task_6ddfe8a7f8b504ed1700e53d"
    D54d2_T2d5 = "task_93801ef64613efee3f8c36db"
    D54d2_T4d0 = "task_b7dfe54c4458db0da5430c22"
    D54d2_ls = [D54d2_T0d05, D54d2_T0d525, D54d2_T1d0, D54d2_T1d375, D54d2_T1d75, D54d2_T2d125, D54d2_T2d5, D54d2_T4d0]
    

    D125d0_T0d05 = "task_00315ea0d133f9be486455ea"
    D125d0_T0d525 = "task_6721a72b5d44f6896710f07f" ## NEW EXPT
    D125d0_T1d0 = "task_c2db90fc4042c2bf1fd8c4b3"
    D125d0_T1d375 = "task_bc9e6c1244ccb355e3ae069f"
    D125d0_T1d75 = "task_c5c247b32831797f463cc543"
    D125d0_T2d125 = "task_d0c2eadd5b392ec6c313c9a3" # peak
    D125d0_T2d5 = "task_fc852096a824ad782ad34d82"
    D125d0_T4d0 = "task_0e0f300cc516e6f67fe37838"
    D125d0_ls = [D125d0_T0d05, D125d0_T0d525, D125d0_T1d0, D125d0_T1d375, D125d0_T1d75, D125d0_T2d125, D125d0_T2d5, D125d0_T4d0]

    
    
    D2d71_t0d525_timestamp = 1774626495
    D54d2_t0d525_timestamp = 1774731710
    D125d0_t0d525_timestamp = 1774866671


    sim_125_allt = "task_5fb40b8e1eb9f1cc8956bd30" # demo5a
    sim_542_allt = "task_ed9d7eee9317b5a644e4201f" # demo6a
    sim_271_allt = "task_a7d98fcffb940592e5f38f77" # demo7a

    one_ham_chaotic = "task_1d092a93d800af23a95f1c7e" # demo11
    chaotic_expt_20timepts = "task_dc5c98f11ff4d89e87af2917" # demo12

    ## 200 time points for numerics
    num_271 = "task_e1863e3d088dfe12500a4d77"
    num_542 = "task_5cd0bfd9c4fc13d02734571e"
    num_125 = "task_c96b150f2fd48db682f65b4b"

    D2d71_no_err = "task_556886b14a2b72fed9c5e05b"
    D125d0_no_err = "task_674ce88dad3fbfd139509d1e"
    D54d2_no_err = "task_67d6229bc32f0413e3ef6e1e"

    D2d71_no_err_100 = "task_f8ac2fc4dc3a338ae2024c9b"
    D54d2_no_err_100 = "task_d27a40c4019f7c6edb9deb71"
    D125d0_no_err_100 = "task_e41d237d81b51a754185eb0d"

    D125d0_nU30_nens30 = "task_103fcd6030475edaa308c672"


    timestamp_005 = 1766200542 ## CHANGE FOR EXPT
    timestamp_10 = 1768489892
    t_fixed = list(np.linspace(0.05, 4.1, 200)) ## LOWER THIS FOR BLOQADE SIM

    get_task = False
    same_h_ls_task = expt_task_125_wo2125
    Delta_local_ls_expt = [-125.0]
    num_ens = 20
    n_U = 25

    if get_task:
        _get_task(N, Delta_local_ls_expt, t_fixed, num_ens ,Delta_mean, a, Omega, phi, n_shots, n_gates, n_U, t_ramp = t_ramp, Delta_local_ramp_time = Delta_local_ramp_time, Omega_delay_time=Omega_delay_time, f=f, dir_root=dir_root, same_h_ls_task= same_h_ls_task)

    # task_ls = [theory_task]
    # timestamp_ls = [0]
    # opt_ls = ['bloqade-sim-no-rc']

    # task_ls = [theory_task, theory_task, expt_task_name_005, expt_task_name_10]
    # timestamp_ls = [0, 0, timestamp_005, timestamp_10]
    # opt_ls = ['num', 'bloqade-sim-no-rc', 'bloqade-expt', 'bloqade-expt']

    # [expt_sim_15_20]
    # + [sim_125_allt, sim_542_allt, sim_271_allt]\

    # localized_copy = "task_6170712a74d8f7fa845f8a47"

    expt_tasks = [expt_task_name_005, expt_task_name_10, expt_125_2125_plateau, expt_task_125_wo2125,  D2d71_T0d525, D54d2_T0d525, D125d0_T0d525] 
    other_tasks = [num_125, num_542, num_271, D2d71_no_err, D54d2_no_err,D125d0_no_err ]
    #  


    T2star_err_ls = D2d71_ls + D54d2_ls + D125d0_ls
    # D125d0_ls
    # 
    # D2d71_ls
    # + D54d2_ls + D125d0_ls

    task_ls =  expt_tasks  + T2star_err_ls  + other_tasks
    # +  other_tasks

    timestamp_ls = [timestamp_005, timestamp_10, expt_125_2125_plateau_timestamp, timestamp_125_wo2125, D2d71_t0d525_timestamp, D54d2_t0d525_timestamp, D125d0_t0d525_timestamp] \
    + [0] * (len(other_tasks) + len(T2star_err_ls))
    
    opt_ls = ['bloqade-expt'] * len(expt_tasks) + ['T2star-sim-rc'] * len(T2star_err_ls) + ['num'] * 3 + ['bloqade-sim-no-rc'] * 3
    

    print("task_ls:", task_ls)
    print("timestamp_ls:", timestamp_ls)
    print("opt_ls:", opt_ls)



    # epsilon_r_ls005, epsilon_r_unc_ls005, epsilon_g_ls005, epsilon_g_unc_ls005, Omega_ls_005, Omega_unc_ls_005 = extract_readouterror_rabi(dir_root, expt_task_name_005, 15, N=6, force_recompute=False)
    # epsilon_r_ls10, epsilon_r_unc_ls10, epsilon_g_ls10, epsilon_g_unc_ls10, Omega_ls_10, Omega_unc_ls_10 = extract_readouterror_rabi(dir_root, expt_task_name_10, 20, N=6, force_recompute=False)
    # epsilon_r_ls_125_wo2125, epsilon_r_unc_ls_125_wo2125, epsilon_g_ls_125_wo2125, epsilon_g_unc_ls_125_wo2125, Omega_ls_125_wo2125, Omega_unc_ls_125_wo2125 = extract_readouterror_rabi(dir_root, expt_task_125_wo2125, 20, N=6, force_recompute=False)


    # epsilon_r_ens_ls = [0.0, 0.0, 0.0, 0.0, epsilon_r_ls005, epsilon_r_ls10, epsilon_r_ls_125_wo2125] + [0] * 3 + [epsilon_r_ls005, epsilon_r_ls10, epsilon_r_ls_125_wo2125]
    # epsilon_g_ens_ls = [0.0, 0.0, 0.0, 0.0, epsilon_g_ls005, epsilon_g_ls10, epsilon_g_ls_125_wo2125] + [0] * 3 + [epsilon_g_ls005, epsilon_g_ls10, epsilon_g_ls_125_wo2125]
    # epsilon_r_ens_ls = [0.1] * len(task_ls)
    # epsilon_g_ens_ls = [0.01] * len(task_ls)


    epsilon_r_ens_ls = [0.1] * len(task_ls)
    epsilon_g_ens_ls = [0.05] * len(task_ls)


    # specify_ensemble_ls = [list(range(15-6)) for _ in range(len(task_ls))] + [None] * 6
    specify_ensemble_ls = [list(range(15)) for _ in range(len(task_ls))]
    # for n in range(1, 6+1, 1):
    #     specify_ensemble_ls[-n] = [0,1,2]
    # specify_ensemble_ls[-1] = [0]




    # specify_ensemble_ls[4] = list(range(7))
    # specify_ensemble_ls[-6:-1] = [None] * 6
    # specify_ensemble_ls[-2] = None

    # for i in range(1, 4+1, 1):
    #     print(f"Setting specify_ensemble_ls[{-i}] to..")
    #     specify_ensemble_ls[-i] = [0]

    # specify_ensemble_ls[-3] = list(range(6))
    # epsilon_g_ens_ls[-3] = [0.1]

    

    t0 = time.time()
    print(f"Starting entropy sweep protocol at {t0}")
    print(specify_ensemble_ls)
    time_sweep_protocol(task_ls, timestamp_ls, opt_ls, fontsize=20, force_recompute=False, force_recompute_processing=False, epsilon_r_ens_ls = epsilon_r_ens_ls, epsilon_g_ens_ls = epsilon_g_ens_ls, dir_root=dir_root, specify_ensemble_ls = specify_ensemble_ls, alpha=0.3, force_recompute_expt=False, T2star_workers=12)
    tf = time.time()
    print(f"Finished entropy sweep protocol at {tf}, total time {(tf-t0)/60:.1f} min")  


def main_expt_figs(opt='expt',  dir_root = "paper_main_data"):

    # dir_root = "alexei_new_demo"
   


    if opt=='expt':
        D2d71_t0d525_timestamp = 1774626495
        D54d2_t0d525_timestamp = 1774731710
        D125d0_t0d525_timestamp = 1774866671

        expt_125_2125_plateau_timestamp = 1772151351
        timestamp_125_wo2125 = 1772408252

        timestamp_005 = 1766200542 ## CHANGE FOR EXPT
        timestamp_10 = 1768489892

        expt_task_name_005 = "task_ddebfeaea3ca63079a4c10c4" # chaotic, 6 qubits
        expt_task_name_10 = "task_15eaace71ebd595814b7bd17" # localized, 6 qubits
        expt_task_125_wo2125 = "task_1f671c4d079b68342b55451a"
        expt_125_2125_plateau = "task_d0c2eadd5b392ec6c313c9a3"
        
        D2d71_T0d05 = "task_5692cdce3a76e63ae3a772f2"
        D2d71_T0d525 = "task_40b2edc8e4622b0d21eaf6b9" ## NEW EXPT
        D2d71_T1d0 = "task_42c55ad726b9f94bdc158967"
        D2d71_T1d375 = "task_0f1bf50951dfaf5fe6bf382c"
        D2d71_T1d75 = "task_bf69383ab41fca45292efca4"
        D2d71_T2d125 = "task_054dbed9870a05386fd07848" # peak
        D2d71_T2d5 = "task_bdca1f5f9559b38d6ac220b2"
        D2d71_T4d0 = "task_2b83a0503df2abfe4e8accdd"
        D2d71_ls = [D2d71_T0d05, D2d71_T0d525, D2d71_T1d0, D2d71_T1d375, D2d71_T1d75, D2d71_T2d125, D2d71_T2d5, D2d71_T4d0]

        D54d2_T0d05 = "task_59e554b7e98b1e23555a8a00"
        D54d2_T0d525 = "task_28501f55cc7b14b219747910" ## NEW EXPT
        D54d2_T1d0 = "task_16733ecbd4817de7c7e3f95b"
        D54d2_T1d375 = "task_e540de6a491931e662c259ce"
        D54d2_T1d75 = "task_45e873589dcb074adcc41fd1"
        D54d2_T2d125 = "task_6ddfe8a7f8b504ed1700e53d"
        D54d2_T2d5 = "task_93801ef64613efee3f8c36db"
        D54d2_T4d0 = "task_b7dfe54c4458db0da5430c22"
        D54d2_ls = [D54d2_T0d05, D54d2_T0d525, D54d2_T1d0, D54d2_T1d375, D54d2_T1d75, D54d2_T2d125, D54d2_T2d5, D54d2_T4d0]
        

        D125d0_T0d05 = "task_00315ea0d133f9be486455ea"
        D125d0_T0d525 = "task_6721a72b5d44f6896710f07f" ## NEW EXPT
        D125d0_T1d0 = "task_c2db90fc4042c2bf1fd8c4b3"
        D125d0_T1d375 = "task_bc9e6c1244ccb355e3ae069f"
        D125d0_T1d75 = "task_c5c247b32831797f463cc543"
        D125d0_T2d125 = "task_d0c2eadd5b392ec6c313c9a3" # peak
        D125d0_T2d5 = "task_fc852096a824ad782ad34d82"
        D125d0_T4d0 = "task_0e0f300cc516e6f67fe37838"
        D125d0_ls = [D125d0_T0d05, D125d0_T0d525, D125d0_T1d0, D125d0_T1d375, D125d0_T1d75, D125d0_T2d125, D125d0_T2d5, D125d0_T4d0]



        time_expt_task_ls = [expt_task_name_005, expt_task_name_10, expt_125_2125_plateau, expt_task_125_wo2125,  D2d71_T0d525, D54d2_T0d525, D125d0_T0d525] 
        time_T2star_task_ls = D2d71_ls + D54d2_ls + D125d0_ls
        time_task_ls = time_expt_task_ls + time_T2star_task_ls

        time_timestamp_ls = [timestamp_005, timestamp_10, expt_125_2125_plateau_timestamp, timestamp_125_wo2125, D2d71_t0d525_timestamp, D54d2_t0d525_timestamp, D125d0_t0d525_timestamp]  + [0] * (len(D2d71_ls) + len(D54d2_ls) + len(D125d0_ls))

        time_opt_ls = ['bloqade-expt'] * len(time_expt_task_ls) + ['T2star-sim-rc'] * (len(time_T2star_task_ls))


        specify_ensemble_time_ls = [list(range(15)) for _ in range(len(time_task_ls))]


        middle_points_tasks = ["task_1a1b10d5afb4687600e767ba", "task_7d1f0dceecf7ebcce140df99", "task_d6f35628b84d6df294fdd9a5", "task_1b3d62cb0a9c01c1b9d9e45b", "task_3e46efcb6c6022b9a937b744", "task_01cf9244e6858929c19ebffa"] 
        middle_points_timestamps = [1771503922] * len(middle_points_tasks)


        disorder_expt_tasks = middle_points_tasks + [expt_125_2125_plateau] + [expt_task_name_005, expt_task_name_10]
        disorder_T2star_tasks = middle_points_tasks + [expt_125_2125_plateau, D2d71_T2d125, D54d2_T2d125]


        disorder_task_ls =  disorder_expt_tasks + disorder_T2star_tasks

        disorder_timestamp_ls = middle_points_timestamps + [expt_125_2125_plateau_timestamp] + [timestamp_005, timestamp_10] + [0] * (len(disorder_T2star_tasks))

        disorder_opt_ls = ['bloqade-expt'] * len(disorder_expt_tasks) + ['T2star-sim-rc'] * len(disorder_T2star_tasks)

        specify_ensemble_disorder_ls = [list(range(15)) for _ in range(len(disorder_task_ls))]

        epsilon_r_ens_time_ls = [0.1] * len(time_task_ls)
        epsilon_g_ens_time_ls = [0.05] * len(time_task_ls)
        epsilon_r_ens_disorder_ls = [0.1] * len(disorder_task_ls)
        epsilon_g_ens_disorder_ls = [0.05] * len(disorder_task_ls)

        use_bc = True
        

    elif opt=='sim':
        Dsweep_noerr = "task_5d19e26015087bf320a09546"
        disorder_task_ls = [Dsweep_noerr]*2
        disorder_timestamp_ls = [0, 0]
        disorder_opt_ls = ['num', 'bloqade-sim-no-rc']

        num_271 = "task_e1863e3d088dfe12500a4d77"
        num_542 = "task_5cd0bfd9c4fc13d02734571e"
        num_125 = "task_c96b150f2fd48db682f65b4b"

        D2d71_no_err = "task_556886b14a2b72fed9c5e05b"
        D125d0_no_err = "task_674ce88dad3fbfd139509d1e"
        D54d2_no_err = "task_67d6229bc32f0413e3ef6e1e"

        Dn125d0_nU30_nens30 = "task_103fcd6030475edaa308c672"
        Dn54d2_nU20_nens20 = "task_47d61f147d41fd88d8deeff8"

        time_task_ls = [num_125, num_542, num_271, D2d71_no_err, D54d2_no_err,D125d0_no_err ]

        time_timestamp_ls = [0] * len(time_task_ls)
        time_opt_ls = ['num'] *3 + ['bloqade-sim-no-rc'] * 3

        specify_ensemble_time_ls = [list(range(15)) for _ in range(len(time_task_ls))]
        specify_ensemble_disorder_ls = [list(range(15)) for _ in range(len(disorder_task_ls))]
        epsilon_r_ens_time_ls = [0.1] * len(time_task_ls)
        epsilon_g_ens_time_ls = [0.05] * len(time_task_ls)
        epsilon_r_ens_disorder_ls = [0.1] * len(disorder_task_ls)
        epsilon_g_ens_disorder_ls = [0.05] * len(disorder_task_ls)
        use_bc = False
        

    else:
        raise ValueError(f"opt {opt} not recognized")
        


    time_disorder_protocol(
    time_task_ls,
    time_timestamp_ls,
    time_opt_ls,
    disorder_task_ls,
    disorder_timestamp_ls,
    disorder_opt_ls,
    fontsize=60,
    force_recompute=False,
    force_recompute_processing=False,
    shot_noise_model='multinomial',
    epsilon_r_ens_time_ls=epsilon_r_ens_time_ls,
    epsilon_g_ens_time_ls=epsilon_g_ens_time_ls,
    epsilon_r_ens_disorder_ls=epsilon_r_ens_disorder_ls,
    epsilon_g_ens_disorder_ls=epsilon_g_ens_disorder_ls,
    dir_root=dir_root,
    specify_ensemble_time_ls=specify_ensemble_time_ls,
    specify_ensemble_disorder_ls=specify_ensemble_disorder_ls,
    use_bc=use_bc
)

def main_T2star_parallel(run_time, specify_ensemble, submit_opt = 'T2star-sim-rc', 
                         extra_note="", custom_tasks=None, T2star_workers=2, launch_in_new_terminal_windows = True, all_tasks_dir = "expt_tasks"):
    
    target_parent_folder="T2star_parallel"

    # _get_task(N=6, Delta_local_ls=[-2.71], t_fixed=[2.125], num_ens=2 ,Delta_mean=2.71, a=10, Omega=15.8, phi=0.0, n_shots=10, n_gates=16, n_U=2, t_ramp = 0.0632, Delta_local_ramp_time = 0.05, Omega_delay_time=0.0, f=1.0, dir_root=dir_root, same_h_ls_task= None)

    expt_task_name_005 = "task_ddebfeaea3ca63079a4c10c4" # chaotic, 6 qubits
    expt_task_name_10 = "task_15eaace71ebd595814b7bd17" # localized, 6 qubits
    expt_125_2125_plateau = "task_d0c2eadd5b392ec6c313c9a3"
    expt_task_125_wo2125 = "task_1f671c4d079b68342b55451a"

    middle_points_tasks = ["task_1a1b10d5afb4687600e767ba", "task_7d1f0dceecf7ebcce140df99", "task_d6f35628b84d6df294fdd9a5", "task_1b3d62cb0a9c01c1b9d9e45b", "task_3e46efcb6c6022b9a937b744", "task_01cf9244e6858929c19ebffa"] 

    if custom_tasks is None:

        if run_time:
            expt_tasks = [expt_task_name_005, expt_task_name_10, expt_task_125_wo2125]
        else:
            expt_tasks = middle_points_tasks + [expt_125_2125_plateau]

    else:
        expt_tasks = custom_tasks

    
    submit_timestamp = 0
    submit_epsilon_r_ens = [0.1]
    submit_epsilon_g_ens = [0.05]
    max_cpu_load_fraction = 0.75
    launch_stagger_seconds = 2.0
    windows_conda_env_name = "oscar_march2"

    for task_name in expt_tasks:
        task_suffix = task_name.removeprefix("task_")
        task_dir_root = os.path.join(target_parent_folder, f"alexei_{task_suffix}_{submit_opt}"+f"_{extra_note}")
        os.makedirs(task_dir_root, exist_ok=True)

        try:
            _stage_task_bundle_for_dir_root(task_name, all_tasks_dir, task_dir_root)
        except Exception as exc:
            print(f"Skipping {task_name}: could not stage task files from {all_tasks_dir}. Error: {exc}")
            continue

        can_submit, load_1min, load_threshold, n_cpu = _cpu_ok_for_submit(max_cpu_load_fraction)
        if load_1min is None:
            print(
                f"CPU check for {task_name}: load1 unavailable on {sys.platform}, "
                f"threshold={load_threshold:.2f} ({n_cpu} cores, {100*max_cpu_load_fraction:.0f}% cap)"
            )
        else:
            print(
                f"CPU check for {task_name}: load1={load_1min:.2f}, "
                f"threshold={load_threshold:.2f} ({n_cpu} cores, {100*max_cpu_load_fraction:.0f}% cap)"
            )

        if not can_submit:
            print(f"Skipping submission for {task_name} to avoid CPU overload.")
            continue

        if launch_in_new_terminal_windows:
            print(f"Launching new Terminal run for {task_name} with dir_root={task_dir_root}")
            try:
                _launch_run_for_N_protocol_in_new_terminal(
                    task_name,
                    submit_timestamp,
                    submit_opt,
                    submit_epsilon_r_ens,
                    submit_epsilon_g_ens,
                    task_dir_root,
                    specify_ensemble,
                    force_recompute=False,
                    force_recompute_processing=False,
                    T2star_workers=T2star_workers,
                    conda_env_name=windows_conda_env_name,
                )
            except Exception as exc:
                print(f"Failed to launch terminal run for {task_name}. Error: {exc}")
                continue

            if launch_stagger_seconds > 0:
                time.sleep(launch_stagger_seconds)
        else:
            print(f"Submitting {task_name} with dir_root={task_dir_root}")
            run_for_N_protocol(
                task_name,
                submit_timestamp,
                submit_opt,
                epsilon_r_ens=submit_epsilon_r_ens,
                epsilon_g_ens=submit_epsilon_g_ens,
                dir_root=task_dir_root,
                specify_ensemble=specify_ensemble,
                force_recompute=False,
                force_recompute_processing=False,
                T2star_workers=T2star_workers,
                save_one_at_a_time=True
            )

def compile_individual(subdir_path, target_dir):
    
    for folder in ("data", "mdata", "tasks", "combos"):
        src_folder = os.path.join(subdir_path, folder)
        if not os.path.isdir(src_folder):
            continue

        dst_folder = os.path.join(target_dir, folder)
        os.makedirs(dst_folder, exist_ok=True)

        for root, dirs, files in os.walk(src_folder):
            rel_path = os.path.relpath(root, src_folder)
            dst_root = os.path.join(dst_folder, rel_path)
            os.makedirs(dst_root, exist_ok=True)

            for fname in files:
                src_file = os.path.join(root, fname)
                dst_file = os.path.join(dst_root, fname)

                if os.path.exists(dst_file):
                    print(f"WARNING: overwriting {dst_file} with {src_file}")

                shutil.copy2(src_file, dst_file)
        print(f"Copied {folder}/ from {subdir_path} -> {target_dir}/{folder}/")

def compile_T2star(source_dir, target_dir):
    """
    Walk each alexei_* subdirectory inside source_dir and copy
    the contents of its data/ and mdata/ folders into
    target_dir/data/ and target_dir/mdata/.
    """
    for subdir_name in sorted(os.listdir(source_dir)):
        subdir_path = os.path.join(source_dir, subdir_name)
        if not os.path.isdir(subdir_path):
            continue

        compile_individual(subdir_path, target_dir)



if __name__ == "__main__":
    # compile_T2star("T2star_parallel_4_10_26", "paper_main_data")
    # main_time_sweep()
    # main_disorder_sweep()

    # main_expt_figs(opt='sim') # 'expt'
    main_expt_figs(opt='expt') # 'expt'

    N = 6
    Delta_local_ls = [-54.2]
    t_fixed = list(np.linspace(0.05, 4.1, 200))
    num_ens = 20
    n_U = 20
    Delta_mean = 2.71
    a = 10
    Omega = 15.8
    phi = 0
    n_shots = 200
    n_gates = 16

    # _get_task(N, Delta_local_ls, t_fixed, num_ens ,Delta_mean, a, Omega, phi, n_shots, n_gates, n_U)

    # _get_task(N=6, Delta_local_ls=[-54.2], t_fixed=[
    #     0.05,
    #     1.0,
    #     1.375,
    #     1.75,
    #     2.125,
    #     2.5,
    #     4
    # ], num_ens=15 ,Delta_mean=0.5*5.42, a=10, Omega=15.8, phi=0, n_shots=200, n_gates=16, n_U=20, t_ramp = 0.0632, Delta_local_ramp_time = 0.05, Omega_delay_time=0.0, f=1, dir_root="alexei_new_demo", same_h_ls_task="task_15eaace71ebd595814b7bd17")

    # localized_copy = "task_6170712a74d8f7fa845f8a47"

    D2d71_T0d05 = "task_5692cdce3a76e63ae3a772f2"
    D2d71_T0d525 = "task_40b2edc8e4622b0d21eaf6b9" ## NEW EXPT
    D2d71_T1d0 = "task_42c55ad726b9f94bdc158967"
    D2d71_T1d375 = "task_0f1bf50951dfaf5fe6bf382c"
    D2d71_T1d75 = "task_bf69383ab41fca45292efca4"
    D2d71_T2d125 = "task_054dbed9870a05386fd07848" # peak
    D2d71_T2d5 = "task_bdca1f5f9559b38d6ac220b2"
    D2d71_T4d0 = "task_2b83a0503df2abfe4e8accdd"
    D2d71_ls = [D2d71_T0d05, D2d71_T0d525, D2d71_T1d0, D2d71_T1d375, D2d71_T1d75, D2d71_T2d125, D2d71_T2d5, D2d71_T4d0]

    D54d2_T0d05 = "task_59e554b7e98b1e23555a8a00"
    D54d2_T0d525 = "task_28501f55cc7b14b219747910" ## NEW EXPT
    D54d2_T1d0 = "task_16733ecbd4817de7c7e3f95b"
    D54d2_T1d375 = "task_e540de6a491931e662c259ce"
    D54d2_T1d75 = "task_45e873589dcb074adcc41fd1"
    D54d2_T2d125 = "task_6ddfe8a7f8b504ed1700e53d"
    D54d2_T2d5 = "task_93801ef64613efee3f8c36db"
    D54d2_T4d0 = "task_b7dfe54c4458db0da5430c22"
    D54d2_ls = [D54d2_T0d05, D54d2_T0d525, D54d2_T1d0, D54d2_T1d375, D54d2_T1d75, D54d2_T2d125, D54d2_T2d5, D54d2_T4d0]
    

    D125d0_T0d05 = "task_00315ea0d133f9be486455ea"
    D125d2_T0d525 = "task_6721a72b5d44f6896710f07f" ## NEW EXPT
    D125d0_T1d0 = "task_c2db90fc4042c2bf1fd8c4b3"
    D125d0_T1d375 = "task_bc9e6c1244ccb355e3ae069f"
    D125d0_T1d75 = "task_c5c247b32831797f463cc543"
    D125d0_T2d125 = "task_d0c2eadd5b392ec6c313c9a3" # peak
    D125d0_T2d5 = "task_fc852096a824ad782ad34d82"
    D125d0_T4d0 = "task_0e0f300cc516e6f67fe37838"

    D125_ls = [D125d0_T0d05, D125d2_T0d525, D125d0_T1d0, D125d0_T1d375, D125d0_T1d75, D125d0_T2d125, D125d0_T2d5, D125d0_T4d0]


  
    # idx = 0
    # if idx == 0:
    #     specify_ensemble = list(range(5))
    # elif idx == 1:
    #     specify_ensemble = list(range(5,11))
    # elif idx == 2:
    #     specify_ensemble = list(range(11,15))

    # specify_ensemble = list(range(15))
    
    # main_T2star_parallel(run_time=True, specify_ensemble = specify_ensemble, submit_opt = 'T2star-sim-rc', extra_note=f"all_4_7_26", custom_tasks=D2d71_ls, T2star_workers=2, launch_in_new_terminal_windows=True)



    #### time dependent numerics 

    # N = 6
    # t_fixed = list(np.linspace(0.05, 4.0, 100))
    # # t_fixed = [0.525]
    # same_h_ls_task_ls = [D2d71_T0d05, D54d2_T0d05, D125d0_T0d05]
    # Delta_local_ls_expt = [-2.71, -54.2, -125.0]
    # Delta_mean = 2.71
    # num_ens = 15
    # n_U_ls = [15, 20, 25]
    # for same_h_ls_task, n_U, Delta_local in zip(same_h_ls_task_ls, n_U_ls, Delta_local_ls_expt):
    #     print("n_U:", n_U, "Delta_local:", Delta_local)
    #     _get_task(N, [Delta_local], t_fixed, num_ens ,Delta_mean, a=10, Omega=15.8, phi=0, n_shots=200, n_gates=16, n_U=n_U, t_ramp = 0.0632, Delta_local_ramp_time = 0.05, Omega_delay_time=0, f=1, dir_root="alexei_new_demo", same_h_ls_task= same_h_ls_task)

    # D2d71_no_err = "task_556886b14a2b72fed9c5e05b"
    # D125d0_no_err = "task_674ce88dad3fbfd139509d1e"
    # D54d2_no_err = "task_67d6229bc32f0413e3ef6e1e"


    ######## NUMERICS TASKS!!!

    D2d71_no_err_100 = "task_f8ac2fc4dc3a338ae2024c9b"
    D54d2_no_err_100 = "task_d27a40c4019f7c6edb9deb71"
    D125d0_no_err = "task_e41d237d81b51a754185eb0d"

    specify_ensemble = None

    # run_for_N_protocol(D125d0_no_err, 0, 'bloqade-sim-no-rc', epsilon_r_ens = 0.1, epsilon_g_ens = 0.05,  dir_root="alexei_copy3",force_recompute=False, force_recompute_processing=False, shot_noise_model = 'multinomial', specify_ensemble=specify_ensemble, force_recompute_expt=False, T2star_workers=1, save_one_at_a_time=False)


    # N = 6
    # t_fixed = [2.125]
    # same_h_ls_task_ls = [None]
    # Delta_local_ls_expt = list(-np.logspace(np.log10(2.71), np.log10(125), 100))
    # Delta_mean = 2.71
    # num_ens = 15
  
    # _get_task(N, Delta_local_ls_expt, t_fixed, num_ens ,Delta_mean, a=10, Omega=15.8, phi=0, n_shots=200, n_gates=16, n_U=25, t_ramp = 0.0632, Delta_local_ramp_time = 0.05, Omega_delay_time=0, f=1, dir_root="alexei_new_demo", same_h_ls_task= None)

    Dsweep_noerr = "task_5d19e26015087bf320a09546"
    
    # run_for_N_protocol(Dsweep_noerr, 0, 'bloqade-sim-no-rc', epsilon_r_ens = 0.1, epsilon_g_ens = 0.05,  dir_root="alexei_copy3",force_recompute=False, force_recompute_processing=False, shot_noise_model = 'multinomial', specify_ensemble=specify_ensemble, force_recompute_expt=False, T2star_workers=1, save_one_at_a_time=False)

    # for n in range(1,4):
    #     print(f"Compiling T2star for alexei_copy{n} -> alexei_new_demo")
    #     compile_individual(f"alexei_copy{n}", "alexei_new_demo")
    # check_0detuning(plot_results=True, make_tasks=False, run_protocol=False)

    




