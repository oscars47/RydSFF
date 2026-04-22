## this file stores the main experimental and gate parameters which are referred to in process_rbp.py
import os, json, time, math, shutil
import numpy as np
import pandas as pd
from QuEraToolbox.hamiltonian import get_h_ls
from process_rbp import get_all_single_hams_rand, get_all_single_hams_rand_calib_check, process_bitstrings, get_all_qutip_probs
from QuEraToolbox.helper_rbp import report_to_bins, bins_to_probs, apply_readout_noise, restrict_probabilities
from QuEraToolbox.random_bp_prep import create_parallelized_x
from QuEraToolbox.expt_file_manager import ExptStore, unique_filename
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import curve_fit
from tqdm import trange
from copy import deepcopy
from copy import deepcopy

from QuEraToolbox.helper_rbp import restrict_probabilities, bins_to_probs, apply_readout_noise, report_to_bins 

# -------- parse tasks
def read_expt_task_stem(stem_task_name, dir_root):
    stem_task_path = os.path.join(dir_root, "tasks", stem_task_name + ".json")
    with open(stem_task_path, 'r') as f:
        stem_task_params = json.load(f)
    
    t_plateau_ls = stem_task_params['t_plateau_ls']
    base_params = stem_task_params['base_params']
    Delta_mean_ls = stem_task_params['Delta_mean_ls']
    Delta_local_ls = stem_task_params['Delta_local_ls']
    gate_params_all = stem_task_params['gate_params_all']
    cluster_spacing = stem_task_params['cluster_spacing']
    override_local = stem_task_params['override_local']
    try: 
        manual_parallelization = stem_task_params['manual_parallelization']
    except KeyError:
        # if not specified, default to False
        manual_parallelization = False

    try:
        x0_y0_offset = stem_task_params['x0_y0_offset']
    except KeyError:
        x0_y0_offset = (0,0)

    try:
        t_delay = stem_task_params['t_delay']
    except KeyError:
        t_delay = None

    try:
        start_Delta_from0 = stem_task_params['start_Delta_from0']   
    except KeyError:
        start_Delta_from0 = True

    try: 
        uniform_Omega_Delta_ramp = stem_task_params['uniform_Omega_Delta_ramp']
    except KeyError:
        uniform_Omega_Delta_ramp = True

    try:
        phi_mode = stem_task_params['phi_mode']
    except KeyError:
        phi_mode = 'binary'

    try:
        Delta_local_ramp_time = stem_task_params['Delta_local_ramp_time']
    except KeyError:
        Delta_local_ramp_time = 0.05
        
    try:
        Omega_delay_time = stem_task_params['Omega_delay_time']
    except KeyError:
        Omega_delay_time = None

    a = base_params['ev_params']['a']

    return t_plateau_ls, Delta_mean_ls, Delta_local_ls, gate_params_all, cluster_spacing, manual_parallelization, base_params, a, override_local, x0_y0_offset, t_delay, start_Delta_from0, uniform_Omega_Delta_ramp, phi_mode, Delta_local_ramp_time, Omega_delay_time
    
def read_expt_task(task_name, dir_root):
    task_filename = os.path.join(dir_root, "tasks", f"{task_name}.json")

    with open(task_filename, 'r') as f:
        task_params = json.load(f)

    stem_task_filename = task_params['stem']
    t_plateau_ls, Delta_mean_ls, Delta_local_ls, gate_params_all, cluster_spacing, manual_parallelization, base_params, a, override_local, x0_y0_offset, t_delay, start_Delta_from0, uniform_Omega_Delta_ramp, phi_mode,Delta_local_ramp_time, Omega_delay_time = read_expt_task_stem(stem_task_filename, dir_root)
    
    # specific to the task
    seq_ls_pre_all = task_params['seq_ls_pre_all']
    h_ls_pre = task_params['h_ls_pre']
    x_pre = task_params['x_pre'] if 'x_pre' in task_params else None
    if x_pre is None:
        N = len(h_ls_pre[0])  # assuming all h_ls_pre have the same length
        x_pre_single = [(i*a, 0) for i in range(N)] 
        x_pre = [x_pre_single for _ in range(len(h_ls_pre))]  

    return h_ls_pre, x_pre, t_plateau_ls, seq_ls_pre_all, base_params, Delta_mean_ls, Delta_local_ls, gate_params_all, cluster_spacing, manual_parallelization, override_local, x0_y0_offset, t_delay, start_Delta_from0, uniform_Omega_Delta_ramp, phi_mode, Delta_local_ramp_time, Omega_delay_time

def get_parall_shots(n_shots, x, cluster_spacing):
    if cluster_spacing is not None:
        x_par = create_parallelized_x(x, cluster_spacing, bound_x=75, bound_y=128)
        n_copies = len(x_par) // len(x) 
        print(f"Parallelization: {n_copies} copies of each ensemble")
    else:
        n_copies = 1

    return n_shots // n_copies

# -------- create tasks
def gen_seq_ls(n_U, n_gates, phi_mode): # for one time
        seq_ls = []
        if phi_mode=='binary':
            for _ in range(n_U):
                seq = np.random.randint(1, 3, size=n_gates).tolist()
                seq_ls.append(seq)
        else:
            for _ in range(n_U):
                seq = np.random.uniform(-np.pi, np.pi, size=n_gates).tolist()
                seq_ls.append(seq)
        return seq_ls

def gen_seq_ls_pre(n_Dm, n_Dl, n_U_ls, n_gates_ls,n_ens, n_t_plateau, same_U_all_time=False, phi_mode='binary'):
    seq_pre_ls_all = np.zeros((n_Dm, n_Dl, n_ens,n_t_plateau), dtype=object)
    for i in range(n_Dm):
        for j in range(n_Dl):
            for l in range(n_ens):
                n_U = n_U_ls[i][j]
                n_gates = n_gates_ls[i][j]
                if n_U != 0 and n_gates != 0:
                    if not same_U_all_time:
                        for k in range(n_t_plateau):
                            seq_ls = gen_seq_ls(n_U, n_gates, phi_mode)
                            seq_pre_ls_all[i, j, l, k] = seq_ls
                        
                    else:
                        seq_ls = gen_seq_ls(n_U, n_gates, phi_mode) # fix the sequence for all time
                        for k in range(n_t_plateau):
                            seq_pre_ls_all[i, j, l, k] = seq_ls

    return seq_pre_ls_all.tolist()


def gen_h_ls_pre(N, n_ens, threshold=0):
    h_ls_pre = []
    for _ in range(n_ens):
        h_ls_pre.append(get_h_ls(N, threshold=threshold).tolist())  

    return h_ls_pre

def gen_tasks(N, Delta_mean_ls, Delta_local_ls, base_params, gate_params_all, cluster_spacing, t_plateau_ls, dir_root, same_U_all_time=False, h_ls_pre=None, seq_ls_pre_all = None, manual_parallelization=False, x_pre=None, override_local=False, x0_y0_offset = (0,0), t_delay=None, start_Delta_from0=True, num_y=1, uniform_Omega_Delta_ramp=False, phi_mode='binary', Delta_local_ramp_time=0.05, Omega_delay_time=None):

    if h_ls_pre is not None:
        assert len(h_ls_pre[0]) == N, "h_ls_pre[0] must have the same length as N"

    manager = ExptStore(dir_root)   
    os.makedirs(os.path.join(dir_root, "tasks"), exist_ok=True)

    if cluster_spacing is None:
        manual_parallelization = False

    # first create task_stem
    # if x0_y0_offset != (0,0), then invlude it in the task stem. otherwise, do not.
    # if t_delay is not None, include it in the task stem. otherwise, do not.

    task_stem_payload = {
            "N": N,
            "Delta_mean_ls": Delta_mean_ls,
            "Delta_local_ls": Delta_local_ls,
            "base_params": base_params,
            "gate_params_all": gate_params_all,
            "t_plateau_ls": list(t_plateau_ls),
            "cluster_spacing": cluster_spacing,
            "manual_parallelization": manual_parallelization,
            "override_local": override_local,
        }
    if x0_y0_offset != (0,0):
        task_stem_payload['x0_y0_offset'] = x0_y0_offset
    if t_delay == 0:
            t_delay = None
    if t_delay is not None:
        task_stem_payload['t_delay'] = t_delay
    if not start_Delta_from0:
        task_stem_payload['start_Delta_from0'] = start_Delta_from0
    if not uniform_Omega_Delta_ramp:
        task_stem_payload['uniform_Omega_Delta_ramp'] = uniform_Omega_Delta_ramp
    if phi_mode != 'binary':
        assert phi_mode == 'random'
        task_stem_payload['phi_mode'] = phi_mode
    if Delta_local_ramp_time != 0.05:
        task_stem_payload['Delta_local_ramp_time'] = Delta_local_ramp_time
    if Omega_delay_time is not None:
        task_stem_payload['Omega_delay_time'] = Omega_delay_time

    suid, added = manager.add(task_stem_payload, timestamp=0)
    stem_task_name = f"stem_{suid}"
    # if added:
    stem_task_filename = os.path.join(dir_root, "tasks", f"{stem_task_name}.json")
    with open(stem_task_filename, 'w') as f:
        json.dump(task_stem_payload, f, indent=4)

    n_ens = base_params['n_ens']
    if h_ls_pre is None and not override_local:
        h_ls_pre = gen_h_ls_pre(N=N, n_ens=n_ens)
    elif override_local and h_ls_pre is None:
        h_ls_pre = [[0]*N for _ in range(n_ens)]


    # need to build n_U_ls and n_gates_ls from gate_params_all   

    n_Dm = len(Delta_mean_ls)
    n_Dl = len(Delta_local_ls)
    n_t_plateau = len(t_plateau_ls)

    n_U_ls = np.zeros((n_Dm, n_Dl), dtype=object)
    n_gates_ls = np.zeros((n_Dm, n_Dl), dtype=object)

    for i in range(n_Dm):
        for j in range(n_Dl):
            try:
                n_U_ls[i, j] = gate_params_all[i][j]['n_U']
                n_gates_ls[i, j] = gate_params_all[i][j]['n_gates']
            except IndexError:
                raise IndexError(f"gate_params_all must have shape ({n_Dm}, {n_Dl}) to match the lengths of Delta_mean_ls and Delta_local_ls. Currently has shape {np.shape(gate_params_all)}")

    if seq_ls_pre_all is None:
        seq_ls_pre_all = gen_seq_ls_pre(n_Dm, n_Dl, n_U_ls, n_gates_ls, n_ens, n_t_plateau, same_U_all_time=same_U_all_time, phi_mode=phi_mode)

    # print("seq_ls_pre_all", seq_ls_pre_all)
    # x_pre_chain = [[(i*base_params['ev_params']['a'], 0) for i in range(N)] for _ in range(len(h_ls_pre))]
    # divide N into num_y rows
    if x_pre is None:
        x_pre = []
        a = base_params['ev_params']['a']
        n_x = N // num_y
        assert n_x * num_y == N, "N must be divisible by num_y"
        for _ in range(len(h_ls_pre)):
            x_coords = []
            for iy in range(num_y):
                for ix in range(n_x):
                    x_coords.append( (ix*a, iy*a) )
            x_pre.append(x_coords)

    task_payload = {
        "stem": stem_task_name,
        "h_ls_pre": h_ls_pre,
        "x_pre": x_pre,
        "seq_ls_pre_all": seq_ls_pre_all,
    }
    

    tuid, added = manager.add(task_payload, timestamp=0)
    task_name = f"task_{tuid}"
    if added:
        task_filename = os.path.join(dir_root, "tasks", f"{task_name}.json")
        with open(task_filename, 'w') as f:
            json.dump(task_payload, f, indent=4)
    print("Created task:", task_name)
    print("-"*20)
    return task_name

     
def det_cost(task_name, dir_root):
    ppt = 0.3 # in usd
    pps = 0.01

    # determine num tasks and shots per task
    # for every ensemble, we have all combinations of Delta_mean and Delta_local, for every combination we have t_plateau_ls. each time has the same list of n_U unitaries. the point of keeping same collection of U per time point is that we only have to take one set of data for the initial state

    h_ls_pre, x_pre, t_plateau_ls, seq_ls_pre_all, base_params, Delta_mean_ls, Delta_local_ls, gate_params_all, cluster_spacing, manual_parallelization ,override_local, x0_y0_offset, t_delay, start_Delta_from0, uniform_Omega_Delta_ramp, phi_mode, Delta_local_ramp_time, Omega_delay_time = read_expt_task(task_name, dir_root)
    
    n_ens = base_params['n_ens']
    n_Dm = len(Delta_mean_ls)
    n_Dl = len(Delta_local_ls)
    
    n_U_ls = np.zeros((n_Dm, n_Dl), dtype=object)
    for i in range(n_Dm):
        for j in range(n_Dl):
            n_U_ij = gate_params_all[i][j]['n_U']
            n_U_ls[i, j] = n_U_ij if n_U_ij > 0 else 1  # if no unitaries, we still ramp up and plateau and ramp down

    n_tasks_ls = np.zeros((n_Dm, n_Dl), dtype=int)
    shots_ls = np.zeros((n_Dm, n_Dl), dtype=int)
    for i in range(n_Dm):
        for j in range(n_Dl):
            tasks_ij = (len(t_plateau_ls)) * n_U_ls[i, j] * n_ens
            n_tasks_ls[i, j] = tasks_ij
            n_shots_raw = gate_params_all[i][j]['n_shots']
            shots_ls[i, j] = tasks_ij*n_shots_raw

    n_tasks = np.sum(n_tasks_ls)
    n_shots = np.sum(shots_ls)

    # cost in dollars
    cost = n_tasks * ppt + n_shots * pps
    print(f"Total number of tasks: {n_tasks}, total number of shots: {n_shots},\nCost: ${cost:.3g}")
    return cost


# -------- run bloqade tasks
def get_subdirname(name, task_name, timestamp, preset_opt):
    return f"{name}_{task_name.replace('task_', '')}_{timestamp}" if preset_opt is None else f"{name}_{task_name.replace('task_', '')}_{timestamp}_{preset_opt}"

def execute_bloqade_task(task_name, name, is_expt_data, timestamp, dir_root, force_recompute=False, debug=False, preset_opt=None, data_subdir=None, save_mode=False, backup_dirs=None, after_how_many_ham_run_check=None, ham_check_dir_main =None, allow_override_name=False): ## if doing experiment, just run this function first to submit the data to AWS
    print("Executing Bloqade task:", task_name)
    h_ls_pre, x_pre, t_plateau_ls, seq_ls_pre_all, base_params, Delta_mean_ls, Delta_local_ls, gate_params_all, cluster_spacing, manual_parallelization, override_local, x0_y0_offset, t_delay, start_Delta_from0, uniform_Omega_Delta_ramp, phi_mode, Delta_local_ramp_time, Omega_delay_time = read_expt_task(task_name, dir_root)

    # print("Using preset_opt:", preset_opt)
    # print("Using manual_parallelization:", manual_parallelization, "cluster_spacing:", cluster_spacing)
    # print("Using override_local:", override_local)

    if data_subdir is None:
        name = get_subdirname(name, task_name, timestamp, preset_opt)
    else:
        name = data_subdir

    ## check if this data_subdir already exists, in wich case we add a number to the end. it will be a directory
    if allow_override_name:
        name = unique_filename(os.path.join(dir_root, "data", name))
        # remove the dir_root/data/ from name
        name = name.split(os.path.join(dir_root, "data"))[-1].strip("/")

    if after_how_many_ham_run_check is None or ham_check_dir_main is None:
        get_all_single_hams_rand(h_ls_pre, x_pre, t_plateau_ls, seq_ls_pre_all, base_params, Delta_mean_ls, Delta_local_ls, gate_params_all, cluster_spacing, manual_parallelization, is_expt_data, timestamp, dir_root, force_recompute=force_recompute, name=name, save_mode=save_mode, x0_y0_offset=x0_y0_offset, debug=debug, preset_opt=preset_opt, backup_dirs=backup_dirs, override_local=override_local, t_delay=t_delay, start_Delta_from0=start_Delta_from0, uniform_Omega_Delta_ramp=uniform_Omega_Delta_ramp, phi_mode=phi_mode, Delta_local_ramp_time=Delta_local_ramp_time, Omega_delay_time=Omega_delay_time)
    else:
        get_all_single_hams_rand_calib_check(h_ls_pre, x_pre, t_plateau_ls, seq_ls_pre_all, base_params, Delta_mean_ls, Delta_local_ls, gate_params_all, cluster_spacing, manual_parallelization, is_expt_data, timestamp, dir_root, force_recompute=force_recompute, name=name, after_how_many_ham_run_check=after_how_many_ham_run_check, ham_check_dir_main=ham_check_dir_main,save_mode=save_mode, x0_y0_offset=x0_y0_offset, debug=debug, preset_opt=preset_opt, backup_dirs=backup_dirs, override_local=override_local, t_delay=t_delay, start_Delta_from0=start_Delta_from0, uniform_Omega_Delta_ramp=uniform_Omega_Delta_ramp, phi_mode=phi_mode, Delta_local_ramp_time=Delta_local_ramp_time, Omega_delay_time=Omega_delay_time)


def chunk_task(task_name, starting_idx, num_ham_in_chunk, dir_root):
    h_ls_pre, x_pre, t_plateau_ls, seq_ls_pre_all, base_params, Delta_mean_ls, Delta_local_ls, gate_params_all, cluster_spacing, manual_parallelization, override_local, x0_y0_offset, t_delay, start_Delta_from0, uniform_Omega_Delta_ramp, phi_mode, Delta_local_ramp_time, Omega_delay_time = read_expt_task(task_name, dir_root)

    ## we break it up here. add a param, chunk_index. selects out the ensemble at this location and corresponding seq_ls. keep t_plateau list
    ## include chunk_idx + num_ham_in_chunk to select the range; if chunk_idx + num_ham_in_chunk > n_ens, then just go to the end
    # h_ls_pre = [h_ls_pre[chunk_idx]]
    # x_pre = [x_pre[chunk_idx]]
    # seq_ls_pre_all = [[seq_ls_pre_all[i][j][chunk_idx] for i in range(len(Delta_mean_ls))] for j in range(len(Delta_local_ls))] 
    n_ens = base_params['n_ens']
    end_idx = min(starting_idx + num_ham_in_chunk, n_ens)
    n_ham_actual_in_chunk = end_idx - starting_idx
    # print("end_idx", end_idx)
    h_ls_pre = h_ls_pre[starting_idx:end_idx]
    x_pre = x_pre[starting_idx:end_idx]
    print("start index", starting_idx, " end index:", end_idx)
    # print("seq_ls_pre before", seq_ls_pre_all)
    # seq_ls_pre_all = [[seq_ls_pre_all[i][j][l] for l in range(chunk_idx, end_idx) for i in range(len(Delta_mean_ls)) for j in range(len(Delta_local_ls))]]
    seq_ls_pre_all_old = deepcopy(seq_ls_pre_all)
    n_Dm = len(Delta_mean_ls)
    n_Dl = len(Delta_local_ls)
    n_t_plateau = len(t_plateau_ls)
    seq_pre_ls_all = np.zeros((n_Dm, n_Dl, n_ham_actual_in_chunk,n_t_plateau), dtype=object)
    print("starting_idx", starting_idx, " end_idx", end_idx)

    for i in range(n_Dm):
        for j in range(n_Dl):
            for l in range(starting_idx, end_idx):
                for k in range(n_t_plateau):
                    seq_ijlk = seq_ls_pre_all_old[i][j][l][k]
                    # Map absolute index l to relative chunk index
                    chunk_l_idx = l - starting_idx
                    seq_pre_ls_all[i][j][chunk_l_idx][k] = seq_ijlk

    seq_pre_ls_all = seq_pre_ls_all.tolist()
    return h_ls_pre, x_pre, t_plateau_ls, seq_pre_ls_all, base_params, Delta_mean_ls, Delta_local_ls, gate_params_all, cluster_spacing, manual_parallelization, override_local, x0_y0_offset, t_delay, start_Delta_from0, uniform_Omega_Delta_ramp, phi_mode, Delta_local_ramp_time, Omega_delay_time


def execute_bloqade_task_chunk(starting_idx, num_ham_in_chunk, task_name, name, is_expt_data, timestamp, dir_root, force_recompute=False, debug=False, preset_opt=None, data_subdir=None, save_mode=False, backup_dirs=None, after_how_many_ham_run_check=None, ham_check_dir_main =None, allow_override_name=False):
    print("Executing Bloqade task:", task_name, " starting index:", starting_idx)

    h_ls_pre, x_pre, t_plateau_ls, seq_ls_pre_all, base_params, Delta_mean_ls, Delta_local_ls, gate_params_all, cluster_spacing, manual_parallelization, override_local, x0_y0_offset, t_delay, start_Delta_from0, uniform_Omega_Delta_ramp, phi_mode, Delta_local_ramp_time, Omega_delay_time = chunk_task(task_name, starting_idx, num_ham_in_chunk, dir_root)

    print("h_ls in the chunk", h_ls_pre)
    

    # print("seq_ls_pre after", seq_ls_pre_all)

    # print("Using preset_opt:", preset_opt)
    # print("Using manual_parallelization:", manual_parallelization, "cluster_spacing:", cluster_spacing)
    # print("Using override_local:", override_local)

    if data_subdir is None:
        name = get_subdirname(name, task_name, timestamp, preset_opt)
    else:
        name = data_subdir

    if allow_override_name:
        name = unique_filename(os.path.join(dir_root, "data", name))
        # remove the dir_root/data/ from name
        name = name.split(os.path.join(dir_root, "data"))[-1].strip("/")

    if after_how_many_ham_run_check is None or ham_check_dir_main is None:
        get_all_single_hams_rand(h_ls_pre, x_pre, t_plateau_ls, seq_ls_pre_all, base_params, Delta_mean_ls, Delta_local_ls, gate_params_all, cluster_spacing, manual_parallelization, is_expt_data, timestamp, dir_root, force_recompute=force_recompute, name=name, save_mode=save_mode, x0_y0_offset=x0_y0_offset, debug=debug, preset_opt=preset_opt, backup_dirs=backup_dirs, override_local=override_local, t_delay=t_delay, start_Delta_from0=start_Delta_from0, uniform_Omega_Delta_ramp=uniform_Omega_Delta_ramp, phi_mode=phi_mode, Delta_local_ramp_time=Delta_local_ramp_time, Omega_delay_time=Omega_delay_time)
    else:
        get_all_single_hams_rand_calib_check(h_ls_pre, x_pre, t_plateau_ls, seq_ls_pre_all, base_params, Delta_mean_ls, Delta_local_ls, gate_params_all, cluster_spacing, manual_parallelization, is_expt_data, timestamp, dir_root, force_recompute=force_recompute, name=name, after_how_many_ham_run_check=after_how_many_ham_run_check, ham_check_dir_main=ham_check_dir_main,save_mode=save_mode, x0_y0_offset=x0_y0_offset, debug=debug, preset_opt=preset_opt, backup_dirs=backup_dirs, override_local=override_local, t_delay=t_delay, start_Delta_from0=start_Delta_from0, uniform_Omega_Delta_ramp=uniform_Omega_Delta_ramp, phi_mode=phi_mode, Delta_local_ramp_time=Delta_local_ramp_time, Omega_delay_time=Omega_delay_time)


def combine_tasks(tasks, Delta_mean_ls_all, Delta_local_ls_all, n_U, n_t_plateau):
    ## NOTE: we combine by extending h_ls_pre and seq_ls_pre at the i,j from the master Delta_mean_ls and Delta_local_ls
    h_ls_pre = np.zeros((len(Delta_mean_ls_all), len(Delta_local_ls_all)), dtype=object)
    x_pre_all = np.zeros((len(Delta_mean_ls_all), len(Delta_local_ls_all)), dtype=object)
    seq_ls_pre_all = np.zeros((len(Delta_mean_ls_all), len(Delta_local_ls_all), n_t_plateau), dtype=object)

    for task in tasks:
        h_ls_pre, x_pre, t_plateau_ls, seq_ls_pre, base_params, Delta_mean_ls, Delta_local_ls, gate_params_all, cluster_spacing = task
        
        i = Delta_mean_ls_all.index(Delta_mean_ls)
        j = Delta_local_ls_all.index(Delta_local_ls)

        # h_ls
        h_ls_pre_ij = h_ls_pre[i, j]
        if h_ls_pre not in h_ls_pre_ij:
            h_ls_pre_ij.extend(h_ls_pre)
        h_ls_pre[i, j] = h_ls_pre_ij
        # x_pre
        x_pre_all_ij = x_pre_all[i, j]
        if x_pre not in x_pre_all_ij:
            x_pre_all_ij.extend(x_pre)
        x_pre_all[i, j] = x_pre_all_ij

        seq_ls_pre_all_ij = seq_ls_pre_all[i, j]
        for k in range(n_t_plateau):
            seq_ls = seq_ls_pre[k]
            if seq_ls not in seq_ls_pre_all_ij[k]:
                seq_ls_pre_all_ij[k].extend(seq_ls)
        seq_ls_pre_all[i, j] = seq_ls_pre_all_ij
    return h_ls_pre, x_pre_all, seq_ls_pre_all
  

## helper func
def set_to_zero(data):
        """Recursively set all elements to 0 while preserving structure."""
        if isinstance(data, list):
            return [set_to_zero(item) for item in data]
        elif isinstance(data, (int, float)):
            return 0.0
        else:
            return data


def compare_single_qubit_effect_phiquenches(
    task_name,
    timestamp,
    name_prefix = "quench_task",
    n_sequences=4,
    aquila_timestamp=None,
    epsilon_r_ls=(0.1,),
    epsilon_g_ls=(0.1,),
    fontsize=20,
    J=5.42,
    force_recompute=False,
    dir_root=".",
    Omega_ens_override=None,
):
    """
    Plot per-sequence computational-basis distributions (averaged across ensembles & plateaus),
    organized with one row per 'sequence index' s.

    Interpretation of timestamps:
      - If timestamp != 0: interpret as Aquila (experimental) data timestamp and plot Aquila.
      - If timestamp == 0: interpret as Bloqade simulation baseline WITHOUT error correction.

    Special compare mode (timestamp == 0):
      - If aquila_timestamp is provided, also plot:
          (i) Bloqade WITH readout error correction + Omega_ens_override (if provided)
          (ii) Aquila data at aquila_timestamp

    Caching:
      - Calls to process_bitstrings are memoized inside this function so multiple "versions"
        don't trigger recomputation unless force_recompute=True.
    """

    # ---------------------------
    # Styling
    # ---------------------------
    mpl.rcParams.update({"font.size": fontsize})
    os.environ["PATH"] += os.pathsep + "/Library/TeX/texbin/"
    plt.rc("text", usetex=True)
    mpl.rc(
        "text.latex",
        preamble=r"""
        \usepackage{amsmath}
        \usepackage{newtxtext,newtxmath}
        """,
    )

    # ---------------------------
    # Read task parameters once
    # ---------------------------
    (
        h_ls_pre,
        x_pre,
        t_plateau_ls,
        seq_ls_pre_all,
        base_params,
        Delta_mean_ls,
        Delta_local_ls,
        gate_params_all,
        cluster_spacing,
        manual_parallelization,
        override_local,
        x0_y0_offset,
        t_delay,
        start_Delta_from0,
        uniform_Omega_Delta_ramp,
        phi_mode,
        Delta_local_ramp_time,
        Omega_delay_time
    ) = read_expt_task(task_name, dir_root)

    N = len(h_ls_pre[0])
    d_state = 2**N


    cache = {}

    def _run_bitstrings(is_expt_data, ts, base_params_in):
        """
        Returns raw_results from process_bitstrings (process_opt='raw').
        Memoized by (is_expt_data, ts, apply_correction, omega_key).
        """
        key = (bool(is_expt_data), int(ts), base_params_in.get("ev_params", {}).get("Omega", None))
        if (not force_recompute) and (key in cache):
            return cache[key]

        name = get_subdirname(name_prefix, task_name, ts, None)
        print(f"[info] Running bitstring processing for {name} (is_expt_data={is_expt_data}, ts={ts})")
        # if ts != 0:
        #     raise NotImplementedError

        raw_results, _, _ = process_bitstrings(
            h_ls_pre,
            x_pre,
            t_plateau_ls,
            seq_ls_pre_all,
            base_params_in,
            Delta_mean_ls,
            Delta_local_ls,
            gate_params_all,
            cluster_spacing,
            manual_parallelization,
            is_expt_data,
            ts,
            dir_root,
            name,
            process_opt="raw",
            force_recompute=True,
            is_bloqade=True,
            epsilon_r_ens=0.0,
            epsilon_g_ens=0.0,
            apply_correction=False,
            start_Delta_from0=start_Delta_from0,
            uniform_Omega_Delta_ramp=uniform_Omega_Delta_ramp,
            phi_mode=phi_mode,
            force_recompute_processing=True,
            override_local=override_local,
            Delta_local_ramp_time=Delta_local_ramp_time, Omega_delay_time=Omega_delay_time
        )

        cache[key] = raw_results
        return raw_results

    base_params_plain = deepcopy(base_params)

    base_params_omega = deepcopy(base_params)
    if Omega_ens_override is not None:
        base_params_omega["ev_params"]["Omega"] = np.mean(Omega_ens_override[0][0])
        print(f"[info] Overriding Omega for corrected Bloqade run.", base_params_omega["ev_params"]["Omega"])
        # raise NotImplementedError("Omega_ens_override not implemented yet")

    versions = []

    if int(timestamp) != 0:
        versions.append(
            dict(
                name="Aquila",
                is_expt_data=True,
                ts=int(timestamp),
                apply_correction=False,  
                base_params_in=base_params_plain,
                label=r"$\mathrm{Aquila}$",
                color="red",
                alpha=0.35,
            )
        )
    else:
        # show Bloqade no correction baseline
        versions.append(
            dict(
                name="Bloqade_uncorrected",
                is_expt_data=False,
                ts=0,
                apply_correction=False,
                base_params_in=base_params_plain,
                label=r"$\mathrm{Bloqade}$",
                color="purple",
                alpha=0.25,
                fmt="o",
                linestyle='dashed'
            )
        )
        

        # compare mode: Aquila vs Bloqade(corrected + Omega_override)
        if aquila_timestamp is not None:
            # Check if epsilon values are close to zero
            epsilon_near_zero = all(
                abs(epsilon_r_ls[i][j][l]) < 1e-6 and abs(epsilon_g_ls[i][j][l]) < 1e-6
                for i in range(len(epsilon_r_ls))
                for j in range(len(epsilon_r_ls[i]))
                for l in range(len(epsilon_r_ls[i][j]))
            )
            
            # Determine label based on epsilon values and Omega override
            if epsilon_near_zero:
                if Omega_ens_override is not None:
                    bloqade_label = r"$\mathrm{Bloqade\ (\Omega \, c.)}$"
                else:
                    bloqade_label = r"$\mathrm{Bloqade}$"
            else:
                if Omega_ens_override is not None:
                    bloqade_label = r"$\mathrm{Bloqade\ (r.c.\, and \, \Omega \, c.)}$"
                else:
                    bloqade_label = r"$\mathrm{Bloqade\ (r.c.)}$"
            
            versions.append(
                dict(
                    name="Bloqade_corrected",
                    is_expt_data=False,
                    ts=0,
                    apply_correction=True,
                    base_params_in=base_params_omega,
                    label=bloqade_label,
                    color="blue",
                    alpha=0.35,
                    fmt="^",
                    linestyle='dotted'
                )
            )
            versions.append(
                dict(
                    name="Aquila_compare",
                    is_expt_data=True,
                    ts=int(aquila_timestamp),
                    apply_correction=False,  
                    base_params_in=base_params_plain,
                    label=r"$\mathrm{Aquila}$",
                    color="red",
                    alpha=0.35,
                    fmt = "s",
                    linestyle='-'
                )
            )
        else:
            print("[warn] timestamp==0 and aquila_timestamp=None: skipping Aquila comparison.")





    def _probs_by_sequence(raw_results, apply_correction=False):
        """
        Returns dict: s -> avg_probs (length 2^N), averaged over ensembles and plateaus
        """
        out = {s: [] for s in range(n_sequences)}
        out_restr = {s: [] for s in range(n_sequences)}

        ### ASSUME ONLY 1 DELTA MEAN/LOCAL FOR NOW
        for i, _Delta_mean in enumerate(Delta_mean_ls):
            for j, _Delta_local in enumerate(Delta_local_ls):
                blk = raw_results[i][j]  # ensembles -> plateaus -> sequences -> array
                for s in range(n_sequences):
                    samples = []
                    samples_restr = []
                    for l in range(len(blk)):          # ensembles
                        # for m in range(len(blk[l])):   # plateau times # only m = 0
                        if s < len(blk[l][0]):
                            arr = np.array(blk[l][0][s])
                            #check sum of arr counts
                            total_counts = np.sum([count for _, count in arr])
                            print(f"[debug] total counts for i={i}, j={j}, ens={l}, seq={s}: {total_counts}")
                            bins = report_to_bins(arr)
                            probs = bins_to_probs(bins, N)
                            if apply_correction:
                                epsilon_r = epsilon_r_ls[i][j][l]
                                epsilon_g = epsilon_g_ls[i][j][l]
                                print("APPLYING READOUT CORRECTION", epsilon_r, epsilon_g)
                                probs = apply_readout_noise(probs, epsilon_g, epsilon_r)
                                print("PROBS", probs)
                                assert np.isclose(np.sum(probs), 1.0), f"probs do not sum to 1 after correction: sum={np.sum(probs)}"

                            probs_qidx = [restrict_probabilities(probs, [q_idx])[1] for q_idx in range(N-1, -1, -1)] 
                                
                            samples.append(probs)
                            samples_restr.append(probs_qidx)
                    # print("SAMPLES FOR SEQUENCE", s, samples)
                    # raise NotImplementedError("readout correction not implemented yet")
                    if len(samples) == 0:
                        continue
                    samples = np.concatenate(samples, axis=0)  # (n_total_samples, 2^N)
                    out[s].append(samples)
                    samples_restr = np.concatenate(samples_restr, axis=0)
                    out_restr[s].append(samples_restr)
        # print("FINAL OUT", out)
        return out, out_restr


    version_probs = []
    for v in versions:
        # if v["ts"] == 0 and v['apply_correction']==False:
        #     pass
        # else:
        raw = _run_bitstrings(
            is_expt_data=v["is_expt_data"],
            ts=v["ts"],
            base_params_in=v["base_params_in"],
        )
        
        probs, probs_restr = _probs_by_sequence(raw, apply_correction=v["apply_correction"])
        if v['name']=="Aquila_compare":
            print(raw)

        # print(f"[debug] probs computed for version {v['name']}", probs)
        version_probs.append((v, probs, probs_restr))
    
    fig, axes = plt.subplots(n_sequences, 2, figsize=(20, 5 * n_sequences), squeeze=False)
    x_states = np.arange(d_state)

    for seq_n in range(n_sequences):
        ax_states = axes[seq_n, 0]
        ax_qubits = axes[seq_n, 1]

        # overlay each version
        for v, probs_by_s, probs_restr_by_s in version_probs:
          
            p = np.array(probs_by_s[seq_n], dtype=float)[0] # select i,j
            # print(f"[debug] probs for version {v['name']}, sequence {seq_n}: ", p)
            
            states = np.arange(len(p))

            mean_probs_single_qubit = np.mean(np.array(probs_restr_by_s[seq_n], dtype=float), axis=0)  # average over all single qubits

            # formatted_probs = ', '.join([f'{mean:.3g}' for mean in mean_probs_single_qubit])
            label = v["label"]
           
            ax_states.plot(
                states,
                p,
                alpha=v["alpha"],
                color=v["color"],
                linewidth=0,
                label=label,
                marker=v.get("fmt", "o"),
                markersize=8,
            )
            
            # Plot per-qubit probabilities in second column
            qubit_indices = np.arange(N)
            ax_qubits.bar(
                qubit_indices + (version_probs.index((v, probs_by_s, probs_restr_by_s)) * 0.2),
                mean_probs_single_qubit,
                width=0.2,
                alpha=v["alpha"],
                color=v["color"],
                label=v["label"],
                edgecolor='black',
                linewidth=0.5
            )
            

        ylabel = rf"$P(|\psi\rangle)$" + "\n" + rf"$\mathrm{{Num \, flips}}={seq_n}$" if seq_n > 0 else rf"$P(|\mathrm{{Basis\, state}}\rangle)$"
        ax_states.set_ylabel(ylabel, fontsize=fontsize * 0.8)
        ylabel_qubits = rf"$\langle P(|r\rangle_i) \rangle$" + "\n" + rf"$\mathrm{{Num \, flips}}={seq_n}$" if seq_n > 0 else rf"$\langle P(|r\rangle_i) \rangle$"
        ax_qubits.set_ylabel(ylabel_qubits, fontsize=fontsize * 0.8)

        # x tick labels - only on last sequence
        if seq_n == 0:
             ax_qubits.legend(fontsize=fontsize * 0.8, loc="upper right", framealpha=0.7)
        if seq_n == n_sequences - 1:
            if d_state <= 16:
                ax_states.set_xticks(x_states)
                ax_states.set_xticklabels([format(i, f"0{N}b") for i in x_states], rotation=45, ha="right")
                ax_states.set_xlabel(r"$|\mathrm{Basis\, state}\rangle$", fontsize=fontsize * 0.8)
            else:
                ax_states.set_xlabel(rf"$|\mathrm{{Basis\, state}}\rangle$", fontsize=fontsize * 0.8)
            ax_qubits.set_xlabel("$\mathrm{Qubit\, } i$", fontsize=fontsize * 0.8)
        else:
            ax_states.set_xticks(x_states)
            ax_states.set_xticklabels([])  # Remove labels but keep ticks

        ax_states.set_xlim(-0.5, d_state - 0.5)
        ax_qubits.set_xlim(-0.5, N - 0.5)
        ax_qubits.set_xticks(np.arange(N))
        ax_qubits.set_xticklabels([f"{i}" for i in range(N)])
        
        # ax_qubits.legend(fontsize=fontsize * 0.8, loc="upper right", framealpha=0.7)

    plt.tight_layout()

    os.makedirs(os.path.join(dir_root, "results"), exist_ok=True)
    manager = ExptStore(dir_root)
    puid_payload = {
        "task_name": task_name,
        "timestamp": int(timestamp),
        "aquila_timestamp": None if aquila_timestamp is None else int(aquila_timestamp),
        "epsilon_r_ls": list(epsilon_r_ls),
        "epsilon_g_ls": list(epsilon_g_ls),
        "Omega_ens_override": None if Omega_ens_override is None else "provided",
        "type": "compare_single_qubit_effect_phiquenches",
    }
    puid, _added = manager.add(puid_payload, timestamp=0)

    plot_filepath = os.path.join(dir_root, "results", f"{puid}_qubit_distributions.pdf")
    plt.savefig(plot_filepath, bbox_inches="tight")
    print(f"[info] Saved plot to {plot_filepath}")
    plt.show()


# ------- carefully redoing the phases
def format_value_uncertainty(value, uncertainty):
    if uncertainty == 0:
        return f"{value:.3g} \pm 0"
    
    exponent = math.floor(math.log10(abs(uncertainty)))
    first_digit = int(uncertainty / 10**exponent)
    
    n_sig = 2 if first_digit in [1, 2] else 1
    
    unc_rounded = round(uncertainty, -exponent + n_sig - 1)
    
    decimals = -int(math.floor(math.log10(unc_rounded))) + n_sig - 1
    val_rounded = round(value, decimals)
    
    return f"{val_rounded:.{decimals}f} \pm {unc_rounded:.{decimals}f}"

def do_preset(task_name, timestamp, t_plateau_qutip, ax=None, color_ls = ['purple', 'red', 'blue'], fontsize=60, force_recompute=False, neg_phi=True, do_ramsey=False, overridename=None, T2_star=3.96, Omega_rabi=6.19, pop_max = 0.83, extra_expt_dir=None, dir_root=".", error_g = 0.1, error_r = 0.1, error_g_unc=0, error_r_unc=0, debug=False, name=None, q_index=-1, show_qutip=True, global_file=None,   override_error=False, fit_num_pts=1000, alpha_val=1.0):

    ## pass in ax if want to overplot
    passed_ax = ax is not None

    timestamp = int(timestamp)
    print("inside do_preset", timestamp)

    if do_ramsey:
        preset_opt = "ramsey"
        if name is None:
            name = "ramsey"
    else:
        preset_opt = None
        if name is None:
            name = "rabi"

    if overridename is not None:
        name = overridename

    def rabi_func(t, A, Omega, varphi, B, t_c):
        return A*np.exp(-(t/t_c)**2) * np.sin(Omega * t + varphi) + B
    
    def ramsey_func(t, a, T2_star):
        return a*np.exp(-(t/T2_star)**2)+0.5

    # expt_name = f"{name}_{task_name}_{timestamp}"
    # sim_name = f"{name}_{task_name}_0"
    expt_name = get_subdirname(name, task_name, timestamp, preset_opt)
    sim_name = get_subdirname(name, task_name, 0, preset_opt)

    h_ls_pre, x_pre, t_plateau_ls, seq_ls_pre_all, base_params, Delta_mean_ls, Delta_local_ls, gate_params_all, cluster_spacing, manual_parallelization, override_local, x0_y0_offset, t_delay, start_Delta_from0, uniform_Omega_Delta_ramp, phi_mode, Delta_local_ramp_time, Omega_delay_time = read_expt_task(task_name, dir_root)

    seq_ls_pre_all_orig = deepcopy(seq_ls_pre_all)

    print("T_DELAY:", t_delay)

    if t_delay is not None:
        t_plateau_ls = np.array(t_plateau_ls) + t_delay


    print("expt_name:", expt_name)
    print("X0_Y0_OFFSET", x0_y0_offset)


    # assert len(h_ls_pre[0]) == 1, "Rabi task should have only one qubit"
    N = len(h_ls_pre[0])

    q_index += N  # convert to python index
    q_index = q_index % N  # wrap around
    assert 0 <= q_index < N, f"q_index {q_index} out of range for N={N}"

    assert len(Delta_mean_ls) == 1 and len(Delta_local_ls) == 1, "Rabi task should have only one Delta_mean and Delta_local"

    # create new expt task
    if len(t_plateau_qutip) != len(t_plateau_ls):
        print("SEQUENCE", seq_ls_pre_all) #[[[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]]];  [[[[[[1, 1, 2, 1, 1, 2, 2, 2]], [[1, 1, 2, 1, 1, 2, 2, 2]], [[1, 1, 2, 1, 1, 2, 2, 2]], [[1, 1, 2, 1, 1, 2, 2, 2]], [[1, 1, 2, 1, 1, 2, 2, 2]], [[1, 1, 2, 1, 1, 2, 2, 2]], [[1, 1, 2, 1, 1, 2, 2, 2]], [[1, 1, 2, 1, 1, 2, 2, 2]], [[1, 1, 2, 1, 1, 2, 2, 2]], [[1, 1, 2, 1, 1, 2, 2, 2]]]]]]
        print(seq_ls_pre_all[0][0][0][0])
        # if not type(seq_ls_pre_all[0][0][0][0]) == int:
        #     seq_ls_pre_all= [[[[seq_ls_pre_all[0][0][0][0] for _ in range(len(t_plateau_qutip))]]]]  # repeat the same sequence for each new time
        # else:
        seq_ls_pre_all= [[[[seq_ls_pre_all[0][0][0][0] for _ in range(len(t_plateau_qutip))]]]]

    smuid, simulated_data = get_all_single_hams_rand(h_ls_pre, x_pre, t_plateau_qutip, seq_ls_pre_all, base_params, Delta_mean_ls, Delta_local_ls, gate_params_all, cluster_spacing, manual_parallelization, is_expt_data=False, timestamp=0, dir_root=dir_root,force_recompute=force_recompute, preset_opt=preset_opt, name=sim_name, override_local=override_local, debug=debug, x0_y0_offset=x0_y0_offset, start_Delta_from0=start_Delta_from0,uniform_Omega_Delta_ramp=uniform_Omega_Delta_ramp, phi_mode=phi_mode, Delta_local_ramp_time=Delta_local_ramp_time, Omega_delay_time=Omega_delay_time) 

    qutip_task = gen_tasks(N, Delta_mean_ls, Delta_local_ls, base_params, gate_params_all, cluster_spacing, t_plateau_qutip, dir_root, same_U_all_time=True, h_ls_pre=h_ls_pre, seq_ls_pre_all=seq_ls_pre_all, start_Delta_from0=start_Delta_from0, uniform_Omega_Delta_ramp=uniform_Omega_Delta_ramp,phi_mode=phi_mode, Delta_local_ramp_time=Delta_local_ramp_time, Omega_delay_time=Omega_delay_time )

    # read the task
    if show_qutip:
        h_ls_pre_qutip, x_pre_qutip, t_plateau_qutip, seq_ls_pre_all, base_params_qutip, Delta_mean_ls_qutip, Delta_local_ls_qutip, gate_params_all_qutip, cluster_spacing_qutip, manual_parallelization_qutip, override_local_qutip, x0_y0_offset_qutip, t_delay_qutip, start_Delta_from0,uniform_Omega_Delta_ramp, phi_mode, Delta_local_ramp_time, Omega_delay_time  = read_expt_task(qutip_task, dir_root)

        qmuid, qutip_data = get_all_qutip_probs(h_ls_pre_qutip, x_pre_qutip, t_plateau_qutip, seq_ls_pre_all, base_params_qutip, Delta_mean_ls_qutip, Delta_local_ls_qutip, gate_params_all_qutip, dir_root=dir_root, force_recompute=force_recompute, neg_phi=neg_phi, preset_opt=preset_opt, override_local=override_local_qutip, start_Delta_from0=start_Delta_from0, uniform_Omega_Delta_ramp=uniform_Omega_Delta_ramp)

       
    Delta_global = Delta_mean_ls[0] - Delta_local_ls[0]/2
    Delta = Delta_global + h_ls_pre[0][q_index]* Delta_local_ls[0] # the global detuning is Delta_mean - Delta_local/2
    # Delta_corrected = Delta + correction

    simulated = simulated_data[0][0][0] # list of times, data
    if show_qutip:
        qutip_ = qutip_data[0][0][0] # list of times, data

    density_sim_ls = []
    density_sim_err_ls = []
    n_shots = gate_params_all[0][0]['n_shots']
    for m in range(len(t_plateau_qutip)):
        simulated_t = simulated[m][0]
        # print(simulated_t)
        probs_sim_t = bins_to_probs(report_to_bins(simulated_t),N)
        density_sim = [restrict_probabilities(probs_sim_t, [q_idx])[1] for q_idx in range(N-1, -1, -1)]  # reverse order to match little endian
        density_sim_ls.append(density_sim)
        density_sim_err_ls.append([np.sqrt(density_sim_ / n_shots) for density_sim_ in density_sim]) # get error bars due to poisson noise, sqrt(p / N)

    if show_qutip:
        density_qutip_ls = []
        density_qutip_extraOmega_ls = []
        for m in range(len(t_plateau_qutip)):
            qutip_t = qutip_[m][0]
            density_qutip = [restrict_probabilities(qutip_t, [q_idx])[1] for q_idx in range(N-1, -1, -1)]
            density_qutip_ls.append(density_qutip)

            # qutip_extraOmega_t = qutip_extraOmega_[m][0]
            # density_qutip_extraDelta = [restrict_probabilities(qutip_extraOmega_t, [q_idx])[1] for q_idx in range(N-1, -1, -1)]
            # density_qutip_extraOmega_ls.append(density_qutip_extraDelta)

    # set font
    mpl.rcParams.update({'font.size': fontsize})
    plt.rc('text', usetex=True)
    mpl.rc('text.latex', preamble=r"""
    \usepackage{amsmath}
    \usepackage{newtxtext,newtxmath}
    """)

    if not passed_ax:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    Omega = base_params['ev_params']['Omega']
    if show_qutip:
        print(density_qutip_ls)
        print("Plotting QuTip results") ## when running the open dynamics I get a nonetype error here
        density_qutip_ls = [density_qutip_ls_t[q_index] for density_qutip_ls_t in density_qutip_ls]
        print(color_ls)
        ax.plot(t_plateau_qutip, density_qutip_ls, label=rf'$\mathrm{{QuTip}}, \Omega = {Omega:.3g}\mu\mathrm{{s}}^{{-1}}, \Delta = {Delta}\mu\mathrm{{s}}^{{-1}}$', color=color_ls[0], linestyle='-', linewidth=6, alpha=alpha_val)

    # ax.scatter(t_plateau_ls, density_sim_ls, label=rf'$\mathrm{{Bloqade \, Sim}}$')
    if not show_qutip:
        # for q_idx in range(N):
        #     print("Delta for q_idx", q_idx, "is", Delta_mean_ls[0] + (h_ls_pre[0][q_idx]-0.5)* Delta_local_ls[0])
        #     print(q_idx, q_index)
        #     density_sim_ls_q = [density_sim_ls_t[q_idx] for density_sim_ls_t in density_sim_ls]
        #     density_sim_err_ls_q = [density_sim_err_ls_t[q_idx] for density_sim_err_ls_t in density_sim_err_ls]
        #     # ax.errorbar(t_plateau_qutip, density_sim_ls_q, yerr=density_sim_err_ls_q, label=rf'$\mathrm{{Bloqade \, Sim}}, \Omega = {Omega:.3g}\mu\mathrm{{s}}^{{-1}}, \Delta = {Delta}\mu\mathrm{{s}}^{{-1}}, \mathrm{{index}}={q_index}$' if q_idx==q_index else None, fmt='o-',   markeredgecolor='black', markeredgewidth=1.5, markersize=8)

        # average the nearest 5 points to smooth out the curve
        density_sim_ls_q_smooth = []
        t_plateau_qutip_smoothed = []
        window_size = 50
        density_sim_ls_q = [density_sim_ls_t[q_index] for density_sim_ls_t in density_sim_ls]
        density_sim_err_ls_q = [density_sim_err_ls_t[q_index] for density_sim_err_ls_t in density_sim_err_ls]
        for i in range(len(density_sim_ls_q)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(density_sim_ls_q), i + window_size // 2 + 1)
            window = density_sim_ls_q[start_idx:end_idx]
            density_sim_ls_q_smooth.append(np.mean(window))
            t_plateau_qutip_smoothed.append(t_plateau_qutip[i])


        ax.plot(t_plateau_qutip_smoothed, density_sim_ls_q_smooth,  label=rf'$\mathrm{{Bloqade \, Sim}}, \Omega = {Omega:.3g}\mu\mathrm{{s}}^{{-1}}, \Delta = {Delta}\mu\mathrm{{s}}^{{-1}}, \mathrm{{index}}={q_index}$', linestyle='-', linewidth=6)

    ## simulated fit
    # if not do_ramsey:
    #     sim_rabi_params, sim_cov = curve_fit(rabi_func, t_plateau_ls, density_sim_ls, p0=[1, Omega, 0, 1], bounds=([0, 0, -np.pi, -2], [np.inf, np.inf, np.pi, 2]))
    #     rabi_sim = [rabi_func(t, *sim_rabi_params) for t in t_plateau_qutip]
    #     rabi_sim_unc = np.sqrt(np.diag(sim_cov))[1]
    #     ax.plot(t_plateau_qutip, rabi_sim, label=rf'$\Omega_{{\mathrm{{fit}}}} = {sim_rabi_params[1]:.3g} \pm {rabi_sim_unc:.1g} \mu\mathrm{{s}}^{{-1}}$')

    if timestamp == 0: ## plot the simulated
        emuid, emu_data = get_all_single_hams_rand(h_ls_pre, x_pre, t_plateau_ls, seq_ls_pre_all_orig, base_params, Delta_mean_ls, Delta_local_ls, gate_params_all, cluster_spacing, manual_parallelization, is_expt_data=False, timestamp=0, dir_root=dir_root, force_recompute=force_recompute, debug=False, preset_opt=preset_opt, name=expt_name, override_local=override_local, save_mode=False, backup_dirs=None, x0_y0_offset=x0_y0_offset, start_Delta_from0=start_Delta_from0, uniform_Omega_Delta_ramp=uniform_Omega_Delta_ramp)

        expt_ = emu_data[0][0][0] # list of times, data
        density_expt_err_ls = []
        density_expt_ls = []
        for m in range(len(t_plateau_ls)):
            expt_t = expt_[m][0]
            probs_expt_t = bins_to_probs(report_to_bins(expt_t),N)
            density_expt = [restrict_probabilities(probs_expt_t, [q_idx])[1] for q_idx in range(N-1, -1, -1)]  # reverse order to match little endian
            density_expt_ls.append(density_expt)
            density_expt_err_ls.append([np.sqrt(density_expt_ / n_shots) for density_expt_ in density_expt]) # get error bars due to poisson noise, sqrt(p / N) 
        print("density_expt_ls:", density_expt_ls)
        density_expt_ls = [density_expt_ls_t[q_index] for density_expt_ls_t in density_expt_ls]
        density_expt_err_ls = [density_expt_err_ls_t[q_index] for density_expt_err_ls_t in density_expt_err_ls]
        ax.errorbar(t_plateau_ls, density_expt_ls, yerr=density_expt_err_ls, label=rf'$\mathrm{{Bloqade \, Emul}}$', fmt='o',   markeredgecolor='black', markeredgewidth=1.5, markersize=8, ecolor='black')

        # add vertical lines
        # vertical = [0.09, 0.13, .17, 0.23, 0.29, 0.33, 0.37, 0.43, 0.49, 0.53]
        # for v in vertical:
        #     ax.axvline(x=v, color='gray', linestyle='--', alpha=0.5)


    
    qutip_data_extraOmega = None
    if timestamp!=0 or extra_expt_dir is not None:
        if timestamp != 0:
            print(f"inside, {x0_y0_offset}")
            euid, expt_data = get_all_single_hams_rand(h_ls_pre, x_pre, t_plateau_ls, seq_ls_pre_all_orig, base_params,
            Delta_mean_ls, Delta_local_ls, gate_params_all, cluster_spacing,
            manual_parallelization, is_expt_data=True, timestamp=timestamp, dir_root=dir_root,
            force_recompute=force_recompute, debug=False, preset_opt=preset_opt,
            name=expt_name, override_local=override_local, save_mode=True, backup_dirs=None, x0_y0_offset=x0_y0_offset, start_Delta_from0=start_Delta_from0, uniform_Omega_Delta_ramp=uniform_Omega_Delta_ramp, phi_mode=phi_mode, Delta_local_ramp_time=Delta_local_ramp_time, Omega_delay_time=Omega_delay_time)
            
            expt_ = expt_data[0][0][0] # list of times, data
            print("expt_:", expt_)
            density_expt_err_ls = []
            density_expt_ls = []
            for m in range(len(t_plateau_ls)):
                expt_t = expt_[m][0]
                probs_expt_t = bins_to_probs(report_to_bins(expt_t),1)
                density_expt = [restrict_probabilities(probs_expt_t, [q_idx])[1] for q_idx in range(N-1, -1, -1)]  # reverse order to match little endian
                density_expt_ls.append(density_expt)
                density_expt_err_ls.append([np.sqrt(density_expt_ / n_shots) for density_expt_ in density_expt]) # get error bars due to poisson noise, sqrt(p / N)

            print("density_expt_ls:", density_expt_ls)
            # if for any of the times it's all 0000, then remove this time point
            # valid_indices = [i for i, d in enumerate(density_expt_ls) if d > 0]
            # valid_indices = [i for i, d in enumerate(density_expt_ls) if not (d[q_index] == 0 or d[q_index] == 1)]
            # ensure no nans
            valid_indices = [i for i, d in enumerate(density_expt_ls) if not (np.isnan(d[q_index]) or d[q_index] == 0 or d[q_index] == 1)]
            print("num of invalid indices = ", len(t_plateau_ls) - len(valid_indices))
            t_plateau_ls = [t_plateau_ls[i] for i in valid_indices]
            density_expt_ls = [density_expt_ls[i] for i in valid_indices]
            density_expt_err_ls = [density_expt_err_ls[i] for i in valid_indices]
       
        elif extra_expt_dir is not None:
            # .npy
            data_files = os.listdir(extra_expt_dir)
            data_files = [f for f in data_files if f.endswith('.npy')]
            density_expt_ls = []
            density_expt_err_ls = []
            expt_t_ls = []
            for data_file in data_files:
                # each file is the time point
                data_path = os.path.join(extra_expt_dir, data_file)
                expt_data = np.load(data_path, allow_pickle=True)
                # open the corresponnding json file
                with open(os.path.join(extra_expt_dir, data_file.replace('.npy', '.json')), 'r') as f:
                    expt_params = json.load(f)
                t = expt_params['bloqade.analog.task.batch.RemoteBatch']['tasks'][0][1]['bloqade.analog.task.braket.BraketTask']['task_ir']['effective_hamiltonian']['rydberg']['rabi_frequency_amplitude']['global']['times'][-1]
                density = expt_data[0][0] * expt_data[0][1] + expt_data[1][0] * expt_data[1][1]
                n_shots = expt_data[0][1] + expt_data[1][1] 
                density /= n_shots
                density_expt_ls.append(density)
                density_expt_err_ls.append(np.sqrt(density / n_shots)) # get error bars due to poisson noise, sqrt(p / N)
                expt_t_ls.append(t)

            # sort the density_expt_ls by expt_t_ls
            sorted_indices = np.argsort(expt_t_ls)
            density_expt_ls = [density_expt_ls[i] for i in sorted_indices]

        # ax.scatter(t_plateau_ls, density_expt_ls, label=rf'$\mathrm{{Aquila}}$', marker='s', color='red')
        density_expt_ls = [density_expt_ls_t[q_index] for density_expt_ls_t in density_expt_ls]
        density_expt_err_ls = [density_expt_err_ls_t[q_index] for density_expt_err_ls_t in density_expt_err_ls]
        # density_expt_ls = np.array(density_expt_ls)
        # density_expt_err_ls = np.array(density_expt_err_ls)

        ax.errorbar(t_plateau_ls, density_expt_ls, yerr=density_expt_err_ls, label=rf'$\mathrm{{Aquila}}$', fmt='s', markerfacecolor=color_ls[1],   markeredgecolor='black', markeredgewidth=2, markersize=15, ecolor=color_ls[1], elinewidth=3, capsize=6)

        #### GETTING THE FIT PARAMS HERE!!!
        if not do_ramsey:
            # pop_max = np.max(density_expt_ls)
            # expt_rabi_params, expt_cov = curve_fit(rabi_func, t_plateau_ls, density_expt_ls, p0=[1, Omega, 0, 1], bounds=([0, 0, -np.pi, -2], [np.inf, np.inf, np.pi, 2]))

            t   = np.asarray(t_plateau_ls, float)
            y   = np.asarray(density_expt_ls, float)
            print("density_expt_ls:", density_expt_ls)
            sig = np.asarray(density_expt_err_ls, float)

            # Fit with experimental uncertainties
            p0     = [(Omega**2 / (Omega**2 + Delta**2))/2, Omega, 0, 1, 5]
            bounds = ([0, 0, -np.pi, -2, 0], [1, 10*Omega,  np.pi, 2, 10])

            expt_rabi_params, expt_cov = curve_fit(
                rabi_func, t, y, p0=p0, bounds=bounds,
                sigma=sig, absolute_sigma=True, maxfev=20000
            )

            A_expt, Omega_rabi, varphi_expt, B_expt, t_c_expt = expt_rabi_params
            A_expt_unc, Omega_expt_unc, varphi_expt_unc, B_expt_unc, t_c_expt_unc = np.sqrt(np.diag(expt_cov))

            # Reduced chi-square (uses the same σ as in the fit)
            resid   = y - rabi_func(t, *expt_rabi_params)
            chi2    = np.sum((resid / sig)**2)
            dof     = y.size - expt_rabi_params.size
            chi2_red = chi2 / dof


            t_fit_ls = np.linspace(min(t_plateau_ls), max(t_plateau_ls), fit_num_pts)
            rabi_expt_ls = [rabi_func(t, *expt_rabi_params) for t in t_fit_ls]
            
            ax.plot(t_fit_ls, rabi_expt_ls, label=rf'$\Omega_{{\mathrm{{fit}}}} = {Omega_rabi:.3g} \pm {Omega_expt_unc:.1g} \mu\mathrm{{s}}^{{-1}}$', color='red', linewidth=6)

            # print("fractional diff Omega:", np.abs(Omega_rabi - sim_rabi_params[1]) / sim_rabi_params[1])
            # print("how many sigma from expt to sim", np.abs(Omega_rabi - sim_rabi_params[1]) / Omega_expt_unc)

            print(f"{expt_name}, {Omega_rabi:.3g} ± {Omega_expt_unc:.1g} MHz, A = {A_expt:.3g} ± {A_expt_unc:.1g}, varphi = {varphi_expt:.3g} ± {varphi_expt_unc:.1g}, B = {B_expt:.3g} ± {B_expt_unc:.1g} MHz")

             # correction = - 1.7
            if show_qutip:
                base_params_qutip = deepcopy(base_params_qutip)
                Omega = base_params_qutip['ev_params']['Omega']

                Omega_eff_theory = np.sqrt(Omega**2 + Delta**2)
                
                delta_Omega = Omega_eff_theory - Omega_rabi
                num_sigma = np.abs(delta_Omega) / Omega_expt_unc

                # if num_sigma > 3:
                #     fractional_diff = delta_Omega / Omega
                #     Omega_corrected = Omega * (1 - fractional_diff)

                #     base_params_qutip['ev_params']['Omega'] = Omega_corrected

                #     qmuid, qutip_data_extraOmega = get_all_qutip_probs(h_ls_pre_qutip, x_pre_qutip, t_plateau_qutip, seq_ls_pre_all, base_params_qutip, Delta_mean_ls_qutip, Delta_local_ls_qutip, gate_params_all_qutip, dir_root=dir_root, force_recompute=force_recompute, neg_phi=neg_phi, preset_opt=preset_opt, override_local=override_local_qutip)

                #     qutip_extraOmega_ = qutip_data_extraOmega[0][0][0] # list of times, data

                #     density_qutip_extraOmega_ls = []
                #     for m in range(len(t_plateau_qutip)):
                #         qutip_extraOmega_t = qutip_extraOmega_[m][0]
                #         density_qutip_extraDelta = [restrict_probabilities(qutip_extraOmega_t, [q_idx])[1] for q_idx in range(N-1, -1, -1)]
                #         density_qutip_extraOmega_ls.append(density_qutip_extraDelta)

                #     ax.plot(t_plateau_qutip, density_qutip_extraOmega_ls, label=rf'$\mathrm{{QuTip}}, \Omega = {Omega_corrected:.3g}\mu\mathrm{{s}}^{{-1}}, \Delta = {Delta}\mu\mathrm{{s}}^{{-1}}$', color='orange', linestyle='--', linewidth=6)

        if do_ramsey:
            ramsey_params, ramsey_cov = curve_fit(ramsey_func, t_plateau_ls, density_expt_ls, p0=[1, 1], bounds=([0, 0], [np.inf, np.inf]))
            T2_star = ramsey_params[1]
            T2_star_unc = np.sqrt(np.diag(ramsey_cov))[1]
            ramsey_fit = [ramsey_func(t, *ramsey_params) for t in t_plateau_qutip]
            ax.plot(t_plateau_qutip, ramsey_fit, label=rf'$T_2^* = {T2_star:.3g} \pm {T2_star_unc:.1g} \mu s$')

    # using the T2* let's perform hamiltonian averaging
    if T2_star is not None and not do_ramsey :
        n_ens = 1

        Delta_std = np.sqrt(2) / T2_star
        ensemble = []
        for _ in trange(n_ens):
            # pick Delta_globally from gaussian centered at 0 with std Delta_std
            Delta_global = np.random.normal(0, Delta_std)
            # now rerun qutip
            bmuid, bloqade_data = get_all_qutip_probs(h_ls_pre_qutip, x_pre_qutip, t_plateau_qutip, seq_ls_pre_all, base_params_qutip, [Delta_global], Delta_local_ls_qutip, gate_params_all_qutip, dir_root=dir_root, force_recompute=force_recompute, neg_phi=neg_phi, preset_opt=preset_opt)
            bloqade_ = bloqade_data[0][0][0]  # list of times, data
            bloqade_density_ls = []
            for m in range(len(t_plateau_qutip)):
                bloqade_t = bloqade_[m][0]
                density_bloqade = bloqade_t[1]
                bloqade_density_ls.append(density_bloqade)
            ensemble.append(bloqade_density_ls)
        ensemble_mean = np.mean(ensemble, axis=0)
        ax.plot(t_plateau_qutip, ensemble_mean, label=rf'${n_ens}\mathrm{{Ensemble \, Avg \, }} \Delta \, \sim \, N(0, \sqrt{{2}}/T_2^*$', color='magenta', linestyle=':')

        Delta_std = np.sqrt(2) / T2_star * 6
        ensemble = []
        for _ in trange(n_ens):
            # pick Delta_globally from gaussian centered at 0 with std Delta_std
            Delta_global = np.random.normal(0, Delta_std)
            # now rerun qutip
            bmuid, bloqade_data = get_all_qutip_probs(h_ls_pre_qutip, x_pre_qutip, t_plateau_qutip, seq_ls_pre_all, base_params_qutip, [Delta_global], Delta_local_ls_qutip, gate_params_all_qutip, dir_root=dir_root, force_recompute=force_recompute, neg_phi=neg_phi, preset_opt=preset_opt)
            bloqade_ = bloqade_data[0][0][0]  # list of times, data
            bloqade_density_ls = []
            for m in range(len(t_plateau_qutip)):
                bloqade_t = bloqade_[m][0]
                density_bloqade = bloqade_t[1]
                bloqade_density_ls.append(density_bloqade)
            ensemble.append(bloqade_density_ls)
        ensemble_mean = np.mean(ensemble, axis=0)
        ax.plot(t_plateau_qutip, ensemble_mean, label=rf'${n_ens}\mathrm{{Ensemble \, Avg \, }} \Delta \, \sim \, N(0, 6\sqrt{{2}}/T_2^*$', color='turquoise', linestyle=':')

   
    # print("qutip density before correction:", density_qutip_ls)


    ###### CALCULTING READOUT ERRORS FROM THE FIT
    if error_r is None or error_g is None:
        # density_qutip_ls = density_qutip_ls[q_index]
        if timestamp != 0 and not override_error:
            # f = Omega**2 / (Omega**2 + Delta**2)
            # error_r =  1 - (A_expt + B_expt) / f
            # error_r_unc = np.sqrt(A_expt_unc**2 + B_expt_unc**2)
            # error_g = B_expt - A_expt 
            # error_g_unc = np.sqrt(A_expt_unc**2 + B_expt_unc**2)

            iA, iB = 0, 3

            varA = expt_cov[iA, iA]
            varB = expt_cov[iB, iB]
            covAB = expt_cov[iA, iB]

            print("NEW READOUTERROR ERROR", varA, varB, covAB)

            f = Omega**2 / (Omega**2 + Delta**2)

            assert np.isclose(f, 1.0, atol=1e-6), f"f={f} too far from 1, cannot use this method to extract readout errors"

            # g = B - A
            error_g = B_expt - A_expt
            error_g_unc = np.sqrt(varA + varB - 2 * covAB)

            # r = 1 - (A + B)/f
            error_r = 1 - (A_expt + B_expt) / f
            error_r_unc = np.sqrt(varA + varB + 2 * covAB) / abs(f)


            if error_r < 0:
                error_r = 0
            if error_g < 0:
                error_g = 0
        else:
            error_r = error_r
            error_r_unc = 0
            error_g = error_g
            error_g_unc = 0

    if error_r is not None and error_g is not None:
        density_qutip_ls_corr = np.array(deepcopy(density_qutip_ls))
        density_qutip_ls_corr = density_qutip_ls_corr * (1 - error_r) + error_g * (1 - density_qutip_ls_corr)
        # density_qutip_ls = (density_qutip_ls - error_g) / (1 - error_g - error_r)  

        ax.plot(t_plateau_qutip, density_qutip_ls_corr, label=rf'$\mathrm{{QuTip}}, \Omega = {Omega:.3g}\mu\mathrm{{s}}^{{-1}}, \Delta = {Delta}\mu\mathrm{{s}}^{{-1}}, \epsilon_r = {format_value_uncertainty(error_r, error_r_unc)}, \epsilon_g = {format_value_uncertainty(error_g, error_g_unc)}$', color=color_ls[2], linestyle=':', linewidth=6, alpha=alpha_val)
        # print("qutip density after correction:", density_qutip_ls)

    # correct also the extraDelta
    if qutip_data_extraOmega is not None:
        density_qutip_extraOmega_ls_corr = np.array(deepcopy(density_qutip_extraOmega_ls))
        density_qutip_extraOmega_ls_corr = density_qutip_extraOmega_ls_corr *   (1 - error_r) + error_g * (1 - density_qutip_extraOmega_ls_corr)


        ax.plot(t_plateau_qutip, density_qutip_extraOmega_ls_corr, label=rf'$\mathrm{{QuTip}}, \Omega = {Omega_corrected:.3g}\mu\mathrm{{s}}^{{-1}}, \Delta = {Delta:.3g}\mu\mathrm{{s}}^{{-1}}, \epsilon_r = {format_value_uncertainty(error_r, error_r_unc)}, \epsilon_g = {format_value_uncertainty(error_g, error_g_unc)}$', color='cyan', linestyle='-.', linewidth=6)

    # else: # leave error_r and error_g as is
    #     error_r_unc = 0
    #     error_g_unc = 0

    # find 15 time points from density_qutip_extraOmega_ls_corr and density_qutip_ls_corr that are maximally separated in value
    # density_diff = np.abs(np.array(density_qutip_extraOmega_ls_corr) - np.array(density_qutip_ls_corr))
    # # get indices of the 15 largest values, but impose a minum spacing of .1 in the unit of t_plateau_qutip
    # selected_indices = []
    # for _ in range(15):
    #     if len(selected_indices) == 0:
    #         idx = np.argmax(density_diff)
    #         selected_indices.append(idx)
    #     else:
    #         idx = np.argmax(density_diff)
    #         if all(abs(idx - s) > 2 for s in selected_indices):  
    #             selected_indices.append(idx)
    #     density_diff[idx] = -1  # so we don't pick it again
    # selected_indices = sorted(selected_indices)




    # plot the value from rabi
    # if Omega_rabi is not None and pop_max is not None:
    #     Delta_rabi = Omega * np.sqrt((1/pop_max -1))
    #     # do it for bloqade
    #     bmuid, bloqade_data = get_all_single_hams_rand(h_ls_pre, x_pre, t_plateau_ls, seq_ls_pre_all, base_params, Delta_mean_ls, Delta_local_ls, gate_params_all, cluster_spacing, manual_parallelization, is_expt_data=False, timestamp=0, dir_root=dir_root, force_recompute=force_recompute, preset_opt=preset_opt, name=name)
    #     bloqade_ = bloqade_data[0][0][0]
    #     bloqade_density_ls = []
    #     for m in range(len(t_plateau_ls)):
    #         bloqade_t = bloqade_[m][0]
    #         probs_bloqade_t = bins_to_probs(report_to_bins(bloqade_t),1)
    #         density_bloqade = probs_bloqade_t[1]
    #         bloqade_density_ls.append(density_bloqade)  
    #     bloqade_rabi_params, bloqade_cov = curve_fit(rabi_func, t_plateau_ls, bloqade_density_ls, p0=[1, 2*np.pi, 0, 1], bounds=([0, 0, -np.pi, -2], [np.inf, np.inf, np.pi, 2]))
    #     bloqade_Omega_rabi = bloqade_rabi_params[1]
    #     bloqade_rabi_fit = [rabi_func(t, *bloqade_rabi_params) for t in t_plateau_qutip]
    #     bloqade_Omega_rabi_unc = np.sqrt(np.diag(bloqade_cov))[1]
    #     ax.plot(t_plateau_qutip, bloqade_rabi_fit, label=rf'$\mathrm{{Bloqade \, Ensemble \, Avg}}$, $\Omega_{{\mathrm{{fit}}}} = {bloqade_Omega_rabi:.3g} \pm {bloqade_Omega_rabi_unc:.1g} \mu\mathrm{{s}}^{{-1}}$', color='blue', linestyle=':')

   
    for spine in ax.spines.values():
        spine.set_visible(True)
        ax.tick_params(axis='both', which='both', top=True, right=True)
        ax.minorticks_on()
        ax.tick_params(axis='both', which='both', length=4, color='gray')
        

    ax.set_xlabel(r'$t_{\mathrm{evol}}\, (\mu \mathrm{s})$', fontsize=1.3*fontsize)
    ax.set_ylabel(r'$\mathrm{P}(|r\rangle)$', fontsize=1.3*fontsize)
    ax.set_ylim(0, 1.05)
    
    
    if not passed_ax:
        # plt.suptitle(rf"$\mathrm{{Timestamp}} = {timestamp},\, \mathrm{{index}}={q_index}$")
        plt.suptitle(rf"{name}")
        ax.legend(fontsize=0.6*fontsize, loc='upper left')
        plt.tight_layout()
        os.makedirs(os.path.join(dir_root, "results"), exist_ok=True)
        savename = f"rabi_{task_name}_timestamp_{timestamp}_negphi_{neg_phi}.png"

        plt.savefig(os.path.join(dir_root, "results", savename),  dpi=300)
        # plt.show()
        print(f"Saved rabi results to {os.path.join(dir_root, 'results', savename)}")

    # if timestamp != 0:
    #     return Omega_rabi, Omega_expt_unc, varphi_expt, varphi_expt_unc, A_expt, A_expt_unc, B_expt, B_expt_unc
    # return None

    # save the fit params to a global csv file. include task_name, timestamp, Omega_rabi, Omega_expt_unc, varphi_expt, varphi_expt_unc, A_expt, A_expt_unc, B_expt, B_expt_unc
    if global_file is not None:
        # check if the file exists 
        file_exists = os.path.isfile(global_file)
        # assume csv
        if file_exists:
            df_global = pd.read_csv(global_file)
        else:
            df_global = pd.DataFrame(columns=['name', 'task_name', 'timestamp', 'chi2_red', 'Omega', 'Delta', 'Omega_eff_theory','Omega_eff_expt', 'Omega_eff_expt_unc', 'varphi', 'varphi_unc', 'A', 'A_unc', 'B', 'B_unc', 'epsilon_r', 'epsilon_r_unc', 'epsilon_g', 'epsilon_g_unc'])
        if timestamp != 0:
            new_row = {
                'name': name,
                'task_name': task_name.split('task_')[-1],
                'timestamp': timestamp,
                'chi2_red': chi2_red,
                'Omega': Omega,
                'Delta': Delta,
                'Omega_eff_theory': np.sqrt(Omega**2 + Delta**2),
                'Omega_eff_expt': Omega_rabi,
                'Omega_eff_expt_unc': Omega_expt_unc,
                'varphi': varphi_expt,
                'varphi_unc': varphi_expt_unc,
                'A': A_expt,
                'A_unc': A_expt_unc,
                'B': B_expt,
                'B_unc': B_expt_unc,
                'epsilon_r': error_r,
                'epsilon_r_unc': error_r_unc,
                'epsilon_g': error_g,
                'epsilon_g_unc': error_g_unc
            }
            df_global = pd.concat([df_global, pd.DataFrame([new_row])], ignore_index=True)
            # remove duplicates based on name, keep the one with the highest timestamp
            df_global = df_global.sort_values('timestamp', ascending=False).drop_duplicates(subset=['name'], keep='first')
            df_global.to_csv(global_file, index=False)
            print(f"Saved global rabi fit results to {global_file}")

    if passed_ax:
        return ax
    else:
        return None



if __name__ == "__main__":
    pass