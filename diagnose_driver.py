## where we run all the experiments in order to understand the quera data
import numpy as np
import time, json, os, shutil
from master_params_rbp import do_preset, gen_tasks, det_cost, execute_bloqade_task, get_subdirname, read_expt_task
from QuEraToolbox.expt_file_manager import ExptStore
from QuEraToolbox.random_bp_prep import get_duid
from QuEraToolbox.hamiltonian import get_h_ls
import matplotlib as mpl
import matplotlib.pyplot as plt

"""
instructions for running:
1. run submit_expts
2. run save_expts, zip the entire directory and download
3. locally, run process_expts to generate the plots
"""


def submit_expts(tasks_ls, preset_opts_ls, name_ls, dir_root, is_expt_data=True, debug=False, force_recompute=False, override=False, after_how_many_ham_run_check=None, ham_check_dir_main =None):
    assert len(tasks_ls) == len(preset_opts_ls) == len(name_ls), \
        "tasks_ls, preset_opts_ls, and name_ls must have the same length"
    print("SUBMITTING EXPTS", "-"*20)
    total_cost = sum([det_cost(task, dir_root) for task in tasks_ls])
    
    if not override:
        proceed = input(f"Total cost for submitting {len(tasks_ls)} experiments: ${total_cost}. Proceed? (y/n) ").strip().lower()
    else:
        proceed = 'y'


    if proceed != 'y':
        print("Aborting submission.")
        return None
    
    submitted = []
    for i, (task, preset_opt, name) in enumerate(zip(tasks_ls, preset_opts_ls, name_ls), start=1):
        cost = det_cost(task, dir_root)

        print("-"*20)
        print(f"[{i}/{len(tasks_ls)}] cost for {task}: {cost}. Auto-submitting (is_expt_data=False).")
        if is_expt_data:
            timestamp = int(time.time())
        else:
            timestamp = 0
        data_subdir = get_subdirname(name, task, timestamp, preset_opt)
        execute_bloqade_task(
            task, name, is_expt_data, timestamp, dir_root,
            force_recompute=force_recompute, debug=debug, preset_opt=preset_opt, data_subdir=data_subdir,
            after_how_many_ham_run_check=after_how_many_ham_run_check, ham_check_dir_main =ham_check_dir_main,
        )
        submitted.append((task, name, timestamp, preset_opt, cost, data_subdir))
    
    # save list of submitted expts
    manager = ExptStore(dir_root)
    manager_params = {
        "is_expt_data": is_expt_data,
        'tasks_ls': [t[0] for t in submitted],
        'name_ls': [t[1] for t in submitted],
        'timestamp_ls': [t[2] for t in submitted],
        'preset_opts_ls': [t[3] for t in submitted],
        'cost_ls': [t[4] for t in submitted],
        'data_subdirs': [t[5] for t in submitted]
    }
    suid, added = manager.add(manager_params, timestamp=0)
    print(f"Saved to {dir_root}/combos/{suid}.json.")
    # copy into the data folder for convenience
    os.system(f"cp {dir_root}/combos/{suid}.json {dir_root}/data/{suid}.json")
    print("Total cost:", sum(manager_params['cost_ls']))
    return suid

def save_expts(submitted_muid, dir_root, is_expt_data=True, main_aws_dir=None): 
    # for subdir in subdir_ls:
    #     name_name, task_name, timestamp, preset_opt = read_subdir_name(subdir)
    submitted_file = f"{dir_root}/combos/{submitted_muid}.json"
    with open(submitted_file, "r", encoding="utf-8") as f:
        payload = json.load(f)['payload']
        tasks_ls = payload['tasks_ls']
        name_ls = payload['name_ls']
        timestamp_ls = payload['timestamp_ls']
        preset_opts_ls = payload['preset_opts_ls']
    submitted = [(task, name, timestamp, preset_opt) for task, name, timestamp, preset_opt in zip(tasks_ls, name_ls, timestamp_ls, preset_opts_ls)]

    if main_aws_dir is None:
        for task, name, timestamp, preset_opt in submitted:
            execute_bloqade_task(task, name, is_expt_data, timestamp, dir_root, force_recompute=True, allow_override_name=False, debug=False, preset_opt=preset_opt, save_mode=True) # retrieve the data
    else:
        # get all the dirs inside main_aws_dir  
        aws_subdirs = [d for d in os.listdir(main_aws_dir) if os.path.isdir(os.path.join(main_aws_dir, d))]
        aws_subdirs = sorted(aws_subdirs, key=lambda x: x.split("__")[0])
        aws_index = 0

        for task, name, timestamp, preset_opt in submitted:
            h_ls_pre, x_pre, t_plateau_ls, seq_ls_pre_all, base_params, Delta_mean_ls, Delta_local_ls, gate_params_all, cluster_spacing, manual_parallelization, override_local, x0_y0_offset, t_delay = read_expt_task(task, dir_root)

            n_t_plateau = len(t_plateau_ls)
            n_seq = len(seq_ls_pre_all[0][0])
            needed_dirs = n_t_plateau * n_seq
            print(f"task {task} has {n_t_plateau} t_plateau and {n_seq} sequences, total {needed_dirs} subdirs needed.")

            dirs_collected = None
            placed_count = 0
            for idx in range(needed_dirs):
                if aws_index >= len(aws_subdirs):
                    print("Not enough AWS subdirectories to match the tasks. Stopping.")
                    return
                aws_subdir = aws_subdirs[aws_index]
                aws_index += 1
                print(f"Processing AWS subdir: {aws_subdir}")
                full_dir = os.path.join(main_aws_dir, aws_subdir)
                if dirs_collected is None:
                    dirs_collected = [
                        [
                            [
                                [[None for _ in range(n_seq)] for _ in range(n_t_plateau)]
                                for _ in range(len(h_ls_pre))
                            ]
                            for _ in range(len(Delta_local_ls))
                        ]
                        for _ in range(len(Delta_mean_ls))
                    ]
                t_idx, s_idx = np.unravel_index(idx, (n_t_plateau, n_seq))
                dirs_collected[0][0][0][t_idx][s_idx] = full_dir
                placed_count += 1
            print(f"Collected {placed_count} subdirs for task {task}. Now executing...")
            execute_bloqade_task(task, name, is_expt_data, timestamp, dir_root, force_recompute=True, debug=False, preset_opt=preset_opt, save_mode=True, backup_dirs=dirs_collected)
                
        # data_subdir = get_subdirname(name, task, timestamp, preset_opt)
        # print("data_subdir:", data_subdir)
        # save_data(data_subdir, dir_root) # save it as csv

    # dir_path = Path(zip_targ).resolve()  # path to the directory you want to zip
    # shutil.make_archive(
    #     base_name=str(dir_path),              # -> ".../dirname.zip"
    #     format="zip",
    #     root_dir=str(dir_path.parent),        # parent directory
    #     base_dir=dir_path.name                # include "dirname/" as top-level in the zip
    # )

def process_expts(submitted_muid, dir_root,epsilon_r, epsilon_g, epsilon_r_unc, epsilon_g_unc, ax_ret=False, colors_main = None, q_index=-1, show_qutip=True,  num_t_qutip_points=100, fontsize=25, default_color=True, global_csv='rabi_fitted_params.csv'):
    # q_index can be an int or list
    # if q_index then assert len(epsilon_r) == len(epsilon_g) == q_index. OR if epilson_r, epsilon_g are floats, then use same for all; OR they can be None

    submitted_file = f"{dir_root}/combos/{submitted_muid}.json"
    with open(submitted_file, "r", encoding="utf-8") as f:
        payload = json.load(f)['payload']
        tasks_ls = payload['tasks_ls']
        print("tasks_ls:", tasks_ls)
        name_ls = payload['name_ls']
        timestamp_ls = payload['timestamp_ls']
        preset_opts_ls = payload['preset_opts_ls']
    submitted = [(task, name, timestamp, preset_opt) for task, name, timestamp, preset_opt in zip(tasks_ls, name_ls, timestamp_ls, preset_opts_ls)]
    
    if ax_ret:
        mpl.rcParams.update({'font.size': fontsize})
        plt.rc('text', usetex=True)
        mpl.rc('text.latex', preamble=r"""
        \usepackage{amsmath}
        \usepackage{newtxtext,newtxmath}
        """)

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    else:
        ax = None

    ## reverse order
    submitted = submitted[::-1]
    # print(submitted)
    # submitted = [submitted[-1]]
    # print(submitted)
  
    for i, (task, name, timestamp, preset_opt) in enumerate(submitted):
        print(f"Processing task {task} with name {name} and timestamp {timestamp}")
        # if preset_opt is None and "rabi" not in name.lower() and single_params is not None:
        #     n_t, n_s, n_ens = single_params
        #     compare_bs_probs_single(task, name, dir_root, timestamp, n_t, n_s, n_ens, fontsize=16, J=J, include_bloqade_sim=True, include_qutip=True, neg_phi=False, force_recompute=True, preset_opt=preset_opt, error_g = 0.01, error_r = 0.1, show_poisson_errorbar=True, check_postarray=True)

        # else:
        # get num qubits
        h_ls_pre, x_pre, t_plateau_ls, seq_ls_pre_all, base_params, Delta_mean_ls, Delta_local_ls, gate_params_all, cluster_spacing, manual_parallelization, override_local, x0_y0_offset, t_delay, start_Delta_from0, uniform_Omega_Delta_ramp, phi_mode,Delta_local_ramp_time, Omega_delay_time  = read_expt_task(task, dir_root)

        t0 = min(0.05, min(t_plateau_ls))

        t_qutip_ls = np.linspace(t0, max(t_plateau_ls), num_t_qutip_points)

        # if q_index is a list, then index thru it by task position in tasks_ls
        if isinstance(q_index, list):
            idx = tasks_ls.index(task)
            q_idx = q_index[idx]
        else:
            q_idx = q_index 

        if isinstance(epsilon_r, list):
            error_r = epsilon_r[q_idx]
            error_r_unc = epsilon_r_unc[q_idx]
        else:
            error_r = epsilon_r
            error_r_unc = epsilon_r_unc

        if isinstance(epsilon_g, list):
            error_g = epsilon_g[q_idx]
            error_g_unc = epsilon_g_unc[q_idx]
        else:
            error_g = epsilon_g
            error_g_unc = epsilon_g_unc

        if colors_main is not None:
            color_ls = [colors_main[name_ls.index(name)]]*3 # same color for all 3 lines
        elif default_color:
            color_ls = ['purple', 'red', 'blue']
        else: # define one
            color_ls =  ['C0', 'C1', 'C2']

        # plot the last one with alpha = 1, others with 0.5
        if i == len(submitted) - 1 or default_color:
            alpha_val = 1.0
        else:
            alpha_val = 0.5

        ax = do_preset(task, timestamp, t_qutip_ls, ax=ax, fontsize=fontsize, force_recompute=False, neg_phi=True, do_ramsey=("ramsey" in name.lower()),name=name, overridename=None, T2_star=None, dir_root=dir_root, error_r = error_r, error_g = error_g, error_r_unc=error_r_unc, error_g_unc=error_g_unc, debug=False, q_index=q_idx, show_qutip=show_qutip, global_file=global_csv,color_ls=color_ls, alpha_val=alpha_val)
        
    if ax_ret:
        if 0 in timestamp_ls:
            # add gray vertical lines at the time points
            for t in t_plateau_ls:
                plt.axvline(x=t, color='black', linestyle='--', linewidth=6, alpha=0.8)

        plt.tight_layout()
        # ax.legend(fontsize=fontsize*.2)
        os.makedirs(os.path.join(dir_root, "results"), exist_ok=True)
        plt.savefig(os.path.join(dir_root, "results", f"{submitted_muid}.png"),  dpi=300)
        # plt.show()
        print(f"Saved rabi results to {os.path.join(dir_root, 'results', f'{submitted_muid}.png')}")
            
def combine_expts(submitted_muid, dir_root):
    """
    ONLY WORKS IF 1 HAM IN ENSEMBLE, 1 DELTA MEAN, 1 DELTA LOCAL, 1 SEQ, 1 T_PLATEAU PER SUBDIR
    """
    submitted_file = f"{dir_root}/combos/{submitted_muid}.json"
    with open(submitted_file, "r", encoding="utf-8") as f:
        payload = json.load(f)['payload']
        tasks_ls = payload['tasks_ls']
        name_ls = payload['name_ls']
        timestamp_ls = payload['timestamp_ls']
        preset_opts_ls = payload['preset_opts_ls']
    submitted = [(task, name, timestamp, preset_opt) for task, name, timestamp, preset_opt in zip(tasks_ls, name_ls, timestamp_ls, preset_opts_ls)]

    unique_tasks = list(set([t[0] for t in submitted]))
    print(f"Unique tasks found: {unique_tasks}")

    manager = ExptStore(dir_root)

    # bundle the submitted expts by unique task
    task_groups = {task: [] for task in unique_tasks}
    for task, name, timestamp, preset_opt in submitted:
        task_groups[task].append((name, timestamp, preset_opt))

    # now go through each of the keys of task_groups, create a combined submitted_muid and copy all the data into a new folder so can be procesed as normal task
    for task, group in task_groups.items():
        name_ls = [item[0] for item in group]
        timestamp_ls = [item[1] for item in group]
        preset_opts_ls = [item[2] for item in group]

        # make one str combining all the timestamps
        combined_timestamp = "_".join(map(str, timestamp_ls))
        
        manifest = {
            "tasks_ls":[task],
            "name_ls": [f"combined_{name_ls[0]}"],
            "timestamp_ls": [combined_timestamp],
            "preset_opts_ls":[preset_opts_ls[0]]  # or some other logic if needed

        }
        muid, added = manager.add(manifest, timestamp=0)
        print(f"Combined tasks saved to {dir_root}/combos/{muid}.json")
        # save also to data folder
        os.system(f"cp {dir_root}/combos/{muid}.json {dir_root}/data/{muid}.json")
        
        # now copy all the data into a new folder
        combined_subdir = get_subdirname(f"combined_{name_ls[0]}", task, combined_timestamp, preset_opts_ls[0])
        combined_full_path = os.path.join(dir_root, "data", combined_subdir)
        os.makedirs(combined_full_path, exist_ok=True)

        all_data = {}  # to accumulate counts for each basis state
        for name, timestamp, preset_opt in group:
            # subdir = get_subdirname(name, task, timestamp, preset_opt)
            # full_path = os.path.join(dir_root, "data", subdir)
            # if os.path.exists(full_path):
            #     for item in os.listdir(full_path):
            #         s = os.path.join(full_path, item)
            #         d = os.path.join(combined_full_path, item)
            #         if os.path.isdir(s):
            #             shutil.copytree(s, d, dirs_exist_ok=True)
            #         else:
            #             shutil.copy2(s, d)
            # else:
            #     print(f"Warning: {full_path} does not exist and will be skipped.")

            # instead of copying, read all the npy and combine into one npy
            subdir = get_subdirname(name, task, timestamp, preset_opt)
            full_path = os.path.join(dir_root, "data", subdir)
            if os.path.exists(full_path):
                npy_files = [f for f in os.listdir(full_path) if f.endswith(".npy")]
                
                for npy_file in npy_files:
                    npy_full_path = os.path.join(full_path, npy_file)
                    data = np.load(npy_full_path, allow_pickle=True)

                    # the data is a list of [basis_states, counts]
                    for pair in data:
                        basis_state = pair[0]
                        counts = pair[1]
                        if basis_state in all_data:
                            all_data[basis_state] += counts
                        else:
                            all_data[basis_state] = counts
            else:
                print(f"Warning: {full_path} does not exist and will be skipped.")

        # now convert back to an array of [basis_state, counts], sorted by counts descending
        combined_data = sorted(all_data.items(), key=lambda x: x[1], reverse=True)
        combined_data = np.array(combined_data, dtype=object)

        # need to get the correct duid name
        h_ls_pre, x_pre, t_plateau_ls, seq_ls_pre_all, base_params, Delta_mean_ls, Delta_local_ls, gate_params_all, cluster_spacing, manual_parallelization, override_local, x0_y0_offset, t_delay = read_expt_task(task, dir_root)

        n_ens = base_params['n_ens']
        assert n_ens == 1, "Combining only supported for n_ens = 1"

        ev_params = base_params['ev_params']

         # need to override the Delta_global and Delta_local in base_params
        Delta_global = Delta_mean_ls[0] - 1/2 * Delta_local_ls[0]
        ev_params['Delta_global'] = Delta_global
        ev_params['Delta_local'] = Delta_local_ls[0]

        print("ev_params used for duid:", ev_params)


        x = x_pre[0]
        h_ls = h_ls_pre[0]
        t_plateau = t_plateau_ls[0]
        seq = seq_ls_pre_all[0][0][0][0][0]
        print("seq used for duid:", seq)
        gate_params = gate_params_all[0][0]
        n_shots = gate_params['n_shots']
        is_expt_data = True  # since we are combining expt data
        
        
        uid, added = get_duid(h_ls, x, ev_params, t_plateau, seq, n_shots, gate_params, is_expt_data, dir_root, combined_timestamp, cluster_spacing=cluster_spacing, manual_parallelization=manual_parallelization, override_local=override_local, preset_opt=preset_opt, full_ev=True)

        combined_npy_path = os.path.join(combined_full_path, f"{uid}.npy")
        np.save(combined_npy_path, combined_data)
        print(f"Combined data saved to {combined_npy_path}.")
        
        
if __name__ == "__main__":

    dir_root = "diagnose_dir4_6"
    ## ---- expt0: single qubit ---- 
    a = 10
    J = 5.42

    base_params0_rabi = {
            "ev_params": {
                "Omega": 15.8,
                "Delta_local": 0, # placeholder, overwritten later
                "Delta_global": 0,  # placeholder, overwritten later
                "phi": 0,
                "t_ramp": 0.0632,
                "a": a
            },
            "n_ens": 1
    }
    base_params0_ramsey = {
            "ev_params": {
                "Omega": 6.29,
                "Delta_local": 0, # placeholder, overwritten later
                "Delta_global": 0,  # placeholder, overwritten later
                "phi": 0,
                "t_ramp": 0.0632,
                "a": a
            },
            "n_ens": 1
    }
    t_plateau_rabi = np.linspace(0.05, 1, 30).tolist()
    t_plateau_ramsey = np.linspace(0.05, 3.5, 30).tolist()
    
    gate_params0 = [[{"n_U":0, "n_gates":0, "Delta_global": 0, "Delta_local": 0, "gate_duration": 0, "n_shots":200}]]

    expt0_rabi_task = gen_tasks(N=1, Delta_mean_ls=[0], Delta_local_ls=[0], base_params=base_params0_rabi, gate_params_all=gate_params0, cluster_spacing = None, t_plateau_ls = t_plateau_rabi, dir_root = dir_root, override_local=True)
    expt0_ramsey_resonant = gen_tasks(N=1, Delta_mean_ls=[0], Delta_local_ls=[0], base_params=base_params0_ramsey, gate_params_all=gate_params0, cluster_spacing = None, t_plateau_ls = t_plateau_ramsey, dir_root = dir_root, override_local=True)
    
    t_qutip_rabi = np.linspace(0.05, 1, 1000).tolist()
    t_qutip_rasmey = np.linspace(0.05, 3.5, 1000).tolist()


    ## ---- N qubit expt, no local detuning ----
    t_plat_single = [1.1]
    # t_plat_single = 0.5
    # t_plat_single = np.linspace(0.05, 3, 10).tolist()

    base_params1 = {
            "ev_params": {
                "Omega": 15.8,
                "Delta_local": 0, # placeholder, overwritten later
                "Delta_global": 0,  # placeholder, overwritten later
                "phi": 0,
                "t_ramp": 0.0632,
                "a": a
            },
            "n_ens": 1
    }
    gate_params1 = [[{"n_U":0, "n_gates":0, "Delta_global": 0, "Delta_local": 0, "gate_duration": 0, "n_shots":1000}]]
    expt1_get_task = lambda N: gen_tasks(N=N, Delta_mean_ls=[J], Delta_local_ls=[0], base_params=base_params1, gate_params_all=gate_params1, cluster_spacing = None, t_plateau_ls = t_plat_single, dir_root = dir_root, override_local=True)
    expt1_name = lambda N: f"expt1-{N}"
    
    # just do N = 2
    # non0 phi
    base_params1_non0phi = {
            "ev_params": {
                "Omega": 15.8,
                "Delta_local": 0, # placeholder, overwritten later
                "Delta_global": 0,  # placeholder, overwritten later
                "phi": np.pi/2,
                "t_ramp": 0.0632,
                "a": a
            },
            "n_ens": 1
    }
    expt1_non0phi_get_task = lambda N: gen_tasks(N=N, Delta_mean_ls=[10*J], Delta_local_ls=[0], base_params=base_params1_non0phi, gate_params_all=gate_params1, cluster_spacing = None, t_plateau_ls = t_plat_single, dir_root = dir_root, override_local=True)
    expt1_non0phi_name = lambda N: f"expt1-non0phi-{N}" 

    # for the following, do 2, 4, 6 qubits
    # all h_i = 1, using 0 phi. 
    get_h_ls_pre_h1 = lambda N: [[1]*N]
    expt2_h1_get_task = lambda N: gen_tasks(N=N, Delta_mean_ls=[1.5*J], Delta_local_ls=[-J], base_params=base_params1_non0phi, gate_params_all=gate_params1, cluster_spacing = None, t_plateau_ls = t_plat_single, dir_root = dir_root, override_local=False, h_ls_pre=get_h_ls_pre_h1(N))
    expt2_h1_name = lambda N: f"expt2-h1-{N}"

    # random h_i
    # need to save it to file so can reproduce same task
    random_h_ls_file = f"{dir_root}/data/h_ls_random.json"
    try:
        with open(random_h_ls_file, "r", encoding="utf-8") as f:
            h_ls_all = json.load(f)
    except FileNotFoundError: # write empty file
        h_ls_all = {}

    def get_h_ls_pre_hrandom(N): 
        # read the h_ls_all_file and see if we have a value for the key N
        key = str(N)
        if key in h_ls_all.keys():
            return h_ls_all[key]
        else:
            h_ls_all[key] = [list(get_h_ls(N, threshold=0))]
            with open(random_h_ls_file, "w", encoding="utf-8") as f:
                json.dump(h_ls_all, f, indent=4)
            return h_ls_all[key]
    
    
    expt2_hrandom_get_task = lambda N: gen_tasks(N=N, Delta_mean_ls=[1.5*J], Delta_local_ls=[-J], base_params=base_params1_non0phi, gate_params_all=gate_params1, cluster_spacing = None, t_plateau_ls = t_plat_single, dir_root = dir_root, override_local=False, h_ls_pre=get_h_ls_pre_hrandom(N))
    expt2_hrandom_name = lambda N: f"expt2-hrandom-{N}"

    # flipping phi a few times--
    # first, no local detuning
    gate_params2 = [[{"n_U":1, "n_gates":4, "Delta_global": J, "Delta_local": 0, "gate_duration": 0.06, "n_shots":1000}]]

    seq_pre_ls_1 = [[[[[[1, 2, 2, 1, 2]]]]]]
    seq_pre_ls_2 = [[[[[[1, 2, 1, 2, 2, 1, 1, 1, 1, 2, 2, 1]]]]]]
    expt3_flipphi_get_task_nolocal_1 = lambda N: gen_tasks(N=N, Delta_mean_ls=[J], Delta_local_ls=[0], base_params=base_params1_non0phi, gate_params_all=gate_params2, cluster_spacing = None, t_plateau_ls = t_plat_single, dir_root = dir_root, override_local=True, seq_ls_pre_all=seq_pre_ls_1)
    expt3_flipphi_name_nolocal_1 = lambda N: f"expt3-flipphi-nolocal-1-{N}"

    expt3_flipphi_get_task_nolocal_2 = lambda N: gen_tasks(N=N, Delta_mean_ls=[J], Delta_local_ls=[0], base_params=base_params1_non0phi, gate_params_all=gate_params2, cluster_spacing = None, t_plateau_ls = t_plat_single, dir_root = dir_root, override_local=True, seq_ls_pre_all=seq_pre_ls_2)
    expt3_flipphi_name_nolocal_2 = lambda N: f"expt3-flipphi-nolocal-2-{N}"

    # then, with local detuning
    gate_params3 = [[{"n_U":1, "n_gates":4, "Delta_global": 1.5*J, "Delta_local": -J, "gate_duration": 0.06, "n_shots":1000}]]

    expt3_flipphi_get_task_h1_1 = lambda N: gen_tasks(N=N, Delta_mean_ls=[1.5*J], Delta_local_ls=[-J], base_params=base_params1_non0phi, gate_params_all=gate_params3, cluster_spacing = None, t_plateau_ls = t_plat_single, dir_root = dir_root, override_local=False, h_ls_pre=get_h_ls_pre_h1(N), seq_ls_pre_all=seq_pre_ls_1)
    expt3_flipphi_name_h1 = lambda N: f"expt3-flipphi-h1-1-{N}"

    expt3_flipphi_get_task_h1_2 = lambda N: gen_tasks(N=N, Delta_mean_ls=[1.5*J], Delta_local_ls=[-J], base_params=base_params1_non0phi, gate_params_all=gate_params3, cluster_spacing = None, t_plateau_ls = t_plat_single, dir_root = dir_root, override_local=False, h_ls_pre=get_h_ls_pre_h1(N), seq_ls_pre_all=seq_pre_ls_2)
    expt3_flipphi_name_h1_2 = lambda N: f"expt3-flipphi-h1-2-{N}"

    expt3_flipphi_get_task_hrandom_1 = lambda N: gen_tasks(N=N, Delta_mean_ls=[1.5*J], Delta_local_ls=[-J], base_params=base_params1_non0phi, gate_params_all=gate_params3, cluster_spacing = None, t_plateau_ls = t_plat_single, dir_root = dir_root, override_local=False, h_ls_pre=get_h_ls_pre_hrandom(N), seq_ls_pre_all=seq_pre_ls_1)
    expt3_flipphi_name_hrandom_1 = lambda N: f"expt3-flipphi-hrandom-1-{N}"

    expt3_flipphi_get_task_hrandom_2 = lambda N: gen_tasks(N=N, Delta_mean_ls=[1.5*J], Delta_local_ls=[-J], base_params=base_params1_non0phi, gate_params_all=gate_params3, cluster_spacing = None, t_plateau_ls = t_plat_single, dir_root = dir_root, override_local=False, h_ls_pre=get_h_ls_pre_hrandom(N), seq_ls_pre_all=seq_pre_ls_2)
    expt3_flipphi_name_hrandom_2 = lambda N: f"expt3-flipphi-hrandom-2-{N}"


    N = 1
    tasks_ls = [expt2_h1_get_task(N), expt2_hrandom_get_task(N)]
    name_ls = [expt2_h1_name(N), expt2_hrandom_name(N)]
    preset_opts_ls = [None for _ in range(len(tasks_ls))]

    is_expt_data = False  

    suid = submit_expts(tasks_ls, preset_opts_ls, name_ls, dir_root, is_expt_data=is_expt_data, debug=False, force_recompute=False)

    # save_expts(suid, dir_root, is_expt_data=is_expt_data,)
    # combine_expts(suid, dir_root)
    process_expts(suid, dir_root, J=J)

    