## the funcs from master_params_rbp needed to submit the rabi calibs
from QuEraToolbox.expt_file_manager import ExptStore
from QuEraToolbox.hamiltonian import get_h_ls
import os, json
import numpy as np

# -------- create tasks
def gen_seq_ls_pre(n_Dm, n_Dl, n_U_ls, n_gates_ls,n_ens, n_t_plateau, same_U_all_time=False):

    def gen_seq_ls(n_U, n_gates): # for one time
        seq_ls = []
        for _ in range(n_U):
            seq = np.random.randint(1, 3, size=n_gates).tolist()
            seq_ls.append(seq)
        return seq_ls

    seq_pre_ls_all = np.zeros((n_Dm, n_Dl, n_ens,n_t_plateau), dtype=object)
    for i in range(n_Dm):
        for j in range(n_Dl):
            for l in range(n_ens):
                n_U = n_U_ls[i][j]
                n_gates = n_gates_ls[i][j]
                if n_U != 0 and n_gates != 0:
                    if not same_U_all_time:
                        for k in range(n_t_plateau):
                            seq_ls = gen_seq_ls(n_U, n_gates)
                            seq_pre_ls_all[i, j, l, k] = seq_ls
                        
                    else:
                        seq_ls = gen_seq_ls(n_U, n_gates) # fix the sequence for all time
                        for k in range(n_t_plateau):
                            seq_pre_ls_all[i, j, l, k] = seq_ls

    return seq_pre_ls_all.tolist()

def gen_h_ls_pre(N, n_ens, threshold=0):
    h_ls_pre = []
    for _ in range(n_ens):
        h_ls_pre.append(get_h_ls(N, threshold=threshold).tolist())  

    return h_ls_pre

def gen_tasks(N, Delta_mean_ls, Delta_local_ls, base_params, gate_params_all, cluster_spacing, t_plateau_ls, dir_root, same_U_all_time=False, h_ls_pre=None, seq_ls_pre_all = None, manual_parallelization=False, x_pre=None, override_local=False, x0_y0_offset = (0,0)):

    manager = ExptStore(dir_root)   
    os.makedirs(os.path.join(dir_root, "tasks"), exist_ok=True)

    if cluster_spacing is None:
        manual_parallelization = False

    # first create task_stem
    if x0_y0_offset != (0,0):
        task_stem_payload = {
            "N": N,
            "Delta_mean_ls": Delta_mean_ls,
            "Delta_local_ls": Delta_local_ls,
            "base_params": base_params,
            "gate_params_all": gate_params_all,
            "t_plateau_ls": list(t_plateau_ls),
            "cluster_spacing": cluster_spacing,
            "manual_parallelization": manual_parallelization,
            "x0_y0_offset": x0_y0_offset,
            "override_local": override_local,
        }
    else:
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

    suid, added = manager.add(task_stem_payload, timestamp=0)
    stem_task_name = f"stem_{suid}"
    if added:
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
            n_U_ls[i, j] = gate_params_all[i][j]['n_U']
            n_gates_ls[i, j] = gate_params_all[i][j]['n_gates']

    if seq_ls_pre_all is None:
        seq_ls_pre_all = gen_seq_ls_pre(n_Dm, n_Dl, n_U_ls, n_gates_ls, n_ens, n_t_plateau, same_U_all_time=same_U_all_time)

    # print("seq_ls_pre_all", seq_ls_pre_all)

    task_payload = {
        "stem": stem_task_name,
        "h_ls_pre": h_ls_pre,
        "x_pre": x_pre if x_pre is not None else [[(i*base_params['ev_params']['a'], 0) for i in range(N)] for _ in range(len(h_ls_pre))],
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
    return task_name, seq_ls_pre_all


def get_calib_task(dir_root, ct, ham_count, J = 5.42):

    calib_params = [[{"n_U":0, "n_gates":0, "Delta_global": 0, "Delta_local": 0, "gate_duration": 0, "n_shots":100}]]

    calib_base_params = {
            "ev_params": {
                "Omega": 15.8,
                "Delta_local": 0, # placeholder, overwritten later
                "Delta_global": 0,  # placeholder, overwritten later
                "phi": 0, # changed from pi/2 to 0
                "t_ramp": 0.0632,
                "a": 10
            },
            "n_ens": 1
    }

    calib_t_plat_ls = np.linspace(0.05, 0.6, 10).tolist()

    # calib_Delta_mean = 1.5*J
    # calib_Delta_local = -J
    calib_Delta_mean = 0
    calib_Delta_local = 0
    calib_x0_y0_offset = (0,0)

    calib_task, calib_seq_ls_pre_all = gen_tasks(N=1, Delta_mean_ls=[calib_Delta_mean], Delta_local_ls=[calib_Delta_local], base_params=calib_base_params, gate_params_all=calib_params, cluster_spacing = None, t_plateau_ls = calib_t_plat_ls, dir_root = dir_root, override_local=True, h_ls_pre=[[1]], x0_y0_offset=calib_x0_y0_offset)

    calib_name = f"calib-{ham_count}_{calib_task.replace('task_', '')}_{ct}"


    return calib_name, calib_seq_ls_pre_all, calib_base_params, calib_params, calib_Delta_mean, calib_Delta_local, calib_t_plat_ls, calib_x0_y0_offset

    