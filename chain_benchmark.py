from master_params_rbp import gen_tasks, det_cost, read_expt_task
import numpy as np
from diagnose_driver import submit_expts, save_expts, process_expts
import json


def gen_task_wrapper(dir_root, N=6, num_t_plateau_ls=30, n_shots=100, res_idx=-1, Delta_local=-10*5.42, t_start=0, uniform_Omega_Delta_ramp=False, start_Delta_from0=True, non_uniform_t_sampling=False):
    # generate the task
    same_U_all_time = True  # whether to repeat the same set of U params for all time points; saves lot of money and time if running sp
    Delta_mean_ls = [Delta_local/2] # so Delta_global = 0
    Delta_local_ls = [Delta_local]
    a = 10


    base_params = {
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
    
    cluster_spacing = None 
    manual_parallelization = False

    # x0 = [(i*a, 0) for i in range(N)]  # initial positions of the qubits
    x_pre = None


    gate_params_all = [ # has dimension n_Dm, n_Dl
            [{
                "gate_duration": 0, "n_gates": 0, 
                "Delta_local": 0, "Delta_global": 0,
                "n_U": 1, "n_shots": n_shots
            },
    ]]

    if not non_uniform_t_sampling:
        t_plateau_expt = np.linspace(0.05, .6, num_t_plateau_ls)  # in us
    elif non_uniform_t_sampling and num_t_plateau_ls == 10: # want to sample more densely around min and max of the oscillations  given that we have a strong prior on the experimebntal rabi frequency
        t_plateau_expt = np.array([0.09, 0.13, .17, 0.23, 0.29, 0.33, 0.37, 0.43, 0.49, 0.53])
    t_plateau_expt += t_start
    print("t_plateau_expt:", t_plateau_expt)

    h_ls_pre = [1]*N
    h_ls_pre[res_idx]= 0
    h_ls_pre = [h_ls_pre]  

    print(len(h_ls_pre[0]), "qubits in the task")


    expt_task_name = gen_tasks(N, Delta_mean_ls, Delta_local_ls, base_params, gate_params_all, cluster_spacing, t_plateau_expt, dir_root,same_U_all_time=same_U_all_time, h_ls_pre=h_ls_pre, manual_parallelization=manual_parallelization, x_pre=x_pre, uniform_Omega_Delta_ramp=uniform_Omega_Delta_ramp, start_Delta_from0=start_Delta_from0)

    det_cost(expt_task_name, dir_root)
    return expt_task_name

def submit_expt_wrapper(tasks_ls, preset_opts_ls, name_ls, dir_root, is_expt_data=False):
    suid = submit_expts(tasks_ls, preset_opts_ls, name_ls, dir_root, is_expt_data=is_expt_data, debug=False, force_recompute=False, override=True, after_how_many_ham_run_check=None, ham_check_dir_main =None)
    print("finished expt with suid:", suid)
    return suid

def save_expt_wrapper(dir_root, suid, is_expt_data=False):
    save_expts(suid, dir_root, is_expt_data=is_expt_data)

def process_expt_wrapper(dir_root, suid, q_index=-1, show_qutip=True, open_dynamics=False, num_repeats_coherent_noise=10, num_t_qutip_points=100, epsilon_r_ls=None, epsilon_r_unc_ls=None, epsilon_g_ls=None, epsilon_g_unc_ls=None):
    process_expts(suid, dir_root, q_index=q_index, show_qutip=show_qutip, open_dynamics=open_dynamics, num_repeats_coherent_noise=num_repeats_coherent_noise, num_t_qutip_points=num_t_qutip_points, epsilon_r=epsilon_r_ls, epsilon_r_unc=epsilon_r_unc_ls, epsilon_g=epsilon_g_ls, epsilon_g_unc=epsilon_g_unc_ls, default_color=True)


def run_benchmark_expt(dir_root, chunk_idx, expt_task, is_expt_data = False, N = 6, num_t_plateau_ls = 10, n_shots = 100, uniform_Omega_Delta_ramp = False, start_Delta_from0=False):        
    res_idx_ls = list(range(N//2)) ## only need readout error for first half of qubits    

    tasks_ls = []
    for res_idx in res_idx_ls:
        expt_task_name = gen_task_wrapper(dir_root, N=N, num_t_plateau_ls=num_t_plateau_ls, n_shots=n_shots, res_idx=res_idx, Delta_local=-125, t_start=0.0, uniform_Omega_Delta_ramp=uniform_Omega_Delta_ramp, start_Delta_from0=start_Delta_from0, non_uniform_t_sampling=True)
        tasks_ls.append(expt_task_name)
        
    preset_opts_ls = [None] * len(tasks_ls)
    if "task_" in expt_task:
        expt_task_name = expt_task.split("task_")[-1]
    name_ls = [f"q{res_idx}-chunk{chunk_idx}-expt{expt_task}" for res_idx in res_idx_ls]
    suid = submit_expt_wrapper(tasks_ls, preset_opts_ls, name_ls, dir_root, is_expt_data=is_expt_data)
    print("finished benchmark expt with suid:", suid)

if __name__ == '__main__':
    dir_root = "chain_benchmark_expt_new"
    # dir_root = 'gaugamela_chunk_expt'
    # dir_root = "6atom_test"
    # dir_root = "chain_benchmark_expt_testhybrid"
    # dir_root = "chain_benchmark_redo_delta2"

    N = 6
    res_idx_ls = list(range(N//2))
    # res_idx_ls = [2]
    t_start_ls = [0.0]
    is_expt_data = True
    num_t_plateau_ls = 2
    n_shots = 3

    mode = 'PROCESS'  # 'RUN', 'SAVE', 'PROCESS'
    uniform_Omega_Delta_ramp = False
    start_Delta_from0=False

    if mode == 'RUN':
        tasks_ls = []
        for t_start in t_start_ls:
            for res_idx in res_idx_ls:
                expt_task_name = gen_task_wrapper(dir_root, N=N, num_t_plateau_ls=num_t_plateau_ls, n_shots=n_shots, res_idx=res_idx, Delta_local=-125, t_start=t_start, uniform_Omega_Delta_ramp=uniform_Omega_Delta_ramp, start_Delta_from0=start_Delta_from0)
                tasks_ls.append(expt_task_name)
        
        preset_opts_ls = [None] * len(tasks_ls)
        name_ls = [f"cb-residx{res_idx}-tstart{t_start}" for res_idx in res_idx_ls for _ in t_start_ls]
        suid = submit_expt_wrapper(tasks_ls, preset_opts_ls, name_ls, dir_root, is_expt_data=is_expt_data)
        # run_benchmark_expt("6atom_test", 0, "task_12345", is_expt_data = False, N = 6, num_t_plateau_ls = 10, n_shots = 100, uniform_Omega_Delta_ramp = False, start_Delta_from0=False)
    elif mode == 'SAVE':
        suid = "5bc47f763856f34135868a8e" # ENTER HERE
        save_expt_wrapper(dir_root, suid, is_expt_data=is_expt_data)
    elif mode == 'PROCESS':
        # suid = "e55a26f767160a1a5b32ab42" # t_start = 0.0
        # suid = "888e60796ee6f51cf16b7452" # t_start = 4.0
        # suid = "10577d668b84b904577b3fbe"
        suid = "8b9c54a25005a3b96bd37522" # numerical test of non uniform sampling
        

        ## results from rabi fitting to get readout errors:
        # task_name,timestamp,chi2_red,Omega,Delta,Omega_eff_theory,Omega_eff_expt,Omega_eff_expt_unc,varphi,varphi_unc,A,A_unc,B,B_unc,epsilon_r,epsilon_r_unc,epsilon_g,epsilon_g_unc
        # bc570de77fdecb2529df4298,1762299060,0.4730753810555413,15.8,0.0,15.8,16.283724973896714,0.2760836130179322,-0.4757743986452318,0.0961470723335748,0.4515735442728953,0.0488431533653649,0.4391639565557548,0.0110847669804078,0.1092624991713497,0.0500851843331188,0.0,0.0500851843331188
        # c980e4cf56fc711fdb7d1167,1762299094,0.5024030518927932,15.8,0.0,15.8,16.617231956767974,0.2736110904670405,-0.6177029553006288,0.0959352082460471,0.4437336994508923,0.0509860648318344,0.4579674767279873,0.0114794542092944,0.0982988238211203,0.0522623829917782,0.0142337772770949,0.0522623829917782
        # ae0a50948310553de6ee77d3,1762299121,2.953531767859577,15.8,0.0,15.8,15.441938500530233,0.3307224802909143,-0.0922122024161463,0.1144223253295634,0.3878122752166817,0.0433838589301282,0.4172003759859631,0.011025424392947,0.1949873487973551,0.044762922142258,0.0293881007692814,0.044762922142258
        # df1d698422a5d4a60d29d206,1762299149,0.3200822019935859,15.8,0.0,15.8,16.152639916531808,0.3057536441465704,-0.441678257812821,0.106192746629002,0.4096224022005598,0.0475217838679457,0.4230201180765748,0.0110038671153763,0.1673574797228653,0.0487791454771873,0.013397715876015,0.0487791454771873
        # 1da5a5325f74f05741f99c31,1762299176,0.4143764981359698,15.8,0.0,15.8,16.22129453576158,0.3097665549644148,-0.4750132963638885,0.107897950333058,0.3885174873836354,0.0462330109245744,0.4066511242022373,0.0107929529594607,0.2048313884141271,0.0474760901163622,0.0181336368186019,0.0474760901163622
        # 004037a68a66cc9db60e6e20,1762299205,0.6004077401003053,15.8,0.0,15.8,16.245946229057523,0.2789952296236902,-0.46526938131924667,0.09703310800816965,0.4221525626350413,0.04704568117477396,0.41225810459608947,0.010736114131286183,0.16558933276886922,0.048255157898803774,0.0,0.048255157898803774
        # epsilon_r_ls = [0.1092624991713497, 0.0982988238211203, 0.1949873487973551, 0.1673574797228653, 0.2048313884141271, 0.16558933276886922]
        # epsilon_r_unc_ls = [0.0500851843331188, 0.0522623829917782, 0.044762922142258, 0.0487791454771873, 0.0474760901163622, 0.048255157898803774]
        # epsilon_g_ls = [0.0, 0.0142337772770949, 0.0293881007692814, 0.013397715876015, 0.0181336368186019, 0.0]
        # epsilon_g_unc_ls = [0.0500851843331188, 0.0522623829917782, 0.044762922142258, 0.0487791454771873, 0.0474760901163622, 0.048255157898803774]

        # # print the epsilon_r +- unc and epsilon_g +- unc for each res_idx
        # for idx in range(len(res_idx_ls)):
        #     print(f"res_idx {res_idx_ls[idx]}: epsilon_r = {epsilon_r_ls[idx]:.2g} +- {epsilon_r_unc_ls[idx]:.1g}, epsilon_g = {epsilon_g_ls[idx]:.2g} +- {epsilon_g_unc_ls[idx]:.1g}")
        epsilon_r_ls = None
        epsilon_r_unc_ls = None
        epsilon_g_ls = None
        epsilon_g_unc_ls = None

        try:
            q_idx_total = []
            for _ in range(len(t_start_ls)):
                q_idx_total += res_idx_ls
        except:
            q_idx_total = res_idx_ls

        process_expt_wrapper(dir_root, suid, q_index=q_idx_total, show_qutip=True, open_dynamics=False, num_repeats_coherent_noise=10, num_t_qutip_points=100, epsilon_r_ls=epsilon_r_ls, epsilon_r_unc_ls = epsilon_r_unc_ls, epsilon_g_ls=epsilon_g_ls, epsilon_g_unc_ls=epsilon_g_unc_ls)  # index of the qubit to plot