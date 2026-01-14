from expt_file_manager import ExptStore
from fig_styling import style_axis
from fig1_fghi import diagonalize_aquila, repeat_quantities_general
from hamiltonian import drive_main, get_h_ls, get_J_arr, H_int
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
from tqdm import tqdm


def collect_eval_only(N, H_int, Delta_local, Delta_mean, Omega, a=0, threshold=0, NN_only=False):
    h_ls = get_h_ls(N, threshold=threshold)
    evals, evecs = diagonalize_aquila(Delta_local, Delta_mean, N, Omega=Omega, phi=0, H_int=H_int, a=a, threshold=threshold, h_ls=h_ls, NN_only=NN_only)
    return {
        'evals': evals
    } 

def plot_results(N, Delta_local_ls, Delta_mean_ls, n_repeats, Omega, dir_root, fontsize=15, J = 5.42):

    mpl.rcParams.update({'font.size': fontsize})
    # use text.usetex = True
    os.environ["PATH"] += os.pathsep + "/Library/TeX/texbin/"
    plt.rc('text', usetex=True)
    mpl.rc('text.latex', preamble=r'''
    \usepackage{amsmath}
    \usepackage{newtxtext,newtxmath}
    ''')

    a = 10
    J_arr = get_J_arr([(i*a,0) for i in range(N)], N)
    H_int_ = H_int(J_arr, N)

    fig, ax = plt.subplots(figsize=(6,6))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(Delta_local_ls)))

    total_combinations = len(Delta_local_ls) * len(Delta_mean_ls)
    with tqdm(total=total_combinations, desc="Overall Progress") as pbar:
        for j, Delta_local in enumerate(Delta_local_ls):
            median_eigenvals = []
            for Delta_mean in Delta_mean_ls:
                pbar.set_postfix({"Δ_local": f"{Delta_local/J:.2f}J", "Δ_mean": f"{Delta_mean/J:.2f}J"})
                all_results = repeat_quantities_general(N, None, H_int_, Delta_local, Delta_mean, n_repeats, dir_root=dir_root, func=collect_eval_only, fileprefix='apx_eval', Omega=Omega)
                eigenval_samples = [res['evals'] for res in all_results]
                median_eigenvals.append(np.median(eigenval_samples))
                pbar.update(1)
            
            median_eigenvals = np.array(median_eigenvals)
            ax.plot(Delta_mean_ls/J, median_eigenvals/J, label=rf'$\Delta_{{\mathrm{{local}}}}={Delta_local/J:.3g}/J$', color=colors[j])

    ax.set_box_aspect(1)
    ax.set_xlabel(r'$\langle \Delta \rangle / J $')
    ax.set_ylabel(r'$\mathrm{Median}\,  E_n / J$')
    ax.legend(fontsize=fontsize*0.9)
    style_axis(ax, fontsize=fontsize)
    plt.tight_layout()
    manager = ExptStore(dir_root)
    payload = {
        'N': N,
        'Delta_local_ls': Delta_local_ls,
        'Delta_mean_ls': Delta_mean_ls.tolist() if isinstance(Delta_mean_ls, np.ndarray) else Delta_mean_ls,
        'n_repeats': n_repeats,
        'Omega': Omega,
    }
    uid, added = manager.add(payload, timestamp=0)
    results_dir = os.path.join(dir_root, "results")
    os.makedirs(results_dir, exist_ok=True)
    out = os.path.join(results_dir, f"apx_eigenval_{uid}.pdf")
    plt.savefig(out)
    plt.close(fig)
 
if __name__ == "__main__":
    dir_root = "appendix"
    N = 8
    Omega = 15.8
    J = 5.42
    Delta_local_ls = [-0.5*J, -3*J, -5*J, -10*J]  
    Delta_mean_ls = np.linspace(-10*J, 10*J, 200) 
    n_repeats = 1000
    plot_results(N, Delta_local_ls, Delta_mean_ls, n_repeats, Omega, dir_root)            
