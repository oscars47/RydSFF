## script to make the 4 panel fig in fig 1 of rydsff paper: (f) varrho(tilde r) with inset density of states, (g) SFF, (h) area vs volume law  
import numpy as np
from hamiltonian import drive_main, get_h_ls, get_J_arr, H_int
import qutip as qt
from tqdm import trange
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import os, json, glob
from expt_file_manager import ExptStore
from fig_styling import style_axis

get_H_indep, get_H_ramp, H_d = drive_main(neg_phi=True, ret_H_d=True)

### ---- diagonalize ---- ###
def diagonalize_aquila(Delta_local, Delta_mean, N, Omega=15.8, phi=0, H_int=None, a=10):
    h_ls = get_h_ls(N)

    Delta_global = Delta_mean - 1/2 * Delta_local

    if H_int is None:
        x = [(i*a, 0) for i in range(N)] 
        H_plateau = get_H_indep(Omega, phi, Delta_global, Delta_local, h_ls, x=x)
    else:
        H_d_ = H_d(Omega, phi, Delta_global, Delta_local, h_ls)
        H_plateau = H_d_ + H_int
    evals, evecs = H_plateau.eigenstates()
    return evals, evecs

### ---- numerical quantities ---- ###
def SFF(evals, times):
    """
    compute spectral form factor for given evals and times
    """
    Z_t = np.sum(np.exp(-1j * evals[:, None] * times), axis=0)
    SFF_vals = np.abs(Z_t)**2 / (len(evals)**2)
    return SFF_vals

def calc_level_spacing(eigenvals):
    """
    compute level spacing of the eigenvalues of H
    """
    ev = np.sort(eigenvals)
    return np.diff(ev)

def level_spacing_ratios_tilde(eigenvals):
    """
    compute tilde r ratios (min/max ratio) of consecutive level spacings
    """
    s = calc_level_spacing(eigenvals)
    return np.minimum(s[:-1], s[1:]) / np.maximum(s[:-1], s[1:])

def level_spacing_ratios(eigenvals):
    """
    compute level spacing ratios r_n = s_n / s_{n-1} of the eigenvalues of H
    """
    s = calc_level_spacing(eigenvals)
    return s[1:] / s[:-1]

def vn_halfcut_in_time(eigenvals, eigenstates, N, t_ls, log_base=np.e, eps=1e-15):
    """
    compute S(rho_A) for arbitrary cut A|B for a pure state psi
    """
    n_A = N // 2
    dA = 2**n_A
    dB = 2**(N - n_A)

    # Build V = [|E_0>, |E_1>, ...] with Fortran layout (good for BLAS and F-order reshapes)
    # Important: use order='F' to match QuTiP tensor left-to-right subsystem ordering.
    V = np.column_stack([ket.full().ravel(order='F') for ket in eigenstates]).astype(np.complex64, copy=False)
    V = np.asfortranarray(V)    # (d, d), F-contiguous

    # |psi0> = |00...0>
    psi0_vec = np.zeros(V.shape[0], dtype=np.complex64)
    psi0_vec[0] = 1.0

    # coefficients c_j = <E_j|psi0>
    c = V.conj().T @ psi0_vec     # (d,)

    E = np.asarray(eigenvals, dtype=np.float32)  # (d,)

    entropies = np.empty(len(t_ls), dtype=np.float32)

    for idx, t in enumerate(np.asarray(t_ls, dtype=np.float32)):
        # phase-weighted coefficients
        phases = np.exp(-1j * E * t, dtype=np.complex64)   # (d,)
        a = c * phases                                     # (d,)

        # |psi(t)⟩ = V @ a
        psi_t = V @ a                                      # (d,)

        # Reshape to (dA, dB) matching the chosen bipartition (A = first half)
        Psi_mat = psi_t.reshape((dA, dB), order='F')

        # Schmidt coeffs via SVD; s are singular values; probabilities p = s^2
        s = np.linalg.svd(Psi_mat, compute_uv=False)
        p = (s * s).real
        # stable entropy
        p = p[p > eps]
        S = -np.dot(p, np.log(p))
        if log_base == 2:
            S = S / np.log(2.0)
        elif log_base != np.e:
            S = S / np.log(log_base)
        entropies[idx] = S

    return entropies

def vn_entropy_eigenstate_subsys(eigenstate, N, n_A_list=None, log_base=np.e, eps=1e-15):
    """
    compute von Neumann entropy for multiple subsystem cuts of an eigenstate
    
    Parameters:
    - eigenstate: QuTiP ket representing the eigenstate
    - N: total number of qubits
    - n_A_list: list of subsystem sizes to compute entropy for (default: all sizes 1 to N-1)
    - log_base: base for logarithm (default: natural log)
    - eps: threshold for numerical stability
    
    Returns:
    - dict mapping subsystem size to entropy value
    """
    if n_A_list is None:
        n_A_list = list(range(1, N))
    
    # Convert eigenstate to numpy array with proper ordering
    psi_vec = eigenstate.full().ravel(order='F').astype(np.complex64)
    
    entropies = {}
    
    for n_A in n_A_list:
        dA = 2**n_A
        dB = 2**(N - n_A)
        
        # Reshape to (dA, dB) for the bipartition
        Psi_mat = psi_vec.reshape((dA, dB), order='F')
        
        # Schmidt coeffs via SVD; s are singular values; probabilities p = s^2
        s = np.linalg.svd(Psi_mat, compute_uv=False)
        p = (s * s).real
        
        # Stable entropy calculation
        p = p[p > eps]
        S = -np.dot(p, np.log(p))
        
        # Convert to desired log base
        if log_base == 2:
            S = S / np.log(2.0)
        elif log_base != np.e:
            S = S / np.log(log_base)
            
        entropies[n_A] = S
    
    return entropies

def all_quantites_one_time(N, t_ls, H_int, Delta_local, Delta_mean, Omega=15.8, phi=0, a = 10, N_excited=10):
    evals, evecs = diagonalize_aquila(Delta_local, Delta_mean, N, Omega=Omega, phi=phi, H_int=H_int, a=a)

    sff_vals = SFF(evals, t_ls)
    ls_ratios_tilde = level_spacing_ratios_tilde(evals)
    
    vn_subsys_all = {}
    for n in range(1, N_excited+1):
        vn_subsys_all[n] = vn_entropy_eigenstate_subsys(evecs[n], N, n_A_list=list(range(1, N)))

    vn_time = vn_halfcut_in_time(evals, evecs, N, t_ls)
    
    return {
        'evals': evals,
        'sff_vals': sff_vals,
        'ls_ratios_tilde': ls_ratios_tilde,
        'vn_subsys_all': vn_subsys_all,
        'vn_time': vn_time
    }

def repeat_quantities(N, t_ls, H_int, Delta_local, Delta_mean, n_repeats, dir_root, **kwargs):
    
    manager = ExptStore(dir_root)
    payload = {
        "N": N,
        "t_ls": t_ls.tolist() if isinstance(t_ls, np.ndarray) else t_ls,
        "Delta_local": Delta_local,
        "Delta_mean": Delta_mean,
    }
    uid, added = manager.add(payload, timestamp=0)
    
    quantities_dir = os.path.join(dir_root, "data")
    os.makedirs(quantities_dir, exist_ok=True)
    
    filename = os.path.join(quantities_dir, f"fig1_{n_repeats}_{uid}.json")
    
    # Search for any existing files with the same UID (potentially different repeat counts)
    existing_data = None
    
    # Look for files matching pattern: fig1_*_{uid}.json
    pattern = os.path.join(quantities_dir, f"fig1_*_{uid}.json")
    matching_files = glob.glob(pattern)
    
    if matching_files:
        # Find the file with the most repeats
        best_file = None
        best_repeats = 0
        
        for file_path in matching_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                file_repeats = len(data.get('results', []))
                if file_repeats > best_repeats:
                    best_repeats = file_repeats
                    best_file = file_path
                    existing_data = data
            except (json.JSONDecodeError, KeyError):
                continue
        
        if best_file and existing_data:
            if best_repeats >= n_repeats:
                print(f"Found existing file {os.path.basename(best_file)} with {best_repeats} repeats (>= {n_repeats} requested). Loading...")
                return existing_data['results'][:n_repeats]
            else:
                print(f"Found existing file {os.path.basename(best_file)} with {best_repeats} repeats (< {n_repeats} requested). Computing additional repeats...")
                all_results = existing_data['results']
                remaining_repeats = n_repeats - best_repeats
        else:
            print(f"Found matching files but couldn't read data. Starting fresh...")
            all_results = []
            remaining_repeats = n_repeats
    else:
        print(f"No existing file found for UID {uid}. Computing {n_repeats} repeats...")
        all_results = []
        remaining_repeats = n_repeats
    
    # Compute remaining repeats
    for i in trange(remaining_repeats, desc="Sampling hamiltonians..."):
        result = all_quantites_one_time(N, t_ls, H_int, Delta_local, Delta_mean, **kwargs)
        
        # Helper function to convert numpy types to JSON serializable types
        def numpy_to_python(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.complex64, np.complex128)):
                return complex(obj)
            elif isinstance(obj, dict):
                return {k: numpy_to_python(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [numpy_to_python(item) for item in obj]
            else:
                return obj
        
        # Convert result to JSON serializable format
        json_result = numpy_to_python(result)
        all_results.append(json_result)
        
        # save incrementally every 5 iterations to avoid data loss
        if (i + 1) % 5 == 0 or i == remaining_repeats - 1:
            # Create filename with current total repeats
            current_total = len(all_results)
            current_filename = os.path.join(quantities_dir, f"fig1_{current_total}_{uid}.json")
            
            data_to_save = {
                'metadata': payload,
                'uid': uid,
                'total_repeats': current_total,
                'results': all_results
            }
            
            # Save with current total repeats in filename
            with open(current_filename, 'w') as f:
                json.dump(data_to_save, f, indent=2)
            
            # Delete old file if it exists and is different from current
            if current_filename != filename and os.path.exists(filename):
                os.remove(filename)
            
            # Update filename reference for next iteration
            filename = current_filename
    
    # final save
    data_to_save = {
        'metadata': payload,
        'uid': uid,
        'total_repeats': len(all_results),
        'results': all_results
    }
    with open(filename, 'w') as f:
        json.dump(data_to_save, f, indent=2)
    
    print(f"Saved {len(all_results)} repeats to {filename}")
    return all_results

def make_fig1_fghi(N, t_ls, Delta_local_ls, Delta_mean_ls, n_repeats, dir_root, fontsize=40,alpha=0.8, J=5.42,**kwargs):
    a = 10
    J_arr = get_J_arr([(i*a,0) for i in range(N)], N)
    H_int_ = H_int(J_arr, N)

    # set font
    mpl.rcParams.update({'font.size': fontsize})
    plt.rc('text', usetex=True)
    mpl.rc('text.latex', preamble=r"""
    \usepackage{amsmath}
    \usepackage{newtxtext,newtxmath}
    """)

    # colors = ['black','red']
    colors = ['black','red']
    colors2 = ['gray','lightcoral' ]
    markers = ['s', 'o',]
    linestyles = [ '-', '--',]

    fig, axs = plt.subplots(2, 2, figsize=(22, 22))

    # inset on (a): eigenvalue histogram outline
    axins = inset_axes(axs[0, 0], width="50%", height="50%", loc='upper right')

    for i, Delta_mean in enumerate(Delta_mean_ls):
        for j, Delta_local in enumerate(Delta_local_ls):
            print(f"Calculating for Delta_mean={Delta_mean}, Delta_local={Delta_local}...")

            all_results = repeat_quantities(N, t_ls, H_int_, Delta_local, Delta_mean, n_repeats, dir_root=dir_root, **kwargs)

            sff_vals_all = np.array([res['sff_vals'] for res in all_results])
            sff_vals = np.mean(sff_vals_all, axis=0)
            ls_ratios_tilde_all = np.array([res['ls_ratios_tilde'] for res in all_results])
            vn_time_all = np.array([res['vn_time'] for res in all_results])
            vn_time = np.mean(vn_time_all, axis=0)
            eigevals_all = np.array([res['evals'] for res in all_results])

            # Get the number of excited states from the first result
            N_excited = len(all_results[0]['vn_subsys_all'].keys())
            vn_subsys_all = {}
            for n in range(1, N_excited+1):
                # Handle both string and integer keys (JSON converts keys to strings)
                key = str(n) if str(n) in all_results[0]['vn_subsys_all'] else n
                vn_subsys_all[n] = np.array([res['vn_subsys_all'][key] for res in all_results])

            # --- panel (a): level spacing ratio density with density of states inset ---
            tilde_counts, tilde_edges = np.histogram(ls_ratios_tilde_all, bins=200, density=True)
            tilde_centers = (tilde_edges[:-1] + tilde_edges[1:]) / 2
            axs[0, 0].plot(
                tilde_centers, tilde_counts,
                linestyle=linestyles[j], alpha=alpha,
                color=colors[j],
                linewidth=6
            )

            # density of states in the inset
            ev = eigevals_all.flatten() / J
            if j == 0:
                ev_min_0 = np.min(ev)*1.2
                ev_max_0 = np.max(ev)*1.5
            else:
                ev = ev[ev >= ev_min_0]
                ev = ev[ev <= ev_max_0]
            ev_counts, ev_edges = np.histogram(ev, bins=100, density=True)
            ev_centers = (ev_edges[:-1] + ev_edges[1:]) / 2
            axins.plot(ev_centers, ev_counts, linestyle=linestyles[j], color=colors[j], alpha=alpha)

            # --- panel (b): spectral form factor ---
            axs[0, 1].plot(
                t_ls, sff_vals, 
                linestyle=linestyles[j], alpha=alpha, color=colors[j], linewidth=6
            )

            # (c) entropy by subsystem size
            for n_excited in range(1, N_excited+1):  # Limit to available excited states
                if n_excited in vn_subsys_all:
                    vn_subsys_n_all = vn_subsys_all[n_excited]
                    alpha_scaled = alpha * (N_excited - n_excited) / N_excited
                    
                    # Convert array of dicts to proper numerical array for averaging
                    # Each element in vn_subsys_n_all is a dict {subsys_size: entropy}
                    # We need to convert this to a 2D array for proper averaging
                    subsys_sizes = list(vn_subsys_n_all[0].keys())  # Get subsystem sizes from first dict
                    vn_subsys_matrix = np.array([[sample_dict[size] for size in subsys_sizes] 
                                                for sample_dict in vn_subsys_n_all])
                    vn_subsys_n = np.mean(vn_subsys_matrix, axis=0)

                    subsys_sizes = [int(size) for size in subsys_sizes]
                    
                    # Plot mean
                    axs[1, 0].scatter(
                        subsys_sizes, vn_subsys_n,
                        marker=markers[j], alpha=alpha_scaled, color=colors[j], s=200,
                        label=fr'$\Delta_{{\mathrm{{local}}}}={Delta_local/J:.3g} J$' if n_excited==1 else None
                    )
                
            # (d) entropy in time
            # vn_time_all has shape (n_repeats, n_time_points)
            # Plot individual realizations (each row is one realization)
            for k in range(vn_time_all.shape[0]):  # Plot all individual realizations
                axs[1, 1].plot(
                    t_ls, vn_time_all[k],
                    linestyle=linestyles[j], alpha=alpha*0.2, color=colors2[j], linewidth=2
                )
            # Plot the mean across all realizations
            axs[1, 1].plot(
                t_ls, vn_time,
                linestyle=linestyles[j], alpha=alpha, color=colors[j], linewidth=6, label=fr'$\Delta_{{\mathrm{{local}}}}={Delta_local/J:.3g} J$'
            )
           
    axs[0, 0].set_xlabel(r'$r_n$', fontsize=1.3*fontsize)
    axs[0, 0].set_ylabel(r'$\varrho (r_n)$', fontsize=1.3*fontsize)
    axs[1, 0].set_xlabel(r'$N_A$', fontsize=1.3*fontsize)
    # axs[1, 0].set_ylabel(r'$S_{1, \, A}^{(1-10)}$', fontsize=1.3*fontsize)
    axs[1, 0].set_ylabel(r'$S_{1, \, A}$', fontsize=1.3*fontsize)
    axs[0, 1].set_xlabel(r'$t_{\mathrm{evolve}} \, (\mu \mathrm{s})$', fontsize=1.3*fontsize)
    axs[0, 1].set_ylabel(r'$\mathrm{SFF}$', fontsize=1.3*fontsize)
    axs[1, 1].set_xlabel(r'$t_{\mathrm{evolve}} \, (\mu \mathrm{s})$', fontsize=1.3*fontsize)
    axs[1, 1].set_ylabel(r'$S_{1, \, A}(t_{\mathrm{evolve}})$', fontsize=1.3*fontsize)

    axins.set_xlabel(r'$\tilde{r_n}$', fontsize=fontsize*0.99)
    axins.set_ylabel(r'$\varrho (\tilde{r_n})$', fontsize=fontsize*0.99)
    
    axs[0, 1].set_xscale('log')
    axs[0, 1].set_ylim(1e-10, 1.1)
    axs[0, 1].set_yscale('log')
    axs[1, 1].set_xscale('log')

    # add legend to [1,0] and [1,1]
    axs[1, 0].legend(fontsize=fontsize*0.9)
    axs[1, 1].legend(fontsize=fontsize*0.9)


    # panel labels
    labels = [r'$\sf{\textbf{f}}$', 
                  r'$\sf{\textbf{g}}$', 
                  r'$\sf{\textbf{h}}$', 
                  r'$\sf{\textbf{i}}$']
    
     # For each axes (including inset)
    for ax_ in axs.flat:
        style_axis(ax_, fontsize=fontsize)

    style_axis(axins, fontsize=fontsize)

    for ax, lab in zip(axs.flat, labels):
        ax.text(
            # -0.15, 1.05, lab,
            -0.2, 1.05, lab,
            transform=ax.transAxes,
            fontsize=fontsize*1.3,
            weight='bold'
        )

    axs[1,1].set_ylim(-0.1, 2.2)
    plt.tight_layout()
    manager = ExptStore(dir_root)
    payload = {
        "N": N,
        "t_ls": t_ls.tolist() if isinstance(t_ls, np.ndarray) else t_ls,
        "Delta_local_ls": Delta_local_ls,
        "Delta_mean_ls": Delta_mean_ls,
        "n_repeats": n_repeats,
    }
    uid, added = manager.add(payload, timestamp=0)
    results_dir = os.path.join(dir_root, "results")
    os.makedirs(results_dir, exist_ok=True)
    out = os.path.join(results_dir, f"fig1_fghi_{uid}.pdf")
    plt.savefig(out)
    plt.close(fig)


if __name__ == "__main__":
    N = 12
    J = 5.42
    t_ls = np.logspace(-2, 3, 500) 
    Delta_local_ls = [-0.5*J, 10*J] 
    Delta_mean_ls = [0.5*J] 
    n_repeats = 100
    make_fig1_fghi(N, t_ls, Delta_local_ls, Delta_mean_ls, n_repeats, dir_root='fig1', fontsize=40, alpha=0.8, J=5.42)

    
    





    