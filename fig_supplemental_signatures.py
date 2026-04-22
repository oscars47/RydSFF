"""
4 main signatures of chaos for the supplemental: level spacing ratios, SFF, area vs volume law of eigenstates in middle of spectrum, and disorder dependent time growth of entanglement
"""


import numpy as np
from QuEraToolbox.hamiltonian import drive_main, get_h_ls, get_J_arr, H_int
import qutip as qt
from tqdm import trange
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import os, json, glob
from QuEraToolbox.expt_file_manager import ExptStore
from fig_styling import style_axis

get_H_indep, get_H_ramp, H_d = drive_main(neg_phi=True, ret_H_d=True)

### ---- diagonalize ---- ###
def diagonalize_aquila(Delta_local, Delta_mean, N, Omega=15.8, phi=0, H_int=None, a=10, threshold=0, h_ls=None, NN_only=False):
    # print("OMEGA", Omega)
    if h_ls is None:
        h_ls = get_h_ls(N, threshold=threshold)
    else:
        assert len(h_ls) == N, f'Length of h_ls ({len(h_ls)}) does not match N ({N})'

    Delta_global = Delta_mean - 1/2 * Delta_local

    if NN_only:
        # only i, i+1 couplings
        print("NN only couplings")
        J_arr = np.zeros((N, N))
        for i in range(N-1):
            J_arr[i, i+1] = 5.42 
            J_arr[i+1, i] = J_arr[i, i+1]

    if H_int is None:
        x = [(i*a, 0) for i in range(N)] 
        H_plateau = get_H_indep(Omega, phi, Delta_global, Delta_local, h_ls, x=x)
    else:
        H_d_ = H_d(Omega, phi, Delta_global, Delta_local, h_ls)
        H_plateau = H_d_ + H_int

    evals, evecs = H_plateau.eigenstates()
    assert len(evals) == 2**N, f'Expected {2**N} eigenvalues, got {len(evals)}'
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
    # use order='F' to match QuTiP tensor left-to-right subsystem ordering.
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

def renyi2_halfcut_in_time(eigenvals, eigenstates, N, t_ls, log_base=np.e, eps=1e-15):
    """
    compute S_2(rho_A) = -log(Tr(rho_A^2)) for cut A|B for a pure state psi
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
        # purity Tr(rho_A^2) = sum_i p_i^2
        purity = np.dot(p, p)
        # stable Renyi-2 entropy
        S = -np.log(max(purity, eps))
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
    
    psi_vec = eigenstate.full().ravel(order='F')
    
    entropies = {}
    
    for n_A in n_A_list:
        dA = 2**n_A
        dB = 2**(N - n_A)
        
        # Reshape to (dA, dB) for the bipartition
        Psi_mat = psi_vec.reshape((dA, dB), order='F')
        
        # Schmidt coeffs via SVD, probab p = s^2
        s = np.linalg.svd(Psi_mat, compute_uv=False)
        p = (s * s).real
        
        p = p[p > eps]
        S = -np.dot(p, np.log(p))
        
        if log_base == 2:
            S = S / np.log(2.0)
        elif log_base != np.e:
            S = S / np.log(log_base)
            
        entropies[n_A] = S
    
    return entropies

def page_value_eqbi(n):
    """Page value (natural log) for equal bipartition https://arxiv.org/pdf/gr-qc/9305007
    """
    n_dim = int(n)
    if n_dim <= 1:
        return 0.0
    # Equal bipartition: m = n
    harmonic_sum = np.sum([1.0 / k for k in range(n_dim + 1, n_dim * n_dim + 1)])
    return float(harmonic_sum - (n_dim - 1) / (2 * n_dim))

def vn_entropy_halfcut_all_eigenstates(eigenstates, N, log_base=np.e, eps=1e-15):
    """Half-cut VN entropy for all eigenstates.

    Returns array S_i = S_A(|E_i>) for equal bipartition A|B (A = first N//2 qubits).
    """
    n_A = N // 2
    dA = 2 ** n_A
    dB = 2 ** (N - n_A)

    entropies = np.empty(len(eigenstates), dtype=np.float32)
    log_denom = 1.0
    if log_base == 2:
        log_denom = np.log(2.0)
    elif log_base != np.e:
        log_denom = np.log(log_base)

    for idx, ket in enumerate(eigenstates):
        psi_vec = ket.full().ravel(order='F')
        Psi_mat = psi_vec.reshape((dA, dB), order='F')
        s = np.linalg.svd(Psi_mat, compute_uv=False)
        p = (s * s).real
        p = p[p > eps]
        S = -np.dot(p, np.log(p))
        if log_denom != 1.0:
            S = S / log_denom
        entropies[idx] = S
    return entropies

def evals_only(N, H_int, Delta_local, Delta_mean, t_ls=None, Omega=15.8, phi=0, a=10, threshold=0, NN_only=False):
    h_ls = get_h_ls(N, threshold=threshold)
    evals, evecs = diagonalize_aquila(
        Delta_local, Delta_mean, N,
        Omega=Omega, phi=phi, H_int=H_int, a=a,
        threshold=threshold, h_ls=h_ls, NN_only=NN_only
    )
    return {'evals': evals}

def all_quantites_one_time(N, t_ls, H_int, Delta_local, Delta_mean,  Omega=15.8, phi=0, a = 10, threshold=0, NN_only=False, middle_h_1=False, add_renyi2=False):
    h_ls = get_h_ls(N, threshold=threshold)
    if middle_h_1:
        h_ls[N//2] = 1.0
    evals, evecs = diagonalize_aquila(
        Delta_local, Delta_mean, N,
        Omega=Omega, phi=phi, H_int=H_int, a=a,
        threshold=threshold, h_ls=h_ls, NN_only=NN_only, 
    )

    sff_vals = SFF(evals, t_ls)
    ls_ratios_tilde = level_spacing_ratios_tilde(evals)

    vn_halfcut_eigs = vn_entropy_halfcut_all_eigenstates(evecs, N, log_base=np.e)

    evals_np = np.asarray(evals, dtype=np.float64)
    closest_indices = np.argsort(np.abs(evals_np - 0.0))[:10]
    vn_subsys_closest = []
    for idx in closest_indices:
        vn_subsys_closest.append(
            vn_entropy_eigenstate_subsys(evecs[int(idx)], N, n_A_list=list(range(1, N)))
        )

    vn_time = vn_halfcut_in_time(evals, evecs, N, t_ls)
    
    results_dict = {
        'evals': evals,
        'sff_vals': sff_vals,
        'ls_ratios_tilde': ls_ratios_tilde,
        'vn_halfcut_eigs': vn_halfcut_eigs,
        'closest_indices': closest_indices,
        'vn_subsys_closest': vn_subsys_closest,
        'vn_time': vn_time,
    }

    if add_renyi2:
        renyi2_time = renyi2_halfcut_in_time(evals, evecs, N, t_ls)
        results_dict['renyi2_time'] = renyi2_time

    return results_dict

def repeat_quantities_general(N, t_ls, H_int, Delta_local, Delta_mean, n_repeats, dir_root, fileprefix, func, **kwargs):
    
    manager = ExptStore(dir_root)
    payload = {
        "N": N,
        "Delta_local": Delta_local,
        "Delta_mean": Delta_mean,
    }
    if t_ls is not None:
        payload["t_ls"] = t_ls.tolist() if isinstance(t_ls, np.ndarray) else t_ls
    if 'threshold' in kwargs:
        threshold = kwargs['threshold']
        if threshold > 0:
            payload['threshold'] = threshold
    if 'Omega' in kwargs:
        if kwargs['Omega'] != 15.8:
            payload['Omega'] = kwargs['Omega']
    if 'NN_only' in kwargs:
        if kwargs['NN_only']:
            payload['NN_only'] = kwargs['NN_only']
    if 'middle_h_1' in kwargs:
        if kwargs['middle_h_1']:
            payload['middle_h_1'] = kwargs['middle_h_1']
    if 'add_renyi2' in kwargs:
        if kwargs['add_renyi2']:
            payload['add_renyi2'] = kwargs['add_renyi2']

    uid, added = manager.add(payload, timestamp=0)
    
    quantities_dir = os.path.join(dir_root, "data")
    os.makedirs(quantities_dir, exist_ok=True)
    
    filename = os.path.join(quantities_dir, f"{fileprefix}_{n_repeats}_{uid}.json")
    
    existing_data = None
    
    # look for files matching pattern: fig1_*_{uid}.json
    pattern = os.path.join(quantities_dir, f"{fileprefix}_*_{uid}.json")
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
    
    for i in trange(remaining_repeats, desc="Sampling hamiltonians..."):
        if t_ls is not None:
            result = func(N, t_ls, H_int, Delta_local, Delta_mean, **kwargs)
        else:
            result = func(N, H_int, Delta_local, Delta_mean, **kwargs)
        
        # convert numpy types to JSON serializable types
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
        
        # convert result to JSON serializable format
        json_result = numpy_to_python(result)
        all_results.append(json_result)
        
        # save incrementally every 5 iterations to avoid data loss
        if (i + 1) % 5 == 0 or i == remaining_repeats - 1:
            current_total = len(all_results)
            current_filename = os.path.join(quantities_dir, f"{fileprefix}_{current_total}_{uid}.json")
            
            data_to_save = {
                'metadata': payload,
                'uid': uid,
                'total_repeats': current_total,
                'results': all_results
            }
            
            with open(current_filename, 'w') as f:
                json.dump(data_to_save, f, indent=2)
            
            if current_filename != filename and os.path.exists(filename):
                os.remove(filename)
            
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

def _extract_vn_spectrum_points(all_results, J):
    all_energies = []
    all_entropies = []

    for result in all_results:
        if 'evals' not in result or 'vn_halfcut_eigs' not in result:
            continue
        evals = np.asarray(result['evals'], dtype=np.float64)
        vn_half = np.asarray(result['vn_halfcut_eigs'], dtype=np.float64)

        n = min(len(evals), len(vn_half))
        if n == 0:
            continue
        all_energies.extend((evals[:n] / J).tolist())
        all_entropies.extend(vn_half[:n].tolist())

    return np.asarray(all_energies, dtype=np.float64), np.asarray(all_entropies, dtype=np.float64)


def plot_vn_entropy_vs_energy(
    all_results,
    N,
    J,
    fontsize=40,
    dir_root='fig1',
    Delta_mean=None,
    Delta_local=None,
    out_name=None,
    energies=None,
    entropies=None,
    e_edges=None,
    s_edges=None,
    c_vmin=None,
    c_vmax=None,
    bins_e=120,
    bins_s=120,
):
    """
    Create separate figure plotting VN entropy vs eigenenergy with Page value comparison
    """

    mpl.rcParams.update({'font.size': fontsize})
    plt.rc('text', usetex=True)
    mpl.rc('text.latex', preamble=r"""
    \usepackage{amsmath}
    \usepackage{newtxtext,newtxmath}
    """)

    fig = plt.figure(figsize=(10, 10), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.06])
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])

    if energies is None or entropies is None:
        if all_results is None:
            raise ValueError("Provide either all_results or (energies, entropies).")
        all_energies, all_entropies = _extract_vn_spectrum_points(all_results, J)
    else:
        all_energies = np.asarray(energies, dtype=np.float64)
        all_entropies = np.asarray(entropies, dtype=np.float64)

    if all_energies.size == 0 or all_entropies.size == 0:
        print("No VN spectrum points found; skipping VN entropy vs energy plot.")
        plt.close(fig)
        return

    if e_edges is None or s_edges is None:
        H, e_edges, s_edges = np.histogram2d(all_energies, all_entropies, bins=[bins_e, bins_s])
    else:
        H, _, _ = np.histogram2d(all_energies, all_entropies, bins=[e_edges, s_edges])

    e_idx = np.clip(np.digitize(all_energies, e_edges) - 1, 0, H.shape[0] - 1)
    s_idx = np.clip(np.digitize(all_entropies, s_edges) - 1, 0, H.shape[1] - 1)
    counts = H[e_idx, s_idx]
    c = np.log10(counts + 1.0)

    norm = None
    if (c_vmin is not None) or (c_vmax is not None):
        norm = mpl.colors.Normalize(vmin=c_vmin, vmax=c_vmax, clip=True)

    sc = ax.scatter(all_energies, all_entropies, c=c, s=6, cmap='viridis', norm=norm, linewidths=0)
    cbar = fig.colorbar(sc, cax=cax)
    cbar.set_label(r'$\log_{10}(\mathrm{count}+1)$')

    dim_half = 2 ** (N // 2)
    page_val = page_value_eqbi(dim_half)
    ax.axhline(y=page_val, color='black', linestyle=':', linewidth=3, label='$\mathrm{Page}$')

    S_max = np.log(dim_half)
    ax.axhline(y=S_max, color='red', linestyle='--', linewidth=3, label='$\log{d_A}$')

    ax.set_xlabel(r'$E_n / J$', fontsize=1.3 * fontsize)
    ax.set_ylabel(r'$S_{1,A}(E_n)$', fontsize=1.3 * fontsize)
    ax.legend(fontsize=0.9 * fontsize, loc='upper right')
    ax.set_box_aspect(1)
    style_axis(ax, fontsize=fontsize)

    results_dir = os.path.join(dir_root, "results")
    os.makedirs(results_dir, exist_ok=True)

    if out_name is None:
        if (Delta_mean is not None) and (Delta_local is not None):
            out_name = f"vn_entropy_vs_energy_Dm{Delta_mean}_Dl{Delta_local}.png"
        else:
            out_name = "vn_entropy_vs_energy.png"

    out = os.path.join(results_dir, out_name)
    plt.savefig(out)
    plt.close(fig)
    print(f"Saved VN entropy vs energy plot to {out}")


def fig_signatures(N, t_ls, Delta_local_ls, Delta_mean_ls, n_repeats, dir_root, fontsize=40, alpha=0.8, J=5.42, bins_e = 120, bins_s = 120, **kwargs):
    a = 10
    J_arr = get_J_arr([(i*a,0) for i in range(N)], N)
    H_int_ = H_int(J_arr, N)

    mpl.rcParams.update({'font.size': fontsize})
    plt.rc('text', usetex=True)
    mpl.rc('text.latex', preamble=r"""
    \usepackage{amsmath}
    \usepackage{newtxtext,newtxmath}
    """)

    colors = ['red', 'black']
    colors2 = ['lightcoral', 'gray' ]
    markers = ['s', 'o']
    linestyles = ['-', '--',]

    fig, axs = plt.subplots(2, 2, figsize=(24, 24))


    # standardize color normalization
    vn_datasets = []

    for i, Delta_mean in enumerate(Delta_mean_ls):
        for j, Delta_local in enumerate(Delta_local_ls):
            print(f"Calculating for Delta_mean={Delta_mean}, Delta_local={Delta_local}...")

            all_results = repeat_quantities_general(N, t_ls, H_int_, Delta_local, Delta_mean, n_repeats, func = all_quantites_one_time, fileprefix='fig1',dir_root=dir_root, **kwargs)
            
            E_pts, S_pts = _extract_vn_spectrum_points(all_results, J)
            if E_pts.size and S_pts.size:
                vn_datasets.append({
                    'Delta_mean': Delta_mean,
                    'Delta_local': Delta_local,
                    'E': E_pts,
                    'S': S_pts,
                })

            sff_vals_all = np.array([res['sff_vals'] for res in all_results])
            sff_vals = np.mean(sff_vals_all, axis=0)
            ls_ratios_tilde_all = np.array([res['ls_ratios_tilde'] for res in all_results])
            vn_time_all = np.array([res['vn_time'] for res in all_results])
            vn_time = np.mean(vn_time_all, axis=0)

            # --- panel (a): level spacing ratio density with density of states inset ---
            tilde_counts, tilde_edges = np.histogram(ls_ratios_tilde_all, bins=200, density=True)
            tilde_centers = (tilde_edges[:-1] + tilde_edges[1:]) / 2
            axs[0, 0].plot(
                tilde_centers, tilde_counts,
                linestyle=linestyles[j], alpha=alpha,
                color=colors[j],
                linewidth=6,
                label =fr'$\Delta_{{\mathrm{{local}}}}={Delta_local/J:.3g} J$'
            )

            # --- panel (b): spectral form factor ---
            axs[0, 1].plot(
                t_ls, sff_vals, 
                linestyle=linestyles[j], alpha=alpha, color=colors[j], linewidth=6
            )

            max_ranks = 0
            for res in all_results:
                if 'vn_subsys_closest' in res and isinstance(res['vn_subsys_closest'], list):
                    max_ranks = max(max_ranks, len(res['vn_subsys_closest']))

            for rank in range(min(10, max_ranks)):
                alpha_scaled = np.exp(-4.0 * rank / 9.0) if max_ranks > 1 else 1.0

                subsys_sizes = list(range(1, N))
                mean_entropies = []
                for n_A in subsys_sizes:
                    vals = []
                    for res in all_results:
                        if 'vn_subsys_closest' not in res:
                            continue
                        if rank >= len(res['vn_subsys_closest']):
                            continue
                        vn_dict = res['vn_subsys_closest'][rank]
                        # JSON may stringify keys
                        if n_A in vn_dict:
                            vals.append(vn_dict[n_A])
                        elif str(n_A) in vn_dict:
                            vals.append(vn_dict[str(n_A)])
                    mean_entropies.append(np.mean(vals) if len(vals) else np.nan)

                mask = ~np.isnan(mean_entropies)
                if np.any(mask):
                    axs[1, 0].scatter(
                        np.asarray(subsys_sizes)[mask],
                        np.asarray(mean_entropies)[mask],
                        marker=markers[j], alpha=alpha_scaled, color=colors[j], s=200
                    )
                
            # (d) entropy in time
            for k in range(vn_time_all.shape[0]):  
                axs[1, 1].plot(
                    t_ls, vn_time_all[k],
                    linestyle=linestyles[j], alpha=0.03, color=colors2[j], linewidth=2
                )
           
            axs[1, 1].plot(
                t_ls, vn_time,
                linestyle=linestyles[j], alpha=alpha, color=colors[j], linewidth=6, label=fr'$\Delta_{{\mathrm{{local}}}}={Delta_local/J:.3g} J$'
            )

           
    axs[0, 0].set_xlabel(r'$\tilde r_n$', fontsize=1.3*fontsize)
    axs[0, 0].set_ylabel(r'$\varrho (\tilde r_n)$', fontsize=1.3*fontsize)
    axs[1, 0].set_xlabel(r'$N_A$', fontsize=1.3*fontsize)
    # axs[1, 0].set_ylabel(r'$S_{1, \, A}^{(1-10)}$', fontsize=1.3*fontsize)
    axs[1, 0].set_ylabel(r'$S_{1, \, A}(N_A)$', fontsize=1.3*fontsize)
    axs[0, 1].set_xlabel(r'$t_{\mathrm{evol}} \, (\mu \mathrm{s})$', fontsize=1.3*fontsize)
    axs[0, 1].set_ylabel(r'$\mathrm{SFF}(t_{\mathrm{evol}})$', fontsize=1.3*fontsize)
    axs[1, 1].set_xlabel(r'$t_{\mathrm{evol}} \, (\mu \mathrm{s})$', fontsize=1.3*fontsize)
    axs[1, 1].set_ylabel(r'$S_{1, \, A}(t_{\mathrm{evol}})$', fontsize=1.3*fontsize)

    axs[0, 1].set_xscale('log')
    axs[0, 1].set_ylim(1e-10, 1.1)
    axs[0, 1].set_yscale('log')
    axs[1, 1].set_xscale('log')

    # add legend to [0,0] with both line styles and markers
    legend_elements = []
    for j, Delta_local in enumerate(Delta_local_ls):
        legend_elements.append(Line2D([0], [0], 
                                    linestyle=linestyles[j], 
                                    color=colors[j], 
                                    marker=markers[j],
                                    markersize=12,
                                    linewidth=6,
                                    label=fr'$\Delta_{{\mathrm{{local}}}}={Delta_local/J:.3g} J$'))
    axs[0, 0].legend(handles=legend_elements, fontsize=fontsize*0.9, loc='lower center')
    


    # panel labels
    labels = [r'$\sf{\textbf{a}}$', 
                  r'$\sf{\textbf{b}}$', 
                  r'$\sf{\textbf{c}}$', 
                  r'$\sf{\textbf{d}}$']
    
     # For each axes (including inset)
    for ax_ in axs.flat:
        style_axis(ax_, fontsize=fontsize)


    for ax, lab in zip(axs.flat, labels):
        ax.text(
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
    extr = 'all_h'

    if 'threshold' in kwargs:
        threshold = kwargs['threshold']
        if threshold > 0:
            payload['threshold'] = threshold
    if 'Omega' in kwargs:
        if kwargs['Omega'] != 15.8:
            payload['Omega'] = kwargs['Omega']
    if 'NN_only' in kwargs:
        if kwargs['NN_only']:
            payload['NN_only'] = kwargs['NN_only']
    if 'middle_h_1' in kwargs:
        if kwargs['middle_h_1']:
            payload['middle_h_1'] = kwargs['middle_h_1']
            extr = 'middle_h_1'

    uid, added = manager.add(payload, timestamp=0)
    results_dir = os.path.join(dir_root, "results")
    os.makedirs(results_dir, exist_ok=True)
    out = os.path.join(results_dir, f"fig1_fghi_{uid}_{extr}.pdf")
    plt.savefig(out)
    plt.close(fig)

    # VN entropy vs energy plots, standardized intensity scale across conditions
    if len(vn_datasets):
        all_E = np.concatenate([d['E'] for d in vn_datasets])
        all_S = np.concatenate([d['S'] for d in vn_datasets])

        e_edges = np.linspace(np.min(all_E), np.max(all_E), bins_e + 1)
        s_edges = np.linspace(np.min(all_S), np.max(all_S), bins_s + 1)
        
        y_min = np.min(all_S)
        y_max = np.max(all_S)

        max_count = 0.0
        for d in vn_datasets:
            H, _, _ = np.histogram2d(d['E'], d['S'], bins=[e_edges, s_edges])
            if H.size:
                max_count = max(max_count, float(np.max(H)))

        c_vmin = 0.0
        c_vmax = float(np.log10(max_count + 1.0)) if max_count > 0 else 1.0

        n_datasets = len(vn_datasets)
        fig = plt.figure(figsize=(10 * n_datasets, 10), constrained_layout=True)
        gs = fig.add_gridspec(1, n_datasets + 1, width_ratios=[1.0] * n_datasets + [0.06])
        
        dim_half = 2 ** (N // 2)
        page_val = page_value_eqbi(dim_half)
        S_max = np.log(dim_half)
        
        labels = [r'$\sf{\textbf{a}}$', r'$\sf{\textbf{b}}$', r'$\sf{\textbf{c}}$', r'$\sf{\textbf{d}}$']
        
        for idx, d in enumerate(vn_datasets):
            ax = fig.add_subplot(gs[0, idx])
            
            all_energies = np.asarray(d['E'], dtype=np.float64)
            all_entropies = np.asarray(d['S'], dtype=np.float64)
            
            H, _, _ = np.histogram2d(all_energies, all_entropies, bins=[e_edges, s_edges])
            
            e_idx = np.clip(np.digitize(all_energies, e_edges) - 1, 0, H.shape[0] - 1)
            s_idx = np.clip(np.digitize(all_entropies, s_edges) - 1, 0, H.shape[1] - 1)
            counts = H[e_idx, s_idx]
            
            c = np.log10(counts + 1.0)
            norm = mpl.colors.Normalize(vmin=c_vmin, vmax=c_vmax, clip=True)
            
            sc = ax.scatter(all_energies, all_entropies, c=c, s=6, cmap='viridis', norm=norm, linewidths=0)
            

            if idx == n_datasets - 1:
                ax.axhline(y=page_val, color='black', linestyle=':', linewidth=3, label='$\mathrm{Page}$')
                ax.legend(fontsize=0.9 * fontsize, loc='upper right')
            else:
                ax.axhline(y=page_val, color='black', linestyle=':', linewidth=3)
            
            ax.set_xlabel(r'$E_n / J$', fontsize=1.3 * fontsize)
            ax.set_ylabel(r'$S_{1,A}(E_n)$', fontsize=1.3 * fontsize)
            ax.set_ylim(y_min, y_max)
            ax.set_box_aspect(1)
            
            if idx < len(labels):
                ax.text(
                    -0.2, 1.05, labels[idx],
                    transform=ax.transAxes,
                    fontsize=fontsize*1.3,
                    weight='bold'
                )
            
            ax.set_title(fr'$\Delta_{{\mathrm{{local}}}}={d["Delta_local"]/J:.3g} J$', 
                        fontsize=1.1 * fontsize, pad=20)
            
            style_axis(ax, fontsize=fontsize)
        
        cax = fig.add_subplot(gs[0, n_datasets])
        cbar = fig.colorbar(sc, cax=cax)
        cbar.set_label(r'$\log_{10}(\mathrm{count}+1)$', fontsize=1.1 * fontsize)
        
        results_dir = os.path.join(dir_root, "results")
        os.makedirs(results_dir, exist_ok=True)
        out_name = f"vn_entropy_vs_energy_combined_{uid}.png"
        out = os.path.join(results_dir, out_name)
        plt.savefig(out)
        plt.close(fig)
        print(f"Saved combined VN entropy vs energy plot to {out}")



def make_fig1_dos(N, Delta_local_ls, Delta_mean_ls, n_repeats, dir_root, fontsize=40, alpha=0.8, J=5.42, x_min_max=None, bins_e = 120, bins_s = 120, **kwargs):
    a = 10
    J_arr = get_J_arr([(i*a,0) for i in range(N)], N)
    H_int_ = H_int(J_arr, N)

    mpl.rcParams.update({'font.size': fontsize})
    plt.rc('text', usetex=True)
    mpl.rc('text.latex', preamble=r"""
    \usepackage{amsmath}
    \usepackage{newtxtext,newtxmath}
    """)

    colors = ['red', 'black']
    linestyles = ['-', '--',]

    n_mean = len(Delta_mean_ls)
    fig, axs = plt.subplots(1, n_mean, figsize=(10 * n_mean, 10))
    
    if n_mean == 1:
        axs = [axs]
    
    labels = [r'$\sf{\textbf{a}}$', r'$\sf{\textbf{b}}$', r'$\sf{\textbf{c}}$', r'$\sf{\textbf{d}}$']

    for i, Delta_mean in enumerate(Delta_mean_ls):
        ax = axs[i]
        
        for j, Delta_local in enumerate(Delta_local_ls):
            print(f"Calculating for Delta_mean={Delta_mean}, Delta_local={Delta_local}...")

            all_results = repeat_quantities_general(N, None, H_int_, Delta_local, Delta_mean, n_repeats, func=evals_only, fileprefix='fig1_dos', dir_root=dir_root, **kwargs)

            eigevals_all = np.array([res['evals'] for res in all_results])

            # density of states
            ev = eigevals_all.flatten() / J
            ev_counts, ev_edges = np.histogram(ev, bins=100, density=True)
            ev_centers = (ev_edges[:-1] + ev_edges[1:]) / 2
            if np.max(ev_counts) > 0:
                ev_counts_plot = ev_counts / np.max(ev_counts)
            else:
                ev_counts_plot = ev_counts
            ax.plot(
                ev_centers, ev_counts_plot,
                linestyle=linestyles[j], color=colors[j], alpha=alpha,
                linewidth=6,
                label=fr'$\Delta_{{\mathrm{{local}}}}={Delta_local/J:.3g} J$'
            )
            if x_min_max[i] is not None:
                ax.set_xlim(x_min_max[i])

        ax.set_xlabel(r'$E_n / J$', fontsize=1.3*fontsize)
        ax.set_ylabel(r'$\varrho (E_n / J)$', fontsize=1.3*fontsize)
        ax.set_title(fr'$\langle \Delta_{{i}} \rangle={Delta_mean/J:.3g} J$', 
                    fontsize=1.1 * fontsize, pad=20)
        ax.legend(fontsize=0.9 * fontsize, loc='upper left')
        
        if i < len(labels):
            ax.text(
                -0.2, 1.05, labels[i],
                transform=ax.transAxes,
                fontsize=fontsize*1.3,
                weight='bold'
            )
        
        style_axis(ax, fontsize=fontsize)

    plt.tight_layout()
    manager = ExptStore(dir_root)
    payload = {
        "type": "dos_only",
        "N": N,
        "Delta_local_ls": Delta_local_ls,
        "Delta_mean_ls": Delta_mean_ls,
        "n_repeats": n_repeats,
    }
    if 'threshold' in kwargs:
        threshold = kwargs['threshold']
        if threshold > 0:
            payload['threshold'] = threshold
    if 'Omega' in kwargs:
        if kwargs['Omega'] != 15.8:
            payload['Omega'] = kwargs['Omega']

    uid, added = manager.add(payload, timestamp=0)
    results_dir = os.path.join(dir_root, "results")
    os.makedirs(results_dir, exist_ok=True)
    out = os.path.join(results_dir, f"fig1_dos_{uid}.png")
    plt.savefig(out)
    plt.close(fig)


def magnetisation_fig(
    base_drive,
    Delta_i_avg_ls_mag,
    Delta_i_avg_ls_one_P,
    Delta_i_std_ls,
    n_q_ls,
    dir_main,
    alpha=0.8,
    fontsize=40,
    force_recompute=False,
    J_value=5.42,
    n_repeats=1,
):

    def _to_list(x):
        return x.tolist() if isinstance(x, np.ndarray) else list(x)

    def _mag_onep_one_time(N, H_int, Delta_local, Delta_mean, Omega=15.8, phi=0, threshold=0, NN_only=False):
        h_ls = get_h_ls(N, threshold=threshold)
        evals, evecs = diagonalize_aquila(
            Delta_local,
            Delta_mean,
            N,
            Omega=Omega,
            phi=phi,
            H_int=H_int,
            threshold=threshold,
            h_ls=h_ls,
            NN_only=NN_only,
        )

        mbgs = evecs[0]
        mbgs_probs = np.abs(mbgs.full().ravel(order='F'))**2
        hamming_weights = np.array([bin(j).count('1') for j in range(2**N)])
        one_p = np.array([
            np.sum(mbgs_probs[hamming_weights == weight])
            for weight in range(N + 1)
        ], dtype=np.float64)
        mag_mbgs = np.sum((N - 2 * np.arange(N + 1)) * one_p)

        return {
            'magnetization_mbgs': float(np.real(mag_mbgs)),
            'one_P': one_p,
            'evals': np.asarray(evals, dtype=np.float64),
        }

    ev_params = base_drive.get('ev_params', {}) if isinstance(base_drive, dict) else {}
    Omega = ev_params.get('Omega', 15.8)
    phi = ev_params.get('phi', 0)
    threshold = ev_params.get('threshold', 0)
    NN_only = bool(ev_params.get('NN_only', False))

    mpl.rcParams.update({'font.size': fontsize})
    plt.rc('text', usetex=True)
    mpl.rc('text.latex', preamble=r"""
    \usepackage{amsmath}
    \usepackage{newtxtext,newtxmath}
    """)

    fig, axs = plt.subplots(1, 2, figsize=(24, 12))
    colors = ['black', 'red']
    colors2 = [['black', 'red'], ['gray', 'lightcoral']]
    linestyles = ['--', '-']

    cached_eval_sets = None

    for n_q in n_q_ls:
        a = 10
        J_arr = get_J_arr([(i * a, 0) for i in range(n_q)], n_q)
        H_int_ = H_int(J_arr, n_q)

        for j, Delta_local in enumerate(Delta_i_std_ls):
            mag_vals = []
            for Delta_mean in Delta_i_avg_ls_mag:
                all_results = repeat_quantities_general(
                    n_q,
                    None,
                    H_int_,
                    Delta_local,
                    Delta_mean,
                    n_repeats,
                    dir_root=dir_main,
                    fileprefix='fig2_mag',
                    func=_mag_onep_one_time,
                    Omega=Omega,
                    phi=phi,
                    threshold=threshold,
                    NN_only=NN_only,
                )
                mag_vals.append(np.mean([res['magnetization_mbgs'] for res in all_results]))

            mag_mean = np.asarray(mag_vals, dtype=np.float64)
            axs[0].plot(
                np.asarray(Delta_i_avg_ls_mag, dtype=np.float64) / J_value,
                mag_mean / n_q,
                alpha=alpha,
                label=fr'$\Delta_{{\mathrm{{local}}}} = {Delta_local/J_value:.3g}J$',
                linewidth=6,
                linestyle=linestyles[j % len(linestyles)],
                color=colors[j % len(colors)],
            )

        if n_q == n_q_ls[-1]:
            x = np.arange(n_q + 1)
            bar_width = 0.4

            one_p_means = np.zeros((len(Delta_i_avg_ls_one_P), len(Delta_i_std_ls), n_q + 1), dtype=np.float64)
            for i, Delta_mean in enumerate(Delta_i_avg_ls_one_P):
                for j, Delta_local in enumerate(Delta_i_std_ls):
                    all_results = repeat_quantities_general(
                        n_q,
                        None,
                        H_int_,
                        Delta_local,
                        Delta_mean,
                        n_repeats,
                        dir_root=dir_main,
                        fileprefix='fig2_onep',
                        func=_mag_onep_one_time,
                        Omega=Omega,
                        phi=phi,
                        threshold=threshold,
                        NN_only=NN_only,
                    )

                    one_p_arr = np.asarray([res['one_P'] for res in all_results], dtype=np.float64)
                    one_p_means[i, j] = np.mean(one_p_arr, axis=0)

                    if i == 0 and j == 0:
                        cached_eval_sets = [np.asarray(res['evals'], dtype=np.float64) for res in all_results]

                    offset = (i * len(Delta_i_std_ls) + j) - (len(Delta_i_avg_ls_one_P) * len(Delta_i_std_ls) - 1) / 2
                    axs[1].bar(
                        x + offset * bar_width,
                        one_p_means[i, j],
                        width=bar_width,
                        alpha=alpha,
                        color=colors2[i % len(colors2)][j % len(colors2[0])],
                        label=fr'$\Delta_{{\mathrm{{local}}}}={Delta_local/J_value:.3g}J$'
                    )

    axs[0].set_xlabel(r'$\langle \Delta_i \rangle / J$', fontsize=fontsize * 1.2)
    axs[0].set_ylabel(r'$\langle \mathcal{M} \rangle / N$', fontsize=fontsize * 1.2)
    axs[1].set_xlabel(r'$N_r$', fontsize=fontsize * 1.2)
    axs[1].set_ylabel(r'$P(N_r)$', fontsize=fontsize * 1.2)
    axs[0].legend(loc='upper right', fontsize=fontsize * 0.9)
    axs[1].legend(loc='upper left', bbox_to_anchor=(-0.018, 1), fontsize=fontsize * 0.9)

    labels = [r'$\sf{\textbf{a}}$', r'$\sf{\textbf{b}}$']
    for ax, lab in zip(axs.flat, labels):
        ax.text(-0.15, 1.05, lab, transform=ax.transAxes, fontsize=fontsize * 1.2)

    for ax in [axs[0], axs[1]]:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(2)
        ax.minorticks_on()
        ax.tick_params(axis='both', which='major', top=True, right=True, bottom=True, left=True)
        ax.tick_params(axis='both', which='minor', top=True, right=True, bottom=True, left=True, length=4)
        ax.set_box_aspect(1)
        style_axis(ax, fontsize=fontsize)

    plt.tight_layout()

    manager = ExptStore(dir_main)
    payload = {
        'type': 'figure_2',
        'Delta_i_avg_ls_mag': _to_list(Delta_i_avg_ls_mag),
        'Delta_i_avg_ls_one_P': _to_list(Delta_i_avg_ls_one_P),
        'Delta_i_std_ls': _to_list(Delta_i_std_ls),
        'n_q_ls': _to_list(n_q_ls),
        'n_repeats': int(n_repeats),
        'J_value': float(J_value),
        'force_recompute': bool(force_recompute),
        'Omega': float(Omega),
        'phi': float(phi),
        'threshold': float(threshold),
        'NN_only': bool(NN_only),
    }
    uid, _ = manager.add(payload, timestamp=0)
    results_dir = os.path.join(dir_main, 'results')
    os.makedirs(results_dir, exist_ok=True)

    out = os.path.join(results_dir, f'fig2_mag_oneP_{uid}.pdf')
    plt.savefig(out)
    plt.close(fig)
    print(f"Figure 2 saved to {out}")

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.set_xlabel(r'$\varepsilon / J$', fontsize=fontsize * 0.8)
    ax.set_ylabel(r'$\varrho(\varepsilon)$', fontsize=fontsize * 0.8)
    if cached_eval_sets is None or len(cached_eval_sets) == 0:
        print("No eigenvalues found for histogram; skipping eigenvalue histogram figure.")
        plt.close(fig)
        return

    eigenvals = np.concatenate(cached_eval_sets).flatten() / J_value
    counts, edges = np.histogram(eigenvals, bins=100, density=True)
    centers = (edges[:-1] + edges[1:]) / 2
    ax.plot(centers, counts, linestyle='-', alpha=alpha, color='black', linewidth=3)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2)
    ax.minorticks_on()
    ax.tick_params(axis='both', which='minor', top=True, right=True, bottom=True, left=True, length=4)
    ax.tick_params(axis='both', which='major', top=True, right=True, bottom=True, left=True, length=4)
    ax.set_box_aspect(1)
    style_axis(ax, fontsize=fontsize)

    out_hist = os.path.join(results_dir, f'fig2_eigenvalue_hist_{uid}.pdf')
    plt.savefig(out_hist)
    plt.close(fig)
    print(f"Figure 2 eigenvalue histogram saved to {out_hist}")


if __name__ == "__main__":
    N = 6
    J = 5.42
    t_ls = np.logspace(-2, np.log10(4), 500) 
    # Delta_local_ls = [-0.5*J] 
    Delta_local_ls = [-10*J, -0.5*J] 
    # Delta_mean_ls = [0.5*J] 
    # Delta_local_ls = [-0.5*J] 
    Delta_mean_ls = [0.5*J] 
    n_repeats = 1000
    # fig_signatures(N, t_ls, Delta_local_ls, Delta_mean_ls, n_repeats, dir_root='fig1', fontsize=40, alpha=0.8, J=5.42, threshold=0.0, Omega = 15.8, NN_only=False)

    # make_fig1_dos(N, Delta_local_ls, Delta_mean_ls, n_repeats, dir_root='fig1', fontsize=40, alpha=0.8, J=5.42, threshold=0.0, Omega = 2*J, NN_only=False, x_min_max=[(-25,25), None])
    
    
    # fig_signatures(N, t_ls, Delta_local_ls, Delta_mean_ls, n_repeats, dir_root='fig1', fontsize=40, alpha=0.8, J=5.42, threshold=0.0, Omega = 2*J, NN_only=True)
    # fig_signatures(N, t_ls, Delta_local_ls, Delta_mean_ls, n_repeats, dir_root='fig1_test', fontsize=20, alpha=0.8, J=5.42, threshold=0.0, Omega = 15.8, NN_only=False, middle_h_1=True)
    # fig_signatures(N, t_ls, Delta_local_ls, Delta_mean_ls, n_repeats, dir_root='fig1_test', fontsize=20, alpha=0.8, J=5.42, threshold=0.0, Omega = 15.8, NN_only=False, middle_h_1=False)

    base_drive = {
                "ev_params": {
                    'Omega': 15.8,
                    'phi': 0,
                    'a': 10,
                    'eps_std': 0,
                    't_ramp': 0.0632,
                    'n_q': 6
                },
                'n_ens': 500
            }
    
    Delta_i_avg_ls_mag = np.linspace(-15, 15, 10) * J
    Delta_i_avg_ls_one_P = [0.5*J]
    Delta_i_std_ls = [-0.5*J, -10*J]
    n_q_ls = [6]

    magnetisation_fig(base_drive, Delta_i_avg_ls_mag, Delta_i_avg_ls_one_P, Delta_i_std_ls, n_q_ls, dir_main='fig2', alpha=0.8, fontsize=40, force_recompute=False, J_value=5.42, n_repeats=1000)

    
    





    