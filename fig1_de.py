## script for (1)-- Fig 1d and e to show the behavior @ single qubit level of chaotic and localized Delta_local
##.           (2) showing how the gates work
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm
from fig_styling import style_axis
from hamiltonian import drive_main, get_J_arr, H_int
from expt_file_manager import ExptStore
import qutip as qt
import os
from tqdm import trange



def draw_bloch_sphere(ax, fontsize=20, show_theta_phi=True):
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))

    ax.plot_surface(xs, ys, zs, alpha=0.05, linewidth=0.5, color='gray')
    
    # Add state labels at poles using Computer Modern for g/r
    ax.text(0, 0, 1.25, r'$|\mathcm{g}\rangle$', fontsize=fontsize, ha='center', va='center')
    ax.text(0, 0, -1.3, r'$|\mathcm{r}\rangle$', fontsize=fontsize, ha='center', va='center')
    
    # Add coordinate circles
    # Equatorial circle (xy-plane, z=0)
    theta_eq = np.linspace(0, 2*np.pi, 100)
    x_eq = np.cos(theta_eq)
    y_eq = np.sin(theta_eq)
    z_eq = np.zeros_like(theta_eq)
    ax.plot(x_eq, y_eq, z_eq, 'k-', linewidth=2, alpha=0.3)
    
    # Polar circle (xz-plane, y=0)
    theta_pol = np.linspace(0, 2*np.pi, 100)
    x_pol = np.cos(theta_pol)
    y_pol = np.zeros_like(theta_pol)
    z_pol = np.sin(theta_pol)
    ax.plot(x_pol, y_pol, z_pol, 'k-', linewidth=2, alpha=0.3)
    
    # Add coordinate axes inside sphere (both positive and negative directions)
    ax.plot([-1.0, 1.0], [0, 0], [0, 0], 'k-', linewidth=2, alpha=0.3)  # x-axis
    ax.plot([0, 0], [-1.0, 1.0], [0, 0], 'k-', linewidth=2, alpha=0.3)  # y-axis  
    ax.plot([0, 0], [0, 0], [-1.0, 1.0], 'k-', linewidth=2, alpha=0.3)  # z-axis
    
    # Orient sphere so pole faces up
    ax.view_init(elev=20, azim=45)
    
    # Add angle indicators
    # Theta (polar angle) - arc from z-axis down towards xy-plane, on sphere surface
    if show_theta_phi:
        # Draw theta arc in the -xz-plane (reflected by π), starting from +z axis, on sphere surface
        theta_arc = np.linspace(0, np.pi/3, 20)
        x_theta = -np.sin(theta_arc)  # radius = 1 (sphere surface), reflected by π
        z_theta = np.cos(theta_arc)
        y_theta = np.zeros_like(theta_arc)
        ax.plot(x_theta, y_theta, z_theta, 'k-', linewidth=2)
        ax.text(-0.27, 0, 0.85, r'$\theta$', fontsize=fontsize, ha='center', va='center')
        
        # Phi (azimuthal angle) - arc in xy-plane around z-axis, on upper sphere surface, reflected by π radians
        phi_arc = np.linspace(np.pi/2, np.pi, 20)  # Reflected by π radians (from +y to -x axis)
        z_level = 0.7  # Upper part of sphere
        radius_at_z = np.sqrt(1 - z_level**2)  # Radius of circle at this z-level
        x_phi = radius_at_z * np.cos(phi_arc)
        y_phi = radius_at_z * np.sin(phi_arc)
        z_phi = np.full_like(phi_arc, z_level)
        ax.plot(x_phi, y_phi, z_phi, 'k-', linewidth=2)
        ax.text(-0.38, 0.45, 0.7, r'$\varphi$', fontsize=fontsize, ha='center', va='center')
    return ax, u, v


def get_bloch_coords(rho_single_qubit):
    a_x = 2 * rho_single_qubit[1, 0].real
    a_y = 2 * rho_single_qubit[1, 0].imag
    a_z = (2 * rho_single_qubit[0, 0] - 1).real
    return (a_x, a_y, a_z)

def evolve_state(evals, evecs, psi0, t_ls, q_idx=None):
    #  <E_j|psi0>
    overlaps = np.array([evecs[i].dag() * psi0 for i in range(len(evals))])
    overlaps = np.array([float(overlap) for overlap in overlaps])  # Convert to complex array
    
    # Vectorized phase computation for all times
    phases_matrix = np.exp(-1j * np.outer(evals, t_ls))  # Shape: (n_states, n_times)
    
    # Compute coefficients for all times at once
    coeffs_matrix = overlaps[:, None] * phases_matrix  # Shape: (n_states, n_times)

    if q_idx is not None:
        # Check if q_idx is a list/array or a single value
        if isinstance(q_idx, (list, np.ndarray)):
            # Return dictionary with results for each q_idx
            results = {}
            for q_i in q_idx:
                psi_t_ls = []
                for t_idx in range(len(t_ls)):
                    psi_t = sum(evecs[i] * coeffs_matrix[i, t_idx] for i in range(len(evals)))
                    rho_t = qt.ket2dm(psi_t)
                    rho_single = rho_t.ptrace(q_i)
                    psi_t_ls.append(rho_single)
                results[q_i] = psi_t_ls
            return results
        else:
            # Single q_idx case
            psi_t_ls = []
            for t_idx in range(len(t_ls)):
                psi_t = sum(evecs[i] * coeffs_matrix[i, t_idx] for i in range(len(evals)))
                rho_t = qt.ket2dm(psi_t)
                rho_single = rho_t.ptrace(q_idx)
                psi_t_ls.append(rho_single)
            return psi_t_ls
    else:
        # Fully vectorized reconstruction for full quantum states
        psi_t_ls = [sum(evecs[i] * coeffs_matrix[i, t_idx] for i in range(len(evals))) 
                    for t_idx in range(len(t_ls))]
        return psi_t_ls

def fig1_de(t_ls, Delta_local_ls=[ -0.5*5.42, -10*5.42,], Delta_mean=0.5*5.42, h_ls=[
            0.24996654759766535,
            0.09286441773368292,
            0.03781977537875725,
            0.5848043295638031,
            0.1759548522567862,
            0.3655913041729373
        ], J=5.42, a = 10, Omega = 15.8, phi = 0, fontsize=20, dir_root='fig1', force_recompute=False):
    
    # Set font first so it applies to all text including Bloch sphere lab
    mpl.rcParams.update({
     'font.size': fontsize,
})   

    plt.rc('text', usetex=True)
    mpl.rc('text.latex', preamble=r"""
    \usepackage{amsmath}
    \usepackage{newtxtext,newtxmath}
    \DeclareSymbolFont{cmletters}{OML}{cmm}{m}{it}
    \DeclareSymbolFontAlphabet{\mathcm}{cmletters}
    """)
    
    cmap_chaos = mcolors.LinearSegmentedColormap.from_list(
        "chaos", ["#000000", "#A1A1A1"]
    )
    cmap_localized = mcolors.LinearSegmentedColormap.from_list(
        "localized", ["#FF0000", "#FFA1A1"]
    )

    norm = LogNorm(vmin=t_ls.min(), vmax=t_ls.max())

    t_ls_norm = norm(t_ls)
    point_colors_chaos = [cmap_chaos(val) for val in t_ls_norm]
    point_colors_localized  = [cmap_localized(val) for val in t_ls_norm]


    all0 = qt.tensor([qt.basis(2, 0) for _ in range(len(h_ls))])
    get_H_indep, get_H_ramp, H_d = drive_main(neg_phi=True, ret_H_d=True)
    Delta_local_localized = Delta_local_ls[1]
    Delta_local_chaos = Delta_local_ls[0]

    Delta_global_localized = Delta_mean - .5 * Delta_local_localized
    Delta_global_chaos = Delta_mean - .5 * Delta_local_chaos

    x= [(i * a, 0) for i in range(len(h_ls))]

    payload = {
        "figure": "fig1_de",
        "t_ls": t_ls.tolist(),
        "Delta_local_localized": Delta_local_localized,
        "Delta_local_chaos": Delta_local_chaos,
        "Delta_mean": Delta_mean,
        "h_ls": h_ls,
        "J": J,
        "a": a,
        "Omega": Omega,
        "phi": phi
    }
    uid, added = ExptStore(dir_root).add(payload, timestamp=0)
    os.makedirs(os.path.join(dir_root, "data"), exist_ok=True)

    data_filename = os.path.join(dir_root, "data", f"fig1_chain_{uid}.npy")
    bloch_sphere_filename = os.path.join(dir_root, "data", f"fig1_bloch_{uid}.npy")

    N = len(h_ls)  # Number of qubits


    if not os.path.exists(data_filename) or force_recompute:

        H_plateau_localized = get_H_indep(Omega=Omega, phi=phi, Delta_global=Delta_global_localized, Delta_local=Delta_local_localized, h_ls=h_ls, x=x)
        H_plateau_chaos = get_H_indep(Omega=Omega, phi=phi, Delta_global=Delta_global_chaos, Delta_local=Delta_local_chaos, h_ls=h_ls, x=x)

        evals_localized, evecs_localized = H_plateau_localized.eigenstates()
        evals_chaos, evecs_chaos = H_plateau_chaos.eigenstates()

        

        # Evolve states for all qubits at once 
        q_idx_list = list(range(N))
        psi_t_ls_localized_all = evolve_state(evals_localized, evecs_localized, all0, t_ls, q_idx=q_idx_list)
        psi_t_ls_chaos_all = evolve_state(evals_chaos, evecs_chaos, all0, t_ls, q_idx=q_idx_list)

        # Convert to numpy arrays for saving
        data_to_save = {
            "psi_t_ls_localized_all": np.array([[rho.full() for rho in psi_t_ls_localized_all[q]] for q in range(N)]),
            "psi_t_ls_chaos_all": np.array([[rho.full() for rho in psi_t_ls_chaos_all[q]] for q in range(N)])
        }
        np.save(data_filename, data_to_save, allow_pickle=True)

        bloch_coords_localized_all = np.array([[get_bloch_coords(rho.full()) for rho in psi_t_ls_localized_all[q]] for q in range(N)])
        bloch_coords_chaos_all = np.array([[get_bloch_coords(rho.full()) for rho in psi_t_ls_chaos_all[q]] for q in range(N)])
        bloch_data_to_save = {
            "bloch_coords_localized": bloch_coords_localized_all,
            "bloch_coords_chaos": bloch_coords_chaos_all
        }
        np.save(bloch_sphere_filename, bloch_data_to_save, allow_pickle=True)
    
    elif os.path.exists(data_filename) and not os.path.exists(bloch_sphere_filename):
        print("Loading data from file:", data_filename)
        data = np.load(data_filename, allow_pickle=True).item()
        psi_t_ls_localized_all = {}
        psi_t_ls_chaos_all = {}
        N = len(h_ls)
        for q in range(N):
            psi_t_ls_localized_all[q] = [qt.Qobj(rho) for rho in data["psi_t_ls_localized_all"][q]]
            psi_t_ls_chaos_all[q] = [qt.Qobj(rho) for rho in data["psi_t_ls_chaos_all"][q]]
        bloch_coords_localized_all = np.array([[get_bloch_coords(rho.full()) for rho in psi_t_ls_localized_all[q]] for q in range(N)])
        bloch_coords_chaos_all = np.array([[get_bloch_coords(rho.full()) for rho in psi_t_ls_chaos_all[q]] for q in range(N)])
        bloch_data_to_save = {
            "bloch_coords_localized": bloch_coords_localized_all,
            "bloch_coords_chaos": bloch_coords_chaos_all
        }
        np.save(bloch_sphere_filename, bloch_data_to_save, allow_pickle=True)
    else:        
        print("Loading bloch sphere data from file:", bloch_sphere_filename)
        bloch_data = np.load(bloch_sphere_filename, allow_pickle=True).item()
        bloch_coords_localized_all = bloch_data["bloch_coords_localized"]
        bloch_coords_chaos_all = bloch_data["bloch_coords_chaos"]

    
    # Create figure with 2 rows x N columns
    fig = plt.figure(figsize=(3*N, 8))

    # Loop over all qubits
    for q_idx in trange(N, desc="Plotting qubits..."):
        # Get reduced density matrices for this qubit
        bloch_coords_localized = bloch_coords_localized_all[q_idx]
        bloch_coords_chaos = bloch_coords_chaos_all[q_idx]

        
        
        ax1 = fig.add_subplot(2, N, N + q_idx + 1, projection='3d')
        draw_bloch_sphere(ax1, fontsize=fontsize, show_theta_phi=False)
        
        # Plot time-colored points
        for i, (x, y, z) in enumerate(bloch_coords_localized):
            ax1.scatter(x, y, z, c=[point_colors_localized[i]], s=10, alpha=0.7)
        
        # if q_idx == 0:
        #     ax1.set_title(rf'$\Delta_{{\mathrm{{local}}}} = {Delta_local_localized/J:.3g}J$', fontsize=fontsize, y=1.0)

        ax1.set_xlim([-1.1, 1.1])
        ax1.set_ylim([-1.1, 1.1])
        ax1.set_zlim([-1.1, 1.1])
        ax1.set_box_aspect([1,1,1])
        
        # Remove background panes and grid completely
        ax1.xaxis.pane.fill = False
        ax1.yaxis.pane.fill = False
        ax1.zaxis.pane.fill = False
        ax1.xaxis.pane.set_edgecolor('none')
        ax1.yaxis.pane.set_edgecolor('none')
        ax1.zaxis.pane.set_edgecolor('none')
        ax1.xaxis.pane.set_alpha(0)
        ax1.yaxis.pane.set_alpha(0)
        ax1.zaxis.pane.set_alpha(0)
        ax1.grid(False)
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_zticks([])
        ax1.axis('off')

        # Make axes background fully transparent so overlapping axes don't mask neighbors
        ax1.set_facecolor((1, 1, 1, 0))
        ax1.patch.set_alpha(0)
        ax1.set_zorder(10)

        # (optionally) also kill any remaining frame edge
        ax1.patch.set_edgecolor('none')
                
        # Chaotic case (bottom row)
        ax2 = fig.add_subplot(2, N, q_idx + 1, projection='3d')
        draw_bloch_sphere(ax2, fontsize=fontsize, show_theta_phi=False)
        
        # Plot time-colored points
        for i, (x, y, z) in enumerate(bloch_coords_chaos):
            ax2.scatter(x, y, z, c=[point_colors_chaos[i]], s=5, alpha=0.7)
        
        # if q_idx == 0:
        #     ax2.set_title(rf'$\Delta_{{\mathrm{{local}}}} = {Delta_local_chaos/J:.3g}J$', fontsize=fontsize, y=1.0)
       
        ax2.set_xlim([-1.1, 1.1])
        ax2.set_ylim([-1.1, 1.1])
        ax2.set_zlim([-1.1, 1.1])
        ax2.set_box_aspect([1,1,1])
        
        # Remove background panes and grid completely
        ax2.xaxis.pane.fill = False
        ax2.yaxis.pane.fill = False
        ax2.zaxis.pane.fill = False
        ax2.xaxis.pane.set_edgecolor('none')
        ax2.yaxis.pane.set_edgecolor('none')
        ax2.zaxis.pane.set_edgecolor('none')
        ax2.xaxis.pane.set_alpha(0)
        ax2.yaxis.pane.set_alpha(0)
        ax2.zaxis.pane.set_alpha(0)
        ax2.grid(False)
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.set_zticks([])
        ax2.axis('off')

        # Make axes background fully transparent so overlapping axes don't mask neighbors
        ax2.set_facecolor((1, 1, 1, 0))
        ax2.patch.set_alpha(0)
        ax2.set_zorder(10)

        # (optionally) also kill any remaining frame edge
        ax2.patch.set_edgecolor('none')
            
    plt.subplots_adjust(left=0, right=0.999, top=0.99, bottom=0.01, wspace=-0.6, hspace=-0.1)
    os.makedirs(os.path.join(dir_root, "results"), exist_ok=True)
    # fig_path = os.path.join(dir_root, "results", f"fig1_de.pdf")
    fig_path = os.path.join(dir_root, "results", f"fig1_de_{max(t_ls)}.pdf")
    plt.savefig(fig_path)


if __name__ == "__main__":
    n_pts = 10000
    for t_max in [1e-3, 0.2, 1, 2, 4]:
        t_ls = np.logspace(-3, np.log10(t_max), max(int(n_pts * t_max / 2), 1))
        # t_ls = np.logspace(-3, 1, 10)
        fig1_de(t_ls, fontsize=30) 




