## script for (1)-- Fig 1d and e to show the behavior @ single qubit level of chaotic and localized Delta_local
##.           (2) showing how the gates work
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm
from fig_styling import style_axis
from hamiltonian import drive_main, get_J_arr, H_int
import qutip as qt
import os


def draw_bloch_sphere(ax, fontsize=20, show_theta_phi=True):
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))

    ax.plot_surface(xs, ys, zs, alpha=0.3, linewidth=0.5, color='gray')
    
    # Add state labels at poles using Computer Modern for g/r
    ax.text(0, 0, 1.3, r'$|\mathcm{g}\rangle$', fontsize=fontsize, ha='center', va='center')
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
    # Theta (polar angle) - arc from z-axis
    if show_theta_phi:
        theta_arc = np.linspace(0, np.pi/4, 20)
        x_theta = 0.3 * np.sin(theta_arc)
        z_theta = 0.3 * np.cos(theta_arc)
        y_theta = np.zeros_like(theta_arc)
        ax.plot(x_theta, y_theta, z_theta, 'k-', linewidth=1.5)
        ax.text(0.2, 0, 0.25, r'$\theta$', fontsize=fontsize*0.8, ha='center', va='center')
        
        # Phi (azimuthal angle) - arc in xy-plane
        phi_arc = np.linspace(0, np.pi/3, 15)
        x_phi = 0.4 * np.cos(phi_arc)
        y_phi = 0.4 * np.sin(phi_arc)
        z_phi = np.zeros_like(phi_arc)
        ax.plot(x_phi, y_phi, z_phi, 'k-', linewidth=1.5)
        ax.text(0.3, 0.2, 0, r'$\phi$', fontsize=fontsize*0.8, ha='center', va='center')
    return ax

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
        psi_t_ls = []
        for t_idx in range(len(t_ls)):
            psi_t = sum(evecs[i] * coeffs_matrix[i, t_idx] for i in range(len(evals)))
            rho_t = qt.ket2dm(psi_t)
            rho_single = rho_t.ptrace(q_idx)
            psi_t_ls.append(rho_single)
    else:
        # Fully vectorized reconstruction for full quantum states
        psi_t_ls = [sum(evecs[i] * coeffs_matrix[i, t_idx] for i in range(len(evals))) 
                    for t_idx in range(len(t_ls))]
    
    return psi_t_ls

def fig1_de(t_ls, Delta_local_ls=[-10*5.42, -0.5*5.42], Delta_mean=0.5*5.42, q_idx=3, h_ls=[
            0.24996654759766535,
            0.09286441773368292,
            0.03781977537875725,
            0.5848043295638031,
            0.1759548522567862,
            0.3655913041729373
        ], J=5.42, a = 10, Omega = 15.8, phi = 0, fontsize=20, dir_root='fig1'):
    
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
    Delta_local_localized = Delta_local_ls[0]
    Delta_local_chaos = Delta_local_ls[1]

    Delta_global_localized = Delta_mean - .5 * Delta_local_localized
    Delta_global_chaos = Delta_mean - .5 * Delta_local_chaos

    x= [(i * a, 0) for i in range(len(h_ls))]

    H_plateau_localized = get_H_indep(Omega=Omega, phi=phi, Delta_global=Delta_global_localized, Delta_local=Delta_local_localized, h_ls=h_ls, x=x)
    H_plateau_chaos = get_H_indep(Omega=Omega, phi=phi, Delta_global=Delta_global_chaos, Delta_local=Delta_local_chaos, h_ls=h_ls, x=x)

    evals_localized, evecs_localized = H_plateau_localized.eigenstates()
    evals_chaos, evecs_chaos = H_plateau_chaos.eigenstates()
    psi_t_ls_localized = evolve_state(evals_localized, evecs_localized, all0, t_ls, q_idx=q_idx)
    psi_t_ls_chaos = evolve_state(evals_chaos, evecs_chaos, all0, t_ls, q_idx=q_idx)
    
    # Extract Bloch coordinates from reduced density matrices
    bloch_coords_localized = np.array([get_bloch_coords(rho.full()) for rho in psi_t_ls_localized])
    bloch_coords_chaos = np.array([get_bloch_coords(rho.full()) for rho in psi_t_ls_chaos])

    # Create figure with two 3D subplots side by side
    fig = plt.figure(figsize=(10, 5))
    
    # Localized case (panel d)
    ax1 = fig.add_subplot(121, projection='3d')
    draw_bloch_sphere(ax1, fontsize=fontsize, show_theta_phi=False)
    
    # Plot time-colored points
    for i, (x, y, z) in enumerate(bloch_coords_localized):
        ax1.scatter(x, y, z, c=[point_colors_localized[i]], s=10, alpha=0.7)
    
    ax1.set_title(rf'$\Delta_{{\mathrm{{local}}}} = {Delta_local_localized/J:.3g}J$', fontsize=fontsize)
    ax1.set_xlim([-1.2, 1.2])
    ax1.set_ylim([-1.2, 1.2])
    ax1.set_zlim([-1.2, 1.2])
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
    
    # Chaotic case (panel e)
    ax2 = fig.add_subplot(122, projection='3d')
    draw_bloch_sphere(ax2, fontsize=fontsize, show_theta_phi=False)
    
    # Plot time-colored points
    for i, (x, y, z) in enumerate(bloch_coords_chaos):
        ax2.scatter(x, y, z, c=[point_colors_chaos[i]], s=5, alpha=0.7)
    
    ax2.set_title(rf'$\Delta_{{\mathrm{{local}}}} = {Delta_local_chaos/J:.3g}J$', fontsize=fontsize)
    ax2.set_xlim([-1.2, 1.2])
    ax2.set_ylim([-1.2, 1.2])
    ax2.set_zlim([-1.2, 1.2])
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


    
    # Panel labels
    axs = [ax1, ax2]
    labels = [r'$\sf{\textbf{d}}$', r'$\sf{\textbf{e}}$']
    
    for ax, lab in zip(axs, labels):
        ax.text(
            -0.2, 1.05, 16.5, lab,
            transform=ax.transAxes,
            fontsize=fontsize*1.3,
            weight='bold'
        )
    
    plt.tight_layout()
    os.makedirs(os.path.join(dir_root, "results"), exist_ok=True)
    fig_path = os.path.join(dir_root, "results", f"fig1_de.pdf")
    plt.savefig(fig_path)


if __name__ == "__main__":
    t_ls = np.logspace(-3, 1, 10000)
    fig1_de(t_ls) 




