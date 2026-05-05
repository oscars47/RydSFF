import numpy as np
import qutip as qt
import os, time
from QuEraToolbox.random_bp_qutip import get_probs_seq_ls
from master_params_rbp import gen_seq_ls_pre
import matplotlib.pyplot as plt 
import matplotlib as mpl
from fig_styling import style_axis

def get_bloch_coords(rho_single_qubit):
    a_x = 2 * rho_single_qubit[1, 0].real
    a_y = 2 * rho_single_qubit[1, 0].imag
    a_z = (2 * rho_single_qubit[0, 0] - 1).real
    return (a_x, a_y, a_z)

def draw_bloch_sphere(ax, fontsize=20, show_theta_phi=True):
    fontsize*=1.3
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))

    ax.plot_surface(xs, ys, zs, alpha=0.05, linewidth=0.5, color='gray')
    
    # Add state labels at poles using Computer Modern for g/r
    ax.text(0, 0, 1.19, r'$|\mathcm{g}\rangle$', fontsize=fontsize, ha='center', va='center', clip_on=False)
    ax.text(0, 0, -1.24, r'$|\mathcm{r}\rangle$', fontsize=fontsize, ha='center', va='center')
    
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
    
    ax.plot([-1.0, 1.0], [0, 0], [0, 0], 'k-', linewidth=2, alpha=0.3)  
    ax.plot([0, 0], [-1.0, 1.0], [0, 0], 'k-', linewidth=2, alpha=0.3)  
    ax.plot([0, 0], [0, 0], [-1.0, 1.0], 'k-', linewidth=2, alpha=0.3)  
    
    ax.view_init(elev=20, azim=45)
    
    if show_theta_phi:
        theta_arc = np.linspace(0, np.pi/3, 20)
        x_theta = -np.sin(theta_arc)  # radius = 1 (sphere surface), reflected by pi
        z_theta = np.cos(theta_arc)
        y_theta = np.zeros_like(theta_arc)
        ax.plot(x_theta, y_theta, z_theta, 'k-', linewidth=2)
        ax.text(-0.27, 0, 0.85, r'$\theta$', fontsize=fontsize, ha='center', va='center')
        
        phi_arc = np.linspace(np.pi/2, np.pi, 20)  # quadrant 4 reflected by pi
        z_level = 0.3  # Lower part of upper sphere
        radius_at_z = np.sqrt(1 - z_level**2)  # Radius of circle at this z-level
        x_phi = radius_at_z * np.cos(phi_arc)
        y_phi = radius_at_z * np.sin(phi_arc)
        z_phi = np.full_like(phi_arc, z_level)
        ax.plot(x_phi, y_phi, z_phi, 'k-', linewidth=2)
        ax.text(-0.55, 0.7, 0.6, r'$\varphi$', fontsize=fontsize, ha='center', va='center')
    return ax, u, v


def compare_methods(n_U, N=6, rand_seed=47, t_plateau=1.0, a = 10, Delta_mean = 0.5*5.42, Delta_local=-10*5.42, qubit_index=2, n_gate=16, h_ls=None):

    # np.random.seed(rand_seed)
    if h_ls is None or len(h_ls) != N:
        h_ls = np.random.uniform(0, 1, size=N)
    x = [(i * a, 0) for i in range(N)]  

    in_gate_params = {
                "gate_duration": 0.06220645598688665, 
                # "gate_duration": 0.005,
                "n_gates": n_gate, 
                "Delta_local": -102.72161226237358, "Delta_global": 6.733840091053594,
                "n_U": n_U, "n_shots": 200, 'cluster_spacing': None
            }
    
    J = 5.42
    
    base_params = {
            "ev_params": {
                "Omega": 15.8,
                "Delta_local": -0.5*J, # placeholder, overwritten later
                "Delta_global": 0.5*J - .5 * (-0.5*J),  # placeholder, overwritten later
                # "phi": np.pi/7,
                "phi": 0,
                "t_ramp": 0.0632,
                # "t_ramp": 0.0,
                "a": a
            },
            "n_ens": 1
        }
    
    # sample n_U length n_gates sequences
    seq_ls_pre = gen_seq_ls_pre(1, 1, [[n_U]], [[n_gate]],1, 1, same_U_all_time=False, phi_mode='binary')[0][0][0]

    # [[[2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 1, 1, 2, 1, 2, 2], [2, 1, 2, 1, 1, 1, 1, 1, 2, 2, 1, 2, 1, 2, 2, 2], [1, 1, 2, 2, 2, 2, 2, 2, 2, 1, 2, 1, 1, 2, 2, 2], ...., x n_U]]

    _, global_gate_states = get_probs_seq_ls(h_ls, x, [t_plateau], seq_ls_pre, base_params, Delta_mean, Delta_local, in_gate_params, neg_phi=True, preset_opt=None, override_local=False, start_Delta_from0=False, uniform_Omega_Delta_ramp=False, phi_mode='binary', ret_probs=False, ret_continuous=False, return_last_state=True)
    # print("global length", len(global_gate_states))

    # Get final state for no gates (without continuous evolution)
    _, evolve_no_gate_state = get_probs_seq_ls(h_ls, x, [t_plateau], [[]], base_params, Delta_mean, Delta_local, in_gate_params, neg_phi=True, preset_opt=None, override_local=False, start_Delta_from0=False, uniform_Omega_Delta_ramp=False, phi_mode='binary', ret_probs=False, ret_continuous=False, return_last_state=True)
    evolve_no_gate_state = evolve_no_gate_state[0][0]  # Extract the final state
    
    # Get continuous evolution for the first sequence only (for trajectory plotting)
    _, cont_times, cont_probs, cont_states = get_probs_seq_ls(h_ls, x, [t_plateau], [[seq_ls_pre[0][0]]], base_params, Delta_mean, Delta_local, in_gate_params, neg_phi=True, preset_opt=None, override_local=False, start_Delta_from0=False, uniform_Omega_Delta_ramp=False, phi_mode='binary', ret_probs=False, ret_continuous=True, continuous_gates_only=True, n_continuous_steps=500, skip_to_this_state=evolve_no_gate_state)
    
    # Extract continuous trajectory for plotting (trace to single qubit)
    cont_trajectory_states = [state.ptrace(qubit_index) for state in cont_states]
    cont_bloch_coords = [get_bloch_coords(state) for state in cont_trajectory_states]
    # print(evolve_no_gate_state)
    evolve_haar_states = []
    for i in range(n_U):
        U_haar = qt.tensor([qt.rand_unitary(2) for _ in range(N)])
        # print(U_haar.dims, evolve_no_gate_state.dims)
        psi_after = U_haar * evolve_no_gate_state
        evolve_haar_states.append(psi_after)


    # now compare; trace over all but qubit_index
    global_gate_q_index_state = []
    for seq_states in global_gate_states[0]:  # global_gate_states is now [final_states_for_plateau_0]
        global_gate_q_index_state.append([state.ptrace(qubit_index) for state in seq_states])
    no_gate_q_index_state = evolve_no_gate_state.ptrace(qubit_index) 
    evolve_haar_q_index_state = [evolve_haar_states[i].ptrace(qubit_index) for i in range(n_U)]

    bloch_coords_no_gate = get_bloch_coords(no_gate_q_index_state) # all the same
    bloch_coords_global_gate = []
    for i in range(n_U):
        seq_states = global_gate_q_index_state[i]
        bloch_coords_global_gate.append([get_bloch_coords(seq_states[s]) for s in range(len(seq_states))])
    bloch_coords_haar = [get_bloch_coords(evolve_haar_q_index_state[i]) for i in range(n_U)]
    print("SEQ_LS", seq_ls_pre)

    return bloch_coords_no_gate, bloch_coords_global_gate, bloch_coords_haar, cont_bloch_coords, h_ls

def make_figure(bloch_coords_no_gate, bloch_coords_global_gate, bloch_coords_haar, cont_bloch_coords, h_ls, qubit_index, fontsize=16, main_dir="gate_demo"):
    """
    bloch_coords_no_gate: single (x, y, z) for the bare state without any gates
    bloch_coords_global_gate: list of length n_U sublists for each (x, y, z) coorindate for each element in the sequence 
    bloch_coords_haar: list of length n_U sublists for each (x, y, z) coorindate applying haar random
    """

    # subpanel a: 
    # - show trajectories in faint lines from no_gate (a black solid circle) to each element of the sequence within each separate realization of global gate (blue smooth lines connecting small blue squares) to the final global gate state (denoted with a solid blue square)
    # - and to haar (solid orange triangle) 
    # - all on a 3d sphere
    #####
    # subpanel b: plot distribution of z values for global gate final state (last in sequence) vs haar
    ####
    # subpanel c: plot the purity for each state in the sequence for global gate; overplot each realizalization with faint lines and show the average as a bold line
    ####

    # --- Normalize / coerce input shapes ---
    bloch_coords_no_gate = np.asarray(bloch_coords_no_gate, dtype=float).reshape(3)

    bloch_coords_global_gate = np.asarray(bloch_coords_global_gate, dtype=float)
    if bloch_coords_global_gate.ndim == 2:
        # assume shape (T, 3) → single realization
        bloch_coords_global_gate = bloch_coords_global_gate[None, :, :]
    assert bloch_coords_global_gate.shape[-1] == 3, "bloch_coords_global_gate must end with size-3 (x,y,z)"

    bloch_coords_haar = np.asarray(bloch_coords_haar, dtype=float)
    if bloch_coords_haar.ndim == 1:
        # single Haar state
        bloch_coords_haar = bloch_coords_haar[None, :]
    assert bloch_coords_haar.shape[-1] == 3, "bloch_coords_haar must end with size-3 (x,y,z)"

    n_U, T, _ = bloch_coords_global_gate.shape
    if bloch_coords_haar.shape[0] != n_U:
        raise ValueError(f"Number of Haar states ({bloch_coords_haar.shape[0]}) "
                         f"must match number of global-gate realizations ({n_U}).")
    

    colors = ['blue','#B95C83','#E5A774','#579C4C']

    # --- Set up figure and grid ---
    mpl.rcParams.update({'font.size': fontsize})
    # use text.usetex = True
    os.environ["PATH"] += os.pathsep + "/Library/TeX/texbin/"
    plt.rc('text', usetex=True)
    mpl.rc('text.latex', preamble=r"""
    \usepackage{amsmath}
    \usepackage{newtxtext,newtxmath}
    \DeclareSymbolFont{cmletters}{OML}{cmm}{m}{it}
    \DeclareSymbolFontAlphabet{\mathcm}{cmletters}
    """)
    fig = plt.figure(figsize=(22, 22))   # adjusted figure size
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0])
    
    # ============================================================
    # Subpanel (a): Bloch sphere trajectories
    # ============================================================
    ax_a = fig.add_subplot(gs[0, 0], projection='3d')

    # Draw Bloch sphere
    ax_a, u, v = draw_bloch_sphere(ax_a, fontsize=fontsize, show_theta_phi=True)

    # Starting point: no gate (final state after evolution without gates)
    x0, y0, z0 = bloch_coords_no_gate
    print("No-gate final state Bloch coords:", x0, y0, z0)
    
    # Check if continuous trajectory starts from the same initial point
    cont_coords = np.array(cont_bloch_coords)
    x_cont_start, y_cont_start, z_cont_start = cont_coords[0, 0], cont_coords[0, 1], cont_coords[0, 2]
    print("Continuous trajectory start Bloch coords:", x_cont_start, y_cont_start, z_cont_start)
    
    # Draw vector from origin to the start of the gate sequence
    ax_a.quiver(0, 0, 0, x_cont_start, y_cont_start, z_cont_start, linewidth=6, color='black', alpha=0.8, arrow_length_ratio=0.2)
    # ax_a.scatter([x0], [y0], [z0], marker='o', s=120, label=r'$\mathrm{No \, gate}$', zorder=10, color='black')
    

    # plot an inner sphere of radius at purity
    purity = (1.0 + np.sum(bloch_coords_no_gate**2)) / 2.0
    purity_radius = np.sqrt(2 * purity - 1)
    xs_p = purity_radius * np.outer(np.cos(u), np.sin(v))
    ys_p = purity_radius * np.outer(np.sin(u), np.sin(v))
    zs_p = purity_radius * np.outer(np.ones_like(u), np.cos(v))
    ax_a.plot_surface(xs_p, ys_p, zs_p, alpha=0.05, color='purple', linewidth=0, antialiased=True)
    # remove edge lines
    ax_a.set_xticks([])
    ax_a.set_yticks([])
    ax_a.set_zticks([])

    ax_a.set_xticklabels([])
    ax_a.set_yticklabels([])
    ax_a.set_zticklabels([])

    # Make panes transparent and remove the 3D cube edges (axis lines)
    for axis in (ax_a.xaxis, ax_a.yaxis, ax_a.zaxis):
        # documented methods on Axis3D
        axis.set_pane_color((1.0, 1.0, 1.0, 0.0))   # RGBA, alpha=0
        axis.line.set_color((1.0, 1.0, 1.0, 0.0))   # hide the axis line itself

    ax_a.grid(False)
    # Optional: no labels at all
    ax_a.set_xlabel("")
    ax_a.set_ylabel("")
    ax_a.set_zlabel("")

    num_show = min(10, n_U)  

    # Trajectories for each global gate realization
    for i in range(num_show):
        traj = bloch_coords_global_gate[i]       # shape (T, 3)
        x, y, z = traj[:, 0], traj[:, 1], traj[:, 2]

        # solid square for final global gate state
        ax_a.scatter([x[-1]], [y[-1]], [z[-1]],
                     marker='s', s=200, color=colors[0], label=r'$\phi \, \mathrm{Quench}$' if i == 0 else None, alpha=1.0) # s = 10 from before

        # Haar point and a line from no-gate to Haar
        xh, yh, zh = bloch_coords_haar[i]
        ax_a.scatter([xh], [yh], [zh],
                     marker='^', s=200, color=colors[1], label=r'$\mathrm{Local \, Haar}$' if i == 0 else None, alpha=1.0)
        
            
        
        # Plot continuous trajectory for the first sequence only
        if i == 0:
            # Plot gate sequence continuous trajectory
            cont_coords = np.array(cont_bloch_coords)
            x_cont, y_cont, z_cont = cont_coords[:, 0], cont_coords[:, 1], cont_coords[:, 2]
            
            # Plot smooth continuous trajectory
            ax_a.plot(x_cont, y_cont, z_cont, linewidth=2, alpha=1.0, color='black', 
                     label=r'$\mathrm{Evolution}$')
            
            # outline the final point with black box
            print("Continuous trajectory final Bloch coords:", x_cont[-1], y_cont[-1], z_cont[-1])
            # plot the final blue square with black outline
            ax_a.scatter([x_cont[-1]], [y_cont[-1]], [z_cont[-1]],
                         marker='s', s=150, color=colors[0])
            # add black square outline
            ax_a.scatter([x_cont[-1]], [y_cont[-1]], [z_cont[-1]],
                         marker='s', s=350, facecolors='none', edgecolors='black', linewidth=1.5) # s = 100 before
          
        

    ax_a.set_box_aspect((1, 1, 1))
    ax_a.set_proj_type('persp')  # slightly tighter view
    
    # Make the sphere bigger and remove whitespace
    ax_a.view_init(elev=20, azim=45)  # Better viewing angle
    ax_a.dist = 2.5  # Make sphere appear much larger (smaller dist = bigger appearance)
    
    # Remove all margins and padding
    ax_a.margins(0, 0, 0)

    # ============================================================
    # Subpanel (b): Distribution of z values (final global vs Haar)
    # ============================================================
    ax_b = fig.add_subplot(gs[1, 0])
    # ax_b.set_title("(b) Distribution of $z$")

    # Final global-gate z-components
    z_global_final = bloch_coords_global_gate[:, -1, 2]
    z_haar = bloch_coords_haar[:, 2]

    # Compute histograms without plotting
    bins = np.linspace(-1, 1, 20)
    counts_global, edges = np.histogram(z_global_final, bins=bins, density=True)
    counts_haar, _     = np.histogram(z_haar,         bins=bins, density=True)

    # Bin centers
    centers = 0.5 * (edges[1:] + edges[:-1])

    # Bar width for each dataset inside a single histogram bin.
    # Because width < bin_width for side-by-side bars, rescale heights so
    # each displayed histogram still has area 1.
    bin_width = edges[1] - edges[0]
    width = 0.45 * bin_width
    print("width", width)
    area_rescale = bin_width / width

    # Side-by-side bars with area-preserving height scaling
    ax_b.bar(centers - width/2, counts_global * area_rescale, width=width,
            alpha=1.0, label=r'$\varphi \, \mathrm{Quench}$', color=colors[0])
    ax_b.bar(centers + width/2, counts_haar * area_rescale, width=width,
            alpha=1.0, label=r'$\mathrm{Local \, Haar}$', color=colors[1])
    
    # print(np.diff(edges))
    # print(np.sum(counts_global * np.diff(edges)))
    # print(np.sum(counts_haar * np.diff(edges)))
    print("integral", np.sum(counts_haar * area_rescale * width))
    
     # what is variance of blue as a fraction of orange
    var_global = np.var(z_global_final)
    var_haar = np.var(z_haar)
    print(f"Variance of final z (global quench): {var_global}, Haar: {var_haar}, ratio: {var_global/var_haar}")
    
    ## set x lim to be min and max of opulated data
    all_z = np.concatenate([z_global_final, z_haar])
    z_min, z_max = all_z.min(), all_z.max()
    z_range = z_max - z_min
    ax_b.set_xlim(z_min - 0.1*z_range, z_max + 0.1*z_range)

    ax_b.set_xlabel(r'$\langle \sigma_i^z \rangle$', fontsize=fontsize*1.3)
    ax_b.set_ylabel(r'$\varrho(\langle \sigma_i^z \rangle)$', fontsize=fontsize*1.3)
    ax_b.set_box_aspect(1)
    ax_b.set_xlim(-1, 1)
    

    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='s', color=colors[0], linestyle='None', 
                             markersize=16, label=r'$\phi \, \mathrm{Quench}$'),
                      Line2D([0], [0], marker='^', color=colors[1], linestyle='None', 
                             markersize=16, label=r'$\mathrm{Local \, Haar}$')]
    ax_b.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(.8, 1.36), 
                fontsize=fontsize, handletextpad=0.3, frameon=False) #(.7, 1.2)

    # ============================================================
    # Subpanel (c): Purity along global-gate sequence
    # ============================================================
    ax_c = fig.add_subplot(gs[0, 1])
    # ax_c.set_title("(c) Purity along sequence")

    r2 = np.sum(bloch_coords_global_gate**2, axis=-1)  # |r|^2, shape (n_U, T)
    purity = (1.0 + r2) / 2.0

    steps = np.arange(T)

    # plot each realization in faint lines
    for i in range(n_U):
        ax_c.plot(steps, purity[i], alpha=0.05, linewidth=1, color='black')

    # mean purity as bold line
    purity_mean = purity.mean(axis=0)
    ax_c.plot(steps, purity_mean, linewidth=7, label=r'$\overline{\mathrm{Tr}(\rho_i^2)}$', color='black')

    ax_c.set_xlabel(r'$\phi \, \mathrm{Quench \, Number}$', fontsize=fontsize*1.3)
    ax_c.set_ylabel(r'$\mathrm{Tr}(\rho_i^2)$', fontsize=fontsize*1.3)
    # ax_c.set_ylim(0.5, 1.05)
    ax_c.set_box_aspect(1)
    # ax_c.legend(loc='lower left')

    # ============================================================
    # Subpanel (d): Theta and phi along global-gate sequence
    # ============================================================
    ax_d = fig.add_subplot(gs[1, 1])
    # ax_d.set_title("(d) Angles along sequence")

    # Convert Cartesian to spherical coordinates
    x_coords = bloch_coords_global_gate[:, :, 0]  # shape (n_U, T)
    y_coords = bloch_coords_global_gate[:, :, 1]  # shape (n_U, T)
    z_coords = bloch_coords_global_gate[:, :, 2]  # shape (n_U, T)
    
    # Calculate theta (polar angle from z-axis) and phi (azimuthal angle)
    r_coords = np.sqrt(x_coords**2 + y_coords**2 + z_coords**2)
    theta = np.arccos(np.clip(z_coords / np.maximum(r_coords, 1e-10), -1, 1))  # [0, pi]
    phi = np.arctan2(y_coords, x_coords)  # [-pi, pi]

    # plot each realization in faint lines
    for i in range(n_U):
        ax_d.plot(steps, phi[i]/np.pi, alpha=0.1, linewidth=1, color=colors[3])
    for i in range(n_U):
        ax_d.plot(steps, theta[i]/np.pi, alpha=0.1, linewidth=1, color=colors[2])
        
    
    # Add legend entries with full opacity (empty plots just for legend)
    ax_d.plot([], [], color=colors[2], linewidth=2, label=r'$\theta$', alpha=1.0)
    ax_d.plot([], [], color=colors[3], linewidth=2, label=r'$\varphi$', alpha=1.0)
    
    

    # Add horizontal reference lines
    ax_d.axhline(y=1, color='gray', linestyle='--', alpha=0.7, linewidth=1)
    ax_d.axhline(y=0, color='gray', linestyle='--', alpha=0.7, linewidth=1)
    ax_d.axhline(y=-1, color='gray', linestyle='--', alpha=0.7, linewidth=1)

    ax_d.set_xlabel(r'$\phi \, \mathrm{Quench \, Number}$', fontsize=fontsize*1.3)
    ax_d.set_ylabel(r'$\mathrm{Angle} / \, \pi$', fontsize=fontsize*1.3)
    ax_d.set_box_aspect(1)

    axs = [ax_b, ax_c, ax_d]

    for ax in axs:
        style_axis(ax, major_len=10, major_w=2, minor_len=5, minor_w=1.5,
            pad=10, label_weight='bold', fontsize=fontsize)

    ax_d.yaxis.set_label_coords(-0.18, 0.5)

    plt.tight_layout()
    
    pos_a = ax_a.get_position()
    pos_b = ax_b.get_position()
    
    new_width = pos_b.width*1.8
    new_height = pos_b.height *3  # Make it slightly taller
    new_x = pos_b.x0 - 0.1605  # Align left edges
    new_y = pos_b.y0 - pos_b.height + 0.5  # Small gap above panel b
    
    ax_a.set_position([new_x, new_y, new_width, new_height])

    # Add legend for panel d above the panel in a single horizontal row
    ax_d.legend(loc='lower center', bbox_to_anchor=(0.53, 0.95), ncol=2,
                frameon=False, handlelength=3.0, handletextpad=0.6,
                columnspacing=1.8)

    # Force a draw to ensure all positioning is finalized
    fig.canvas.draw_idle()
    
    # Add "a" label above the sphere, y-aligned with the purity panel (ax_c) label
    bbox_c = ax_c.get_position()
    fig.text(pos_b.x0 - 0.03, bbox_c.y1 + 0.011, r'$\sf{\textbf{a}}$',
             transform=fig.transFigure,
             fontsize=fontsize*1.2,
             va="bottom", ha="right",
             clip_on=False)
    
    fig.canvas.draw_idle()
    
    # Add subplot labels for b, c, and d
    labels = [ r'$\sf{\textbf{b}}$', r'$\sf{\textbf{c}}$']
    axes = [ax_c, ax_d]
    
    for ax, lab in zip(axes, labels):
        bbox = ax.get_position()
        x_fig = bbox.x0 - 0.03         # a bit to the left of the axes
        y_fig = bbox.y1 + 0.011        # a bit above the axes

        fig.text(
            x_fig, y_fig, lab,
            transform=fig.transFigure,
            fontsize=fontsize*1.2,
            va="bottom", ha="right",
            clip_on=False,
        )

    # plt.tight_layout()
    os.makedirs(main_dir, exist_ok=True)
    plt.savefig(f"{main_dir}/compare_global_haar_{list(h_ls)}_{qubit_index}.pdf", bbox_inches='tight')
    # plt.show()
    
if __name__ == "__main__":
    n_U = 500
    N = 6
    t_plateau = 1.0
    chaos_h_ls_ls = []

    plot_dir = "gate_demo_correct"
    h_ls = [0.5351778897933708, 0.8762572741949045, 0.1483627705168249, 0.7102834530996069, 0.6273169859842198, 0.4275076110819733]
    qubit_index = 2



    bloch_coords_no_gate, bloch_coords_global_gate, bloch_coords_haar, cont_bloch_coords, h_ls = compare_methods(n_U, N=N, t_plateau=t_plateau, h_ls=h_ls, n_gate=16, qubit_index=qubit_index)

    make_figure(bloch_coords_no_gate, bloch_coords_global_gate, bloch_coords_haar, cont_bloch_coords, h_ls, qubit_index=qubit_index, fontsize=47, main_dir=plot_dir)