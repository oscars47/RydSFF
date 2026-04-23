## file for postprocessing the results of random bloqade experiments

from QuEraToolbox.random_bp_prep import expt_run
from QuEraToolbox.expt_file_manager import ExptStore, make_uid
import matplotlib.pyplot as plt
import numpy as np
import os, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from collections import Counter
from uncertainties import UFloat, ufloat
from uncertainties import unumpy as unp
from QuEraToolbox.helper_rbp import get_hamming_matrix, est_purity, est_fidelity, get_ee, get_sp
from QuEraToolbox.random_bp_qutip import get_probs_seq_ls
from process_rbp_calib_helper import get_calib_task


def _resolve_per_ensemble_value(value, ensemble_idx, num_ensembles):
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        if value.shape[0] == num_ensembles:
            return value[ensemble_idx]
        return value

    if isinstance(value, (list, tuple)):
        if len(value) == num_ensembles:
            return value[ensemble_idx]
        return value

    return value


def _scalarize_delta_local(value):
    if np.isscalar(value):
        return float(value)

    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError("Cannot resolve Delta_local from an empty value")
    return float(arr[0])


def _t2_star_from_delta_local(delta_local):
    delta_local_scalar = _scalarize_delta_local(delta_local)
    # return (1 / 4.67 + np.abs(delta_local_scalar) / (2 * np.pi * 5.9)) ** (-1)
    return 1 / (1/6.175127255235175 + np.abs(delta_local_scalar)/(4.779473335372458)) ## updated formula from arinjoy


def _sample_delta_local_disorder(base_delta_local):
    base_delta_local_scalar = _scalarize_delta_local(base_delta_local)
    t2_star = _t2_star_from_delta_local(base_delta_local_scalar)
    return np.random.normal(loc=base_delta_local_scalar, scale=np.sqrt(2) / t2_star)


def _sample_delta_local_disorder_per_qubit(base_delta_local, h_ls):
    """
    Sample per-qubit disorder accounting for h_i scaling in T2star computation.
    
    Each qubit's dephasing is characterized by T2star computed from h_i*Delta_local,
    but the sampled value is centered at base_delta_local 
    The Hamiltonian will apply the h_i factor once to each sample.
    
     N(delta_local, sqrt(2)/T2star*(h_i*delta_local))
    
    Parameters:
    -----------
    base_delta_local : scalar or array-like
        The base Delta_local value (center of sampling distribution)
    h_ls : list or array
        List of Hamiltonian coefficients, one per qubit
    
    Returns:
    --------
    list of scalars
        Per-qubit sampled Delta_local values, each with T2 computed from h_i*Delta_local
        Final effective field per qubit: h_i * sampled_value[i]
    """

    base_delta_scalar = _scalarize_delta_local(base_delta_local)
    h_arr = np.asarray(h_ls, dtype=float)
    
    sampled_values = []
    for h_i in h_arr:
        effective_delta_for_t2star = h_i * base_delta_scalar
        # effective_t2star = (1 / 4.67 + np.abs(effective_delta_for_t2star) / (2 * np.pi * 5.9)) ** (-1)
        effective_t2star = 1 / (1/6.175127255235175 + np.abs(effective_delta_for_t2star)/(4.779473335372458))
        sampled = np.random.normal(loc=base_delta_scalar, scale=np.sqrt(2) / effective_t2star)
        sampled_values.append(sampled)
    
    return sampled_values


def _resolve_time_seq_delta_local(value, time_idx, n_times, seq_idx=None, n_seqs=None):
    resolved = _resolve_per_ensemble_value(value, time_idx, n_times)
    if seq_idx is not None and n_seqs is not None:
        resolved = _resolve_per_ensemble_value(resolved, seq_idx, n_seqs)
    return _scalarize_delta_local(resolved)

def get_single_ham_rand(h_ls, x, t_plateau_ls, seq_ls_pre, base_params,Delta_mean, Delta_local, gate_params, cluster_spacing, manual_parallelization, is_expt_data, timestamp, dir_root,debug=False, check_postarray=False, preset_opt=None, data_subdir="data", override_local=False, save_mode=True, backup_dir_t_s=None, x0_y0_offset = (0,0), t_delay=None, start_Delta_from0=False, uniform_Omega_Delta_ramp=False, phi_mode='binary', Delta_local_ramp_time=0.05, Omega_delay_time=None, sample_Delta_local_each_t_seq=False):
    """
    returns data_all, shape: (t_plateau, seq, bitstrings)
    """

    os.makedirs(os.path.join(dir_root, "data", data_subdir), exist_ok=True)
    
    ev_params = base_params['ev_params']
    t_ramp = ev_params['t_ramp']
    n_shots = gate_params['n_shots']
    Delta_local_init = _resolve_time_seq_delta_local(Delta_local, 0, len(t_plateau_ls))
    ev_params['Delta_global'] = Delta_mean-1/2*Delta_local_init
    ev_params['Delta_local'] = Delta_local_init

    assert not (is_expt_data and timestamp == 0), "timestamp should not be 0 for experiment data"

    def _prepare_delta_locals_for_call(time_idx, seq_idx=None, n_seqs=None):
        evol_delta_local = _resolve_time_seq_delta_local(Delta_local, time_idx, len(t_plateau_ls), seq_idx=seq_idx, n_seqs=n_seqs)

        gate_delta_source = gate_params.get("Delta_local", evol_delta_local)
        gate_delta_local = _resolve_time_seq_delta_local(gate_delta_source, time_idx, len(t_plateau_ls), seq_idx=seq_idx, n_seqs=n_seqs)

        if sample_Delta_local_each_t_seq:
            evol_delta_local = _sample_delta_local_disorder(evol_delta_local)
            gate_delta_local = _sample_delta_local_disorder(gate_delta_local)

        return evol_delta_local, gate_delta_local
    
    data_all = []
    for i, t_plateau in enumerate(t_plateau_ls):
        print("t_plateau progress:", i/len(t_plateau_ls))    
        data_t_fixed_ls = []
        seq_ls = seq_ls_pre[i] # get the sequences for this time plateau
        if type(seq_ls) is list:
            for seq_idx, seq in enumerate(seq_ls):
                if backup_dir_t_s is not None:
                    backup_dir = backup_dir_t_s[i][seq_idx]
                else:
                    backup_dir = None

                evol_delta_local, gate_delta_local = _prepare_delta_locals_for_call(i, seq_idx=seq_idx, n_seqs=len(seq_ls))
                ev_params['Delta_global'] = Delta_mean - 1/2 * evol_delta_local
                ev_params['Delta_local'] = evol_delta_local

                gate_params_run = gate_params
                if sample_Delta_local_each_t_seq or "Delta_local" in gate_params:
                    gate_params_run = deepcopy(gate_params)
                    gate_params_run["Delta_local"] = gate_delta_local

                package = expt_run(h_ls, x, ev_params, t_plateau, t_ramp, seq, n_shots, gate_params_run, is_expt_data,dir_root, timestamp, cluster_spacing=cluster_spacing, manual_parallelization=manual_parallelization, debug=debug, check_postarray=check_postarray, preset_opt=preset_opt, data_subdir=data_subdir,
                override_local=override_local, save_mode=save_mode, backup_dir=backup_dir, x0_y0_offset = x0_y0_offset, t_delay=t_delay, start_Delta_from0=start_Delta_from0, uniform_Omega_Delta_ramp=uniform_Omega_Delta_ramp, phi_mode=phi_mode, Delta_local_ramp_time=Delta_local_ramp_time, Omega_delay_time=Omega_delay_time)

                if len(package) > 0:
                    data_t_fixed_ls.append(package[1]) # append the bitstrings if it's been processed
        else:
            if backup_dir_t_s is not None:
                backup_dir = backup_dir_t_s[i][0]
            else:
                backup_dir = None

            evol_delta_local, gate_delta_local = _prepare_delta_locals_for_call(i)
            ev_params['Delta_global'] = Delta_mean - 1/2 * evol_delta_local
            ev_params['Delta_local'] = evol_delta_local

            gate_params_run = gate_params
            if sample_Delta_local_each_t_seq or "Delta_local" in gate_params:
                gate_params_run = deepcopy(gate_params)
                gate_params_run["Delta_local"] = gate_delta_local

            package = expt_run(h_ls, x, ev_params, t_plateau, t_ramp, [], n_shots, gate_params_run, is_expt_data,dir_root, timestamp, cluster_spacing=cluster_spacing, manual_parallelization=manual_parallelization, debug=debug, check_postarray=check_postarray, preset_opt=preset_opt, data_subdir=data_subdir,
            override_local=override_local, save_mode=save_mode, backup_dir=backup_dir, x0_y0_offset = x0_y0_offset, t_delay=t_delay, start_Delta_from0= start_Delta_from0, uniform_Omega_Delta_ramp=uniform_Omega_Delta_ramp, phi_mode=phi_mode, Delta_local_ramp_time=Delta_local_ramp_time, Omega_delay_time=Omega_delay_time)
            if len(package) > 0:
                data_t_fixed_ls.append(package[1]) # append the bitstrings if it's been processed
        data_all.append(data_t_fixed_ls)
    return data_all

def get_all_single_hams_rand_calib_check(h_ls_pre, x_pre, t_plateau_ls, seq_ls_pre_all, base_params, Delta_mean_ls_all, Delta_local_ls_all, gate_params_all, cluster_spacing, manual_parallelization, is_expt_data, timestamp, dir_root, force_recompute=False, debug=False,check_postarray=False, preset_opt=None, name = "", override_local=False, save_mode=True, backup_dirs=None, x0_y0_offset = (0,0), after_how_many_ham_run_check=2, ham_check_dir_main="calib_check", t_delay=None, start_Delta_from0=False, uniform_Omega_Delta_ramp=False, phi_mode='binary', Delta_local_ramp_time=0.05, Omega_delay_time=None, ignore_seq_for_muid=False, backwards_compatible_muid=False, sample_Delta_local_each_t_seq=False, specify_ensemble=None):
    # print("get_all_single_hams_rand_calib_check called")

    assert after_how_many_ham_run_check is not None, "after_how_many_ham_run_check should be an integer"
    assert ham_check_dir_main is not None, "ham_check_dir_main should be a string"

    assert not (is_expt_data and timestamp == 0), "timestamp should not be 0 for experiment data"

    if not is_expt_data:
        timestamp = 0

    if not is_expt_data:
        save_mode = False # calculate directly if numerical

    specify_ensemble_set = set(specify_ensemble) if specify_ensemble is not None else None

    manager = ExptStore(dir_root)
    # don't include seq_ls_pre_all or timestamp since we want to compile the data for different independent runs on difference accounts
    # could put an exclusion rule for certain timestamps or users

    manager_params = {
        'h_ls_pre': h_ls_pre,
        'x_pre': x_pre,
        't_plateau_ls': t_plateau_ls,
        'base_params': base_params,
        'Delta_mean_ls_all': Delta_mean_ls_all,
        'Delta_local_ls_all': Delta_local_ls_all,
        'gate_params_all': gate_params_all,
        'cluster_spacing': cluster_spacing,
        'manual_parallelization': manual_parallelization,
        'override_local': override_local,
        'is_expt_data': is_expt_data,
        'timestamp': timestamp,
        'preset_opt': preset_opt,
        'start_Delta_from0': start_Delta_from0,
        'phi_mode': phi_mode,
        'type': 'all_single_hams_rand'
    }
    if not backwards_compatible_muid:
        manager_params['Delta_local_ramp_time'] = Delta_local_ramp_time
        manager_params['Omega_delay_time'] = Omega_delay_time
    if x0_y0_offset != (0,0):
        manager_params['x0_y0_offset'] = x0_y0_offset
    if t_delay is not None:
        manager_params['t_delay'] = t_delay
    if not uniform_Omega_Delta_ramp:
        manager_params['uniform_Omega_Delta_ramp'] = uniform_Omega_Delta_ramp
    if specify_ensemble is not None:
        manager_params['specify_ensemble'] = specify_ensemble
    if not ignore_seq_for_muid:
        manager_params["seq_ls_pre_all"] = seq_ls_pre_all ## ADDED FEB 3 2026']

    muid, added = manager.add(manager_params, timestamp=timestamp)

    os.makedirs(os.path.join(dir_root, "mdata"), exist_ok=True) # make sure the mdata directory exists
    os.makedirs(os.path.join(dir_root, "data", name), exist_ok=True) # make sure the mdata directory exists
    mdata_filename = os.path.join(dir_root, "mdata", f"{muid}.npy")

    if not os.path.exists(mdata_filename) or force_recompute:
        # print("computing all single hams rand")
        data_master_results = np.zeros((len(Delta_mean_ls_all), len(Delta_local_ls_all)), dtype=object) # shape: (Delta_mean, Delta_local)
        for i, Delta_mean in enumerate(Delta_mean_ls_all):
            for j, Delta_local in enumerate(Delta_local_ls_all):
                gate_params = gate_params_all[i][j]
        
                x_h_ensemble = []
                ham_count = 0

                for h_index, (x, h_ls) in enumerate(zip(x_pre, h_ls_pre)):
                    if specify_ensemble_set is not None and h_index not in specify_ensemble_set:
                        continue

                    if backup_dirs is not None:
                        backup_dir_t_s = backup_dirs[i][j][h_index]
                    else:
                        backup_dir_t_s = None

                    # print("backup_dir_t_s", backup_dir_t_s)
                   
                    print("Delta mean progress:", i/len(Delta_mean_ls_all))
                    print("Delta local progress:", j/len(Delta_local_ls_all))
                    print("ensemble progress", h_index / len(h_ls_pre))

                    seq_ls_pre = seq_ls_pre_all[i][j][h_index]

                    Delta_local_h = _resolve_per_ensemble_value(Delta_local, h_index, len(h_ls_pre))
                    gate_params_h = deepcopy(gate_params)
                    if "Delta_local" in gate_params_h:
                        gate_params_h["Delta_local"] = _resolve_per_ensemble_value(gate_params_h["Delta_local"], h_index, len(h_ls_pre))
              
                    data_all_times_single_ham = get_single_ham_rand(h_ls, x, t_plateau_ls, seq_ls_pre, base_params, Delta_mean, Delta_local_h, gate_params_h, cluster_spacing, manual_parallelization, is_expt_data, timestamp, dir_root, debug=debug, check_postarray=check_postarray, preset_opt=preset_opt, data_subdir=name, override_local=override_local, save_mode=save_mode, backup_dir_t_s=backup_dir_t_s, x0_y0_offset=x0_y0_offset, t_delay=t_delay, start_Delta_from0=start_Delta_from0, uniform_Omega_Delta_ramp=uniform_Omega_Delta_ramp, phi_mode=phi_mode, Delta_local_ramp_time=Delta_local_ramp_time, Omega_delay_time=Omega_delay_time, sample_Delta_local_each_t_seq=sample_Delta_local_each_t_seq) # per time, per seq
                    ham_count += 1
                    if ham_count % after_how_many_ham_run_check == 0:
                        ct = int(time.time())
                        print(f"Running diagnose check at {ct}...")
                        # diagnose_main(dir_root = "lucky_turtle_calib", override = True, mode="RUN", suid_filename = None, is_expt_data=is_expt_data)

                        if not is_expt_data:
                            ct = 0

                        calib_name, calib_seq_ls_pre_all, calib_base_params, calib_params, calib_Delta_mean, calib_Delta_local, calib_t_plat_ls, calib_x0_y0_offset = get_calib_task(dir_root=ham_check_dir_main, ham_count=ham_count,ct=ct)

                        calib_params = calib_params[0][0] # only one set of gate params
                        calib_seq_ls_pre_all = calib_seq_ls_pre_all[0][0][0] # only one set of sequences

                        get_single_ham_rand(h_ls=[1], x=[(0,0)], t_plateau_ls=calib_t_plat_ls, seq_ls_pre=calib_seq_ls_pre_all, base_params=calib_base_params,Delta_mean=calib_Delta_mean, Delta_local=calib_Delta_local, gate_params=calib_params, cluster_spacing=None, manual_parallelization=False, is_expt_data=is_expt_data, timestamp=ct, dir_root=ham_check_dir_main,debug=False, check_postarray=False, preset_opt=None, data_subdir=calib_name, override_local=override_local, save_mode=False, backup_dir_t_s=None, x0_y0_offset = calib_x0_y0_offset, t_delay=t_delay, start_Delta_from0=start_Delta_from0, uniform_Omega_Delta_ramp=uniform_Omega_Delta_ramp, Delta_local_ramp_time=Delta_local_ramp_time, Omega_delay_time=Omega_delay_time, sample_Delta_local_each_t_seq=False)
                        
                data_master_results[i, j] = np.array(x_h_ensemble) # returns list of t_plateau, seq, bitstrings

        # save data_master_results with muid as key
        np.save(mdata_filename, data_master_results, allow_pickle=True)

    else:
        data_master_results = np.load(mdata_filename, allow_pickle=True)

    return muid, data_master_results

def get_all_single_hams_rand(h_ls_pre, x_pre, t_plateau_ls, seq_ls_pre_all, base_params, Delta_mean_ls_all, Delta_local_ls_all, gate_params_all, cluster_spacing, manual_parallelization, is_expt_data, timestamp, dir_root, force_recompute=False, debug=False,check_postarray=False, preset_opt=None, name = "", override_local=False, save_mode=True, backup_dirs=None, x0_y0_offset = (0,0), specify_ensemble=None, t_delay=None, start_Delta_from0=False, uniform_Omega_Delta_ramp=False, phi_mode='binary', save_sample_hist=False, Delta_local_ramp_time=0.05, Omega_delay_time=None, ignore_seq_for_muid=False, backwards_compatible_muid=False, skip_store=False, sample_Delta_local_each_t_seq=False):

    ## name is: descriptor of task_taskid_timestamp_preset_opt
    # print("sequences pre all", seq_ls_pre_all)

    print("SPECIFY ENSEMBLE", specify_ensemble)

    # print('get all', check_postarray)
    assert not (is_expt_data and timestamp == 0), "timestamp should not be 0 for experiment data"

    if not is_expt_data:
        timestamp = 0

    if not is_expt_data:
        save_mode = False # calculate directly if numerical

    specify_ensemble_set = set(specify_ensemble) if specify_ensemble is not None else None

    muid = None
    mdata_filename = None

    if not skip_store:
        manager = ExptStore(dir_root)
        # don't include seq_ls_pre_all or timestamp since we want to compile the data for different independent runs on difference accounts
        manager_params = { 
            'h_ls_pre': h_ls_pre,
            'x_pre': x_pre,
            't_plateau_ls': t_plateau_ls.tolist() if isinstance(t_plateau_ls, np.ndarray) else t_plateau_ls,
            'base_params': base_params,
            'Delta_mean_ls_all': Delta_mean_ls_all,
            'Delta_local_ls_all': Delta_local_ls_all,
            'gate_params_all': gate_params_all,
            'cluster_spacing': cluster_spacing,
            'manual_parallelization': manual_parallelization,
            'override_local': override_local,
            'is_expt_data': is_expt_data,
            'timestamp': timestamp,
            'preset_opt': preset_opt,
            'start_Delta_from0': start_Delta_from0,
            'phi_mode': phi_mode,
            "name": name,
            'type': 'all_single_hams_rand'
        }
        if not backwards_compatible_muid:
            manager_params['Delta_local_ramp_time'] = Delta_local_ramp_time
            manager_params['Omega_delay_time'] = Omega_delay_time

        if x0_y0_offset != (0,0):
            manager_params['x0_y0_offset'] = x0_y0_offset
        if specify_ensemble is not None:
            manager_params['specify_ensemble'] = specify_ensemble
        if t_delay is not None:
            manager_params['t_delay'] = t_delay
        if not uniform_Omega_Delta_ramp:
            manager_params['uniform_Omega_Delta_ramp'] = uniform_Omega_Delta_ramp
        if not ignore_seq_for_muid:
            manager_params["seq_ls_pre_all"] = seq_ls_pre_all ## ADDED FEB 3 2026']

        print(manager_params)

        muid, added = manager.add(manager_params, timestamp=0)
        print("MUID", muid)

        os.makedirs(os.path.join(dir_root, "mdata"), exist_ok=True) # make sure the mdata directory exists
        os.makedirs(os.path.join(dir_root, "data", name), exist_ok=True)
        mdata_filename = os.path.join(dir_root, "mdata", f"{muid}.npy")
    else:
        os.makedirs(os.path.join(dir_root, "data", name), exist_ok=True)
        muid = f"skip_store_{time.time_ns()}"

    if skip_store or not os.path.exists(mdata_filename) or force_recompute:
        # print("computing all single hams rand")
        data_master_results = np.zeros((len(Delta_mean_ls_all), len(Delta_local_ls_all)), dtype=object) # shape: (Delta_mean, Delta_local)
        for i, Delta_mean in enumerate(Delta_mean_ls_all):
            for j, Delta_local in enumerate(Delta_local_ls_all):
                gate_params = gate_params_all[i][j]
        
                # Initialize x_h_ensemble as numpy array with full size, filled with None
                x_h_ensemble = np.full(len(h_ls_pre), None, dtype=object)
               
                for h_index, (x, h_ls) in enumerate(zip(x_pre, h_ls_pre)):
                    run_this_h = True
                    
                    if specify_ensemble_set is not None:
                        run_this_h = h_index in specify_ensemble_set
                        print("SPECIFY ENSEMBLE IS NOT NONE, h_index", h_index, "run_this_h", run_this_h)

                    if backup_dirs is not None:
                        backup_dir_t_s = backup_dirs[i][j][h_index]
                    else:
                        backup_dir_t_s = None

                    # print("backup_dir_t_s", backup_dir_t_s)
                    if run_this_h:
                    
                        print("Delta mean progress:", i/len(Delta_mean_ls_all))
                        print("Delta local progress:", j/len(Delta_local_ls_all))
                        print("ensemble progress", h_index / len(h_ls_pre))

                        # print(f"seq_ls at {i}, {j}", seq_ls_pre_all[i][j], len(seq_ls_pre_all[i][j]))
                        # print("h_ls index", h_index)

                        seq_ls_pre = seq_ls_pre_all[i][j][h_index]

                        Delta_local_h = _resolve_per_ensemble_value(Delta_local, h_index, len(h_ls_pre))
                        gate_params_h = deepcopy(gate_params)
                        if "Delta_local" in gate_params_h:
                            gate_params_h["Delta_local"] = _resolve_per_ensemble_value(gate_params_h["Delta_local"], h_index, len(h_ls_pre))
                
                        data_all_times_single_ham = get_single_ham_rand(h_ls, x, t_plateau_ls, seq_ls_pre, base_params, Delta_mean, Delta_local_h, gate_params_h, cluster_spacing, manual_parallelization, is_expt_data, timestamp, dir_root, debug=debug, check_postarray=check_postarray, preset_opt=preset_opt, data_subdir=name, override_local=override_local, save_mode=save_mode, backup_dir_t_s=backup_dir_t_s, x0_y0_offset=x0_y0_offset, t_delay=t_delay, start_Delta_from0=start_Delta_from0, uniform_Omega_Delta_ramp=uniform_Omega_Delta_ramp, phi_mode=phi_mode, Delta_local_ramp_time=Delta_local_ramp_time, Omega_delay_time=Omega_delay_time, sample_Delta_local_each_t_seq=sample_Delta_local_each_t_seq) # per time, per seq
                        
                        # Generate sample histograms if requested
                        if save_sample_hist and len(data_all_times_single_ham) > 0:
                            _save_sample_hist(data_all_times_single_ham, t_plateau_ls, seq_ls_pre, 
                                                muid, Delta_mean, Delta_local_h, is_expt_data, 
                                                timestamp, dir_root, h_index)
                        
                        # Store data at the correct index corresponding to the original ensemble index
                        x_h_ensemble[h_index] = data_all_times_single_ham

                data_master_results[i, j] = x_h_ensemble # returns array of t_plateau, seq, bitstrings with correct indexing

    
        # save data_master_results with muid as key
        if not skip_store:
            np.save(mdata_filename, data_master_results, allow_pickle=True)

    else:
        data_master_results = np.load(mdata_filename, allow_pickle=True)

    return muid, data_master_results

def extract_value(arr, idx_tuple):
        """
        helper func to deal with the epsilon_r_ens

        arr: scalar or array
        idx_tuple: e.g. (i, j) or (i, j, l)
        Returns:
        - float if scalar
        - list with nested arrays converted to lists if composite
        """
        def convert_nested_arrays_to_lists(obj):
            """Recursively convert nested arrays to lists"""
            if hasattr(obj, "ndim") and obj.ndim > 0:
                # Convert array to list and recursively process elements
                return [convert_nested_arrays_to_lists(item) for item in obj]
            elif isinstance(obj, (list, tuple)):
                # Recursively process list/tuple elements
                return [convert_nested_arrays_to_lists(item) for item in obj]
            else:
                # Return scalar as-is
                return obj
        
        # case 1: arr is scalar
        if not hasattr(arr, "ndim") or arr.ndim == 0:
            return float(arr)

        # case 2: array
        idx = idx_tuple[:arr.ndim]
        try:
            val = arr[idx]
        except IndexError:
            print("ERROR, RETURNING 0", idx_tuple)
            raise IndexError("Index out of bounds in extract_value")

        # val may be numpy scalar, python scalar, list, arr...
        if isinstance(val, (list, tuple)):
            return convert_nested_arrays_to_lists(val)  # convert nested arrays to lists
        if hasattr(val, "ndim") and val.ndim > 0:
            return convert_nested_arrays_to_lists(val)  # convert nested arrays to lists                     

        # Otherwise scalar -> convert to float
        return float(val)
    
def get_all_qutip_probs(h_ls_pre, x_pre, t_plateau_ls, seq_ls_pre_all, base_params, Delta_mean_ls_all, Delta_local_ls_all, gate_params_all, dir_root, force_recompute=False, neg_phi=False, preset_opt=None, override_local=False, specify_ensemble=None, include_T2=False,  start_Delta_from0=False, uniform_Omega_Delta_ramp=False, phi_mode='binary', local_haar=False, indep_haar=True,Delta_local_ramp_time=0.05, Omega_delay_time=0, ignore_seq_for_muid=False, skip_store=False, sample_Delta_local_each_t_seq=False, per_qubit_delta_local=True):
    """
    cluster_spacing is currently not used

    per_qubit_delta_local default changed to True
    """
    specify_ensemble_set = set(specify_ensemble) if specify_ensemble is not None else None

    # don't include seq_ls_pre_all or timestamp since we want to compile the data for different independent runs on difference accounts
    # could put an exclusion rule for certain timestamps or users

    # print("get all qutip", seq_ls_pre_all)

    # manager_params = { 
    #     'h_ls_pre': h_ls_pre if specify_ensemble is None else [h_ls_pre[i] for i in specify_ensemble],
    #     'x_pre': x_pre,
    #     'seq_ls_pre_all': seq_ls_pre_all,
    #     't_plateau_ls': t_plateau_ls,
    #     'base_params': base_params,
    #     'Delta_mean_ls_all': Delta_mean_ls_all,
    #     'Delta_local_ls_all': Delta_local_ls_all,
    #     'gate_params_all': gate_params_all,
    #     "preset_opt": preset_opt,
    #     'override_local': override_local,
    #     "open_dynamics": open_dynamics,
    #     "num_repeats_coherent_noise": num_repeats_coherent_noise if open_dynamics else None,
    #     'start_Delta_from0': start_Delta_from0,
    #     'phi_mode': phi_mode,
    #     'local_haar': local_haar,
    #     'type': 'get_probs_seq_ls_pre_all_qutip'
    # }
    # if not uniform_Omega_Delta_ramp:
    #     manager_params['uniform_Omega_Delta_ramp'] = uniform_Omega_Delta_ramp
    # print(base_params)
    manager_params = { 
        'h_ls_pre': h_ls_pre,
        'x_pre': x_pre,
        't_plateau_ls': t_plateau_ls.tolist() if isinstance(t_plateau_ls, np.ndarray) else t_plateau_ls,
        'base_params': base_params,
        'Delta_mean_ls_all': Delta_mean_ls_all,
        'Delta_local_ls_all': Delta_local_ls_all,
        'gate_params_all': gate_params_all,
        'local_haar': local_haar,
        'override_local': override_local,
        'preset_opt': preset_opt,
        'start_Delta_from0': start_Delta_from0,
        'phi_mode': phi_mode,
        'specify_ensemble': specify_ensemble,
        'include_T2': include_T2,
        'Delta_local_ramp_time': Delta_local_ramp_time,
        'Omega_delay_time': Omega_delay_time,
        'per_qubit_delta_local': per_qubit_delta_local,
        'type': 'qutip'
    }
    if not indep_haar:
        manager_params['indep_haar'] = indep_haar

    # print("manager", manager_params)

    # raise NotImplementedError

    muid = None
    mdata_filename = None
    if not skip_store:
        manager = ExptStore(dir_root)

        if not uniform_Omega_Delta_ramp:
            manager_params['uniform_Omega_Delta_ramp'] = uniform_Omega_Delta_ramp

        muid, added = manager.add(manager_params, timestamp=0)

        os.makedirs(os.path.join(dir_root, "mdata"), exist_ok=True) # make sure the qutip directory exists

        mdata_filename = os.path.join(dir_root, "mdata", f"{muid}.npy")

        print("DOES mdata file exist?", os.path.exists(mdata_filename), 'mduid', muid)

    def _extract_single_seq_result(single_result):
        if single_result is None:
            return None

        if isinstance(single_result, np.ndarray) and single_result.ndim == 1:
            return single_result

        if isinstance(single_result, (list, tuple, np.ndarray)) and len(single_result) > 0:
            first_t = single_result[0]
            if isinstance(first_t, (list, tuple, np.ndarray)) and len(first_t) > 0:
                return first_t[0]
            return first_t

        return single_result

    if skip_store or not os.path.exists(mdata_filename) or force_recompute:
        data_master_results = np.zeros((len(Delta_mean_ls_all), len(Delta_local_ls_all)), dtype=object) # shape: (Delta_mean, Delta_local)
        for i, Delta_mean in enumerate(Delta_mean_ls_all):
            for j, Delta_local in enumerate(Delta_local_ls_all):
                gate_params = gate_params_all[i][j]
                # print("inside get all qutip", seq_ls_pre_all[i][j])

                try:
                    Omega_ij = base_params['ev_params']['Omega_ens'][i][j]
                    
                    
                    def safe_abs(val):
                        if isinstance(val, (list, tuple, np.ndarray)):
                            return np.mean([safe_abs(v) for v in val])
                        else:
                            return abs(val)
                    
                    def is_nonzero(val, threshold=1e-6):
                        if isinstance(val, (list, tuple, np.ndarray)):
                            return any(is_nonzero(v, threshold) for v in val)
                        else:
                            return abs(val) > threshold
                    
                    # Calculate average of nonzero Omega values
                    nonzero_values = [safe_abs(om) for om in Omega_ij if is_nonzero(om)]
                    if nonzero_values:
                        avg_nonzero_Omega = np.mean(nonzero_values)
                    else:
                        avg_nonzero_Omega = 15.8  # Default fallback value
                    
                    # Replace any zero Omega with avg_nonzero_Omega
                    Omega_ij = [om if is_nonzero(om) else avg_nonzero_Omega for om in Omega_ij]
                    print("Extracted Omega_ij:", Omega_ij)
                except Exception as e:
                    print("Error extracting Omega_ij:", e)
                    Omega_ij = None
                    

        
                # Initialize x_h_ensemble as numpy array with full size, filled with None
                x_h_ensemble = np.full(len(h_ls_pre), None, dtype=object)
               
                for h_idx, (x, h_ls) in enumerate(zip(x_pre, h_ls_pre)):
                    run_this_h = True
                    
                    if specify_ensemble_set is not None:
                        run_this_h = h_idx in specify_ensemble_set

                    if run_this_h:
                   
                        print("Delta mean progress:", i/len(Delta_mean_ls_all))
                        print("Delta local progress:", j/len(Delta_local_ls_all))
                        print("ensemble progress", h_idx / len(h_ls_pre))

                        seq_ls_pre = seq_ls_pre_all[i][j][h_idx]

                        Delta_local_h = _resolve_per_ensemble_value(Delta_local, h_idx, len(h_ls_pre))
                        gate_params_h = deepcopy(gate_params)
                        if "Delta_local" in gate_params_h:
                            gate_params_h["Delta_local"] = _resolve_per_ensemble_value(gate_params_h["Delta_local"], h_idx, len(h_ls_pre))

                        if Omega_ij is not None:
                            print("Omega_ij", Omega_ij)
                            print("len Omega_ij", len(Omega_ij), "h_idx", h_idx)
                            print("len h_ls", len(h_ls))
                            # assert len(Omega_ij[h_idx]) == len(h_ls), f"Length mismatch: len(Omega_ij)={len(Omega_ij)} vs len(h_ls)={len(h_ls)}"
                            try:
                                base_params['Omega'] = Omega_ij[h_idx] ## expect list containing one element for each qubit
                                print("Setting Omega to", base_params['Omega'], "for h index", h_idx)
                            except Exception as e:
                                print("Error setting Omega for h index", h_idx, ":", e)
                                base_params['Omega'] = Omega_ij[0]  # Fallback to first element
                                print("Falling back to Omega", base_params['Omega'])
                            # if there are any zeros in Omega_ij[h_idx], replace with the average of the non-zero Omegas
                            if isinstance(base_params['Omega'], (list, np.ndarray)):
                                Omega_list = base_params['Omega']
                                nonzero_Omegas = [abs(om) for om in Omega_list if abs(om) > 1e-6]
                                if len(nonzero_Omegas) > 0:
                                    avg_Omega = np.mean(nonzero_Omegas)
                                else:
                                    raise ValueError("All Omega values are zero")
                                Omega_list = [om if abs(om) > 1e-6 else avg_Omega for om in Omega_list]
                                base_params['Omega'] = Omega_list
                                print("Corrected Omega list:", base_params['Omega'])

                        

                        if sample_Delta_local_each_t_seq:
                            data_all_times_single_ham = []
                            for t_idx, t_plateau in enumerate(t_plateau_ls):
                                seq_ls = seq_ls_pre[t_idx]
                                data_t_fixed_ls = []

                                if type(seq_ls) is list:
                                    for seq_idx, seq in enumerate(seq_ls):
                                        evol_delta_local_base = _resolve_time_seq_delta_local(Delta_local_h, t_idx, len(t_plateau_ls), seq_idx=seq_idx, n_seqs=len(seq_ls))
                                        gate_delta_local_base = _resolve_time_seq_delta_local(gate_params_h.get("Delta_local", evol_delta_local_base), t_idx, len(t_plateau_ls), seq_idx=seq_idx, n_seqs=len(seq_ls))

                                        # Use per-qubit Delta_local if enabled, otherwise use scalar
                                        if per_qubit_delta_local:
                                            # raise NotImplementedError
                                            evol_delta_local_disorder = _sample_delta_local_disorder_per_qubit(evol_delta_local_base, h_ls)
                                            gate_delta_local_disorder = _sample_delta_local_disorder_per_qubit(gate_delta_local_base, h_ls)
                                        else:
                                            evol_delta_local_disorder = _sample_delta_local_disorder(evol_delta_local_base)
                                            gate_delta_local_disorder = _sample_delta_local_disorder(gate_delta_local_base)

                                        gate_params_seq = deepcopy(gate_params_h)
                                        gate_params_seq["Delta_local"] = gate_delta_local_disorder

                                        single_result = get_probs_seq_ls(h_ls, x, [t_plateau], [[seq]], base_params, Delta_mean, evol_delta_local_disorder, gate_params_seq, neg_phi=neg_phi, preset_opt=preset_opt, override_local=override_local, start_Delta_from0=start_Delta_from0, uniform_Omega_Delta_ramp=uniform_Omega_Delta_ramp, phi_mode=phi_mode, local_haar=local_haar,indep_haar = indep_haar, include_T2=include_T2, Delta_local_ramp_time=Delta_local_ramp_time, Omega_delay_time=Omega_delay_time)
                                        single_seq_result = _extract_single_seq_result(single_result)
                                        if single_seq_result is not None:
                                            data_t_fixed_ls.append(single_seq_result)
                                else:
                                    evol_delta_local_base = _resolve_time_seq_delta_local(Delta_local_h, t_idx, len(t_plateau_ls))
                                    gate_delta_local_base = _resolve_time_seq_delta_local(gate_params_h.get("Delta_local", evol_delta_local_base), t_idx, len(t_plateau_ls))

                                    # Use per-qubit Delta_local if enabled, otherwise use scalar
                                    if per_qubit_delta_local:
                                        evol_delta_local_disorder = _sample_delta_local_disorder_per_qubit(evol_delta_local_base, h_ls)
                                        gate_delta_local_disorder = _sample_delta_local_disorder_per_qubit(gate_delta_local_base, h_ls)
                                    else:
                                        evol_delta_local_disorder = _sample_delta_local_disorder(evol_delta_local_base)
                                        gate_delta_local_disorder = _sample_delta_local_disorder(gate_delta_local_base)

                                    gate_params_seq = deepcopy(gate_params_h)
                                    gate_params_seq["Delta_local"] = gate_delta_local_disorder

                                    single_result = get_probs_seq_ls(h_ls, x, [t_plateau], [[]], base_params, Delta_mean, evol_delta_local_disorder, gate_params_seq, neg_phi=neg_phi, preset_opt=preset_opt, override_local=override_local, start_Delta_from0=start_Delta_from0, uniform_Omega_Delta_ramp=uniform_Omega_Delta_ramp, phi_mode=phi_mode, local_haar=local_haar, indep_haar=indep_haar, include_T2=include_T2, Delta_local_ramp_time=Delta_local_ramp_time, Omega_delay_time=Omega_delay_time)
                                    single_seq_result = _extract_single_seq_result(single_result)
                                    if single_seq_result is not None:
                                        data_t_fixed_ls.append(single_seq_result)

                                data_all_times_single_ham.append(data_t_fixed_ls)
                        else:
                            # For the non-sampled case, optionally apply per-qubit Delta_local sampling
                            if per_qubit_delta_local:
                                Delta_local_h_list = _sample_delta_local_disorder_per_qubit(Delta_local_h, h_ls)
                                gate_params_h_copy = deepcopy(gate_params_h)
                                if "Delta_local" in gate_params_h_copy:
                                    gate_params_h_copy["Delta_local"] = _sample_delta_local_disorder_per_qubit(gate_params_h_copy["Delta_local"], h_ls)
                            else:
                                Delta_local_h_list = Delta_local_h
                                gate_params_h_copy = gate_params_h
                            
                            data_all_times_single_ham = get_probs_seq_ls(h_ls, x, t_plateau_ls, seq_ls_pre, base_params, Delta_mean, Delta_local_h_list, gate_params_h_copy, neg_phi=neg_phi, preset_opt=preset_opt, override_local=override_local, start_Delta_from0=start_Delta_from0, uniform_Omega_Delta_ramp=uniform_Omega_Delta_ramp, phi_mode=phi_mode, local_haar=local_haar, indep_haar=indep_haar, include_T2=include_T2, Delta_local_ramp_time=Delta_local_ramp_time, Omega_delay_time=Omega_delay_time) # per time, per seq
                        # else:
                        #     data_all_times_single_ham = get_probs_seq_ls_noisy(h_ls, x, t_plateau_ls, seq_ls_pre, base_params, Delta_mean, Delta_local, gate_params, neg_phi=neg_phi, preset_opt=preset_opt, override_local=override_local, n_repeats = num_repeats_coherent_noise, start_Delta_from0=start_Delta_from0, uniform_Omega_Delta_ramp=uniform_Omega_Delta_ramp, phi_mode=phi_mode) # per time, per seq

                        # Store data at the correct index corresponding to the original ensemble index
                        x_h_ensemble[h_idx] = data_all_times_single_ham 

                # data_master_results[i, j] = np.array(x_h_ensemble) # returns list of t_plateau, seq, bitstrings
                data_master_results[i, j] = x_h_ensemble

        # save data_master_results with muid as key
        if not skip_store:
            np.save(mdata_filename, data_master_results, allow_pickle=True)

    else:
        data_master_results = np.load(mdata_filename, allow_pickle=True)

    if muid is not None:
        return muid, data_master_results
    else:
        return data_master_results

def process_bitstrings(h_ls_pre, x_pre, t_plateau_ls, seq_ls_pre_all, base_params, Delta_mean_ls_all, Delta_local_ls_all, gate_params_all, cluster_spacing, manual_parallelization, is_expt_data, timestamp, dir_root, name, process_opt = "ee", force_recompute=False, force_recompute_processing=False, same_U_all_time=False, is_bloqade=True, epsilon_r_ens = 0.0, epsilon_g_ens = 0.0, apply_correction = False, after_how_many_ham_run_check=None, ham_check_dir_main =None, specify_ensemble=None, shot_noise_model="multinomial", t_delay=None, start_Delta_from0=False, uniform_Omega_Delta_ramp=False, phi_mode='binary', save_sample_hist=False, total_system=False, override_local=False, Delta_local_ramp_time=0.05, Omega_delay_time=None, ignore_seq_for_muid=False, backwards_compatible_muid=False, muid_data_master_results_override=None, include_T2star=False, include_T2=False, seed=42, T2star_workers=None, haar_opt=None):
    """
    returns estimate of the quantity specified by "process_opt" : either "ee" for entanglement entropy, "sp" for survival probability, or "raw" for raw bitstrings

    correction is applied for readout error only if not is_bloqade and apply_correction is True

    """

    print('-'*20)
    print("ERROR MODELING:")
    print("\nUse T2star = ", include_T2star)
    print("\nUse T2 = ", include_T2)
    print('-'*20)

    # Set random seed for reproducibility
    if seed is not None:
        np.random.seed(seed)

    print("in process bitstrings, epsilon_r_ens", epsilon_r_ens, "epsilon_g_ens", epsilon_g_ens, "apply_correction", apply_correction, "shot_noise_model", shot_noise_model)
    

    assert haar_opt in [None, 'local-indep', 'local-same']
    if haar_opt is not None:
        local_haar = True
        indep_haar = (haar_opt == 'local-indep')
    else:
        local_haar = False 
        indep_haar = False

  

    print("local_haar", local_haar, "indep_haar", indep_haar)

    assert process_opt in ["ee", "sp", "raw"], f"process_opt must be either 'ee', 'sp', or 'raw', got {process_opt}"

    assert not (apply_correction and is_expt_data), "silly, don't apply correction to actual data"

    if not apply_correction:
        epsilon_r_ens = 0.0
        epsilon_g_ens = 0.0

    def _t2_star(delta_local):
        return (1 / 4.67 + np.abs(delta_local) / (2 * np.pi * 5.9)) ** (-1)

    def _combine_data_master_results(data_master_results_all):
        if not data_master_results_all:
            return None

        valid_results = [data_mr for data_mr in data_master_results_all if data_mr is not None]
        if not valid_results:
            return None

        def _is_count_report(elem):
            if not isinstance(elem, (list, tuple, np.ndarray)) or len(elem) == 0:
                return False

            first = elem[0]
            if isinstance(first, np.ndarray):
                return first.ndim == 1 and first.size == 2

            if isinstance(first, (list, tuple)) and len(first) == 2:
                return np.isscalar(first[0]) and np.isscalar(first[1])

            return False

        def _is_probability_vector(elem):
            if isinstance(elem, np.ndarray):
                return elem.ndim >= 1 and np.issubdtype(elem.dtype, np.number)

            if isinstance(elem, (list, tuple)) and len(elem) > 0:
                return all(np.isscalar(v) for v in elem)

            return False

        def _is_container(elem):
            if elem is None:
                return False

            if _is_count_report(elem) or _is_probability_vector(elem):
                return False

            if isinstance(elem, np.ndarray):
                return elem.ndim > 0

            return isinstance(elem, (list, tuple))

        def _merge_shot_elements(shot_elements):
            if not shot_elements:
                return None

            if _is_count_report(shot_elements[0]):
                merged_counts = {}
                for elem in shot_elements:
                    if elem is None:
                        continue
                    for basis_state_idx, count in elem:
                        basis_state_idx = int(basis_state_idx)
                        count = int(count)
                        merged_counts[basis_state_idx] = merged_counts.get(basis_state_idx, 0) + count
                return sorted(merged_counts.items(), key=lambda x: x[1], reverse=True)

            # qutip/probability path: treat each shot element as one sampled outcome,
            # then merge counts like Bloqade. If downstream expects probabilities
            # (is_bloqade=False), return empirical probabilities from merged counts.
            merged_counts = {}
            total_samples = 0
            state_dim = None

            for elem in shot_elements:
                if elem is None:
                    continue

                arr = np.asarray(elem, dtype=float).reshape(-1)
                if arr.size == 0:
                    continue
                if state_dim is None:
                    state_dim = int(arr.size)
                elif int(arr.size) != state_dim:
                    raise ValueError(f"Incompatible shot element shapes in T2* combine: {(state_dim,)} vs {arr.shape}")

                if arr.size == 1:
                    sampled_idx = 0
                else:
                    prob_sum = float(np.sum(arr))
                    if not np.isfinite(prob_sum) or prob_sum <= 0:
                        continue
                    probs = arr / prob_sum
                    sampled_idx = int(np.random.choice(arr.size, p=probs)) ## simulate a shot

                merged_counts[sampled_idx] = merged_counts.get(sampled_idx, 0) + 1
                total_samples += 1

            if total_samples == 0:
                return None

            merged_tuples = sorted(merged_counts.items(), key=lambda x: x[1], reverse=True)

            if is_bloqade:
                return merged_tuples

            empirical_probs = np.zeros(state_dim, dtype=float)
            for basis_state_idx, count in merged_tuples:
                empirical_probs[basis_state_idx] = count / float(total_samples)

            return empirical_probs

        def _merge_nested(shot_elements):
            valid_elements = [elem for elem in shot_elements if elem is not None]
            if not valid_elements:
                return None

            base_elem = valid_elements[0]

            if not _is_container(base_elem):
                return _merge_shot_elements(valid_elements)

            base_list = list(base_elem)
            looks_like_time_axis = len(base_list) == len(t_plateau_ls)
            merged_children = []

            for idx in range(len(base_list)):
                child_shot_elements = []
                for elem in valid_elements:
                    try:
                        child = elem[idx]
                    except (IndexError, TypeError):
                        continue
                    if child is not None:
                        child_shot_elements.append(child)

                if not child_shot_elements:
                    merged_children.append(None)
                    continue

                merged_child = _merge_nested(child_shot_elements)
                base_child = base_list[idx]

                if looks_like_time_axis and _is_probability_vector(base_child):
                    merged_children.append([merged_child])
                else:
                    merged_children.append(merged_child)

            return merged_children

        shape = valid_results[0].shape
        combined = np.empty(shape, dtype=object)

        for i in range(shape[0]):
            for j in range(shape[1]):
                shot_entries = []
                for data_mr in valid_results:
                    try:
                        entry = data_mr[i, j]
                    except (IndexError, TypeError):
                        continue
                    if entry is not None:
                        shot_entries.append(entry)

                combined[i, j] = _merge_nested(shot_entries)

        return combined

    def _resolve_T2star_workers(n_shots, force_serial=False):
        if n_shots <= 1 or force_serial:
            return 1

        if T2star_workers is not None:
            try:
                parsed_workers = int(T2star_workers)
            except (TypeError, ValueError):
                raise ValueError(f"T2star_workers must be None or an integer, got {T2star_workers}")
            return max(1, min(parsed_workers, n_shots))

        if is_expt_data:
            return 1

        cpu_count = os.cpu_count() or 1
        return min(n_shots, max(1, cpu_count - 1))

    def _prepare_t2star_shot_inputs(shot_idx):
        gate_params_shot = deepcopy(gate_params_all)
        Delta_local_ls_all_shot = deepcopy(Delta_local_ls_all)

        num_h_ensembles = len(h_ls_pre)

        for i in range(len(Delta_mean_ls_all)):

            for j in range(len(Delta_local_ls_all)):
                base_delta_local = gate_params_all[0][j].get("Delta_local", Delta_local_ls_all[j])
                delta_local_disorder_h = []
                evol_delta_disorder_h = []

                for h_idx in range(num_h_ensembles):
                    base_delta_local_h = _resolve_per_ensemble_value(base_delta_local, h_idx, num_h_ensembles)
                    t2_star_gate = _t2_star(base_delta_local_h)
                    delta_local_disorder_h.append(np.random.normal(loc=base_delta_local_h, scale=np.sqrt(2) / t2_star_gate))

                    evol_delta_local_h = _resolve_per_ensemble_value(Delta_local_ls_all[j], h_idx, num_h_ensembles)
                    evol_delta_t2star = _t2_star(evol_delta_local_h)
                    evol_delta_disorder_h.append(np.random.normal(loc=evol_delta_local_h, scale=np.sqrt(2) / evol_delta_t2star))

                Delta_local_ls_all_shot[j] = evol_delta_disorder_h

                gate_params_shot[i][j]["Delta_local"] = delta_local_disorder_h
                gate_params_shot[i][j]["n_shots"] = 1

        base_params_shot = deepcopy(base_params)
        if "ev_params" in base_params_shot and len(Delta_local_ls_all_shot) > 0:
            base_params_shot["ev_params"]["Delta_local"] = Delta_local_ls_all_shot[0]

        return shot_idx, base_params_shot, Delta_local_ls_all_shot, gate_params_shot

    def _run_t2star_shots(shot_runner, force_serial=False):
        n_shots = gate_params_all[0][0].get("n_shots", 1)
        workers = _resolve_T2star_workers(n_shots, force_serial=force_serial)

        print(f"Running T2* averaging with n_shots={n_shots}, workers={workers}")

        shot_inputs = [_prepare_t2star_shot_inputs(n) for n in range(n_shots)]
        data_master_results_all = [None] * n_shots

        if workers == 1:
            for shot_idx, base_params_shot, Delta_local_ls_all_shot, gate_params_shot in shot_inputs:
                print('-' * 50)
                print(f"SHOT {shot_idx+1}/{n_shots}!!!")
                print("base_params_shot", base_params_shot)

                shot_results = shot_runner(base_params_shot, Delta_local_ls_all_shot, gate_params_shot)

                data_master_results_all[shot_idx] = shot_results

                print(f"Completed processing for shot {shot_idx+1}/{n_shots}!!!")
                print('-' * 50)
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_idx = {}
                for shot_idx, base_params_shot, Delta_local_ls_all_shot, gate_params_shot in shot_inputs:
                    print('-' * 50)
                    print(f"Submitting SHOT {shot_idx+1}/{n_shots}...")
                    print("base_params_shot", base_params_shot)
                    future = executor.submit(shot_runner, base_params_shot, Delta_local_ls_all_shot, gate_params_shot)
                    future_to_idx[future] = shot_idx

                completed = 0
                for future in as_completed(future_to_idx):
                    shot_idx = future_to_idx[future]
                    shot_results = future.result()

                    data_master_results_all[shot_idx] = shot_results

                    completed += 1
                    print(f"Completed processing for shot {shot_idx+1}/{n_shots}, ({completed}/{n_shots})!!!")
                    print('-' * 50)

        data_master_results = _combine_data_master_results(data_master_results_all)
        return data_master_results, n_shots, workers
############
    def _get_t2star_muid_candidates(manager_params):
        def _parse_single_index(value):
            if isinstance(value, bool):
                return None
            if isinstance(value, (int, np.integer)):
                return int(value)
            return None

        def _normalize_lookup(value):
            if isinstance(value, range):
                value = list(value)
            elif isinstance(value, np.ndarray):
                value = value.tolist()
            elif isinstance(value, tuple):
                value = list(value)

            if value is None:
                return None

            if not isinstance(value, list):
                value = [value]

            normalized = []
            seen = set()
            for idx in value:
                idx_int = _parse_single_index(idx)
                if idx_int is None:
                    continue
                if idx_int < 0 or idx_int >= len(h_ls_pre):
                    continue
                if idx_int in seen:
                    continue
                seen.add(idx_int)
                normalized.append(idx_int)

            return normalized

        specify_ensemble_lookup = manager_params.get('specify_ensemble')
        specify_ensemble_norm = _normalize_lookup(specify_ensemble_lookup)

        full_ensemble = list(range(len(h_ls_pre)))
        manager_params_candidates = [manager_params]

        if specify_ensemble_norm is None:
            # None means all ensembles — also try the explicit full list.
            alt_manager_params = dict(manager_params)
            alt_manager_params['specify_ensemble'] = full_ensemble
            manager_params_candidates.append(alt_manager_params)
        else:
            # Only add equivalent representations of the SAME request.
            # Proper subsets must NOT be included here: if a subset cache exists,
            # it is an incomplete result and must be handled by
            # _try_load_partial_t2star_cache, not treated as a full hit.
            alt_manager_params = dict(manager_params)
            alt_manager_params['specify_ensemble'] = specify_ensemble_norm
            manager_params_candidates.append(alt_manager_params)

            # Support legacy/alternate storage where [k] may be stored as k.
            if len(specify_ensemble_norm) == 1:
                alt_manager_params_scalar = dict(manager_params)
                alt_manager_params_scalar['specify_ensemble'] = specify_ensemble_norm[0]
                manager_params_candidates.append(alt_manager_params_scalar)

            # If the requested set equals the full ensemble, None is equivalent.
            if specify_ensemble_norm == full_ensemble:
                alt_manager_params = dict(manager_params)
                alt_manager_params['specify_ensemble'] = None
                manager_params_candidates.append(alt_manager_params)

        muid_candidates = []
        for params_candidate in manager_params_candidates:
            payload_candidate = dict(params_candidate)
            payload_candidate['timestamp'] = 0
            muid_candidate = make_uid(payload_candidate)
            if muid_candidate not in muid_candidates:
                muid_candidates.append(muid_candidate)

        return muid_candidates

    def _make_muid_for_ensemble(manager_params, ensemble_indices):
        """Build the muid that would correspond to a specific specify_ensemble value."""
        params = dict(manager_params)
        params['specify_ensemble'] = ensemble_indices
        params['timestamp'] = 0
        return make_uid(params)

    def _find_cached_subset_muids(manager_params):
        """
        For each possible contiguous sub-range of the target ensemble,
        check if a cached .npy exists. Returns a list of (indices, muid, filepath) tuples.
        """
        specify_ensemble = _normalize_specify_ensemble(manager_params.get('specify_ensemble'))
        if specify_ensemble is None:
            specify_ensemble = list(range(len(h_ls_pre)))

        mdata_dir = os.path.join(dir_root, "mdata")
        if not os.path.isdir(mdata_dir):
            return []

        available = []
        n = len(specify_ensemble)

        # Check all contiguous sub-slices of the target ensemble
        for start in range(n):
            for end in range(start + 1, n + 1):
                subset = specify_ensemble[start:end]

                # Try both list and None (None only if subset == full ensemble)
                candidates_to_try = [subset]
                full_ensemble = list(range(len(h_ls_pre)))
                if subset == full_ensemble:
                    candidates_to_try.append(None)

                # Support legacy/alternate storage for single-index subsets.
                if len(subset) == 1 and isinstance(subset[0], (int, np.integer)) and not isinstance(subset[0], bool):
                    candidates_to_try.append(int(subset[0]))

                for ensemble_val in candidates_to_try:
                    muid = _make_muid_for_ensemble(manager_params, ensemble_val)
                    fpath = os.path.join(mdata_dir, f"{muid}.npy")
                    if os.path.exists(fpath):
                        available.append((subset, muid, fpath))
                        break  # No need to check None variant if list variant found

        return available

    def _find_covering_subsets(target_indices, available_subsets):
        """
        Greedy set-cover: find a minimal collection of cached subsets
        that together cover all target_indices.
        available_subsets: list of (indices, muid, filepath)
        Returns list of (indices, muid, filepath) that cover target, or None if impossible.
        """
        target_set = set(target_indices)
        remaining = set(target_set)
        chosen = []

        while remaining:
            # Pick the subset that covers the most remaining indices
            best = None
            best_cover = 0
            for entry in available_subsets:
                subset_indices = set(entry[0])
                # Only allow subsets that don't include indices outside our target
                if not subset_indices <= target_set:
                    continue
                cover = len(subset_indices & remaining)
                if cover > best_cover:
                    best_cover = cover
                    best = entry

            if best is None or best_cover == 0:
                return None  # Can't cover remaining indices

            chosen.append(best)
            remaining -= set(best[0])

        return chosen

    def _try_load_t2star_cache(manager_params, preferred_muid):
        if force_recompute:
            return preferred_muid, None

        os.makedirs(os.path.join(dir_root, "mdata"), exist_ok=True)

        # 1) Try exact match first (original logic)
        muid_candidates = _get_t2star_muid_candidates(manager_params)
        if preferred_muid not in muid_candidates:
            muid_candidates.insert(0, preferred_muid)

        print("MUID candidates to check for T2* cache:", muid_candidates)

        for muid_candidate in muid_candidates:
            mdata_filename = os.path.join(dir_root, "mdata", f"{muid_candidate}.npy")
            print(f"Checking for T2* cache at {mdata_filename} for MUID candidate {muid_candidate}...")
            if os.path.exists(mdata_filename):
                print(f"Loading cached T2* data from {mdata_filename}")
                return muid_candidate, np.load(mdata_filename, allow_pickle=True)

        # 2) Try merging cached subsets
        specify_ensemble = _normalize_specify_ensemble(manager_params.get('specify_ensemble'))
        if specify_ensemble is None:
            specify_ensemble = list(range(len(h_ls_pre)))

        available = _find_cached_subset_muids(manager_params)
        if available:
            covering = _find_covering_subsets(specify_ensemble, available)
            if covering is not None:
                print(f"Found {len(covering)} cached subsets that cover the target ensemble:")
                # Sort by the first index in each subset to merge in order
                covering.sort(key=lambda entry: entry[0][0])

                merged_parts = []
                for subset_indices, muid, fpath in covering:
                    print(f"  Loading subset {subset_indices} from {fpath}")
                    data = np.load(fpath, allow_pickle=True)
                    merged_parts.append((subset_indices, data))

                # Build index mapping: for each target index, find which
                # subset it came from and its position within that subset
                merged_results = _merge_cached_results(specify_ensemble, merged_parts)
                if merged_results is not None:
                    print(f"Successfully merged {len(covering)} cached subsets")
                    return preferred_muid, merged_results

        return preferred_muid, None
    def _merge_cached_results(target_indices, parts):
        try:
            base_data = parts[0][1]
            merged = np.empty(base_data.shape, dtype=object)

            for i in range(base_data.shape[0]):
                for j in range(base_data.shape[1]):
                    full_list = [None] * len(h_ls_pre)

                    for subset_indices, data in parts:
                        entry = data[i, j]
                        if entry is None:
                            continue
                        entry_list = list(entry)

                        if len(entry_list) == len(h_ls_pre):
                            # Full-length array: entries are at their natural indices
                            for h_idx in subset_indices:
                                if h_idx < len(entry_list) and entry_list[h_idx] is not None:
                                    full_list[h_idx] = entry_list[h_idx]
                        else:
                            # Subset-length array: positional mapping
                            for pos, h_idx in enumerate(subset_indices):
                                if pos < len(entry_list) and entry_list[pos] is not None:
                                    full_list[h_idx] = entry_list[pos]

                    if target_indices == list(range(len(h_ls_pre))):
                        merged[i, j] = np.array(full_list, dtype=object)
                    else:
                        result = [full_list[idx] for idx in target_indices]
                        merged[i, j] = np.array(result, dtype=object)

            return merged
        except Exception as e:
            print(f"Warning: failed to merge cached subsets: {e}")
            import traceback
            traceback.print_exc()
            return None


    def _save_t2star_cache(muid, data_master_results):
        os.makedirs(os.path.join(dir_root, "mdata"), exist_ok=True)
        mdata_filename = os.path.join(dir_root, "mdata", f"{muid}.npy")
        np.save(mdata_filename, data_master_results, allow_pickle=True)
        print(f"Saved T2* cache to {mdata_filename}")

    def _normalize_specify_ensemble(specify_ensemble_value):
        if isinstance(specify_ensemble_value, range):
            specify_ensemble_value = list(specify_ensemble_value)
        elif isinstance(specify_ensemble_value, np.ndarray):
            specify_ensemble_value = specify_ensemble_value.tolist()
        elif isinstance(specify_ensemble_value, tuple):
            specify_ensemble_value = list(specify_ensemble_value)

        if specify_ensemble_value is None:
            return None

        if not isinstance(specify_ensemble_value, list):
            specify_ensemble_value = [specify_ensemble_value]

        normalized = []
        seen = set()
        for idx in specify_ensemble_value:
            try:
                idx_int = int(idx)
            except (TypeError, ValueError):
                continue

            if idx_int < 0 or idx_int >= len(h_ls_pre):
                continue

            if idx_int not in seen:
                seen.add(idx_int)
                normalized.append(idx_int)

        return normalized
#######################
    def _requested_t2star_ensembles(specify_ensemble_value):
        # Convention: specify_ensemble=None means run all h_i in h_ls_pre.
        normalized = _normalize_specify_ensemble(specify_ensemble_value)
        if normalized is None:
            return list(range(len(h_ls_pre)))
        return normalized

    def _get_missing_t2star_ensembles(cached_results, requested_ensembles):
        if cached_results is None:
            return list(requested_ensembles)

        if not hasattr(cached_results, "shape") or len(cached_results.shape) != 2:
            return list(requested_ensembles)

        missing = []
        for h_idx in requested_ensembles:
            has_data = False
            for i in range(cached_results.shape[0]):
                for j in range(cached_results.shape[1]):
                    try:
                        entry = cached_results[i, j]
                    except (IndexError, TypeError):
                        entry = None

                    if entry is None:
                        continue

                    try:
                        h_entry = entry[h_idx]
                    except (IndexError, TypeError):
                        h_entry = None

                    if h_entry is not None:
                        has_data = True
                        break
                if has_data:
                    break

            if not has_data:
                missing.append(h_idx)

        return missing

    def _merge_t2star_results_by_ensemble(base_results, update_results, update_ensembles):
        if base_results is None:
            return update_results
        if update_results is None:
            return base_results

        if not hasattr(base_results, "shape") or not hasattr(update_results, "shape"):
            return update_results

        if base_results.shape != update_results.shape:
            raise ValueError(
                f"Cannot merge T2* results with different shapes: {base_results.shape} vs {update_results.shape}"
            )

        update_set = set(update_ensembles) if update_ensembles is not None else None
        merged = np.empty(base_results.shape, dtype=object)

        for i in range(base_results.shape[0]):
            for j in range(base_results.shape[1]):
                base_entry = base_results[i, j]
                update_entry = update_results[i, j]

                if base_entry is None:
                    merged[i, j] = update_entry
                    continue
                if update_entry is None:
                    merged[i, j] = base_entry
                    continue

                base_list = list(base_entry)
                update_list = list(update_entry)

                if len(base_list) < len(h_ls_pre):
                    base_list.extend([None] * (len(h_ls_pre) - len(base_list)))
                if len(update_list) < len(h_ls_pre):
                    update_list.extend([None] * (len(h_ls_pre) - len(update_list)))

                for h_idx in range(len(h_ls_pre)):
                    if update_set is not None and h_idx not in update_set:
                        continue
                    if update_list[h_idx] is not None:
                        base_list[h_idx] = update_list[h_idx]

                merged[i, j] = np.array(base_list, dtype=object)

        return merged
    def _try_load_partial_t2star_cache(manager_params):
        if force_recompute:
            return None, None

        manager_specify_ensemble = manager_params.get("specify_ensemble")
        if manager_specify_ensemble is None:
            requested_ensembles = list(range(len(h_ls_pre)))
        else:
            requested_ensembles = _requested_t2star_ensembles(manager_specify_ensemble)

        remaining = set(requested_ensembles)
        if not remaining:
            return None, None

        exact_muid_candidates = set(_get_t2star_muid_candidates(manager_params))

        payload_target = dict(manager_params)
        payload_target.pop("specify_ensemble", None)

        manager_scan = ExptStore(dir_root)

        # Build a list of all compatible caches with their ensemble coverage
        compatible_caches = []
        for muid_candidate in manager_scan.list_ids():
            if muid_candidate in exact_muid_candidates:
                continue

            mdata_filename = os.path.join(dir_root, "mdata", f"{muid_candidate}.npy")
            if not os.path.exists(mdata_filename):
                continue

            record = manager_scan.get(muid_candidate)
            if record is None:
                continue

            payload_candidate = dict(record.get("payload", {}))
            if payload_candidate.pop("timestamp", None) != 0:
                continue

            candidate_specify = payload_candidate.pop("specify_ensemble", None)
            if payload_candidate != payload_target:
                continue

            candidate_ensembles = set(_requested_t2star_ensembles(candidate_specify))
            overlap = remaining.intersection(candidate_ensembles)
            if not overlap:
                continue

            compatible_caches.append((muid_candidate, candidate_ensembles))

        if not compatible_caches:
            return None, None

        # Greedily pick caches that cover the most remaining ensembles
        merged_result = None
        merged_muids = []

        while remaining and compatible_caches:
            # Find the cache with the best overlap with what's still missing
            best_idx = max(
                range(len(compatible_caches)),
                key=lambda i: len(remaining.intersection(compatible_caches[i][1]))
            )
            best_muid, best_ensembles = compatible_caches.pop(best_idx)
            overlap = remaining.intersection(best_ensembles)
            if not overlap:
                continue

            mdata_filename = os.path.join(dir_root, "mdata", f"{best_muid}.npy")
            try:
                cached_data = np.load(mdata_filename, allow_pickle=True)
            except Exception:
                continue

            print(f"Loading partial T2* cache from {mdata_filename} (ensembles {sorted(overlap)})")
            merged_result = _merge_t2star_results_by_ensemble(
                merged_result, cached_data, list(overlap)
            )
            merged_muids.append(best_muid)
            remaining -= overlap

            if not remaining:
                break

        if merged_result is None:
            return None, None

        # Return the first muid as the "base" identifier
        return merged_muids[0], merged_result

    if muid_data_master_results_override is None:

        if local_haar:
            is_bloqade = False

        use_qutip_backend = not is_bloqade or include_T2 
        print(f"use_qutip_backend={use_qutip_backend} (is_bloqade={is_bloqade}, include_T2={include_T2})")
        # raise NotImplementedError

        if include_T2:
            print("include_T2=True detected; forcing qutip backend (get_all_qutip_probs)")
            use_qutip_backend = True
            is_bloqade = False

        if not use_qutip_backend:
            assert not (is_expt_data and timestamp == 0), "timestamp should not be 0 for experiment data"
            
            if not is_expt_data:
                timestamp = 0

            if include_T2star:
                run_calib_check = not (after_how_many_ham_run_check is None or ham_check_dir_main is None)

                manager_params = {
                    'h_ls_pre': h_ls_pre,
                    'x_pre': x_pre,
                    't_plateau_ls': t_plateau_ls.tolist() if isinstance(t_plateau_ls, np.ndarray) else t_plateau_ls,
                    'base_params': base_params,
                    'Delta_mean_ls_all': Delta_mean_ls_all,
                    'Delta_local_ls_all': Delta_local_ls_all,
                    'gate_params_all': gate_params_all,
                    'local_haar': local_haar,
                    'override_local': override_local,
                    'start_Delta_from0': start_Delta_from0,
                    'phi_mode': phi_mode,
                    'specify_ensemble': specify_ensemble,
                    'include_T2': include_T2,
                    'Delta_local_ramp_time': Delta_local_ramp_time,
                    'Omega_delay_time': Omega_delay_time,
                    'type': 'bloqade-with-T2star', ## old, before march 26 2026 fix to make each qubit have different T2*
                    # 'type': 'bloqade-with-T2star-per-qubit',
                }

                manager = ExptStore(dir_root)
                muid, added = manager.add(manager_params, timestamp=0)

                requested_ensembles = _requested_t2star_ensembles(specify_ensemble)
                specify_ensemble_run = None if specify_ensemble is None else list(requested_ensembles)
                partial_base_results = None
                missing_ensembles = None

                muid_cached, cached_results = _try_load_t2star_cache(manager_params, muid)
                if cached_results is not None:
                    muid = muid_cached
                    data_master_results = cached_results
                else:
                    partial_muid, partial_results = _try_load_partial_t2star_cache(manager_params)

                    if partial_results is not None:
                        partial_base_results = partial_results
                        missing_ensembles = _get_missing_t2star_ensembles(partial_results, requested_ensembles)

                        if len(missing_ensembles) == 0:
                            data_master_results = partial_results
                            print("Partial-cache lookup already covers all requested ensembles.")
                            if partial_muid != muid:
                                _save_t2star_cache(muid, data_master_results)
                        else:
                            print(f"Partial-cache lookup hit; recomputing only missing ensembles: {missing_ensembles}")
                            specify_ensemble_run = missing_ensembles

                    if partial_results is None or (missing_ensembles is not None and len(missing_ensembles) > 0):

                        def _run_t2star_shot_bloqade(base_params_shot, Delta_local_ls_all_shot, gate_params_shot):
                            if not run_calib_check:
                                return get_all_single_hams_rand(
                                    h_ls_pre,
                                    x_pre,
                                    t_plateau_ls,
                                    seq_ls_pre_all,
                                    base_params_shot,
                                    Delta_mean_ls_all,
                                    Delta_local_ls_all_shot,
                                    gate_params_shot,
                                    cluster_spacing,
                                    manual_parallelization,
                                    is_expt_data,
                                    timestamp,
                                    dir_root,
                                    force_recompute=force_recompute,
                                    name=name,
                                    specify_ensemble=specify_ensemble_run,
                                    start_Delta_from0=start_Delta_from0,
                                    uniform_Omega_Delta_ramp=uniform_Omega_Delta_ramp,
                                    phi_mode=phi_mode,
                                    save_sample_hist=save_sample_hist,
                                    override_local=override_local,
                                    Delta_local_ramp_time=Delta_local_ramp_time,
                                    Omega_delay_time=Omega_delay_time,
                                    ignore_seq_for_muid=ignore_seq_for_muid,
                                    backwards_compatible_muid=backwards_compatible_muid,
                                    skip_store=True,
                                    sample_Delta_local_each_t_seq=True,
                                )

                            return get_all_single_hams_rand_calib_check(
                                h_ls_pre,
                                x_pre,
                                t_plateau_ls,
                                seq_ls_pre_all,
                                base_params_shot,
                                Delta_mean_ls_all,
                                Delta_local_ls_all_shot,
                                gate_params_shot,
                                cluster_spacing,
                                manual_parallelization,
                                is_expt_data,
                                timestamp,
                                dir_root,
                                force_recompute=force_recompute,
                                name=name,
                                after_how_many_ham_run_check=after_how_many_ham_run_check,
                                ham_check_dir_main=ham_check_dir_main,
                                specify_ensemble=specify_ensemble_run,
                                start_Delta_from0=start_Delta_from0,
                                uniform_Omega_Delta_ramp=uniform_Omega_Delta_ramp,
                                phi_mode=phi_mode,
                                override_local=override_local,
                                Delta_local_ramp_time=Delta_local_ramp_time,
                                Omega_delay_time=Omega_delay_time,
                                ignore_seq_for_muid=ignore_seq_for_muid,
                                backwards_compatible_muid=backwards_compatible_muid,
                                sample_Delta_local_each_t_seq=True,
                            )

                        if run_calib_check and T2star_workers not in (None, 1):
                            print("Calibration-check mode is enabled; forcing serial T2* execution.")

                        computed_results, n_shots, workers = _run_t2star_shots(
                            _run_t2star_shot_bloqade,
                            force_serial=run_calib_check,
                        )

                        if partial_base_results is not None and missing_ensembles is not None:
                            data_master_results = _merge_t2star_results_by_ensemble(
                                partial_base_results,
                                computed_results,
                                missing_ensembles,
                            )
                        else:
                            data_master_results = computed_results

                        _save_t2star_cache(muid, data_master_results)
            
            elif after_how_many_ham_run_check is None or ham_check_dir_main is None:
                muid, data_master_results = get_all_single_hams_rand(h_ls_pre, x_pre, t_plateau_ls, seq_ls_pre_all, base_params, Delta_mean_ls_all, Delta_local_ls_all, gate_params_all, cluster_spacing, manual_parallelization, is_expt_data, timestamp, dir_root, force_recompute=force_recompute, name=name, specify_ensemble=specify_ensemble, start_Delta_from0=start_Delta_from0, uniform_Omega_Delta_ramp=uniform_Omega_Delta_ramp, phi_mode=phi_mode, save_sample_hist=save_sample_hist, override_local=override_local, Delta_local_ramp_time=Delta_local_ramp_time, Omega_delay_time=Omega_delay_time, ignore_seq_for_muid=ignore_seq_for_muid, backwards_compatible_muid=backwards_compatible_muid)
            else:
                muid, data_master_results = get_all_single_hams_rand_calib_check(h_ls_pre, x_pre, t_plateau_ls, seq_ls_pre_all, base_params, Delta_mean_ls_all, Delta_local_ls_all, gate_params_all, cluster_spacing, manual_parallelization, is_expt_data, timestamp, dir_root, force_recompute=force_recompute, name=name, after_how_many_ham_run_check=after_how_many_ham_run_check, ham_check_dir_main=ham_check_dir_main, specify_ensemble=specify_ensemble, start_Delta_from0=start_Delta_from0, uniform_Omega_Delta_ramp=uniform_Omega_Delta_ramp, phi_mode=phi_mode, override_local=override_local, Delta_local_ramp_time=Delta_local_ramp_time, Omega_delay_time=Omega_delay_time, ignore_seq_for_muid=ignore_seq_for_muid, backwards_compatible_muid=backwards_compatible_muid)

        
        else: # qutip
            if include_T2star:

                manager_params = {
                    'h_ls_pre': h_ls_pre,
                    'x_pre': x_pre,
                    't_plateau_ls': t_plateau_ls.tolist() if isinstance(t_plateau_ls, np.ndarray) else t_plateau_ls,
                    'base_params': base_params,
                    'Delta_mean_ls_all': Delta_mean_ls_all,
                    'Delta_local_ls_all': Delta_local_ls_all,
                    'gate_params_all': gate_params_all,
                    'local_haar': local_haar,
                    'override_local': override_local,
                    'start_Delta_from0': start_Delta_from0,
                    'phi_mode': phi_mode,
                    'specify_ensemble': specify_ensemble,
                    'include_T2': include_T2,
                    'Delta_local_ramp_time': Delta_local_ramp_time,
                    'Omega_delay_time': Omega_delay_time,
                    'type': 'qutip-with-T2star-per-qubit-enabled-new-T2star-func-us',
                }

                if not indep_haar:
                    manager_params['indep_haar'] = indep_haar

                manager = ExptStore(dir_root)
                muid, added = manager.add(manager_params, timestamp=0)

                print("QUTIP T2* MUID", muid)


                requested_ensembles = _requested_t2star_ensembles(specify_ensemble)
                specify_ensemble_run = None if specify_ensemble is None else list(requested_ensembles)
                partial_base_results = None
                missing_ensembles = None

                muid_cached, cached_results = _try_load_t2star_cache(manager_params, muid)
                if cached_results is not None:
                    muid = muid_cached
                    data_master_results = cached_results
                else:
                    partial_muid, partial_results = _try_load_partial_t2star_cache(manager_params)

                    if partial_results is not None:
                        partial_base_results = partial_results
                        missing_ensembles = _get_missing_t2star_ensembles(partial_results, requested_ensembles)

                        if len(missing_ensembles) == 0:
                            data_master_results = partial_results
                            print("Partial-cache lookup already covers all requested ensembles.")
                            if partial_muid != muid:
                                _save_t2star_cache(muid, data_master_results)
                        else:
                            print(f"Partial-cache lookup hit; recomputing only missing ensembles: {missing_ensembles}")
                            specify_ensemble_run = missing_ensembles

                    if partial_results is None or (missing_ensembles is not None and len(missing_ensembles) > 0):

                        def _run_t2star_shot_qutip(base_params_shot, Delta_local_ls_all_shot, gate_params_shot):
                            return get_all_qutip_probs(
                                h_ls_pre,
                                x_pre,
                                t_plateau_ls,
                                seq_ls_pre_all,
                                base_params_shot,
                                Delta_mean_ls_all,
                                Delta_local_ls_all_shot,
                                gate_params_shot,
                                dir_root,
                                force_recompute=force_recompute,
                                specify_ensemble=specify_ensemble_run,
                                include_T2=include_T2,
                                start_Delta_from0=start_Delta_from0,
                                uniform_Omega_Delta_ramp=uniform_Omega_Delta_ramp,
                                phi_mode=phi_mode,
                                local_haar=local_haar,
                                indep_haar=indep_haar,
                                Delta_local_ramp_time=Delta_local_ramp_time,
                                Omega_delay_time=Omega_delay_time,
                                ignore_seq_for_muid=ignore_seq_for_muid,
                                skip_store=True,
                                sample_Delta_local_each_t_seq=True,
                            )

                        computed_results, n_shots, workers = _run_t2star_shots(
                            _run_t2star_shot_qutip,
                        )

                        if partial_base_results is not None and missing_ensembles is not None:
                            data_master_results = _merge_t2star_results_by_ensemble(
                                partial_base_results,
                                computed_results,
                                missing_ensembles,
                            )
                        else:
                            data_master_results = computed_results

                        _save_t2star_cache(muid, data_master_results)
            else:
                muid, data_master_results = get_all_qutip_probs(h_ls_pre, x_pre, t_plateau_ls, seq_ls_pre_all, base_params, Delta_mean_ls_all, Delta_local_ls_all, gate_params_all, dir_root, force_recompute=force_recompute, specify_ensemble=specify_ensemble, include_T2=include_T2, start_Delta_from0=start_Delta_from0, uniform_Omega_Delta_ramp=uniform_Omega_Delta_ramp, phi_mode=phi_mode, local_haar=local_haar, indep_haar=indep_haar,Delta_local_ramp_time=Delta_local_ramp_time, Omega_delay_time=Omega_delay_time, ignore_seq_for_muid=ignore_seq_for_muid)
    
    else:
        print("OVERRIDING!!!!!!")
        muid, data_master_results = muid_data_master_results_override

    print("data_master_results", data_master_results)

    # get a uid for the epsilon_r, epsilon_g
    epsilon_params = {
        'epsilon_r_ens': epsilon_r_ens,
        'epsilon_g_ens': epsilon_g_ens,
        'is_bloqade': is_bloqade,
        'apply_correction': apply_correction,
        'shot_noise_model': shot_noise_model,
        'start_Delta_from0': start_Delta_from0,
        'phi_mode': phi_mode,
        'specify_ensemble': specify_ensemble,
        'unform_Omega_Delta_ramp': uniform_Omega_Delta_ramp,
        'local_haar': local_haar,
        'total_system': total_system,
        "override_local": override_local,
        "Delta_local_ramp_time": Delta_local_ramp_time,
        "Omega_delay_time": Omega_delay_time,
        "muid_data_master_results_override": muid_data_master_results_override is not None,
        'include_T2star': include_T2star,
        'include_T2': include_T2,
        'type': 'process_bitstrings'
        }
    
    if not indep_haar:
        epsilon_params['indep_haar'] = indep_haar
    print("epsilon_r_ens", epsilon_r_ens, "epsilon_g_ens", epsilon_g_ens)
    
    manager = ExptStore(dir_root)
    epsilon_muid, added = manager.add(epsilon_params, timestamp=0)

    os.makedirs(os.path.join(dir_root, "mdata"), exist_ok=True)

    # if specify_ensemble is None:
    total_filename = os.path.join(dir_root, "mdata", f"{muid}_{process_opt}_{epsilon_muid}_total.npy")
    sem_total_filename = os.path.join(dir_root, "mdata", f"{muid}_{process_opt}_{epsilon_muid}_sem_total.npy")
    no_avg_total_filename = os.path.join(dir_root, "mdata", f"{muid}_{process_opt}_{epsilon_muid}_no_avg_total.npy")
    # else:
    #     ensemble_str = ''
    #     for ens in specify_ensemble:
    #         ensemble_str+=str(ens)+'_'
    #     total_filename = os.path.join(dir_root, "mdata", f"{muid}_{process_opt}_{ensemble_str}_{epsilon_muid}_total.npy")
    #     sem_total_filename = os.path.join(dir_root, "mdata", f"{muid}_{process_opt}_{ensemble_str}_{epsilon_muid}_sem_total.npy")
    #     no_avg_total_filename = os.path.join(dir_root, "mdata", f"{muid}_{process_opt}_{ensemble_str}_{epsilon_muid}_no_avg_total.npy")

    print("total filename", total_filename)

    if (not os.path.exists(total_filename) or not os.path.exists(sem_total_filename)) or force_recompute or force_recompute_processing:

        # For raw bitstrings, skip all processing and just extract the data
        if process_opt == "raw":
            raw_results = np.zeros((len(Delta_mean_ls_all), len(Delta_local_ls_all)), dtype=object)
            
            for i, Delta_mean in enumerate(Delta_mean_ls_all):
                for j, Delta_local in enumerate(Delta_local_ls_all):
                    Delta_mean_local_results = data_master_results[i, j]
                    raw_results[i, j] = Delta_mean_local_results
                    
            # Save and return raw results
            np.save(total_filename, raw_results, allow_pickle=True)
            return raw_results, None, None

        N = len(h_ls_pre[0])  # number of qubits, assuming all h_ls have the same length
        if not total_system:
            n_A = N // 2 # assume bipartition
        else:
            n_A = N
        # n_A = N 
        qubits_A = list(range(n_A))  # qubits in subsystem A
        
        if process_opt == "ee":
            hamming_sub = get_hamming_matrix(n_A)
        elif process_opt == "sp":
            hamming = get_hamming_matrix(N)

        manager = ExptStore(dir_root)

        total_results = np.zeros((len(Delta_mean_ls_all), len(Delta_local_ls_all)), dtype=object) # shape: (Delta_mean, Delta_local)
        sem_total_results = np.zeros((len(Delta_mean_ls_all), len(Delta_local_ls_all)), dtype=object) # shape: (Delta_mean, Delta_local)
        no_avg_results = np.zeros((len(Delta_mean_ls_all), len(Delta_local_ls_all)), dtype=object) 

        epsilon_r_ens = np.asarray(epsilon_r_ens)
        epsilon_g_ens = np.asarray(epsilon_g_ens)

       


        for i, Delta_mean in enumerate(Delta_mean_ls_all):
            for j, Delta_local in enumerate(Delta_local_ls_all):

                Delta_mean_local_results = data_master_results[i, j]
                t_ls_ham_ens=[]
                processed_ensemble_indices = []
                ensemble_invalid_reasons = {}
                nan_uf = ufloat(np.nan, np.nan)

                def _record_invalid_reason(h_idx, t_idx, reason):
                    h_reason_map = ensemble_invalid_reasons.setdefault(h_idx, {})
                    t_reasons = h_reason_map.setdefault(t_idx, [])
                    t_reasons.append(reason)
                print("processing Delta mean progress:", i/len(Delta_mean_ls_all), epsilon_g_ens)

                epsilon_r_ij = extract_value(np.array(epsilon_r_ens), (i, j))
                epsilon_g_ij = extract_value(np.array(epsilon_g_ens), (i, j))
                print("HERE epsilon_r_ij, epsilon_g_ij", epsilon_r_ij, epsilon_g_ij)

                # get uids for combo of Delta_mean, Delta_local, muid
                delta_muid_params = {
                    'Delta_mean': Delta_mean,
                    'Delta_local': Delta_local,
                    'epsilon_r_ens_ij': epsilon_r_ij,
                    'epsilon_g_ens_ij': epsilon_g_ij,
                    'muid': muid
                }
                delta_muid, added = manager.add(delta_muid_params, timestamp=timestamp)

                vals_filename = os.path.join(dir_root, "mdata", f"{delta_muid}_{process_opt}_vals.npy")
                sems_filename = os.path.join(dir_root, "mdata", f"{delta_muid}_{process_opt}_sems.npy")
                no_avg_filename = os.path.join(dir_root, "mdata", f"{delta_muid}_{process_opt}_no_avg.npy")

                # if same_U_all_time and process_opt == "sp": # get the one reference
                #     gate_params = gate_params_all[i][j]
                #     seq_pre_ls = seq_ls_pre_all[i][j][0]

                #     bitstrings_0_ls = get_single_ham_rand(h_ls, x_pre[l], [0]*len(t_plateau_ls), seq_pre_ls, base_params,Delta_mean, Delta_local, gate_params, cluster_spacing, is_expt_data, timestamp, dir_root,debug=False)
                            # shape: (t_plateau, seq, bitstrings)

                if (not os.path.exists(vals_filename) or not os.path.exists(sems_filename)) or force_recompute or force_recompute_processing:

                    requested_ensembles = _requested_t2star_ensembles(specify_ensemble)
                    expected_h_count = len(requested_ensembles)
                    allowed_ensembles = set(requested_ensembles)

                    # Debug: Print dimensions of Delta_mean_local_results
                    print(f"Debug: Delta_mean_local_results shape: {len(Delta_mean_local_results) if Delta_mean_local_results is not None else 'None'}")
                    if Delta_mean_local_results is not None and len(Delta_mean_local_results) > 0:
                        print(f"Debug: First element shape: {len(Delta_mean_local_results[0]) if Delta_mean_local_results[0] is not None else 'None'}")

                    for l,h_ls in enumerate(h_ls_pre):
                        if l in allowed_ensembles:
                            # Debug: Check if ensemble index l exists
                            print(f"Debug: Checking ensemble l={l}, total ensembles available: {len(Delta_mean_local_results) if Delta_mean_local_results is not None else 'None'}")
                            
                            if Delta_mean_local_results is None or l >= len(Delta_mean_local_results):
                                print(f"Error: Ensemble index l={l} is out of range. Available ensembles: {len(Delta_mean_local_results) if Delta_mean_local_results is not None else 'None'}")
                                continue
                                
                            xh_t_ls = [] # ee list for single hamiltonian in time
                            seq_pre_ls = seq_ls_pre_all[i][j][l] # get the sequences for this Delta_mean, Delta_local, x
                            # if process_opt == "sp" and not same_U_all_time:  # get the reference
                            #         gate_params = gate_params_all[i][j]

                            #         bitstrings_0_ls = get_single_ham_rand(h_ls, x_pre[l], [0]*len(t_plateau_ls), seq_pre_ls, base_params,Delta_mean, Delta_local, gate_params, cluster_spacing, is_expt_data, timestamp, dir_root,debug=False)
                            #         # shape: (t_plateau, seq, bitstrings)
                            epsilon_r = extract_value(np.array(epsilon_r_ens), (i, j, l))
                            epsilon_g = extract_value(np.array(epsilon_g_ens), (i, j, l))

                            epsilon_r = np.array(epsilon_r)
                            epsilon_g = np.array(epsilon_g)
                            
                            # Keep as scalar if 0-dimensional, otherwise keep as numpy array
                            if epsilon_r.ndim == 0:
                                epsilon_r = float(epsilon_r)
                            if epsilon_g.ndim == 0:
                                epsilon_g = float(epsilon_g)
                            
                            no_avg = []
                            for m in range(len(t_plateau_ls)):
                                ## average over the random unitaries
                                no_avg_fixed_t = []
                                X_fixed_t =[]
                                seq_ls = seq_pre_ls[m]
                                
                                # Debug: Check dimensions before accessing
                                print(f"Debug: l={l}, m={m}, len(seq_ls)={len(seq_ls)}")
                                if Delta_mean_local_results[l] is None:
                                    print(f"Error: Delta_mean_local_results[{l}] is None")
                                    _record_invalid_reason(l, m, "ensemble-data-none")
                                    xh_t_ls.append(nan_uf)
                                    continue
                                print(f"Debug: len(Delta_mean_local_results[{l}]) = {len(Delta_mean_local_results[l])}")
                                if m >= len(Delta_mean_local_results[l]):
                                    print(f"Error: Time plateau index m={m} is out of range for ensemble {l}. Available time plateaus: {len(Delta_mean_local_results[l])}")
                                    _record_invalid_reason(l, m, "time-index-out-of-range")
                                    xh_t_ls.append(nan_uf)
                                    continue
                                if Delta_mean_local_results[l][m] is None:
                                    print(f"Error: Delta_mean_local_results[{l}][{m}] is None")
                                    _record_invalid_reason(l, m, "time-entry-none")
                                    xh_t_ls.append(nan_uf)
                                    continue
                                print(f"Debug: len(Delta_mean_local_results[{l}][{m}]) = {len(Delta_mean_local_results[l][m])}")
                            
                                for s in range(len(seq_ls)):
                                    # Debug: Check sequence index
                                    print(f"Debug: Accessing Delta_mean_local_results[{l}][{m}][{s}]")
                                    if s >= len(Delta_mean_local_results[l][m]):
                                        print(f"Error: Sequence index s={s} is out of range for ensemble {l}, time plateau {m}. Available sequences: {len(Delta_mean_local_results[l][m])}")
                                        _record_invalid_reason(l, m, "sequence-index-out-of-range")
                                        continue
                                    
                                    bitstrings = Delta_mean_local_results[l][m][s]

                                    # epsilon_r = epsilon_r_ens[i][j][l] if type(epsilon_r_ens) is not float else epsilon_r_ens
                                    # epsilon_g = epsilon_g_ens[i][j][l] if type(epsilon_g_ens) is not float else epsilon_g_ens
                                    

                                    if l == 0:
                                        print("HERE epsilon_r_ens, epsilon_g_ens", epsilon_r, epsilon_g)

                                    #### ADD SHOT NOISE IN EST_PURITY AND EST_FIDELITY
                                    if process_opt == "ee":
                                        X_est = est_purity(bitstrings, N,  qubits_A, epsilon_r, epsilon_g, is_bloqade=is_bloqade, hamming=hamming_sub, shot_noise_model=shot_noise_model,n_shots=gate_params_all[i][j]['n_shots'])
        
                                    elif process_opt == "sp":
                                        if same_U_all_time:
                                            bitstring_0 = Delta_mean_local_results[l][0][s]
                                        
                                            X_est = est_fidelity(bitstrings, bitstring_0, N, epsilon_r, epsilon_g, is_bloqade=is_bloqade, hamming=hamming, shot_noise_model=shot_noise_model)
                                        else:
                                            raise NotImplementedError("Survival probability not implemented for different unitaries at different times")

                                    if X_est is not None:
                                        X_fixed_t.append(X_est)

                                X_fixed_t = np.array(X_fixed_t)
                                no_avg_fixed_t.append(X_fixed_t)

                                if len(X_fixed_t) == 0:
                                    _record_invalid_reason(l, m, "no-valid-samples")
                                    xh_t_ls.append(nan_uf)
                                    continue

                                try:

                                    # if any elements are ufloat, treat accordingly
                                    if isinstance(X_fixed_t[0], UFloat):
                                        
                                        vals = unp.nominal_values(X_fixed_t)
                                        errs = unp.std_devs(X_fixed_t)

                                        mean_X = np.mean(vals)
                                        # standard error from both sample scatter and individual uncertainties
                                        sem_X = np.sqrt(np.mean(errs**2) + np.var(vals, ddof=1)/len(vals))
                                    else:
                                        mean_X = X_fixed_t.mean()
                                        sem_X = X_fixed_t.std(ddof=1) / np.sqrt(len(X_fixed_t))

                                    if process_opt == "ee":
                                        ee_val = -np.log(mean_X)
                                        sem = sem_X / mean_X  # linearized uncertainty propagation
                                        xh_t_ls.append(ufloat(ee_val, sem))
                                    elif process_opt == "sp":
                                        xh_t_ls.append(ufloat(mean_X, sem_X)) 
                                except Exception as e:
                                    print("Error in processing ensemble", l, "time", m, "error:", e, "proceeding (adding NaN placeholder)...")
                                    _record_invalid_reason(l, m, f"processing-exception:{type(e).__name__}")
                                    # Add a placeholder value to maintain consistent array lengths
                                    xh_t_ls.append(nan_uf)

                            # take average over the hamiltonians but not over the time
                            t_ls_ham_ens.append(xh_t_ls)
                            processed_ensemble_indices.append(l)
                            no_avg.append(no_avg_fixed_t)

                    raw_h_count = len(t_ls_ham_ens)
                    if raw_h_count != expected_h_count:
                        raise ValueError(
                            f"Hamiltonian count mismatch before averaging at (i={i}, j={j}): "
                            f"expected {expected_h_count} from specify_ensemble={requested_ensembles}, "
                            f"got {raw_h_count}"
                        )

                    # now compute the average over the hamiltonians
                    # print("before avg:", t_ls_ham_ens)
                    # remove any Nan
                    
                    expected_t_len = len(t_plateau_ls)
                    normalized_ham_rows = []
                    for row in t_ls_ham_ens:
                        row_list = list(row)
                        if len(row_list) < expected_t_len:
                            row_list.extend([nan_uf] * (expected_t_len - len(row_list)))
                        elif len(row_list) > expected_t_len:
                            row_list = row_list[:expected_t_len]
                        normalized_ham_rows.append(row_list)

                    t_ls_ham_ens = np.array(normalized_ham_rows, dtype=object)

                    # Drop only ensembles that are fully invalid across all times.
                    # Rows with partial NaNs are still useful and should contribute where valid.
                    nominal_vals = unp.nominal_values(t_ls_ham_ens)
                    valid_row_mask = ~np.isnan(nominal_vals).all(axis=1)
                    removed_ensemble_indices = [
                        h_idx for h_idx, keep in zip(processed_ensemble_indices, valid_row_mask.tolist()) if not keep
                    ]
                    t_ls_ham_ens = t_ls_ham_ens[valid_row_mask]

                    filtered_h_count = t_ls_ham_ens.shape[0]
                    if filtered_h_count != expected_h_count:
                        print(
                            f"Warning: Hamiltonian count after filtering at (i={i}, j={j}) is "
                            f"{filtered_h_count}/{expected_h_count} for specify_ensemble={requested_ensembles}."
                        )
                        print(
                            f"Warning: fully invalid ensembles removed at (i={i}, j={j}): "
                            f"{removed_ensemble_indices}"
                        )
                        for removed_h_idx in removed_ensemble_indices:
                            time_reason_map = ensemble_invalid_reasons.get(removed_h_idx, {})
                            if not time_reason_map:
                                print(
                                    f"Warning: removed ensemble {removed_h_idx} has no explicit captured reason; "
                                    "all computed values may have become NaN downstream."
                                )
                                continue

                            reason_parts = []
                            for t_idx in sorted(time_reason_map.keys()):
                                reason_counts = Counter(time_reason_map[t_idx])
                                reason_parts.append(f"t={t_idx}:{dict(reason_counts)}")

                            print(
                                f"Warning: invalidity details for removed ensemble {removed_h_idx}: "
                                + "; ".join(reason_parts)
                            )

                    if t_ls_ham_ens.shape[0] == 0:
                        vals_ls = np.full(expected_t_len, np.nan)
                        sems_ls = np.full(expected_t_len, np.nan)
                        np.save(vals_filename, vals_ls)
                        np.save(sems_filename, sems_ls)
                        np.save(no_avg_filename, t_ls_ham_ens, allow_pickle=True)
                        total_results[i, j] = vals_ls
                        sem_total_results[i, j] = sems_ls
                        no_avg_results[i, j] = t_ls_ham_ens
                        continue

                    t_ls_avg = t_ls_ham_ens.mean(axis=0)

                    # save the results using muid
                    vals_ls = unp.nominal_values(t_ls_avg)
                    sems_ls = unp.std_devs(t_ls_avg)

                    # no_avg = np.array(no_avg, dtype=object)
                    
                    np.save(vals_filename, vals_ls)
                    np.save(sems_filename, sems_ls)
                    np.save(no_avg_filename, t_ls_ham_ens) ## replaced no_avg by t_ls_ham_ens
                else:
                    vals_ls = np.load(vals_filename, allow_pickle=True)
                    sems_ls = np.load(sems_filename, allow_pickle=True)
                    no_avg = np.load(no_avg_filename, allow_pickle=True)

                total_results[i, j] = vals_ls
                sem_total_results[i, j] = sems_ls
                no_avg_results[i, j] = no_avg

        # save the results using muid
        np.save(total_filename, total_results, allow_pickle=True)
        np.save(sem_total_filename, sem_total_results, allow_pickle=True)
        np.save(no_avg_total_filename, no_avg_results, allow_pickle=True)

    else:
        total_results = np.load(total_filename, allow_pickle=True)
        
        # For raw mode, sem_total_results and no_avg_results don't exist
        if process_opt == "raw":
            return total_results, None, None
            
        sem_total_results = np.load(sem_total_filename, allow_pickle=True)
        no_avg_results = np.load(no_avg_total_filename, allow_pickle=True)
    
    return total_results, sem_total_results, no_avg_results


def _save_sample_hist(data_all_times, t_plateau_ls, seq_ls_pre, muid, Delta_mean, Delta_local, is_expt_data, timestamp, dir_root, h_index):
    """
    Helper function to save sample histograms at first, middle, last time plateaus and sequences
    """
    
    # Create results directory
    hist_dir = os.path.join(dir_root, "results", "bit_hist")
    os.makedirs(hist_dir, exist_ok=True)
    
    n_times = len(data_all_times)
    if n_times == 0:
        return
        
    # Select sample time indices: first, middle, last
    time_indices = [0]
    if n_times > 1:
        time_indices.append(n_times // 2)  # middle
    if n_times > 2:
        time_indices.append(n_times - 1)  # last
    
    for t_idx in time_indices:
        if t_idx >= len(data_all_times) or len(data_all_times[t_idx]) == 0:
            continue
            
        t_plateau = t_plateau_ls[t_idx]
        seq_data = data_all_times[t_idx]
        
        n_seqs = len(seq_data)
        if n_seqs == 0:
            continue
            
        # Select sample sequence indices: first, middle, last
        seq_indices = [0]
        if n_seqs > 1:
            seq_indices.append(n_seqs // 2)  # middle
        if n_seqs > 2:
            seq_indices.append(n_seqs - 1)  # last
            
        for s_idx in seq_indices:
            if s_idx >= len(seq_data) or len(seq_data[s_idx]) == 0:
                continue
                
            bitstrings = seq_data[s_idx]
            
            # Convert bitstrings to integers for histogram
            if len(bitstrings) > 0:
                # Convert binary strings to integers
                if isinstance(bitstrings[0], str):
                    bit_values = [int(bs, 2) for bs in bitstrings]
                    n_qubits = len(bitstrings[0])
                else:
                    # Flatten in case it's nested and convert to list
                    bit_values = np.array(bitstrings).flatten().tolist()
                    n_qubits = int(np.log2(np.max(bit_values) + 1))
                
                # Create histogram
                plt.figure(figsize=(10, 6))
                bins = range(2**n_qubits + 1)
                
                plt.hist(bit_values, bins=bins, alpha=0.7, edgecolor='black', color='blue')
                
                # Create title with all relevant information
                title = (f"muid_{muid}_Dm_{Delta_mean:.3g}_Dl_{Delta_local:.3g}_"
                        f"t_{t_plateau:.3g}_seq_{s_idx}_expt_{is_expt_data}_ts_{timestamp}_h_{h_index}")
                plt.title(title, fontsize=10)
                plt.xlabel('Bitstring Value')
                plt.ylabel('Count')
                # plt.grid(True, alpha=0.3)
                
                # Save with the same name as title
                filename = f"{title}.png"
                filepath = os.path.join(hist_dir, filename)
                plt.savefig(filepath, dpi=150, bbox_inches='tight')
                plt.close()
                
                print(f"Saved histogram: {filename}")



def numerical(h_ls_pre, x_pre, t_plateau_ls,  base_params, Delta_mean_ls, Delta_local_ls, dir_root, process_opt = "ee", force_recompute=False, time_dep=True, start_Delta_from0=False, uniform_Omega_Delta_ramp=False, Delta_local_ramp_time=0.05, Omega_delay_time=0.0): 
    """
    numerical comparison for entanglement entropy or survival probability (process_opt = "ee" or "sp")

    time_dep == True means do the ramping like experiment
    """
    
    assert process_opt in ["ee", "sp"], f"process_opt must be either 'ee' or 'sp', got {process_opt}"
    
    manager = ExptStore(dir_root)
    manager_params = {
        'h_ls_pre': h_ls_pre,
        'x_pre': x_pre,
        't_plateau_ls': t_plateau_ls.tolist() if isinstance(t_plateau_ls, np.ndarray) else t_plateau_ls,
        'base_params': base_params,
        'Delta_mean_ls': Delta_mean_ls,
        'Delta_local_ls': Delta_local_ls,
        'time_dep': time_dep,
        'start_Delta_from0': start_Delta_from0,
        'Delta_local_ramp_time': Delta_local_ramp_time,
        'Omega_delay_time': Omega_delay_time,
        'type': 'numerical'
    }  
    if not uniform_Omega_Delta_ramp:
        manager_params['uniform_Omega_Delta_ramp'] = uniform_Omega_Delta_ramp 
    muid, added = manager.add(manager_params, timestamp=0)  # timestamp is not used here, so we can set it to 0
    os.makedirs(os.path.join(dir_root, "ndata"), exist_ok=True) 

    num_total_filename = os.path.join(dir_root, "ndata", f"{muid}_num_{process_opt}_total.npy")  
    no_avg_total_filename = os.path.join(dir_root, "ndata", f"{muid}_num_{process_opt}_no_avg_total.npy")

    if not os.path.exists(num_total_filename) or force_recompute:
        num_results = np.zeros((len(Delta_mean_ls), len(Delta_local_ls)), dtype=object)
        no_avg_results = np.zeros((len(Delta_mean_ls), len(Delta_local_ls)), dtype=object)

        for i, Delta_mean in enumerate(Delta_mean_ls):
            for j, Delta_local in enumerate(Delta_local_ls):
                delta_muid_params = {
                    'Delta_mean': Delta_mean,
                    'Delta_local': Delta_local,
                    'time_dep': time_dep,
                    'start_Delta_from0': start_Delta_from0,
                    'muid': muid
                }
                delta_muid, added = manager.add(delta_muid_params, timestamp=0)
                delta_muid_filename = os.path.join(dir_root, "ndata", f"{delta_muid}_num_{process_opt}_vals.npy")

                if not os.path.exists(delta_muid_filename) or force_recompute:
                    num_t_ls_ham_ens = []
                    for x, h_ls in zip(x_pre, h_ls_pre):
                        print("Delta mean progress:", i/len(Delta_mean_ls))
                        print("Delta local progress:", j/len(Delta_local_ls))
                        print("ensemble progress", h_ls_pre.index(h_ls) / len(h_ls_pre))

                        if process_opt == "ee":
                            num_xh_t_ls = get_ee(h_ls, x, t_plateau_ls, base_params, Delta_mean, Delta_local, time_dep=time_dep, start_Delta_from0=start_Delta_from0, uniform_Omega_Delta_ramp=uniform_Omega_Delta_ramp, Delta_local_ramp_time=Delta_local_ramp_time, Omega_delay_time=Omega_delay_time)

                        elif process_opt == "sp":
                            num_xh_t_ls = get_sp(h_ls, x, t_plateau_ls, base_params, Delta_mean, Delta_local, time_dep=time_dep, start_Delta_from0=start_Delta_from0, uniform_Omega_Delta_ramp=uniform_Omega_Delta_ramp, Delta_local_ramp_time=Delta_local_ramp_time, Omega_delay_time=Omega_delay_time)

                        num_t_ls_ham_ens.append(num_xh_t_ls)
                    
                    
                    num_t_ls_ham_ens = np.array(num_t_ls_ham_ens)
                    num_t_ls_avg = num_t_ls_ham_ens.mean(axis=0)

                    # save the results using delta_muid
                    np.save(delta_muid_filename, num_t_ls_avg, allow_pickle=True)
                    np.save(no_avg_total_filename, num_t_ls_ham_ens, allow_pickle=True)

                else:
                    num_t_ls_avg = np.load(delta_muid_filename, allow_pickle=True)
                    num_t_ls_ham_ens = np.load(no_avg_total_filename, allow_pickle=True)

                num_results[i, j] = num_t_ls_avg
                no_avg_results[i, j] = num_t_ls_ham_ens

        # save the results using muid
        np.save(num_total_filename, num_results, allow_pickle=True)
        np.save(no_avg_total_filename, no_avg_results, allow_pickle=True)
        # print("Numerical entanglement entropy results saved to", num_total_filename)
    else:
        num_results = np.load(num_total_filename, allow_pickle=True)
        no_avg_results = np.load(no_avg_total_filename, allow_pickle=True)
    return num_results, no_avg_results

if __name__ == '__main__':
    for D_l in [-2.71, -54.2, -125]:
        print(_t2_star_from_delta_local(D_l))