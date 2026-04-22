## file to prepare random program in bloqade
## updated version of run_expt_headless.py that complies with experiment
import bloqade.analog as ba
import numpy as np
import os, json, shutil
from copy import deepcopy
try:
    from QuEraToolbox.expt_file_manager import ExptStore, unique_filename
except ImportError:
    from expt_file_manager import ExptStore, unique_filename
from pathlib import Path
## AWS
from braket.devices import Devices
from braket.jobs import (
    InstanceConfig,
    hybrid_job,
)

_Omega_slope = 250.0
_Delta_slope = 2500.0

my_bucket = "qeg-braket-us-east-1-00" ### SPECIFY THESE HERE IF NECESSARY!!!!!
my_prefix = "simulation-results/ors35"
# s3_destination_folder = (my_bucket, my_prefix) # or 'default'
s3_destination_folder = 'default'

gate_params = {
    1: {
        'phi': 0,
        'Omega': 15.8,
    },
    2: {
        'phi': np.pi/2,
        'Omega': 15.8,
    },
}

# -------- process results helpers

def reassign_bits(k):
    k_new = []
    for ki in list(k):
        if ki == "0":
            k_new.append("1")
        else:
            k_new.append("0")
    k_new_str = "".join(k_new)
    return k_new_str

def common_process(bitstrings_ls, N, manual_parallelization=False):
    if manual_parallelization: # need to take qubit index mod N
        bitstrings_ls = [(i % (2**N), v) for i, v in bitstrings_ls]
        # now recombine the bitstrings that share same 0 value in the tuple
        bitstrings_ls = [(i, sum(v for j, v in bitstrings_ls if j == i)) for i in set(i for i, v in bitstrings_ls)]

    # check if all bitstrings are present; if not, add them with 0 counts
    for i in range(2**N):
        if i not in [r[0] for r in bitstrings_ls]:
            bitstrings_ls.append((i, 0))
    return bitstrings_ls


def process_exp(report, N, manual_parallelization=False, is_expt_data=False):
    # bitstrings_dict = report.counts()[0]
    try:
        unfiltered_dict = report.counts()[0]
        bitstrings_dict = report.counts(filter_perfect_filling=True)[0]

        # sum of all values is the total shots
        unfiltered_shots = sum(unfiltered_dict.values())
        filtered_shots = sum(bitstrings_dict.values())
        
        if is_expt_data:
            print(f"Total unfiltered shots: {unfiltered_shots}, shots postselected with perfect filling: {filtered_shots}")

    except IndexError: # make empty dict if no perfect filling shots
        bitstrings_dict = {}
    # print("report raw", report.counts())
    # print("filtered bitstrings", report.bitstrings(filter_perfect_filling=True))
    bitstrings_ls = [(int(reassign_bits(k),2), v) for k, v in bitstrings_dict.items()]
    return common_process(bitstrings_ls, N, manual_parallelization)

    
def process_manual(backup_file, N, manual_parallelization=False):
    backup = json.load(open(backup_file, "r"))
    measurements = backup['measurements']

    bitstrings = []
    for meas in measurements:
        if meas['shotResult']['preSequence']==[1]*N: # need to be all filled
            raw_result = meas['shotResult']["postSequence"] # this is a list of 0,1
            corrected_result = [1 - r for r in raw_result] # flip 0,1
            bitstring = "".join([str(r) for r in corrected_result])
            bitstrings.append(bitstring)

    unique, counts = np.unique(bitstrings, return_counts=True)
    bitstrings_ls = [(int(k,2), v) for k, v in zip(unique, counts)]
    return common_process(bitstrings_ls, N, manual_parallelization)

# -------- bloqade compilation
def compile_rand_seq(seq,
                     in_gate_params,
                     t_ramp,
                     t_Delta_ramp,
                     durations = [],
                     Delta_durations = [],
                     durations_phi = [],
                     values_dict={
                                'Omega': [], 'Delta_global': [], 'Delta_local': [], 'phi': []
                        },
                    min_dt = 0.05, 
                    start_Delta_from0 = False,
                    phi_mode='binary',
                    Omega_delay_time=None
                    ): # keep appending to the durations and values
    
    # print('PHI MODE', phi_mode)
    vals_add = {
        'Omega': [],
        'Delta_global': [],
        'Delta_local': [],
        'phi': []
    }
    params = ['Omega', 'Delta_global', 'Delta_local', 'phi']
    gate_duration = in_gate_params['gate_duration']


    # first populate the gate parameters
    # if phi_mode=='binary':
    gate_params[1]['Delta_global'] = in_gate_params['Delta_global']
    gate_params[1]['Delta_local'] = in_gate_params['Delta_local']
    gate_params[2]['Delta_global'] = in_gate_params['Delta_global']
    gate_params[2]['Delta_local'] = in_gate_params['Delta_local']

    durations_add = []
    durations_phi_add = []

    # Delta_global_delta = values_dict['Delta_global'][-1] - gate_params[1]['Delta_global']
    # Delta_local_delta = values_dict['Delta_local'][-1] - gate_params[1]['Delta_local']
    
    # t_ramp_Delta_delta = max(min_dt, abs(Delta_global_delta) / _Delta_slope, abs(Delta_local_delta) / _Delta_slope)
    t_ramp_Delta_delta = max(min_dt, t_Delta_ramp)

    durations_add.append(t_ramp_Delta_delta)
    durations_phi_add.append(t_ramp_Delta_delta) # remain at the original phi while the others ramp down

    phi_original = values_dict['phi'][-1] 
    vals_add['phi'].append(phi_original) # keep the original phi

    vals_add['Delta_global'].append(gate_params[1]['Delta_global'])
    vals_add['Delta_local'].append(gate_params[1]['Delta_local'])  
    vals_add['Omega'].append(values_dict['Omega'][-1]) 

    # now we begin the sequence. everything is fixed but phi
    for s in seq:
        durations_phi_add.append(gate_duration)
        if phi_mode=='binary':
            vals_add['phi'].append(gate_params[s]['phi'])
        else:
            vals_add['phi'].append(s)  # here the sequence stores thde actual value of phi
    
    vals_add['Delta_global'].append(gate_params[1]['Delta_global'])
    vals_add['Delta_local'].append(gate_params[1]['Delta_local'])  
    vals_add['Omega'].append(values_dict['Omega'][-1]) 
    durations_add.append(gate_duration * len(seq))

    # now we ramp down Omega but keep the detunings fixed
    # vals_add['Delta_global'].append(0)
    # vals_add['Delta_local'].append(0)

    ## need to wait at 0 Omega for t_Omega_wait AND t_ramp
    # values_dict['Omega'].extend([0.0, 0.0])
    # durations.extend([t_ramp, Omega_delay_time + t_ramp_Delta])
    # Delta_durations.extend([t_ramp + Omega_delay_time, t_ramp_Delta])
    # phi_durations.append(t_ramp + Omega_delay_time+ t_ramp_Delta)
    
    # if not start_Delta_from0:
    #     vals_add['Delta_global'].append(gate_params[1]['Delta_global'])
    # else:
    #     raise NotImplementedError("start_Delta_from0 (global) option not allowed")

   
    # t_Delta_local_to_0 = gate_params[1]['Delta_local'] / _Delta_slope
    t_Delta_final = max(t_ramp_Delta_delta, min_dt)

    vals_add['phi'].append(phi_original)

    if Omega_delay_time is not None:
        vals_add['Omega'].extend([0,0])
        vals_add['Delta_global'].append(gate_params[1]['Delta_global'])
        vals_add['Delta_local'].append(gate_params[1]['Delta_local'])

    else:
        vals_add['Omega'].append(0)

    vals_add['Delta_local'].append(0)
    vals_add['Delta_global'].append(0)

    # now append the values to the existing values_dict
    for param in params:
        values_dict[param].extend(vals_add[param])
        
    durations.extend(durations_add)
    Delta_durations.extend(durations_add)
    durations_phi.extend(durations_phi_add)

    
    if Omega_delay_time is not None:
        Delta_durations.extend([t_ramp + Omega_delay_time, t_ramp_Delta_delta])
        durations.extend([t_ramp, Omega_delay_time + t_ramp_Delta_delta])
        durations_phi.append(t_ramp + Omega_delay_time+ t_ramp_Delta_delta)
        

    else:
        durations += [t_ramp] 
        Delta_durations[-1] += (t_ramp - t_Delta_final) # need to spend more time at the final gate Delta value
        Delta_durations += [t_Delta_final]
        durations_phi.append(t_ramp)  

   
    # print("Delta_local", values_dict['Delta_local'])
    # print("Delta_global", values_dict['Delta_global'])
    # print("Delta_durations", Delta_durations)
    
    return values_dict, durations, Delta_durations, durations_phi

def get_program(x, h_ls, durations, Delta_durations, durations_phi, values_dict, manual_parallelization, override_local,  x0_y0_offset):

    x = [(xi[1] + x0_y0_offset[1], xi[0] + x0_y0_offset[0]) for xi in x]  # ensure x is a list of tuples (x, y), which we have to check since saving x_pre in the json file
    ## NOTE: we swap x and y since quera says better to put the array in the y direction
    atoms = ba.start.add_position(x)
    
    if not override_local:
        h_ls = list(h_ls)
        if manual_parallelization:
            h_ls = h_ls * int(len(x) / len(h_ls))  # repeat h_ls to match the length of x
    
        # print("Delta_durations inside get_program", Delta_durations)
        # print("Delta_local inside", values_dict['Delta_local'])
        # print("Delta_global inside", values_dict['Delta_global'])
        
        program = ( 
            atoms
            .rydberg.rabi.amplitude.uniform.piecewise_linear(
                durations = durations, values = values_dict['Omega']
                )
            .rydberg.rabi.phase.uniform.piecewise_constant(
                durations = durations_phi, values = values_dict['phi']
                )
            .rydberg.detuning.scale(h_ls).piecewise_linear(
                durations = Delta_durations, values = values_dict['Delta_local']
                )
            .rydberg.detuning.uniform.piecewise_linear(
                durations = Delta_durations, values = values_dict['Delta_global']
                )
        )
    else: # no local detuning
        program = ( 
            atoms
            .rydberg.rabi.amplitude.uniform.piecewise_linear(
                durations = durations, values = values_dict['Omega']
                )
            .rydberg.rabi.phase.uniform.piecewise_constant(
                durations = durations_phi, values = values_dict['phi']
                )
            .rydberg.detuning.uniform.piecewise_linear(
                durations = Delta_durations, values = values_dict['Delta_global']
                )
        )
    return program

def compile_program_oneU(x, ev_params, t_ramp, t_plateau, seq, h_ls, gate_params, manual_parallelization, t_min = 0.05, return_program=True, override_local=False, dt_min = 0.05, x0_y0_offset = (0,0), t_delay=None, start_Delta_from0=False, uniform_Omega_Delta_ramp=False, phi_mode='binary', Delta_local_ramp_time=0.05, Omega_delay_time=None):
    # print("start Delta from 0", start_Delta_from0)
    t_ramp = max(t_min, ev_params['Omega'] / _Omega_slope, abs(ev_params['Delta_global']) / _Delta_slope, abs(ev_params['Delta_local']) / _Delta_slope)
    if not uniform_Omega_Delta_ramp:
        # t_ramp_Delta =  max(t_min, abs(ev_params['Delta_global']) / _Delta_slope, abs(ev_params['Delta_local']) / _Delta_slope) if start_Delta_from0 else max(t_min, abs(ev_params['Delta_local']) / _Delta_slope)
        t_ramp_Delta = max(t_min, Delta_local_ramp_time)
    else:
        t_ramp_Delta = t_ramp
        

    # initialize with the trapezoidal time evolution
    if t_delay is None:
        if Omega_delay_time is not None:
            if np.isclose(Omega_delay_time, 0.0, 1e-7):
                t_Omega_wait = 0
            else:
                t_Omega_wait = max(t_min, Omega_delay_time)
            durations = [t_ramp_Delta + t_Omega_wait, t_ramp, t_plateau]
            Delta_durations = [t_ramp_Delta, t_Omega_wait + t_ramp + t_plateau ]
            phi_durations = [t_min, t_ramp_Delta + t_Omega_wait + t_ramp + t_plateau - t_min]
        else:
            t_ramp_diff = t_ramp - t_ramp_Delta
            durations = [t_ramp, t_plateau]
            Delta_durations = [t_ramp_Delta, t_plateau + t_ramp_diff]
            phi_durations = [t_min, t_plateau + (t_ramp - t_min)]
        # values_dict= {
        #     'Omega': [0, ev_params['Omega'], ev_params['Omega']],
        #     'Delta_global': [0, ev_params['Delta_global'], ev_params['Delta_global']],
        #     'Delta_local': [0, ev_params['Delta_local'], ev_params['Delta_local']],
        #     'phi': [0, ev_params['phi']]
        # }
        if not start_Delta_from0:
            if Omega_delay_time is None:
                Omega_ls = [0.0, ev_params['Omega'], ev_params['Omega']]
            else:
                Omega_ls = [0.0, 0.0, ev_params['Omega'], ev_params['Omega']]
            Delta_local_ls = [0.0, ev_params['Delta_local'], ev_params['Delta_local']]
            Delta_global_ls = [ev_params['Delta_global'], ev_params['Delta_global'], ev_params['Delta_global']]

            values_dict= {
                'Omega': Omega_ls,
                'Delta_global': Delta_global_ls,
                'Delta_local': Delta_local_ls,
                'phi': [0.0, ev_params['phi']]
            }
        else:
            raise NotImplementedError("start_Delta_from0 (global) option not allowed")
            # values_dict= {
            #     'Omega': [0.0, ev_params['Omega'], ev_params['Omega']],
            #     'Delta_global': [ 0.0, ev_params['Delta_global'], ev_params['Delta_global']],
            #     'Delta_local': [0.0, ev_params['Delta_local'], ev_params['Delta_local']],
            #     'phi': [0.0, ev_params['phi']]
            # }
    else: # wait for a value of t_delay with everything set to 0
        # durations = [t_delay, t_ramp, t_plateau]
        # Delta_durations = [t_delay, t_ramp_Delta, t_plateau + t_ramp_diff]
        # phi_t0 = t_min if t_delay < t_min else t_delay
        # phi_durations = [phi_t0, t_plateau + (t_ramp - t_min)]
        # if not start_Delta_from0:
        #     values_dict= {
        #         'Omega': [0.0, 0.0, ev_params['Omega'], ev_params['Omega']],
        #         'Delta_global': [ev_params['Delta_global'], ev_params['Delta_global'], ev_params['Delta_global'], ev_params['Delta_global']],
        #         'Delta_local': [0.0,  0.0, ev_params['Delta_local'], ev_params['Delta_local']],
        #         'phi': [0.0, ev_params['phi']] 
        #     }
        # else:
        #     values_dict= {
        #         'Omega': [0.0, 0.0, ev_params['Omega'], ev_params['Omega']],
        #         'Delta_global': [0.0, 0.0, ev_params['Delta_global'], ev_params['Delta_global'], ev_params['Delta_global']],
        #         'Delta_local': [0.0,  0.0, ev_params['Delta_local'], ev_params['Delta_local']],
        #         'phi': [0.0, ev_params['phi']] 
        #     }
        raise NotImplementedError("t_delay option not implemented")
    
    if len(seq) > 0:
        values_dict, durations, Delta_durations, phi_durations = compile_rand_seq(seq, gate_params, t_ramp, t_ramp_Delta, durations, Delta_durations, phi_durations, values_dict, min_dt=dt_min, start_Delta_from0=start_Delta_from0, phi_mode=phi_mode, Omega_delay_time=Omega_delay_time)
    else: # no gates, just ramp down
        # ramp down from the plateau
        if Omega_delay_time is None:    
            values_dict['Omega'].append(0.0)
            Delta_durations[-1] += t_ramp_diff # account for difference in ramp, need to remain at plateau for longer
            durations.append(t_ramp)
            Delta_durations.append(t_ramp_Delta)
            phi_durations.append(t_ramp)
        else:
            ## need to wait at 0 Omega for t_Omega_wait AND t_ramp
            values_dict['Omega'].extend([0.0, 0.0])
            values_dict['Delta_global'].append(ev_params['Delta_global'])
            values_dict['Delta_local'].append(ev_params['Delta_local'])
            # Delta_durations[-1] += t_ramp + Omega_delay_time # account for difference in ramp, need to remain at plateau for longer
            durations.extend([t_ramp, Omega_delay_time + t_ramp_Delta])
            Delta_durations.extend([t_ramp + Omega_delay_time, t_ramp_Delta])
            phi_durations.append(t_ramp + Omega_delay_time+ t_ramp_Delta)
        
        # if not start_Delta_from0:
        #     values_dict['Delta_global'].append(ev_params['Delta_global'])
        # else:
        #     raise NotImplementedError("start_Delta_from0 (global) option not allowed")

        values_dict['Delta_local'].append(0.0)
        values_dict['Delta_global'].append(0.0)
        values_dict['phi'].append(ev_params['phi'])

    # confirm the timestep between changes is at least min_dt
    durations_cumsum = np.cumsum(durations)
    # print('durations cumsum', durations_cumsum)
    for dt in np.abs(np.diff(durations_cumsum)):
        assert dt > dt_min or np.isclose(dt, dt_min, 1e-7), f"timestep {dt} in durations is less than minimum {dt_min}, have {durations_cumsum}"

        Delta_durations_cumsum = np.cumsum(Delta_durations)
    # print('durations cumsum', Delta_durations_cumsum)
    for dt in np.abs(np.diff(Delta_durations_cumsum)):
        assert dt > dt_min or np.isclose(dt, dt_min, 1e-7), f"timestep {dt} in durations is less than minimum {dt_min}, have {durations_cumsum}"

    # print("Delta durations", Delta_durations)
    # print("Delta local", values_dict['Delta_local'])
    # print("Delta global", values_dict['Delta_global'])
    # print("Omega durations", durations)
    # print("Omega ",  values_dict['Omega'])

    if return_program:
        try:
            return get_program(x, h_ls, durations, Delta_durations, phi_durations, values_dict, manual_parallelization, override_local, x0_y0_offset)
        except Exception as e:
            print("Error in get_program:", e)
            print("durations:", durations)
            print("Delta_durations:", Delta_durations)
            print("phi_durations:", phi_durations)
            print("values_dict:", values_dict)
            raise e
    else:
        return durations, Delta_durations, phi_durations, values_dict

# -------- manual parallelization of arrays
# based on https://github.com/QuEraComputing/bloqade-analog/blob/main/src/bloqade/analog/ir/location/location.py

def create_parallelized_x(x, a_c, bound_x=75, bound_y=128):
    # assumed input x is a list of tuples (x, y)
    x_new = []
    # how many times can we copy in the x and y direction?
    sx0 = max([xi[0] for xi in x]) - min([xi[0] for xi in x]) # length of x span
    sy0 = max([xi[1] for xi in x]) - min([xi[1] for xi in x]) # length of y span

    x_compl = bound_x - sx0 # remaining x space
    y_compl = bound_y - sy0 # remaining y space
    n_c_x = int(np.floor(x_compl / (sx0 + a_c))) + 1  # number of clusters in x direction, including the original
    n_c_y = int(np.floor(y_compl / (sy0 + a_c))) + 1  # number of clusters in y direction, including the original

    for i in range(n_c_x):
        for j in range(n_c_y):
            x_new.extend([(xi[0] + i * a_c, xi[1] + j * a_c) for xi in x])
    
    return x_new
        

# ------- preset experiments
def get_duid(h_ls, x, ev_params, t_plateau, seq, n_shots, gate_params, is_expt_data, dir_root, timestamp, cluster_spacing=70, manual_parallelization=True, override_local=False, preset_opt=None, full_ev=True, x0_y0_offset = (0,0), t_delay=None, start_Delta_from0=False, uniform_Omega_Delta_ramp=False, phi_mode='binary', Delta_local_ramp_time=0.05, Omega_delay_time=None):
    timestamp = int(timestamp)

    store = ExptStore(dir_root)
   
    payload = {
            "h_ls": h_ls,
            "x": x,
            "ev_params": ev_params,
            "t_plateau": t_plateau,
            "seq": seq,
            "n_shots": n_shots, 
            "gate_params": gate_params,
            "is_expt_data": is_expt_data,
            "timestamp": timestamp,
            "cluster_spacing": cluster_spacing,
            "manual_parallelization": manual_parallelization,
            "preset_opt": preset_opt,
            "full_ev": full_ev,
            "override_local": override_local,
            "type": "duid"
    }
    if x0_y0_offset != (0,0):
        payload["x0_y0_offset"] = x0_y0_offset
    if t_delay is not None:
        payload["t_delay"] = t_delay
    if start_Delta_from0:
        payload["start_Delta_from0"] = start_Delta_from0
    if not full_ev:
        del payload["ev_params"]
        del payload["t_plateau"]
    if not uniform_Omega_Delta_ramp:
        payload["uniform_Omega_Delta_ramp"] = uniform_Omega_Delta_ramp
    if phi_mode != 'binary':
        payload["phi_mode"] = phi_mode
    if Delta_local_ramp_time != 0.05:
        payload["Delta_local_ramp_time"] = Delta_local_ramp_time
    if Omega_delay_time is not None:
        payload["Omega_delay_time"] = Omega_delay_time

    uid, added = store.add(payload, timestamp=timestamp)
    return uid, added, payload



def expt_run(h_ls, x, ev_params, t_plateau, t_ramp, seq, n_shots, gate_params, is_expt_data,dir_root, timestamp, data_subdir, full_ev=True, cluster_spacing=70, debug=False, check_postarray=False, manual_parallelization=True, override_local=False, preset_opt=None, save_mode=True, backup_dir=None, x0_y0_offset = (0,0), t_delay=None, start_Delta_from0=False, uniform_Omega_Delta_ramp=False, phi_mode='binary', Delta_local_ramp_time=0.05, Omega_delay_time=None):
    """
    main workhorse function to run real and simualted experiments with bloqade
    """

    # print("backup_dir", backup_dir)
    # print("DEBUG:",debug)

    assert preset_opt in [None, 'ramsey'], f"preset_opt must be None or 'ramsey', got {preset_opt}"
    
    # file management
    uid, added, payload = get_duid(h_ls, x, ev_params, t_plateau, seq, n_shots, gate_params, is_expt_data, dir_root, timestamp, cluster_spacing=cluster_spacing, manual_parallelization=manual_parallelization, override_local=override_local, preset_opt=preset_opt, full_ev=full_ev, x0_y0_offset = x0_y0_offset, t_delay=t_delay, start_Delta_from0=start_Delta_from0, uniform_Omega_Delta_ramp=uniform_Omega_Delta_ramp, phi_mode=phi_mode, Delta_local_ramp_time=Delta_local_ramp_time, Omega_delay_time=Omega_delay_time)
    

    if added and is_expt_data:
        print(f"Added new experiment with UID: {uid}, running on AWS...")

    # if "old" not in data_subdir:
    subdir_path = os.path.join(dir_root, "data", data_subdir)
    filename = os.path.join(subdir_path, f"{uid}.json")
    processed_filename = os.path.join(subdir_path, f"{uid}.npy")
    # else:
    #     filename = os.path.join(dir_root, "data", f"{uid}.json")
    #     processed_filename = os.path.join(dir_root, "data", f"{uid}.npy")

    def process(filename):
        data = ba.load(filename)
        if is_expt_data:
            print("Processing data from file:", filename)
            data.fetch()
            ba.save(data, filename)  # save the fetched data back to the file
        report = data.report()
        # if check_postarray:
        #     print("Postarray:")
        #     report.show()
        bitstrings_ls = process_exp(report, len(h_ls), manual_parallelization=manual_parallelization, is_expt_data=is_expt_data)

        np.save(processed_filename, bitstrings_ls)
        return (bitstrings_ls)
    
    if os.path.exists(filename) or os.path.exists(processed_filename) and not debug and backup_dir is None:
        print(f"Processing: existing data.  h_ls: {h_ls}, ev_params: {ev_params}, t_plateau: {t_plateau}, seq: {seq}, n_shots: {n_shots}")
        # return uid
        # open the file and process it
        if not os.path.exists(processed_filename) or check_postarray:
            bitstrings_ls = process(filename)
        else:
            bitstrings_ls = np.load(processed_filename, allow_pickle=True)
            
        return (uid, bitstrings_ls)
    
    elif save_mode and backup_dir is not None and os.path.exists(backup_dir):
        # process directly
        # check if it failed
        task_status_file = os.path.join(backup_dir, f"task_status.txt")
        # read the txt file
        if os.path.exists(task_status_file):
            with open(task_status_file, "r") as f:
                task_status = f.read().strip()
        # check if it contains "COMPLETED"
        if "COMPLETED" not in task_status:
            print(f"Task status for backupfile provided for {uid} does not indicate COMPLETED, cannot process backup file.")
            bitstrings_ls = []
            bitstrings_ls = common_process(bitstrings_ls, len(h_ls), manual_parallelization=manual_parallelization)
            return (uid,bitstrings_ls)
        else:
            backup_file = os.path.join(backup_dir, f"results.json")
            print(f"Processing data from backup file: {backup_file}")
            bitstrings_ls = process_manual(backup_file, len(h_ls), manual_parallelization=manual_parallelization)
            np.save(processed_filename, bitstrings_ls)
            return (uid, bitstrings_ls)
    
    elif not save_mode: # file doesnt exist, we need to compile the program

        # print("DEBUG-running:",debug)

        if manual_parallelization and cluster_spacing is not None:
            # reassign the lattice x
            x = create_parallelized_x(x, cluster_spacing)

        if preset_opt is None:
            if full_ev:
                program = compile_program_oneU(x, ev_params, t_ramp, t_plateau, seq, h_ls, gate_params, manual_parallelization, override_local=override_local, x0_y0_offset = x0_y0_offset, t_delay=t_delay, start_Delta_from0=start_Delta_from0, uniform_Omega_Delta_ramp=uniform_Omega_Delta_ramp, phi_mode=phi_mode, Delta_local_ramp_time=Delta_local_ramp_time, Omega_delay_time=Omega_delay_time)
            else: # gates only
                raise NotImplementedError("non-full_ev option not implemented")
        elif preset_opt == 'ramsey':
            raise NotImplementedError("preset_opt 'ramsey' not implemented")
        
        if debug:
            program.show()
        
        # now run the program
        ###### HANDLE FAILED TASKS GRACEFULLY WITH RETRY
        if is_expt_data:
            # for k,v in payload.items():
            #     print(f"{k}: {v}")
            # print("UID:", uid)
            # print("data_subdir:", data_subdir)

            max_retries = 3 ## since API will wait until unlocking is fixed, technically only need one retry
            for attempt in range(max_retries):
                try:
                    if s3_destination_folder != 'default':
                        if cluster_spacing is not None and not manual_parallelization:
                            results = program.parallelize(cluster_spacing).braket.aquila().run_async(shots=n_shots, use_experimental=True,  s3_destination_folder=s3_destination_folder)
                        else:
                            results = program.braket.aquila().run(shots=n_shots, use_experimental=True,  s3_destination_folder=s3_destination_folder)
                    else:
                        if cluster_spacing is not None and not manual_parallelization:
                            results = program.parallelize(cluster_spacing).braket.aquila().run_async(shots=n_shots, use_experimental=True)
                        else:
                            results = program.braket.aquila().run(shots=n_shots, use_experimental=True)
                    ba.save(results, filename)
                    print("Results saved to", filename)
                    return (uid)
                except Exception as e:
                    print(f"Attempt {attempt + 1}/{max_retries} failed with error: {e}")
                    if attempt < max_retries - 1:
                        print(f"Retrying task submission...")
                    else:
                        print(f"All retry attempts exhausted. Task failed for UID: {uid}")
                        raise e
        else:
            if cluster_spacing is not None:
                results = program.parallelize(cluster_spacing).bloqade.python().run(n_shots)
            else:
                results = program.bloqade.python().run(n_shots)
            ba.save(results, filename)
            # can run processing immediately if only simulated
            bitstrings_ls = process(filename)
            return (uid, bitstrings_ls)
        
    else:
        print(f"File {filename} does not exist, and save_mode is True, so not compiling or running anything. h_ls: {h_ls}, ev_params: {ev_params}, t_plateau: {t_plateau}, seq: {seq}, n_shots: {n_shots}")
        bitstrings_ls = []
        bitstrings_ls = common_process(bitstrings_ls, len(h_ls), manual_parallelization=manual_parallelization)
        return (uid,bitstrings_ls)
        

if __name__ == '__main__':

    # h_ls = [0.4, .5]
    J = 5.42
    x = [(i*10, 0) for i in range(6)]  # initial positions of the qubits
    Delta_local = -0.5*J
    Delta_mean = 0.5*J
    Delta_global = Delta_mean - 1/2 * Delta_local
    ev_params = {
            'Omega': 15.8,
            'Delta_local': Delta_local,
            'Delta_global': Delta_global,
            'phi': 0,
            't_ramp': 0.0632,
            'a':10,
            'N': 6
        }
    in_gate_params = {
        'gate_duration': 0.06220645598688665, 'n_gates': 2, 
        'Delta_local': -102.7, 'Delta_global': 26.7,
        'n_U': 1, 'n_shots': 100
    }

    h_ls = [
            0.24996654759766535,
            0.09286441773368292,
            0.03781977537875725,
            0.5848043295638031,
            0.1759548522567862,
            0.3655913041729373
        ]

    seq = [2,1,2,2,1,2,1,2,2,1,1,2]
            
    t_plateau = 0.5
    t_ramp = 0.0632
    n_shots = 10
    dir_root = 'test_random_bp_prep'
    data_subdir = 'test_data'
    is_expt_data = False
    # import time
    # timestamp = time.time()
    timestamp = 0
    Delta_local_ramp_time=0.05
    Omega_delay_time=0.0
    expt_run(h_ls, x, ev_params, t_plateau, t_ramp, seq, n_shots, in_gate_params, is_expt_data,dir_root, timestamp, data_subdir, full_ev=True, cluster_spacing=None, debug=True, check_postarray=False, manual_parallelization=False, override_local=False, preset_opt=None, save_mode=False, backup_dir=None, x0_y0_offset = (0,0), t_delay=None, start_Delta_from0=False, uniform_Omega_Delta_ramp=False, phi_mode='binary', Delta_local_ramp_time=Delta_local_ramp_time, Omega_delay_time=Omega_delay_time)
   
    
    
    
