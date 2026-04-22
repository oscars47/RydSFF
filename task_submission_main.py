## this is the main (head) script to use to run the chaotic vs localized experiment
## for the most recent experiment (Nov 6 2025), we use nqcc_cost.py to generate the tasks 
from master_params_rbp import execute_bloqade_task_chunk, read_expt_task, get_subdirname
import datetime as _dt_logging
import diagnose_driver, master_params_rbp, QuEraToolbox, process_rbp, process_rbp_calib_helper, chain_benchmark
import numpy as np
import os, json, tempfile
import shutil
from pathlib import Path
from chain_benchmark import run_benchmark_expt
from diagnose_driver import save_expts, process_expts
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
from fig_styling import style_axis

import datetime as _dt
import time as _time
import traceback as _traceback
from tqdm import tqdm
from datetime import datetime, timezone

from parse_majd_calib import parse_majd

_aquila_time_rules = {
    'time_per_100_shots': 60,
    'max_chunk_time':36000,  # 10 hours
}

my_bucket = "qeg-braket-us-east-1-00"
my_prefix = "simulation-results/ors35"

_FORCE_TRY_HYBRID = False ### CHANGE IF YOU DONT WANT TO TRY RUNNING IN HYBRID MODE

_RABI_CALIB_LAST_UTC_DATE = None

def _maybe_run_rabi_calibration(cancel_Rabi, dir_root, chunk_idx, task_name, is_expt_data, n_shots_rabi, mode_label=""):
    global _RABI_CALIB_LAST_UTC_DATE
    if cancel_Rabi:
        print("Rabi calibration canceled via cancel_Rabi flag.")
        return False

    today_utc = _dt.datetime.now(tz=_dt.timezone.utc).date()
    if _RABI_CALIB_LAST_UTC_DATE == today_utc:
        print(f"Skipping Rabi calibration: already ran today (UTC {today_utc}).")
        return False

    label = f" {mode_label}" if mode_label else ""
    print(f"Running Rabi calibration{label}...")
    run_benchmark_expt(
        dir_root,
        chunk_idx,
        task_name,
        is_expt_data=is_expt_data,
        n_shots=n_shots_rabi,
        uniform_Omega_Delta_ramp=False,
        start_Delta_from0=False,
    )
    _RABI_CALIB_LAST_UTC_DATE = today_utc
    return True

def _log_error_to_file(dir_root, exception, context_info=""):
    """
    Log an exception to a timestamped file in the error_logs subdirectory.
    """
    try:
        error_logs_dir = os.path.join(dir_root, "error_logs")
        os.makedirs(error_logs_dir, exist_ok=True)
        
        timestamp = _dt_logging.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"error_{timestamp}.txt"
        filepath = os.path.join(error_logs_dir, filename)
        
        with open(filepath, "w") as f:
            f.write(f"Error Log - {_dt_logging.datetime.now().isoformat()}\n")
            f.write(f"Context: {context_info}\n")
            f.write(f"Exception Type: {type(exception).__name__}\n")
            f.write(f"Exception Message: {str(exception)}\n")
            f.write("\nFull Traceback:\n")
            f.write(_traceback.format_exc())
        
        print(f"Error logged to: {filepath}")
    except Exception as log_error:
        print(f"Failed to log error: {log_error}")

def download_and_merge_results(job, target_root="."):
    """
    - Download model.tar.gz for `job` and extract into a temp dir.
    - If the archive contains a top-level dir equal to job.name,
      strip that level so you don't get job.name/job.name/...
    - Copy files directly into target_root, preserving subdirectory structure.
    - Only copy files that don't already exist locally (no overwriting).
    """
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"{job.name}-results-"))
    job.download_result(extract_to=str(tmp_dir))

    # --- determine the true source root ---
    # If the tarball has tmp_dir/job.name/, use that as src_root;
    # otherwise, use tmp_dir directly.
    candidate = tmp_dir / job.name
    if candidate.is_dir():
        src_root = candidate
    else:
        src_root = tmp_dir

    target_root = Path(target_root)
    target_root_name = target_root.name
    
    # Check if the source contains the target_root directory structure
    # If so, we need to go one level deeper to avoid target_root/target_root
    target_root_candidate = src_root / target_root_name
    if target_root_candidate.is_dir():
        print(f"Found nested target_root directory '{target_root_name}' in results, using it as source root")
        src_root = target_root_candidate

    target_root.mkdir(parents=True, exist_ok=True)

    copied_files = 0
    skipped_files = 0

    for src in src_root.rglob("*"):
        if src.is_dir():
            continue

        # Get relative path from source root
        rel = src.relative_to(src_root)
        # Copy directly to target_root (not target_root/job.name)
        dst = target_root / rel
        
        # Create parent directories if they don't exist
        dst.parent.mkdir(parents=True, exist_ok=True)

        # Only copy if file doesn't already exist
        if dst.exists():
            print(f"Skipping existing file: {dst}")
            skipped_files += 1
        else:
            shutil.copy2(src, dst)
            print(f"Copied: {src} -> {dst}")
            copied_files += 1

    print(f"Download and merge complete: {copied_files} files copied, {skipped_files} files skipped (already exist)")
    shutil.rmtree(tmp_dir, ignore_errors=True)


def _read_all_json_files_locally(dir_root, data_subdir):
    """
    Read all JSON files from dir_root/tasks/ and dir_root/data/ directories locally.
    Returns a dictionary with the structure: {subdir_name: {filename: json_data}}
    This ensures all necessary JSON configuration files are passed to the AWS environment.
    """
    try:
        # Directories to search for JSON files
        source_dirs = [
            os.path.join(dir_root, "tasks"),
            os.path.join(dir_root, "data", data_subdir)
        ]
        
        all_json_data = {}
        
        for source_dir in source_dirs:
            if not os.path.exists(source_dir):
                print(f"Warning: Source directory not found: {source_dir}")
                all_json_data[source_dir] = {}
                continue
                
            subdir_json_data = {}
            
            # Find all JSON files in the source directory
            for filename in os.listdir(source_dir):
                if filename.endswith('.json'):
                    source_path = os.path.join(source_dir, filename)
                    
                    try:
                        with open(source_path, 'r') as f:
                            json_data = json.load(f)
                            subdir_json_data[filename] = json_data
                            print(f"Read JSON file: {source_path}")
                    except Exception as e:
                        print(f"Error reading JSON file {source_path}: {e}")
            
            all_json_data[source_dir] = subdir_json_data
        
        total_files = sum(len(subdir_data) for subdir_data in all_json_data.values())
        print(f"Successfully read {total_files} JSON files locally")
        return all_json_data
        
    except Exception as e:
        print(f"Error reading JSON files locally: {e}")
        return {}


def _write_json_files_in_aws_environment(all_json_data):
    """
    Write the JSON data to files in the AWS hybrid job environment.
    Creates the same directory structure as the local environment.
    """
    try:
        
        written_files = []
        
        for subdir_name, subdir_data in all_json_data.items():
            if not subdir_data:
                continue
                
            # Create subdirectory
            os.makedirs(subdir_name, exist_ok=True)
            
            # Write each JSON file
            for filename, json_data in subdir_data.items():
                target_path = os.path.join(subdir_name, filename)
                
                with open(target_path, 'w') as f:
                    json.dump(json_data, f, indent=2)
                    
                written_files.append(target_path)
                print(f"Wrote JSON file in AWS environment: {target_path}")
        
        print(f"Successfully wrote {len(written_files)} JSON files in AWS environment")
        return True
        
    except Exception as e:
        print(f"Error writing JSON files in AWS environment: {e}")
        return False

if _FORCE_TRY_HYBRID:
    try:
        from braket.devices import Devices
        from braket.jobs import (
            InstanceConfig,
            hybrid_job,
        )
       
        @hybrid_job(
            device = Devices.QuEra.Aquila, 
            dependencies="requirements_revised.txt",
            instance_config=InstanceConfig(instanceType='ml.m5.large', instanceCount=1),
            wait_until_complete=True,
            include_modules=[diagnose_driver, master_params_rbp, QuEraToolbox, process_rbp, process_rbp_calib_helper, chain_benchmark], 
            image_uri="292282985366.dkr.ecr.us-east-1.amazonaws.com/amazon-braket-base-jobs:1.0-cpu-py310-ubuntu22.04" ## due to update on Jan 21, forces hyrbid to run in python 3.12. see: https://docs.aws.amazon.com/braket/latest/developerguide/braket-jobs-script-environment.html; https://docs.aws.amazon.com/braket/latest/developerguide/braket-troubleshooting-python312.html#braket-troubleshooting-python312-hybrid-job
        )
        def run_chunk(chunk_idx, num_ham_in_chunk, task_name, name, is_expt_data, timestamp, dir_root, save_mode, all_json_data, cancel_Rabi=False):
    
            # Write all JSON files to the AWS environment using the data passed from local environment
            # This ensures all necessary configuration files are available in the AWS environment
            _write_json_files_in_aws_environment(all_json_data)

            if not save_mode: ## run Rabi
                print("Few shot rabi:", cancel_Rabi)
                n_shots_rabi = 100
                _maybe_run_rabi_calibration(
                    cancel_Rabi,
                    dir_root,
                    chunk_idx,
                    task_name,
                    is_expt_data,
                    n_shots_rabi,
                    mode_label="in hybrid mode",
                )
        
            return execute_bloqade_task_chunk(chunk_idx, num_ham_in_chunk, task_name, name, is_expt_data, timestamp, dir_root, force_recompute=True, debug=False, preset_opt=None, data_subdir=None, save_mode=save_mode, backup_dirs=None, after_how_many_ham_run_check=None, ham_check_dir_main = None, allow_override_name = False)
    
        _IS_AWS_ENVIRONMENT = True
        
    except Exception as e:
        print(f"Exception occurred while setting up hybrid job: {e}")
        def run_chunk(chunk_idx, num_ham_in_chunk, task_name, name, is_expt_data, timestamp, dir_root, save_mode, all_json_data=None, cancel_Rabi=False):
            # In local environment, all_json_data is not needed since we have direct access to files
            if not save_mode: ## run Rabi
                n_shots_rabi = 100
                _maybe_run_rabi_calibration(
                    cancel_Rabi,
                    dir_root,
                    chunk_idx,
                    task_name,
                    is_expt_data,
                    n_shots_rabi,
                )

            return execute_bloqade_task_chunk(chunk_idx, num_ham_in_chunk, task_name, name, is_expt_data, timestamp, dir_root, force_recompute=True, debug=False, preset_opt=None, data_subdir=None, save_mode=save_mode, backup_dirs=None, after_how_many_ham_run_check=None, ham_check_dir_main = None, allow_override_name = False)
        
        _IS_AWS_ENVIRONMENT = False
else:
    def run_chunk(chunk_idx, num_ham_in_chunk, task_name, name, is_expt_data, timestamp, dir_root, save_mode, all_json_data=None, cancel_Rabi=False):
        # In local environment, all_json_data is not needed since we have direct access to files
        if not save_mode: ## run Rabi
            n_shots_rabi = 100
            _maybe_run_rabi_calibration(
                cancel_Rabi,
                dir_root,
                chunk_idx,
                task_name,
                is_expt_data,
                n_shots_rabi,
            )

        return execute_bloqade_task_chunk(chunk_idx, num_ham_in_chunk, task_name, name, is_expt_data, timestamp, dir_root, force_recompute=True, debug=False, preset_opt=None, data_subdir=None, save_mode=save_mode, backup_dirs=None, after_how_many_ham_run_check=None, ham_check_dir_main = None, allow_override_name = False)
        
    _IS_AWS_ENVIRONMENT = False


def _get_calibration_windows_utc(day: _dt.date):
    """
    Return a list of (start_datetime, end_datetime) UTC calibration
    windows for a given day.

    QuEra calibration schedule in UTC:
        - Sunday-Monday (weekday 5,6,0): 00:00-01:00, 12:00-13:00
        - Friday (weekday 4): 12:00-13:00
        - Tuesday (weekday 1) 00:00 - 01:00 and 12:00-13:00 is calibration, then internal maintenance 13:00-23:59.
        - Thursday (weekday 3): 12:00-13:00 is calibration, then internal maintenance 13:00-23:59. 
        - Wednesday (weekday 2): 12:00-18:00
    """
    wd = day.weekday()  # Monday=0, Sunday=6

    def dt(h, m=0):
        return _dt.datetime(day.year, day.month, day.day, h, m, tzinfo=_dt.timezone.utc)

    if wd in (0, 5, 6):  # Mon, Sat, Sun
        return [(dt(0), dt(1)), (dt(12), dt(13))]
    elif wd ==4 :  # Fri
        return [(dt(12), dt(13))]
    elif wd == 1:      # Tue
        return [(dt(0), dt(1)), (dt(12), dt(23, 59))]
    elif wd == 3 :      # Thurs
        return [(dt(12), dt(23, 59))]
    elif wd == 2:           # Wed
        return [(dt(12), dt(18))]
    else:
        return []  

def _wait_until_outside_calibration_window(chunk_duration_s):
    """
    Block until we are OUTSIDE any calibration window and have enough time
    to complete a chunk of duration `chunk_duration_s` (seconds) before 
    the next calibration window starts.

    Strategy:
      - If currently inside a calibration window, wait until it ends.
      - If outside calibration windows, check if we have enough time before 
        the next one starts. If not, wait until after the next window ends.
    """
    while True:
        now = _dt.datetime.now(tz=_dt.timezone.utc)
        today = now.date()
        
        # Get windows for today and tomorrow 
        windows_today = _get_calibration_windows_utc(today)
        tomorrow = today + _dt.timedelta(days=1)
        windows_tomorrow = _get_calibration_windows_utc(tomorrow)
        
        all_windows = windows_today + windows_tomorrow
        print(f"Current time UTC: {now}.")

        # Check if we're currently inside any calibration window
        inside_window = False
        for start, end in all_windows:
            if start <= now < end:
                # We're inside a calibration window - wait until it ends
                sleep_s = (end - now).total_seconds()
                print(f"Inside calibration window until {end}. Sleeping for {sleep_s:.0f} seconds.")
                if sleep_s > 0:
                    _time.sleep(sleep_s)
                inside_window = True
                break
        
        if inside_window:
            continue  # Recheck after sleeping
        
        # We're outside all current windows. Find the next upcoming window.
        upcoming_windows = [(start, end) for start, end in all_windows if start > now]
        
        if not upcoming_windows:
            # No upcoming windows in next 2 days - safe to run
            print(f"No upcoming calibration windows. Safe to run chunk.")
            return
        
        # Find the earliest upcoming window
        next_start, next_end = min(upcoming_windows, key=lambda x: x[0])
        time_until_next_window = (next_start - now).total_seconds()
        
        if time_until_next_window >= chunk_duration_s:
            # Enough time to complete chunk before next calibration window
            print(f"Enough time ({time_until_next_window:.0f}s) before next calibration window at {next_start}. Safe to run chunk.")
            return
        else:
            # Not enough time - wait until after the next window ends
            sleep_s = (next_end - now).total_seconds()
            print(f"Not enough time ({time_until_next_window:.0f}s < {chunk_duration_s:.0f}s) before next calibration window. "
                  f"Waiting until after window ends at {next_end}. Sleeping for {sleep_s:.0f} seconds.")
            if sleep_s > 0:
                _time.sleep(sleep_s)


def _calculate_chunk_params(task_name, dir_root, override_num_ham_per_chunk=None):
    """
    Returns: (time_per_ham, num_ham_per_chunk, total_hams, num_chunks, hams_in_chunk_func)
    """
    h_ls_pre, x_pre, t_plateau_ls, seq_ls_pre_all, base_params, Delta_mean_ls, Delta_local_ls, gate_params_all, cluster_spacing, manual_parallelization, override_local, x0_y0_offset, t_delay, start_Delta_from0, uniform_Omega_Delta_ramp, phi_mode, Delta_local_ramp_time, Omega_delay_time = read_expt_task(task_name, dir_root)
    
    n_shots = gate_params_all[0][0]['n_shots']
    n_U = gate_params_all[0][0]['n_U']
    n_t_pts = len(t_plateau_ls)

    # Time for a single Hamiltonian (one element of h_ls_pre)
    time_per_ham = n_shots * n_U * n_t_pts * _aquila_time_rules['time_per_100_shots'] / 100.0  # seconds

    max_time_per_chunk = _aquila_time_rules['max_chunk_time']

    if override_num_ham_per_chunk is None:
        num_ham_per_chunk = int(max_time_per_chunk // time_per_ham)
        if num_ham_per_chunk < 1:
            raise ValueError(
                f"Chunk duration exceeds max chunk time: "
                f"time_per_ham={time_per_ham:.2f}s, max_time_per_chunk={max_time_per_chunk}s"
            )
    else:
        num_ham_per_chunk = override_num_ham_per_chunk

    total_hams = len(h_ls_pre)
    num_chunks = int(np.ceil(total_hams / num_ham_per_chunk))

    def hams_in_chunk(idx):
        start = idx * num_ham_per_chunk
        end = min(start + num_ham_per_chunk, total_hams)
        return max(0, end - start)
    
    return time_per_ham, num_ham_per_chunk, total_hams, num_chunks, hams_in_chunk


def schedule_run_all_chunks(task_name, name, is_expt_data, timestamp, dir_root, specific_chunk_idx=None, override_num_ham_per_chunk=1, cancel_Rabi=False):
    # Read all JSON files locally before submitting jobs to AWS
    print("[schedule_run_all_chunks] Reading JSON files locally...")
    data_subdir = get_subdirname(name, task_name, timestamp, preset_opt=None)
    all_json_data = _read_all_json_files_locally(dir_root, data_subdir)
    
    time_per_ham, num_ham_per_chunk, total_hams, num_chunks, hams_in_chunk = _calculate_chunk_params(task_name, dir_root, override_num_ham_per_chunk=override_num_ham_per_chunk)


    # Validate that we don't have empty chunks at the end
    # actual_num_chunks = 0
    # for i in range(num_chunks):
    #     if hams_in_chunk(i) > 0:
    #         actual_num_chunks = i + 1
    
    print(f"[schedule_run_all_chunks] Total Hamiltonians: {total_hams}, "
          f"Hams per chunk: {num_ham_per_chunk}, "
          f"Time per chunk: {time_per_ham * num_ham_per_chunk} ")

    if specific_chunk_idx is None:
        starting_ham_idx= 0
        for chunk_idx in range(num_chunks):
            n_ham_chunk = hams_in_chunk(chunk_idx)
            if n_ham_chunk <= 0:
                print(f"[schedule_run_all_chunks] Skipping chunk {chunk_idx}: no Hamiltonians")
                continue
            if not cancel_Rabi:
                chunk_duration_s = n_ham_chunk * time_per_ham + 2700 # 45 extra mins for the Rabi callibration (should be 30 min) and buffer
            else: # remove extra time if Rabi is cancelled
                chunk_duration_s = n_ham_chunk * time_per_ham

            print(f"[schedule_run_all_chunks] Processing chunk {chunk_idx}: {n_ham_chunk} Hamiltonians. Duration: {chunk_duration_s} seconds")
            
            # Wait until outside calibration windows with enough time
            if _IS_AWS_ENVIRONMENT:
                _wait_until_outside_calibration_window(chunk_duration_s)

            # Run with protection so a failure doesn't kill the whole loop
            try:
                if _IS_AWS_ENVIRONMENT:
                    job = run_chunk(starting_ham_idx, n_ham_chunk, task_name, name, is_expt_data, timestamp, dir_root, save_mode=False, all_json_data=all_json_data, cancel_Rabi=cancel_Rabi)
                    download_and_merge_results(job, target_root=dir_root)
                else:
                    run_chunk(starting_ham_idx, n_ham_chunk, task_name, name, is_expt_data, timestamp, dir_root, save_mode=False, all_json_data=all_json_data, cancel_Rabi=cancel_Rabi)
                    
            except Exception as exc:
                print(f"[schedule_run_all_chunks] Error in run_chunk(starting_ham_idx={starting_ham_idx}): {exc}")
                _traceback.print_exc()
                # Log error to file
                context = f"Main loop - chunk_idx={chunk_idx}, starting_ham_idx={starting_ham_idx}, n_ham_chunk={n_ham_chunk}"
                _log_error_to_file(dir_root, exc, context)
                # Continue with next chunk
            starting_ham_idx += n_ham_chunk
    elif isinstance(specific_chunk_idx, list):
        # Handle list of specific chunk indices
        for chunk_idx in specific_chunk_idx:
            assert chunk_idx < num_chunks, f"chunk_idx {chunk_idx} exceeds maximum {num_chunks-1}"
            n_ham_chunk = hams_in_chunk(chunk_idx)
            if n_ham_chunk <= 0:
                print(f"[schedule_run_all_chunks] Skipping chunk {chunk_idx}: no Hamiltonians")
                continue
            
            # starting Hamiltonian index for this specific chunk
            starting_ham_idx = 0
            for i in range(chunk_idx):
                starting_ham_idx += hams_in_chunk(i)
            
            print(f"[schedule_run_all_chunks] Processing specific chunk {chunk_idx}: {n_ham_chunk} Hamiltonians, starting from Hamiltonian {starting_ham_idx}")
            if not cancel_Rabi:
                chunk_duration_s = n_ham_chunk * time_per_ham + 2700 # 45 extra mins for the Rabi callibration (should be 30 min) and buffer
            else: # remove extra time if Rabi is cancelled
                chunk_duration_s = n_ham_chunk * time_per_ham

            if _IS_AWS_ENVIRONMENT:
                _wait_until_outside_calibration_window(chunk_duration_s)

            try:
                if _IS_AWS_ENVIRONMENT:
                    job = run_chunk(starting_ham_idx, n_ham_chunk, task_name, name, is_expt_data, timestamp, dir_root, save_mode=False, all_json_data=all_json_data, cancel_Rabi=cancel_Rabi)
                    download_and_merge_results(job, target_root=dir_root)
                else:
                    run_chunk(starting_ham_idx, n_ham_chunk, task_name, name, is_expt_data, timestamp, dir_root, save_mode=False, all_json_data=all_json_data, cancel_Rabi=cancel_Rabi)
                    
            except Exception as exc:
                print(f"[schedule_run_all_chunks] Error in run_chunk(starting_ham_idx={starting_ham_idx}): {exc}")
                _traceback.print_exc()
                # Log error to file
                context = f"Specific chunk list - chunk_idx={chunk_idx}, starting_ham_idx={starting_ham_idx}, n_ham_chunk={n_ham_chunk}"
                _log_error_to_file(dir_root, exc, context)
                # Continue with next chunk
    else:
        # single specific chunk index
        assert specific_chunk_idx < num_chunks, f"specific_chunk_idx {specific_chunk_idx} exceeds maximum {num_chunks-1}"
        n_ham_chunk = hams_in_chunk(specific_chunk_idx)
        if n_ham_chunk <= 0:
            print(f"[schedule_run_all_chunks] specific_chunk_idx={specific_chunk_idx} has no Hamiltonians to run.")
            return
        
        # starting Hamiltonian index for this specific chunk
        starting_ham_idx = 0
        for i in range(specific_chunk_idx):
            starting_ham_idx += hams_in_chunk(i)
        
        print(f"[schedule_run_all_chunks] Processing specific chunk {specific_chunk_idx}: {n_ham_chunk} Hamiltonians, starting from Hamiltonian {starting_ham_idx}")
        if not cancel_Rabi:
            chunk_duration_s = n_ham_chunk * time_per_ham + 2700 # 45 extra mins for the Rabi callibration (should be 30 min) and buffer
        else: # remove extra time if Rabi is cancelled
            chunk_duration_s = n_ham_chunk * time_per_ham

        if _IS_AWS_ENVIRONMENT:
            _wait_until_outside_calibration_window(chunk_duration_s)

        try:
            if _IS_AWS_ENVIRONMENT:
                job = run_chunk(starting_ham_idx, n_ham_chunk, task_name, name, is_expt_data, timestamp, dir_root, save_mode=False, all_json_data=all_json_data, cancel_Rabi=cancel_Rabi)
                download_and_merge_results(job, target_root=dir_root)
            else:
                run_chunk(starting_ham_idx, n_ham_chunk, task_name, name, is_expt_data, timestamp, dir_root, save_mode=False, all_json_data=all_json_data, cancel_Rabi=cancel_Rabi)
                
        except Exception as exc:
            print(f"[schedule_run_all_chunks] Error in run_chunk(starting_ham_idx={starting_ham_idx}): {exc}")
            _traceback.print_exc()
            # Log error to file
            context = f"Single chunk - specific_chunk_idx={specific_chunk_idx}, starting_ham_idx={starting_ham_idx}, n_ham_chunk={n_ham_chunk}"
            _log_error_to_file(dir_root, exc, context)

   
def save_chunks(task_name, name, is_expt_data, timestamp, dir_root, max_num_chunks=None, override_num_ham_per_chunk=1):
    """
    Save chunks without waiting for calibration windows.
    Processes up to max_num_chunks chunks with save_mode=True.
    """
    # Read all JSON files locally before processing chunks
    print("[save_chunks] Reading JSON files locally...")
    data_subdir = get_subdirname(name, task_name, timestamp, preset_opt=None)
    all_json_data = _read_all_json_files_locally(dir_root, data_subdir)
    
    time_per_ham, num_ham_per_chunk, total_hams, num_chunks, hams_in_chunk = _calculate_chunk_params(task_name, dir_root, override_num_ham_per_chunk=override_num_ham_per_chunk)
    
    # Ensure we don't try to save more chunks than exist
    if max_num_chunks is not None:
        actual_max_chunks = min(max_num_chunks, num_chunks)
    else:
        actual_max_chunks = num_chunks
    
    print(f"[save_chunks] Total Hamiltonians: {total_hams}, "
          f"Hams per chunk: {num_ham_per_chunk}, "
          f"Saving chunks 0 to {actual_max_chunks-1}")
    
    for chunk_idx in range(actual_max_chunks):
        n_ham_chunk = hams_in_chunk(chunk_idx)
        if n_ham_chunk <= 0:
            print(f"[save_chunks] Skipping chunk {chunk_idx}: no Hamiltonians")
            continue
        
        print(f"[save_chunks] Saving chunk {chunk_idx}: {n_ham_chunk} Hamiltonians")
        
        try:
            run_chunk(chunk_idx, n_ham_chunk, task_name, name, is_expt_data, timestamp, dir_root, save_mode=True, all_json_data=all_json_data)
            print(f"[save_chunks] Successfully saved chunk {chunk_idx}")
        except Exception as exc:
            print(f"[save_chunks] Error saving chunk {chunk_idx}: {exc}")
            _traceback.print_exc()
            # Continue with next chunk even if this one fails

def ret_all_suids(dir_root):
    search_dir = os.path.join(dir_root, "data")
    suid_ls = []
    # look for all .json files in search_dir (only direct files, not subdirectories)
    # extract the name (before.json) and run save_expt_wrapper()
    if os.path.exists(search_dir):
        for file in os.listdir(search_dir):
            if file.endswith(".json") and os.path.isfile(os.path.join(search_dir, file)):
                suid = file[:-5]  # remove .json
                suid_ls.append(suid)
    else:
        print(f"Warning: Search directory not found: {search_dir}")
    return suid_ls

def save_all_suids(dir_root):
    suid_ls = ret_all_suids(dir_root)
    print(f"Found {len(suid_ls)} SUIDs to save.")
    for suid in suid_ls:
        try:
            print(f"Saving expt for SUID: {suid}")
            save_expts(suid, dir_root)
        except Exception as exc:
            print(f"Error saving expt for SUID {suid}: {exc}")
            _traceback.print_exc()
            # Continue with next SUID
                    
def extract_readouterror_rabi(main_dir,chosen_task, total_ens, N=6, force_recompute=True):
    main_rabi_csv = f"{N}rabi_calib.csv"
    rabi_csv_path = os.path.join(main_dir, main_rabi_csv)
    if force_recompute:
        ### use do_preset ###
        # step 1: run processing
        suid_ls = ret_all_suids(main_dir)
        print(suid_ls)
        res_idx_ls = list(range(N//2))
        for suid in suid_ls:
            print("SUID:", suid)
            print(f"Processing readout error extraction for SUID: {suid}")
            try:
                process_expts(suid, dir_root, None, None, None, None, ax_ret=False, colors_main = None, q_index=res_idx_ls, show_qutip=True, num_t_qutip_points=100, fontsize=25, default_color=True, global_csv=rabi_csv_path)
                print(f"Extracted readout error for SUID: {suid}")
            except Exception as exc:
                print(f"Error extracting readout error for SUID {suid}: {exc}")
                _traceback.print_exc()
                # Continue with next SUID

        # step 2: read rabi_csv: access the task and timestamp columns, match to the dir of calib data to extract the resonant index: this is the qubit for which we extract the readout error
        ## step 2 alternative: read in what they give us from expt
        ## continue logic here: we need to build the list: 
        # - read the "name" column, select out the entries that contain chosen_task in the name
        # expect name to be: q{idx}-chunk{n}-task_{chosen_task}
        # create a of lists list: for all chunks n, give a list of idx = 0, 1, ..., up to the highest given. if idx < N-1, then just fill with  0.0 for the rest. if maximum nuber of chunks < total_ens, fill with 0.0 lists
        df_rabi = pd.read_csv(rabi_csv_path)
        df_filtered = df_rabi[df_rabi['name'].str.contains(chosen_task)]
        readout_err_dict = {
            "epsilon_r_ls": [[0.0 for _ in range(N)] for _ in range(total_ens)], 
            "epsilon_r_unc_ls": [[0.0 for _ in range(N)] for _ in range(total_ens)],
            "epsilon_g_ls": [[0.0 for _ in range(N)] for _ in range(total_ens)],
            "epsilon_g_unc_ls": [[0.0 for _ in range(N)] for _ in range(total_ens)],
            'Omega_ls': [[0.0 for _ in range(N)] for _ in range(total_ens)],
            'Omega_unc_ls': [[0.0 for _ in range(N)] for _ in range(total_ens)],
            'timestamp': [0.0 for _ in range(total_ens)],
        }
        max_chunk_num = -1
        for _, row in df_filtered.iterrows():
            name = row['name']
            parts = name.split('-')
            q_part = parts[0]  # e.g., 'q2'
            chunk_part = parts[1]  # e.g., 'chunk3'
            q_idx = int(q_part[1:])  # extract index after 'q'
            chunk_num = int(chunk_part[5:])  # extract number after 'chunk'
            if chunk_num > max_chunk_num:
                max_chunk_num = chunk_num
            epsilon_r = row['epsilon_r']
            epsilon_r_unc = row['epsilon_r_unc']
            epsilon_g = row['epsilon_g']
            epsilon_g_unc = row['epsilon_g_unc']
            Omega = row['Omega_eff_expt']
            Omega_unc = row['Omega_eff_expt_unc']
            timestamp_row = row['timestamp']
            if chunk_num < total_ens and q_idx < N:
                readout_err_dict["epsilon_r_ls"][chunk_num][q_idx] = epsilon_r
                readout_err_dict["epsilon_r_unc_ls"][chunk_num][q_idx] = epsilon_r_unc
                readout_err_dict["epsilon_g_ls"][chunk_num][q_idx] = epsilon_g
                readout_err_dict["epsilon_g_unc_ls"][chunk_num][q_idx] = epsilon_g_unc
                readout_err_dict["Omega_ls"][chunk_num][q_idx] = Omega
                readout_err_dict["Omega_unc_ls"][chunk_num][q_idx] = Omega_unc
                readout_err_dict["timestamp"][chunk_num] = timestamp_row

        print(f"Extracted readout error data for {max_chunk_num+1} chunks.")
        # save this
        save_path = os.path.join(main_dir, f"readout_error_rabi_extracted_{chosen_task}.json")
        with open(save_path, 'w') as f:
            json.dump(readout_err_dict, f, indent=2)
        print(f"Saved extracted readout error data to {save_path}")
    else:
        # read existing
        save_path = os.path.join(main_dir, f"readout_error_rabi_extracted_{chosen_task}.json")
        with open(save_path, 'r') as f:
            readout_err_dict = json.load(f)
        print(f"Loaded existing readout error data from {save_path}")
    return readout_err_dict['epsilon_r_ls'], readout_err_dict['epsilon_r_unc_ls'], readout_err_dict['epsilon_g_ls'], readout_err_dict['epsilon_g_unc_ls'], readout_err_dict['Omega_ls'], readout_err_dict['Omega_unc_ls']
    

### func to compare majd and my calibs
def compare_calibs(my_calib_path, majd_root, majd_output_path="majd_calibs.json", majd_indices_compare=[2, 6, 10], fontsize=20, dir_root='gaugamela_chunk_expt', rerun_majd=False, avg_comp_time=(2025, 12, 28)):
    "my_calib_path : expects readout_error_rabi_extracted_{chosen_task}.json -- FUTURE NOTE: WANT TO COMBINE BOTH THE CHAOTIC AND LOCALIZED CALIB FILES TOGETHER"

    year_comp, month_comp, day_comp = avg_comp_time
    # Create date string for filenames (e.g., "dec28" from (2025, 12, 28))
    month_abbr = datetime(year_comp, month_comp, day_comp, tzinfo=timezone.utc).strftime('%b').lower()
    date_str = f"{month_abbr}{day_comp:02d}"
    
    import time
    start_time = time.time()

    if rerun_majd or not os.path.exists(majd_output_path):
        print(f"[compare_calibs] Parsing majd data...")
        parse_majd(majd_root = majd_root, output_path= majd_output_path)
        print(f"[compare_calibs] Parsing completed in {time.time() - start_time:.2f}s")

    assert os.path.exists(my_calib_path) , f"my_calib_path {my_calib_path} does not exist."

    # read in the files
    print(f"[compare_calibs] Reading JSON files...")
    read_start = time.time()
    with open(majd_output_path, 'r') as f:
        majd_calib = json.load(f)
    with open(my_calib_path, 'r') as f:
        my_calib = json.load(f)
    print(f"[compare_calibs] JSON files loaded in {time.time() - read_start:.2f}s")

    print("read the files")

    my_timestamps = my_calib['timestamp']
    majd_timestamps = majd_calib['timestamp']
    
    # Convert timestamps to datetime objects for plotting
    # Handle both Unix timestamps (int/float) and ISO format strings
    from matplotlib.dates import date2num
    
    def convert_to_datetime(ts):
        if ts == 0 or ts is None:
            return None
        try:
            # Try Unix timestamp first
            if isinstance(ts, (int, float)):
                return date2num(datetime.fromtimestamp(ts, tz=timezone.utc))
            # Try ISO format string
            elif isinstance(ts, str):
                return date2num(datetime.fromisoformat(ts.replace('Z', '+00:00')))
        except:
            return None
    
    my_timestamps_plot = [convert_to_datetime(ts) for ts in my_timestamps]
    majd_timestamps_plot = [convert_to_datetime(ts) for ts in majd_timestamps]
    
    my_Omega_ls = my_calib['Omega_ls']
    my_Omega_unc_ls = my_calib['Omega_unc_ls']
    majd_Omega_ls = majd_calib['Omega_ls']
    majd_Omega_unc_ls = majd_calib['Omega_unc_ls']
    my_epsilon_r_ls = my_calib['epsilon_r_ls']
    my_epsilon_r_ls_unc = my_calib['epsilon_r_unc_ls']
    majd_epsilon_r_ls = majd_calib['epsilon_r_ls']
    majd_epsilon_r_ls_unc = majd_calib['epsilon_r_unc_ls']
    my_epsilon_g_ls = my_calib['epsilon_g_ls']
    my_epsilon_g_ls_unc = my_calib['epsilon_g_unc_ls']
    majd_epsilon_g_ls = majd_calib['epsilon_g_ls']
    majd_epsilon_g_ls_unc = majd_calib['epsilon_g_unc_ls']
    
    print(f"[compare_calibs] My data: {len(my_timestamps)} timestamps, Majd data: {len(majd_timestamps)} timestamps")

    print(f"[compare_calibs] Setting up matplotlib...")
    setup_start = time.time()
    mpl.rcParams.update({'font.size': fontsize})
    plt.rc('text', usetex=True)
    mpl.rc('text.latex', preamble=r"""
    \usepackage{amsmath}
    \usepackage{newtxtext,newtxmath}
    """)
    print(f"[compare_calibs] Matplotlib setup completed in {time.time() - setup_start:.2f}s")

    # 3 rows: Omega, epsilon_r, epsilon_g
    print(f"[compare_calibs] Creating plots...")
    plot_start = time.time()
    fig, axs = plt.subplots(3, 1, figsize=(30, 10), sharex=True)
    colors_compare = ['blue', 'orange', 'green']
    # my values are squares, majd are circles
    
    # Filter my data to only include non-zero timestamps
    my_valid_indices = [i for i, ts in enumerate(my_timestamps_plot) if ts is not None]
    
    # iterate over qubits, collect values across ensembles
    print(f"[compare_calibs] Plotting my data...")
    for q_idx, color in tqdm(enumerate(colors_compare), total=len(colors_compare), desc="My data"):
        # Collect values for this qubit across all ensembles with non-zero timestamps
        Omega_vals = [my_Omega_ls[i][q_idx] for i in my_valid_indices]
        Omega_unc_vals = [my_Omega_unc_ls[i][q_idx] for i in my_valid_indices]
        epsilon_r_vals = [my_epsilon_r_ls[i][q_idx] for i in my_valid_indices]
        epsilon_r_unc_vals = [my_epsilon_r_ls_unc[i][q_idx] for i in my_valid_indices]
        epsilon_g_vals = [my_epsilon_g_ls[i][q_idx] for i in my_valid_indices]
        epsilon_g_unc_vals = [my_epsilon_g_ls_unc[i][q_idx] for i in my_valid_indices]
        my_timestamps_filtered = [my_timestamps_plot[i] for i in my_valid_indices]
        
        label_omega = rf'$\mathrm{{OxCam}}, \, q_{{\mathrm{{idx}}}} = {q_idx}$' 
        
        axs[0].errorbar(my_timestamps_filtered, Omega_vals, yerr=Omega_unc_vals, fmt='s', color=color, label=label_omega, alpha=0.7)
        axs[1].errorbar(my_timestamps_filtered, epsilon_r_vals, yerr=epsilon_r_unc_vals, fmt='s', color=color, label=None, alpha=0.7)
        axs[2].errorbar(my_timestamps_filtered, epsilon_g_vals, yerr=epsilon_g_unc_vals, fmt='s', color=color, label=None, alpha=0.7)
    
    # majd data: filter ensembles by non-zero timestamps
    majd_valid_indices = [i for i, ts in enumerate(majd_timestamps_plot) if ts is not None]
    
    print(f"[compare_calibs] Plotting QuEra data...")
    for color_idx, s_idx in tqdm(enumerate(majd_indices_compare), total=len(majd_indices_compare), desc="QuEra data"):
        color = colors_compare[color_idx]
        # Collect values for this qubit across valid ensembles
        Omega_vals = [majd_Omega_ls[i][s_idx] for i in majd_valid_indices if i < len(majd_Omega_ls)]
        Omega_unc_vals = [majd_Omega_unc_ls[i][s_idx] for i in majd_valid_indices if i < len(majd_Omega_unc_ls)]
        epsilon_r_vals = [majd_epsilon_r_ls[i][s_idx] for i in majd_valid_indices if i < len(majd_epsilon_r_ls)]
        epsilon_r_unc_vals = [majd_epsilon_r_ls_unc[i][s_idx] for i in majd_valid_indices if i < len(majd_epsilon_r_ls_unc)]
        epsilon_g_vals = [majd_epsilon_g_ls[i][s_idx] for i in majd_valid_indices if i < len(majd_epsilon_g_ls)]
        epsilon_g_unc_vals = [majd_epsilon_g_ls_unc[i][s_idx] for i in majd_valid_indices if i < len(majd_epsilon_g_ls_unc)]
        
        # Filter timestamps to match
        majd_timestamps_filtered = [majd_timestamps_plot[i] for i in majd_valid_indices]
        
        # Convert None to NaN in both values and uncertainties
        Omega_vals_clean = [v if v is not None else np.nan for v in Omega_vals]
        Omega_unc_vals_clean = [unc if unc is not None else np.nan for unc in Omega_unc_vals]
        epsilon_r_vals_clean = [v if v is not None else np.nan for v in epsilon_r_vals]
        epsilon_r_unc_vals_clean = [unc if unc is not None else np.nan for unc in epsilon_r_unc_vals]
        epsilon_g_vals_clean = [v if v is not None else np.nan for v in epsilon_g_vals]
        epsilon_g_unc_vals_clean = [unc if unc is not None else np.nan for unc in epsilon_g_unc_vals]
        
        # Sort all data by timestamp to ensure fill_between works correctly
        sorted_data = sorted(zip(majd_timestamps_filtered, Omega_vals_clean, Omega_unc_vals_clean, 
                                epsilon_r_vals_clean, epsilon_r_unc_vals_clean,
                                epsilon_g_vals_clean, epsilon_g_unc_vals_clean))
        
        if sorted_data:
            majd_timestamps_sorted, Omega_vals_sorted, Omega_unc_vals_sorted, \
            epsilon_r_vals_sorted, epsilon_r_unc_vals_sorted, \
            epsilon_g_vals_sorted, epsilon_g_unc_vals_sorted = zip(*sorted_data)
        else:
            majd_timestamps_sorted = []
            Omega_vals_sorted = []
            Omega_unc_vals_sorted = []
            epsilon_r_vals_sorted = []
            epsilon_r_unc_vals_sorted = []
            epsilon_g_vals_sorted = []
            epsilon_g_unc_vals_sorted = []
        
        label_omega = rf'$\mathrm{{QuEra}}, \, \mathrm{{site}} = {s_idx}$' 
        
        # Add shaded regions for uncertainty
        Omega_upper = [v + u for v, u in zip(Omega_vals_sorted, Omega_unc_vals_sorted)]
        Omega_lower = [v - u for v, u in zip(Omega_vals_sorted, Omega_unc_vals_sorted)]
        axs[0].fill_between(majd_timestamps_sorted, Omega_lower, Omega_upper, color=color, alpha=0.2, label=label_omega)
        
        epsilon_r_upper = [v + u for v, u in zip(epsilon_r_vals_sorted, epsilon_r_unc_vals_sorted)]
        epsilon_r_lower = [v - u for v, u in zip(epsilon_r_vals_sorted, epsilon_r_unc_vals_sorted)]
        axs[1].fill_between(majd_timestamps_sorted, epsilon_r_lower, epsilon_r_upper, color=color, alpha=0.2)
        
        epsilon_g_upper = [v + u for v, u in zip(epsilon_g_vals_sorted, epsilon_g_unc_vals_sorted)]
        epsilon_g_lower = [v - u for v, u in zip(epsilon_g_vals_sorted, epsilon_g_unc_vals_sorted)]
        axs[2].fill_between(majd_timestamps_sorted, epsilon_g_lower, epsilon_g_upper, color=color, alpha=0.2)

    ## Save the average in a CSV comparing me and majd, Omega, Omega_unc, epsilon_r, epsilon_r_unc, epsilon_g, epsilon_g_unc for data since Dec 28
    cutoff_date = datetime(year_comp, month_comp, day_comp, tzinfo=timezone.utc)
    cutoff_numdate = date2num(cutoff_date)
    allowed_my_indices = [i for i, ts in enumerate(my_timestamps_plot) if ts is not None and ts >= cutoff_numdate]
    allowed_majd_indices = [i for i, ts in enumerate(majd_timestamps_plot) if ts is not None and ts >= cutoff_numdate]
    print("fraction of my data allowed:", len(allowed_my_indices)/len(my_timestamps_plot))
    print("fraction of majd data allowed:", len(allowed_majd_indices)/len(majd_timestamps_plot))
    
    # Helper function to format mean ± sem with proper significant figures
    def format_mean_sem(mean, sem):
        if sem == 0.0 or mean == 0.0:
            return "0.0 ± 0.0"
        # Find the order of magnitude of the first significant digit in sem
        sem_order = int(np.floor(np.log10(abs(sem))))
        # Determine number of decimal places (negative of order gives decimal places)
        decimal_places = max(0, -sem_order)
        # Round both values
        mean_rounded = round(mean, decimal_places)
        sem_rounded = round(sem, decimal_places)
        # Format string
        return f"{mean_rounded:.{decimal_places}f} ± {sem_rounded:.{decimal_places}f}"
    
    # Prepare data for CSV
    csv_data = []
    
    # Calculate and collect averages for each qubit
    for q_idx, majd_site_idx in enumerate(majd_indices_compare):
        # OxCam data
        my_Omega_vals = [my_Omega_ls[i][q_idx] for i in allowed_my_indices if my_Omega_ls[i][q_idx] != 0.0]
        my_Omega_uncs = [my_Omega_unc_ls[i][q_idx] for i in allowed_my_indices if my_Omega_ls[i][q_idx] != 0.0]
        my_epsilon_r_vals = [my_epsilon_r_ls[i][q_idx] for i in allowed_my_indices if my_epsilon_r_ls[i][q_idx] != 0.0]
        my_epsilon_r_uncs = [my_epsilon_r_ls_unc[i][q_idx] for i in allowed_my_indices if my_epsilon_r_ls[i][q_idx] != 0.0]
        my_epsilon_g_vals = [my_epsilon_g_ls[i][q_idx] for i in allowed_my_indices if my_epsilon_g_ls[i][q_idx] != 0.0]
        my_epsilon_g_uncs = [my_epsilon_g_ls_unc[i][q_idx] for i in allowed_my_indices if my_epsilon_g_ls[i][q_idx] != 0.0]
        
        # QuEra data
        majd_Omega_vals = [majd_Omega_ls[i][majd_site_idx] for i in allowed_majd_indices if i < len(majd_Omega_ls) and majd_Omega_ls[i][majd_site_idx] is not None]
        majd_Omega_uncs = [majd_Omega_unc_ls[i][majd_site_idx] for i in allowed_majd_indices if i < len(majd_Omega_unc_ls) and majd_Omega_ls[i][majd_site_idx] is not None]
        majd_epsilon_r_vals = [majd_epsilon_r_ls[i][majd_site_idx] for i in allowed_majd_indices if i < len(majd_epsilon_r_ls) and majd_epsilon_r_ls[i][majd_site_idx] is not None]
        majd_epsilon_r_uncs = [majd_epsilon_r_ls_unc[i][majd_site_idx] for i in allowed_majd_indices if i < len(majd_epsilon_r_ls_unc) and majd_epsilon_r_ls[i][majd_site_idx] is not None]
        majd_epsilon_g_vals = [majd_epsilon_g_ls[i][majd_site_idx] for i in allowed_majd_indices if i < len(majd_epsilon_g_ls) and majd_epsilon_g_ls[i][majd_site_idx] is not None]
        majd_epsilon_g_uncs = [majd_epsilon_g_ls_unc[i][majd_site_idx] for i in allowed_majd_indices if i < len(majd_epsilon_g_ls_unc) and majd_epsilon_g_ls[i][majd_site_idx] is not None]
        
        # Calculate means and std errors (combining statistical SEM and individual uncertainties)
        my_Omega_mean = np.mean(my_Omega_vals) if my_Omega_vals else 0.0
        my_Omega_stat_sem = np.std(my_Omega_vals) / np.sqrt(len(my_Omega_vals)) if len(my_Omega_vals) > 0 else 0.0
        my_Omega_prop_unc = np.sqrt(np.sum([unc**2 for unc in my_Omega_uncs])) / len(my_Omega_uncs) if len(my_Omega_uncs) > 0 else 0.0
        my_Omega_std = np.sqrt(my_Omega_stat_sem**2 + my_Omega_prop_unc**2)
        
        my_epsilon_r_mean = np.mean(my_epsilon_r_vals) if my_epsilon_r_vals else 0.0
        my_epsilon_r_stat_sem = np.std(my_epsilon_r_vals) / np.sqrt(len(my_epsilon_r_vals)) if len(my_epsilon_r_vals) > 0 else 0.0
        my_epsilon_r_prop_unc = np.sqrt(np.sum([unc**2 for unc in my_epsilon_r_uncs])) / len(my_epsilon_r_uncs) if len(my_epsilon_r_uncs) > 0 else 0.0
        my_epsilon_r_std = np.sqrt(my_epsilon_r_stat_sem**2 + my_epsilon_r_prop_unc**2)
        
        my_epsilon_g_mean = np.mean(my_epsilon_g_vals) if my_epsilon_g_vals else 0.0
        my_epsilon_g_stat_sem = np.std(my_epsilon_g_vals) / np.sqrt(len(my_epsilon_g_vals)) if len(my_epsilon_g_vals) > 0 else 0.0
        my_epsilon_g_prop_unc = np.sqrt(np.sum([unc**2 for unc in my_epsilon_g_uncs])) / len(my_epsilon_g_uncs) if len(my_epsilon_g_uncs) > 0 else 0.0
        my_epsilon_g_std = np.sqrt(my_epsilon_g_stat_sem**2 + my_epsilon_g_prop_unc**2)
        
        majd_Omega_mean = np.mean(majd_Omega_vals) if majd_Omega_vals else 0.0
        majd_Omega_stat_sem = np.std(majd_Omega_vals) / np.sqrt(len(majd_Omega_vals)) if len(majd_Omega_vals) > 0 else 0.0
        majd_Omega_prop_unc = np.sqrt(np.sum([unc**2 for unc in majd_Omega_uncs])) / len(majd_Omega_uncs) if len(majd_Omega_uncs) > 0 else 0.0
        majd_Omega_std = np.sqrt(majd_Omega_stat_sem**2 + majd_Omega_prop_unc**2)
        
        majd_epsilon_r_mean = np.mean(majd_epsilon_r_vals) if majd_epsilon_r_vals else 0.0
        majd_epsilon_r_stat_sem = np.std(majd_epsilon_r_vals) / np.sqrt(len(majd_epsilon_r_vals)) if len(majd_epsilon_r_vals) > 0 else 0.0
        majd_epsilon_r_prop_unc = np.sqrt(np.sum([unc**2 for unc in majd_epsilon_r_uncs])) / len(majd_epsilon_r_uncs) if len(majd_epsilon_r_uncs) > 0 else 0.0
        majd_epsilon_r_std = np.sqrt(majd_epsilon_r_stat_sem**2 + majd_epsilon_r_prop_unc**2)
        
        majd_epsilon_g_mean = np.mean(majd_epsilon_g_vals) if majd_epsilon_g_vals else 0.0
        majd_epsilon_g_stat_sem = np.std(majd_epsilon_g_vals) / np.sqrt(len(majd_epsilon_g_vals)) if len(majd_epsilon_g_vals) > 0 else 0.0
        majd_epsilon_g_prop_unc = np.sqrt(np.sum([unc**2 for unc in majd_epsilon_g_uncs])) / len(majd_epsilon_g_uncs) if len(majd_epsilon_g_uncs) > 0 else 0.0
        majd_epsilon_g_std = np.sqrt(majd_epsilon_g_stat_sem**2 + majd_epsilon_g_prop_unc**2)
        
        # Add row to CSV data with formatted strings
        csv_data.append({
            'qubit_idx': q_idx,
            'majd_site_idx': majd_site_idx,
            'OxCam_Omega': format_mean_sem(my_Omega_mean, my_Omega_std),
            'QuEra_Omega': format_mean_sem(majd_Omega_mean, majd_Omega_std),
            'OxCam_epsilon_r': format_mean_sem(my_epsilon_r_mean, my_epsilon_r_std),
            'QuEra_epsilon_r': format_mean_sem(majd_epsilon_r_mean, majd_epsilon_r_std),
            'OxCam_epsilon_g': format_mean_sem(my_epsilon_g_mean, my_epsilon_g_std),
            'QuEra_epsilon_g': format_mean_sem(majd_epsilon_g_mean, majd_epsilon_g_std),
        })
    
    # Save formatted CSV
    csv_path = os.path.join(dir_root, f"calib_comparison_since_{date_str}.csv")
    df_comparison = pd.DataFrame(csv_data)
    df_comparison.to_csv(csv_path, index=False)
    print(f"[compare_calibs] Saved calibration comparison to {csv_path}")
    
    # Save raw (unrounded) CSV with separate mean and SEM columns
    csv_data_raw = []
    for q_idx, majd_site_idx in enumerate(majd_indices_compare):
        # OxCam data
        my_Omega_vals = [my_Omega_ls[i][q_idx] for i in allowed_my_indices if my_Omega_ls[i][q_idx] != 0.0]
        my_Omega_uncs = [my_Omega_unc_ls[i][q_idx] for i in allowed_my_indices if my_Omega_ls[i][q_idx] != 0.0]
        my_epsilon_r_vals = [my_epsilon_r_ls[i][q_idx] for i in allowed_my_indices if my_epsilon_r_ls[i][q_idx] != 0.0]
        my_epsilon_r_uncs = [my_epsilon_r_ls_unc[i][q_idx] for i in allowed_my_indices if my_epsilon_r_ls[i][q_idx] != 0.0]
        my_epsilon_g_vals = [my_epsilon_g_ls[i][q_idx] for i in allowed_my_indices if my_epsilon_g_ls[i][q_idx] != 0.0]
        my_epsilon_g_uncs = [my_epsilon_g_ls_unc[i][q_idx] for i in allowed_my_indices if my_epsilon_g_ls[i][q_idx] != 0.0]
        
        # QuEra data
        majd_Omega_vals = [majd_Omega_ls[i][majd_site_idx] for i in allowed_majd_indices if i < len(majd_Omega_ls) and majd_Omega_ls[i][majd_site_idx] is not None]
        majd_Omega_uncs = [majd_Omega_unc_ls[i][majd_site_idx] for i in allowed_majd_indices if i < len(majd_Omega_unc_ls) and majd_Omega_ls[i][majd_site_idx] is not None]
        majd_epsilon_r_vals = [majd_epsilon_r_ls[i][majd_site_idx] for i in allowed_majd_indices if i < len(majd_epsilon_r_ls) and majd_epsilon_r_ls[i][majd_site_idx] is not None]
        majd_epsilon_r_uncs = [majd_epsilon_r_ls_unc[i][majd_site_idx] for i in allowed_majd_indices if i < len(majd_epsilon_r_ls_unc) and majd_epsilon_r_ls[i][majd_site_idx] is not None]
        majd_epsilon_g_vals = [majd_epsilon_g_ls[i][majd_site_idx] for i in allowed_majd_indices if i < len(majd_epsilon_g_ls) and majd_epsilon_g_ls[i][majd_site_idx] is not None]
        majd_epsilon_g_uncs = [majd_epsilon_g_ls_unc[i][majd_site_idx] for i in allowed_majd_indices if i < len(majd_epsilon_g_ls_unc) and majd_epsilon_g_ls[i][majd_site_idx] is not None]
        
        # Calculate means and std errors (combining statistical SEM and individual uncertainties)
        my_Omega_mean = np.mean(my_Omega_vals) if my_Omega_vals else 0.0
        my_Omega_stat_sem = np.std(my_Omega_vals) / np.sqrt(len(my_Omega_vals)) if len(my_Omega_vals) > 0 else 0.0
        my_Omega_prop_unc = np.sqrt(np.sum([unc**2 for unc in my_Omega_uncs])) / len(my_Omega_uncs) if len(my_Omega_uncs) > 0 else 0.0
        my_Omega_std = np.sqrt(my_Omega_stat_sem**2 + my_Omega_prop_unc**2)
        
        my_epsilon_r_mean = np.mean(my_epsilon_r_vals) if my_epsilon_r_vals else 0.0
        my_epsilon_r_stat_sem = np.std(my_epsilon_r_vals) / np.sqrt(len(my_epsilon_r_vals)) if len(my_epsilon_r_vals) > 0 else 0.0
        my_epsilon_r_prop_unc = np.sqrt(np.sum([unc**2 for unc in my_epsilon_r_uncs])) / len(my_epsilon_r_uncs) if len(my_epsilon_r_uncs) > 0 else 0.0
        my_epsilon_r_std = np.sqrt(my_epsilon_r_stat_sem**2 + my_epsilon_r_prop_unc**2)
        
        my_epsilon_g_mean = np.mean(my_epsilon_g_vals) if my_epsilon_g_vals else 0.0
        my_epsilon_g_stat_sem = np.std(my_epsilon_g_vals) / np.sqrt(len(my_epsilon_g_vals)) if len(my_epsilon_g_vals) > 0 else 0.0
        my_epsilon_g_prop_unc = np.sqrt(np.sum([unc**2 for unc in my_epsilon_g_uncs])) / len(my_epsilon_g_uncs) if len(my_epsilon_g_uncs) > 0 else 0.0
        my_epsilon_g_std = np.sqrt(my_epsilon_g_stat_sem**2 + my_epsilon_g_prop_unc**2)
        
        majd_Omega_mean = np.mean(majd_Omega_vals) if majd_Omega_vals else 0.0
        majd_Omega_stat_sem = np.std(majd_Omega_vals) / np.sqrt(len(majd_Omega_vals)) if len(majd_Omega_vals) > 0 else 0.0
        majd_Omega_prop_unc = np.sqrt(np.sum([unc**2 for unc in majd_Omega_uncs])) / len(majd_Omega_uncs) if len(majd_Omega_uncs) > 0 else 0.0
        majd_Omega_std = np.sqrt(majd_Omega_stat_sem**2 + majd_Omega_prop_unc**2)
        
        majd_epsilon_r_mean = np.mean(majd_epsilon_r_vals) if majd_epsilon_r_vals else 0.0
        majd_epsilon_r_stat_sem = np.std(majd_epsilon_r_vals) / np.sqrt(len(majd_epsilon_r_vals)) if len(majd_epsilon_r_vals) > 0 else 0.0
        majd_epsilon_r_prop_unc = np.sqrt(np.sum([unc**2 for unc in majd_epsilon_r_uncs])) / len(majd_epsilon_r_uncs) if len(majd_epsilon_r_uncs) > 0 else 0.0
        majd_epsilon_r_std = np.sqrt(majd_epsilon_r_stat_sem**2 + majd_epsilon_r_prop_unc**2)
        
        majd_epsilon_g_mean = np.mean(majd_epsilon_g_vals) if majd_epsilon_g_vals else 0.0
        majd_epsilon_g_stat_sem = np.std(majd_epsilon_g_vals) / np.sqrt(len(majd_epsilon_g_vals)) if len(majd_epsilon_g_vals) > 0 else 0.0
        majd_epsilon_g_prop_unc = np.sqrt(np.sum([unc**2 for unc in majd_epsilon_g_uncs])) / len(majd_epsilon_g_uncs) if len(majd_epsilon_g_uncs) > 0 else 0.0
        majd_epsilon_g_std = np.sqrt(majd_epsilon_g_stat_sem**2 + majd_epsilon_g_prop_unc**2)
        
        csv_data_raw.append({
            'qubit_idx': q_idx,
            'majd_site_idx': majd_site_idx,
            'OxCam_Omega_mean': my_Omega_mean,
            'OxCam_Omega_sem': my_Omega_std,
            'QuEra_Omega_mean': majd_Omega_mean,
            'QuEra_Omega_sem': majd_Omega_std,
            'OxCam_epsilon_r_mean': my_epsilon_r_mean,
            'OxCam_epsilon_r_sem': my_epsilon_r_std,
            'QuEra_epsilon_r_mean': majd_epsilon_r_mean,
            'QuEra_epsilon_r_sem': majd_epsilon_r_std,
            'OxCam_epsilon_g_mean': my_epsilon_g_mean,
            'OxCam_epsilon_g_sem': my_epsilon_g_std,
            'QuEra_epsilon_g_mean': majd_epsilon_g_mean,
            'QuEra_epsilon_g_sem': majd_epsilon_g_std,
        })
    
    csv_path_raw = os.path.join(dir_root, f"calib_comparison_since_{date_str}_raw.csv")
    df_comparison_raw = pd.DataFrame(csv_data_raw)
    df_comparison_raw.to_csv(csv_path_raw, index=False)
    print(f"[compare_calibs] Saved raw calibration comparison to {csv_path_raw}")

    # Format x-axis to show actual timestamps
    from matplotlib.dates import num2date
    from matplotlib.ticker import FuncFormatter
    
    def date_formatter(x, pos):
        """Convert matplotlib date number to formatted date string (non-LaTeX)"""
        dt = num2date(x)
        return dt.strftime('%Y-%m-%d\n%H:%M UTC')
    
    for ax in axs:
        ax.xaxis.set_major_formatter(FuncFormatter(date_formatter))
        ax.tick_params(axis='x', rotation=45)
    
    axs[2].set_xlabel(r"$\mathrm{Timestamp}$")
    axs[0].set_ylabel(r"$\Omega \, (\mu \mathrm{s}^{-1})$ ")
    axs[1].set_ylabel(r"$\epsilon_r$ ")
    axs[2].set_ylabel(r"$\epsilon_g$ ")
    axs[0].legend(fontsize=fontsize*0.8, ncol=2, bbox_to_anchor=(0.5, 1.15), loc='lower center')
    for ax in axs:
        style_axis(ax, fontsize=fontsize)

    print(f"[compare_calibs] Plotting completed in {time.time() - plot_start:.2f}s")
    print(f"[compare_calibs] Saving figure...")
    save_start = time.time()
    
    # Temporarily disable usetex for tight_layout to avoid LaTeX processing date strings
    original_usetex = plt.rcParams['text.usetex']
    plt.rcParams['text.usetex'] = False
    fig.canvas.draw()  # Force rendering with usetex=False for tick labels
    plt.rcParams['text.usetex'] = original_usetex
    
    plt.tight_layout()
    plt.savefig(f"{dir_root}/compare_calibs.pdf")
    print(f"[compare_calibs] Figure saved in {time.time() - save_start:.2f}s")
    print(f"[compare_calibs] Total time: {time.time() - start_time:.2f}s")

def extract_readout_error_majd(my_calib_path, majd_output_path, total_ens, N, majd_indices_compare=[2, 6, 10]):
    """
    Extract readout error data from majd calibration, matching timestamps from my_calib.
    For each timestamp in my_calib, find the closest timestamp in majd data and extract
    the values at majd_indices_compare positions.
    
    Returns the same format as extract_readouterror_rabi:
    epsilon_r_ls, epsilon_r_unc_ls, epsilon_g_ls, epsilon_g_unc_ls, Omega_ls, Omega_unc_ls
    """
    # Read both JSON files
    with open(my_calib_path, 'r') as f:
        my_calib = json.load(f)
    with open(majd_output_path, 'r') as f:
        majd_calib = json.load(f)
    
    my_timestamps = my_calib['timestamp']
    majd_timestamps = majd_calib['timestamp']
    
    
    # Initialize output structure
    result_dict = {
        "epsilon_r_ls": [[0.0 for _ in range(N)] for _ in range(total_ens)],
        "epsilon_r_unc_ls": [[0.0 for _ in range(N)] for _ in range(total_ens)],
        "epsilon_g_ls": [[0.0 for _ in range(N)] for _ in range(total_ens)],
        "epsilon_g_unc_ls": [[0.0 for _ in range(N)] for _ in range(total_ens)],
        "Omega_ls": [[0.0 for _ in range(N)] for _ in range(total_ens)],
        "Omega_unc_ls": [[0.0 for _ in range(N)] for _ in range(total_ens)],
    }
    
    # Convert timestamp to numeric for comparison
    def to_numeric_timestamp(ts):
        if isinstance(ts, (int, float)):
            return float(ts)
        elif isinstance(ts, str):
            try:
                return datetime.fromisoformat(ts.replace('Z', '+00:00')).timestamp()
            except:
                return 0.0
        return 0.0
    
    my_timestamps_numeric = [to_numeric_timestamp(ts) for ts in my_timestamps]
    majd_timestamps_numeric = [to_numeric_timestamp(ts) for ts in majd_timestamps]
    
    # For each my_calib timestamp, find closest majd timestamp
    for ens_idx, my_ts in enumerate(my_timestamps_numeric):
        if my_ts == 0.0:
            continue
        
        # Find closest majd timestamp
        min_diff = float('inf')
        closest_majd_idx = 0
        for majd_idx, majd_ts in enumerate(majd_timestamps_numeric):
            if majd_ts == 0.0:
                continue
            diff = abs(my_ts - majd_ts)
            if diff < min_diff:
                min_diff = diff
                closest_majd_idx = majd_idx
        
        # Extract values at majd_indices_compare for this closest timestamp
        for result_idx, majd_site_idx in enumerate(majd_indices_compare):
            if closest_majd_idx < len(majd_calib['epsilon_r_ls']) and majd_site_idx < len(majd_calib['epsilon_r_ls'][closest_majd_idx]):
                result_dict["epsilon_r_ls"][ens_idx][result_idx] = majd_calib['epsilon_r_ls'][closest_majd_idx][majd_site_idx]
                result_dict["epsilon_r_unc_ls"][ens_idx][result_idx] = majd_calib['epsilon_r_unc_ls'][closest_majd_idx][majd_site_idx]
                result_dict["epsilon_g_ls"][ens_idx][result_idx] = majd_calib['epsilon_g_ls'][closest_majd_idx][majd_site_idx]
                result_dict["epsilon_g_unc_ls"][ens_idx][result_idx] = majd_calib['epsilon_g_unc_ls'][closest_majd_idx][majd_site_idx]
                result_dict["Omega_ls"][ens_idx][result_idx] = majd_calib['Omega_ls'][closest_majd_idx][majd_site_idx]
                result_dict["Omega_unc_ls"][ens_idx][result_idx] = majd_calib['Omega_unc_ls'][closest_majd_idx][majd_site_idx]
    
    print(f"Extracted readout error data from majd for {total_ens} ensembles, {N} sites each")
    return result_dict['epsilon_r_ls'], result_dict['epsilon_r_unc_ls'], result_dict['epsilon_g_ls'], result_dict['epsilon_g_unc_ls'], result_dict['Omega_ls'], result_dict['Omega_unc_ls']
    
    


if __name__ == "__main__":
    #### CHECK THE OPTION _FORCE_TRY_HYBRID AT TOP OF FILE BEFORE RUNNING!!!  ####
    mode = "RUN" ## OPTIONS: "RUN" to run the full pipeline --  ENFORCE _FORCE_TRY_HYBRID==TRUE, "SAVE" to only save SUIDs and chunks  ENFORCE _FORCE_TRY_HYBRID==FALSE !!!, "GET-CALIB" to only extract and save calibration data
    dir_root = "gaugamela_ramp"

    # 0.05 us starting time
    expt_task_name_005="task_ddebfeaea3ca63079a4c10c4" # chaotic, 6 qubits
    expt_task_name_10="task_15eaace71ebd595814b7bd17" # localized, 6 qubits

    timestamp_005 = 1766200542 ## CHANGE FOR EXPT
    timestamp_10 = 1768489892

    expt_task_name_125 = "task_1f671c4d079b68342b55451a"
    timestamp_125 = 1772408252

    is_expt_data = False ## CHANGE FOR EXPT

    name = "full_expt"
    cancel_Rabi = False  ## CHANGE FOR EXPT: whether to do few-shot Rabi calibration before each chunk
 

    chosen_task_ls = ["task_40b2edc8e4622b0d21eaf6b9"]
    chosen_timestamp = 1774626495
    
    
    specific_chunk_idx = [0] ## CHANGE FOR EXPT: set to None to run all chunks, or specify chunk index to run only that chunk

    # avg_comp_time=(2026, 1, 15) # for comparing OxCam and QuEra calibs since this date
    avg_comp_time=(2025, 12, 28) # for comparing OxCam and QuEra calibs since this date
    

    if mode =="RUN":
        for chosen_task in chosen_task_ls:
            schedule_run_all_chunks(chosen_task, name, is_expt_data, chosen_timestamp, dir_root, specific_chunk_idx=specific_chunk_idx, override_num_ham_per_chunk=1, cancel_Rabi=cancel_Rabi)
    elif mode == "SAVE":
        save_all_suids(dir_root)
        for chosen_task in chosen_task_ls:
            save_chunks(chosen_task, name, is_expt_data, chosen_timestamp, dir_root)
    elif mode == "GET-CALIB":
        # main_dir,chosen_task, total_ens
        for chosen_task in chosen_task_ls: 
            h_ls_pre, x_pre, t_plateau_ls, seq_ls_pre_all, base_params, Delta_mean_ls, Delta_local_ls, gate_params_all, cluster_spacing, manual_parallelization, override_local, x0_y0_offset, t_delay, start_Delta_from0, uniform_Omega_Delta_ramp ,phi_opt, Delta_local_ramp_time, Omega_delay_time = read_expt_task(chosen_task, dir_root) 

            extract_readouterror_rabi(dir_root, chosen_task, total_ens=len(h_ls_pre))
            
   