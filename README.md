# RydSFF
Repository to run real experiments, simulated experiments, and purely numerical calculations for QuEra's Aquila device for our paper (arXiv:https://arxiv.org/abs/2604.24854)

This code is not yet refined enough to contain a tutorial for using AWS Braket, so if you have any questions about that in particular or want to discuss other parts of the code, please contact me at oscar.scholin@physics.ox.ac.uk or orsa2020@mymail.pomona.edu.

### version requirements: 

For AWS, we enforce Python 3.10.20 (this is handled in the S3 environment creation during hybrid job; otherwise we run into errors submitting programs to Aquila). For classical backend simulation, we use Python 3.12.2.

## File descriptions:

### Experimental data plotting
- `main_expt_plot.py`
    - I use a hash code to reference the input configuration to bloqade -- this is denoted by files with the name structure `task_{uid}.json`, which are stored inside `dir_main/tasks/*` subdirectory. Tasks have corresponding stem_tasks which record the overall structure of the hamiltonian but not the specific `h_i` values or `seq_ls` that are actually executed. 
    - Task and stem ids can be generated using `_get_task`
    - `run_for_N_protocol` is the lowest level of the plotting functions -- it calls `process_rbp.py` after reading the tasks
    - `time_disorder_protocol` is the next-highest level plotting function we use that actually creates the figures 
    - the main function to call that specifies all the tasks and experimental timestamps to pass to `time_disorder_protocol` is `main_expt_figs` which takes as arguments `opt=='sim'` for simulated bloqade and `opt=='expt'` for experimental tasks.
    - to run parallelised T2star simulations use `main_T2star_parallel` and then `compile_T2star` to recombine the data into the main directory
    

### AWS:
  - `task_submission_main.py`
        - `schedule_run_all_chunks` is the main, high level function to run experiments on Aquila. automatically checks to avoid submitting during the calibration period so all data can be run together
            - First in an AWS environment, run `option=="RUN"` with `_FORCE_TRY_HYBRID == True` to initialise hybrid job and submit quantum tasks via braket. Then, run `option=="SAVE"` but  `_FORCE_TRY_HYBRID == True` while still in the AWS environment to fetch the `.json` and return results as `.npy`. The directory now may be zipped and downloaded to your local machine for further processing.
            - `requirements_revised.txt` is used to setup the 
- helpers
    - `cancel_all_tasks.py`
        - terminates all actively queued tasks in amazon braket
    - `cancel_task_arn.py`
        - terminates specific hybrid task given arn
    - `make_tasks_table.py`
        - compile results of quantum tasks submitted to braket
    - `parse_majd_calib.py`
        - compile QuEra calibrations and compare against our own
    - `manual_download.py`
        - download the S3 bucket containing the hybrid task result
    - `chain_benchmark.py`, `diagnose_driver.py` are helper functions for the Rabi oscillation calibration which is called in `run_for_N_protocol`

### Middle level functions
- `master_params_rbp.py` 
    - called by `task_submission_main.py`: contains funcs to create task uids (`read_expt_task` hashes the input parameters to the Hamiltonian and other input configs; the hashing is done by `QuEraToolbox.expt_file_manager.py`), and to execute these tasks on simulated backend or on AWS hardware (`execute_bloqade_task`)
- `process_rbp.py`
    - `process_bitstrings` is the main function of this file: tasks a list of `h_ls` and `seq_ls` extracted from the task by functions in `main_expt_plot` (`run_for_N_protocol`); first calls the respective functions to gather simulated or experimental bitstrings and then in lines 1965 - 1974 will helper functions to actually calculate the purity (`est_purity`). Includes support for T2* uncertainty (bloqade simulator and qutip) and T2 dephasing noise (qutip only).
        -  `get_all_qutip_probs` is the main function for qutip emulation tasks
        - `get_all_single_hams_rand` is the main function for bloqade (emulation and simulation) tasks; calls `get_single_ham_rand` for individual `h_i` list, which calls 
    - `numerical` computes numerically the second order renyi entropy including the time dependent ramp structure of the experiment but excluding the randomised measurements

### QuEraToolbox -- low-level processing and execution helpers
- `random_bp_prep.py`
    - lowest level file to run experiments (or emulate) on Aquila. (!!!!!)
    - `expt_run` is the main function of this file: data is either loaded if the corresponding files exist (either the `.json` from Braket or the `.npy` from our processing) or the program is compiled using `compile_program_oneU` which gets the initial ramp up, plateau, ramp down and `compile_rand_seq` which intersplices the randomised phi-flip sequence. 
    - `random_bp_qutip.py` emulates this function but also includes option for T2 modeling
- `expt_file_manager.py`
    - hashes an input dict with json-friendly keys and values through the object `ExptStore`
- `hamiltonian.py`
    - defines the Aquila hamiltonian in QuTip with helpers for ramp structures -- main function is the function factory `drive_main`
- `helper_rbp.py`
    - contains the functions to estimate purity using the randomised measurement toolbox formula (`est_purity`) and apply readout error to numerical probability vectors (`apply_readout_channel`)
    

### Illustrative figures
- `fig_supplemental_signatures.py`
    - all figures shown in the Supplemental Information, which focus on the ergodicity to localisation transition
- `fig1_single_qubit_chain.py`
    - illustrative figure to show the growth of entanglement for Fig 1 in paper
- `fig2_randmeas.py`
    - Fig 2 in the paper to illustrate the behavior of the $phi$-quench gates at single qubit level.
- `appendix_spectrum.py`
    - computes the median eigenvalue as a function of the mean local detuning


## Data: 
DOI 10.5287/ora-dow5oapxe contains all the data needed to reproduce the figures in the main text of the paper. Due to upload size constraints we have to separate the main directory `paper_main_data` into two .zip files: `data.zip` and `paper_main_data_rest.zip`.  

Inside `data/`, the following files can be found:
- `full_expt_{task_id}_{timestamp}`: experimental data from Aquila for the hash identifier `task_id` (see `tasks/task_{task_id}.json`) and the integer timestamp
- `bloqade-sim-no-rc_{task_id}_{0}`: simulated data for the task `task_id` for the noiseless emulation of experimental protocol
- `avg_{uid}.npy` and `result_{uid}.npy`: simulated data for the task `task_id` for noisy emulation of experimental protocol including readout error and $T_2^*$ effects. Note, the meaning of the uid can be determined by accessing the corresponding `{uid}.json` file in `paper_main_data_rest/combos/`

In `paper_main_data_rest` are subdirectories that specify:
    - the `tasks/*` for all tasks and stem tasks needed for the experiments, as called in `main_expt_plot.py`
    - the `combos/*` which explain the input dict to generate the hashes
    - the supporting `mdata` and `ndata` for the plots,
    - and the `results/*` themselves

`expt_tasks.zip` contains the `expt_tasks` directory needed to run `main_T2star_parallel`
