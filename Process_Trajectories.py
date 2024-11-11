"""
Python file for batch processing trajectory xyz files into dI/I and s ranges that can be used later. dI/I and s matrices are saved to an .h5 file
with the relevant information about the trajectory in the data naming scheme. This code uses the gued_theory.py package, specifically the 
gued_theory.trajectory_sim() function. In deveopment.
"""


import gued_theory as gt
import glob
from datetime import date
import time
import concurrent.futures
from functools import partial
from multiprocessing import Lock
import h5py

# # Initialize the multiprocessing lock for safe access to the HDF5 file
# lock = Lock()

# def append_to_h5_with_lock(file_name, group_name, run_id, data_dictionary):
#     """
#     Safely appends data to an HDF5 file in a thread/process-safe manner using a lock.
    
#     ARGUMENTS:
    
#     file_name (str): Name of the HDF5 file.
#     group_name (str): Group name where the dataset should be written.
#     run_id (str): Unique identifier for the run.
#     data_dictionary (dict): Dictionary containing variable names and data arrays.
#     """
#     with lock:  # Ensure this block is synchronized across processes
#         # Open the HDF5 file in append mode
#         with h5py.File(file_name, 'a') as f:
#             # Ensure the group exists
#             if group_name not in f:
#                 group = f.create_group(group_name)
#             else:
#                 group = f[group_name]
            
#             # Save each variable in the dictionary to the HDF5 group
#             for var_name, var_data in data_dictionary.items():
#                 dataset_name = f"{var_name}_run_{run_id}"
                
#                 # Delete the dataset if it already exists (overwrite behavior)
#                 if dataset_name in group:
#                     print(f"Overwriting dataset '{dataset_name}' in group '{group_name}'.")
#                     del group[dataset_name]
                
#                 # Create the dataset within the group
#                 group.create_dataset(dataset_name, data=var_data)
#                 print(f"Variable '{dataset_name}' added to group '{group_name}' successfully.")


# def batch_process_trajectories(mol_name, file_type, traj_folder, file_name, group_name):
#     """
#     Processes trajectory simulations concurrently and saves the results into an HDF5 file.
    
#     ARGUMENTS:
    
#     mol_name (str): Molecule name to pass to the simulation function.
#     file_type (str): File type to pass to the simulation function.
#     traj_folder (list): List of paths to trajectory files.
#     file_name (str): Name of the HDF5 file to save data into.
#     group_name (str): Group in the HDF5 file where data will be saved.
#     """

#     # Use ProcessPoolExecutor to run trajectory simulations concurrently
#     with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executor:
#         # Submit tasks for each path in traj_folder using partial to fix some arguments
#         futures = [executor.submit(partial(gt.trajectory_sim, mol_name, file_type), path) for path in traj_folder]
        
#         # Get results from futures
#         results = [future.result() for future in futures]

#     # Create dictionaries and save data from each result
#     for path, result in zip(traj_folder, results):
#         dI_I_raw, dI_I_conv, s, t_fs = result
        
#         # Create the data dictionary
#         data_dictionary = {
#             "dI_I_raw": dI_I_raw,
#             "dI_I_conv": dI_I_conv,
#             "s": s,
#             "time": t_fs
#         }
        
#         # Extract a unique identifier from the path (e.g., folder[-6:-2]) and save the data
#         run_id = path[-6:-2]  # Adjust this to extract the correct part of the path
        
#         # Append the data to the HDF5 file using the lock for safety
#         append_to_h5_with_lock(file_name, group_name, run_id, data_dictionary)


# Define saving details
file_label = "NB_Trajectories"
today = date.today()
print(today)

file_path = 'C:\\Users\\laure\\OneDrive - University of Nebraska-Lincoln\\Documents\\Centurion Lab\\QC data and code\\Theory Structures\\'
file_name = file_path + f"{file_label}_{today}.h5"
print(f"writing data to {file_name}")
group_name = "s4"

# Define the folder path to the trajectory files
path_traj = "C:/Users/laure/OneDrive - University of Nebraska-Lincoln/Documents/Centurion Lab/QC data and code/Theory Structures/NB/Singlet_4/*/"
traj_folder = glob.glob(path_traj)
# sort the folder names in order of trajectory number


start = time.perf_counter()
print(f"Processing {len(traj_folder)} trajectory files.")
mol_name = "output"
file_type = ".xyz"

for folder in traj_folder:
    print(f"getting trajectory for {folder[-6:-2]}")
    dI_I_raw, dI_I_conv, s, t_fs = gt.trajectory_sim(folder, mol_name, file_type, return_data=True)
    data_dictionary = {"dI_I_raw": dI_I_raw, "dI_I_conv":dI_I_conv, "s": s, "time": t_fs}
    gt.save_data(file_name, group_name, folder[-6:-2], data_dictionary)

stop = time.perf_counter()
print(f"Finished processing {len(traj_folder)} trajectories in {(stop-start)/60} minutes")


