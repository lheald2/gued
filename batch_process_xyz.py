import numpy as np
import gued_theory as gt
import concurrent.futures
import glob
import time
import pandas as pd

def calculate_I_for_structure(structure, atom_sum):
    """Helper function to calculate I_tot for a single structure."""
    return gt.get_I_from_xyz(structure, atom_sum)

def process_all_structures_for_key(key, xyz_dict, I_totals, I_stds):
    """Process all structures for a given key."""
    print(f"Calculating I total for all {key} structures")
    
    # Prepare data for concurrent processing
    structures = xyz_dict[key]["coordinates"]
    atom_sum = xyz_dict[key]["atom_sums"]
    
    all_I = []
    
    # Use ProcessPoolExecutor to parallelize the inner loop
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(calculate_I_for_structure, structure, atom_sum) 
            for structure in structures
        ]
        
        # Process results as they are completed
        count = 0
        for future in concurrent.futures.as_completed(futures):
            I_tot, _, _, _ = future.result()
            all_I.append(np.array(I_tot))
            count += 1
            if count % 1000 == 0:
                print(f"Processed {count}")
    
    # Calculate the final totals and standard deviations
    return np.mean(all_I, axis=0), np.std(all_I, axis=0)

def process_trajectory(folder, mol_name, file_type):
    print(f"getting trajectory for {folder[-6:-2]}")
    xyz, at_sum, counts = gt.load_time_evolving_xyz(folder, mol_name, file_type)
    #print(at_sum)
    I0, _, _ , s_vals = gt.get_I_from_xyz(xyz[0], at_sum)
    return I0


if __name__=='__main__':
    file_path = "C:\\Users\\laure\\OneDrive - University of Nebraska-Lincoln\\Documents\\Centurion Lab\\QC data and code\\hot_structures\\"
    file_names = ["ethanol", "CP", "QC", "NBD"]
    file_type = ".xyz"

    structure_dict = {key: {"coordinates": [], "atom_sums": [],} for key in file_names}

    start = time.perf_counter()
        
    for file_name in file_names:
        coors, atom_sum = gt.load_hot_xyz(file_path, file_name, file_type)
        structure_dict[file_name]["coordinates"] = coors
        structure_dict[file_name]["atom_sums"] = atom_sum
    
    stop = time.perf_counter()

    print(f"Finished getting hot structures in {(stop-start)/60:.2f} minutes")
    
    I_totals = {key: [] for key in file_names}
    I_stds = {key: [] for key in file_names}

    start1 = time.perf_counter()

    for key in file_names:
        I_mean, I_std = process_all_structures_for_key(key, structure_dict, I_totals, I_stds)
        I_totals[key] = I_mean
        I_stds[key] = I_std
        #print(I_totals[key])

    stop1 = time.perf_counter()
    print(f"Finished getting I for hot structures in {(stop1-start1)/60:.2f} minutes")
    
    start2 = time.perf_counter()
    I0_all = []

    path_traj = f"C:/Users/laure/OneDrive - University of Nebraska-Lincoln/Documents/Centurion Lab/QC data and code/Theory Structures/*/*/*/"

    traj_folder = glob.glob(path_traj)
    # sort the folder names in order of trajectory number

    print(f"Processing {len(traj_folder)} trajectory files.")
    mol_name = "output"
    file_type = ".xyz"

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(process_trajectory, file, mol_name, file_type) for file in traj_folder]
        for future in concurrent.futures.as_completed(futures):
            I0_all.append(np.array(future.result()))

    I0_all = np.array(I0_all)
    #print(I0_all.shape)
    I0_mean = np.mean(np.array(I0_all), axis=0)
    I0_all = np.array(I0_all)
    #print(I0_all.shape)
    I0_mean = np.mean(np.array(I0_all), axis=0)

    stop2 = time.perf_counter()
    print(f"Finished getting I for trajectory structures in {(stop2-start2)/60:.2f} minutes")

    key_words = ["NBD", "QC", "frags"]
    dI_I_hot = {key: [] for key in key_words}
    dI_I_hot["QC"] = (I_totals["QC"]-I0_mean)/I0_mean
    dI_I_hot["NBD"] = (I_totals["NBD"]-I0_mean)/I0_mean
    dI_I_hot["frags"] = ((I_totals["CP"]+I_totals["ethanol"])-I0_mean)/I0_mean

    s_vals = np.linspace(0, 12, 500)
    for key in key_words:
        file_name = key+".txt"
        frag_dict = {"s_values": s_vals, "dI_I": dI_I_hot[key]}
        frag_df = pd.DataFrame(frag_dict)
        frag_df.to_csv(file_name, sep='\t', index=False)
    stop3 = time.perf_counter()
    print(f"total processing time was {(stop3-start)/60:.2f} minutes")

