import h5py
from chemspipy import ChemSpider
import time
import random
import concurrent.futures
import gued_theory as gt


API_KEY = '67ABV69AUpV3Q0ALBQA69zcGjqGbYCby'
FILE_NAME = 'packages\\molecular_coordinates\\training_set.h5'
cs = ChemSpider(API_KEY)
MAX_WORKERS = 6

def print_h5_structure(name, obj):
    # Print the name of the object (group or dataset)
    print(name)
    # Print attributes of the object, if any
    for key, value in obj.attrs.items():
        print(f"  Attribute: {key} = {value}")

# Function to count groups in the HDF5 file
def count_groups(file):
    group_count = 0
    
    def count_groups_recursive(name, obj):
        nonlocal group_count
        if isinstance(obj, h5py.Group):
            group_count += 1

    # Open the HDF5 file in read mode
    with h5py.File(file, 'r') as h5file:
        # Traverse the file to count groups
        h5file.visititems(count_groups_recursive)

    return group_count

def save_many_structures(elements_in, elements_out, include_all=True, complexity='single', max_amount = 100):
    print("Searching ChemSpider...")
    search = cs.filter_element(elements_in, elements_out, include_all = include_all, complexity=complexity)

    while cs.filter_status(search)['status'] != 'Complete':
        time.sleep(1E-3)
    print(cs.filter_status(search))
    cids = cs.filter_results(search)
    # Define the range for the indices


    rand_mols = random.sample(cids, max_amount)
    with h5py.File(FILE_NAME, 'a') as f:
        # Create or access the group
        print("Writing h5 file")
        repeat = 0
        for idx in rand_mols:
            mol = cs.get_details(idx)
            coor, atom_sum = gt.mol2_to_xyz(mol['mol2D'])
            coor = gt._get_modified_coor(coor, atom_sum)
            #mol['xyz'] = [coor]
            _, I_at, I_mol, s_new = gt.get_I_from_xyz(coor, atom_sum)
            sM = s_new * (I_mol/I_at)
            group_name = f"{mol['id']}"
            mol['sM'] = [sM]

            print(group_name)
            if group_name in f:
                group = f[group_name]
                repeat =+ 1 
                print(f"Rewriting group {group_name}")
            else:
                group = f.create_group(group_name)

            group.attrs['note'] = f"Random download set featuring {elements_in} and without {elements_out}"
            for dataset_name, data in mol.items():
            # Append run number to the dataset name
                run_dataset_name = f"{dataset_name}"
        
                # Create or overwrite the dataset within the group
                if run_dataset_name in group:
                    del group[run_dataset_name]

                group.create_dataset(run_dataset_name, data=data)
                time.sleep(1E-2)
        print(f"Data for {len(rand_mols)} molecules saved to group in {FILE_NAME} successfully with {repeat} repeats.")

if __name__ == '__main__':
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = [executor.submit(save_many_structures, ['H', 'C', 'I'], ['D', 'N', 'O'], max_amount=50) for _ in range(5)]
        for r in concurrent.futures.as_completed(results):
            print(r.result)
    num_groups = count_groups(FILE_NAME)
    print(f"Total number of data sets in training data is now {num_groups}")