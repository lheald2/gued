import subprocess
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


# === CONFIGURATION ===
input_dir = Path("C:\\Users\\laure\\Documents\\Centurion_Lab\\orca_simulations\\HBr_Scan\\")  # folder with .inp files
orca_path = Path("C:/ORCA_6.1.0/orca.exe")  # full path to ORCA executable



def run_orca(inp_file, out_file = None):
        if out_file == None:
            out_file = inp_file.with_suffix(".out")
        print(f"\n📄 Running: {inp_file.name}")
        
        # Run ORCA using subprocess
        try:
            with open(out_file, "w") as out_f:
                subprocess.run([str(orca_path), str(inp_file)], stdout=out_f, stderr=subprocess.STDOUT)
            print(f"Finished: {out_file.name}")
        except Exception as e:
            print(f"Failed to run {inp_file.name}: {e}")

def get_final_energy(out_file):
    find_string_start = 'FINAL SINGLE POINT ENERGY'
    
    file = open(out_file, 'r')
    inlines = file.readlines()
    file.close()

    # First step : Find block of interest
    linenum = 0
    found = False
    for line in inlines:
        if not found:
            if (line.find(find_string_start)) > -1:
                found = True
            else:
                linenum += 1
    if not found:
         raise Exception(f"Couldn't find {find_string_start}")
    
    temp = list(map(str, inlines[linenum].split()))
    #print(temp[-1])
    final_energy = float(temp[-1])

    return final_energy


def get_absorption_spectrum(out_file):
    
    find_string_start = "                     ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS    "
    find_string_end = "                     ABSORPTION SPECTRUM VIA TRANSITION VELOCITY DIPOLE MOMENTS    "

    eV_values = {"states": [], "abs_energies": [], "rel_energies": []}
 
    # Retrieve single point energy
    final_energy = get_final_energy(out_file)

    file = open(out_file, 'r')
    inlines = file.readlines()
    file.close()

    # First step : Find block of interest
    linenum = 0
    found = False
    for line in inlines:
        if not found:
            if (line.find(find_string_start)) > -1:
                found = True
            else:
                linenum += 1
    if not found:
         raise Exception(f"Couldn't find {find_string_start}")

    # Remove all of the stuff before the first occurrence
    inlines = inlines[linenum:]

    linenum = 0
    found = False
    for line in inlines:
        if not found:
            if (line.find(find_string_end)) > -1:
                found = True
            else:
                linenum += 1

    if not found:
        raise Exception(f"Couldn't find {find_string_end} line")

    inlines = inlines[:linenum]

    eV_values['states'].append('0')
    eV_values['rel_energies'].append(0.0)
    eV_values['abs_energies'].append(final_energy)

    for line in inlines[5:-2]:
        string = list(map(str, line.split()))
        eV_values["states"].append(string[2][:-3])
        eV_values["rel_energies"].append(string[3])
        eV_values["abs_energies"].append(float(string[3])+final_energy)
    
    return eV_values




# === RUN LOOP ===
inp_files = sorted(input_dir.glob("*.inp"))  # sort for consistency
# print(inp_files)

# === RUNS ORCA CALCULATIONS === 
# for inp_file in inp_files:
#     run_orca(inp_file)

out_files = sorted(input_dir.glob("*.out"))

# # === Testing final energy function ===
# for out_file in out_files[:]:
#     get_final_energy(out_file)

#print(out_files)
all_results = {}
for out_file in out_files[:]:
    string = str(out_file)
    temp = list(map(str, string.split('\\')))
    dict_key = temp[-1][-7:-4]
    dictionary = get_absorption_spectrum(out_file)
    all_results[f"{dict_key}"] = dictionary

# # Reconstruct data

state_info = {state: [] for state in all_results['001']['states']}

for key in all_results.keys():
    for i, state in enumerate(all_results[key]['states']):
        #print(f"state: {state} energy: {all_results[key]['energies'][i]}")
        #state_info[state].append(float(all_results[key]["rel_energies"][i]))
        state_info[state].append(float(all_results[key]["abs_energies"][i]))

print(state_info.keys())
gs_min = np.min(state_info['0'])
x_vals = np.arange(0.7, 4, 0.15)
plt.figure()
for state in list(state_info.keys())[:]:
    temp = np.array(state_info[state])+np.abs(gs_min)
    plt.plot(x_vals[:], temp, "k")
    plt.xlabel("Reaction Coordinate")
    plt.ylabel("eV")
    plt.ylim(-5, 30)
plt.show()

# TODO 
"""Incorporate ground state well in the PES scan and make into all one function"""
