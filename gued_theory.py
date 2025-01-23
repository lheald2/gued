import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import h5py
import re
from collections import defaultdict
sys.path.append('packages')
import warnings
from scipy import signal
import scipy.interpolate as interp
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
import scipy.stats
from PIL import Image
from matplotlib.colors import TwoSlopeNorm
from mendeleev import element
from colour import Color
warnings.simplefilter("ignore")
from gued_globals import *

#Authors: Lauren F. Heald, Keke Chen
#Contact: lheald2@unl.edu, ckk20@mails.tsinghua.edu.cn
# Path for Theory Package


#path_dcs = '/sdf/home/l/lheald2/GUED/jupyter_notebook/user_notebooks/dcs_repository/3.7MeV/'
TABLE = pd.read_csv(PATH_DCS+'Periodic_Table.csv')
angs = '\u00C5' 

def import_s():
    """ This functions uses the C.dat files as an example file to generate values of s for the simulation calculations. 
    
    RETURNS:

    s (array) = array of s values which correspond to calculated scattering intensities for each atom.
    """
    
    qe=1.602176565e-19 
    me=9.10938291e-31
    c=299792458 
    h=6.62606957e-34
    E=3700000*qe+me*c**2 #kinetic energy=3.7MeV
    p=(E**2/c**2-me**2*c**2)**0.5
    lamb=h/p
    k=2*np.pi/lamb #wave vector of the incident electron

    path=PATH_DCS+'C.dat'

    with open(path,'r') as file:
        a=file.read()

    a0=a.split('\n')
    theta_deg=np.empty(130)

    for i in range(130):
        a31=str(a0[31+i]).split(' ')
        theta_deg[i]=a31[2]
    
    theta=theta_deg*np.pi/180
    S=2*k*np.sin(0.5*theta)
    s=np.array(S)
    return s


def read_dat_dcs(atom_no):
    """ 
    Reads in the scattering intensity (form factors) for each atom in the molecule of interest from the .dat files calculated using ELSEPA. 
    
    ARGUMENTS: 

    atom_no (int):
        maximum atomic number of interest (default value is 55)
    
    GLOBAL VARIABLES:

    PATH_DCS (string):
        path to the folder containing the .dat files
    
    RETURNS:

    data (array):
        values of the scattering intensities in cm taken from the .dat files. 
    """
    
    atom_sym=no_to_sym(atom_no)
    path=PATH_DCS+atom_sym+'.dat'
    with open(path,'r') as file:
        a=file.read()
    a0=a.split('\n')
    data=np.empty(130)
    for i in range(130):
        a31=str(a0[31+i]).split(' ')
        #print(a31)
        data[i]=a31[6]
    #print(data)
    return data**0.5 ## returns in cm


def sym_to_no(atom_symbol):
    """ 
    Short cut for getting the atomic number from the atomic symbol.
    
    ARGUMENTS: 

    atom_symbol (string):
        atomic symbol
    
    RETURNS:

    atom_number (int):
        atomic number
    """
    
    n=np.where(TABLE['Symbol']==atom_symbol)
    atom_number = int(n[0]+1)
    return atom_number


def no_to_sym(atom_number):
    """ 
    Short cut for getting the atomic symbol from the atomic number. 
    
    ARGUMENTS: 
    atom_number (int):
        atomic number
    
    RETURNS:
    atom_symbol (string):
        atomic symbol    
    """
    
    atom_symbol = TABLE['Symbol'][atom_number-1]
    return atom_symbol


def import_DCS(max_at_no=55):
    """ 
    Uses read_dat_dcs to get the form factors for all the atoms available. 
    
    ARGUMENTS: 
    max_at_no (int):
        maximum atomic number of interest (default value is 55)
    
    RETURNS:

    f (array):
        form factors for all atoms
    """
    
    f=np.empty((max_at_no+1,130))
    for i in range(max_at_no):
        f[i+1]=read_dat_dcs(i+1)
    return f

# Set up Constants
FORM_FACTORS = import_DCS()
S_THEORY = import_s()

def load_molecular_structure(path_mol, mol_name, file_type = [], mol2_str = None):
    """ Reads in either a .csv or .xyz file containing moleculear coordinates and adds a column containing the atomic number using the 
    hidden function _get_modified_coor for each atom in the molecule. When reading in an .xyz file runs the hidden function _load_xyz. 
    Errors are thrown if an improper file type is chosen or if the .xyz or .csv file needs further formatting.
    
    ARGUMENTS:
    
    path_mol (string):
        path to the directory of the molecular structure
    mol_name (string):
        file name of the structural file used for the simulation
    file_type (string):
        either xyz or csv depending on what the file being used is. Determines treatment
    
    RETURNS:
    
    coor (array): 
        N x 5 array where N = # of atoms. Column 0 contains the atomic symbol, columns 1, 2, and 3 contain x, y, and z coordinates
        and column 4 contains the atomic number. 
    atom_sum (int):
        total number of atoms in the molecule
    """
    
    filename=path_mol + mol_name + file_type

    if file_type=='.xyz':
        coor_xyz, atom_sum = _load_xyz(filename)
        coor = _get_modified_coor(coor_xyz, atom_sum)
        
    if file_type =='.csv':
        mol_filename = mol_name+'.csv'
        coor_M = pd.read_csv(path_mol+mol_filename)
        coor = np.array(coor_M)
        num = np.array(coor[:,3])
        atom_sum = int(len(num))
        coor = _get_modified_coor(coor, atom_sum)
    if file_type == None and type(mol2_str) == str:
        coor, atom_sum = mol2_to_xyz(mol2_str)
        coor = _get_modified_coor(coor, atom_sum)
    elif file_type!='.csv' and file_type!='.xyz':
        print('error! Please type in the right molecular coordinate file type, .xyz or .csv')

        
    return coor,atom_sum


def mol2_to_xyz(mol2_str):
    """Function for converting a mol2 string taken from an online source such as pubchem or chemspider and converts to xyz coordinates."""
    atoms_section = False
    coor = []

    mol2_str = mol2_str.split('\n')
    start = 4 
    max_line = len(mol2_str)
    for i in range(start, max_line):
        string = mol2_str[i].split()
        #print(string)
        if len(string) > 10:
            atom_name = string[3]
            x = np.float64(string[0])
            y = np.float64(string[1])
            z = np.float64(string[2])
            coor.append([atom_name, x, y, z])
    coor = np.array(coor)
    atom_sum = len(coor)
    return coor, atom_sum


def _load_xyz(xyz_file):
    """
    Reads in an .xyz generated from programs such as Gaussian or ORCA.
    
    ARGUMENTS: 

    xyz_file (string):
        full path to the .xyz file of interest.
    
    RETURNS: 

    coordinates (array):
        coordinate array of N (# of atoms) x 4 shape with column 0 containing atomic symbol, and columns 1, 2, and 3 containing x, y, z 
        coordinates
    atom_sum (int):
        total number of atoms in the molecule
    """

    file = open(xyz_file, 'r')
    text = file.readlines()
    file.close()
    count = len(text)
    coordinates = []
    for j in range(0, count):
        try:
            string = list(map(str, text[j].split()))
            coordinates.append(string)
        except Exception:
            pass    
    atom_sum = coordinates[0]
    atom_sum = int(np.array(atom_sum))
    coordinates = np.array(coordinates[2:])

    return coordinates, atom_sum


def get_bonds(xyz_file):

    coordinates, atom_sum = _load_xyz(xyz_file)
    atom_ids = coordinates[:,0]
    x = coordinates[:,1].astype(float)
    y = coordinates[:,2].astype(float)
    z = coordinates[:,3].astype(float)


    atomic_num = []
    radii = []
    colors = []

    for ident in atom_ids:
        elem = element(ident)
        num = elem.atomic_number
        atomic_num.append(num)
        rad = elem.atomic_radius
        radii.append(rad / 75)
        color = elem.cpk_color
        new = Color(color).rgb
        colors.append(new)

    connections = []
    lengths = []
    for i in range(0, len(atomic_num)):
        for j in range(i+1, len(atomic_num)):
            if i == j:
                pass
            bond_length = (x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2 + (z[i] - z[j]) ** 2
            bond_length = bond_length ** 0.5
            max_bond_length = 1.1 * (radii[i] + radii[j])

            if bond_length < max_bond_length:
                #connections.append((atom_ids[i], atom_ids[j]))
                print(atom_ids[i], atom_ids[j])
                print(bond_length, max_bond_length)
                #lengths.append(bond_length)

    # lengths = np.array(lengths)
    # return connections, lengths


def load_freq_xyz(path_mol, mol_name, file_type):
    """
    Reads in a frequency trajectory .xyz file containing many structures which evolve over time generated from programs such as Gaussian or
    ORCA. The file also contains information on the time points for each structural evolution.

    ARGUMENTS:

    path_mol (string):
        path to the directory of the molecular structure
    mol_name (string):
        file name of the structural file used for the simulation
    file_type (string):
        either xyz or csv depending on what the file being used is.

    RETURNS:

    re (array):
        array of atom symbol, x, y, z, and atom number for each time step
    atom_sum (int):
        total number of atoms in the molecule
    time (array):
        time points corresponding to the simulation in fs
    """

    filename = path_mol + mol_name + file_type
    xyz_file = filename
    file = open(xyz_file, 'r')
    text = file.readlines()
    file.close()
    count = len(text)
    re = []

    atom_sum = list(map(int, text[0].split()))
    atom_sum = atom_sum[0]
    iteration = atom_sum + 2

    groups = np.arange(0, count, (iteration))

    temp = (list(map(str, text[atom_sum + 3].split())))
    inv_cm = float(temp[2])
    print(inv_cm)
    fs = (1 / (inv_cm * 2.99e10)) * 1e15
    print("fs=", fs)
    fs_step = fs / len(groups)
    time = np.arange(0, fs, fs_step)

    for j in range(len(groups)):
        temp = []
        lines = np.arange(groups[j] + 2, groups[j] + iteration)
        for line in lines:
            string = list(map(str, text[line].split()))
            atom_num = sym_to_no(string[0])
            info = string[0:4] + [str(atom_num)]
            temp.append(info)
            # print(string)
        re.append(temp)

    re = np.array(re)

    return re, atom_sum, time


def load_traj_xyz(path_mol, mol_name, file_type):
    """
    Reads in a simulated trajectory .xyz file containing many structures which evolve over time generated from programs such as Gaussian or
    ORCA. The file also sometimes contains information on the time points for each structural evolution.

    ARGUMENTS:

    path_mol (string):
        path to the directory of the molecular structure
    mol_name (string):
        file name of the structural file used for the simulation
    file_type (string):
        either xyz or csv depending on what the file being used is.

    RETURNS:

    coordinates (array):
        array of atom symbol, x, y, z, and atom number for each time step
    atom_sum (int):
        total number of atoms in the molecule
    time (array):
        time points corresponding to the simulation in fs
    """

    filename = path_mol + mol_name + file_type
    xyz_file = filename
    file = open(xyz_file, 'r')
    text = file.readlines()
    file.close()
    count = len(text)
    coordinates = []

    atom_sum = list(map(int, text[0].split()))
    atom_sum = atom_sum[0]
    iteration = atom_sum + 2

    groups = np.arange(0, count, (iteration))

    for j in range(len(groups)):
        temp = []
        lines = np.arange(groups[j] + 2, groups[j] + iteration)
        for line in lines:
            string = list(map(str, text[line].split()))
            atom_num = sym_to_no(string[0])
            info = string[0:4] + [str(atom_num)]
            temp.append(info)
            # print(string)
        coordinates.append(temp)

    coordinates = np.array(coordinates)
    return coordinates


def _get_modified_coor(coor, atom_sum):
    """ 
    Appends a column of atomic numbers to the coordinate array read from the .xyz file
    
    ARGUMENTS: 

    re (array):
        coordinate array of N (# of atoms) x 4 shape with column 0 containing atomic symbol, and columns 1, 2, and 3 containing x, y, z 
        coordinates
    atom_sum (int):
        total number of atoms in the molecule
    
    RETURNS: 

    coor (array): 
        N x 5 array where N = # of atoms. Column 0 contains the atomic symbol, columns 1, 2, and 3 contain x, y, and z coordinates
        and column 4 contains the atomic number. 
    """
    atom_num=[0 for i in range(atom_sum)]
    for i in range(atom_sum):
        atom_num[i]=sym_to_no(coor[i][0])
            
    atom_num=np.array(atom_num)
    atom_num=atom_num[:,np.newaxis]
    coor=np.hstack((coor, atom_num))

    return coor
    

def get_I_atomic(coor, atom_sum, s_val=12):
    """
    Calculates the I_atomic scattering pattern for the molecule of interest. 

    ARGUMENTS: 

    f (array):
        Array of form factors for all elements
    s000 (array):
        s values of interest in calculation
    s_max (int):
        maximum s value for consideration
    coor (array): 
        coordinates of molecule
    atom_sum (int):
        total number of atoms in molecule
    
    RETURNS:

    I_at (array):
        values for I atomic
    s_new (array):
        s values related to I atomic
    """

    I_at_all = []
    s_angstrom = S_THEORY*1e-10 #in angstroms

    
    if isinstance(s_val, int):
        s_new = np.linspace(0, s_val, 500)
    elif isinstance(s_val, np.ndarray):
        s_new = s_val
    elif isinstance(s_val, list):
        s_new = np.array(s_val)


    for i in range(atom_sum):
        I_atomic = []
        I_at = 0
        amps = FORM_FACTORS[int(coor[i,4])]
        #print(amps)
        interp_amps = interp.interp1d(s_angstrom[0:125], amps[0:125])
        amps_new = interp_amps(s_new)
        for k in range(len(amps_new)):
            f_new = amps_new[k]
            I_at = np.abs(f_new)**2
            I_atomic.append(float(I_at))
        I_at_all.append(I_atomic)
    I_at = sum(np.array(I_at_all))

    return I_at, s_new


def get_I_molecular(coor, atom_sum, s_val=12):
    """
    Calculates the I_molecular scattering pattern for the molecule of interest. 

    ARGUMENTS: 

    f (array):
        Array of form factors for all elements
    s000 (array):
        s values of interest in calculation
    s_max (int):
        maximum s value for consideration
    coor (array): 
        coordinates of molecule
    atom_sum (int):
        total number of atoms in molecule
    
    RETURNS:

    I_mol (array):
        values for I molecular
    s_new (array):
        s values related to I molecular
    """
    x = np.array(coor[:, 1])
    y = np.array(coor[:, 2])
    z = np.array(coor[:, 3])




    s_angstrom = S_THEORY * 1e-10 #convert to angstroms
    
    if isinstance(s_val, int):
        s_new = np.linspace(0, s_val, 500)
    elif isinstance(s_val, np.ndarray):
        s_new = s_val
    elif isinstance(s_val, list):
        s_new = np.array(s_val)

    I_mol = np.zeros(len(s_new))
    for i in range(atom_sum):
        for j in range(atom_sum): # Does each atom pair calculation twice
            if i != j:
                r_ij = (float(x[i]) - float(x[j])) ** 2 + (float(y[i]) - float(y[j])) ** 2 + (float(z[i]) - float(z[j])) ** 2
                r_ij = r_ij ** 0.5
                #print(f"bond length between {coor[i, 0]} and {coor[j, 0]} = {r_ij}")
                amps_i = FORM_FACTORS[int(coor[i,4])]
                amps_j = FORM_FACTORS[int(coor[j,4])]
                interp_amps_i = interp.interp1d(s_angstrom[0:125], amps_i[0:125])
                interp_amps_j = interp.interp1d(s_angstrom[0:125], amps_j[0:125])
                amps_new_i = interp_amps_i(s_new)
                amps_new_j = interp_amps_j(s_new)
                #print(len(amps_new_j))
                I_mol[0]+=FORM_FACTORS[int(coor[i,4]),0]*FORM_FACTORS[int(coor[j,4]),0]
                I_mol[1:len(s_new)]+=amps_new_i[1:len(s_new)]*amps_new_j[1:len(s_new)]*np.sin(
                    s_new[1:len(s_new)]*r_ij)/s_new[1:len(s_new)]/r_ij
                
    
    return I_mol, s_new


def get_I_from_xyz(coor, atom_sum, s_val=12):
    """
    Calculates the total scattering of the molecule of interest by using the functions get_I_atomic and get_I_molecular. 
    
    ARGUMENTS: 

    f (array):
        Array of form factors for all elements
    s000 (array):
        s values of interest in calculation
    s_max (int):
        maximum s value for consideration
    coor (array): 
        coordinates of molecule
    atom_sum (int):
        total number of atoms in molecule
        
    RETURNS:

    I (array): 
        sum of I atomic and I molecular
    I_at (array):
        values for I atomic
    I_mol (array):
        values for I molecular
    s_new (array):
        s values related to I
    """
    I_at, s_new = get_I_atomic(coor, atom_sum, s_val)
    I_mol, _ = get_I_molecular(coor, atom_sum, s_val)
    I = I_at + I_mol

    return I, I_at, I_mol, s_new


def get_sM_and_PDF_from_I(I_at,I_mol,s,r_max,damp_const):
    """ 
    Calculates the sM and PDF from the simulated I atomic and I molecular for the molecule of interest. 
    
    ARGUMENTS: 
    
    I_at (array):
        values for I atomic
    I_mol (array):
        values for I molecular
    s (array):
        scattering range of interest
    r_max (int):
        maximum radius for consideration
    damp_const (int):
        damping constant for the fourier transform 
    
    RETURNS:

    sM (array): 
        modified scattering intensity
    PDF (array):
        pair distribution function
    r (array):
        radial values corresponding to the PDF
    """

    sM=I_mol/I_at*s #calculate sM from I
    r_max = r_max * 1; # convert to picometer
    r=range(r_max)
    #print(r)
    PDF=[0 for i in range(r_max)]

    for i in range(len(s)-1): 
        PDF+=sM[i]*np.sin(s[i]*1e10*np.array(r)*1e-12)*(s[i+1]-s[i])*np.exp(-s[i]**2/damp_const)
    #for i in range(r_max):
     #   PDF[i]+=sum(sM*np.sin(s*1e10*np.array(r[i])*1e-12)*(s[1]-s[0])*np.exp(-s**2/damp_const))

    return sM,PDF,np.array(r)


def apply_conv(matrix_before_conv,x_range,col,t_interval,nt,space_for_convol, plot=False):
    """ ADD DOC STRING"""

    x0 = np.linspace(-col,col,int(255/t_interval))
    h = np.exp(-x0**2*t_interval**2/8000)/(np.pi*8000/t_interval**2)**0.5 #normalize the gaussian
    M1 = get_2d_matrix(x_range,nt+space_for_convol*2)
    for i in range(x_range):
        M1[i]=signal.convolve(matrix_before_conv[:,i],h,mode='same')

    #M1=np.transpose(M1)
    M1 = np.array(M1)
    print(M1.shape)
    
    if plot == True:
        norm = TwoSlopeNorm(vmin=M1.min(),vcenter=0,vmax=M1.max())
        plt.figure(figsize=(15,5))
        pc=plt.imshow(M1[:, 0:nt+space_for_convol-1],norm=norm,cmap=plt.get_cmap('seismic'),alpha=0.65)
        plt.colorbar(pc)
        ax=plt.gca()
        ax.invert_yaxis()
        ax.xaxis.set_ticks_position('bottom')
        plt.xlabel('time/fs')
        #plt.xticks(np.arange(0,nt+space_for_convol,100),np.arange(-space_for_convol*t_interval,nt*t_interval,100*t_interval))
        #plt.axhline(y=space_for_convol,linestyle='--')
        plt.grid()
    return M1


def plot_I_sM_PDF(I,sM,PDF,s,r,title_I,title_sM,title_PDF):
    """ Plots I total, sM and PDF"""

    plt.figure()
    plt.subplot(1,3,1)
    plt.plot(s,I/I.max())
    plt.xlabel('s/angs^-1')
    plt.title(title_I)
    plt.grid()
    
    plt.subplot(1,3,2)    
    plt.plot(s,sM)
    plt.xlabel('s/angs^-1')
    plt.title(title_sM)
    plt.grid()
    
    plt.subplot(1,3,3)    
    plt.plot(r,PDF)
    plt.xlabel('r/pm')
    plt.title(title_PDF)
    plt.grid()
    
    plt.tight_layout()
    plt.show()
    return


def load_time_evolving_xyz(path_mol, mol_name, file_type):
    """
    Reads in a trajectory .xyz file containing many structures which evolve over time generated from programs such as Gaussian or ORCA. The
        file also contains information on the time points for each structural evolution.

    ARGUMENTS:

    path_mol (string):
        path to the directory of the molecular structure
    mol_name (string):
        file name of the structural file used for the simulation
    file_type (string):
        either xyz or csv depending on what the file being used is.

    RETURNS:

    coor_txyz (array):
        array of atom symbol, x, y, z, and atom number for each time step
    atom_sum (int):
        total number of atoms in the molecule
    time (array):
        time points corresponding to the simulation in fs (??)
    """

    mol_filename = mol_name + file_type
    with open(path_mol + mol_filename, 'r') as f:
        a = f.read()

    a0 = a.split('\n')
    atom_sum = int(a0[0])  # get the total atom number in the molecule, this does great help

    time_count = int((len(a0) - 1) / (atom_sum + 2))  # find how many time points are there in the time evolution file
    time = [0 for i in range(time_count)]
    print("count = ", time_count)

    coor_txyz = get_3d_matrix(time_count, atom_sum, 4)
    # coor_txyz[time order number][atom type][coordinate xyz]

    m = 0
    n = 0
    o = 0
    # just little tricks to move the data into right place, from the file to array
    # don't be confused by these parameters
    for i in range(time_count):
        m = 0
        a1 = str(a0[(atom_sum + 2) * i + 1]).split(' ')
        for j in a1:
            if j == 't=':
                m = 1
            if j != '' and j != 't=' and m == 1:
                time[i] = j
                break
        for j in range(atom_sum):
            a1 = str(a0[(atom_sum + 2) * i + 2 + j]).split(' ')
            for k in a1:
                if k != '':
                    coor_txyz[i][n][o] = k
                    o += 1
            o = 0
            n += 1
        n = 0
    print(len(coor_txyz[0][0]))
    for i in range(time_count):
        coor1 = _get_modified_coor(coor_txyz[i][:][:], atom_sum)
        coor_txyz[i] = coor1

    return np.array(coor_txyz), atom_sum, time


def static_sim(path_mol, mol_name, file_type, s_max = 12, r_max = 800, damp_const = 33, plot=True, return_data = False):
    """
    Calculates the static scattering of a molecule based on the structure. First, reads in the molecular structure using 
    load_molecular_structure then calculates the I atomic and I molecular using get_I_from_xyz. Then, using the I atomic and I molecular,
    calculated the sM and PDF using the get_sM_and_PDF_from_I function.
    
    ARGUMENTS:

    path_mol (str):
        path to the folder with the trajectory simulation
    mol_name (str):
        name of the xyz file
    file_type (str):
        '.xyz' or '.csv' but I think it only works with xyz currently 

    OPTIONAL ARGUMENTS:

    s_max (int):
        Default set to 12 inverse angstroms. Defines the maximum scattering range for consideration
    r_max (int):
        Default set to 800 (picometers). Defines the maximum r distance for consideration
    damp_const (int):
        Default set to 33. Defines the size of the damping constant used in the Fourier transform
    plot (boolean):
        Default set to True. When true, plots the dI/I and PDF with respect to time and distance
    return_data (boolean):
        Default set to False. When true, returns I atomic, I molecular, s, sM, PDF, and r.
        
    """

    coor, atom_sum  = load_molecular_structure(path_mol,mol_name,file_type)
    _, I_at, I_mol, s_new = get_I_from_xyz(coor, atom_sum, s_max)
    sM, PDF, r_new = get_sM_and_PDF_from_I(I_at, I_mol, s_new, r_max, damp_const)

    if plot == True:
        plt.figure(figsize=(8,4))
            
        plt.subplot(1,2,1)    
        plt.plot(s_new,sM)
        plt.xlabel(r'S, ['+angs+'$^{-1}$]')
        plt.ylabel('sM(s)')
        plt.title('Modified Scattering Intensity')
        plt.grid()
            
        plt.subplot(1,2,2)    
        plt.plot(r_new, PDF)
        plt.xlabel(r'R, [pm]')
        plt.ylabel('PDF')
        plt.title('Pair Distribution Function')
        plt.grid()
            
        plt.tight_layout()
        plt.show()
    
    if return_data == True:
        return I_at, I_mol, s_new, sM, PDF, r_new
    else:
        return


def dI_I_from_trajectory(path_mol, mol_name, file_type, s_max=12, tstep = 0.5, plot=False):
    """ 
    Calculates the scattering and plots results of a trajectory simulation with a series of xyz coordinates and corresponding time steps.
    Uses the functions load_time_evolving_xyz, get_I_from_xyz, get_2d_matrix, and plot_delay_simulation_with_conv. 
    
    ARGUMENTS:
    
    path_mol (str):
        path to the folder with the trajectory simulation
    mol_name (str):
        name of trajectory file
    file_type (str):
        '.xyz' or '.csv' but I think it only works with xyz currently 
        
    OPTIONAL ARGUMENTS:
    
    s_max (int):
        Default set to 12 inverse angstroms. Defines the maximum scattering range for consideration
        
    RETURNS: 
    
    nothing right now lol
    """
    coor_txyz, atom_sum, time=load_time_evolving_xyz(path_mol,mol_name,file_type) #load xyz data
    #options: load_time_evolving_xyz, or load_time_evolving_xyz1

    time_steps = len(time)
    if float(time[1])-float(time[0])!=0:
        t_interval = float(time[1])-float(time[0])
        print(t_interval)
    elif float(time[1])-float(time[0]) == 0:
        t_interval = tstep
        print(t_interval)

    I0,I0_at,I0_mol,s_new = get_I_from_xyz(coor_txyz[0],atom_sum, s_max)
    dI_I_t = []
    neg_time = 100/t_interval

    for i in range(0, int(neg_time)):
        dI_I = np.zeros(I0.shape)
        dI_I_t.append(dI_I)
    for i in range(time_steps):
        I,I_at,I_mol,s = get_I_from_xyz(coor_txyz[i],atom_sum, s_max)
        dI_I_t.append((I-I0)/I)

    dI_I_t=np.array(dI_I_t)
    t_fs = np.linspace(-100, 1000, len(dI_I_t))

    if plot == True:
        norm = TwoSlopeNorm(vmin=dI_I_t.min(),vcenter=0,vmax=dI_I_t.max())
        plt.figure(figsize=FIGSIZE)
        plt.pcolor(t_fs, s, dI_I_t.T, norm=norm, cmap="bwr")
        plt.xlabel("time (fs)")
        plt.ylabel("s (A^-1)")
        plt.colorbar()
        plt.show()
    
    return dI_I_t, s, t_fs

    
def trajectory_sim(path_mol, mol_name, file_type, s_max=12, tstep = 0.5, plot=False, return_data=False):
    """ 
    Calculates the scattering and plots results of a trajectory simulation with a series of xyz coordinates and corresponding time steps.
    Uses the functions load_time_evolving_xyz, get_I_from_xyz, get_2d_matrix, and plot_delay_simulation_with_conv. 
    
    ARGUMENTS:
    
    path_mol (str):
        path to the folder with the trajectory simulation
    mol_name (str):
        name of trajectory file
    file_type (str):
        '.xyz' or '.csv' but I think it only works with xyz currently 
        
    OPTIONAL ARGUMENTS:
    
    s_max (int):
        Default set to 12 inverse angstroms. Defines the maximum scattering range for consideration
    plot (boolean):
        Default set to False. When true, shows a plot of data with a convolution applied
    return_data (boolean):
        Default set to False. When true, returns dI/I, s, and time arrays
        
    RETURNS: 
        see above
    """

    coor_txyz, atom_sum, time=load_time_evolving_xyz(path_mol,mol_name,file_type) #load xyz data
    #options: load_time_evolving_xyz, or load_time_evolving_xyz1

    time_steps = len(time)
    if float(time[1])-float(time[0])!=0:
        t_interval = float(time[1])-float(time[0])
        print(t_interval)
    elif float(time[1])-float(time[0]) == 0:
        t_interval = tstep
        print(t_interval)

    col = int(160/t_interval)
    space_for_convol = int(200/t_interval)

        
    I0,I0_at,I0_mol,s_new = get_I_from_xyz(coor_txyz[0],atom_sum, s_max)
    delta_I_over_I_t = get_2d_matrix(time_steps+space_for_convol*2,len(s_new))

    for i in range(time_steps):
        I,I_at,I_mol,s = get_I_from_xyz(coor_txyz[i],atom_sum, s_max)
        delta_I_over_I_t[i+space_for_convol]=(I-I0)/I
    
    for i in range(space_for_convol):
        delta_I_over_I_t[i+time_steps+space_for_convol]=delta_I_over_I_t[time_steps+space_for_convol-1]
    
    delta_I_over_I_t=np.array(delta_I_over_I_t)
    t_fs = np.linspace(-space_for_convol, 1000+space_for_convol, len(delta_I_over_I_t))
    if plot == True:
        dI_I_conv = apply_conv(delta_I_over_I_t*100,len(s),col,t_interval,time_steps,space_for_convol, plot=True)

        plt.ylabel('s/angs^-1')
        plt.yticks(np.arange(0,len(s),len(s)/s.max()),np.arange(0,s.max(),1))
        plt.axvline(x=space_for_convol,linestyle='--')
        plt.title('delta_I/I')
        plt.show()
    else:
        dI_I_conv = apply_conv(delta_I_over_I_t*100, len(s), col, t_interval, time_steps, space_for_convol, plot=False)
    if return_data == True:
        return delta_I_over_I_t, dI_I_conv.T, s, t_fs
    else:
        return


def freq_sim(path_mol, tra_mol_name, file_type, s_max=12, evolutions=10, r_max=800, damp_const=33, plot=True):
    """
    Calculates the delta I/I and PDF for a frequency simulation done by ORCA and saved as a .hess file. Plots a set mumber of evolutions 
    of the vibrational mode. Uses functions load_freq_xyz, get_I_from_xyz, and get_sM_and_PDF_from_I
    
    ARGUMENTS:

    path_mol (str):
        path to the folder with the trajectory simulation
    mol_name (str):
        name of trajectory file
    file_type (str):
        '.xyz' or '.csv' but I think it only works with xyz currently 

    OPTIONAL ARGUMENTS:

    s_max (int):
        Default set to 12 inverse angstroms. Defines the maximum scattering range for consideration
    evolutions (int):
        Default set to 10. Defines the number of frequency iterations to calculate and plot
    r_max (int):
        Default set to 800 (picometers). Defines the maximum r distance for consideration
    damp_const (int):
        Default set to 33. Defines the size of the damping constant used in the Fourier transform
    plot (boolean):
        Default set to True. When true, plots the dI/I and PDF with respect to time and distance

    RETURNS:
    
    dI_I (3d array):
        calculated delta I/I for the frequency motion 
    new_time (1d array):
        time steps within the frequency calculation
    s (1d array):
        corresponding s values for the dI/I
    PDF (3d array):
        calculated pair distribution function for the frequency motion
    r (1d array):
        corresponding pair distances
    """

    coor_txyz,atom_sum,time=load_freq_xyz(path_mol,tra_mol_name,file_type) #load xyz data
    nt=len(time)*evolutions
    max_time = max(time)*evolutions
    t_interval=float(time[1])-float(time[0])
    new_time = np.linspace(0, max_time, nt)
    # col=int(160/t_interval)
    # space_for_convol=int(200/t_interval)
        
    I0,I0_at,I0_mol,s = get_I_from_xyz(coor_txyz[0],atom_sum, s_max)
    dI_I= []
    PDF = []
    k = 0
    for i in range(nt):
        j = i%20
        I,I_at,I_mol,s = get_I_from_xyz(coor_txyz[j],atom_sum, s_max)
        dI_temp = (I-I0)/I
        dI_I.append(dI_temp)
        sM,pdf,r = get_sM_and_PDF_from_I(I_at,I_mol,s,r_max,damp_const)
        PDF.append(pdf)
    dI_I=np.array(dI_I)
    PDF = np.array(PDF)
    
    if plot == True:
        plt.figure(figsize=(15,10))

        plt.subplot(2,1,1)
        plt.pcolor(new_time, s, dI_I.T, cmap='bwr')
        plt.colorbar()
        plt.xlabel("time (fs)")
        plt.ylabel("s")
        plt.title("dI/I")

        plt.subplot(2,1,2)
        plt.pcolor(new_time, r, PDF.T, cmap='bwr')
        plt.colorbar()
        plt.xlabel("time (fs)")
        plt.ylabel("r (pm)")
        plt.ylim(50, 500)
        plt.title("PDF")
        plt.show()

    return dI_I, new_time, s, PDF, r


def dissoc_sim(path_mol, reactant, products, file_type, s_max=12, r_max=800, damp_const=33, plot=False):
    """
    Calculates the diffraction pattern of a dissociation reaction (or just any kind of structural change) from 2 or more structure files.
    Uses funcitons load_molecular_structure, get_I_from_xyz, and get_sM_and_PDF_from_I
    
    ARGUMENTS:

    path_mol (str):
        path to the folder with the trajectory simulation
    reactant (str):
        name of reactant xyz file
    products (list):
        list of product xyz files
    file_type (str):
        '.xyz' or '.csv' but I think it only works with xyz currently 

    OPTIONAL ARGUMENTS:

    s_max (int):
        Default set to 12 inverse angstroms. Defines the maximum scattering range for consideration
    evolutions (int):
        Default set to 10. Defines the number of frequency iterations to calculate and plot
    r_max (int):
        Default set to 800 (picometers). Defines the maximum r distance for consideration
    damp_const (int):
        Default set to 33. Defines the size of the damping constant used in the Fourier transform
    plot (boolean):
        Default set to True. When true, plots the dI/I and PDF with respect to time and distance

    RETURNS:

    dsM (2d array):
        delta sM of products/reactant
    s (1d array):
        s values for dsM
    dPDF (2d array):
        delta PDF with products/reactant
    r (1d array):
        corresponding pair distances
    """
    coor0, atom_sum0 = load_molecular_structure(path_mol, reactant, file_type)
    I0, I0_at,I0_mol,s = get_I_from_xyz(coor0,atom_sum0, s_max)
    sM0, pdf0, r = get_sM_and_PDF_from_I(I0_at,I0_mol,s,r_max,damp_const)
    

    I_prods = []
    sM_prods = []
    pdf_prods = []
    for i in range(len(products)):
        frag_name = str(products[i])
        coor, atom_sum = load_molecular_structure(path_mol, frag_name, file_type)
        I,I_at,I_mol,s = get_I_from_xyz(coor, atom_sum, s_max)
        I_prods.append(I)
        sM,pdf,r = get_sM_and_PDF_from_I(I0_at,I_mol,s,r_max,damp_const)
        pdf_prods.append(pdf)
        sM_prods.append(sM)
    
    I_prods = np.sum(I_prods, axis=0)
    sM_prods = np.sum(sM_prods, axis=0)
    pdf_prods = np.sum(pdf_prods, axis=0)
    dsM = s*(I_prods-I0)/I0_at
    
    if plot == True:
        angs = '\u00C5' # Angstrom sign
        plt.figure(figsize=(8,4))
        
        plt.subplot(1,2,1)
        plt.plot(s, sM0, label = "reactant")
        plt.plot(s, sM_prods, label="products")
        plt.xlabel(r'S, ['+angs+'$^{-1}$]');plt.ylabel('sM(s)');
        plt.legend()
        plt.title("sM")
        
        plt.subplot(1,2,2)
        plt.plot(r, pdf0, label="reactant")
        plt.plot(r, pdf_prods, label="products")
        plt.xlabel(r'R, [pm]');
        plt.legend()
        plt.title("PDF")

    dPDF=[0 for i in range(r_max)]
    for i in range(len(s)-1): 
        dPDF+=dsM[i]*np.sin(s[i]*1e10*np.array(r)*1e-12)*(s[i+1]-s[i])*np.exp(-s[i]**2/damp_const)
    
    return dsM, s, dPDF, r


def poly_fit(data_array, x_vals, degree = 2, plot=True, return_baseline=False):
    """
    Calculates a polynomial fit of the data_array with respect to the x_vals. 

    ARGUMENTS:

    data_array (1d or 2d array):
        1d or 2d data array to be fit, normally used on the dI/I or dI values after azimuthal averaging. Code checks the shape of the array
    x_vals (1d array):
        list of x values related to the data array (i.e., s values)

    OPTIONAL ARGUMENTS:

    degree (int):
        default set to True. Defines the degree of the polynomial used for fitting
    return_baseline (boolean):
        default set to False. When true, returns both the corrected data and the calculated baseline
    
    RESULTS:
    
    corrected_data (2d array):
        input 2d array - calculated baselines
    baselines (2d array):
        calculated baseline for each data set in the array. Only returned when return_baseline == True
    
    """

    if len(data_array.shape) == 2:
        baseline2d = []
        for i in range(len(data_array)):
            temp_data = np.copy(data_array[i])
            idx_nan = ~np.isnan(temp_data)
            coeff = np.polyfit(x_vals[idx_nan],temp_data[idx_nan], degree)
            baseline = np.polyval(coeff,x_vals)
            baseline2d.append(baseline)

        baseline2d = np.array(baseline2d)
        corrected_data = data_array - baseline2d
        if plot == True:
            plt.figure()
            plt.subplot(1,2,1)
            plt.plot(data_array[0])
            plt.plot(baseline2d[0])
            plt.xlabel("pixel")
            plt.title("delta I/I original with fit line")

            plt.subplot(1,2,2)
            plt.plot(corrected_data[0])
            plt.xlabel("pixel")
            plt.title("delta I/I corrected")

            plt.tight_layout()
            plt.show()
        
    elif len(data_array.shape) == 1:
        temp_data = data_array
        idx_nan = ~ np.isnan(temp_data)
        coeff = np.polyfit(x_vals[idx_nan], temp_data[idx_nan], degree)
        baseline2d = np.polyval(coeff, x_vals)
        
        corrected_data = data_array - baseline2d

        if plot == True:
            plt.figure()
            plt.subplot(1,2,1)
            plt.plot(data_array)
            plt.plot(baseline2d)
            plt.xlabel("pixel")
            plt.title("delta I/I original with fit line")

            plt.subplot(1,2,2)
            plt.plot(corrected_data)
            plt.xlabel("pixel")
            plt.title("delta I/I corrected")

            plt.tight_layout()
            plt.show()
    else:
        print("Data Array must be 1D or 2D array")

    if return_baseline == True:
        return corrected_data, baseline2d
    else:
        return corrected_data


def fit_high_s(data_array, x_vals, s_range=300, return_baseline=False, plot=False):
    """ 
    Calculates a linear fit to the high s values and subtracts the high s data from that line based on the slope. 
    
    ARGUMENTS:
    
    data_array (array):
        1D or 2D array
    x_vals (1D array):
        x_vals relating to the data_array (i.e., s_exp)
    
    OPTIONAL ARGUMENTS:
    
    s_range (int):
        Default set to 300. Index of the lowest s value to fit. 
    return_baseline (boolean):
        Default set to False. When true, returns the calculated baseline along with the corrected array
    plot (boolean):
        Default set to False. When true, plots a figure showing the baseline with respect to the original data and the new data.
        
    RETURNS:
    
    corrected_data (array):
        array matching the size of the input data_array with the baseline subtracted
    baseline (array):
        calculated baselines. Only returned when return_baseline==True"""

    data_fix = data_array
    if len(data_array.shape) == 2:
        corrected_data = []
        baseline = []
        for i in range(len(data_array)):
            temp_data = data_array[i]
            coeff = np.polyfit(x_vals[s_range:], temp_data[s_range:], 1)
            line = np.polyval(coeff, x_vals[s_range:])
            baseline.append(line)
            data_fix[i, s_range:] = temp_data[s_range:] - line
            corrected_data.append(data_fix[i])
        
        if plot==True:
            plt.figure()
            plt.subplot(2,1,1)
            plt.plot(x_vals, data_array[0])
            plt.plot(x_vals[s_range:], baseline[0])
            plt.title("Original Data with calculated Baseline")
            
            plt.subplot(2,1,2)
            plt.plot(corrected_data[0])
            plt.title("Baseline Subtracted Data")
            plt.show()

            
    elif len(data_array.shape) == 1:
        coeff = np.polyfit(x_vals[s_range:], data_array[s_range:], 1)
        baseline = np.polyval(coeff, x_vals[s_range:])
        data_fix[s_range:] = data_array[s_range:] - baseline
        corrected_data = data_fix
    
        if plot==True:
            plt.figure()
            plt.subplot(2,1,1)
            plt.plot(x_vals, data_array)
            plt.plot(x_vals[s_range:], baseline)
            plt.title("Original Data with calculated Baseline")
            
            plt.subplot(2,1,2)
            plt.plot(corrected_data)
            plt.title("Baseline Subtracted Data")
            plt.show()
    else:
        print("Data Array must be 1D or 2D array")
    
    corrected_data= np.array(corrected_data)

    if return_baseline == True:
        return corrected_data, baseline
    else:
        return corrected_data
    

def power_fit(data_array, x_vals, return_baseline = False, plot=False):
    """
    Fits the input data to a power function and subracts off the baseline (fit) to flatten data along zero axis. 
    
    ARGUMENTS:

    data_array (array):
        A 1D or 2D data array to be used for finding the power fit
    x_vals (1D array):
        x values associated with the data_array
    
    OPTIONAL ARGUMENTS:
    
    return_baseline (boolean):
        Default set to False. When true, returns both the corrected data and the calculated baselines
    plot (boolean):
        Default set to False. When true, plots and example of the original data with the calculated baseline and the corrected data

    RETURNS:

    corrected_data (array):
        Array of the same shape as input data_array with the calculated baseline subtracted off
    baseline (array):
        Array of the same shape as input data_array containing the calculated baselines (fits) for each data set. Only returned if 
        return_baseline == True
    """
    # Define power function
    def power_function(x, a, b):
        return a * np.power(x, b)

    # Perform the curve fitting
    if len(data_array.shape)==2:
        baseline = []
        for i in range(len(data_array)):
            params, _ = curve_fit(power_function, x_vals, data_array[i])
            a, b = params
            fit = power_function(x_vals, a, b)
            baseline.append(fit)
        baseline = np.array(baseline)
        corrected_data = data_array-baseline

        if plot == True:
            plt.figure()
            plt.subplot(1,2,1)
            plt.plot(data_array[0])
            plt.plot(baseline[0])
            plt.xlabel("pixel")
            plt.title("delta I/I original with fit line")

            plt.subplot(1,2,2)
            plt.plot(corrected_data[0])
            plt.xlabel("pixel")
            plt.title("delta I/I corrected")

            plt.tight_layout()
            plt.show()

    elif len(data_array.shape) == 1:
        params, _ = curve_fit(power_function, x_vals, data_array)
        a, b = params
        baseline = power_function(x_vals, a, b)
        corrected_data = data_array-baseline

        if plot == True:
            plt.figure()
            plt.subplot(1,2,1)
            plt.plot(data_array)
            plt.plot(baseline)
            plt.xlabel("pixel")
            plt.title("delta I/I original with fit line")

            plt.subplot(1,2,2)
            plt.plot(corrected_data)
            plt.xlabel("pixel")
            plt.title("delta I/I corrected")

            plt.tight_layout()
            plt.show()

    else:
        print("Please provide an array of data with a max shape length of 2")

    if return_baseline == True:
        return corrected_data, baseline
    
    else:
        return corrected_data


def bandpass_filter(data_array, ds, min_freq=0.001, max_freq=5, order = 4, plot=False):
    """
    Applies a bandpass filter to the input data to get rid of noise based on the minimum and maximum frequency using the scipy.signal.butter
    function. Min and Max frequencies can be estimated by the inverse of the bond lengths..?
    
    ARGUMENTS:

    data_array (array):
        1D or 2D array of data which will be bandpass filtered
    ds (float):
        s calibration value (or another value representing sampling rate)

    OPTIONAL ARGUMENTS:

    min_freq (float > 0):
        Default set to 0.001. Minimum value in the bandpass filter.
    max_freq (float > min_freq):
        Default set to 5. Maximum value for the bandpass filter.
    order (int):
        Default set to 4. Order of the butter bandpass filter. Higher-order filters will have a sharper cutoff but may introduce more 
        ripple or distortion in the passband. Lower-order filters will have a gentler transition and may be more stable.
    plot (boolean):
        Default set to False. When true, plots before and after filtering

    RETURNS:
    
    filtered_data (array):
        Array with the same shape as the input data_array with the bandpass filter applied. 
    """
    fs = 1 / ds
    nyquist = 0.5 * fs
    low = min_freq/nyquist
    high = max_freq/nyquist

    # Set up filter
    b, a = signal.butter(order, [low, high], btype='band')

    if len(data_array.shape)==2:
        filtered_data = []
        for i in range(len(data_array)):
            filtered_temp = signal.filtfilt(b, a, data_array[i])
            filtered_data.append(filtered_temp)
        filtered_data = np.array(filtered_data)

        if plot == True:
            plt.figure()
            plt.subplot(2,1,1)
            plt.plot(data_array[0])
            plt.title("Original Data")

            plt.subplot(2,1,2)
            plt.plot(filtered_data[0])
            plt.title("Bandpass Filtered Data")
            plt.tight_layout()
            plt.show()
            
    elif len(data_array.shape)==1:
        filtered_data = signal.filtfilt(b, a, data_array)

        if plot == True:
            plt.figure()
            plt.subplot(2,1,1)
            plt.plot(data_array)
            plt.title("Original Data")

            plt.subplot(2,1,2)
            plt.plot(filtered_data)
            plt.title("Bandpass filtered Data")
            plt.tight_layout()
            plt.show()

    return filtered_data
    

def get_exp_sM_PDF(coor, atom_sum, s_exp, dI, freq_filter=False, polyfit=False, degree=2, gauss_filter=False, sigma=1, 
                   powerfit = False, plot=False):
    """
    Calculates the sM and PDF for experimental dI. First, any zero offset at high s is corrected for then the I(atomic) is calculated for
    the appropriate s coordinates relating to the experiment. Then calculates sM and fills in nan values before calculating
    the PDF from the I(atomic). See papers for more details.
    
    ARGUMENTS:
    
    coor (array):
        array of xyz positions for each atom in the molecule 
    atom_sum (int): 
        number of atoms in the molecule
    s_exp (array):
        1D array of s values for the experimental data
    dI (array):
        2D array of dI for each time step in the experiment
    
    OPTIONAL ARGUMENTS:
    
    freq_filter (boolean):
        Default set to False. When true, applies a bandpass filter to the sM to eliminate high frequency noise.
    polyfit (boolean):
        default set to False. When true, applies a polynomial fit to the sM data and subtracts off the background using the poly_fit function
    degree (int):
        Default is set to 2. Degree of the polynomial function used to fit the data
    gauss_filter (boolean):
        Default is set to False. When true, applies a gaussian filter to the sM
    sigma (float):
        Default set to 1.0. Width of the gaussian filter.
    powerfit (boolean):
        Default set to False. When true, fits the sM to a power function of a*x**b and subtracts the baseline.
    plot (boolean):
        default set to False. When true, plots a 2d image (without axes) of the dPDF
    
    RETURNS:

    sM (2D array):
        Calculated delta sM of the input dI
    pdf_exp (2D array):
        Calculated delta pair distribution function
    r (1D array):
        radial distances associated with the PDF
    """

    I_at_all = []
    s_angstrom = S_THEORY*1e-10
    ds= s_exp[1]-s_exp[0]  # step size of s 

    for i in range(atom_sum):
        I_atomic = []
        I_at = 0
        amps = FORM_FACTORS[int(coor[i,4])]
        #print(amps)
        interp_amps = interp.interp1d(s_angstrom[0:125], amps[0:125])
        amps_new = interp_amps(s_exp)
        for k in range(len(amps_new)):
            f_new = amps_new[k]
            I_at = np.abs(f_new)**2
            I_atomic.append(float(I_at))
        I_at_all.append(I_atomic)
    I_at = sum(np.array(I_at_all))
    
    sM = s_exp*(dI/I_at)

    if freq_filter == True:
        sM = bandpass_filter(sM, ds)
    if polyfit == True:
        sM = poly_fit(sM, s_exp, degree=degree, plot=True)
    if gauss_filter == True:
        sM = gaussian_filter(sM, sigma=sigma)
    if powerfit == True:
        sM = power_fit(sM, s_exp, plot=True)
    
    sM_new = []   
    # add numbers for nan values           
    for i in range(len(dI)):
        nan_num = sum(np.isnan(dI[i]))+5 # added 5 to pass over low s noise
        sM_temp = sM[i]
        temp_mean = np.nanmean(sM_temp[nan_num:nan_num+5])
        slope = temp_mean/nan_num
        sM_temp[0:nan_num] = np.arange(0,nan_num)*slope
        sM_new.append(sM_temp)
    sM = np.array(sM_new)/np.nanmax(np.array(sM_new))  # Normalized?

    print(f"s calibration value is {ds}")
    rmax=10; # in angstroms
    r=np.linspace(0,rmax,np.round(500))
    damp_const = np.log(0.01)/(-1 * (np.max(s_exp)**2))
    print(f"1/alpha value for damping constant is {1/damp_const}")

    pdf_exp = []
    for i in range(len(sM)):
        sM_new=sM[i]*np.exp(-ds*(s_exp**2))
        #print(len(sM_new))
        fr_temp = []
        for j in range(len(r)):
            fr=np.nansum(sM_new*np.sin(r[j]*s_exp))*ds*np.exp(-1 * s_exp[i]**2 * damp_const)
            fr_temp.append(fr)
        fr_temp = np.array(fr_temp)
        #print(len(fr_temp))
        pdf_exp.append(fr_temp)
        #     % Calculating fourier transform of theory
    
    pdf_exp = np.array(pdf_exp)

    if plot == True:
        plt.figure()
        plt.pcolor(pdf_exp)
        plt.colorbar()
        plt.show()


    return sM, pdf_exp, r   


def save_data(file_name, group_name, run_number, data_dict, group_note=None):
    """
    Saves the azimuthal average and stage positions after processing to an h5 file with the specified file_name. The group name specifies the 
    group subset the data relates to and the run number tags the number. For example, when running large data sets, each run will be a subset
    of data that was processed. If you have multiple experiments that can be grouped, you can save them with different group names to the same 
    h5 file. The saved data is used for further analysis. 

    ARGUMENTS:

    file_name (str):
        unique file name for the data to be saved. Can specify a full path. 
    group_name (str):
        label for the group of data that is being processed
    run_number (int):
        specifies ths subset of data being processed
    data_dict (dictionary):
        dictionary containing variable labels and data sets to be saved. Can contain any number of data_sets
        i.e., data_dict = {'I' : norm_data, 'stage_positions' : stage_positions, 'centers' : centers}

    OPTIONAL ARGUMENTS:

    group_note (str):
        Note to attach to each group to explain any relevant details about the data processing (i.e., Used average center)
    
    RETURNS:

    Doesn't return anything but creates an h5 file with the stored data or appends already existing file.
    """

    with h5py.File(file_name, 'a') as f:
        # Create or access the group
        if group_name in f:
            group = f[group_name]
        else:
            group = f.create_group(group_name)
        
        # Add a description of the data (if provided)
        if group_note:
            group.attrs['note'] = group_note
        
        for dataset_name, data in data_dict.items():
            # Append run number to the dataset name
            run_dataset_name = f"{dataset_name}_run_{run_number}"
            
            # Create or overwrite the dataset within the group
            if run_dataset_name in group:
                del group[run_dataset_name]
            group.create_dataset(run_dataset_name, data=data)

        f.close()         
    print(f"Data for run {run_number} saved to group '{group_name}' in {file_name} successfully.")
    return


def add_to_h5(file_name, group_name, var_data_dict, run_number=None):
    """
    Appends multiple datasets to a specified group in an h5 file with a specific run number.
    
    ARGUMENTS:
    
    file_name (str):
        Name and path to h5 file you wish to append data to.
    group_name (str):
        Subgroup within the h5 dataset that you wish to append data to.
    var_data_dict (dict):
        Dictionary where keys are variable names and values are arrays of data to add to the h5 file.
    run_number (int):
        Run number to specify which run the data belongs to.
    """

    # Open the HDF5 file in append mode
    with h5py.File(file_name, 'a') as f:
        # Check if the group exists, create if not
        if group_name in f:
            group = f[group_name]
        else:
            group = f.create_group(group_name)

        if run_number == None:
            for var_name, var_data in var_data_dict.items():
                group.create_dataset(var_name, data=var_data)
                print(f"Varriable '{var_name}' added to group '{group_name}' successfully.")
            f.close()
            return
        else:
            for var_name, var_data in var_data_dict.items():
                # Create the run-specific variable name
                run_var_name = f"{var_name}_run_{run_number}"
                
                # Delete the existing dataset if it exists
                if run_var_name in group:
                    print(f"Warning: Dataset '{run_var_name}' already exists in group '{group_name}'. It will be overwritten.")
                    del group[run_var_name]
                
                # Create the dataset within the group
                group.create_dataset(run_var_name, data=var_data)
                print(f"Variable '{run_var_name}' added to group '{group_name}' successfully.")
            f.close()
            return


def load_trajectory_h5(file_name, group_name):
    """
    Reads an HDF5 file and groups datasets by their numeric run ID (e.g., '0014').
    
    ARGUMENTS:
    
    file_name (str):
        Name and path to the HDF5 file to read from.
    group_name (str):
        The name of the group to search in the HDF5 file.
    
    RETURNS:
    
    A dictionary where the keys are run IDs (e.g., '0014') and the values 
    are dictionaries with dataset names as keys and the dataset values as NumPy arrays.
    """
    run_dict = defaultdict(dict)  # Dictionary to group datasets by run ID
    
    # Open the HDF5 file
    with h5py.File(file_name, 'r') as f:
        if group_name not in f:
            print(f"Group '{group_name}' not found in the file.")
            return run_dict
        
        group = f[group_name]
        
        # Regex pattern to extract the numeric run ID (e.g., '0014')
        pattern = re.compile(r'run_(\d{4})')
        
        # Loop through the datasets in the group
        for dataset_name in group.keys():
            match = pattern.search(dataset_name)
            if match:
                run_id = match.group(1)  # Extract the numeric run ID (e.g., '0014')
                variable_name = dataset_name.split(f'_run_{run_id}')[0]  # Extract the variable name
                run_dict[run_id][variable_name] = group[dataset_name][:]
            else:
                print(f"Warning: Could not extract run ID from '{dataset_name}'")
    
    return dict(run_dict)  # Convert defaultdict to regular dict


def _print_h5_structure(group_name, run_number):
    if isinstance(run_number, h5py.Group):
        print(f"Group: {group_name}")
    elif isinstance(run_number, h5py.Dataset):
        print(f"Dataset: {group_name}")


def inspect_h5(file_name):
    """ Inspects and prints structure of the h5 file of interest"""
    with h5py.File(file_name, 'r') as f:
        f.visititems(_print_h5_structure)
        f.close()

# written by SLAC People
def remove_nan_from_data(s_exp,I_exp):
    #this function is to cut off nans in the experimental data
    start=0 #the parameter start reveals the end of nans
    for i in range(len(I_exp)):
        if np.isnan(I_exp[i]) or I_exp[i]==0:
            I_exp[i]=0
            start+=1
        else:
            break

    I_exp=I_exp/I_exp.max()
    #normalize experimental data
    #if start==0:
     #   I_exp1=I_exp
      #  s1=s_exp
    #else:
    if start<20:
        start=20
    I_exp1=I_exp[start:]
    s1=s_exp[start:]

    return I_exp1,s1,start


def get_2d_matrix(x, y):
    # an easy way to set whatever matrix you want
    d = []
    for i in range(x):
        d.append([])
        for j in range(y):
            d[i].append(0)
    return d


def get_3d_matrix(x, y, z):
    # an easy way to set whatever matrix you want
    matrix3d = []
    for i in range(x):
        matrix3d.append([])
        for j in range(y):
            matrix3d[i].append([])
            for k in range(z):
                matrix3d[i][j].append(0)
    return matrix3d


def low_freq_filter(cutoff_freq,s_interval,data):
    fs=1/s_interval
    nyq=0.5*fs
    low=cutoff_freq/nyq
    b,a=signal.butter(5,low,btype='low',analog=False)
    filted_data = signal.filtfilt(b,a,data)
    return filted_data


def plot_delay_matrix(M,norm='',title=''):
    M=np.array(M)
    M=Image.fromarray(M)
    target_size=(200,150)
    new_image=M.resize(target_size)
    im=np.array(new_image)
    
    plt.figure(figsize=(12,8))
    pc=plt.imshow(im,norm=norm,cmap=plt.get_cmap('seismic'),alpha=0.65)
    plt.colorbar(pc)
    ax=plt.gca()
    plt.ylabel('delay',fontsize=20)
    plt.title(title,fontsize=20)
    plt.grid(axis='x',color='indigo',linestyle='--',linewidth=2)
    plt.grid(axis='y',color='olive',linestyle=':',linewidth=1.5)
    return


def find_zeros(I_mol,s):
    #this function is to find the s0 values at which the simulated I_mol equals 0
    zero_count=0
    zeros_max=30 #the total number of zero points should be no larger than this value
    s01=[0 for i in range(zeros_max)]
    for i in range(len(s)-1):
        if I_mol[i]*I_mol[i+1]<=0:
            #s01[zero_count]=s[i]
            s01[zero_count]=(I_mol[i]*s[i+1]-I_mol[i+1]*s[i])/(I_mol[i]-I_mol[i+1])
            zero_count+=1

    s0=s01[0:zero_count]

    return s0


def find_near_zeros(sM,s):
    #it is better to determine near zeros using sM instead of I_mol
    #because I_mol approaches zero at large angles
    zero_count=0
    zeros_max=50 
    s01=[0 for i in range(zeros_max)]
    for i in range(len(s)-1):
        if sM[i]*sM[i+1]<=0 and s[i]>1e-3 and s[i]-s01[zero_count-1]>0.1:
            s01[zero_count]=(sM[i]*s[i+1]-sM[i+1]*s[i])/(sM[i]-sM[i+1])
            zero_count+=1
        if sM[i]*sM[i+1]>0 and abs(sM[i]/sM.max())<0.000001 and s[i]-s01[zero_count-1]>0.1: 
            # abs(sM[i]/sM.max())<0.001 decides whether the point satisfies 'near zero'
            # set a number much smaller than 0.001 if three zeros are too close to each other
            # s[i]-s01[zero_count-1]>0.1 zeros too close to another are discarded
            s01[zero_count]=s[i]
            zero_count+=1

    s0=s01[0:zero_count]
    return s0


def get_I_exp_at_zeros(s_exp,I_exp,s0):
    m=0
    filted_I_exp=low_freq_filter(1.5,s_exp[1]-s_exp[0],I_exp)
    I_0=[0 for i in range(len(s0))]
    for i in range(len(s_exp)-1):
        if abs(s_exp[i]-s0[m])<3e-4:
            I_0[m]=filted_I_exp[i]
            m+=1
            if m==len(s0):
                break
        elif abs(s_exp[i+1]-s0[m])<3e-4:
            I_0[m]=filted_I_exp[i]
            m+=1
            if m==len(s0):
                break
        elif s_exp[i]<s0[m] and s_exp[i+1]>s0[m]:
            I_0[m]=(filted_I_exp[i+1]-filted_I_exp[i])/(s_exp[i+1]-s_exp[i])*(s0[m]-s_exp[i+1])+filted_I_exp[i+1]
            m+=1
            if m==len(s0):
                break
    return I_0


def fit_every_3_points(s0,I_0,s_exp):
    A=np.empty(len(s0)-2)
    B=np.empty(len(s0)-2)
    C=np.empty(len(s0)-2)
    I_frag=np.empty((len(s0)-2,len(s_exp)))
    for i in range(len(s0)-2):
        s1=s0[i]
        s2=s0[i+1]
        s3=s0[i+2]
        g1=np.log(I_0[i])
        g2=np.log(I_0[i+1])
        g3=np.log(I_0[i+2])
    
        alpha1=(g2-g1)/(g3-g1)
        alpha2=(g3-g1)/(g3-g2)
        c=np.linspace(-3,3,800)
        h=alpha1*(s3**c-s1**c)-s2**c+s1**c+alpha2*(s3**c-s2**c)-s3**c+s1**c
        
        c0=find_zeros(h,c)
        c_zero = 0.005 #temp for testing
        for j in c0:
            if abs(j)>0.001:
                c_zero=j

        b=((g2-g1)/(s2**c_zero-s1**c_zero)+(g3-g1)/(s3**c_zero-s1**c_zero))/2
        a=(g1-b*s1**c_zero+g2-b*s2**c_zero+g3-b*s3**c_zero)/3
        A[i]=a
        B[i]=b
        C[i]=c_zero
        I_frag[i]=np.exp(a+b*s_exp**c_zero)
        
    I_fit=[0 for i in range(len(s_exp))]
    m=0
    m_max=len(s0)-1
    for i in range(len(s_exp)):
        if s_exp[i]>=s0[m] and m<m_max:
            m+=1
        if m<=1:
            I_fit[i]=I_frag[0,i]
        if m>=len(s0)-1:
            I_fit[i]=I_frag[m_max-2,i]
        if m>1 and m<len(s0)-1:
            I_fit[i]=I_frag[m-2,i]*(s_exp[i]-s0[m-1])/(s0[m]-s0[m-1])+I_frag[m-1,i]*(s0[m]-s_exp[i])/(s0[m]-s0[m-1])

    return I_fit


def fit_background(s0,I_0,s1):
    def func(x,a,b,c,d):
        return c*np.exp(-a*x**b)+d
    xdata=np.array(s0)
    ydata=np.array(I_0)
    popt,pcov=curve_fit(func,xdata,ydata,p0=[1.3,0.7,1,1],maxfev = 10000,bounds=(0,[3,3,10,10]))
    a=popt[0]
    b=popt[1]
    c=popt[2]
    d=popt[3]
    I_fit=c*np.exp(-a*s1**b)+d
    return I_fit


def rescale_along_y_axis(fstandard,ftoscale):
    fstandard=np.array(fstandard)
    ftoscale=np.array(ftoscale)
    if len(fstandard)!=len(ftoscale):
        print('error! fstandard and ftoscale should have the same length!')
        return
    scale=sum(fstandard**2)/sum(fstandard*ftoscale)
    return ftoscale*scale,scale


def retrieve_PDF(left,right,s_interval1,s_max,coor,atom_sum,damp_const,r_max,I_exp):
    s_exp=np.linspace(0,(len(I_exp)-1)*s_interval1,len(I_exp))
    I,I_at,I_mol, _=get_I_from_xyz(coor,atom_sum, s_val=s_exp)
    I_exp1,s1,start=remove_nan_from_data(s_exp,I_exp)
    I_at1=I_at[start:]
    sM,PDF,r=get_sM_and_PDF_from_I(I_at,I_mol,s_exp,r_max,damp_const)
    s0=find_near_zeros(sM,s_exp)
    print('zeros:')
    print(s0)
    
    for i in range(len(s_exp)):
        if s_exp[i]>=s0[0]:
            start1=i
            break
    cut=start1-start
    I_0=get_I_exp_at_zeros(s1,I_exp1,s0)
    #I_fit=fit_background(s0,I_0,s1)
    I_fit=fit_every_3_points(s0,I_0,s1)

    I_mol_ret=I_exp1-I_fit
    I_mol_ret_filted=low_freq_filter(1.5,s_interval1,I_mol_ret)#filt high frequency noices
    sM_ret,PDF_awful,r=get_sM_and_PDF_from_I(I_at1,I_mol_ret_filted,s1,r_max,damp_const)
    sM_ret=sM_ret/sM_ret.max()
    
    I1,I_at1,I_mol1,s11=get_I_from_xyz(coor,atom_sum, s_val=s_max)
    sM1,PDF1,r=get_sM_and_PDF_from_I(I_at1,I_mol1,s11,r_max,damp_const)

    sM_combined=np.empty(len(s_exp))
    sM_combined[start1:]=sM_ret[cut:]
    
    j=0
    for i in range(len(s_exp)):
        if s_exp[i]>=left and j==0:
            st=i
            j=1
        if s_exp[i]>=right:
            en=i
            break

    st1=st-start
    en1=en-start
    #sM_ret1,sc1=rescale_along_y_axis(sM[st:en],sM_ret[st1:en1])
    #sMscale=1/sc1
    sMscale=1/(sum(abs(sM[st:en]))/sum(abs(sM_ret[st1:en1])))
    
    sM_combined[:start1]=sM[:start1]*sMscale

    PDF_ret=[0 for i in range(r_max)]
    for i in range(len(s_exp)-1): 
        PDF_ret+=sM_combined[i]*np.sin(s_exp[i]*1e10*np.array(r)*1e-12)*(s_exp[i+1]-s_exp[i])*np.exp(-s_exp[i]**2/damp_const)
    
    PDF_ret_scaled,sc3=rescale_along_y_axis(PDF1,PDF_ret)
    fig=plt.figure(figsize=(8,4))
    # fig.set_size_inches(10,5)
    plt.subplot(121)
    plt.plot(s1,I_exp1,label='experimental data')
    plt.scatter(s0,I_0)
    plt.plot(s1,I_fit,label='fitted background')
    plt.plot(s1,I_mol_ret,label='retrieved I_mol')
    plt.grid()
    plt.legend(loc="best")
    plt.xlabel('s/angs^-1')
    plt.title('fit scattering background')

    plt.subplot(122)
    plt.plot(s_exp,sM*sMscale,label='simulation',linestyle='--',linewidth='3')
    plt.plot(s1,sM_ret,label='experiment',linewidth='2')
    plt.axvline(x=s_exp[st],linestyle='--',color='firebrick',label='rescale begin')
    plt.axvline(x=s_exp[en],linestyle='--',color='firebrick',label='rescale end')
    plt.title('retrieved sM')
    plt.grid()
    plt.xlabel('s/angs^-1')
    #plt.legend(loc="best")
    plt.show()
    
    fig=plt.figure(figsize=(8,4))
    # fig.set_size_inches(10,5)
    plt.subplot(121)
    plt.plot(s11,sM1*np.exp(-s11**2/damp_const),label='simulation',linestyle='--',linewidth='3')
    plt.plot(s_exp,sM_combined/(sum(abs(sM_combined[st:en]))/sum(abs(sM1[st:en])))*np.exp(-s_exp**2/damp_const),label='experiment',linewidth='2')
    plt.legend()
    plt.title('damped sM')
    plt.axvline(x=s_exp[start1],linestyle='--',color='black',label='cut')
    plt.axvline(x=s_exp[st],linestyle='--',color='firebrick')
    plt.axvline(x=s_exp[en],linestyle='--',color='firebrick')
    plt.legend(loc="best")
    plt.grid()
    plt.xlabel('s/angs^-1')
    
    plt.subplot(122)
    plt.plot(r,PDF1,label='simulation',linestyle='--',linewidth='3')
    plt.plot(r,PDF_ret_scaled,label='experiment',linewidth='2.5')
    plt.title('retrieved ground state PDF')
    plt.legend(loc="best")
    plt.grid()
    plt.xlabel('r/pm')
    plt.tight_layout()
    plt.show()
    
    return


def sM_err(left,right,s_interval1,s_max,coor,atom_sum,damp_const,r_max,I_exp):
    s_exp = np.linspace(0,(len(I_exp)-1)*s_interval1,len(I_exp))
    I,I_at,I_mol, _ = get_I_from_xyz(coor,atom_sum, s_val = s_exp)
    I_exp1,s1,start = remove_nan_from_data(s_exp,I_exp)
    I_at1=I_at[start:]
    sM,PDF,r=get_sM_and_PDF_from_I(I_at,I_mol,s_exp,r_max,damp_const)
    s0=find_near_zeros(sM,s_exp)
    
    for i in range(len(s_exp)):
        if s_exp[i]>=s0[0]:
            start1=i
            break
    cut=start1-start
    I_0=get_I_exp_at_zeros(s1,I_exp1,s0)
    #I_fit=fit_background(s0,I_0,s1)
    I_fit=fit_every_3_points(s0,I_0,s1)

    I_mol_ret=I_exp1-I_fit
    I_mol_ret_filted=low_freq_filter(1.5,s_interval1,I_mol_ret)#filt high frequency noices
    sM_ret=s1*I_mol_ret_filted/I_at1

    j=0
    for i in range(len(s_exp)):
        if s_exp[i]>=left and j==0:
            st=i
            j=1
        if s_exp[i]>=right:
            en=i
            break
    st1=st-start
    en1=en-start

    s_good=s_exp[st:en]
    #sMscale=1/(sum(abs(sM[st:en]))/sum(abs(sM_ret[st1:en1])))
    
    sM_ret1,sc1=rescale_along_y_axis(sM[st:en],sM_ret[st1:en1])

    r=scipy.stats.mstats.pearsonr(sM_ret1*np.exp(-s_good**2/damp_const),sM[st:en]*np.exp(-s_good**2/damp_const))
    #r=mstats.pearsonr(sM_ret[st1:en1],sM_sim*sMscale)
    
    return r[0]


def scan_s_calibration(scmin,scmax,left,right,s_max,coor,atom_sum,damp_const,r_max,I_ground_state):
    #uses sM_err function
    interval1=0.0002
    interval2=0.00002
    R1=np.empty(int((scmax-scmin)/interval1))
    R2=np.empty(int(2*interval1/interval2))
    for i in range(int((scmax-scmin)/interval1)):
        R1[i]=sM_err(left,right,scmin+i*interval1,s_max,coor,atom_sum,damp_const,r_max,I_ground_state)

    a=np.where(R1==R1.max())
    s_calibration1=scmin+interval1*int(a[0])

    for i in range(int(2*interval1/interval2)):
        R2[i]=sM_err(left,right,s_calibration1-interval1+i*interval2,s_max,coor,atom_sum,damp_const,r_max,I_ground_state)

    x=np.linspace(-2,2,5)
    h=np.exp(-x**2/200)
    c=signal.convolve(R2,h,mode='same')
    c=c[2:-2]

    b=np.where(c==c.max())
    s_calibration2=s_calibration1-interval1+interval2*(int(b[0])+2)
    print('s_calibration:',s_calibration2,'angs^-1/pixel')
    #print('pearson_r:',R2[int(b[0])+2])
    return s_calibration2
