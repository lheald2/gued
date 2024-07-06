# Standard Packages
import numpy as np
import glob
import matplotlib.pyplot as plt
from datetime import date
import time

#multiprocessing
from multiprocessing.dummy import Pool as ThreadPool

# new code 
import gued

if __name__ == '__main__':
    #data_path = 'C:\\Users\\laure\\OneDrive - University of Nebraska-Lincoln\\Documents\\Centurion Lab\\nitrophenyl code\\20180823\\Run\\*\\'
    data_path = 'C:\\Users\\laure\\OneDrive - University of Nebraska-Lincoln\\Documents\\Centurion Lab\\nitrophenyl code\\20180623\\Run\\'
    run_path = "*\\*\\ANDOR1_*.tif"


    bkg_path = '/work/centurion/shared/UED_data/FY18_o-nitrophenol/20180823/Background/*/*/ANDOR1_*.tif'


    full_path = data_path + run_path
    print(full_path)
    print(bkg_path)

    files = glob.glob(full_path) 
    bkg_files = glob.glob(bkg_path)
    print(len(files))
    print(len(bkg_files))
    print('Loading diffraction signal');
    all_data, all_stages, all_orders, all_counts = gued.get_image_details(files, sort=True, filter_data=False, plot=False)

    exp_label = "o-ntph_data"
    today = date.today()
    print(today)

    file_name = f"{exp_label}_{today}.h5"
    print(file_name)
    #group_name = "s1"
    group_name = "s4"

    #center = [489, 464]
    center = [571, 494]

    #img_r_max = 460
    img_r_max = 450

    #data_sets = [[0, 225], [275, 400], [400,600], [600, 800], [800, 1000], [1000,1200], [1200,1400], [1400,1600], [1600, 1800], [1800,1990]]
    data_sets = [[0,200], [200, 400], [400,600], [600, 800], [800, 1000], [1000,1200], [1200, 1400], [1400, 1600], [1600, 1800], [1800, 2000]]

    start = time.perf_counter()

    for i in range(len(data_sets)):
        print(f"Started processing files {data_sets[i][0]} to {data_sets[i][1]}")

        data_array = all_data[data_sets[i][0]:data_sets[i][1]]
        stage_positions = all_stages[data_sets[i][0]:data_sets[i][1]]
        file_numbers = all_orders[data_sets[i][0]:data_sets[i][1]]
        counts = all_counts[data_sets[i][0]:data_sets[i][1]]

        counts_mean = np.mean(counts)        # Mean values of Total Counts of all images
        counts_std  = np.std(counts)         # the STD of all the tc for all the iamges
        uni_stage = np.unique(stage_positions)# Pump-probe stage position

        print(f"Filtering based on counts for files {data_sets[i][0]} to {data_sets[i][1]}")
        data_array, stage_positions, file_numbers, counts = gued.remove_counts(data_array, stage_positions, 
                                                                file_numbers, counts, plot=False)
        
        print(f"Removing background for files {data_sets[i][0]} to {data_sets[i][1]}")
        data_array = gued.remove_background_pool(data_array, plot=False)

        print(f"Removing xrays for files {data_sets[i][0]} to {data_sets[i][1]}")
        data_array = gued.remove_xrays_pool(data_array, plot=False)

        print(f"Masking Data for files {data_sets[i][0]} to {data_sets[i][1]}")
        masked_data = gued.apply_mask(data_array, plot=False)

        print(f"Removing Radial Outliers for files {data_sets[i][0]} to {data_sets[i][1]}")

        cleaned_data = []
        cleaned_data = gued.remove_radial_outliers_pool(masked_data, center, plot=False)

        data_array = cleaned_data 
        del cleaned_data, masked_data

        print(f"Median Filtering for files {data_sets[i][0]} to {data_sets[i][1]}")
        clean_data = gued.median_filter_pool(data_array, plot=False)

        data_array = clean_data
        del clean_data
        
        print(f"Calculating the Azimuthal average and Normalizing for files {data_sets[i][0]} to {data_sets[i][1]}")
        norm_data, norm_std = gued.get_azimuthal_average_pool(data_array, center, normalize=True, plot=True)

        print(f"Saving Data for files {data_sets[i][0]} to {data_sets[i][1]} as run number {i}")
        gued.save_data(file_name, group_name, i, norm_data, stage_positions)
        del data_array, stage_positions, file_numbers, counts, counts_mean, counts_std

    stop = time.perf_counter()

    print(f"Finished processing {int(data_sets[-1][-1])-(data_sets[0][0])} in {(stop-start)/60} minutes")