# Standard Packages
import numpy as np
import glob
import matplotlib.pyplot as plt
from datetime import date
import time

# new code 
import gued

if __name__ == "__main__":
    data_path = 'C:\\Users\\laure\\OneDrive - University of Nebraska-Lincoln\\Documents\\Centurion Lab\\nitrophenyl code\\20180823\\Run\\*\\'
    #data_path = 'C:\\Users\\laure\\OneDrive - University of Nebraska-Lincoln\\Documents\\Centurion Lab\\nitrophenyl code\\20180623\\Run\\'
    run_path = "*\\*\\ANDOR1_*.tif"
    
    #bkg_path = '/work/centurion/shared/UED_data/FY18_o-nitrophenol/20180823/Background/*/*/ANDOR1_*.tif'


    full_path = data_path + run_path
    files = glob.glob(full_path) 

    #bkg_files = glob.glob(bkg_path)
    print(f"{len(files)} found in {full_path} folders")

    print('Loading diffraction signal');
    all_data, all_stages, all_orders, all_counts = gued.get_image_details(files, sort=True, filter_data=False, plot=False)

    exp_label = "o-ntph_data"
    today = date.today()
    print(today)

    file_name = f"{exp_label}_{today}.h5"
    print(f"writing data to {file_name}")
    group_name = "s1"
    #group_name = "s4"

    group_size = 200
    groups = np.arange(0, len(all_data)+1, group_size)

    start = time.perf_counter()

    for i in range(len(groups)-1):
        print(f"Started processing files {groups[i]} to {groups[i+1]}")

        data_array = all_data[groups[i]:groups[i+1]]
        stage_positions = all_stages[groups[i]:groups[i+1]]
        file_numbers = all_orders[groups[i]:groups[i+1]]
        counts = all_counts[groups[i]:groups[i+1]]

        counts_mean = np.mean(counts)        # Mean values of Total Counts of all images
        counts_std  = np.std(counts)         # the STD of all the tc for all the iamges
        uni_stage = np.unique(stage_positions)# Pump-probe stage position

        print(f"Filtering based on counts for files {groups[i]} to {groups[i+1]}")
        data_array, stage_positions, file_order, counts = gued.remove_counts(data_array, stage_positions, file_numbers, 
                                                                             counts, added_range=[], plot=False)

        print(f"Removing background for files {groups[i]} to {groups[i+1]}")
        data_array = gued.remove_background_pool(data_array, remove_noise=True, plot=False)

        print(f"Removing xrays for files {groups[i]} to {groups[i+1]}")
        data_array, pct_xrays = gued.remove_xrays_pool(data_array, plot=False, return_pct=True)

        print(f"Masking Data with 0.0 value for files {groups[i]} to {groups[i+1]}")
        data_array = gued.apply_mask(data_array, fill_value=0.0, plot=False)

        print(f"Finding center for files {groups[i]} to {groups[i+1]}")
        center_x, center_y = gued.find_center_pool(data_array, plot=False)
        centers = list(zip(center_x, center_y))
        average_center = np.mean(centers, axis=0)
        #average_center = [455, 460]
        print(f"Average center of data set {i} is ({average_center[0]:.2f}, {average_center[1]:.2f})")

        print(f"Remasking Data with np.nan value for files {groups[i]} to {groups[i+1]}")
        data_array = gued.apply_mask(data_array, fill_value=np.nan, plot=False)

        print(f"Removing Radial Outliers for files {groups[i]} to {groups[i+1]}")
        
        cleaned_data, pct_outliers = gued.remove_radial_outliers_pool(data_array, centers, plot=False, return_pct=True)

        data_array = cleaned_data 
        del cleaned_data

        print(f"Median Filtering for files {groups[i]} to {groups[i+1]}")
        clean_data = gued.median_filter_pool(data_array, plot=False)

        data_array = clean_data
        del clean_data
        
        print(f"Calculating the Azimuthal average and Normalizing for files {groups[i]} to {groups[i+1]}")
        norm_data, norm_std = gued.get_azimuthal_average_pool(data_array, centers, normalize=True, plot=False)

        print(f"Saving Data for files {groups[i]} to {groups[i+1]} as run number {i}")
        # Make dictionary for saving to h5 file

        data_dictionary = {
            'clean_images': data_array,
            'I': norm_data, 
            'stage_positions': stage_positions, 
            'centers': centers, 
            'total_counts': counts, 
            'percent_xrays': pct_xrays, 
            'percent_outliers': pct_outliers}
        
        data_note = "Using center value for each image"
        gued.save_data(file_name, group_name, i, data_dictionary, group_note=data_note)
        del data_array, stage_positions, file_numbers, counts, counts_mean, counts_std

    stop = time.perf_counter()

    print(f"Finished processing {int(groups[-1])-(groups[0])} in {(stop-start)/60} minutes")