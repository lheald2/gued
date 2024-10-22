# Standard Packages
import numpy as np
import glob
import matplotlib.pyplot as plt
from datetime import date
import time
from scipy.signal import savgol_filter

# new code 
import gued

if __name__ == "__main__":
    # path to raw data
    data_path = 'C:\\Users\\laure\\OneDrive - University of Nebraska-Lincoln\\Documents\\Centurion Lab\\UV_IR_exp\\07222024\\Time_Scans\\Scan 9\\'
    run_path = "\\Image_*.tiff"

    full_path = data_path + run_path
    files = glob.glob(full_path) 
    print(f"{len(files)} found in {full_path} folders")
    
    print('Loading diffraction signal')
    all_data, all_stages, all_orders, all_counts, _ = gued.get_image_details_keV(files, sort=True, multistage=True, 
                                                                                 filter_data=False, plot=False)
    
    # details for saving processed data
    exp_label = "cf3i_ir"
    today = date.today()
    print(today)

    file_path = 'C:\\Users\\laure\\OneDrive - University of Nebraska-Lincoln\\Documents\\Centurion Lab\\UV_IR_exp\\07222024\\Time_Scans\\'
    file_name = file_path + f"{exp_label}_{today}.h5"
    print(f"writing data to {file_name}")
    group_name = "scan_9"
    power = 0.5
    #group_name = "s4"
    save_factor = 0

    group_size = 300
    groups = np.arange(0, len(all_data)+1, group_size)

    # Load masking information
    mask_path = "C:\\Users\\laure\\OneDrive - University of Nebraska-Lincoln\\Documents\\Centurion Lab\\UV_IR_exp\\"

    import scipy.io # needed to do it this way because of an outdated version of matlab and how it creates .mat files

    mask_d = scipy.io.loadmat((mask_path + "mask_d.mat"))
    #print(mask_d.keys())
    mask_exp = mask_d['mask_d']
    mask_sign = 'yes'
    #print("mask_exp has a shape of ", mask_exp.shape)

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

        #print(f"Filtering based on counts for files {groups[i]} to {groups[i+1]}")
        data_array, stage_positions, file_order, counts = gued.remove_counts(data_array, stage_positions, file_numbers, 
                                                                             counts, added_range=[], plot=False)

        #print(f"Applying detector mask for files {groups[i]} to {groups[i+1]}")
        if mask_sign == 'yes':
            data_array = gued.add_detector_mask(data_array, mask_exp, fill_value=np.nan, plot=False)
        
        #print(f"Removing background for files {groups[i]} to {groups[i+1]}")
        data_array = gued.remove_background_pool(data_array, remove_noise=True, plot=False)

        #print(f"Removing xrays for files {groups[i]} to {groups[i+1]}")
        data_array, pct_xrays = gued.remove_xrays_pool(data_array, plot=False, return_pct=True)

        #print(f"Masking Data with 0.0 value for files {groups[i]} to {groups[i+1]}")
        if mask_sign=='yes':
            data_array = gued.add_detector_mask(data_array, mask_exp, fill_value=0.0, plot=False) # when using a detector mask
        data_array = gued.apply_mask(data_array, fill_value=0.0, add_rectangular=True, plot=False)

        print(f"Finding center for files {groups[i]} to {groups[i+1]}")
        center_x, center_y = gued.find_center_pool(data_array, plot=False)
        centers = list(zip(center_x, center_y))
        average_center = np.mean(centers, axis=0)
        #average_center = [455, 460]
        print(f"Average center of data set {i} is ({average_center[0]:.2f}, {average_center[1]:.2f})")

        #print(f"Remasking Data with np.nan value for files {groups[i]} to {groups[i+1]}")
        if mask_sign=='yes':
            data_array = gued.add_detector_mask(data_array, mask_exp, fill_value=np.nan, plot=False)
        data_array = gued.apply_mask(data_array, fill_value=np.nan, add_rectangular=True, plot=False)

        #print(f"Median Filtering for files {groups[i]} to {groups[i+1]}")
        #clean_data = gued.median_filter_pool(data_array, plot=False)

        # data_array = clean_data
        # del clean_data

        print(f"Saving Data for files {groups[i]} to {groups[i+1]} as run number {i}")
        # Make dictionary for saving to h5 file

        data_dictionary = {
            'clean_images': data_array,
            'stage_positions': stage_positions, 
            'centers': centers, 
            'total_counts': counts, 
            'percent_xrays': pct_xrays}
        
        data_note = f"Power Scan: {str(power)} W"
        gued.save_data(file_name, group_name, (i+save_factor), data_dictionary, group_note=data_note)
        del data_array, stage_positions, file_numbers, counts, counts_mean, counts_std

    stop = time.perf_counter()

    print(f"Finished pre-processing {int(groups[-1])-(groups[0])} in {(stop-start)/60} minutes")
    del all_counts, all_data, all_orders, all_stages

    variable_names = ["centers", "clean_images"]
    run_numbers = list(np.arange(save_factor,(save_factor+len(groups)),1))
    
    combined_data = gued.read_combined_data(file_name, group_name, variable_names, run_numbers=run_numbers)
    centers_x = combined_data['centers'][:,0]
    centers_y = combined_data['centers'][:,1]

    print(f"Smoothing center values using Savitzky-Golay filter")
    # Apply Savitzky-Golay filter
    window_size = 101  # Choose an odd number for the window size
    poly_order = 1  # Choose the polynomial order

    if group_name == "s1":
        smoothed_x = np.concatenate((savgol_filter(centers_x[:1440], window_size, poly_order),
                                    savgol_filter(centers_x[1441:1505], 35, poly_order), 
                                    savgol_filter(centers_x[1505:], window_size, poly_order)))

        smoothed_y = np.concatenate((savgol_filter(centers_y[:990], window_size, poly_order),
                                    savgol_filter(centers_y[991:1627], window_size, poly_order), 
                                    savgol_filter(centers_y[1627:], window_size, poly_order)))
        smoothed_centers = list(zip(smoothed_x, smoothed_y))
    else:
        smoothed_x = savgol_filter(centers_x, window_size, poly_order)
        smoothed_y = savgol_filter(centers_y, window_size, poly_order)
        smoothed_centers = list(zip(smoothed_x, smoothed_y))

    for i in range(len(groups)-1):
        data_array = combined_data["clean_images"][groups[i]:groups[i+1]]
        centers = smoothed_centers[groups[i]:groups[i+1]]

        print(f"Removing Radial Outliers for files {groups[i]} to {groups[i+1]}")
        cleaned_data, pct_outliers = gued.remove_radial_outliers_pool(data_array, centers, plot=False, return_pct=True)

        data_array = cleaned_data 
        del cleaned_data

        print(f"Calculating the Azimuthal average and Normalizing for files {groups[i]} to {groups[i+1]}")
        norm_data, norm_std = gued.get_azimuthal_average_pool(data_array, centers, normalize=True, plot=False)

        data_dictionary = {"clean_images": data_array,
                           "percent_outliers": pct_outliers, 
                           "new_centers": centers,
                           "I": norm_data}
        
        gued.add_to_h5(file_name, group_name, data_dictionary, run_number=(i+save_factor))
        del norm_data, norm_std, data_array, centers, pct_outliers
        print(f"Done post processing files {groups[i]} to {groups[i+1]}")
    
    stop2 = time.perf_counter()
    print(f"Finished all processing {int(groups[-1])-(groups[0])} in {(stop2-start)/60} minutes")