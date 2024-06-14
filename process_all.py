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

def azimuthal_avg_correct(args):
    """Returns the azimuthal average of a diffraction image based on the radial distance of the x, y positions in the image."""
    data, x, y = args
    r_max=len(x)
    I=np.empty(r_max)
    for ri in range(r_max):
        I_r=data[x[ri],y[ri]]
        ave=np.nanmean(I_r)
        sigma=np.nanstd(I_r)
        I_r[np.abs(I_r-ave)>=5*sigma]=np.nan
        I[ri]=np.nanmean(I_r)
    return I

def get_azimuthal_average(data,x,y):
    """Runs the azimuthal average function in parallel for large data sets."""
    p = ThreadPool(3)
    I=p.map(azimuthal_avg_correct, [(data_i,x,y) for data_i in data]) 
    return np.array(I)

if __name__ == '__main__':
    data_path = 'C:\\Users\\laure\\OneDrive - University of Nebraska-Lincoln\\Documents\\Centurion Lab\\nitrophenyl code\\20180823\\Run\\'
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
    group_name = "s4"

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
        center = [560, 500]
        cleaned_data = []
        cleaned_data = gued.remove_radial_outliers_pool(masked_data, center, plot=False)

        data_array = cleaned_data 
        del cleaned_data, masked_data

        print(f"Median Filtering for files {data_sets[i][0]} to {data_sets[i][1]}")
        clean_data = gued.median_filter_pool(data_array, plot=False)

        data_array = clean_data
        del clean_data

        X_CENTER = center[0]
        Y_CENTER = center[1]
        GRID_SIZE = 1024

        pointer = np.empty((1024,1024))
        for j in range(GRID_SIZE):
            for k in range(GRID_SIZE):
                pointer[j,k]=int(np.sqrt((j-X_CENTER)**2+(k-Y_CENTER)**2))
        img_r_max = 460
        x=[];y=[]
        for ri in range(img_r_max):
            [X_temp,Y_temp]=np.where(pointer==ri)
            x.append(X_temp)
            y.append(Y_temp)
        print(r'max Q index is ' + str(img_r_max));
        r_max=len(x)

        #print(f"Getting Azimuthal Average for files {data_sets[i][0]} to {data_sets[i][1]}")
        azi_data = get_azimuthal_average(data_array, x, y)

        print(f"Normalizing Data for files {data_sets[i][0]} to {data_sets[i][1]}")
        min_val = 50
        max_val = 100
        norm_data = gued.normalize_to_baseline(azi_data) 
        del azi_data

        s_cali = 0.026
        #posi_0    = 154.405 # The reference T0
        #posi_0 = 108.61
        posi_0 = 26.9
        s = np.arange(0,len(norm_data[0]))*s_cali # The Q axis

        print(f"Saving Data for files {data_sets[i][0]} to {data_sets[i][1]} as run number {i}")
        gued.save_data(file_name, group_name, norm_data, stage_positions, i)
        del data_array, stage_positions, file_numbers, counts, counts_mean, counts_std

    stop = time.perf_counter()

    print(f"Finished processing {int(data_sets[-1][-1])-(data_sets[0][0])} in {(stop-start)/60} minutes")