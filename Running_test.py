import glob
import time

import GUED as gued
import numpy as np

import matplotlib.pyplot as plt


# Load in Data
dataPath = 'C:\\Users\\laure\\OneDrive - University of Nebraska-Lincoln\\Documents\\Centurion Lab\\nitrophenyl code\\20180823'
runPath = "\\*\\*\\ANDOR1_*.tif"

bkg_path = '\\work\\centurion\\shared\\UED_data\\FY18_o-nitrophenol\\20180823\\Background\\*\\*\\ANDOR1_*.tif'

newPath = dataPath + runPath
print("Path to images ", newPath)
print("Path to background images ", bkg_path)

files = glob.glob(newPath)
bkg_files = glob.glob(bkg_path)
print("Number of files loaded: ", len(files))
print("Number of background files loaded: ", len(bkg_files))


print('Load diffraction signal');
data_array, stage_positions, file_order, counts = gued.get_image_details(files[:], sort=True, plot=False, filter_data=False)

counts_mean = np.mean(counts)        # Mean values of Total Counts of all images
counts_std  = np.std(counts)         # the STD of all the tc for all the iamges
uni_stage = np.unique(stage_positions)# Pump-probe stage position


print('Image number read: ', len(counts))
print('Stage positions: ', len(uni_stage))
print(len(np.unique(file_order)))

# Remove Outlier Images Based on Total Counts
data_array, stage_positions, file_order, counts = gued.remove_counts(data_array, stage_positions, file_order, counts)

#Find Centers

start = time.perf_counter()
center_x, center_y = gued.find_center_pool(data_array, plot=False)
stop = time.perf_counter()

print(f"Found centers for {len(data_array): .2f} images in {(stop-start): .2f} seconds using thread pool")


# Remove Background based on Corners

start = time.perf_counter()
data_array = gued.remove_background(data_array, plot=False)
stop = time.perf_counter()

print(f"Removed background from {len(data_array): .2f} images in {(stop - start): .2f} seconds.")


# Removing Hot Pixels

start = time.perf_counter()
data_array = gued.remove_xrays_pool(data_array, plot=False)
stop = time.perf_counter()

print(f"Removed hot pixels from {len(data_array): .2f} images in {(stop - start): .2f} seconds.")

# Mask Beam Block and Artifacts

start = time.perf_counter()

# Find Mask
mean_data = np.nanmean(data_array, axis=0)
mask_center = [475,475]
mask_radius = 45

# Apply Mask
data_array = gued.apply_mask(data_array, mask_center, mask_radius, add_mask=[[440, 435, 30]], plot=False)

stop = time.perf_counter()
print(f"Masked data in {stop-start} seconds")



# Remove Radial Outliers and Apply Median Filter
ave_cx = np.nanmean(center_x)
ave_cy = np.nanmean(center_y)
center = [ave_cx, ave_cy]
center = [500, 500]

start = time.perf_counter()
# for i in range(0,10):
#     temp = gued.remove_radial_outliers(data_array[i], center)
#     print(i, " : completed")

test = gued.remove_radial_outliers(data_array[0], center, plot=True, fill_value='ave')


stop = time.perf_counter()

print(f"Radial outliers for {len(data_array)} images took {(stop-start):2f} seconds.")