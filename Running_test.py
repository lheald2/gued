import glob
import time

import GUED as gued
import numpy as np
import center_finding as cf
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

# Remove Hot Pixels
data_array, stage_positions, file_order, counts = gued.remove_counts(data_array, stage_positions, file_order, counts)

#Find Centers

start = time.perf_counter()
center_x, center_y = gued.find_center_parallel(data_array)
stop = time.perf_counter()
#
print(f"Found centers for {len(data_array)} images in {(stop-start)} seconds while running in parallel")
#



