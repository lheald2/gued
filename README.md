# GUED_Analysis

Written by Lauren F Heald 
Email: lheald2@unl.edu

## About:
The goal is to create this package to contain all the necessary functions for analyzing time resolved gas phase electron diffraction data (especially written for 
experiments conducted at the MeV-UED facility at the Stanford Linear Accelerator).

### Current Status:
  Able to process all data to get the total scattering and ΔI/I
  Working on getting the pair distribution function from the total scattering. Updates are posted often

## Current Functionality

1. __Import all images__
    * Function called `gued.get_image_details` is used to load in all .tif files in the specified folder of interest. The function returns the images as a 3D data array, a 1D array of stage positions, file order, and total counts per image.

2. __Reject images with bad total counts__
    * Function called `gued.remove_counts` loads in the returns from `gued.get_image_details` and removes any images based on their total counts then returns the inputs with bad images removed.

3. __Subtract background__
    * The function called `gued.remove_background_pool` takes in a 3D array containing all image files and runs the hidden function `_remove_background` which creates a background image based on the corners of the original image. Then can either return the interpolated backgrounds or the cleaned data.
    * In cases where background images are taken as part of the experiment, use the `gued.subtract_background` function with the data array and an average background image.

4. __Remove outlier instances of identical pixels__
    * This is generally referred to as removing x-ray hits or hot pixels. When working with large data sets, use the `gued.remove_xrays_pool` function. This function takes the 3D data array and runs the hidden function `_remove_xrays` in parallel. The function looks for instances of outlier pixels with respect to the average pixel value for all data. Returns the original data array with hot pixel values replaced with `np.nan`.

5. __Mask detector hole__
    * The function `gued.apply_mask` uses the `gued.mask_generator_alg` function to create a mask of `np.nan` values based on center and set radius. Returns masked data. Has the capability to apply multiple masks.

6. __Calculate diffraction center__
    * The function `gued.find_center_pool` runs the function `gued.find_center_alg` in parallel to find the center of all images. The pool function speeds up the process significantly but with small data sets can run `gued.find_center_alg` directly.

7. __Remove radial outliers__
    * The function `gued.remove_radial_outliers_pool` uses the hidden function `gued._preprocess_radial_data` which converts the data to polar coordinates, creates an interpolated average image from radial averages, then looks for instances of radial outliers and replaces them with `np.nan`.
    * This is by far the most time-consuming part of data processing. __Only do this with small data sets (i.e., after stage averaging) unless you're willing to spend a long time processing data.__ Takes 10 minutes per 100 images running in parallel.

8. __Apply median filter__
    * The function `gued.median_filter` applies a median filter to the data.

9. __Retrieve Azimuthal Average__
    * The function `gued.azimuthal_average` takes the 3D data array and returns the azimuthal average for each data set. This function also applies
    a normalization to the data if desired.
   
10. __Apply Polynomial Fit__
    * The function `gued.poly_fit` fits the azimuthally averaged data to a polynomial then subtracts this baseline from the data.

13. __Generate PDF__
    * This is a work in progress but updating often. 
   

## Usage: 

The first step is to use `Set_up_Globals.ipynb` file to test and optimize the __global variables__ stored in `gued_globals.py` which need to be adjusted for each experiment. This notebook uses the average of all the data and plots examples to see if your variables are set properly. Once you feel good about the global variables, move on to the next step.  

    Example of a `gued_globals.py` file:
```### Global Variables for s1 data set

# Variable for reading files
SEPARATORS = ['-', '_']

# Variables for Center Finding Algorithm
CENTER_GUESS = (460, 460)
RADIUS_GUESS = 35
DISK_RADIUS = 3
THRESHOLD = 150 # When average data, set to 0

# Variable for Generating Background
CORNER_RADIUS = 20
CHECK_NUMBER = 50

# Variables for Masking
MASK_CENTER = [475, 475]
MASK_RADIUS = 40
ADDED_MASK = [[440, 435, 30], [460, 450, 30]]

# Used throughout code as the threshold for cutting out date. This is the default value but other values can be set for the functions using
# std_factor = 4
STD_FACTOR = 3

# Specifies the maximum number of workers to be used when running concurrent.futures
MAX_PROCESSORS = 6

# Adjust figure size 
FIGSIZE = (12,4)

# Path for Theory Package

PATH_DCS = 'C:\\Users\\laure\\OneDrive - University of Nebraska-Lincoln\\Documents\\Centurion Lab\\Coding Lab Notebook\\gued_package\\GUED_Analysis\\packages\\dcs_repositiory\\3.7MeV\\'
#path_dcs = '/sdf/home/l/lheald2/GUED/jupyter_notebook/user_notebooks/dcs_repository/3.7MeV/'
```

An example notebook named `Fast_Analysis.ipynb` should be run as the second step in the data processing. This notebook applies and plots all the above steps after having averaged based on the stage position associated with the data. This notebook will get you to the ΔI/I. 

Once the global variables are set, it is possible to run all the functions above on a large data set using the `process_all.py` file. This file interatively processes images following the above steps (without averaging based on stage position) and saves the total scattering and stage positions to an h5 file. Running __2000__ images takes __~ 25 minutes__ on a personal laptop. 

After processing all of the images and saving to an .h5 file, it can be useful to check drifts with respect to lab time. An example of tracking drifts in t0 with respect to labtime is done in the `T0_Analysis.ipynb` notebook. The data is broken up into groups and the rise time is fit to the different subsets of data to look for changes due to drifts during the data collection.

Finally, after data has been thoroughly cleaned and processed, the .h5 file can be read into the `PDF_Generation.ipynb` notebook to convert the ΔI/I to the pair distribution function (PDF). Note: This notebook is still a work in progress.

Another notebook that will likely be helpful is the `GUED_Simulations.ipynb` notebook which can be used to simulate scattering data from input structure files such as .xyz and .csv files. Additionally, can simulate time resolved diffraction patterns from trajectory files and vibrational .hess files generated through ORCA. 


## Acknowledgements: 
Code was adapted from multiple sources including:  
Caidan Moore  
Cuong Le  
Yusong Liu  

The entire Centurion group at the University of Nebraska - Lincoln offered advice and guidance throughout the development. 
The Stanford National Accelerator Laboratory - MeV-UED Facility Staff offered advice and guidance throughout development.

## Dependencies:

__* numpy  *  scipy  *  matplotlib  *  pandas  *  tifffile  *  skimage  *  concurrent  *  h5py  *  glob  *  functools *__  

