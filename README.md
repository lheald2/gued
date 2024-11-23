# gued package

Written by Lauren F Heald 
Email: lheald2@unl.edu

## About:
This package was created for data processing and analysis for experiments conducted at the MeV-UED facility at the Linac Coherent Light Source at the Stanford Linear Accelerator. The main packages, `gued.py` and `gued_theory.py` contain all the relevant functions for data processing, and each Jupyter Notebook serves a specific function for cleaning, processing, and saving image based scattering data. This package is developed and maintained by Lauren F. Heald, PhD.

### Current Status:
  The package has been rigorously tested with multiple data sets collected at the MeV-UED facility. However, updates are posted often as lessons are learned and new tricks are implemented. Please note, this package is meant to serve as a backbone for data processing, but further noise reduction and analysis are encouraged. If you have questions or concerns, email Lauren Heald at lheald2@unl.edu with subject line "GUED Help". 

## Current Functionality
Different notebooks within the repository serve different purposes but the general data processing pipeline is outlined below (as found in `Fast_Analysis.py`).
1. __Import all images__
    * Function called `gued.get_image_details` is used to load in all .tif files in the specified folder of interest. The function returns the images as a 3D data array, a 1D array of stage positions, file order, and total counts per image.

2. __Average Based on Stage Positions__
    * Group all images based on the stage positions in order to speed up data processing steps for large data sets

3. __Find Centers__
    * Find diffraction center for all images using the function called `gued.find_centers_pool` which runs the `find_center_alg` function in parallel

4. __Reject images with bad total counts__
    * Function called `gued.remove_counts` loads in the returns from `gued.get_image_details` and removes any images based on their total counts then returns the inputs with bad images removed.

5. __Subtract background__
    * The function called `gued.remove_background_pool` takes in a 3D array containing all image files and runs the hidden function `_remove_background` which creates a background image based on the corners of the original image. Then can either return the interpolated backgrounds or the cleaned data.
    * In cases where background images are taken as part of the experiment, use the `subtract_background` function with the data array and an average background image.

6. __Remove outlier instances of identical pixels__
    * This is generally referred to as removing x-ray hits or hot pixels. When working with large data sets, use the `gued.remove_xrays_pool` function. This function takes the 3D data array and runs the hidden function `_remove_xrays` in parallel. The function looks for instances of outlier pixels with respect to the average pixel value for all data. Returns the original data array with hot pixel values replaced with `np.nan`.

7. __Mask detector hole__
    * The function `gued.apply_mask` uses the `gued.mask_generator_alg` function to create a mask of `np.nan` values based on center and set radius. Returns masked data. Has the capability to apply multiple masks.

8. __Calculate diffraction center__
    * The function `gued.find_center_pool` runs the function `gued.find_center_alg` in parallel to find the center of all images. The pool function speeds up the process significantly but with small data sets can run `gued.find_center_alg` directly.

9. __Remove radial outliers__
    * The function `gued.remove_radial_outliers_pool` uses the hidden function `gued._preprocess_radial_data` which converts the data to polar coordinates, creates an interpolated average image from radial averages, then looks for instances of radial outliers and replaces them with `np.nan`.
    * This is by far the most time-consuming part of data processing. __Only do this with small data sets (i.e., after stage averaging) unless you're willing to spend a long time processing data.__ Takes 10 minutes per 100 images running in parallel.


10. __Fill Missing Values (built in to median filter)__
    * `gued.fill_missing` can be used to replace NaN values with the radial average for that detector position. This helps remove artifacts that could be caused by median filtering with NaN values present. This functionality is still being tested. 

11. __Apply median filter__
    * The function `gued.median_filter` applies a median filter to the data. Must replace `np.nan` values with radial average so this function is done in concert with the radial outlier removal (often not necessary and occasionally buggy. Still working on it).

12. __Retrieve Azimuthal Average__
    * The function `gued.azimuthal_average` takes the 3D data array and returns the azimuthal average for each data set.

13. __Plot Pump/Probe Results__
    * Following the azimuthal average calculations, generate a plot of the time resolved data for visualization. 

14. __Apply Polynomial Fit__
    * The function `gued.poly_fit` is used to apply a polynomial fit (with adjustable order) to correct any baseline offsets. 

15. __Save Data__
    * The `gued.save_data` function can be used to save a dictionary of important results to a .h5 for further processing. 
   

Additional notebooks are included for other key purposes and are discussed below. Additionally, some functions are written in the `gued.py` and the `gued_theory.py` files but are not currently in use in any notebooks. 

## Usage: 

The first step when using this package at a MeV-UED experiment is to use `Set_up_Globals.ipynb` file to test and optimize the __global variables__ stored in `gued_globals.py`. These variables need to be adjusted for each experiment. This notebook uses the average of all the data and plots examples to see if your variables are set properly. Once the global variables are set, move on to the `Fast_Analysis.ipynb` notebook for processing pump/probe data. 

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
PATH_DCS = 'gued_package\\GUED_Analysis\\packages\\dcs_repositiory\\3.7MeV\\'
```

An example notebook named `Fast_Analysis.ipynb` should be run as the second step in the data processing. This notebook applies and plots all the above steps after having averaged based on the stage position associated with the data. This notebook will get you to the ΔI/I. 

Another useful notebook is the `Tracking_LabTime.ipynb` notebook which allows for visualization of experimental drifts (i.e., center drifts) by grouping images based on acquisition time. 

Once the global variables are set, it is possible to run all the functions above on a large data set using the `process_all.py` file. This file interatively processes images following the above steps (without averaging based on stage position) and saves the total scattering and stage positions to an h5 file. Running __2000__ images takes __~ 25 minutes__ on a personal laptop. 

After processing all of the images and saving to an .h5 file, it can be useful to check drifts with respect to lab time. An example of tracking drifts in t0 with respect to labtime is done in the `T0_Analysis.ipynb` notebook. The data is broken up into groups and the rise time is fit to the different subsets of data to look for changes due to drifts during the data collection.

Finally, after data has been thoroughly cleaned and processed, the .h5 file can be read into the `PDF_Generation.ipynb` notebook to convert the ΔI/I to the pair distribution function (PDF). 

Another notebook that will likely be helpful is the `GUED_Simulations.ipynb` notebook which can be used to simulate scattering data from input structure files such as .xyz and .csv files. Additionally, can simulate time resolved diffraction patterns from trajectory files and vibrational .hess files generated through ORCA. 

## Citation
If you're relying heavily on this package, please consider citing us following the citation style for open sources packages following the example below:  
```Heald, L.F. (2024) GUED (Version 1.0.0) [Computer Software] Github Repository. https://github.com/lheald2/gued```
See `LICENSE.md` for more information. 

## Acknowledgements: 
Code was written and adapted by Lauren F. Heald with assistance from multiple sources including:  
Caidan Moore (Case Western University)  
Cuong Le (University of Nebraska - Lincoln)  
Yusong Liu (Stanford Linear Accelerator)  
Keke Chen (Tsinghua University)

Additionally, the entire Centurion group at the University of Nebraska - Lincoln and the Stanford National Accelerator Laboratory - MeV-UED Facility Staff offered advice and guidance throughout the development. 

## Relevant Literature
If you're interested in learning more about gas-phase ultrafast electron diffraction, consider reading the following 
  
* Weathersby, S. P.; Brown, G.; Centurion, M.; Chase, T. F.; Coffee, R.; Corbett, J.; Eichner, J. P.; Frisch, J. C.; Fry, A. R.; Gühr, M.; Hartmann, N.; Hast, C.; Hettel, R.; Jobe, R. K.; Jongewaard, E. N.; Lewandowski, J. R.; Li, R. K.; Lindenberg, A. M.; Makasyuk, I.; May, J. E.; McCormick, D.; Nguyen, M. N.; Reid, A. H.; Shen, X.; Sokolowski-Tinten, K.; Vecchione, T.; Vetter, S. L.; Wu, J.; Yang, J.; Dürr, H. A.; Wang, X. J. Mega-Electron-Volt Ultrafast Electron Diffraction at SLAC National Accelerator Laboratory. Rev. Sci. Instrum. 2015, 86 (7), 073702. https://doi.org/10.1063/1.4926994.  
* Yang, J.; Guehr, M.; Shen, X.; Li, R.; Vecchione, T.; Coffee, R.; Corbett, J.; Fry, A.; Hartmann, N.; Hast, C.; Hegazy, K.; Jobe, K.; Makasyuk, I.; Robinson, J.; Robinson, M. S.; Vetter, S.; Weathersby, S.; Yoneda, C.; Wang, X.; Centurion, M. Diffractive Imaging of Coherent Nuclear Motion in Isolated Molecules. Phys. Rev. Lett. 2016, 117 (15), 153002. https://doi.org/10.1103/PhysRevLett.117.153002.  
* Shen, X.; Nunes, J. P. F.; Yang, J.; Jobe, R. K.; Li, R. K.; Lin, M.-F.; Moore, B.; Niebuhr, M.; Weathersby, S. P.; Wolf, T. J. A.; Yoneda, C.; Guehr, M.; Centurion, M.; Wang, X. J. Femtosecond Gas-Phase Mega-Electron-Volt Ultrafast Electron Diffraction. Structural Dynamics 2019, 6 (5), 054305. https://doi.org/10.1063/1.5120864.  
* Wilkin, K. J.; Parrish, R. M.; Yang, J.; Wolf, T. J. A.; Nunes, J. P. F.; Guehr, M.; Li, R.; Shen, X.; Zheng, Q.; Wang, X.; Martinez, T. J.; Centurion, M. Diffractive Imaging of Dissociation and Ground-State Dynamics in a Complex Molecule. Phys. Rev. A 2019, 100 (2), 023402. https://doi.org/10.1103/PhysRevA.100.023402.  
* Yang, J.; Zhu, X.; F. Nunes, J. P.; Yu, J. K.; Parrish, R. M.; Wolf, T. J. A.; Centurion, M.; Gühr, M.; Li, R.; Liu, Y.; Moore, B.; Niebuhr, M.; Park, S.; Shen, X.; Weathersby, S.; Weinacht, T.; Martinez, T. J.; Wang, X. Simultaneous Observation of Nuclear and Electronic Dynamics by Ultrafast Electron Diffraction. Science 2020, 368 (6493), 885–889. https://doi.org/10.1126/science.abb2235 
* Centurion, M.; Wolf, T. J. A.; Yang, J. Ultrafast Imaging of Molecules with Electron Diffraction. Annu. Rev. Phys. Chem. 2022, 73 (1), 21–42. https://doi.org/10.1146/annurev-physchem-082720-010539.  
* Figueira Nunes, J. P.; Ibele, L. M.; Pathak, S.; Attar, A. R.; Bhattacharyya, S.; Boll, R.; Borne, K.; Centurion, M.; Erk, B.; Lin, M.-F.; Forbes, R. J. G.; Goff, N.; Hansen, C. S.; Hoffmann, M.; Holland, D. M. P.; Ingle, R. A.; Luo, D.; Muvva, S. B.; Reid, A. H.; Rouzée, A.; Rudenko, A.; Saha, S. K.; Shen, X.; Venkatachalam, A. S.; Wang, X.; Ware, M. R.; Weathersby, S. P.; Wilkin, K.; Wolf, T. J. A.; Xiong, Y.; Yang, J.; Ashfold, M. N. R.; Rolles, D.; Curchod, B. F. E. Monitoring the Evolution of Relative Product Populations at Early Times during a Photochemical Reaction. J. Am. Chem. Soc. 2024, 146 (6), 4134–4143. https://doi.org/10.1021/jacs.3c13046.  
* Nunes, J. P. F.; Williams, M.; Yang, J.; Wolf, T. J. A.; Rankine, C. D.; Parrish, R.; Moore, B.; Wilkin, K.; Shen, X.; Lin, M.-F.; Hegazy, K.; Li, R.; Weathersby, S.; Martinez, T. J.; Wang, X. J.; Centurion, M. Photo-Induced Structural Dynamics of o -Nitrophenol by Ultrafast Electron Diffraction. Phys. Chem. Chem. Phys. 2024, 26 (26), 17991–17998. https://doi.org/10.1039/D3CP06253H. 


## Dependencies:

__* numpy  *  scipy  *  matplotlib  *  pandas  *  tifffile  *  skimage  *  concurrent  *  h5py  *  glob  *  functools *__  

