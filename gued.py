"""Code written by Lauren F. Heald with support from Caidan Moore, Cuong Le, and Yusong Liu. Intended to be used for processing ultrafast 
gas phase electron diffraction data collected at the MeV-UED hutch of LCLS at the Stanford Linear Accelerator National Laboratory. 

Questions or Concerns: 
    email: lheald2@unl.edu

Affiliations:
    Centurion Lab at the University of Nebraska -- Lincoln, NE
    Stanford Linear Accelerator National Lab -- Menlo Park, CA
"""


# Standard Packages
import numpy as np
import numpy.ma as ma
from tifffile import tifffile as tf
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from scipy import signal
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import median_filter
import concurrent.futures
from functools import partial
import h5py

# Image stuff
import matplotlib.patches as patches
from skimage.filters import threshold_otsu
from skimage.morphology import closing, square, disk
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops_table
from skimage import util, draw

# Configuration File
from gued_globals import *

# error handling
import warnings

# Suppress warnings about mean of empty slice and degrees of freedome for empty slice (warnings normally appear when taking azimuthal average)
warnings.filterwarnings('ignore', category=RuntimeWarning, message='Mean of empty slice')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='Degrees of freedom <= 0 for slice')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='divide by zero encountered in log')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in log')



### Reading Images Functions

def _show_counts(stage_positions, counts):
    """Function for visualizing and plotting total counts from a set of data. Called within the get_image_details
    function when plot == True"""

    counts_mean = np.mean(counts)  # Mean values of Total Counts of all images
    counts_std = np.std(counts)  # the STD of all the tc for all the iamges
    uni_stage = np.unique(stage_positions)  # Pump-probe stage position
    plt.figure(figsize=FIGSIZE)  # Plot counts rate, images number at each posi, and bad images

    plt.subplot(1, 3, 1)
    plt.plot(counts, '-d')
    plt.axhline(y=counts_mean, color='k', linestyle='-', linewidth=1, label="mean counts")
    plt.axhline(y=counts_mean - (3 * counts_std), color='r', linestyle='-', linewidth=0.5, label="min counts")
    plt.axhline(y=counts_mean + (3 * counts_std), color='r', linestyle='-', linewidth=0.5, label="max counts")
    plt.xlabel('Images orderd in lab time')
    plt.ylabel('Counts')
    plt.legend()
    plt.title('Total counts')

    plt.subplot(1, 3, 2)  # Histogram the number of images at each posi
    plt.plot(uni_stage, '-o')
    plt.xlabel('pp stage posi')
    plt.ylabel('Stg Position [mm]')
    plt.title('Delay Stage Position')

    plt.subplot(1, 3, 3)  # Histogram the number of images at each posi
    posi_edges_bins = np.append(uni_stage - 0.001, uni_stage[-1])
    posi_hist, posi_edges = np.histogram(stage_positions, bins=posi_edges_bins)
    plt.plot(uni_stage, posi_hist, '-*')
    plt.xlabel('pp stage posi [mm]')
    plt.ylabel('Num of Imges')
    plt.title('Num of images at each delay')

    plt.tight_layout()
    plt.show()
    
    return


def _get_counts(data_array, plot=False):
    """
    Generates the counts from the given data by summing over the array elements. Returns 2d array of the same dimension as the
    input images.

    ARGUMENTS:

    data_array (3d array): 
        Array containing the diffraction images.
    
    OPTIONAL ARGUMENTS:

    plot (boolean): 
        Default set to False. When true, plots a graph of the counts data.

    RETURNS:

    counts (2d array): 
        One dimensional array containing the data after summing over each array element.

    """
    counts = np.sum(data_array, axis=(1, 2))
    if len(data_array) == 0:
        raise ValueError("Input data_array is empty.")
    if data_array.ndim != 3:
        raise ValueError("Input data_array is not 3 dimensional.")
    if plot == True:
        plt.figure(figsize=FIGSIZE)
        plt.plot(np.arange(len(data_array[:, 0, 0])), counts)
        plt.show()
    return counts


def get_image_details(file_names, sort=True, filter_data=False, plot=False):
    """
    Reads all images from input file_names and returns the data as a 3d array along with stage positions, order, and counts per image.

    ARGUMENTS:

    file_names (list):
        list of file names to be read in

    OPTIONAL ARGUMENTS:

    sort (boolean): 
        default is set to True. This arguments sorts the data based on when it was saved (i.e. file number)
    plot (boolean): 
        default is set to False. When True, a plot of the data, log(data), and histogram of counts is shown
    filter_data (boolean or list): 
        default is set to False. If you want to select only a fraction of the images, set filter_data = [min_image,]

    GLOBAL VARIABLES:

    SEPARATORS (list):
        list of strings such as '_' or '-' which are used in the file naming scheme to separate values needed for data analysis (i.e. stage
        position)

    RETURNS:

    data_array (3d array): 
        Array of N x 1024 x 1024 where N is the length of tile file_names list. Generated by using tifffile as tf.
    stage_positions (array): 
        An array containing the stage positions of the file. The index of each stage position corresponds to the index of the file name 
        in file_names.
    file_order (array): 
        Returns the image number located in the file name. Reflects the order with which the images are taken.
    counts(array): 
        One dimensional numpy array of length N containing the data after summing over each array element.

    """
    if type(SEPARATORS) == list:
        try:
            stage_positions = []
            file_order = []
            try:
                # stage_pos = [np.float64(file_name[idx_start:idx_end]) for file_name in file_names]
                # stage_pos = np.array(stage_pos)
                for file in file_names:
                    string = list(
                        map(str, file.split("\\")))  # Note standard slash usage for windows todo might need to test
                    folder_number = string[-3][-3:]
                    string = list(map(str, string[-1].split(SEPARATORS[0])))
                    file_number = int(folder_number + string[1])
                    file_order.append(int(file_number))
                    string = list(map(str, string[-1].split(SEPARATORS[1])))
                    stage_positions.append(float(string[0]))
            except ValueError:
                raise ValueError("""Failed to convert a file name to a float. Make sure that index positions are correct for all files in file_names. 
                Also check separators""")
        except IndexError:
            raise ValueError(
                "Invalid index values. Make sure the index values are within the range of the file name strings.")
    elif type(SEPARATORS) == str:
        try:
            stage_positions = []
            file_order = []
            try:
                # stage_pos = [np.float64(file_name[idx_start:idx_end]) for file_name in file_names]
                # stage_pos = np.array(stage_pos)
                for file in file_names:
                    string = list(map(str, file.split("\\")))
                    string = list(map(str, string[-1].split(SEPARATORS)))
                    file_order.append(int(string[2]))
                    stage_positions.append(float(string[3]))
            except ValueError:
                raise ValueError(
                    """Failed to convert a file name to a float. Make sure that index positions are correct for all files in file_names. Also check separators""")
        except IndexError:
            raise ValueError(
                "Invalid index values. Make sure the index values are within the range of the file name strings.")
    else:
        print("Provide valid SEPARATORS")

    stage_positions = np.array(stage_positions)
    file_order = np.array(file_order)
    file_names = np.array(file_names)


    if sort == True:
        idx_sort = np.argsort(file_order)
        file_order = file_order[idx_sort]
        file_names = list(file_names[idx_sort])
        stage_positions = stage_positions[idx_sort]


    if type(filter_data) == list:
        print("Filtering files")
        min_val = filter_data[0]
        max_val = filter_data[1]
        if max_val < len(file_names):
            data_array = tf.imread(file_names[min_val:max_val])  # construct array containing files
            counts = _get_counts(data_array)
            stage_positions = stage_positions[min_val:max_val]
            file_order = file_order[min_val:max_val]
        else:
            print("Max value is larger than the size of the data range, returning all data")
   
    elif filter_data == False:
        data_array = tf.imread(file_names)  # construct array containing files
        counts = _get_counts(data_array)



    if plot == True:
        test = data_array[0]
        plt.figure(figsize=FIGSIZE)
        plt.subplot(1, 3, 1)
        plt.imshow(test, cmap='jet')
        plt.xlabel('Pixel')
        plt.ylabel('Pixel')
        plt.title('Linear Scale(data)')

        plt.subplot(1, 3, 2)
        plt.imshow(np.log(test), cmap='jet')
        plt.xlabel('Pixel')
        plt.ylabel('Pixel')
        plt.title('Log Scale(data)')

        plt.subplot(1, 3, 3)
        plt.hist(test.reshape(-1), bins=30, edgecolor="r", histtype="bar", alpha=0.5)
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Pixel Number')
        plt.title('Hist of the pixel intensity(data)')
        plt.yscale('log')
        plt.tight_layout()
        plt.show()

        _show_counts(stage_positions, counts)

    return data_array, stage_positions, file_order, counts


def get_image_details_keV(file_names, sort=False, multistage=False, filter_data=False, plot=False):
    # todo update to look like other get_image_details code and make for one stage
    """
    Reads all images from input file_names and returns the data as a 3d array along with stage positions, order, and counts per image.

    ARGUMENTS:

    file_names (list):
        list of file names to be read in

    OPTIONAL ARGUMENTS:

    sort (boolean): 
        default is set to True. This arguments sorts the data based on when it was saved (i.e. file number)
    multistage (boolean):
        default is set to False. Use this when file names contain information from multiple stages.
    plot (boolean): 
        default is set to False. When True, a plot of the data, log(data), and histogram of counts is shown
    filter_data (boolean): 
        default is set to False. When True, code prompts you for a minimum and maximum value then
        returns only the information from files within this range

    RETURNS:

    data_array (3d array): 
        Array of N x 1024 x 1024 where N is the length of tile file_names list. Generated by using tifffile as tf.
    stage_positions (array): 
        An array containing the stage positions of the file. The index of each stage position corresponds to the index of the file name 
        in file_names.
    file_order (array): 
        Returns the image number located in the file name. Reflects the order with which the images are taken.
    counts(array): 
        One dimensional numpy array of length N containing the data after summing over each array element.

    """
    data_array = tf.imread(file_names)  # construct array containing files
    if multistage == True:
        try:
            ir_stage_pos = []
            uv_stage_pos = []
            file_order = []
            current = []
            try:
                for file in file_names:
                    string = list(map(str, file.split("\\")))
                    string = list(map(str, string[-1].split("_")))
                    file_number = int(string[1])
                    file_order.append(file_number)
                    ir_stage_pos.append(float(string[4]))
                    uv_stage_pos.append(float(string[6]))
                    current.append(float(string[-1][:-5]))
            except ValueError:
                raise ValueError("""Failed to convert a file name to a float. Make sure that index positions are correct for all files in file_names. 
                Also check separators""")
        except IndexError:
            raise ValueError(
                "Invalid index values. Make sure the index values are within the range of the file name strings.")

        ir_stage_pos = np.array(ir_stage_pos)
        uv_stage_pos = np.array(uv_stage_pos)
        file_order = np.array(file_order)
        current = np.array(current)

        uni_stage_ir = np.unique(ir_stage_pos)# Pump-probe stage position
        uni_stage_uv = np.unique(uv_stage_pos)

        if len(uni_stage_ir) > 1: 
            stage_positions = ir_stage_pos
        elif len(uni_stage_uv) > 1: 
            stage_positions = uv_stage_pos
        else:
            print("Bad Stage Positions")

        if sort == True:
            temp_idx = _sort_files_multistage(file_order, ir_stage_pos, uv_stage_pos)
            data_array = data_array[temp_idx]
            stage_positions = stage_positions[temp_idx]
            file_order = file_order[temp_idx]
            current = current[temp_idx]

        if type(filter_data) == list:
            print("Filtering files")
            min_val = filter_data[0]
            max_val = filter_data[1]
            if max_val < len(file_names):
                data_array = tf.imread(file_names[min_val:max_val])  # construct array containing files
                counts = _get_counts(data_array)
                stage_positions = stage_positions[min_val:max_val]
                file_order = file_order[min_val:max_val]
            else:
                print("Max value is larger than the size of the data range, returning all data")
   
        elif filter_data == False:
            data_array = tf.imread(file_names)  # construct array containing files
            counts = _get_counts(data_array)

        if plot == True:
            test = data_array[0]
            plt.figure(figsize=FIGSIZE)
            plt.subplot(1, 3, 1)
            plt.imshow(test, cmap='jet')
            plt.xlabel('Pixel')
            plt.ylabel('Pixel')
            plt.title('Linear Scale(data)')

            plt.subplot(1, 3, 2)
            plt.imshow(np.log(test), cmap='jet')
            plt.xlabel('Pixel')
            plt.ylabel('Pixel')
            plt.title('Log Scale(data)')

            plt.subplot(1, 3, 3)
            plt.hist(test.reshape(-1), bins=30, edgecolor="r", histtype="bar", alpha=0.5)
            plt.xlabel('Pixel Intensity')
            plt.ylabel('Pixel Number')
            plt.title('Hist of the pixel intensity(data)')
            plt.yscale('log')
            plt.tight_layout()
            plt.show()

            _show_counts(stage_positions, counts)

        return data_array, stage_positions, file_order, counts, current

    if multistage == False:
        try:
            stage_positions = []
            file_order = []
            current = []
            try:
                for file in file_names:
                    string = list(map(str, file.split("/")))
                    string = list(map(str, string[-1].split("_")))
                    file_number = int(string[1])
                    file_order.append(file_number)
                    stage_positions.append(float(string[4]))
                    current.append(float(string[-1][:-5]))
            except ValueError:
                raise ValueError("""Failed to convert a file name to a float. Make sure that index positions are correct for all files in file_names. 
                Also check separators""")
        except IndexError:
            raise ValueError(
                "Invalid index values. Make sure the index values are within the range of the file name strings.")

        stage_positions = np.array(stage_positions)
        file_order = np.array(file_order)
        current = np.array(current)

        if sort == True:
            temp_idx = _sort_files(file_order, stage_positions)
            data_array = data_array[temp_idx]
            stage_positions = stage_positions[temp_idx]
            file_order = file_order[temp_idx]
            current = current[temp_idx]

        if type(filter_data) == list:
            print("Filtering files")
            min_val = filter_data[0]
            max_val = filter_data[1]
            if max_val < len(file_names):
                data_array = tf.imread(file_names[min_val:max_val])  # construct array containing files
                counts = _get_counts(data_array)
                stage_pos = stage_pos[min_val:max_val]
                file_order = file_order[min_val:max_val]
            else:
                print("Max value is larger than the size of the data range, returning all data")
   
        elif filter_data == False:
            data_array = tf.imread(file_names)  # construct array containing files
            counts = _get_counts(data_array)

        if plot == True:
            test = data_array[0]
            plt.figure(figsize=FIGSIZE)
            plt.subplot(1, 3, 1)
            plt.imshow(test, cmap='jet')
            plt.xlabel('Pixel')
            plt.ylabel('Pixel')
            plt.title('Linear Scale(data)')

            plt.subplot(1, 3, 2)
            plt.imshow(np.log(test), cmap='jet')
            plt.xlabel('Pixel')
            plt.ylabel('Pixel')
            plt.title('Log Scale(data)')

            plt.subplot(1, 3, 3)
            plt.hist(test.reshape(-1), bins=30, edgecolor="r", histtype="bar", alpha=0.5)
            plt.xlabel('Pixel Intensity')
            plt.ylabel('Pixel Number')
            plt.title('Hist of the pixel intensity(data)')
            plt.yscale('log')
            plt.tight_layout()
            plt.show()

            _show_counts(stage_pos, counts)

    return data_array, stage_positions, file_order, counts, current


def _sort_files_multistage(file_order, ir_stage_pos, uv_stage_pos):
    """ Hidden function for sorting files for experiments with multiple stage positions"""
    uni_stage_ir = np.unique(ir_stage_pos)  # Pump-probe stage position
    uni_stage_uv = np.unique(uv_stage_pos)

    if len(uni_stage_ir) > 1:
        stage_positions = ir_stage_pos
        print("sorting based on IR stage position")
    elif len(uni_stage_uv) > 1:
        stage_positions = uv_stage_pos
        print("sorting based on UV stage position")
    else:
        print("Bad Stage Positions")
    idx_list = []
    uni_stage = np.unique(stage_positions)
    for i in range(len(uni_stage)):
        # file_numbers = file_order[np.where(stage_positions==uni_stage[i])[0]]
        # file_numbers = file_numbers[idx_temp]
        stage_idx = np.where(stage_positions == uni_stage[i])[0]
        file_numbers = file_order[stage_idx]
        idx_temp = np.argsort(file_numbers)
        # print(file_numbers[idx_temp])
        idx_list.append(stage_idx[idx_temp])
    idx_list = np.array(idx_list)
    idx_list = np.reshape(idx_list, len(stage_positions))
    return idx_list


def _sort_files(file_order, stage_positions):
    """Hidden function for sorting files based on image name"""
    uni_stage = np.unique(stage_positions)  # Pump-probe stage position

    idx_list = []

    for i in range(len(uni_stage)):
        # file_numbers = file_order[np.where(stage_positions==uni_stage[i])[0]]
        # file_numbers = file_numbers[idx_temp]
        stage_idx = np.where(stage_positions == uni_stage[i])[0]
        file_numbers = file_order[stage_idx]
        idx_temp = np.argsort(file_numbers)
        # print(file_numbers[idx_temp])
        idx_list.append(stage_idx[idx_temp])
    idx_list = np.array(idx_list)
    idx_list = np.reshape(idx_list, len(stage_positions))
    return idx_list


### Cleaning Functions 

def remove_counts(data_array, stage_positions, file_order, counts, added_range = [], std_factor=STD_FACTOR, plot=False):
    # todo add edge option
    """
    Filters input parameters by removing any data where the total counts falls outside of the set filter. Default
    value is set to 3 standard deviations from the mean. Returns the same variables as it inputs but with
    different dimensions.

    ARGUMENTS:

    data_array (ndarray): 
        Multidimensional array of N x 1024 x 1024 where N is the length of file_names list
    stage_pos (array): 
        One dimensional array of length N containing the stage positions associated with each image.
    file_order (array): 
        One dimensional array of length N that reflects the order with which the images are taken.
    counts(ndarray): 
        One dimensional array of length N containing the total counts after summing over each array
        element.

    OPTIONAL ARGUMENTS:

    std_factor (int): 
        Default value is 3. Refers to cut off based on number of standard deviations from the mean.
    plot (boolean): 
        Default is False. Returns a plot of new and old counts.

    RETURNS:

    Returns same variables which it received as arguments with new N value.

    """

    init_length = len(counts)
    # Decide to use threshold or selected images
    counts_mean = np.mean(counts)  # Mean values of Total Counts of all images
    counts_std = np.std(counts)  # the STD of all the tc for all the iamges

    tc_good = np.squeeze(
        np.where(abs(counts - counts_mean) < std_factor * counts_std))  # Find out the indices of the low counts images
    new_array = data_array[tc_good]
    new_stage_positions = stage_positions[tc_good]
    new_counts = counts[tc_good]
    new_file_order = file_order[tc_good]

    for rng in added_range:
        new_array = np.concatenate((new_array[:rng[0]], new_array[rng[1]:]))
        print(len(new_array))
        new_stage_positions = np.concatenate((new_stage_positions[:rng[0]], new_stage_positions[rng[1]:]))
        new_counts = np.concatenate((new_counts[:rng[0]], new_counts[rng[1]:]))
        new_file_order = np.concatenate((new_file_order[:rng[0]], new_file_order[rng[1]:]))

    print(init_length - len(new_counts), " number of files removed from ", init_length, " initial files")

    if plot == True:
        plt.figure(figsize=(12, 4))  # Plot counts rate, images number at each posi, and bad images

        plt.plot(new_counts, '-d')
        plt.axhline(y=counts_mean, color='k', linestyle='-', linewidth=1, label="mean counts")
        plt.axhline(y=counts_mean - (3 * counts_std), color='r', linestyle='-', linewidth=0.5, label="min counts")
        plt.axhline(y=counts_mean + (3 * counts_std), color='r', linestyle='-', linewidth=0.5, label="max counts")
        plt.xlabel('Images orderd in lab time')
        plt.ylabel('Counts')
        plt.legend()
        plt.title('Total counts')

        plt.tight_layout()
        plt.show()

    return new_array, new_stage_positions, new_file_order, new_counts


def remove_background(data_array, remove_noise=True, plot=False, print_status=True):  
    """
    Takes in a 3d data array and calculates the means of the corners then linearly interpolates values based on corners across 3d array to 
    generate of background noise values using pandas.DataFrame.interpolate.

    ARGUMENTS:

    data_array (3d ndarray): 
        array containing all data

    OPTIONAL ARGUMENTS:

    remove_noise (boolean): 
        Default set to true, returns image with background subtracted. If false, returns
        interpolated background.
    plot (boolean): 
        Default set to False. Plots images showing the original image, interpolated background, and
        background subtracted image.
    print_status (boolean): 
        Default set to True. Prints a status update every nth image (n defined via CHECK_NUMBER).

    GLOBAL VARIABLES:

    CORNER_RADIUS (int): 
        defines the size of the corners being used in background suptraction.
    CHECK_NUMBER (int): 
        defines how often updates are given when print_status == True

    RETURNS:

    clean_data (3d ndarray): 
        Returns array of images with background subtracted if remove_noise == True, else returns
        array of interpolated background.

    """

    if not isinstance(data_array, np.ndarray):
        raise ValueError("Input data_array must be a numpy array.")
    if not isinstance(CORNER_RADIUS, int) and CORNER_RADIUS > 0:
        raise ValueError("bkg_range must be an integer > 0.")
    if not isinstance(remove_noise, bool):
        raise ValueError("remove_noise must be a boolean.")
    if not (2 * CORNER_RADIUS < len(data_array[:, 0, :]) and
            2 * CORNER_RADIUS < len(data_array[:, :, 0])):
        raise ValueError("2 * bkg-range must be less than both the number of rows and the number of columns.")

    clean_data = []
    bkg_data = []
    for i, image in enumerate(data_array):
        empty_array = np.empty(np.shape(image))
        empty_array = (ma.masked_array(empty_array, mask=True))
        empty_array[0, 0] = np.mean(image[0:CORNER_RADIUS, 0:CORNER_RADIUS])
        empty_array[-1, 0] = np.mean(image[-CORNER_RADIUS:, 0:CORNER_RADIUS])
        empty_array[0, -1] = np.mean(image[0:CORNER_RADIUS, -CORNER_RADIUS:])
        empty_array[-1, -1] = np.mean(image[-CORNER_RADIUS:, -CORNER_RADIUS:])
        empty_array = pd.DataFrame(empty_array).interpolate(axis=0)
        empty_array = pd.DataFrame(empty_array).interpolate(axis=1)
        bkg = pd.DataFrame.to_numpy(empty_array)
        bkg_data.append(bkg)
        clean_data.append(image - bkg)
        if print_status == True:
            if i % CHECK_NUMBER == 0:
                print(f"Subtracting background of {i}th image.")

    if plot == True:
        plt.figure()

        plt.subplot(1, 3, 1)
        plt.imshow(data_array[0])
        plt.title("Original Data")

        plt.subplot(1, 3, 2)
        plt.imshow(bkg_data[0])
        plt.title("Interpolated Background")

        plt.subplot(1, 3, 3)
        plt.imshow(clean_data[0])
        plt.title("Background Free Data")
        plt.tight_layout()
        plt.show()

    if remove_noise == True:
        return np.array(clean_data)
    else:
        return np.array(bkg_data)


def _remove_background(image):
    """
    Takes in a 2d data array (using the mean array is recommended) and calculates the means of the corners. Linearly
    interpolates values across 2d array to generate of background noise values using pandas.DataFrame.interpolate.

    ARGUMENTS:

    image (2d ndarray): 
        single image array

    GLOBAL VARIABLES:

    CORNER_RADIUS (int): 
        defines the size of the corners being used in background suptraction.
    CHECK_NUMBER (int): 
        defines how often updates are given when print_status == True

    RETURNS:

    clean_image (2d ndarray): 
        Returns image with background subtracted if remove_noise == True, else returns array of interpolated background.

    """
    if not isinstance(image, np.ndarray):
        raise ValueError("Input data_array must be a numpy array.")
    if not isinstance(CORNER_RADIUS, int) and CORNER_RADIUS > 0:
        raise ValueError("bkg_range must be an integer > 0.")
    if not (2 * CORNER_RADIUS < len(image[0, :]) and
            2 * CORNER_RADIUS < len(image[:, 0])):
        raise ValueError("2 * bkg-range must be less than both the number of rows and the number of columns.")

    clean_data = []
    bkg_data = []

    empty_array = np.empty(np.shape(image))
    empty_array = (ma.masked_array(empty_array, mask=True))
    empty_array[0, 0] = np.mean(image[0:CORNER_RADIUS, 0:CORNER_RADIUS])
    empty_array[-1, 0] = np.mean(image[-CORNER_RADIUS:, 0:CORNER_RADIUS])
    empty_array[0, -1] = np.mean(image[0:CORNER_RADIUS, -CORNER_RADIUS:])
    empty_array[-1, -1] = np.mean(image[-CORNER_RADIUS:, -CORNER_RADIUS:])
    empty_array = pd.DataFrame(empty_array).interpolate(axis=0)
    empty_array = pd.DataFrame(empty_array).interpolate(axis=1)
    bkg = pd.DataFrame.to_numpy(empty_array)
    bkg_data.append(bkg)
    clean_data.append(image - bkg)

    return np.squeeze(np.array(clean_data)), np.squeeze(np.array(bkg_data))


def remove_background_pool(data_array, remove_noise=True, plot=False):
    """ 
    Removes the background of images based on the corners. Runs the hidden function _remove_background and runs it in parallel.

    ARGUMENTS:

    data_array (3d array): 
        data array of all images

    OPTIONAL ARGUMENTS:

    remove_noise (boolean): 
        Default set to true. Returns data array with noise removed. If false, only returns the interpolated background
    plot (boolean): 
        Default set to false. When true, plots an example of original data, interpolated background, and cleaned image.

    RETURNS:

    clean_data (3d array): 
        Original data with background removed when remove_noise==True
    or
    backgrounds (3d array):
        Interpolated background for each image when remove_noise==False

    """
    clean_data = []
    backgrounds = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_PROCESSORS) as executor:
        results = executor.map(_remove_background, data_array)

    for result in results:
        clean, bkg = result
        clean_data.append(clean)
        backgrounds.append(bkg)

    clean_data = np.array(clean_data)
    backgrounds = np.array(backgrounds)

    if plot == True:   
        fig, axes = plt.subplots(1, 3, figsize=FIGSIZE)
        vmin_1_1, vmax_1_1 = 10**-0, 4000 

        #Before
        img1 = axes[0].imshow(data_array[0], cmap='plasma', norm=matplotlib.colors.LogNorm(vmin=vmin_1_1, vmax=vmax_1_1))
        axes[0].set_title('Original Data with Outliers Removed')
        axes[0].set_xlabel('X-position')
        axes[0].set_ylabel('Y-position')
        colorbar1 = fig.colorbar(img1, ax=axes[0])  # Add colorbar to the first subplot

        #Noise
        img2 = axes[1].imshow(backgrounds[0], cmap='plasma',  norm=matplotlib.colors.LogNorm(vmin=vmin_1_1, vmax=vmax_1_1))  
        # Use the same normalization as img1
        axes[1].set_title('Interpolated Noise Data')
        axes[1].set_xlabel('X-position')
        axes[1].set_ylabel('Y-position')
        colorbar2 = fig.colorbar(img2, ax=axes[1])  # Add colorbar to the second subplot

        # Noise Removed
        img3 = axes[2].imshow(clean_data[0], cmap='plasma',  norm=matplotlib.colors.LogNorm(vmin=vmin_1_1, vmax=vmax_1_1))  # Logarithmic scaling
        axes[2].set_title('After Background Removal')
        axes[2].set_xlabel('X-position')
        axes[2].set_ylabel('Y-position')
        colorbar3 = fig.colorbar(img3, ax=axes[2])  # Add colorbar to the third subplot
    if remove_noise == True:
        return clean_data
    else:
        return backgrounds


def _remove_xrays(mean_data, std_data, image, std_factor=STD_FACTOR):
    """This is the hidden function that is run within the remove_xrays_pool function.

    ARGUMENTS:

    mean_data (2d array): 
        average image of all data in data_array from parent function.
    std_data (2d array): 
        image with standard deviation values from all data in data_array in parent function.
    image (2d array): 
        array of image like data with length N where N is number of images.

    OPTIONAL ARGUMENTS:

    std_factor (int): 
        Default set to 3. Defines the threshold for removing pixels with |pixel_value - mean| > std_factor*std

    RETURNS:

    clean_data (2d array): 
        array of image like data with shape of input data array where errant pixels are now masked based on the set threshold
        amt_rmv (int): count of all pixels removed per image

    """

    upper_threshold = mean_data + std_factor * std_data
    clean_data = ma.masked_greater_equal(image, upper_threshold)
    amt_rmv = np.sum(clean_data.mask)
    return clean_data, amt_rmv


def remove_xrays(data_array, plot=True): # testing for timing

    """
    Filters out any pixels that are more than set threshold value based on the standard deviation of the
    average pixel value by running the hidden function _remove_xrays in parallel.

    ARGUMENTS:

    data_array (3d array): 
        array of image like data with length N where N is number of images.

    OPTIONAL ARGUMENTS:

    plot (boolean): 
        Default set to True. Plots the percentage of pixeled removed during cleaning process
    std_factor (int): 
        Default set to 3. Defines the threshold for removing pixels with |pixel_value - mean| > std_factor*std

    RETURNS:

    clean_data (3d array): 
        array of image like data with shape of input data array where errant pixels are now masked based on the set threshold

    """

    mean_data = np.mean(data_array, axis=0)
    std_data = np.std(data_array, axis=0)
    print("Removing hot pixels from all data")

    clean_data = []
    amt_rmv = []

    for i in range(len(data_array)):
        clean, amt = _remove_xrays(data_array[i], mean_data, std_data)
        clean_data.append(clean)
        amt_rmv.append(amt)

    pct_rmv = np.array(amt_rmv) / (len(data_array[1]) * len(data_array[2])) * 100

    if plot == True:
        plt.figure(figsize=FIGSIZE)
        plt.subplot(1, 3, 1)
        plt.plot(pct_rmv)
        plt.title("Percent Pixels Removed")
        plt.xlabel("Image Number")
        plt.ylabel("Percent")

        plt.subplot(1, 3, 2)
        plt.imshow(data_array[0])
        plt.title("Original Image")

        plt.subplot(1, 3, 3)
        plt.imshow(clean_data[0])
        plt.title("Cleaned Image")
        plt.tight_layout()
        plt.show()

    return clean_data


def remove_xrays_pool(data_array, plot=False, return_pct = False, std_factor=STD_FACTOR):
    """
    Filters out any pixels that are more than set threshold value based on the standard deviation of the
    average pixel value by running the hidden function _remove_xrays in parallel.

    ARGUMENTS:

    data_array (3d array): 
        array of image like data with length N where N is number of images.

    OPTIONAL ARGUMENTS:

    plot (boolean): 
        Default set to True. Plots the percentage of pixeled removed during cleaning process
    return_pct (boolean):
        Default is set to False. When true, returns the pct removed along with the clean_data
    std_factor (int): 
        Default set to 3. Defines the threshold for removing pixels with |pixel_value - mean| > std_factor*std

    RETURNS:

    clean_data (3d array): 
        array of image like data with shape of input data array where errant pixels are now masked based on the set threshold

    """

    mean_data = np.mean(data_array, axis=0)
    std_data = np.std(data_array, axis=0)
    print("Removing hot pixels from all data")
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_PROCESSORS) as executor:
        futures = [executor.submit(partial(_remove_xrays, mean_data, std_data), data) for
                   data in data_array]
        results = [future.result() for future in futures]

    clean_data = []
    amt_rmv = []
    for result in results:
        data, amt = result
        clean_data.append(data)
        amt_rmv.append(amt)

    pct_rmv = np.array(amt_rmv) / (len(data_array[1]) * len(data_array[2])) * 100

    if plot == True:
        plt.figure(figsize=FIGSIZE)
        plt.subplot(1, 3, 1)
        plt.plot(pct_rmv)
        plt.title("Percent Pixels Removed")
        plt.xlabel("Image Number")
        plt.ylabel("Percent")

        plt.subplot(1, 3, 2)
        plt.imshow(data_array[0])
        plt.title("Original Image")

        plt.subplot(1, 3, 3)
        plt.imshow(clean_data[0])
        plt.title("Cleaned Image")
        plt.tight_layout()
        plt.show()
    if return_pct==True:
        return np.array(clean_data), np.array(pct_rmv)
    else:
        return np.array(clean_data)


def subtract_background(data_array, mean_background, plot=True):
    """Takes in 3d data_array and subtracts each image from the input mean_background 2d array. Returns cleaned
    data_array

    ARGUMENTS:

    data_array (3d array): 
        original data array
    mean_background (2d array): 
        average of background images

    OPTIONAL ARGUMENTS:

    plot (boolean): 
        Default is True. Plots example original image and background subtracted image

    RETURNS:

    clean_data (3d array): 
        data_array - mean_background
    """

    clean_data = data_array - mean_background

    if plot == True:
        plt.figure(figsize=FIGSIZE)
        plt.subplot(1, 2, 1)
        plt.imshow(data_array[0])
        plt.title("Original Image")

        plt.subplot(1, 2, 2)
        plt.imshow(clean_data[0])
        plt.title("Cleaned Image")
        plt.tight_layout()
        plt.show()

    return clean_data


def remove_based_on_center(centers, data_array, stage_positions, std_factor=2, plot=False):
    """ TODO ADD DOC STRING """
    centers = np.array(centers)
    center_ave = np.mean(centers, axis=0)
    center_std = np.nanstd(centers, axis=0)
    init_length = len(data_array)
    
    # Use np.logical_and for element-wise logical AND
    good_idx = np.where(
        np.logical_and(
            np.abs(centers[:, 0] - center_ave[0]) < (std_factor * center_std[0]),
            np.abs(centers[:, 1] - center_ave[1]) < (std_factor * center_std[1])
        )
    )[0]

    print(good_idx.shape)
    print(np.sum(good_idx))
    new_array = data_array[good_idx]
    new_stage_positions = stage_positions[good_idx]
    new_centers = centers[good_idx]

    print(init_length - len(new_array), " number of files removed from ", init_length, " initial files")

    if plot:
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(new_centers[:, 0], '-d', label='New Centers')
        plt.plot(centers[:, 0], label='Original Centers')
        plt.axhline(y=center_ave[0], color='k', linestyle='-', linewidth=1, label="mean")
        plt.axhline(y=center_ave[0] - (3 * center_std[0]), color='r', linestyle='-', linewidth=0.5, label="min")
        plt.axhline(y=center_ave[0] + (3 * center_std[0]), color='r', linestyle='-', linewidth=0.5, label="max")
        plt.xlabel('Pixel Index')
        plt.ylabel('Center x Values')
        plt.legend()
        plt.title('New x Centers')

        plt.subplot(1, 2, 2)
        plt.plot(new_centers[:, 1], '-d', label='New Centers')
        plt.plot(centers[:, 1], label='Original Centers')
        plt.axhline(y=center_ave[1], color='k', linestyle='-', linewidth=1, label="mean")
        plt.axhline(y=center_ave[1] - (3 * center_std[1]), color='r', linestyle='-', linewidth=0.5, label="min")
        plt.axhline(y=center_ave[1] + (3 * center_std[1]), color='r', linestyle='-', linewidth=0.5, label="max")
        plt.xlabel('Pixel Index')
        plt.ylabel('Center y Values')
        plt.legend()
        plt.title('New y Centers')

        plt.tight_layout()
        plt.show()

    return new_array, new_stage_positions, new_centers

    
# Masking and Center Finding Functions

def mask_generator_alg(image, fill_value=np.nan, add_rectangular=False, plot=False):
    """
    Generate mask to cover unwanted area

    ARGUMENTS:

    image : 2D array
        Diffraction pattern.

    OPTIONAL ARGUMENTS: 

    fill_value : int, float, or nan, optional
        Value that use to fill the area of the mask. The default is np.nan.
    add_rectangular : boolean, optional
        Additional mask with rectangular shape. The default is True. Uses global variables to define the shape of the rectangle
    showingfigure : boolean, optional
        Show figure of the result of applied masks. The default is False.

    GLOBAL VARIABLES:

    MASK_CENTER (1d array, tuple, or list that contains only two values):
        Center for generating mask cover unscattered electron beam.
    MASK_RADIUS (int):
        Radius of the mask.
    ADDDED_MASK (list of 3-value-lists):
        Additional masks. Input gonna be [[x-center, y-center, radius], [...], ...] The default is [].
    REC_LENGTH (int):
        length of the rectangle in the vertical direction
    REC_EXTENT (tuple of ints):
        defines the length and width of the rectangle


    RETURNS:
    
    mask (binary 2D array):
        Result of all the masks in an image.

    """

    mask = np.ones(image.shape)
    rows, cols = draw.disk((MASK_CENTER[1], MASK_CENTER[0]), MASK_RADIUS, shape=mask.shape)
    mask[rows, cols] = fill_value

    if len(ADDED_MASK) == 0:
        pass
    else:
        for i in ADDED_MASK:
            rows, cols = draw.disk((i[1], i[0]), i[2], shape=mask.shape)
            mask[rows, cols] = fill_value

    # retangular mask
    if add_rectangular == True:
        rr, cc = draw.rectangle((0, REC_LENGTH), extent=REC_EXTENT, shape=image.shape)  # (0,535) for iodobenzene
        mask[rr, cc] = fill_value
        # 515

    if plot == True:
        fig, axs = plt.subplots(1, 3, figsize=FIGSIZE)

        # First subplot: Mean of Unmasked Data Array
        axs[0].imshow(image)
        axs[0].set_title("Mean of Unmasked Data Array")
        cbar = plt.colorbar(axs[0].imshow(image), ax=axs[0])
        cbar.ax.set_ylabel('Intensity')

        masked_data = mask*image
        # Second subplot: Mean of Masked Data Array
        axs[1].imshow(masked_data)
        axs[1].set_title("Mean of Masked Data Array")
        cbar = plt.colorbar(axs[1].imshow(masked_data), ax=axs[1])
        cbar.ax.set_ylabel('Intensity')

        # Third subplot: Contour map of average data
        x = np.arange(300, 700)
        y = np.arange(300, 700)
        X, Y = np.meshgrid(y, x)
        pc = axs[2].pcolormesh(x, y, np.log(masked_data[300:700, 300:700]), shading='auto')
        cs = axs[2].contour(X, Y, np.log(masked_data[300:700, 300:700]), levels=20, colors='w')
        axs[2].set_title('Contour map of average data')
        cbar = fig.colorbar(pc, ax=axs[2])
        cbar.ax.set_ylabel('Log(Intensity)')

        # Adjust layout to prevent overlap
        plt.tight_layout()

        # Show the combined figure
        plt.show()

    return mask


def apply_mask(data_array, fill_value=np.nan, add_rectangular=False, plot=False, print_vals=False):  
    """ Applies a mask to individual images in the data array.

    ARGUMENTS:

    data_array (array):
        2D or 3D array of an image or images.

    OPTIONAL ARUGMENTS:

    fill_value (int, float, or nan):
        Default set to np.nan. Value used to fill the area of the mask.
    add_rectangular (boolean)
        The default is True. Adds an additional mask with rectangular shape. 
    plot (boolean):
        Default set to False. When true, plots a figure with the original data, the masked data, and a contour map of the data
    print_vals (boolean):
        Default set to False. When true, prints the values at each contour line. Useful for setting global variables

    GLOBAL VARIABLES:

    MASK_CENTER : 1D array, tuple, or list that contains only two values
        Center for generating mask cover unscattered electron beam.
    MASK_RADIUS : int
        Radius of the mask.
    ADDED_MASK : list of 3-value-lists, optional
        Additional masks. Input gonna be [[x-center, y-center, radius], [...], ...] The default is [].

    RETURNS:

    mask : binary 2D array
        Result of all the masks in an image.

    """

    if len(data_array.shape) == 2:
        masked_data = data_array * mask_generator_alg(data_array, fill_value, add_rectangular)

        if plot == True:
            fig, axs = plt.subplots(1, 3, figsize=FIGSIZE)

            # First subplot: Mean of Unmasked Data Array
            axs[0].imshow(data_array)
            axs[0].set_title("Mean of Unmasked Data Array")
            cbar = plt.colorbar(axs[0].imshow(data_array), ax=axs[0])
            cbar.ax.set_ylabel('Intensity')

            # Second subplot: Mean of Masked Data Array
            axs[1].imshow(masked_data)
            axs[1].set_title("Mean of Masked Data Array")
            cbar = plt.colorbar(axs[1].imshow(masked_data), ax=axs[1])
            cbar.ax.set_ylabel('Intensity')

            # Third subplot: Contour map of average data
            x = np.arange(300, 700)
            y = np.arange(300, 700)
            X, Y = np.meshgrid(y, x)
            pc = axs[2].pcolormesh(x, y, np.log(masked_data[300:700, 300:700]), shading='auto')
            cs = axs[2].contour(X, Y, np.log(masked_data[300:700, 300:700]), levels=20, colors='w')
            axs[2].set_title('Contour map of average data')
            cbar = fig.colorbar(pc, ax=axs[2])
            cbar.ax.set_ylabel('Log(Intensity)')

            # Retrieve and print the intensity values of the contour lines
            if print_vals == True:
                intensity_values = np.exp(cs.levels)
                for value in intensity_values:
                    print(f"Intensity value: {value:.2f}")

            # Adjust layout to prevent overlap
            plt.tight_layout()

            # Show the combined figure
            plt.show()
        
    if len(data_array.shape) == 3:
        mean_data = np.nanmean(data_array, axis=0)
        masked_data = data_array * mask_generator_alg(mean_data, fill_value, add_rectangular)
        masked_mean = np.nanmean(masked_data, axis=0)

        if plot == True:
            fig, axs = plt.subplots(1, 3, figsize=FIGSIZE)

            # First subplot: Mean of Unmasked Data Array
            axs[0].imshow(mean_data)
            axs[0].set_title("Mean of Unmasked Data Array")
            cbar = plt.colorbar(axs[0].imshow(mean_data), ax=axs[0])
            cbar.ax.set_ylabel('Intensity')

            # Second subplot: Mean of Masked Data Array
            axs[1].imshow(masked_mean)
            axs[1].set_title("Mean of Masked Data Array")
            cbar = plt.colorbar(axs[1].imshow(masked_mean), ax=axs[1])
            cbar.ax.set_ylabel('Intensity')

            # Third subplot: Contour map of average data
            x = np.arange(300, 700)
            y = np.arange(300, 700)
            X, Y = np.meshgrid(y, x)
            pc = axs[2].pcolormesh(x, y, np.log(masked_mean[300:700, 300:700]), shading='auto')
            cs = axs[2].contour(X, Y, np.log(masked_mean[300:700, 300:700]), levels=20, colors='w')
            axs[2].set_title('Contour map of average data')
            cbar = fig.colorbar(pc, ax=axs[2])
            cbar.ax.set_ylabel('Log(Intensity)')

            # Retrieve and print the intensity values of the contour lines
            if print_vals == True:
                intensity_values = np.exp(cs.levels)
                for value in intensity_values:
                    print(f"Intensity value: {value:.2f}")

            # Adjust layout to prevent overlap
            plt.tight_layout()

            # Show the combined figure
            plt.show()

    return masked_data


def finding_center_alg(image, plot=False, title='Reference Image'):
    """
    Algorithm for finding the center of diffraction pattern

    ARGUMENTS:
    
    data_array : 2D array
        Diffraction pattern.

    OPTIONAL ARGUMENTS:

    thresh_input (float):
        Default set to 0. When zero, the threshold value is calculated using threshold_otsu from scikit-images. Often doesn't work
    plot : boolean, optional
        Show figure of the result of center finding. The default is False.
    title : str, optional
        Title of the figure. The default is 'Reference image'.


    GLOBAL VARIABLES:

    DISK_RADIUS : int, optional
        Generates a flat, disk-shaped footprint. The default is 3.
    CENTER_GUESS : tuple contains 2 values, optional
        Guessing center position to generate temporary mask. The default is (532, 520).
    RADIUS_GUESS : int, optional
        Guessing radius of the temporary mask. The default is 80.

    RETURNS
    
    center_x : int
        Center value on x axis.
    center_y : int
        Center value of y axis.
    radius : int
        Radius of ring used for finding center.

    """

    if THRESHOLD == 0:
        thresh = threshold_otsu(image)
    else:
        thresh = THRESHOLD

    cxt, cyt = [], []
    ## apply median filter to help
    image = signal.medfilt2d(image, kernel_size=9)
    for th in [1]:
        thresh *= th
        mask_temp = mask_generator_alg(image, fill_value=False,
                                       add_rectangular=False)
        mask_temp = util.invert(mask_temp.astype(bool))
        bw = closing(image > thresh, disk(
            DISK_RADIUS))  # Return grayscale morphological closing of an image. Square(): generate the footprint to close the gap between data points
        cleared = clear_border(bw + mask_temp)
        label_image = label(cleared)
        props = regionprops_table(label_image, properties=('centroid',
                                                           'axis_major_length',
                                                           'axis_minor_length'))
        dia = np.array([props['axis_major_length'], props['axis_minor_length']])
        dia = np.mean(dia, axis=0)
        radius = np.amax(dia) / 2
        idx = np.where(dia == np.amax(dia))[0][0]
        cxt.append(props['centroid-1'][idx])
        cyt.append(props['centroid-0'][idx])

    center_x = np.mean(cxt)
    center_y = np.mean(cyt)

    if plot == True:
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=FIGSIZE)
        ax1.imshow(image)
        ax2.imshow(label_image)
        ax3.imshow(bw)

        for cc in range(len(cxt)):
            circ = patches.Circle((cxt[cc], cyt[cc]), radius, linewidth=1, edgecolor="r", facecolor="none",
                                  linestyle='--')
            ax1.add_patch(circ)
            circ = patches.Circle((cxt[cc], cyt[cc]), radius, linewidth=2, edgecolor="r", facecolor="none")
            ax2.add_patch(circ)
            circ = patches.Circle((cxt[cc], cyt[cc]), radius, linewidth=2, edgecolor="r", facecolor="none")
            ax3.add_patch(circ)

        for ax in (ax1, ax2, ax3):
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        ax1.set_title(title, fontsize=10)
        ax2.set_title("Center [X = " + str(center_x) + ", Y = " + str(center_y) + "]", fontsize=10)
        ax3.set_title("Binary image", fontsize=20)

        ax1.axvline(center_x, linestyle='--', lw=1, color='tab:red')
        ax1.axhline(center_y, linestyle='--', lw=1, color='tab:red')

        ax2.axvline(center_x, linestyle='--', lw=2, color='tab:red')
        ax2.axhline(center_y, linestyle='--', lw=2, color='tab:red')

        plt.tight_layout()
        plt.show()

    return center_x, center_y, radius, thresh


def find_center_pool(data_array, plot=True, print_stats=False):
    """ Finds center of each image in the data array using concurrent.futures.ThreadPoolExecutor to quickly process
    many data files.

    ARGUMENTS:

    data_array (ndarray): 
        array of image like data with shape Nx1024x1024

    OPTIONAL ARGUMENTS:

    plot (boolean): 
        Default is set to True. When true, plots an image of the values for center_x and center_y with respect to pixel number
    print_stats (boolean): 
        Default is set to True. Prints the average value for center_x and center_y and prints the percent failure rate.

    GLOBAL VARIABLES:

    CENTER_GUESS (tuple): 
        initial guess for center position
    RADIUS_GUESS (int): 
        initial guess for the radius
    DISK_RADIUS (int): 
        value for disk radius used in mapping

    RETURNS:

    center_x (array):
        One-dimensional array of x values for the center position of each image
    center_y (array): 
        One-dimensional array of y values for the center position of each image"""

    center_x = []
    center_y = []
    radii = []
    thresholds = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_PROCESSORS) as executor:
        results = executor.map(finding_center_alg, data_array)

    for result in results:
        cx, cy, radius, thresh = result
        center_x.append(cx)
        center_y.append(cy)
        radii.append(radius)
        thresholds.append(thresh)

    center_x = np.array(center_x)
    center_y = np.array(center_y)
    radii = np.array(radii)
    thresholds = np.array(thresholds)
    if plot == True:
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 2, 1)
        plt.plot(center_x[:])
        plt.title("X values for Centers")
        plt.xlabel("Image Number")
        plt.ylabel("Pixel Value")

        plt.subplot(2, 2, 2)
        plt.plot(center_y[:])
        plt.title("Y values for Centers")
        plt.xlabel("Image Number")
        plt.ylabel("Pixel Value")

        plt.subplot(2,2,3)
        plt.plot(radii[:])
        plt.title("Radii found during center finding")
        plt.xlabel("Image Number")
        plt.ylabel("Radius")

        plt.subplot(2,2,4)
        plt.plot(thresholds[:])
        plt.title("Threshold Values")
        plt.xlabel("Image Number")
        plt.ylabel("Thresholds")

        plt.tight_layout()
        plt.show()

    if print_stats == True:
        x_ave = np.mean(center_x[np.where(center_x != CENTER_GUESS[0])[0]])
        y_ave = np.mean(center_y[np.where(center_y != CENTER_GUESS[1])[0]])
        center_x[np.where(center_x == CENTER_GUESS[0])[0]] = x_ave
        center_y[np.where(center_y == CENTER_GUESS[1])[0]] = y_ave
        center_ave = x_ave, y_ave
        print(r'Averaged ctr is ' + str(center_ave))
        fail_count = np.count_nonzero(np.array(center_x) == CENTER_GUESS[0])

        print(
            f"Percentage of images where the center finding failed (i.e., found the guess value): {fail_count / len(data_array) * 100}")
    return center_x, center_y


### Azimuthal Averaging and Radial Outlier Removal Functions
# todo: clean and optimize
def cart2pol(x, y): 
    """Converts cartesian x,y coordinates to polar coordinates of r, theta."""
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return r, theta


def preprocess_for_azimuthal_checking(center, dat):
    """Preprocesses the data by creating a meshgrid based on the center and shape of the original image, converts the meshgrid to polar
    coordinates to get a list of radial values then returns the length of x (int) and the r values (1d array)."""
    w, h = dat.shape
    xmat, ymat = np.meshgrid(np.arange(0,w,1)-center[0],np.arange(0,h,1)-center[1])
    rmat, _ = cart2pol(xmat, ymat)
    rmat = np.around(rmat)
    dat = dat.astype(float)
    xlength = int(np.amax([np.amax(abs(xmat)),np.amax(abs(ymat))]))
    
    return xlength, rmat


def cleaning_2d_data(center, dat, std_factor=STD_FACTOR, fill_value='nan'):
    """Runs the outlier removal algorithm to check for instances of pixels which are outside of the radial average. """
    xlength, rmat = preprocess_for_azimuthal_checking(center, dat)
    res2d = np.copy(dat)
    
    mask_detect = True
    for i in range(xlength):
        roi = np.copy(dat[rmat==int(i+1)])
        if len(roi)==0:
            break
        if int(i+1)>=500:
            break
        # Check the area of mask so the azimuthal integration will ignore that.
        if mask_detect==True:
            if np.sum(np.isnan(roi)) < len(roi):
                mask_detect=False
                
        if mask_detect==False:
            # remove value that higher or lower than correct_factor*standard deviation
            roi = outlier_rev_algo(roi, std_factor=std_factor, fill_value=fill_value)
        
        res2d[rmat==int(i+1)] = np.copy(roi)
    return res2d


def outlier_rev_algo(dat1d, std_factor=STD_FACTOR, fill_value = 'nan'):
    index = np.logical_or(dat1d>=np.nanmean(dat1d)+std_factor*np.nanstd(dat1d), dat1d<=np.nanmean(dat1d)-std_factor*np.nanstd(dat1d))
    if fill_value == 'nan':
        dat1d[index] = np.nan
    elif fill_value == 'average':
        dat1d[index] = np.nanmean(dat1d)
    return dat1d


def remove_radial_outliers_pool(data_array, centers, fill_value='nan', plot=False, return_pct=False):
    """
    Removes instances of outlier pixels based on the radial average of the image. Runs the hidden function _remove_radial_outliers in parallel. 
    Works by first converting an individual array to polar coordinates and remaps to create an average image. Then performs a logical check on 
    the original image compared to interpolated image. 
    
    ARGUMENTS: 
    
    data_array (3d array):
        Original data 
    center (list):
        Can either be an average center value of form [x, y] or a list of centers of form [[x1,y1], [x2, y2], ...]

    OPTIONAL ARGUMENTS: 

    plot (boolean):
        default set to False. When true, plots an example of original data, the interpolated average image, and the cleaned image
    return_pct (boolean):
        Default set to False. When true, returns the percentage of pixels removed per image. 

    RETURNS:

    clean_data (3d array):
        data with outliers removed

    """

    clean_data = []
    rmv_count = []
    
    if len(centers) > 2:
        print("Using all center values")
        print("Removing radial outliers from all data")
        
        # Use partial to include the optional parameter in the cleaning_2d_data function
        cleaning_with_optional_param = partial(cleaning_2d_data, fill_value=fill_value)
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_PROCESSORS) as executor:
            # Pass the partially defined function to map
            results = list(executor.map(cleaning_with_optional_param, centers, data_array))
        
        for result in results:
            clean_image = result
            clean_data.append(clean_image)
            rmv = np.isnan(clean_image)
            rmv_count.append(np.sum(rmv) / (len(data_array[0][0]) * len(data_array[0][1])))

    elif len(centers) == 2:
        print("Using average center")
        print("Removing radial outliers from all data")
        
        cleaning_with_optional_param = partial(cleaning_2d_data, centers, fill_value=fill_value)
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_PROCESSORS) as executor:
            futures = [executor.submit(cleaning_with_optional_param, data) for data in data_array]
            results = [future.result() for future in futures]
        
        for result in results:
            clean_image = result
            clean_data.append(clean_image)
            rmv = np.isnan(clean_image)
            rmv_count.append(np.sum(rmv) / (len(data_array[0][0]) * len(data_array[0][1])))

    clean_data = np.array(clean_data)
    rmv_count = np.array(rmv_count)*100

    if plot == True:
        plt.figure(figsize=FIGSIZE)
        plt.subplot(1, 3, 1)
        plt.imshow(data_array[0])
        # plt.xlim(300, 600)
        # plt.ylim(300, 600)
        plt.title("Original Image")

        plt.subplot(1, 3, 2)
        plt.imshow(clean_data[0])
        # plt.xlim(300, 600)
        # plt.ylim(300, 600)
        plt.title("Cleaned Image")

        plt.subplot(1, 3, 3)
        plt.plot(rmv_count)
        plt.title("Percent of nan Values per Image")
        plt.tight_layout()
        plt.show()
    if return_pct == True:
        return clean_data, rmv_count
    else:
        return clean_data


def fill_missing(center, image):
    """
    ARGUMENTS:

    center (tuple):
        center value for the image
    image(2D array):
        image array

    NOTES
    Pchip interpolation: We use scipy.interpolate.PchipInterpolator to fill missing values for the first 200 elements (aziFill[1:200]). 
    We create a mask to identify NaN values and then interpolate only those missing points.
    Nearest interpolation: pandas.Series.fillna(method='nearest') is used for filling NaN values from index 400 onwards with the nearest 
    available values.
    """

    xlength, rmat = preprocess_for_azimuthal_checking(center, image)

    _, azi_fill, _ = azimuthal_integration_alg(center, image) # Azimuthal average with nan values

    # Part 1: Fill missing values in the first 200 elements using 'polynomial'  interpolation
    azi_fill[:200] = pd.Series(azi_fill[:200]).interpolate(method='polynomial', order=1).to_numpy()

    # Part 2: Fill missing values from index 400 onwards using 'nearest' interpolation
    azi_fill[400:] = pd.Series(azi_fill[400:]).interpolate(method='nearest').to_numpy()
    print(azi_fill.shape)
    new_image = np.copy(image)

    for i in range(xlength):
        roi = np.copy(image[rmat==int(i+1)])
        if len(roi)==0:
            break
        if int(i+1)>=len(azi_fill):
            break
        index = np.isnan(roi)
        #print(index)
        roi[index] = azi_fill[i+1] #fill outliers with interpolated value
        
        new_image[rmat==int(i+1)] = np.copy(roi)
        
    return new_image



def _median_filter(image, kernel_size = 5):
    """
    Applies the scipy.ndimage.median_filter function to the image then returns the filtered image"""

    #corners = (np.median(data_array_1d[-50:, -50:]), np.median(data_array_1d[-50:, :50]), np.median(data_array_1d[:50, -50:]), 
                    #np.median(data_array_1d[:50, :50]))
    #floor = float(np.mean(corners))
    filt_data = median_filter(image, kernel_size)

    return filt_data


def median_filter_pool(data_array, centers, plot=True):
    """Takes in a large 3D array of data and applies the scipy.ndimage.median_filter on them in parallel processing using the hidden function
    _median_filter.

    ARGUMENTS: 

    data_array (3d array):
        array of all images
    
    OPTIONAL ARGUMENTS:

    plot(boolean): Default set to True
        When true, plots an example of the original and of the filtered image

    RETURNS: 
        
    filtered_data (3d array):
        filtered data array of the same size as the input array"""
    
    nan_idx = np.isnan(data_array) # get nan locations in order to replace later
    
    filled_data = []
    
    if len(centers) > 2:
        print("Using all center values ")
        with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_PROCESSORS) as executor:
            results = list(executor.map(fill_missing, centers, data_array))
        for result in results:
            filled_data.append(result)

    elif len(centers) == 2:
        print("Using average center")
        with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_PROCESSORS) as executor:
            futures = [executor.submit(partial(fill_missing, centers), data) for data in data_array]
            results = [future.result() for future in futures]

        for result in results:
            filled_data.append(result)

    filled_data = np.array(filled_data)
    print("Done filling nan values, starting median filter")
    
    filtered_data = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_PROCESSORS) as executor:
        results = executor.map(_median_filter, filled_data)
        
    for result in results:
        filtered_data.append(result)
    
    filtered_data = np.array(filtered_data)

    filtered_data[nan_idx] = np.nan # replace bad pixels with nan after applying median filter

    if plot == True:
        plt.figure(figsize=FIGSIZE)
        plt.subplot(1,2,1)
        plt.imshow(data_array[0])
        plt.title("Original Image")
        
        plt.subplot(1,2,2)
        plt.imshow(filtered_data[0])
        plt.title("Filtered Image")
        plt.show()

    return filtered_data

def azimuthal_integration_alg(center, image, max_azi=450):
    """
    Generate 1D data from 2D image using azimuthal integration.
    1D data must be clean before applying azimuthal integration.

    Parameters
    ----------
    dat : 2D array
        Diffration pattern (result after background subtraction and masking).
    center : list, 1D array, or tuple that containt only two values
        Center of the image.

    Returns
    -------
    Three 1D array: 
        1: s (length from the center corresponding to each value)
        2: azimuthal data
        3: standard deviation of azimuthal data

    """
    
    xlength, rmat = preprocess_for_azimuthal_checking(center, image)
    #azi_dat, azi_err, s0 = [], [], []

    azi_dat = np.full(max_azi, np.nan)
    azi_err = np.full(max_azi, np.nan)
    s0 = np.full(max_azi, np.nan)
    #dat[dat<-300] = np.nan
    
    mask_detect = True
    for i in range(xlength):
        roi = np.copy(image[rmat==int(i+1)])
        if len(roi)==0:
            break
        
        if int(i+1)>= max_azi:
            break
        # Check the area of mask so the azimuthal integration will ignore that.
        if mask_detect==True:
            if np.sum(np.isnan(roi)) < len(roi):
                mask_detect=False
                
        if mask_detect==False:
            #s0.append(i+1)
            s0[i+1] = i+1
            #azi_dat.append(np.nanmean(roi))
            azi_dat[i+1] = np.nanmean(roi)
            #azi_err.append(np.nanstd(roi)/np.sqrt(abs(np.nansum(roi))))
            azi_err[i+1]=np.nanstd(roi)/np.sqrt(abs(np.nansum(roi)))

    return np.array(s0), np.array(azi_dat), np.array(azi_err)


def get_azimuthal_average_pool(data_array, centers, normalize=False, plot=False):
    """
    Code for getting the azimuthal average of each image for a given center. 

    ARGUMENTS:

    data_array (3d array):
        array of images which will be integrated over. 
    centers (list):
        Can be either a list with length two of [cx, cy] or a list of lists i.e., [[cx1, cy1], [cx2, cy2]...] which define the center positions

    OPTIONAL ARGUMENTS:

    normalize (boolean):
        Default set to False. When true, runs the normalize_to_baseline function
    plot (boolean):
        Default set to False. When true, plots an example of the azimuthally averaged data

    RETURNS:

    average_data (2d array):
        Integrated averages for each image
    std_data (2d array):
        Standard deviations associated with integrated average. 
    """
    average_data = []
    std_data = []
    s_ranges = []

    if len(centers) > 2:
        print("Using all center values ")
        print("Calculating azimuthal average for all data")
        with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_PROCESSORS) as executor:
            # Zip the arrays together and submit to the executor
            results = list(executor.map(azimuthal_integration_alg, centers, data_array))
        for result in results:
            s, ave, std = result
            average_data.append(ave)
            std_data.append(std)
            s_ranges.append(s)

    elif len(centers) == 2:
        print("Using average center")
        print("Calculating azimuthal average for all data")
        with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_PROCESSORS) as executor:
            futures = [executor.submit(partial(azimuthal_integration_alg, centers), data) for data in data_array]
            results = [future.result() for future in futures]
        min_azi = 500
        for result in results:
            s, ave, std = result
            if len(ave) < min_azi:
                min_azi=len(ave)
                print(f"maximum value previously: {min_azi} -- length of array: {len(ave)}")
            average_data.append(ave)
            std_data.append(std)
            s_ranges.append(s)

    average_data = np.array(average_data)
    std_data = np.array(std_data)

    if normalize == True:
        norm_data = normalize_to_baseline(average_data)
        average_data = norm_data


    if plot == True:
        plt.figure()
        plt.plot(average_data[0])
        plt.title("Example of Azimuthal Average")
        plt.show()

    return average_data, std_data


def normalize_to_baseline(data_array2d, min_val=50, max_val=100):
    """
    Normalizes a 2d data set based on the average of the data between the min_val and the max_val

    ARGUMENTS:

    data_array2d (2d array):
        azimuthally averaged 2d data with shape of length unique stage positons vs maximum s distance
    
    OPTIONAL ARGUMENTS:

    min_val (int): 
        Default set to 50. Defines the minimum point for normalization range
    max_val (int): 
        Default set to 100. Defines the maximum point for normalization range

    RESULTS:

    data_norm (2d array):
        normalized array of azimuthally averaged data with same shape as data_array2d
            
    """

    data_array2d[:, :25] = np.nan
    data_mean = np.nanmean(data_array2d, axis=0)
    norm_factor = np.nansum(data_mean[min_val:max_val])
    data_norm = []
    for i in range(len(data_array2d)):
        offset = np.nansum(data_array2d[i, min_val:max_val])
        norm = data_array2d[i] * (norm_factor / offset)
        data_norm.append(norm)

    data_norm = np.array(data_norm)

    return data_norm


def normalize_to_range(data_array, min_val = -1, max_val=1):
    """ Normalizes a 2d array so that it falls within the defined minimum value and maximum value. 
    
    ARGUMENTS:
    
    data_array (2d array):
        array of data such as the azimuthally averaged data with respect to stage position
        
    OPTIONAL ARGUMENTS:
    
    min_val (int):
        Default set to -1. Minimum value that the new data should be within
    max_val (int):
        Default set to 1. Maximum value that the new data should be within

    RETURNS:

    data_norm (2d array):
        new data that falls within minimum and maximum values with the same shape as the input array. 

    """

    data_norm = []

    for i in range(len(data_array)):
        norm = (max_val - min_val)*(data_array[i] - np.nanmin(data_array[i]))/(np.nanmax(data_array[i]) - np.nanmin(data_array[i])) + min_val
        data_norm.append(norm)
    
    return np.array(data_norm)


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
        
    elif len(data_array.shape) == 1:
        temp_data = data_array
        idx_nan = ~ np.isnan(temp_data)
        coeff = np.polyfit(x_vals[idx_nan], temp_data[idx_nan], degree)
        baseline2d = np.polyval(coeff, x_vals)
        
        corrected_data = data_array - baseline2d
    else:
        print("Data Array must be 1D or 2D array")

    if plot == True:
        plt.figure()
        plt.subplot(1,2,1)
        plt.plot(data_array[1])
        plt.plot(baseline2d[1])
        plt.xlabel("pixel")
        plt.title("delta I/I original with fit line")

        plt.subplot(1,2,2)
        plt.plot(corrected_data[1])
        plt.xlabel("pixel")
        plt.title("delta I/I corrected")

        plt.tight_layout()
        plt.show()

    if return_baseline == True:
        return corrected_data, baseline2d
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
            filtered_temp = signal.filtfilt(b, a, data_array[i][25:])
            filtered_data.append(filtered_temp)
        filtered_data = np.array(filtered_data)
        print(filtered_data)

        if plot == True:
            plt.figure()
            plt.subplot(2,1,1)
            plt.plot(data_array[1])
            plt.title("Original Data")

            plt.subplot(2,1,2)
            plt.plot(filtered_data[1])
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
# Saving and Loading Data

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
        

def read_combined_data(file_name, group_name, variable_names, run_numbers = 'all'):
    """Reads in and concatenates all the data within a group from an h5 file.
    
    ARGUMENTS:
    
    file_name (str):
        file name or path that holds the data of interest
    group_name (str):
        group subset within the file
    variable_names (list of strings):
        list of the variable names you're interested in. 

    OPTIONAL ARGUMENTS:

    run_numbers (list):
        default set to 'all' which reads in all runs in the group of interest. If you want to only read in particular run numbers, set 
        run_numbers = list like [1,2,3]
        
    RETURNS:
    
    combined_data (dict):
        dictionary containing the data concatenated along run number containing the variables specified in variable names

    """
    if run_numbers == 'all':
        concatenated_data = {name: None for name in variable_names}

        with h5py.File(file_name, 'r') as f:
            if group_name not in f:
                raise ValueError(f"Group '{group_name}' not found in the HDF5 file.")
            
            group = f[group_name]

            # Initialize lists to accumulate data across runs
            data_accumulators = {name: [] for name in variable_names}

            # Iterate over each run dataset
            for run_dataset_name in group.keys():
                for variable_name in variable_names:
                    if run_dataset_name.startswith(variable_name):
                        data_accumulators[variable_name].append(group[run_dataset_name][:])

            # Concatenate the accumulated data across runs
            for name in variable_names:
                if data_accumulators[name]:
                    concatenated_data[name] = np.concatenate(data_accumulators[name], axis=0)
                else:
                    concatenated_data[name] = None
            f.close()

    elif type(run_numbers) == list:
        concatenated_data = {name: None for name in variable_names}

        with h5py.File(file_name, 'r') as f:
            if group_name not in f:
                raise ValueError(f"Group '{group_name}' not found in the HDF5 file.")
            
            group = f[group_name]

            # Initialize lists to accumulate data across selected runs
            data_accumulators = {name: [] for name in variable_names}

            # Iterate over each run dataset and check if it's in the selected runs
            for run_dataset_name in group.keys():
                run_number = int(run_dataset_name.split('_run_')[-1])
                if run_number in run_numbers:
                    for variable_name in variable_names:
                        if run_dataset_name.startswith(variable_name):
                            data_accumulators[variable_name].append(group[run_dataset_name][:])

            # Concatenate the accumulated data across selected runs
            for name in variable_names:
                if data_accumulators[name]:
                    concatenated_data[name] = np.concatenate(data_accumulators[name], axis=0)
                else:
                    concatenated_data[name] = None
            f.close()
    return concatenated_data
    

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

        


