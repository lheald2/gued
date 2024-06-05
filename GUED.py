# todo Clean importing calls
# Standard Packages
import numpy as np
from math import exp, sqrt, floor
from tifffile import tifffile as tf
import glob
import matplotlib.pyplot as plt
import pandas as pd
import time
from datetime import date
import numpy.ma as ma
import scipy.signal as ss
import scipy.interpolate as interp
from scipy.optimize import curve_fit
from scipy.signal import medfilt2d
from scipy.ndimage import median_filter
from scipy.ndimage import gaussian_filter
from scipy.interpolate import make_interp_spline
import skimage.transform as skt
import concurrent.futures
import matplotlib
#%matplotlib inline

#Image stuff
import matplotlib.patches as patches
from PIL import Image
from skimage.filters import threshold_otsu
from skimage.morphology import closing, square
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops_table
from skimage import util

#Multiprocessing
import os
from multiprocessing.dummy import Pool as ThreadPool


### Reading Images Functions

# todo learn about classes and figure out how to implement
def get_data(file_path, shape=False, file_names=False): # todo don't think I need this but check anyway and look at notes
    """
    Generates an array of images of TIFF files.

    Arguments:

    file_path (string) = Path containing the UED TIFF files. Path can be for a singular image or for a folder containing
                        many files. See examples below for example using //*// format style.
    shape (bool, optional) = Boolean expression. If set to True, function returns the shape of the image data array.
    file_names (bool, optional) = Boolean Expression. If set to true, function returns the list of file names.

    Returns: data_array, shape (if True), file_names (if True)

    numpy array or tuple:
            If shape = False, file_names = False: Returns numpy array containing data from tiff files as a n x shape x shape (shape refers to the
                                dimensions of the TIFF file, typically 1024 x 1024).
            If shape = True: returns tuple. First element is a numpy array containing data from tiff files as a n x shape x shape
                                (shape refers to the dimensions of the TIFF file, typically 1024 x 1024). Second element is the dimensions
                                of the data array as a tuple.
            If file_names = True: returns tuple containing data array and list with file names.


    Examples:

    file_path = "Desktop/Folder/Data_Analysis/20034291//*//*//ANDOR1_*.tif"

    >>>fileReading(file_path)
        [[data1], ..., [dataN]]

    >>>fileReading(file_path)
        [[data1], ..., [dataN]], (220, 1024, 1024)

    Notes:
        - All images must be of same dimension. If images of different shapes needed, either group them by shape and run this function for
            each file shape or create it manually by reading files into a numpy array with dtype = object. Be careful as this can lead to
            later problems
        - Review "*" notation to read in all files of a singular type. Expedites the process.

    """
    files = glob.glob(file_path)  # read in file name
    if not files:
        FileNotFoundError(f"file_path input does not contain any TIFF files: {file_path}")
    data_array = tf.imread(files)  # construct array containing files
    if shape and file_names:
        return data_array, data_array.shape, files
    elif shape:
        return data_array, data_array.shape
    elif file_names:
        return data_array, files
    else:
        return data_array


def stagePosition(file_names, idx_start, idx_end, unique_positions=False): #todo may not be needed. Check
    """
    Finds the stage position in the string for every file name in the list of all TIFF files. Requires the user to find the
    index of the first digit and last digit of a single file and assumes all files are formatted identically. It uses these indices
    to retrieve the digits from the file names and stores them in a list. All files must have the exact same number of characters before
    the stage position in the name of the file. If this is an issue, group the file names by preceding characters and run this function
    on each list of files.

    If unique_positions = False, it returns the stage positions as a numpy array. If unique_positions = True, it returns a tuple
                        containing an array of the stage positions, an array of the unique stage positions, and an array containing
                        the indices of the unique stage positions in the original array.
    Arguments:

    file_names (list containing strings): List of file names
    idxStart (int): the index of the first digit of the stage position in the file name
    idxEnd (int): the index of the last digit of the stage position in the file name
    unique_positions

    Returns: stage_pos, uniq_stage (optional), uniq_stage_idx (optional)

    stage_pos (array): Default. A numpy array containing the stage positions of the file. The index of each stage position corresponds to
                            the index of the file name in file_names.
    uniq_stage (array): Optional. A numpy array containing the unique stage positions listed in ascending order.
    uniq_stage_idx (array): Optional. A numpy array containing the indices of the unique stage positions in the original input array.

    Examples:

    file_names = ['image001_10.tif', 'image002_20.tif', 'image004_40.tif', 'image003_30.tif', 'image004_40.tif']
    >>> stagePosition(file_names, 9, 11)
        [10. 20. 40. 30. 40.]

    >>> stagePosition(file_names, 9, 11, unique_positions = True)
        (array([10., 20., 40., 30., 40.]), array([10., 20., 30., 40.]), array([0, 1, 3, 2]))

    """
    try:
        try:
            stage_pos = [np.float64(file_name[idx_start:idx_end]) for file_name in file_names]
            stage_pos = np.array(stage_pos)
            if unique_positions == True:
                uniq_stage, uniq_stage_idx = np.unique(stage_pos, return_index=True)
                return stage_pos, uniq_stage, uniq_stage_idx
            else:
                return stage_pos
            return stagePos
        except ValueError:
            raise ValueError(
                """Failed to convert a file name to a float. Make sure that index positions are correct for all files in file_names.""")
    except IndexError:
        raise ValueError(
            "Invalid index values. Make sure the index values are within the range of the file name strings.") #tod


def get_counts(data_array, plot=False): # todo clean and update
    """
    Generates the counts from the given data by summing over the array elements. Returns 2d array of the same dimension as the
    input images.

    Arguments:

    data_array (numpy.ndarray): Numpy data array containing the diffraction images.
    plot (bool, optional): If set to true, generates a graph of the counts data.

    Returns:
    counts (numpy.ndarray): One dimensional numpy array containing the data after summing over each array element.

    Example:

    data = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    >>>countData(data)
        array([21, 51])
    """
    counts = np.sum(data_array, axis=(1, 2))
    if len(data_array) == 0:
        raise ValueError("Input data_array is empty.")
    if data_array.ndim != 3:
        raise ValueError("Input data_array is not 3 dimensional.")
    if plot == True:
        plt.plot(np.arange(len(data_array[:, 0, 0])), counts)
        plt.show()
    return counts


def get_image_details(file_names, sort=True, plot=False, filter_data=False): #looks pretty good for now todo look into optional arguments
    """
    Returns the data loaded from the tif files with a floor subtracted based on the median of the corner. Also returns arrays with the stage
    positions, the file order number, and the number of counts per image.

    Arguments:
    data_array (numpy.ndarray): Numpy data array containing the diffraction images.
    file_names = ['image001_10.tif', 'image002_20.tif', 'image004_40.tif', 'image003_30.tif', 'image004_40.tif']

    Optional arguments:
    sort (boolean): default is set to True. This arguments sorts the data based on when it was saved (i.e. file number)
    plot (boolean): default is set to False. When True, a plot of the data, log(data), and histogram of counts is shown
    filter_data (boolean): default is set to False. When True, code prompts you for a minimum and maximum value then
        returns only the information from files within this range

    Returns:
    data_array (ndarray): Array of N x 1024 x 1024 where N is the length of tile file_names list. Generated by using tifffile as tf.
    stage_pos (array): Default. A numpy array containing the stage positions of the file. The index of each stage position corresponds to
                            the index of the file name in file_names.
    file_order (array): Returns the image number located in the file name. Reflects the order with which the images are taken.
    counts(ndarray): One dimensional numpy array of length N containing the data after summing over each array element.

    """
    data_array = tf.imread(file_names)  # construct array containing files

    try:
        stage_pos = []
        file_order = []
        try:
            # stage_pos = [np.float64(file_name[idx_start:idx_end]) for file_name in file_names]
            # stage_pos = np.array(stage_pos)
            for file in file_names:
                string = list(map(str, file.split("\\"))) # Note standard slash usage for windows todo might need to test
                folder_number = string[-3][-3:]
                string = list(map(str, string[-1].split("-")))
                file_number = int(folder_number + string[1])
                file_order.append(int(file_number))
                string = list(map(str, string[-1].split("_")))
                stage_pos.append(float(string[0]))
        except ValueError:
            raise ValueError("""Failed to convert a file name to a float. Make sure that index positions are correct for all files in file_names. 
            Also check separators""")
    except IndexError:
        raise ValueError(
            "Invalid index values. Make sure the index values are within the range of the file name strings.")

    stage_pos = np.array(stage_pos)
    file_order = np.array(file_order)
    counts = get_counts(data_array)

    if sort == True:
        idx_sort = np.argsort(file_order)
        file_order = file_order[idx_sort]
        data_array = data_array[idx_sort]
        stage_pos = stage_pos[idx_sort]
        counts = counts[idx_sort]

    if plot == True:
        test = data_array[0]
        plt.figure(figsize=[12,10])
        plt.subplot(1, 3, 1);
        plt.imshow(test, cmap='jet');
        plt.xlabel('Pixel');
        plt.ylabel('Pixel');
        plt.title('Linear Scale(data)')

        plt.subplot(1, 3, 2);
        plt.imshow(np.log(test), cmap='jet');
        plt.xlabel('Pixel');
        plt.ylabel('Pixel');
        plt.title('Log Scale(data)')

        plt.subplot(1, 3, 3);
        plt.hist(test.reshape(-1), bins=30, edgecolor="r", histtype="bar", alpha=0.5)
        plt.xlabel('Pixel Intensity');
        plt.ylabel('Pixel Number');
        plt.title('Hist of the pixel intensity(data)');
        plt.yscale('log')
        plt.tight_layout()
        plt.show()

    if filterdata == True:
        min_val = int(input("Enter minimum file number: "))
        max_val = int(input("Enter maximum file number: "))
        try:
            good_range = np.arange(min_val, max_val, 1)
            data_array = data_array[good_range]
            stage_pos = stage_pos[good_range]
            counts = counts[good_range]
            file_order = file_order[good_range]
        except:
            print("Max value is larger than the size of the data range, returning all data")


    return data_array, stage_pos, file_order, counts


def get_image_details_slac(file_names, sort=True): # todo Add image option
    """
    WORKS FOR CURRENT DATA COLLECTION
    Returns the data loaded from the tif files with a floor subtracted based on the median of the corner. Also returns arrays with the stage positions, the
    file order number, and the number of counts per image.

    Arguments:
    data_array (numpy.ndarray): Numpy data array containing the diffraction images.
    file_names = ['image001_10.tif', 'image002_20.tif', 'image004_40.tif', 'image003_30.tif', 'image004_40.tif']

    Returns:
    data_array (ndarray): Array of N x 1024 x 1024 where N is the length of tile file_names list. Generated by using tifffile as tf.
    stage_pos (array): Default. A numpy array containing the stage positions of the file. The index of each stage position corresponds to
                            the index of the file name in file_names.
    file_order (array): Returns the image number located in the file name. Reflects the order with which the images are taken.
    counts(ndarray): One dimensional numpy array of length N containing the data after summing over each array element.

    """
    data_array = tf.imread(file_names)  # construct array containing files

    try:
        stage_pos = []
        file_order = []
        try:
            # stage_pos = [np.float64(file_name[idx_start:idx_end]) for file_name in file_names]
            # stage_pos = np.array(stage_pos)
            for file in file_names:
                string = list(map(str, file.split("/")))
                string = list(map(str, string[-1].split("_")))
                file_order.append(int(string[2]))
                stage_pos.append(float(string[3]))
        except ValueError:
            raise ValueError(
                """Failed to convert a file name to a float. Make sure that index positions are correct for all files in file_names. Also check separators""")
    except IndexError:
        raise ValueError(
            "Invalid index values. Make sure the index values are within the range of the file name strings.")

    stage_pos = np.array(stage_pos)
    file_order = np.array(file_order)
    counts = get_counts(data_array)

    if sort == True:
        idx_sort = np.argsort(file_order)
        file_order = file_order[idx_sort]
        data_array = data_array[idx_sort]
        stage_pos = stage_pos[idx_sort]
        counts = counts[idx_sort]

    return data_array, stage_pos, file_order, counts


def get_image_details_keV(file_names, sort=False):
    """
    Returns the data loaded from the tif files with a floor subtracted based on the median of the corner. Also returns arrays with the stage
    positions, the file order number, and the number of counts per image.

    Arguments:
    data_array (numpy.ndarray): Numpy data array containing the diffraction images.
    file_names = ['image001_10.tif', 'image002_20.tif', 'image004_40.tif', 'image003_30.tif', 'image004_40.tif']

    Returns:
    data_array (ndarray): Array of N x 1024 x 1024 where N is the length of tile file_names list. Generated by using tifffile as tf.
    stage_pos (array): Default. A numpy array containing the stage positions of the file. The index of each stage position corresponds to
                            the index of the file name in file_names.
    file_order (array): Returns the image number located in the file name. Reflects the order with which the images are taken.
    counts(ndarray): One dimensional numpy array of length N containing the data after summing over each array element.

    """
    data_array = tf.imread(file_names)  # construct array containing files

    try:
        ir_stage_pos = []
        uv_stage_pos = []
        file_order = []
        current = []
        try:
            for file in file_names:
                string = list(map(str, file.split("/")))
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
    counts = get_counts(data_array)

    if sort == True:
        temp_idx = sort_files(file_order, ir_stage_pos, uv_stage_pos)
        data_array = data_array[temp_idx]
        ir_stage_pos = ir_stage_pos[temp_idx]
        uv_stage_pos = uv_stage_pos[temp_idx]
        file_order = file_order[temp_idx]
        current = current[temp_idx]
        counts = counts[temp_idx]

    return data_array, ir_stage_pos, uv_stage_pos, file_order, counts, current


def sort_files(file_order, ir_stage_pos, uv_stage_pos):
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
        # file_numbers = file_order[np.where(stage_positions==uni_stage[i])[0]];
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

def clean_and_filter(data_array):
    """Takes in a 3D array of data and applies the scipy.ndimage.median_filter on them. Need to consider the box size for these filters.
        Additionally, issues can arise if the data array has nan values.

        Returns: filtered data array of the same size as the input array"""

    print("Cleaning data")
    clean_data_array = []
    for data in data_array:
        new_data = data
        filt_data = mediand_filter(new_data, size=3)
        clean_data_array.append(np.array(filt_data))

    return clean_data_array


def _clean_and_filter(data_array_1d):
    """Takes in a large 3D array of data and applies the scipy.ndimage.median_filter on them in parallel processing.
    Need to consider the box size for these filters. Additionally, issues can arise if the data array has nan values. Eventually would like to add
    A baseline subtraction to this code.

    Returns: filtered data array of the same size as the input array"""

    # corners = (np.median(data_array_1d[-50:, -50:]), np.median(data_array_1d[-50:, :50]), np.median(data_array_1d[:50, -50:]),
    # np.median(data_array_1d[:50, :50]))
    # floor = float(np.mean(corners))
    new_data = data_array_1d
    filt_data = median_filter(new_data, size=3)

    return filt_data


def clean_all(data_array):
    """Takes in a large 3D array of data and applies the scipy.ndimage.median_filter on them in parallel processing using the hidden function
    _clean_and_filter. Need to consider the box size for these filters. Additionally, issues can arise if the data array has nan values.
    Eventually would like to add a baseline subtraction to this code.

    Returns: filtered data array of the same size as the input array"""

    print('Cleaning all data with concurrent.futures.ProcessPoolExecutor')
    filtered_data = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(_clean_and_filter, data_array)

    for result in results:
        filtered_data.append(result)

    filtered_data = np.array(filtered_data)
    print(filtered_data.shape)
    print("Finished cleaning!!")
    return filtered_data


def rmv_xrays_all(data_array):
    """ Requires global variables for the mean and standard deviation. Filters out any pixels that are more than 4 times the standard deviation of
    the average pixel value by running the hidden function _remove_xrays in parallel. Use cleanMean if not a large dataset.

    Returns: xray removed data sets of the same shape."""
    print("Removing xrays from all data")
    clean_data = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(_remove_xrays, data_array)
    for result in results:
        clean_data.append(result)
    pct_rmv = []
    for i in range(len(clean_data)):
        no_rmv = sum(sum(clean_data[i].mask))
        pct_rmv.append(no_rmv / (1024 * 1024) * 100)
    clean_data = np.array(clean_data)
    print(clean_data.shape)

    pct_rmv = np.array(pct_rmv)
    plt.figure()
    plt.plot(pct_rmv)
    plt.title("Percent Pixels Removed")
    plt.xlabel("Image Number")
    plt.ylabel("Percent")
    plt.show()

    return clean_data


def _remove_xrays(data_array_1d):
    upper_threshold = mean_data + 4 * std_data
    clean_data = ma.masked_greater_equal(data_array_1d, upper_threshold)
    return clean_data


def cleanMean(data_array, std=4, return_clean_data=True):
    """
    Takes in a data array and calculate the mean and standard deviation at each index across all images. Then applies a filter
    to the data that masks all values (replaces outliers with nan's) outside a given number of standard deviations. After, the
    mean is taken, returning a 2 dimensional array with the mean data of non-outlier entries across all images.

    If return_clean_data is set to True, the cleaned data is returned as a 3d array without having the mean taken.

    Arguments:

    data_array (numpy.ndarray): Data array containing diffraction image data.
    std (int or float): Number of standard deviations from the mean allowed. Values outside this number of standard deviations
                        are masked as nan's. Set to 3 by default.

    Returns:

    clean_mean_data (numpy.ndarray): Default. Returns a 2 dimensional array containing the mean values of the cleaned data.
    clean_data (numpy.ndarray): Other option. Returns the raw 3 dimensional array containing the cleaned data.

    Examples:

    example_array = np.array([[[8.,2],[1,2]], [[1.,2],[1,2]], [[1.,2],[1,2]], [[1., 2], [1,2]]])

    >>> cleanMean(example_array, std=1)
        np.array([[1.,2],[1.,2]])

    >>> cleanMean(example_array, std=1, return_clean_data = True)
        np.array([[[np.nan, 2],[1,2]], [[1.,2],[1,2]], [[1.,2],[1,2]], [[1.,2],[1,2]]])
    """
    if len(data_array) == 0:
        raise ValueError("Input data array is empty.")
    if data_array.ndim != 3:
        raise ValueError("Input data array is not 3 dimensional.")
    if std <= 0:
        raise ValueError("Number of standard deviations (std) must be non-negative.")
    if not isinstance(return_clean_data, bool):
        raise TypeError("return_clean_data must be a boolean value.")

    mean = np.mean(data_array, axis=0)
    stDev = np.std(data_array, axis=0)
    upper_threshold = mean + std * stDev
    # lower_threshold = mean - std*stDev
    clean_data = ma.masked_greater_equal(data_array, upper_threshold)
    # clean_data = ma.masked_outside(data_array, lower_threshold, upper_threshold)
    pct_rmv = []
    for i in range(len(clean_data)):
        no_rmv = sum(sum(clean_data[i].mask))
        pct_rmv.append(no_rmv / (1024 * 1024) * 100)

    pct_rmv = np.array(pct_rmv)
    plt.figure()
    plt.plot(pct_rmv)
    plt.title("Percent Pixels Removed")
    plt.xlabel("Image Number")
    plt.ylabel("Percent")
    plt.show()
    if return_clean_data == True:
        return clean_data
    else:
        clean_mean_data = np.mean(clean_data, axis=(0))
        return clean_mean_data


def medianFilter(data_array, center_top_left_corner, center_border_length, med_filter_range=3):
    """
    Takes in a data array and applies scipy.signal's median filter. Then replaces the boundary and center values with the
    original values from the input array as to not lose precision in these parts.

    Arguments:

    data_array (np.array): 2d Numpy array containing diffraction image data.
    center_top_left_corner (tuple): Tuple containing the row index (integer) and column index (integer) of the top left corner
                                    (lowest_index_row, lowest_index_column)
    center_border_length (int): Length of one side of the square. Must be an integer.
    med_filter_range (int): Must be odd. Initially set to 3. Shape of the square array for median filtering. Using an odd values makes it so that
    the median filter is centered around the individual point.

    Returns:

    med_filt_data (2d-array): Array containing the median filtered data.

    """
    if not isinstance(data_array, np.ndarray):
        raise ValueError("Input data_array must be a 2D numpy array.")
    if not isinstance(center_top_left_corner, tuple) or len(center_top_left_corner) != 2:
        raise ValueError("Center top left corner must be a tuple of length 2 that contains two integers.")
    if not isinstance(center_border_length, int):
        raise ValueError("center_border_length must be an integer.")
    # if not isinstance(center_top_left_corner[0] + center_border_length < data_array[0,-1]
    if not (0 <= center_top_left_corner[0] < data_array.shape[0] and
            0 <= center_top_left_corner[1] < data_array.shape[1] and
            center_top_left_corner[0] + center_border_length <= data_array.shape[0] and
            center_top_left_corner[1] + center_border_length <= data_array.shape[1]):
        raise ValueError("""center_top_left_corner is out of bounds or adding center_border_length goes beyond the array.
                     Check that the tuple values are positive integers within the bounds of the data array and that 
                     adding the border length does not result in a value beyond the size of the array.""")
    if not isinstance(med_filter_range, int) and med_filter_range % 2 == 1:
        raise ValueError("med_filter_range must be an odd integer.")

    med_filt_data = ss.medfilt2d(data_array, med_filter_range)
    med_filt_data[0:med_filter_range // 2, :] = data_array[0:med_filter_range // 2, :]
    med_filt_data[-(med_filter_range // 2):0, :] = data_array[-(med_filter_range // 2):0, :]
    med_filt_data[:, 0:med_filter_range // 2] = data_array[:, 0:med_filter_range // 2]
    med_filt_data[:, -(med_filter_range // 2):] = data_array[:, -(med_filter_range // 2):]
    row_s, col_s = center_top_left_corner
    row_e, col_e = row_s + center_border_length, col_s + center_border_length
    med_filt_data[row_s:row_e, col_s:col_e] = data_array[row_s:row_e, col_s:col_e]
    return med_filt_data


def backgroundNoise(data_array, bkg_range=20, remove_noise=False):
    """
    Takes in a 2d data array (using the mean array is recommended) and calculatees the means of the corners. Linearly interpolates values across 2d
    array to generate of background noise values using pandas.DataFrame.interpolate. Returns a two dimensional numpy array with the linearly
    interpolated background noise.

    Arguments:

    data_array (2d np.ndarray): Data array used to generate the corner values of the background noise.
    bkg_range (int): Side length of square in each corner used for generating mean value. Initially set to 20.
    remove_noise (Bool): If set to true, generated background values are subtracted from the initial input array. Returns cleaned data.

    Returns:

    bkg_data (2d np.ndarray): Data array containing the linearly interpolated background noise for the image. If remove_noise = True,
                                returned data has background noise removed from original input.
    """
    if not isinstance(data_array, np.ndarray):
        raise ValueError("Input data_array must be a numpy array.")
    if not isinstance(bkg_range, int) and bkg_range > 0:
        raise ValueError("bkg_range must be an integer > 0.")
    if not isinstance(remove_noise, bool):
        raise ValueError("remove_noise must be a boolean.")
    if not (2 * bkg_range < len(data_array[0, :]) and
            2 * bkg_range < len(data_array[:, 0])):
        raise ValueError("2 * bkg-range must be less than both the number of rows and the number of columns.")

    empty_array = np.empty(np.shape(data_array))
    empty_array = (ma.masked_array(empty_array, mask=True))
    empty_array[0, 0] = np.mean(data_array[0:20, 0:20])
    empty_array[-1, 0] = np.mean(data_array[-20:, 0:20])
    empty_array[0, -1] = np.mean(data_array[0:20, -20:])
    empty_array[-1, -1] = np.mean(data_array[-20:, -20:])
    empty_array = pd.DataFrame(empty_array).interpolate(axis=0)
    empty_array = pd.DataFrame(empty_array).interpolate(axis=1)
    bkg_data = pd.DataFrame.to_numpy(empty_array)
    if remove_noise == True:
        return data_array - bkg_data
    else:
        return bkg_data


def gaussian_filter_2d(data_array, sig=1):
    """Applies the scipy.ndimage.gaussian_filter() on the 2D data array. Returns filtered images"""
    gf_filtered = gaussian_filter(data_array, sig)
    return gf_filtered


def normalize_to_baseline(data, min_val=200, max_val=300):
    data[:, :25] = np.nan
    data_mean = np.nanmean(data, axis=0)
    norm_factor = np.nansum(data_mean[min_val:max_val])
    data_norm = []
    for i in range(len(data)):
        offset = np.nansum(data[i, min_val:max_val])
        norm = data[i] * (norm_factor / offset)
        data_norm.append(norm)

    data_norm = np.array(data_norm)
    return data_norm


def power_fit(data_array, x_vals, return_baseline=False):
    if len(data_array.shape) == 2:
        baseline2d = []
        for i in range(len(data_array)):
            temp_data = np.copy(data_array[i])
            idx_nan = ~np.isnan(temp_data)
            coeff = np.polyfit(x_vals[idx_nan], temp_data[idx_nan], 2)
            baseline = np.polyval(coeff, x_vals)
            baseline2d.append(baseline)

        baseline2d = np.array(baseline2d)
        corrected_data = data_array - baseline2d

    elif len(data_array.shape) == 1:
        temp_data = data_array
        idx_nan = ~ np.isnan(temp_data)
        coeff = np.polyfit(x_vals[idx_nan], temp_data[idx_nan], 2)
        baseline2d = np.polyval(coeff, x_vals)

        corrected_data = data_array - baseline2d
    else:
        print("Data Array must be 1D or 2D array")
    if return_baseline == True:
        return corrected_data, baseline2d
    else:
        return corrected_data


def fit_high_s(data_array, x_vals, s_range, return_baseline=False):
    if len(data_array.shape) == 2:
        corrected_data = []
        baseline = []
        for i in range(len(data_array)):
            temp_data = data_array[i]
            coeff = np.polyfit(x_vals[s_range], temp_data[s_range], 2)
            line = np.polyval(coeff, x_vals[s_range])
            baseline.append(line)
            data_array[i, s_range] = temp_data[s_range] - line
            corrected_data.append(data_array[i])

    elif len(data_array.shape) == 1:
        coeff = np.polyfit(x_vals[s_range], data_array[s_range], 1)
        baseline = np.polyval(coeff, x_vals)

        corrected_data = data_array - baseline
    else:
        print("Data Array must be 1D or 2D array")

    corrected_data = np.array(corrected_data)

    if return_baseline == True:
        return corrected_data, baseline
    else:
        return corrected_data


### Masking and Center Finding Functions

def detectorMask(data_array, hole_center, inner_radius, outer_radius, plot_image=True):
    """
    Takes in a 2d data array and applies a circular (donut shaped) detector mask to it, replacing the masked values with np.nan's.
    Returns the masked, 2d data array.

    Arguments:

    data_array (2d np.ndarray): 2d data array to be masked.
    hole_center (tuple): Tuple containing the x and y coordinates of the center of the image, each one of which an int.
    inner_radius (float): Inner radius. Values within the radius of this drawn from the center are masked.
    outer_radius (float): Outer radius of the donut. Values outside the radius of this drawn from the center are masked.
    plot_image (bool, optional): If True, plots the masked image. Default is False.

    Returns:
    ring_data (2d np.ndarray): Data array with the circular detector mask applied.
    """
    if not isinstance(hole_center, tuple) or len(hole_center) != 2:
        raise ValueError(
            "hole_center must be a tuple of length 2 containing the x and y coordinates of the hole center.")
    if not (isinstance(inner_radius, (int, float)) and inner_radius > 0):
        raise ValueError("inner_radius must be a positive float or integer.")
    if not (isinstance(outer_radius, (int, float)) and outer_radius > 0):
        raise ValueError("outer_radius must be a positive float or integer.")
    if inner_radius >= outer_radius:
        raise ValueError("inner_radius must be smaller than outer_radius.")

    hole_cx, hole_cy = hole_center
    x_idx, y_idx = np.meshgrid(np.arange(data_array.shape[2]), np.arange(data_array.shape[1]))
    dist = np.sqrt(((x_idx - hole_cx) ** 2 + (y_idx - hole_cy) ** 2))
    mask = np.logical_and(dist <= outer_radius, dist >= inner_radius)
    ring_data = []
    for i in range(len(data_array)):
        data = np.where(mask, data_array[i], np.nan)
        ring_data.append(data)

    ring_data = np.array(ring_data)
    if plot_image == True:
        img3 = plt.imshow(ring_data[0])
        plt.colorbar(img3)
    return (ring_data)


def mask_hole(I, fit_bor, hole_bor, value_bor, show='yes'):
    [X_fit, Y_fit] = np.where((I[fit_bor[0][0]:fit_bor[0][1], fit_bor[1][0]:fit_bor[1][1]] < 1.1 * value_bor) &
                              (I[fit_bor[0][0]:fit_bor[0][1], fit_bor[1][0]:fit_bor[1][1]] > 0.9 * value_bor))
    if show == 'yes':
        plt.scatter(Y_fit + fit_bor[1][0], X_fit + fit_bor[0][0])
    center_hole, r_hole = fit_circle([X_fit + fit_bor[0][0], Y_fit + fit_bor[1][0]])

    mask = np.ones((hole_bor[0][1] - hole_bor[0][0], hole_bor[1][1] - hole_bor[1][0])).astype(float)
    for xi in range(len(mask)):
        for yi in range(len(mask[xi])):
            if (xi - center_hole[0] + hole_bor[0][0]) ** 2 + (yi - center_hole[1] + hole_bor[1][0]) ** 2 <= (
                    r_hole + 3) ** 2:
                mask[xi, yi] = np.nan
    I[hole_bor[0][0]:hole_bor[0][1], hole_bor[1][0]:hole_bor[1][1]] = I[hole_bor[0][0]:hole_bor[0][1],
                                                                      hole_bor[1][0]:hole_bor[1][1]] * mask

    return mask, center_hole, r_hole


def find_beam_center(I, center=[500, 500], r=200, printr2='no', recursiontime=0):
    recursiontime += 1
    # up down right left,r away pixles average
    # fit_value=average([I[center[0]+r][center[1]],I[center[0]-r][center[1]],I[center[0]][center[1]+r],I[center[0]][center[1]-r]])
    fit_value = np.average([I[round(center[0]) + r][round(center[1])], I[round(center[0]) - r][round(center[1])],
                            I[round(center[0])][round(center[1]) + r], I[round(center[0])][round(center[1]) - r]])

    [X_f, Y_f] = np.where((I > 0.999 * fit_value) & (I < 1.001 * fit_value))

    a = len(X_f)
    i = 0
    # delete fit_points which are too far away from fit_circle, range from 0.5r to 1.5r
    while (i < a):
        ri2 = (X_f[i] - center[0]) ** 2 + (Y_f[i] - center[1]) ** 2
        if (ri2 > (1.5 * r) ** 2) or (ri2 < (0.5 * r) ** 2):
            X_f = np.delete(X_f, i)
            Y_f = np.delete(Y_f, i)
            i -= 1
            a -= 1
        i += 1

    center_new, r_new = fit_circle([X_f, Y_f], printr2)

    if r_new == 0:
        return [0, 0]
    elif ((center[0] - center_new[0]) ** 2 + (center[1] - center_new[1]) ** 2) <= 1:
        # new center pretty close to old center
        return center_new
    elif recursiontime >= 10:
        return [0, 0]
    else:
        # else: iterate
        return find_beam_center(I, center_new, r_new, recursiontime=recursiontime)


def fit_circle(fit_points, printr2='yes'):
    # circle function: ax+by+c=-(x^2+y^2)

    A = np.empty((len(fit_points[0]), 3))  # Find center for 3 thimes
    B = np.empty(len(fit_points[0]))

    for i in range(len(fit_points[0])):
        B[i] = -(fit_points[0][i] ** 2 + fit_points[1][i] ** 2)
        A[i][0] = fit_points[0][i]
        A[i][1] = fit_points[1][i]
        A[i][2] = 1

    # A[i]=[xi,yi,1], B[i]=-(xi^2+yi^2), par=[a,b,c]
    # namely A*par=B, least square method
    if np.linalg.det(np.dot(A.T, A)) == 0:
        return [], 0
    par = np.dot(np.dot(np.linalg.inv(np.dot(A.T, A)), A.T), B)

    # correlation coeff, if not very close to 1(less than 3 nines), be careful
    if printr2 == 'yes':
        y_ave = np.mean(B)
        r2 = sum((np.dot(A, par) - y_ave) ** 2) / sum((B - y_ave) ** 2)
        print(r2)

    center_new = [(-par[0] / 2), (-par[1] / 2)]  # no-Round the center, not working for the moment
    r_new = round(
        np.sqrt(par[0] ** 2 + par[1] ** 2 - 4 * par[2]) / 2)  # no-round the r range, not working for the moment
    # print('ct found:'+str(center_new))

    # center_new=[round(-par[0]/2),round(-par[1]/2)] # Round the center
    # r_new=round(sqrt(par[0]**2+par[1]**2-4*par[2])/2) # round the r range

    return center_new, r_new


def mask_generator_alg(dat, mask_center, mask_radius, fill_value=np.nan, add_mask=[], add_rectangular=True,
                       showingfigure=False):
    """
    Generate mask to cover unwanted area

    Parameters
    ----------
    dat : 2D array
        Diffraction pattern.
    mask_center : 1D array, tuple, or list that contains only two values
        Center for generating mask cover unscatter electron beam.
    mask_radius : int
        Radius of the mask.
    fill_value : int, float, or nan, optional
        Value that use to fill the area of the mask. The default is np.nan.
    add_mask : list of 3-value-lists, optional
        Additional masks. Input gonna be [[x-center, y-center, radius], [...], ...] The default is [].
    add_rectangular : boolean, optional
        Additional mask with rectangular shape. The default is True.
    showingfigure : boolean, optional
        Show figure of the result of applied masks. The default is False.

    Returns
    -------
    mask : binary 2D array
        Result of all the masks in an image.

    """

    mask = np.ones(dat.shape)
    rows, cols = draw.disk((mask_center[1], mask_center[0]), mask_radius, shape=mask.shape)
    mask[rows, cols] = fill_value

    if len(add_mask) == 0:
        pass
    else:
        for i in add_mask:
            rows, cols = draw.disk((i[1], i[0]), i[2], shape=mask.shape)
            mask[rows, cols] = fill_value

    # retangular mask
    if add_rectangular == True:
        rr, cc = draw.rectangle((0, 590), extent=(500, 40), shape=dat.shape)  # (0,535) for iodobenzene
        mask[rr, cc] = fill_value
        # 515

    if showingfigure == True:
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
        ax1.imshow(dat)
        ax2.imshow(dat * mask)

        ax1.axvline(mask_center[0], linestyle='--', lw=1, color='tab:red')
        ax1.axhline(mask_center[1], linestyle='--', lw=1, color='tab:red')

        ax2.axvline(mask_center[0], linestyle='--', lw=1, color='tab:red')
        ax2.axhline(mask_center[1], linestyle='--', lw=1, color='tab:red')

        ax1.set_title("Reference image", fontsize=20)
        ax2.set_title("Reference image + masks", fontsize=20)

        for ax in (ax1, ax2):
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        plt.tight_layout()
        plt.show()

    return mask


def finding_center_alg(dat, disk_radius=3, showingfigure=False, center_guess=(532, 520), radius_guess=80,
                       title='Reference image', thresh_input=0):
    """
    Algorithm for finding the center of diffraction pattern

    Parameters
    ----------
    dat : 2D array
        Diffraction pattern.
    disk_radius : int, optional
        Generates a flat, disk-shaped footprint. The default is 3.
    showingfigure : boolean, optional
        Show figure of the result of center finding. The default is False.
    center_guess : tuple contains 2 values, optional
        Guessing center position to generate temporary mask. The default is (532, 520).
    radius_guess : int, optional
        Guessing radius of the temporary mask. The default is 80.
    title : str, optional
        Title of the figure. The default is 'Reference image'.

    Returns
    -------
    center_x : int
        Center value on x axis.
    center_y : int
        Center value of y axis.
    radius : int
        Radius of ring used for finding center.

    """

    if thresh_input == 0:
        thresh = threshold_otsu(dat)
    else:
        thresh = thresh_input

    cxt, cyt = [], []
    for th in [1]:
        thresh *= th
        mask_temp = mask_generator_alg(dat, center_guess, radius_guess * th, fill_value=False, add_mask=[],
                                       add_rectangular=False, showingfigure=False)
        mask_temp = util.invert(mask_temp.astype(bool))
        bw = closing(dat > thresh, disk(
            disk_radius))  # Return grayscale morphological closing of an image. Square(): generate the footprint to close the gap between data points
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

    if showingfigure == True:
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(20, 10))
        ax1.imshow(dat)
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
        ax1.set_title(title, fontsize=20)
        ax2.set_title("Center [X = " + str(center_x) + ", Y = " + str(center_y) + "]", fontsize=20)
        ax3.set_title("Binary image", fontsize=20)

        ax1.axvline(center_x, linestyle='--', lw=1, color='tab:red')
        ax1.axhline(center_y, linestyle='--', lw=1, color='tab:red')

        ax2.axvline(center_x, linestyle='--', lw=2, color='tab:red')
        ax2.axhline(center_y, linestyle='--', lw=2, color='tab:red')

        plt.tight_layout()
        plt.show()

    return center_x, center_y, radius, thresh


### Azimuthal Averaging and Radial Outlier Removal Functions

def azimuthal_integration(data_array, center, correction_factor=3):
    """
    Performs azimuthal integration over the data set, removing radial outliers as it performs the integration.
    Function coverts to polar coordinates, the calculates the mean and std at every radial distance (distances are
    previously round to the nearest positive integer and grouped by integer). Returns the azimuthal integration data
    and the azimuthal error.

    Arguments:

    data_array (np.ndarray): Two dimensional numpy array containing the diffraction image.
    center (tuple): Tuple containing the integer coordinates of the image center.
    correction_factor (int, optional): Number of standard deviations set as threshold. Preset to 3.

    Return: (azi_dat, azi_err)

    azi_dat (np.ndarray): Array containing the integrated data.
    azi_err (np.ndarray): Array containing the error of the integrated data.

    Example:

    sample_array = np.random.rand(100,100)
    center = (40,40)

    >>> azimuthal_integration(sample_array, center, correction_factor = 2)
        azi_dat, azi_err
    """

    x_mat, y_mat = np.meshgrid(np.arange(0, data_array.shape[1], 1) - center[1],
                               np.arange(0, data_array.shape[0], 1) - center[0])
    # rmat, _ = cart2pol(xmat, ymat)

    r_mat = np.sqrt(x_mat ** 2 + y_mat ** 2)
    # phi = np.arctan2(y_mat,x_mat)

    rmat = np.around(r_mat)
    data_array = data_array.astype(float)
    data_array[data_array == 0] = np.nan
    x_length = int(np.amax([np.amax(abs(x_mat)), np.amax(abs(y_mat))]))
    azi_dat = np.zeros(x_length)
    azi_err = np.copy(azi_dat)
    for i in range(x_length):
        roi = data_array[r_mat == int(i + 1)].astype(float)
        # print(roi)
        if len(roi) == 0:
            break
        std = np.nanstd(roi)
        idxx = np.logical_and(roi <= np.nanmean(roi) + correction_factor * std,
                              roi >= np.nanmean(roi) - correction_factor * std)
        idxx = util.invert(idxx)
        roi[idxx] = np.nan

        azi_dat[i] = np.nanmean(roi)
        azi_err[i] = np.nanstd(roi) / np.sqrt(abs(np.nansum(roi)))
    return azi_dat, azi_err


def azimuthal_avg_correct(args):
    """Returns the azimuthal average of a diffraction image based on the radial distance of the x, y positions in the image."""
    data, x, y = args
    r_max = len(x)
    I = np.empty(r_max)
    for ri in range(r_max):
        I_r = data[x[ri], y[ri]]
        ave = np.nanmean(I_r)
        sigma = np.nanstd(I_r)
        I_r[np.abs(I_r - ave) >= 5 * sigma] = np.nan
        I[ri] = np.nanmean(I_r)
    return I


def get_azimuthal_average(data, x, y):
    """Runs the azimuthal average function in parallel for large data sets."""
    p = ThreadPool(3)
    I = p.map(azimuthal_avg_correct, [(data_i, x, y) for data_i in data])
    return np.array(I)


def get_average(array):
    average = np.nanmean(array)
    stdev = np.nanstd(array)
    return average, stdev


def get_radial_distribution(image):
    """Get average radial intensity after eliminating outliers"""
    radial_range = int(max(X_CENTER, Y_CENTER, GRID_SIZE - X_CENTER,
                           GRID_SIZE - Y_CENTER))  # half the size of the image i.e. 512 for 1024 by 1024 image
    print(radial_range)
    radial_values = [[] for r in range(radial_range)]
    # First get all pixel values for each radial position as an array that we can average over later
    # It will look like
    #   [ (R = 0 values) [...],
    #     (R = 1 values) [...],
    #     (R = 2 values) [...], etc ]

    for Y in range(GRID_SIZE):
        for X in range(GRID_SIZE):
            R = sqrt((X - X_CENTER) ** 2 + (Y - Y_CENTER) ** 2)
            R_index = int(round(R))
            if R_index < len(radial_values):
                radial_values[R_index].append(image[X][Y])
    # So now we have an array where, for each R value, we have a sub-array of each pixel intensity at that R
    # We want to first calculate the average and standard deviation
    # Then eliminate all outliers that are >3sigma away from the average
    # Then re-calculate the average with all outliers removed
    radial_averages, radial_stdevs = [], []
    for r_values in radial_values:
        avg, stdev = get_average(r_values)
        # Create new list with outliers removed
        new_r_values = [r if abs(avg - r) <= 5 * stdev else np.nan for r in r_values]
        new_avg, new_stdev = get_average(new_r_values)
        radial_averages.append(new_avg)
        radial_stdevs.append(new_stdev)

    return radial_averages, radial_stdevs


def remove_radial_outliers(image):
    """After finding average radial values, replace all outlier pixels with the average value at that radius"""
    radial_avgs, radial_stdevs = get_radial_distribution(image)

    new_image = [[0 for Y in range(GRID_SIZE)] for X in range(GRID_SIZE)]  # image of all zeros

    for Y in range(GRID_SIZE):
        for X in range(GRID_SIZE):
            R = sqrt((X - X_CENTER) ** 2 + (Y - Y_CENTER) ** 2)
            # Calculate the interpolated value for the average and stdev at this R
            # (Same thing I do in simulate_image but now I'm calculating the weighted average of the
            # average intensity values so the language is a little confusing)
            interpolated_avg, interpolated_stdev = 0, 0  # Will be filled in with the interpolated values

            R_lower = floor(R)
            R_upper = R_lower + 1
            if R_lower >= len(radial_avgs):
                interpolated_avg = 0  # R value is out of bounds, default to 0
                interpolated_stdev = 0
            elif R_upper >= len(radial_avgs):
                interpolated_avg = radial_avgs[R_lower]  # R value is just outside of bounds, default to edge value
                interpolated_stdev = radial_stdevs[R_lower]
            else:
                # Calculate weighted average
                interpolated_avg = (R - R_lower) * radial_avgs[R_upper] + (R_upper - R) * radial_avgs[R_lower]
                interpolated_stdev = (R - R_lower) * radial_stdevs[R_upper] + (R_upper - R) * radial_stdevs[R_lower]

            if abs(image[X][Y] - interpolated_avg) <= 3 * interpolated_stdev:
                # Value is within acceptance range, do not change
                new_image[X][Y] = image[X][Y]
            else:
                # Value is outside acceptance range, use average value instead
                new_image[X][Y] = interpolated_avg

    return new_image


def remove_radial_outliers_pool(data):
    p = ThreadPool(5)
    I = p.map(remove_radial_outliers, [(data_i) for data_i in data])
    print("FINISHED")
    return np.array(I)


### Simulation Functions
# todo organize and clean!
# path_dcs = '/home/centurion/lheald2/GUED_Analysis/GUED_Analysis/packages/dcs_repository/3.7MeV/'
# # path_dcs = '/sdf/home/l/lheald2/GUED/jupyter_notebook/user_notebooks/dcs_repository/3.7MeV/'
# table = pd.read_csv(path_dcs + 'Periodic_Table.csv')


def import_s():
    """ This functions uses the C.dat files as an example file to generate values of s for the simulation calculations.

    RETURNS:
    s (array) = array of s values which correspond to calculated scattering intensities for each atom.
    """

    qe = 1.602176565e-19
    me = 9.10938291e-31
    c = 299792458
    h = 6.62606957e-34
    E = 3700000 * qe + me * c ** 2  # kinetic energy=3.7MeV
    p = (E ** 2 / c ** 2 - me ** 2 * c ** 2) ** 0.5
    lamb = h / p
    k = 2 * np.pi / lamb  # wave vector of the incident electron

    path = path_dcs + 'C.dat'
    with open(path, 'r') as file:
        a = file.read()
    a0 = a.split('\n')
    theta_deg = np.empty(130)
    for i in range(130):
        a31 = str(a0[31 + i]).split(' ')
        theta_deg[i] = a31[2]

    theta = theta_deg * np.pi / 180
    S = 2 * k * np.sin(0.5 * theta)
    s = np.array(S)
    return s


def read_dat_dcs(atom_no, path_dcs):
    """ Reads in the scattering intensity (form factors) for each atom in the molecule of interest from the .dat files calculated using ELSEPA.

    ARGUMENTS:
    atom_no (int) = maximum atomic number of interest (default value is 55)
    path_dcs (string) = path to the folder containing the .dat files

    RETURNS:
    data (array) = values of the scattering intensities in cm taken from the .dat files.
    """

    atom_sym = no_to_sym(atom_no)
    path = path_dcs + atom_sym + '.dat'
    with open(path, 'r') as file:
        a = file.read()
    a0 = a.split('\n')
    data = np.empty(130)
    for i in range(130):
        a31 = str(a0[31 + i]).split(' ')
        # print(a31)
        data[i] = a31[6]
    # print(data)
    return data ** 0.5  ## returns in cm


def sym_to_no(atom_symbol):
    """ Short cut for getting the atomic number from the atomic symbol.

    ARGUMENTS:
    atom_symbol (string) = atomic symbol

    RETURNS:
    atom_number (int) = atomic number
    """

    n = np.where(table['Symbol'] == atom_symbol)
    atom_number = int(n[0] + 1)
    return atom_number


def no_to_sym(atom_number):
    """ Short cut for getting the atomic symbol from the atomic number.

    ARGUMENTS:
    atom_number (int) = atomic number

    RETURNS:
    atom_symbol (string) = atomic symbol
    """

    atom_symbol = table['Symbol'][atom_number - 1]
    return atom_symbol


def import_DCS(max_at_no=55):
    """ Uses read_dat_dcs to get the form factors for all the atoms available.

    ARGUMENTS:
    max_at_no (int) = maximum atomic number of interest (default value is 55)

    RETURNS:
    f (array) = form factors for all atoms
    """

    f = np.empty((max_at_no + 1, 130))
    for i in range(max_at_no):
        f[i + 1] = read_dat_dcs(i + 1, path_dcs)
    return f


def load_static_mol_coor(path_mol, mol_name, file_type):
    """ Reads in either a .csv or .xyz file containing moleculear coordinates and adds a column containing the atomic number for each atom in
    the molecule. Errors are thrown if an improper file type is chosen or if the .xyz or .csv file needs further formatting.

    ARGUMENTS:

    path_mol (string) = path to the directory of the molecular structure
    mol_name (string) = file name of the structural file used for the simulation
    file_type (string) = either xyz or csv depending on what the file being used is. Determines treatment

    RETURNS:

    coor (array) = N x 5 array where N = # of atoms. Column 0 contains the atomic symbol, columns 1, 2, and 3 contain x, y, and z coordinates
        and column 4 contains the atomic number.
    atom_sum (int) = total number of atoms in the molecule
    """

    filename = path_mol + mol_name + file_type
    if file_type == '.xyz':
        [coor_xyz, atom_sum] = load_xyz_new(filename)
        coor = get_modified_coor_for_xyz(coor_xyz, atom_sum)

    if file_type == '.csv':
        mol_filename = mol_name + '.csv'
        coor_M = pd.read_csv(path_mol + mol_filename)
        coor = np.array(coor_M)
        num = np.array(coor[:, 3])
        atom_sum = int(len(num))
        coor = get_modified_coor_for_csv(coor, atom_sum)

    elif file_type != '.csv' and file_type != '.xyz':
        print('error! Please type in the right molecular coordinate file type, .xyz or .csv')

    return coor, atom_sum


def load_xyz_new(xyz_file):
    """Reads in an .xyz generated from programs such as Gaussian or ORCA.

    ARGUMENTS:
    xyz_file (string) = full path to the .xyz file of interest.

    RETURNS:
    re (array) = coordinate array of N (# of atoms) x 4 shape with column 0 containing atomic symbol, and columns 1, 2, and 3 containing x, y, z
        coordinates
    atom_sum (int) = total number of atoms in the molecule
    """

    file = open(xyz_file, 'r')
    text = file.readlines()
    file.close()
    count = len(text)
    re = []
    for j in range(0, count):
        try:
            string = list(map(str, text[j].split()))
            re.append(string)
        except Exception:
            pass
    atom_sum = re[0]
    atom_sum = int(np.array(atom_sum))
    re = np.array(re[2:])
    return re, atom_sum


def get_modified_coor_for_xyz(re, atom_sum):
    """ Appends a column of atomic numbers to the coordinate array read from the .xyz file

    ARGUMENTS:
    re (array) = coordinate array of N (# of atoms) x 4 shape with column 0 containing atomic symbol, and columns 1, 2, and 3 containing x, y, z
        coordinates
    atom_sum (int) = total number of atoms in the molecule

    RETURNS:
    coor (array) = N x 5 array where N = # of atoms. Column 0 contains the atomic symbol, columns 1, 2, and 3 contain x, y, and z coordinates
        and column 4 contains the atomic number.
    """
    atom_num = [0 for i in range(atom_sum)]
    for i in range(atom_sum):
        atom_num[i] = sym_to_no(re[i][0])

    atom_num = np.array(atom_num)
    atom_num = atom_num[:, np.newaxis]
    coor = np.hstack((re, atom_num))

    return coor


def get_modified_coor_for_csv(coor_csv, atom_sum):
    """ Appends a column of atomic numbers to the coordinate array read from the .csv file

    ARGUMENTS:
    coor_csv (array) = coordinate array of N (# of atoms) x 4 shape with column 0 containing atomic symbol, and columns 1, 2, and 3 containing x,
        y, and z coordinates
    atom_sum (int) = total number of atoms in the molecule

    RETURNS:
    coor (array) = N x 5 array where N = # of atoms. Column 0 contains the atomic symbol, columns 1, 2, and 3 contain x, y, and z coordinates
        and column 4 contains the atomic number.
    """

    atom_num = [0 for i in range(atom_sum)]
    for i in range(atom_sum):
        atom_num[i] = sym_to_no(coor_csv[i, 0])

    atom_num = np.array(atom_num)
    atom_num = atom_num[:, np.newaxis]
    coor = np.hstack((coor_csv, atom_num))

    return coor


def load_time_evolving_xyz(path_mol, mol_name, file_type):
    """Reads in a trajectory .xyz file containing many structures which evolve over time generated from programs such as Gaussian or ORCA. The
        file also contains information on the time points for each structural evolution.

    ARGUMENTS:

    path_mol (string) = path to the directory of the molecular structure
    mol_name (string) = file name of the structural file used for the simulation
    file_type (string) = either xyz or csv depending on what the file being used is.

    RETURNS:
    coor_txyz (array) = array of atom symbol, x, y, z, and atom number for each time step
    atom_sum (int) = total number of atoms in the molecule
    time (array) = time points corresponding to the simulation in fs (??)
    """

    mol_filename = mol_name + '.xyz'
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
        coor1 = get_modified_coor_for_xyz(coor_txyz[i][:][:], atom_sum)
        coor_txyz[i] = coor1

    return np.array(coor_txyz), atom_sum, time


def load_time_evolving_xyz1(path_mol, mol_name, file_type):  # just to load another type of xyz file
    """Reads in a trajectory .xyz file containing many structures which evolve over time generated from programs such as Gaussian or ORCA. The
        file also contains information on the time points for each structural evolution.

    ARGUMENTS:

    path_mol (string) = path to the directory of the molecular structure
    mol_name (string) = file name of the structural file used for the simulation
    file_type (string) = either xyz or csv depending on what the file being used is.

    RETURNS:
    coor_txyz (array) = array of atom symbol, x, y, z, and atom number for each time step
    atom_sum (int) = total number of atoms in the molecule
    time (array) = time points corresponding to the simulation in fs (??)
    """

    mol_filename = mol_name + '.xyz'
    with open(path_mol + mol_filename, 'r') as f:
        a = f.read()

    a0 = a.split('\n')
    atom_sum = int(a0[0])

    time_count = int((len(a0) - 1) / (atom_sum + 2))
    time = [0 for i in range(time_count)]
    print("count = ", time_count)
    coor_txyz = get_3d_matrix(time_count, atom_sum, 4)
    # coor_txyz[time order number][atom type][coordinate xyz]

    m = 0
    n = 0
    o = 0

    for i in range(time_count):
        m = 0
        a1 = str(a0[(atom_sum + 2) * i + 1]).split(' ')
        for j in a1:
            if j == '=':
                m = 1
            if j != '' and j != '=' and j != '1' and m == 1:
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

    for i in range(time_count):
        coor1 = get_modified_coor_for_xyz(coor_txyz[i][:][:], atom_sum)
        coor_txyz[i] = coor1
    return np.array(coor_txyz), atom_sum, time


def get_2d_matrix(x, y):
    # an easy way to set whatever matrix you want
    d = []
    for i in range(x):
        d.append([])
        for j in range(y):
            d[i].append(0)
    return d;


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


def load_freq_xyz(path_mol, mol_name, file_type):
    """Reads in a frequency trajectory .xyz file containing many structures which evolve over time generated from programs such as Gaussian or
        ORCA. The file also contains information on the time points for each structural evolution.

    ARGUMENTS:

    path_mol (string) = path to the directory of the molecular structure
    mol_name (string) = file name of the structural file used for the simulation
    file_type (string) = either xyz or csv depending on what the file being used is.

    RETURNS:
    re (array) = array of atom symbol, x, y, z, and atom number for each time step
    atom_sum (int) = total number of atoms in the molecule
    time (array) = time points corresponding to the simulation in fs
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

    # for i in range(time_count):
    #     coor1=get_modified_coor_for_xyz(coor_txyz[i][:][:],atom_sum)
    #     coor_txyz[i]=coor1
    return re, atom_sum, time


def get_I_from_mol_coor(f, s, s_max, coor, atom_sum):  # f is the form factor array
    # this function is to get scattering intensity from molecular coordinates and atom form factors
    b = range(atom_sum)

    Lm = 125  # Lm confines the maximum of s, for we negelect high angle scattering which has poor signal-noice ratio.
    # You may change this number in order to compute at higher s
    # this Lm should be no larger than 138
    s0 = s[0:Lm] * 1e-10  # this is to change the unit into inverse angstrom
    s1 = np.linspace(0, s_max, 500)

    I = np.zeros(Lm)  # total elastic scattering intensity under the approx of IAM
    I_at = np.zeros(Lm)  # I_atom under IAM
    I_mol = np.zeros(Lm)  # I_molecule under IAM
    R = [0 for i in range(atom_sum ** 2)]
    m = 0
    for i in range(atom_sum):  # the case for single atom scattering, which contributes to I_at
        I_at += np.abs(f[int(coor[i, 4]), 0:Lm]) ** 2

    for i in b:
        for j in b:
            if i != j:  # the case for interatomic interferencing, which contributes to I_mol
                r_ij = ((float(coor[i, 1]) - float(coor[j, 1])) ** 2 + (float(coor[i, 2]) - float(coor[j, 2])) ** 2 + (
                            float(coor[i, 3]) - float(coor[j, 3])) ** 2) ** 0.5 * 1e-10
                # if int(coor[i,4])!=1 and int(coor[j,4])!=1:
                #   R[m]=r_ij
                #  m+=1
                # distance between atom i and j
                I_mol[0] += f[int(coor[i, 4]), 0] * f[int(coor[j, 4]), 0]
                I_mol[1:Lm] += f[int(coor[i, 4]), 1:Lm] * f[int(coor[j, 4]), 1:Lm] * np.sin(s[1:Lm] * r_ij) / s[
                                                                                                              1:Lm] / r_ij
    # y=[1 for i in range(len(R))]
    # y=np.array(y)
    # plt.scatter(np.array(R)*1e12,y,s=5)
    # plt.grid()
    # plt.show()

    I = I_at + I_mol

    I1 = make_interp_spline(s0, I)(s1)
    I_at1 = make_interp_spline(s0, I_at)(s1)
    I_mol1 = make_interp_spline(s0, I_mol)(s1)

    return I1, I_at1, I_mol1, s1


def get_I_atomic(f, s000, s_max, coor, atom_sum):
    I_at_all = []
    s000 = s000 * 1e-10  # in angstroms
    s_new = np.linspace(0, s_max, 500)
    for i in range(atom_sum):
        I_atomic = []
        I_at = 0
        amps = f[int(coor[i, 4])]
        # print(amps)
        interp_amps = interp.interp1d(s000[0:125], amps[0:125])
        amps_new = interp_amps(s_new)
        for k in range(len(amps_new)):
            f_new = amps_new[k]
            I_at = np.abs(f_new) ** 2
            I_atomic.append(float(I_at))
        I_at_all.append(I_atomic)
    I_at = sum(np.array(I_at_all))
    return I_at, s_new


def get_I_molecular(f, s000, s_max, coor, atom_sum):
    x = np.array(coor[:, 1])
    y = np.array(coor[:, 2])
    z = np.array(coor[:, 3])

    s000 = s000 * 1e-10  # convert to angstroms

    s_new = np.linspace(0, s_max, 500)
    I_mol = np.zeros(len(s_new))
    for i in range(atom_sum):
        for j in range(atom_sum):  # Does each atom pair calculation twice
            if i != j:
                r_ij = (float(x[i]) - float(x[j])) ** 2 + (float(y[i]) - float(y[j])) ** 2 + (
                            float(z[i]) - float(z[j])) ** 2
                r_ij = r_ij ** 0.5
                # print(f"bond length between {coor[i, 0]} and {coor[j, 0]} = {r_ij}")
                amps_i = f[int(coor[i, 4])]
                amps_j = f[int(coor[j, 4])]
                interp_amps_i = interp.interp1d(s000[0:125], amps_i[0:125])
                interp_amps_j = interp.interp1d(s000[0:125], amps_j[0:125])
                amps_new_i = interp_amps_i(s_new)
                amps_new_j = interp_amps_j(s_new)
                # print(len(amps_new_j))
                I_mol[0] += f[int(coor[i, 4]), 0] * f[int(coor[j, 4]), 0]
                I_mol[1:len(s_new)] += amps_new_i[1:len(s_new)] * amps_new_j[1:len(s_new)] * np.sin(
                    s_new[1:len(s_new)] * r_ij) / s_new[1:len(s_new)] / r_ij
    return I_mol, s_new


def get_I_from_xyz(f, s000, s_max, coor, atom_sum):
    """USE THIS ONE. OLD ONE INTERPOLATES INCORRECTLY"""

    I_at, s_new = get_I_atomic(f, s000, s_max, coor, atom_sum)
    I_mol, _ = get_I_molecular(f, s000, s_max, coor, atom_sum)
    I = I_at + I_mol
    return I, I_at, I_mol, s_new


def get_I_for_exp_from_mol_coor(f, s, s_exp, coor, atom_sum):
    # slightly different from the function get_I_from_mol_coor
    # this function is to simulate I that matches the s from experiments
    b = range(atom_sum)
    Lm = 125  # Lm should be no larger than 138
    I = np.zeros(Lm)
    I_at = np.zeros(Lm)
    I_mol = np.zeros(Lm)

    for i in range(atom_sum):
        I_at += f[int(coor[i, 4]), 0:Lm] * f[int(coor[i, 4]), 0:Lm]
    for i in b:
        for j in b:
            if i != j:
                r_ij = ((float(coor[i, 1]) - float(coor[j, 1])) ** 2 + (float(coor[i, 2]) - float(coor[j, 2])) ** 2 + (
                            float(coor[i, 3]) - float(coor[j, 3])) ** 2) ** 0.5 * 1e-10
                # distance between atom i and j
                I_mol[0] += f[int(coor[i, 4]), 0] * f[int(coor[j, 4]), 0]
                I_mol[1:Lm] += f[int(coor[i, 4]), 1:Lm] * f[int(coor[j, 4]), 1:Lm] * np.sin(s[1:Lm] * r_ij) / s[
                                                                                                              1:Lm] / r_ij
    I = I_at + I_mol
    s0 = s[0:Lm] * 1e-10

    I1 = make_interp_spline(s0, I)(s_exp)
    I_at1 = make_interp_spline(s0, I_at)(s_exp)
    I_mol1 = make_interp_spline(s0, I_mol)(s_exp)

    return I1, I_at1, I_mol1


def get_sM_and_PDF_from_I(I_at, I_mol, s, r_max, damp_const):
    sM = I_mol / I_at * s  # calculate sM from I
    r_max = r_max * 1;  # convert to picometer
    r = range(r_max)
    # print(r)
    PDF = [0 for i in range(r_max)]
    for i in range(len(s) - 1):
        PDF += sM[i] * np.sin(s[i] * 1e10 * np.array(r) * 1e-12) * (s[i + 1] - s[i]) * np.exp(-s[i] ** 2 / damp_const)
    # for i in range(r_max):
    #   PDF[i]+=sum(sM*np.sin(s*1e10*np.array(r[i])*1e-12)*(s[1]-s[0])*np.exp(-s**2/damp_const))
    return sM, PDF, np.array(r)


def plot_delay_simulation_with_conv(matrix_before_conv, x_range, col, t_interval, nt, space_for_convol):
    x0 = np.linspace(-col, col, int(255 / t_interval))
    h = np.exp(-x0 ** 2 * t_interval ** 2 / 8000) / (np.pi * 8000 / t_interval ** 2) ** 0.5  # normalize the gaussian
    M1 = get_2d_matrix(x_range, nt + space_for_convol * 2)
    for i in range(x_range):
        M1[i] = signal.convolve(matrix_before_conv[:, i], h, mode='same')

    # M1=np.transpose(M1)
    M1 = np.array(M1)

    norm = TwoSlopeNorm(vmin=M1.min(), vcenter=0, vmax=M1.max())
    plt.figure(figsize=(15, 5))
    pc = plt.imshow(M1[:, 0:nt + space_for_convol - 1], norm=norm, cmap=plt.get_cmap('seismic'), alpha=0.65)
    plt.colorbar(pc)
    ax = plt.gca()
    ax.invert_yaxis()
    ax.xaxis.set_ticks_position('bottom')
    plt.xlabel('time/fs')
    # plt.xticks(np.arange(0,nt+space_for_convol,100),np.arange(-space_for_convol*t_interval,nt*t_interval,100*t_interval))
    # plt.axhline(y=space_for_convol,linestyle='--')
    plt.grid()
    return


def plot_I_sM_PDF(I, sM, PDF, s, r, title_I, title_sM, title_PDF):
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.plot(s, I / I.max())
    plt.xlabel('s/angs^-1')
    plt.title(title_I)
    plt.grid()

    plt.subplot(1, 3, 2)
    plt.plot(s, sM)
    plt.xlabel('s/angs^-1')
    plt.title(title_sM)
    plt.grid()

    plt.subplot(1, 3, 3)
    plt.plot(r, PDF)
    plt.xlabel('r/pm')
    plt.title(title_PDF)
    plt.grid()

    plt.tight_layout()
    plt.show()
    return


def trajectory_sim(path_mol, tra_mol_name, file_type, f, s000, s_max):
    coor_txyz, atom_sum, TIME = load_time_evolving_xyz(path_mol, tra_mol_name, file_type)  # load xyz data
    # options: load_time_evolving_xyz, or load_time_evolving_xyz1
    nt = len(TIME)
    t_interval = float(TIME[1]) - float(TIME[0])
    col = int(160 / t_interval)
    space_for_convol = int(200 / t_interval)

    [I0, I0_at, I0_mol, s] = get_I_from_mol_coor(f, s000, s_max, coor_txyz[0], atom_sum)
    delta_I_over_I_t = get_2d_matrix(nt + space_for_convol * 2, len(s))
    for i in range(nt):
        [I, I_at, I_mol, s] = get_I_from_mol_coor(f, s000, s_max, coor_txyz[i], atom_sum)
        delta_I_over_I_t[i + space_for_convol] = (I - I0) / I
    for i in range(space_for_convol):
        delta_I_over_I_t[i + nt + space_for_convol] = delta_I_over_I_t[nt + space_for_convol - 1]
    delta_I_over_I_t = np.array(delta_I_over_I_t)
    plot_delay_simulation_with_conv(delta_I_over_I_t * 100, len(s), col, t_interval, nt, space_for_convol)
    # this simulation assumes full dissociation
    # after taking dissociation percentage into consideration, the change in signal is much smaller
    plt.ylabel('s/angs^-1')
    plt.yticks(np.arange(0, len(s), len(s) / s.max()), np.arange(0, s.max(), 1))
    plt.axvline(x=space_for_convol, linestyle='--')
    plt.title('delta_I/I %')
    plt.show()
    return


def freq_sim(path_mol, tra_mol_name, file_type, f, s000, s_max, evolutions=10, r_max=800, damp_const=33):
    coor_txyz, atom_sum, TIME = load_freq_xyz(path_mol, tra_mol_name, file_type)  # load xyz data
    # print(coor_txyz.shape)
    # options: load_time_evolving_xyz, or load_time_evolving_xyz1
    nt = len(TIME) * evolutions
    max_time = max(TIME) * evolutions
    t_interval = float(TIME[1]) - float(TIME[0])
    new_time = np.linspace(0, max_time, nt)
    col = int(160 / t_interval)
    space_for_convol = int(200 / t_interval)

    [I0, I0_at, I0_mol, s] = get_I_from_mol_coor(f, s000, s_max, coor_txyz[0], atom_sum)
    delta_I_over_I_t = []
    PDF = []
    k = 0
    for i in range(nt):
        j = i % 20
        # print(j, nt)
        [I, I_at, I_mol, s] = get_I_from_mol_coor(f, s000, s_max, coor_txyz[j], atom_sum)
        dI_I = (I - I0) / I
        delta_I_over_I_t.append(dI_I)
        sM, pdf, r = get_sM_and_PDF_from_I(I_at, I_mol, s, r_max, damp_const)
        PDF.append(pdf)
    #     for i in range(space_for_convol):
    #         delta_I_over_I_t[i+nt+space_for_convol]=delta_I_over_I_t[nt+space_for_convol-1]
    delta_I_over_I_t = np.array(delta_I_over_I_t)
    PDF = np.array(PDF)
    return delta_I_over_I_t, new_time, s, PDF, r


def dissoc_sim(path_mol, reactant, products, file_type, f, s000, s_max, r_max=800, damp_const=33):
    [coor0, atom_sum0] = load_static_mol_coor(path_mol, reactant, file_type)
    [I0, I0_at, I0_mol, s] = get_I_from_mol_coor(f, s000, s_max, coor0, atom_sum0)
    [sM0, pdf0, r] = get_sM_and_PDF_from_I(I0_at, I0_mol, s, r_max, damp_const)

    I_prods = []
    sM_prods = []
    pdf_prods = []
    for i in range(len(products)):
        frag_name = str(products[i])
        coor, atom_sum = load_static_mol_coor(path_mol, frag_name, file_type)
        I, I_at, I_mol, s = get_I_from_mol_coor(f, s000, s_max, coor, atom_sum)
        I_prods.append(I)
        sM, pdf, r = get_sM_and_PDF_from_I(I0_at, I_mol, s, r_max, damp_const)
        pdf_prods.append(pdf)
        sM_prods.append(sM)

    I_prods = np.sum(I_prods, axis=0)
    sM_prods = np.sum(sM_prods, axis=0)
    pdf_prods = np.sum(pdf_prods, axis=0)
    dsM = s * (I_prods - I0) / I0_at

    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.plot(s, sM0, label="reactant")
    plt.plot(s, sM_prods, label="products")
    plt.xlabel(r'S, [' + angs + '$^{-1}$]');
    plt.ylabel('sM(s)');
    plt.legend()
    plt.title("sM")

    plt.subplot(1, 2, 2)
    plt.plot(r, pdf0, label="reactant")
    plt.plot(r, pdf_prods, label="products")
    plt.xlabel(r'R, [pm]');
    plt.legend()
    plt.title("PDF")

    dPDF = [0 for i in range(r_max)]
    for i in range(len(s) - 1):
        dPDF += dsM[i] * np.sin(s[i] * 1e10 * np.array(r) * 1e-12) * (s[i + 1] - s[i]) * np.exp(-s[i] ** 2 / damp_const)
    # for i in range(r_max):
    #   PDF[i]+=sum(sM*np.sin(s*1e10*np.array(r[i])*1e-12)*(s[1]-s[0])*np.exp(-s**2/damp_const))

    return dsM, s, dPDF, r


### PDF Generating Functions
# todo add theses functions

