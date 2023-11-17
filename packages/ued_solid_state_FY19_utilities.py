# Utility functions for solid state UED FY19
# Xiaozhe Shen
# Thu Jun 28 18:02:33 PDT 2018
from dirtools import Dir
import pandas as pd
import numpy as np
#import dask.dataframe as df
#from dask import delayed
import cv2

# Cython utilities
import os
import sys


code_path = os.path.abspath('/cds/sw/ds/ana/conda1/inst/envs/ana-4.0.1-py3/lib/python3.7/site-packages/')

if code_path not in sys.path:
    sys.path.append(code_path)

# Timing decorator
from functools import wraps
from time import time

# Gaussian filter
from scipy.ndimage.filters import gaussian_filter

# Image processing
from skimage.morphology import erosion, disk, white_tophat, dilation

# Meanshift clustering algorithm
from sklearn.cluster import MeanShift, estimate_bandwidth, KMeans

# Multiprocessing
import multiprocessing
#from multiprocessing import set_start_method # cannot import
#set_start_method('forkserver')

# Shared memory
import ctypes


#code_path = os.path.abspath('/scratch/xshen/Codes/General/cython_functions/')
#
#if code_path not in sys.path:
#    sys.path.append(code_path)
#
#from gas_phase_UED_cython_functions import *


############## Timing decorator function ##############

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('funcion: %r took: %2.4f sec' % \
          (f.__name__, te-ts))
        return result
    return wrap

######################################################


############## Read diffraction data ##############

## -1. Get image file list
def get_diffraction_file_list(path, keyword='ANDOR1*.tif'):
    '''
    file_list = get_diffraction_file_list(path, keyword='ANDOR1*.tif')
    '''
    temp_dir = Dir(path)
    file_names = temp_dir.files(keyword)
    file_list = [path + '/' + fn for fn in file_names]
    return file_list

## 0. Extract scan number and delay stage reading from formatted file name
def extract_info_from_file_name(file_name):
    temp_fn = file_name.split('/')
    temp_scan = ''
    for t in temp_fn:
        if 'scan' in t:
            temp_scan = t.split('scan')
            if len(temp_scan) > 1 and temp_scan[1].isdigit():
                scan_num = int(temp_scan[1])
                break
        else:
            scan_num = 1
    if len(temp_fn[-1].split('_')) > 5: # old file format
        temp_delay = temp_fn[-1].split('_')
        temp_delay = temp_delay[1].split('-')
        if len(temp_delay) > 3: # negative delay contains one more '-'
            delay = -1*float(temp_delay[-1])
        else:
            delay = float(temp_delay[-1])
    else: # new file format
        delay=float(temp_fn[-1].split('_')[3])
    return scan_num, delay

## 1. Serial code
@timing
def read_diffraction_image_serial(file_names):
    '''
    img_set = read_diffraction_image_serial(file_names)
    '''
    data = [None]*len(file_names)
    for ind, fn in enumerate(file_names):
        data[ind] = cv2.imread(fn, cv2.IMREAD_UNCHANGED)
    return np.array(data, dtype=np.float64)

@timing
def read_diffraction_data_serial(file_names):
    '''
    img_set, scan, delay = read_diffraction_data_serial(file_names)
    '''
    data = [None]*len(file_names)
    scan = [None]*len(file_names)
    delay = [None]*len(file_names)
    for ind, fn in enumerate(file_names):
        data[ind] = cv2.imread(fn, cv2.IMREAD_UNCHANGED)
        scan[ind], delay[ind] = extract_info_from_file_name(fn)
    return np.array(data, dtype=np.float64), scan, delay


## 2. Use shared memory + multithreading to read in data
def assign_shared_memory(dimension):
    '''
    data_set = assign_shared_memory(dimension)
    # dimension is a tuple with the shape of the memory to be created
    '''
    # dimension is a tuple with the shape of the memory to be created
    total = 1
    for d in dimension:
        total = total * d
    return np.ctypeslib.as_array(multiprocessing.Array(ctypes.c_double, total).get_obj()).reshape(dimension)

def read_one_diffraction_data_sharedarray(args):
    ind, file_name, img, scan, delay = args
    img[:,:] = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)
    scan[ind], delay[ind] = extract_info_from_file_name(file_name)
    return

@timing
def read_diffraction_data_sharedarray(file_names, img_array, scan_array, delay_array, num_thread=18):
    '''
    read_diffraction_data_sharedarray(file_names, img_array, scan_array, delay_array, num_thread=18)
    '''
    p = multiprocessing.pool.ThreadPool(processes=num_thread)
    index = np.arange(len(file_names))
    p.map(read_one_diffraction_data_sharedarray, [(ind, fn, img, scan_array, delay_array) for (ind, fn, img) in zip(index, file_names, img_array)])
    p.close()
    p.join()
    return

def read_one_diffraction_image_sharedarray(args):
    file_name, img = args
    img[:,:] = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)
    return

def extract_info_from_file_name_power_scan(file_name):
    temp_fn = file_name.split('/')
    for t in temp_fn:
        if 'scan' in t:
            temp_scan = t
        if 'pumpThrottle' in t:
            temp = t
    for t in temp.split('_'):
        if 'pumpThrottle' in t:
            temp_throttle = t
        if 'longDelay' in t:
            temp_delay = t

    temp_scan = temp_scan.split('scan')
    scan_num = int(temp_scan[1])

    temp_delay = temp_delay.split('-')
    if len(temp_delay) > 3: # negative delay contains one more '-'
        delay = -1*float(temp_delay[-1])
    else:
        delay = float(temp_delay[-1])

    temp_throttle = temp_throttle.split('-')
    if len(temp_throttle) > 3: # negative delay contains one more '-'
        throttle = -1*float(temp_throttle[-1])
    else:
        throttle = float(temp_throttle[-1])
    return scan_num, delay, throttle

def read_one_diffraction_data_power_scan_sharedarray(args):
    ind, file_name, img, scan, delay, throttle = args
    img[:,:] = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)
    scan[ind], delay[ind], throttle[ind] = extract_info_from_file_name_power_scan(file_name)
    return

@timing
def read_diffraction_data_power_scan_sharedarray(file_names, img_array, scan_array, delay_array, throttle_array, num_thread=18):
    p = multiprocessing.pool.ThreadPool(processes=num_thread)
    index = np.arange(len(file_names))
    p.map(read_one_diffraction_data_power_scan_sharedarray, [(ind, fn, img, scan_array, delay_array, throttle_array) for (ind, fn, img) in zip(index, file_names, img_array)])
    p.close()
    p.join()
    return

@timing
def read_diffraction_image_sharedarray(file_names, img_array, num_thread=18):
    p = multiprocessing.pool.ThreadPool(processes=num_thread)
    p.map(read_one_diffraction_image_sharedarray, [(fn, img) for (fn, img) in zip(file_names, img_array)])
    p.close()
    p.join()
    return

###################################################


############## Data analysis for time-zero from Si single crystal pump-probe ##############

# Pre-process diffraction data, get mask for bragg peaks, and determine bragg peak centers, get total intensity

# Using meanshift method, no need to input number of bragg peaks
@timing
def preprocess_diffraction_images(img_set, threshold_bragg = 1000, sigma_filter=-1):
    # From mean image, using mean shift clustering algorithm to estimate Bragg peak centers
    img_mean = img_set.mean(axis=0)
    if sigma_filter > 0:
        img_mean = gaussian_filter(img_mean, [sigma_filter, sigma_filter])

    ## Use the top two corners to estimate camera dark background
    mask_background = np.zeros_like(img_mean)
    mask_background[0:100, 0:100] = 1
    mask_background[0:100, 923:1023] = 1
    count_background = np.median((img_mean*mask_background)[mask_background==1])
    mask_bragg = dilation( (img_mean-count_background) > threshold_bragg, disk(10) )

    ## Use mean shift algorithm to find peak centers
    training_set = np.argwhere(mask_bragg==1)
    bandwidth = estimate_bandwidth(training_set)
    meanshift=MeanShift(bin_seeding=True)
    #meanshift=MeanShift(bandwidth=bandwidth, bin_seeding=True)
    meanshift.fit(training_set)
    centroids_bragg = meanshift.cluster_centers_.astype(int)
    num_bragg = centroids_bragg.shape[0]

    # Total intensity
    inten_total = img_set.sum(axis=1).sum(axis=1)

    return img_mean, mask_background, mask_bragg, centroids_bragg, inten_total

# Using kmeans method, need to input number of bragg peaks
@timing
def gen_bragg_mask(img_set, threshold_bragg=1000, sigma_filter=-1):
    # From mean image, using mean shift clustering algorithm to estimate Bragg peak centers
    img_mean = img_set.mean(axis=0)
    if sigma_filter > 0:
        img_mean = gaussian_filter(img_mean, [sigma_filter, sigma_filter])

    ## Use the top two corners to estimate camera dark background
    mask_background = np.zeros_like(img_mean)
    mask_background[0:50, 0:50] = 1
    #mask_background[0:100, 923:1023] = 1
    count_background = np.median(img_mean[mask_background==1])
    mask_bragg = dilation( (img_mean-count_background) > threshold_bragg, disk(10) )

    # Total intensity
    inten_total = []
    for i in xrange(img_set.shape[0]):
        img_temp = img_set[i,:,:]
        img_temp = img_temp - np.median(img_temp[mask_background==1])
        inten_total.append(img_temp.sum())
    return img_mean, mask_background, mask_bragg, np.array(inten_total)

@timing
def cal_bragg_centroids(mask_bragg, num_bragg):
    training_set = np.argwhere(mask_bragg==1)
    result = KMeans(n_clusters=num_bragg, init='k-means++').fit(training_set)


    centroids_bragg = result.cluster_centers_
    return centroids_bragg

# Process diffraction data by summing up intensities within a ROI
def analyze_one_diffraction_image(args):
    ind, img, mask_background, mask_bragg, centroids_bragg, bragg_inten_set, count_background_set, halfwidth_bragg = args
    count_background = np.median(img[mask_background==1])
    count_background_set[ind] = count_background
    num_bragg = centroids_bragg.shape[0]

    for i in xrange(num_bragg):
        row_ind = centroids_bragg[i,0]
        col_ind = centroids_bragg[i,1]
        bragg_inten_set[i, ind] = (img[row_ind-halfwidth_bragg:row_ind+halfwidth_bragg, col_ind-halfwidth_bragg:col_ind+halfwidth_bragg]-count_background).sum()
    return

@timing
def process_diffraction_data(img_set, mask_background, mask_bragg, centroids_bragg, bragg_inten_set, count_background_set, halfwidth_bragg = 25, num_thread=18):
    p = multiprocessing.pool.ThreadPool(processes=num_thread)
    index = np.arange(img_set.shape[0])
    p.map(analyze_one_diffraction_image, [(ind, img, mask_background, mask_bragg, centroids_bragg, bragg_inten_set, count_background_set, halfwidth_bragg) for (ind, img) in zip(index, img_set)])
    p.close()
    p.join()
    return

# Analyze diffraction data by Gaussian fitting

def gaussian_linear_background(para, x):
    amp, mu, sigma, slope, offset = para
    return amp*np.exp( -(x-mu)**2/2/sigma**2 ) + slope * x + offset

def gaussian_linear_background_fit_func(para, x, y):
    amp, mu, sigma, slope, offset = para
    return amp*np.exp( -(x-mu)**2/2/sigma**2 ) + slope * x + offset - y

def fit_gaussian_linear_background(x, y, para0=None):
    if para0 == None:
        offset0 = y.min()
        amp0 = y.max() - offset0
        mu0 = x[np.argwhere(y==y.max())][0][0] # get the first element if more than two
        try:
            sigma0 = (x[np.argwhere(y>np.exp(-0.5*2**2)*amp0+offset0).max()] - x[np.argwhere(y>np.exp(-0.5*2**2)*amp0+offset0).min()] )/(2*2)
        except:
            sigma0 = 5
        slope0 = 0
        para0 = (amp0, mu0, sigma0, slope0, offset0)
    try:
        fit_result = least_squares(gaussian_linear_background_fit_func, para0, args=(x,y))
        para = fit_result.x
        J = fit_result.jac
        pcov = np.linalg.inv(np.matmul(J.transpose(), J))

        if (len(y) > len(para0)) and pcov is not None:
            mse = gaussian_linear_background_fit_func(para, x, y)
            mse = (mse**2).sum()/(len(y) - len(para0))
            pcov = pcov * mse
        else:
            pcov = np.inf

        para_error = []
        for i in range(len(para0)):
            try:
                para_error.append(np.absolute(pcov[i][i])**0.5)
            except:
                para_error.append( 0.00 )
    except:
        para = np.array(para0)
        para_error = [np.nan, np.nan, np.nan, np.nan, np.nan]

    return para, para_error

def fit_one_bragg_peak(img, flagPlot=-1):
    row_pro = img.sum(axis=1)
    row_axis = np.arange(len(row_pro))
    para_row, para_err_row = fit_gaussian_linear_background(row_axis, row_pro)

    col_pro = img.sum(axis=0)
    col_axis = np.arange(len(col_pro))
    para_col, para_err_col = fit_gaussian_linear_background(col_axis, col_pro)

    if flagPlot>0:
        plt.figure(flagPlot)
        plt.subplot(2,1,1)
        plt.plot(row_axis, gaussian_linear_background(para_row, row_axis), 'r-', label='fitted')
        plt.plot(row_axis, row_pro, 'or', label='raw row projection')
        plt.legend()
        plt.subplot(2,1,2)
        plt.plot(col_axis, gaussian_linear_background(para_col, col_axis), 'g-', label='fitted')
        plt.plot(col_axis, col_pro, 'og', label='raw col projection')

    return para_row, para_err_row, para_col, para_err_col


def gaussian_fit_one_diffraction_image(args):
    ind, img, mask_background, mask_bragg, centroids_bragg, bragg_info_set, count_background_set, halfwidth_bragg = args
    count_background = np.median((img*mask_background)[mask_background==1])
    count_background_set[ind] = count_background

    num_bragg = bragg_info_set.shape[0]

    for i in xrange(num_bragg):
        row_ind = centroids_bragg[i,0]
        col_ind = centroids_bragg[i,1]

        para_row, para_err_row, para_col, para_err_col = fit_one_bragg_peak(img[row_ind-halfwidth_bragg:row_ind+halfwidth_bragg, col_ind-halfwidth_bragg:col_ind+halfwidth_bragg]-count_background)
        para_row[1] += (row_ind-halfwidth_bragg)
        para_col[1] += (col_ind-halfwidth_bragg)

        bragg_info_set[i, 0:5, ind] = para_row
        bragg_info_set[i, 5:10, ind] = para_col

    return

@timing
def process_diffraction_data_gaussian_fit(img_set, mask_background, mask_bragg, centroids_bragg, bragg_info_set, count_background_set, halfwidth_bragg = 25, num_thread=18):
    p = multiprocessing.pool.ThreadPool(processes=num_thread)
    #p = multiprocessing.Pool(processes=num_thread)
    index = np.arange(img_set.shape[0])
    p.map(gaussian_fit_one_diffraction_image, [(ind, img, mask_background, mask_bragg, centroids_bragg, bragg_info_set, count_background_set, halfwidth_bragg) for (ind, img) in zip(index, img_set)])
    p.close()
    p.join()
    return



######################################################################################################


######################## Online gas phase UED analysis  ###########################

# Circle fit
def circle_fit_cost_func(center, x,y):
    x0, y0 = center
    radii = ((x-x0)**2 + (y-y0)**2)**0.5
    return radii - radii.mean()

def circle_fit(x,y):
    x0 = np.mean(x)
    y0 = np.mean(y)
    result = least_squares(circle_fit_cost_func, (x0, y0), loss='soft_l1', f_scale=0.1, args=(x,y))
    return result.x

import matplotlib.pyplot as plt
import matplotlib
from scipy.optimize import least_squares
def cal_one_diffraction_pattern_center(img, sampling_point, half_width=5, flagPlot=-1):
    '''
    Sampling point = np.array([s1, s2, s3, ...]), where the s' are index used as [512, s] to
    retrieve the diffraction intensity for circle fitting
    Best for gas phase data with not very strong rings
    '''

    centers = np.zeros((len(sampling_point),2))

    if flagPlot>0:
        plt.figure(num=flagPlot)
        plt.imshow(img)
    for i, s in enumerate(sampling_point):
        ind = np.logical_and(img>img[512, s-half_width:s+half_width].min(), img<img[512, s-half_width:s+half_width].max())
        ring = np.where(ind)
        if len(ring[0])>0: # in case ring is empty
            x0, y0 = circle_fit(ring[1], ring[0])
            if flagPlot>0:
                plt.scatter(ring[1], ring[0], 1, marker='.')
                plt.scatter(x0, y0, 15, marker='x', label=r'x0=%3.1f,y0=%3.1f'%(x0, y0))
                plt.legend()
            centers[i, 0] = x0
            centers[i, 1] = y0
    return np.array(centers).mean(axis=0)

def cal_one_diffraction_pattern_center_single_threshold(img, threshold, flagPlot=-1):
    '''
    Use a single threshold to get rings for fitting, best for solid state polycrystalline sample with uniform and strong rings
    '''

    centers = np.zeros((1,2))

    if flagPlot>0:
        plt.figure(num=flagPlot)
        plt.imshow(img)

    ind = img>threshold
    ring = np.where(ind)
    if len(ring[0])>0: # in case ring is empty
        x0, y0 = circle_fit(ring[1], ring[0])
        if flagPlot>0:
            plt.scatter(ring[1], ring[0], 1, marker='.')
            plt.scatter(x0, y0, 15, marker='x', label=r'x0=%3.1f,y0=%3.1f'%(x0, y0))
            plt.legend()
        centers[0, 0] = x0
        centers[0, 1] = y0
    return np.array(centers).mean(axis=0)

def cal_one_diffraction_pattern_center_double_threshold(img, threshold_low, threshold_high, flagPlot=-1):
    '''
    Use a single threshold to get rings for fitting, best for solid state polycrystalline sample with uniform and strong rings
    '''

    centers = np.zeros((1,2))

    if flagPlot>0:
        plt.figure(num=flagPlot)
        plt.imshow(img)

    ind = np.logical_and(img>threshold_low, img<threshold_high)
    ring = np.where(ind)
    if len(ring[0])>0: # in case ring is empty
        x0, y0 = circle_fit(ring[1], ring[0])
        if flagPlot>0:
            plt.scatter(ring[1], ring[0], 1, marker='.', color='w')
            plt.scatter(x0, y0, 15, marker='x', label=r'x0=%3.1f,y0=%3.1f'%(x0, y0))
            plt.legend()
        centers[0, 0] = x0
        centers[0, 1] = y0
    return np.array(centers).mean(axis=0)

def gen_circular_mask(x0, y0, radius, num_row, num_col):
    mask = np.zeros((num_row, num_col))

    X,Y = np.meshgrid(np.arange(num_col), np.arange(num_row))
    mask[(X-x0)**2+(Y-y0)**2 < radius**2] = 1
    return mask

# Calculate hole mask from Si diffraction pattern
def gen_ind_tuple(row_low, row_high, col_low, col_high):
    row_ind = []
    col_ind = []
    for i in np.arange(row_low, row_high+1):
        for j in np.arange(col_low, col_high+1):
            row_ind.append(i)
            col_ind.append(j)
    return (row_ind, col_ind)

def cal_hole_mask_from_Si_pattern(img, thres_hole=530, radius_hole=45, flagPlot=-1, center_ind_tuple=gen_ind_tuple(550, 650, 450, 580)):
    mask_hole = np.zeros_like(img)
    mask_hole[np.where(img<thres_hole)] = 1
    m = np.zeros_like(mask_hole)
    #m[550:650, 450:580] = 1 # Only look at the center part for the hole
    m[center_ind_tuple] = 1 # Only look at the center part for the hole
    mask_hole = mask_hole * m
    mask_hole = 1 - mask_hole

    # Fit the center
    pp = np.argwhere(mask_hole==0)
    center = circle_fit(pp[:,1], pp[:,0])

    num_row, num_col = img.shape
    mask_hole = 1-gen_circular_mask(center[0], center[1], radius_hole, num_row, num_col)

    if flagPlot>0:
        plt.figure(flagPlot)
        plt.subplot(1,2,1)
        plt.imshow(img)
        plt.title('Raw')
        plt.subplot(1,2,2)
        plt.imshow(img*mask_hole)
        plt.title('Hole masked')

    return center, mask_hole

# Calculate total intensity using nanmean
@timing
def cal_total_diffraction_intensity(img_set, mask_hole, num_thread=8):
    p = multiprocessing.pool.ThreadPool(processes=num_thread)
    inten_total = p.map(lambda img:np.nansum(img*mask_hole), img_set)
    p.close()
    p.join()
    return inten_total

# Remove pump background from data
def remove_one_pump_background(args):
    img, img_bg = args
    img[:,:] -= img_bg
    return

@timing
def remove_pump_background_from_diffraction_data(img_set, img_bg, num_thread=8):
    p = multiprocessing.pool.ThreadPool(processes=num_thread)
    camera_bg = p.map(remove_one_pump_background, [(img, img_bg) for img in img_set])
    p.close()
    p.join()
    return

# Normalize diffracton by intensity
def normalize_one_diffraction(args):
    img, inten = args
    img[:,:] /= inten
    return

@timing
def normalize_diffraction(img_set, inten_totl, num_thread=8):
    p = multiprocessing.pool.ThreadPool(processes=num_thread)
    camera_bg = p.map(normalize_one_diffraction, [(img, inten) for (img, inten) in zip(img_set, inten_total)])
    p.close()
    p.join()
    return



# Preprocess gas diffraction data

def preprocess_one_gas_diffraction_data(args):
    img, hot_pixel_thres, ind_bg = args
    # Set hot pixel to nan
    ind_nan = np.where(img>hot_pixel_thres)
    img[ind_nan] = np.nan

    camera_bg = np.nanmedian(img[ind_bg])
    img[:,:] -= camera_bg

    return camera_bg

@timing
def preprocess_gas_diffraction_data(img_set, hot_pixel_thres=10000, ind_bg = [np.arange(100), np.arange(100)], num_thread=8):
    p = multiprocessing.pool.ThreadPool(processes=num_thread)
    camera_bg = p.map(preprocess_one_gas_diffraction_data, [(img, hot_pixel_thres, ind_bg) for img in img_set])
    p.close()
    p.join()
    return camera_bg


##################################################################################



######################################################################################################


'''
######################## Online diffuse scattering UED analysis  ###########################

# Generate differential images
@timing
def cal_one_image_centroid(img, bragg_mask, centroids_bragg, halfwidth_bragg=25, flagPlot=-1):
    for i in xrange(centroids_bragg.shape[1]):
        row_ind = centroids_bragg[i,0]
        col_ind = centroids_bragg[i,1]



def gen_differential_image(img_set, mask_hole, delay):
    delay_unique = np.unique(delay)
    mean_img_set = np.zeros((len(delay_unique), img_set.shape[1], img_set.shape[2]))
    mean_total_inten_set = np.zeros((len(delay_unique),))
    for i,d in enumerate(delay_unique):
        ind = np.argwhere(delay==d)[0]
        mean_img_set[i,:,:] = cal_nanmean_image_axis_0(img_set[ind, :,:])
        mean_total_inten_set[i] = np.nanmean(mean_img_set[i]*mask_hole)
        mean_img_set[i,:,:] = mean_img_set[i] / mean_total_inten_set[i]

    return delay_unique, mean_img_set, mean_total_inten_set

# Using cross-correlation algorithm to align images
from skimage.feature import register_translation
from skimage.transform import AffineTransform, warp


@timing
def align_one_image(img, img_ref, accuracy=10, flagPlot=-1):
    offset, error, diffphase = register_translation(img_ref, img, 10)
    transform = AffineTransform(translation=[offset[1], offset[0]])
    return warp(img, transform, mode='wrap', preserve_range=True), offset


##################################################################################
'''


