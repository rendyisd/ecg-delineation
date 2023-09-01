import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pywt
from scipy import stats

import wfdb

# Helper
def grouped(itr, n=3):
    itr = iter(itr)
    end = object()
    while True:
        vals = tuple(next(itr, end) for _ in range(n))
        if vals[-1] is end:
            return
        yield vals

def save_file(fpath, data):
    with open(fpath, 'wb') as f:
        pickle.dump(data, f)

def open_pickle(fpath):
    with open(fpath, 'rb') as f:
        return pickle.load(f)
    
def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

'''
signal: an array (self-explanatory)
wavelet: either wavelet name string or wavelet object
level: an integer, decomposition level
'''
def universal_threshold(coeffs):
    # Flatten the detail coefficients (excluding the approximation coefficients)
    detail_coeffs = np.concatenate(coeffs[1:])

    # Compute the noise variance estimate using Median Absolute Deviation (MAD)
    noise_var = (stats.median_abs_deviation(detail_coeffs) / 0.6745) ** 2

    # Calculate the universal threshold based on the SURE criterion
    threshold = np.sqrt(2 * noise_var * np.log(len(detail_coeffs)))

    return threshold

# TODO:find the reference for SURE threshold
def denoise_dwt(signal, wavelet, level):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    '''
    As per the pywt.wavedec docs:
    coeffs[0] contains approximation coeffs and the rest are detail coeffs
    '''
    # threshold = universal_threshold(coeffs)

    for i in range(1, len(coeffs)):
        threshold = stats.median_abs_deviation(coeffs[i])
        coeffs[i] = pywt.threshold(coeffs[i], threshold, mode='hard')
    
    denoised_signal = pywt.waverec(coeffs, wavelet)

    return denoised_signal

def calculate_snr(original_signal, denoised_signal):
    if len(original_signal) != len(denoised_signal):
        raise ValueError("Original and denoised signal must have the same length")
    
    original_signal = np.asarray(original_signal)
    denoised_signal = np.asarray(denoised_signal)

    original_sq = np.sum(original_signal ** 2)

    noise = original_signal - denoised_signal
    noise_sq = np.sum(noise ** 2)

    # result in decibels
    snr = 10 * np.log10(original_sq / noise_sq)

    return snr

# Below function is from The Optimal Selection of Mother Function ... paper
# def calculate_snr(original_signal, denoised_signal):
#     original_signal = np.asarray(original_signal)
#     denoised_signal = np.asarray(denoised_signal)

#     numerator = np.sum(original_signal ** 2)
#     denominator = np.sum(np.absolute(original_signal - denoised_signal))

#     snr = 10 * np.log10(numerator / denominator)

#     return snr