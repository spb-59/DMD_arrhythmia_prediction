
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import logging as lg
import pandas as pd
import pywt
import numpy as np



'''
This file contains the several functions used for denoising the signal. 
'''

def removeBaselineWander(ecg_signal:pd.DataFrame,sf):
    
    sampling_rate = sf  
    cutoff_frequency = 0.8
    nyquist_rate = sampling_rate / 2

    index=['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'] 

    b, a = signal.butter(1, cutoff_frequency / nyquist_rate, btype='highpass')
    for i in index:    
        ecg_signal[i] = signal.filtfilt(b, a, ecg_signal[i])
    return ecg_signal


def SWT(ecg_signal, wavelet,level):
   

    index=['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'] 
    for i in index:
        coeffs = pywt.swt(ecg_signal[i], wavelet, level=level)

        sigma = np.median(np.abs(coeffs[-1])) / 0.6745

        uthresh = sigma * np.sqrt(2 * np.log(len(ecg_signal[i])))
        

        denoised_coeffs = [pywt.threshold(c, uthresh, mode='soft') for c in coeffs]
        

        ecg = pywt.iswt(denoised_coeffs, wavelet)
        ecg_signal[i]=ecg[0].T
    
    return ecg_signal


def DWT(ecg_signal, wavelet,level):
   

    index=['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'] 
    for i in index:
        coeffs = pywt.wavedec(ecg_signal[i], wavelet, level=level)

        sigma = np.median(np.abs(coeffs[-1])) / 0.6745

        uthresh = sigma * np.sqrt(2 * np.log(len(ecg_signal[i])))
        

        denoised_coeffs = [pywt.threshold(c, uthresh, mode='soft') for c in coeffs]
        

        ecg_signal[i] = pywt.waverec(denoised_coeffs, wavelet)
    
    return ecg_signal

def notchFilter(ecg_signal, sf  ):
    notch_freq=50
    quality_factor=30
    sampling_rate =sf
    nyquist_rate = sampling_rate / 2

    index=['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'] 
    b, a = signal.iirnotch(notch_freq / nyquist_rate, quality_factor)
    for i in index:    
        ecg_signal[i] = signal.filtfilt(b, a, ecg_signal[i])
    return ecg_signal

def removeHighFrequency(ecg_signal,sf):
    sampling_rate = sf  
    cutoff_frequency = 45
    nyquist_rate = sampling_rate / 2

    index=['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'] 

    b, a = signal.butter(4, cutoff_frequency / nyquist_rate, btype='lowpass')
    for i in index:    
        ecg_signal[i] = signal.filtfilt(b, a, ecg_signal[i])
    return ecg_signal


def denoise(signal,sf):
    '''
    This function combines the denoising functions and returns and signal with reduced noise.
    '''

    lg.info(msg="Starting denoising for current record")

    lg.info(msg="Baseline wander removal for current record")
    signal=removeBaselineWander(signal,sf)
    lg.info(msg="Baseline wander removal for current record complete")

    lg.info(msg="High Frequency removal for current record")
    signal=removeHighFrequency(signal,sf)
    lg.info(msg="High Frequency removal for current record complete")

    lg.info(msg="DWT removal for current record ")
    signal=DWT(signal,'sym6',6)
    lg.info(msg="DWT removal for current record complete")

    lg.info(msg="SWT removal for current record ")
    signal=SWT(signal,'bior4.4',3)
    lg.info(msg="SWT removal for current record complete ")

    lg.info('Denoising steps for current record complete')
    return signal
    






