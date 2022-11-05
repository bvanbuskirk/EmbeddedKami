#!/usr/bin/env python
# coding: utf-8

"""
This script works to take in noisy measurements taken in by the IMU and filters them
to provide smoother and more accurate data. 

Additionally, to compress further, the script takes the transform of the data in order
to allow us to transmit the smallest possible state space model to our machine learning
running on the central server.

Author: William Vavrik 
Date: 11-02-2022

sources:
https://www.geeksforgeeks.org/digital-low-pass-butterworth-filter-in-python/
https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.buttord.html
https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html#scipy.signal.butter

"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy import fft as fourier
import math

'''
Takes in a buffer of data, and provides the appropriate transform to de-noise
we implement a butterworth filter and appropriately convolve it with the 
cutoff is typically 0.5 which corresponds to 1/2 the pyquist frequency. The Nyquist frequncy
is half the sampling frequency of the system.

If you want to preserve the signal that looks like it takes the entire sampling period...
say 1/2 period of a sin wave, the frequency of this signal will be as follows:
fN = Nyquist freq
fs = given
T = (2 * len(data)) / fs
f_desired = 1/T = fs / (2*len(data))
f_desired = (fs/2) * (1/len(data))
f_desired = fN * (1/len(data)) # for butterworth, wN is normalized to 1
w = 2pi*f
w_desired = wN * (2pi/len(data))# for butterworth, wN is normalized to 1
wc = 2pi/len(data)

examples:
1)
fs = 1000 #1000Hz
t = np.linspace(0, 1, fs, False)  # 1 second
freqs = [(5,0.5), (1,2.5), (4,50)]
height = 0
sig = 0
for a,freq in freqs:
    sig += a * np.sin(2*np.pi*freq*t)
    height += a
    
filtered = de_noise(sig, fs)
tester = test(sig, fs)
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.plot(t, sig)
ax1.set_title( ", ".join(map(str,freqs)) + 'Hz sinusoids')
ax1.axis([0, 1, -height, height])
ax2.plot(t, filtered)
ax2.set_title('After  low-pass filter')
ax2.axis([0, 1, -height, height])
ax2.set_xlabel('Time [seconds]')
plt.tight_layout()
plt.show()
print(take_fft(filtered))
print(take_fft(sig))

2) Given time series data, (a_x, a_y, a_z)
a_state = get_dc_fundamental(de_noise(a_x))
or 
a_state = get_dc(a_x) # if resource constrained

'''

#convolves the data with the low-pass filter
def de_noise(data, sample_freq):
    # create a butterworth low-pass filter
    cutoff = (1 / len(data)) * 10
    fc = (cutoff * sample_freq / 2)
    wc = 2 * np.pi * cutoff
#     N, Wn = signal.buttord(wp=wc, ws=wc+0.1, gpass=3, gstop=40, analog=False)
#     print(N, Wn)
    sos = signal.butter(4, fc, 'lp', fs=sample_freq, output='sos')
    de_noise_data = signal.sosfilt(sos, data)
    return de_noise_data

#rtakes in 1D set of data, eturns a[0], the real-valued DC component of the signal
def get_dc(data):
    return np.average(np.real(data))

#takes a 1D set of data, returns a tuple of the (a[0],a[1]), both are real values
def get_dc_fundamental(data):
    transformed = fourier.rfft(data)
    return (np.real(transformed[0]), np.real(transformed[1]))




