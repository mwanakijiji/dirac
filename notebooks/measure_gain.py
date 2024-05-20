
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

'''
More rigorous methodology:
https://www.mirametrics.com/tech_note_ccdgain.php
(implement later)

also Howell, p. 71ff

Quick and easy methodology from
https://www.photometrics.com/learn/imaging-topics/gain#:~:text=Calculate%20a%20bias%2Dcorrected%20image,%3A%20gain%20%3D%20mean%20%2Fvariance.

Quick and easy way to measure gain: 

Collect a bias image (zero-integration dark image) and label it “bias”.
Collect two even-illumination images and label them “flat1” and “flat2”.
Calculate a difference image: diff = flat2 – flat1.
Calculate the standard deviation of the central 100 x 100 pixels in the difference image.
Calculate the variance by squaring the standard deviation and dividing by 2 (variance adds per image, so the variance of the difference image is the sum of the variance of flat1 and flat2).
Calculate a bias-corrected image by subtracting the bias from one of the flat images and label it corr: corr = flat1 – bias.
Obtain the mean illumination level by calculating the mean of the central 100 x 100 region of the corr image.
The mean divided by the variance equals the gain: gain = mean /variance.
'''

stem = '/Users/bandari/Documents/git.repos/dirac/vtp_scripts_data/nirao_09_dark_current/'

bias_file_name = stem + 'data/tests_junk_13may/pos1_selected_cold_target_not_cold/100ms.fits'
badpix_file_name = stem + 'data/ersatz_bad_pix.fits'

flat1_file_name = stem + 'data/tests_junk_13may/pos1_selected_cold_target_not_cold/20sec.fits'
flat2_file_name = stem + 'data/tests_junk_13may/pos1_selected_cold_target_not_cold/100ms.fits'

# bad pixel frame (0: good, 1: bad)
# (N.b. these pixels are masked in the detector readout, not corrected)
hdul_badpix = fits.open(badpix_file_name, dtype=int)
badpix = hdul_badpix[0].data

# read in bias frame
hdul_bias = fits.open(bias_file_name)
bias = hdul_bias[0].data.astype(np.float32)
bias[badpix == 1] = np.nan

# read in flat1
hdul_flat1_raw = fits.open(flat1_file_name)
flat1_raw = hdul_flat1_raw[0].data.astype(np.float32)
flat1_raw[badpix == 1] = np.nan

# read in flat2
hdul_flat2_raw = fits.open(flat2_file_name)
flat2_raw = hdul_flat2_raw[0].data.astype(np.float32)
flat2_raw[badpix == 1] = np.nan

# Calculate a difference image: diff = flat2 – flat1.
diff = flat2_raw - flat1_raw

# Calculate the standard deviation of the central 100 x 100 pixels in the difference image.
diff_central = diff[diff.shape[0]//2 - 50:diff.shape[0]//2 + 50, diff.shape[1]//2 - 50:diff.shape[1]//2 + 50]
std_dev = np.std(diff_central)

# Calculate the variance by squaring the standard deviation and dividing by 2 (variance adds per image, so 
# the variance of the difference image is the sum of the variance of flat1 and flat2).
var_image = (std_dev ** 2) / 2

# Calculate a bias-corrected image by subtracting the bias from one of the flat images and label it corr: 
# corr = flat1 – bias.
corr = flat1_raw - bias

# Obtain the mean illumination level by calculating the mean of the central 100 x 100 region of the corr image.
mean_illumination = np.mean(corr[corr.shape[0]//2 - 50:corr.shape[0]//2 + 50, corr.shape[1]//2 - 50:corr.shape[1]//2 + 50])

# The mean divided by the variance equals the gain: gain = mean /variance.
gain_quick = mean_illumination / var_image

print('Gain, as found using the quick and dirty method [ADU/e]:', gain_quick)