# calculates the dark current

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

stem = '/Users/bandari/Documents/git.repos/dirac/'

dark_raw_file_name = stem + '/data/nirao_09_dark_current/tests_junk_13may/pos1_selected_cold_target_not_cold/20sec.fits'
bias_file_name = stem + '/data/nirao_09_dark_current/tests_junk_13may/pos1_selected_cold_target_not_cold/100ms.fits'
badpix_file_name = stem + 'data/nirao_09_dark_current/ersatz_badpix.fits'

## ## TBD: make bad pix correction

gain = 1 # [e/ADU]
exposure_time = 300 # [sec]

hdul_dark_raw = fits.open(dark_raw_file_name)
dark_raw = hdul_dark_raw[0].data

hdul_bias = fits.open(bias_file_name)
bias = hdul_bias[0].data

# TBD: bad pix correction
#hdul_badpix = fits.open(badpix_file_name)
#badpix = hdul_bias[0].data

# number of pixels 
N_pix = dark_raw.size

# dark-subt
dark = dark_raw - bias # [ADU]

# dark current in ADU
dark_curr_adu_full_frame = dark / exposure_time # [ADU/sec]
dark_curr_adu = np.sum( dark ) / (N_pix * exposure_time) # [ADU/pix/sec]

# dark current in e-
dark_curr_e_full_frame = dark_curr_adu_full_frame * gain
dark_curr_e = dark_curr_adu * gain  # [e/pix/sec]

# criterion for success:
# Dark noise < 0.1 e-/px/s.s
print('--------------------------------------------------')
print('Criterion for success: Dark noise < 0.1 e-/px/sec')
print('Dark current (e/pix/sec):',dark_curr_e)
if dark_curr_e < 0.1:
    print('NIRAO-09 result: PASS')
elif dark_curr_e > 0.1:
    print('NIRAO-09 result: FAIL')

# plot histogram
plot_file_name = 'scratch_nirao_09_hist_e_counts.png'
plt.hist(dark_curr_e_full_frame)
plt.title('Histogram of e- across detector')
plt.xlabel('N_e in a pixel')
plt.savefig(plot_file_name)
print('Wrote',plot_file_name)
print('--------------------------------------------------')