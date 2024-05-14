# calculates the dark current

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
import logging
import datetime

log_file_name = 'log_nirao_09_' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.txt'
logging.basicConfig(filename=log_file_name, 
                    level=logging.INFO, format='%(message)s')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(message)s')
console.setFormatter(formatter)
logging.getLogger().addHandler(console)
logger = logging.getLogger()

stem = '/Users/bandari/Documents/git.repos/dirac/vtp_scripts_data/nirao_09_dark_current/'

logger.info('-----------------------------------------------------')
logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': NIRAO-9 Dark Current test')
logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': Criterion for success: Dark noise < 0.1 e-/px/sec')
logger.info('-----------------------------------------------------')

dark_raw_file_name = stem + 'data/tests_junk_13may/pos1_selected_cold_target_not_cold/20sec.fits'
bias_file_name = stem + 'data/tests_junk_13may/pos1_selected_cold_target_not_cold/100ms.fits'
badpix_file_name = stem + 'data/ersatz_bad_pix.fits'

logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': Raw dark: ' + dark_raw_file_name)
logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': Bias: ' + bias_file_name)
logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': Bad pixel mask: ' + badpix_file_name)

gain = 1 # [e/ADU]
exposure_time = 300 # [sec]

# bad pixel frame (0: good, 1: bad)
# (N.b. these pixels are masked in the detector readout, not corrected)
hdul_badpix = fits.open(badpix_file_name, dtype=int)
badpix = hdul_badpix[0].data

# read in dark frame and mask bad pixels with NaNs
hdul_dark_raw = fits.open(dark_raw_file_name)
dark_raw = hdul_dark_raw[0].data.astype(np.float32) # convert to float to allow NaNs
dark_raw[badpix == 1] = np.nan

# read in bias frame
hdul_bias = fits.open(bias_file_name)
bias = hdul_bias[0].data.astype(np.float32)
bias[badpix == 1] = np.nan

# total number of pixels 
N_pix_tot = dark_raw.size

# dark-subtract
dark = dark_raw - bias # [ADU]

# before any further calculations, eliminate any non-finite numbers

# how many pixels are finite?
N_pix_finite = np.nansum(np.isfinite(dark))
# fraction of good pixels within science region of detector (i.e., 4-pixel-wide overscan region of 16320 pixels removed)
frac_finite = N_pix_finite/(N_pix_tot - 16320)

# dark current in ADU
dark_curr_adu_full_frame = dark / exposure_time # entire frame of dark ADU counts [ADU/sec]
dark_curr_adu = np.nansum(dark[np.isfinite(dark)]) / (N_pix_finite * exposure_time) # dark current [ADU/pix/sec]

# dark current in e-
dark_curr_e_full_frame = dark_curr_adu_full_frame * gain # entire frame of dark electrons [e/sec]
dark_curr_e = dark_curr_adu * gain  # dark current [e/pix/sec]

# plot histogram
plot_file_name = 'scratch_nirao_09_hist_e_counts.png'
plt.hist(dark_curr_e_full_frame[np.isfinite(dark_curr_e_full_frame)], bins=100)
plt.title('Histogram of e- across detector')
plt.xlabel('N_e in a pixel')
plt.savefig(plot_file_name)

# criterion for success:
# Dark noise < 0.1 e-/px/s.s
logger.info('-----------------------------------------------------')
logger.info('-----------------------------------------------------')
logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': Fraction of bad pixels: {:.5f}'.format(1. - frac_finite))
logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': Criterion for success: Dark noise < 0.1 e-/px/sec')
logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': Measured dark current: ' + str(dark_curr_e) + ' e/pix/sec')
logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': Wrote plot ' + plot_file_name)
logger.info('--------------------------------------------------')
if dark_curr_e < 0.1:
    logger.info('######   NIRAO-09 result: PASS   ######')
elif dark_curr_e > 0.1:
    logger.info('######   NIRAO-09 result: FAIL   ######')
logger.info('--------------------------------------------------')