# calculates the dark current

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
import logging
import datetime
import glob
from photutils.centroids import centroid_sources, centroid_com

log_file_name = 'log_nirao_12_plate_scales_' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.txt'
logging.basicConfig(filename=log_file_name, 
                    level=logging.INFO, format='%(message)s')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(message)s')
console.setFormatter(formatter)
logging.getLogger().addHandler(console)
logger = logging.getLogger()

stem = '/Users/bandari/Documents/git.repos/dirac/vtp_scripts_data/nirao_12_plate_scales/data/'

logger.info('-----------------------------------------------------')
logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': NIRAO-12 Plate Scale test')
logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': Criterion for success: Measured positions of PSF matches the expected camera pixel position in VTP Table 1 (i.e., PS=32.7 mas/pix).')
logger.info('-----------------------------------------------------')

dark_raw_file_name = stem + 'data/tests_junk_13may/pos1_selected_cold_target_not_cold/20sec.fits'
bias_file_name = stem + 'data/tests_junk_13may/pos1_selected_cold_target_not_cold/100ms.fits'
badpix_file_name = stem + 'ersatz_bad_pix.fits'

# positions of micrometer [mm]
microm_xy = {'cen':np.array([6.565,6.222]),
             'ul':np.array([10.943,1.875]), 
             'ur':np.array([2.190,1.850]), 
             'll':np.array([10.905,10.570]), 
             'lr':np.array([2.205,10.580])}

logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': Raw dark: ' + dark_raw_file_name)
logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': Bias: ' + bias_file_name)
logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': Bad pixel mask: ' + badpix_file_name)

simple_bias_file_names = [stem + 'DIRAC_20240515_135139.fits', stem + 'DIRAC_20240515_135152.fits']
cds_bias_file_names = [stem + 'DIRAC_20240515_135206.fits', stem + 'DIRAC_20240515_135223.fits']

# bad pixel frame (0: good, 1: bad)
# (N.b. these pixels are masked in the detector readout, not corrected)
hdul_badpix = fits.open(badpix_file_name, dtype=int)
badpix = hdul_badpix[0].data

# make net bias
bias_array = []
for file_name in simple_bias_file_names:
    hdul = fits.open(file_name)
    bias_this = hdul[0].data
    bias_array.append(bias_this)
bias_simple = np.median(bias_array, axis=0)
# mask bad pixels
bias_simple[badpix == 1] = np.nan

# total number of pixels 
N_pix_tot = bias_simple.size
# how many pixels are finite?
N_pix_finite = np.nansum(np.isfinite(bias_simple))
# fraction of good pixels within science region of detector (i.e., 4-pixel-wide overscan region of 16320 pixels removed)
frac_finite = N_pix_finite/(N_pix_tot - 16320)

# read in dark
'''
hdul_dark_raw = fits.open(dark_raw_file_name)
dark_raw = hdul_dark_raw[0].data.astype(np.float32) # convert to float to allow NaNs
dark_raw[badpix == 1] = np.nan
'''

def dark_subt_take_median(raw_science_frame_file_names, bias_array):
    # reads in list of raw frames, subtracts bias, and returns median of residuals

    test_array = []
    for i in range(0,len(raw_science_frame_file_names)):

        hdul = fits.open(raw_science_frame_file_names[i])
        sci = hdul[0].data

        sci = sci - bias_array

        test_array.append(sci)

    median_frame = np.median(test_array, axis=0)

    return median_frame

# read/process set of frames corresponding to upper left (of micrometer space; coords are flipped on readout)
ul_raw_frame_file_names = glob.glob(stem + 'DIRAC_20240515_11523[5-9].fits')
ul_raw_frame_file_names += glob.glob(stem + 'DIRAC_20240515_11524*.fits')
ul_raw_frame_file_names += glob.glob(stem + 'DIRAC_20240515_11525[0-4]*.fits')
ul = dark_subt_take_median(raw_science_frame_file_names=ul_raw_frame_file_names, bias_array=bias_simple)

# same, for upper right
ur_raw_frame_file_names = glob.glob(stem + 'DIRAC_20240515_11444[2-9].fits')
ur_raw_frame_file_names += glob.glob(stem + 'DIRAC_20240515_1144[5-9]*.fits')
ur_raw_frame_file_names += glob.glob(stem + 'DIRAC_20240515_11450[0-1]*.fits')
ur = dark_subt_take_median(raw_science_frame_file_names=ur_raw_frame_file_names, bias_array=bias_simple)

# same, for lower left
ll_raw_frame_file_names = glob.glob(stem + 'DIRAC_20240515_11385[7-9].fits')
ll_raw_frame_file_names += glob.glob(stem + 'DIRAC_20240515_1138[6-9]*.fits')
ll_raw_frame_file_names += glob.glob(stem + 'DIRAC_20240515_11390[0-7]*.fits')
ll = dark_subt_take_median(raw_science_frame_file_names=ll_raw_frame_file_names, bias_array=bias_simple)

# same, for lower right
lr_raw_frame_file_names = glob.glob(stem + 'DIRAC_20240515_11411[6-9].fits')
lr_raw_frame_file_names += glob.glob(stem + 'DIRAC_20240515_11412*.fits')
lr_raw_frame_file_names += glob.glob(stem + 'DIRAC_20240515_114130.fits')
lr = dark_subt_take_median(raw_science_frame_file_names=lr_raw_frame_file_names, bias_array=bias_simple)

# same, for center
cen_raw_frame_file_names = glob.glob(stem + 'DIRAC_20240515_11252[7-9].fits')
cen_raw_frame_file_names += glob.glob(stem + 'DIRAC_20240515_11253*.fits')
cen_raw_frame_file_names += glob.glob(stem + 'DIRAC_20240515_11254[0-1].fits')
cen = dark_subt_take_median(raw_science_frame_file_names=cen_raw_frame_file_names, bias_array=bias_simple)

# centroid the PSFs
x_ul, y_ul = centroid_sources(data=ul, xpos=[998], ypos=[30], box_size=21, centroid_func=centroid_com)
x_ur, y_ur = centroid_sources(data=ur, xpos=[21], ypos=[30], box_size=21, centroid_func=centroid_com)
x_ll, y_ll = centroid_sources(data=ll, xpos=[1000], ypos=[1000], box_size=21, centroid_func=centroid_com)
x_lr, y_lr = centroid_sources(data=lr, xpos=[23], ypos=[1000], box_size=21, centroid_func=centroid_com)
x_cen, y_cen = centroid_sources(data=cen, xpos=[511], ypos=[511], box_size=21, centroid_func=centroid_com)

# find the distances between them

# UL del_x, del_y from center
del_y_ul = (y_ul-y_cen)[0]
del_x_ul = (x_ul-x_cen)[0]
del_y_ur = (y_ur-y_cen)[0]
del_x_ur = (x_ur-x_cen)[0]
del_y_ll = (y_ll-y_cen)[0]
del_x_ll = (x_ll-x_cen)[0]
del_y_lr = (y_lr-y_cen)[0]
del_x_lr = (x_lr-x_cen)[0]

# result

# plot histogram
'''
plot_file_name = 'scratch_nirao_09_hist_e_counts.png'
plt.hist(dark_curr_e_full_frame[np.isfinite(dark_curr_e_full_frame)], bins=100)
plt.title('Histogram of e- across detector')
plt.xlabel('N_e in a pixel')
plt.savefig(plot_file_name)
'''

# criterion for success:
# Plate scale 32.7 mas/pix
logger.info('-----------------------------------------------------')
logger.info('-----------------------------------------------------')
logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': Fraction of bad pixels: {:.5f}'.format(1. - frac_finite))
logger.info('----------')
logger.info('Magnitude of expected offsets [pix]: ' + str(3.689 * 4.2 * 1e3 / 32.7 )) # constant * mm  PS
logger.info('Upper left del_x [pix]: ' + str(del_x_ul))
logger.info('Upper left del_y [pix]: ' + str(del_y_ul))
logger.info('Upper right del_x [pix]: ' + str(del_x_ur))
logger.info('Upper right del_y [pix]: ' + str(del_y_ur))
logger.info('Lower left del_x [pix]: ' + str(del_x_ll))
logger.info('Lower left del_y [pix]: ' + str(del_y_ll))
logger.info('Lower right del_x [pix]: ' + str(del_x_lr))
logger.info('Lower right del_y [pix]: ' + str(del_y_lr))
logger.info('----------')
logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': Criterion for success: Plate scale 32.7 mas/pix')
#logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': Measured plate scale: ' + str(dark_curr_e) + ' e/pix/sec')
#logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': Wrote plot ' + plot_file_name)
logger.info('--------------------------------------------------')
'''
if dark_curr_e < 0.1:
    logger.info('######   NIRAO-09 result: PASS   ######')
elif dark_curr_e > 0.1:
    logger.info('######   NIRAO-09 result: FAIL   ######')
'''
logger.info('--------------------------------------------------')