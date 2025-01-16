import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
import logging
import datetime
import glob
import ipdb
import scipy
from scipy import fftpack
from photutils.centroids import centroid_sources, centroid_com
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from skimage.transform import resize
from PIL import Image
from skimage import color, data, restoration
from image_registration import chi2_shift


def dark_subt(raw_science_frame_file_names, dark_array):
    # dark subtracts

    hdul = fits.open(raw_science_frame_file_names)
    sci = hdul[0].data

    sci = sci - dark_array

    return median_frame


def main(data_date = '20240919'):
    # 20240919 is best data

    # start logging
    log_file_name = 'log_nirao_01_installation_' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.txt'

    logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(message)s',
                    handlers=[
                        logging.FileHandler(log_file_name),
                        logging.StreamHandler()
                    ])

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)
    logger = logging.getLogger(__name__)

    if data_date == '20240919':
        stem = '/Users/eckhartspalding/Documents/git.repos/dirac/vtp_scripts_data/nirao_01_installation/data/20240919/'
        dark_frame_file_names = glob.glob(stem + 'calibs/darks/*.fits') # darks from 20240919
        position_baseline_frames = glob.glob(stem + 'position_baseline/*.fits')
        position_1_frames = glob.glob(stem + 'position_1/*.fits')
        position_2_frames = glob.glob(stem + 'position_2/*.fits')
        position_3_frames = glob.glob(stem + 'position_3/*.fits')
        position_4_frames = glob.glob(stem + 'position_4/*.fits')
        position_5_frames = glob.glob(stem + 'position_5/*.fits')

    logger.info('-----------------------------------------------------')
    logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': NIRAO-01 Installation test')
    logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': Criterion for success: PSF centroids '+\
                'are within 22 pixels of each other across all images (corresponds to 0.2 mm camera position)')
    logger.info('-----------------------------------------------------')
    logger.info('Dark frames: ' + str(dark_frame_file_names))
    logger.info('Position baseline frames: ' + str(position_baseline_frames))
    logger.info('Position 1 frames: ' + str(position_1_frames))
    logger.info('Position 2 frames: ' + str(position_2_frames))
    logger.info('Position 3 frames: ' + str(position_3_frames))
    logger.info('Position 4 frames: ' + str(position_4_frames))
    logger.info('Position 5 frames: ' + str(position_5_frames))

    badpix_file_name = stem + 'calibs/ersatz_bad_pix.fits'

    # bad pixel frame (0: good, 1: bad)
    # (N.b. these pixels are masked in the detector readout, not corrected)
    hdul_badpix = fits.open(badpix_file_name, dtype=int)
    badpix = hdul_badpix[0].data

    # make net dark
    dark_array = []
    for file_name in dark_frame_file_names:
        hdul = fits.open(file_name)
        dark_this = hdul[0].data
        dark_array.append(dark_this)
    
    dark_median = np.median(dark_array, axis=0)
    # mask bad pixels
    dark_median[badpix == 1] = np.nan

    # total number of pixels 
    N_pix_tot = dark_median.size
    # how many pixels are finite?
    N_pix_finite = np.nansum(np.isfinite(dark_median))
    # fraction of good pixels within science region of detector (i.e., 4-pixel-wide overscan region of 16320 pixels removed)
    frac_finite = N_pix_finite/(N_pix_tot - 16320)

    def med_frame(file_name_array):
        data_array = []
        for file_name in file_name_array:
            hdul = fits.open(file_name)
            data_this = hdul[0].data
            data_this = np.subtract(data_this, dark_median)  # dark-subtract
            data_array.append(data_this)
        
        # Convert the list to a 3D NumPy array
        data_array = np.array(data_array)
        
        # Compute the median along the first axis
        frame_median = np.median(data_array, axis=0)
        return frame_median
                  

    pos_baseline = med_frame(position_baseline_frames)
    pos_1 = med_frame(position_1_frames)
    pos_2 = med_frame(position_2_frames)
    pos_3 = med_frame(position_3_frames)
    pos_4 = med_frame(position_4_frames)
    pos_5 = med_frame(position_5_frames)

    # centroid on the spots
    x_cen_baseline, y_cen_baseline = centroid_sources(data=pos_baseline, xpos=[536], ypos=[559], box_size=21, centroid_func=centroid_com)
    x_cen_1, y_cen_1 = centroid_sources(data=pos_1, xpos=[536], ypos=[559], box_size=21, centroid_func=centroid_com)
    x_cen_2, y_cen_2 = centroid_sources(data=pos_2, xpos=[536], ypos=[559], box_size=21, centroid_func=centroid_com)
    x_cen_3, y_cen_3 = centroid_sources(data=pos_3, xpos=[536], ypos=[559], box_size=21, centroid_func=centroid_com)
    x_cen_4, y_cen_4 = centroid_sources(data=pos_4, xpos=[536], ypos=[559], box_size=21, centroid_func=centroid_com)
    x_cen_5, y_cen_5 = centroid_sources(data=pos_5, xpos=[536], ypos=[559], box_size=21, centroid_func=centroid_com)

    x_cen_array = [x_cen_baseline, x_cen_1, x_cen_2, x_cen_3, x_cen_4, x_cen_5]
    y_cen_array = [y_cen_baseline, y_cen_1, y_cen_2, y_cen_3, y_cen_4, y_cen_5]

    # absolute distances from baseline

    # distances from the baseline
    plot_file_name = 'nirao_01_installation_' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.png'
    plt.scatter(x_cen_array, y_cen_array)
    circle = plt.Circle((x_cen_baseline, y_cen_baseline), radius=11, color='r', fill=False, linestyle='--')
    plt.gca().add_patch(circle)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('NIRAO-1: Spot centers on detector between lifts\n(Red circle has diameter 22 pixels)')
    plt.xlabel('x (pix)')
    plt.ylabel('y (pix)')
    plt.savefig(plot_file_name)

    def are_all_pairs_within_distance(x_array, y_array, N):
        num_points = len(x_array)
        for i in range(num_points):
            for j in range(i + 1, num_points):
                distance = np.sqrt((x_array[j] - x_array[i])**2 + (y_array[j] - y_array[i])**2)
                if distance > N:
                    return False
        return True

    # Example usage
    N = 22  # Define the distance threshold
    result = are_all_pairs_within_distance(x_cen_array, y_cen_array, N)

    logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': -----------------------------------------------------')
    logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': -----------------------------------------------------')
    logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': Fraction of bad pixels: {:.5f}'.format(1. - frac_finite))
    logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': ----------')
    # 0.1 * lambda/D @ 900 nm is equivalent to
    # 0.1 * (0.9e-6 m) * 206265" /( 4 m * 32.7e-3 "/pix) = 0.14 pix
    logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': Criterion for success: PSF centroids '+\
                'are within 22 pixels of each other across all images (corresponds to 0.2 mm camera position)')
    logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': Condition fulfilled: ' + str(result))
    logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': ----------')
    logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': Wrote plot ' + plot_file_name)
    logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': --------------------------------------------------')


if __name__ == "__main__":
    main(data_date = '20240919')