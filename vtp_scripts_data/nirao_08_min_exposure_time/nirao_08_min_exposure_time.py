# calculates the dark current

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
import logging
import datetime
import glob
import ipdb

def main(data_date = '20240517'):
    # 20240517 is best data

    # start logging
    log_file_name = 'log_nirao_08_min_exposure_time_' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.txt'
    logging.basicConfig(filename=log_file_name, 
                        level=logging.INFO, format='%(message)s')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)
    logger = logging.getLogger(__name__)

    if data_date == '20240517':
        stem = '/Users/eckhartspalding/Documents/git.repos/dirac/vtp_scripts_data/nirao_08_min_exposure_time/data/20240517/'

    logger.info('-----------------------------------------------------')
    logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': NIRAO-08 Minimum Exposure Time test')
    logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': Criterion for success: Confirm successful acquisition of frames with exposure times < 1.2s. Confirm read noise in short exposure frames frames < 18e-.')
    logger.info('-----------------------------------------------------')

    #dark_raw_file_name = stem + 'data/tests_junk_13may/pos1_selected_cold_target_not_cold/20sec.fits'
    #bias_file_name = stem + 'data/tests_junk_13may/pos1_selected_cold_target_not_cold/100ms.fits'
    badpix_file_name = stem + '../calibs/ersatz_bad_pix.fits'

    # bad pixel frame (0: good, 1: bad)
    # (N.b. these pixels are masked in the detector readout, not corrected)
    hdul_badpix = fits.open(badpix_file_name, dtype=int)
    badpix = hdul_badpix[0].data

    ## TBD: Check that frames are in CDS mode

    # read in bias-only frames
    if data_date == '20240517':
        file_list_biases = glob.glob(stem + 'DIRAC_20240517_115929.fits')
        file_list_biases += glob.glob(stem + 'DIRAC_20240517_1159[3-9]*.fits')
        file_list_biases += glob.glob(stem + 'DIRAC_20240517_11[6-9]*.fits')
        file_list_biases += glob.glob(stem + 'DIRAC_20240517_1200[0-1]*.fits')
        file_list_biases += glob.glob(stem + 'DIRAC_20240517_120020*.fits')
    
    logger.info('Bias-only frames: ' + str(file_list_biases))

    # total number of pixels 
    sample_frame_file_name = file_list_biases[0]
    hdul = fits.open(sample_frame_file_name)
    sample_frame = hdul[0].data.astype(np.float32)
    sample_frame[badpix == 1] = np.nan
    N_pix_tot = sample_frame.size
    # how many pixels are finite?
    N_pix_finite = np.nansum(np.isfinite(sample_frame))
    # fraction of good pixels within science region of detector (i.e., 4-pixel-wide overscan region of 16320 pixels removed)
    frac_finite = N_pix_finite/(N_pix_tot - 16320)

    gain = 1.78 # [e/ADU] (email, A. Vaccarella, 2024 05 17)
    logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': Used gain value [e/ADU]: ' + str(gain))

    # read in all frames (darks and biases) and print exposure times
    file_name_list = glob.glob(stem + "*.fits")
    exp_times_array = []
    # extract exposure times from all frames
    for i in range(0,len(file_name_list)):
        hdul = fits.open(file_name_list[i])
        hdr = hdul[0].header
        exp_times_array.append(hdr['EXPTIME'])

    # find read noise of bias frames
    read_noise_array_adu = []
    # get all pair (N=2) permutations of M file names (N pick M)
    pairs = []
    for i in range(len(file_list_biases)):
        for j in range(i+1, len(file_list_biases)):
            pair = [file_list_biases[i], file_list_biases[j]]
            pairs.append(pair)

    for i in range(0,len(pairs)):
        hdul_1 = fits.open(pairs[i][0])
        data_1 = hdul_1[0].data.astype(np.float32)
        data_1[badpix == 1] = np.nan

        hdul_2 = fits.open(pairs[i][1])
        data_2 = hdul_2[0].data.astype(np.float32)
        data_2[badpix == 1] = np.nan

        diff = data_1 - data_2

        # find read noise
        read_noise = np.nanstd(diff)/np.sqrt(2)

        read_noise_array_adu.append(read_noise)


    read_noise_e_array = gain * np.array([read_noise_array_adu])
    read_noise_e_mean = gain * np.mean( read_noise_array_adu )

    hist_file_name = 'hist_nirao_08_min_exposure_time_' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.png'

    plt.clf()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Global plot
    ax1.hist(read_noise_e_array, bins=100)
    ax1.axvline(x=18, linestyle='--', color='k', alpha=0.5)
    ax1.set_xlabel('Read noise (e)')
    ax1.set_title('global')

    # Restricted x-axis plot
    ax2.hist(read_noise_e_array, bins=100)
    ax2.axvline(x=18, linestyle='--', color='k', alpha=0.5)
    ax2.set_xlabel('Read noise (e)')
    ax2.set_xlim([0, 30])  # Adjust the x-axis limits as needed
    ax2.set_title('detail')

    # Save the figure
    plt.suptitle('Histogram of read noise\n(vertical line: success criterion)')
    plt.tight_layout()
    plt.savefig(hist_file_name)
    logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': Wrote ' + hist_file_name)
        
    logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': Exposure times of mix of flats and bias frames [sec]: ' + str(exp_times_array))
    logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': -----------------------------------------------------')
    logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': -----------------------------------------------------')
    logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': Fraction of bad pixels: {:.5f}'.format(1. - frac_finite))
    logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': ----------')
    logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': Criterion for success: Exposure times [sec]: <=1.2 ')
    logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': Min. exposure times of collection of flats and bias frames: ' + str(np.min(exp_times_array)))
    logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': Max. exposure times of collection of flats and bias frames: ' + str(np.max(exp_times_array)))
    logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': ----------')
    logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': Criterion for success: Read noise [e]: < 18 ')
    logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': Measured read noise [e]: {:.2f}'.format(read_noise_e_mean))
    logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': Percent variation from ideal: {:.1f}%'.format(100 * np.abs(read_noise_e_mean-18.)/18.))
    logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': --------------------------------------------------')
    
    if read_noise_e_mean < 18 and np.max(exp_times_array) <= 1.2:
        logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': ######   NIRAO-08 result: PASS   ######')
    else:
        logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': ######   NIRAO-08 result: FAIL   ######')

    logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': --------------------------------------------------')

if __name__ == "__main__":
    main(data_date = '20240517')