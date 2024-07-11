# calculates the dark current

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
import logging
import datetime
import glob
import scipy
import pandas as pd
from scipy import fftpack
from photutils.centroids import centroid_sources, centroid_com
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from skimage.transform import resize
from scipy.optimize import curve_fit
from PIL import Image
from skimage import color, data, restoration


def gaussian_2d(xy_mesh, amplitude, xo, yo, sigma_x_pix, sigma_y_pix, theta):
    x, y = xy_mesh
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta)**2) / (2 * sigma_x_pix**2) + (np.sin(theta)**2) / (2 * sigma_y_pix**2)
    b = -(np.sin(2 * theta)) / (4 * sigma_x_pix**2) + (np.sin(2 * theta)) / (4 * sigma_y_pix**2)
    c = (np.sin(theta)**2) / (2 * sigma_x_pix**2) + (np.cos(theta)**2) / (2 * sigma_y_pix**2)
    g = amplitude * np.exp(-(a * ((x - xo)**2) + 2 * b * (x - xo) * (y - yo) + c * ((y - yo)**2)))
    return g.ravel()


def deconvolve(star, psf):
    star_fft = fftpack.fftshift(fftpack.fftn(star))
    psf_fft = fftpack.fftshift(fftpack.fftn(psf))
    return fftpack.fftshift(fftpack.ifftn(fftpack.ifftshift(star_fft/psf_fft)))


def gen_model_tbs_psf(raw_cutout_size = 25, upsampling = 4):
    # generate model PSF (Gaussian approximation) of the telescope beam simulator
    # based on FWHM = 6 pix * (5.2 um / pix) = 31.2 um
    # and DIRAC pitch of 18 um / pix --> FWHM is 31.2 um * (pix / 18 um) = 1.733 pix on DIRAC detector

    # PARAMETERS:
    # raw_cutout_size: edge lengths of cutout from DIRAC detector (before any upsampling)
    # upsampling: upsampling we intend to apply

    print('NEED TO CORREC THE PARAMS ON THIS MODEL PSF')
    ## import ipdb; ipdb.set_trace()

    size = int(raw_cutout_size * upsampling)
    tbs_psf_model = np.zeros((size, size))

    # FWHM = 2.35 * sigma (Gaussian)
    sigma_x = 1.733 * upsampling / 2.35 # (DIRAC pix) * upsampling / 2.35
    sigma_y = 1.733 * upsampling / 2.35

    # make grid
    x = np.linspace(-int(0.5*size), int(0.5*size), size)
    y = np.linspace(-int(0.5*size), int(0.5*size), size)

    x, y = np.meshgrid(x, y)
    # 2D Gaussian
    z = (1/(2*np.pi*sigma_x*sigma_y) * np.exp(-(x**2/(2*sigma_x**2) + y**2/(2*sigma_y**2))))

    #plt.contourf(x, y, z, cmap='Blues')
    #plt.colorbar()

    #plt.imshow(z)

    #plt.show()

    return z


def get_model_tbs_psf_based_on_empirical(raw_cutout_size = 100, upsampling = 1):
    # generate model of TBS, based on a Gaussian fit to what was measured

    size = raw_cutout_size * upsampling
    #tbs_psf_model = np.zeros((size, size))

    # PSF of the TBS, as measured Apr. 12, 2024 (email from RZ)
    # Note the pixels are the TBS pixels (not DIRAC!)
    psf_tbs_empirical = np.array([[4, 4,  4,  4,  4],
                    [5, 8,  16, 19, 16],
                    [16,43, 79, 80, 52],
                    [50, 129, 186, 175, 115],
                    [103, 212, 236, 226, 180],
                    [132, 236, 233, 216, 195],
                    [95, 198, 230, 216, 159],
                    [38, 93, 151, 148, 94],
                    [12, 25, 47, 48, 31],
                    [5, 5, 8, 7, 6],
                    [3, 3, 4, 3, 3]]) 
    # fit Gaussian model to TBS PSF
    tbs_fit_result, tbs_fwhm_x_pix_tbs, tbs_fwhm_y_pix_tbs, tbs_sigma_x_pix_tbs, tbs_sigma_y_pix_tbs = fit_gaussian(psf_tbs_empirical, np.array([2.5, 5]))
    #import ipdb; ipdb.set_trace()
    # convert to um with factors of 5.2 um/pix (TBS pixel pitch) and 2 (DIRAC camera magnification) from email from RZ, 2024 04 12
    tbs_fwhm_x_um = tbs_fwhm_x_pix_tbs * 5.2 * 2
    tbs_fwhm_y_um = tbs_fwhm_y_pix_tbs * 5.2 * 2
    tbs_sigma_x_um = tbs_sigma_x_pix_tbs * 5.2 * 2
    tbs_sigma_y_um = tbs_sigma_y_pix_tbs * 5.2 * 2
    # convert to DIRAC pixels
    tbs_fwhm_x_pix_dirac = tbs_fwhm_x_um / 18. # DIRAC pixel pitch: 18 um/pix
    tbs_fwhm_y_pix_dirac = tbs_fwhm_y_um / 18.
    tbs_sigma_x_pix_dirac = tbs_sigma_x_um / 18.
    tbs_sigma_y_pix_dirac = tbs_sigma_y_um / 18.

    import ipdb; ipdb.set_trace()

    print('tbs_fwhm_x_um',tbs_fwhm_x_um)
    print('tbs_fwhm_y_um',tbs_fwhm_y_um)

    plt.clf()
    # save a plot of the psf_tbs
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    # Plot psf_tbs_empirical
    im1 = axs[0].imshow(psf_tbs_empirical, cmap='viridis')
    axs[0].set_title('psf_tbs_empirical')
    # Plot tbs_fit_result
    im0 = axs[1].imshow(tbs_fit_result, cmap='viridis')
    axs[1].set_title('psf_tbs_bestfit')
    # Plot resids_psf_tbs
    resids_psf_tbs = tbs_fit_result - psf_tbs_empirical
    im2 = axs[2].imshow(resids_psf_tbs, cmap='viridis')
    axs[2].set_title('resids_psf_tbs')
    # Set the same color scale
    vmin = min(im0.get_array().min(), im1.get_array().min(), im2.get_array().min())
    vmax = max(im0.get_array().max(), im1.get_array().max(), im2.get_array().max())
    im0.set_clim(vmin, vmax)
    im1.set_clim(vmin, vmax)
    im2.set_clim(vmin, vmax)
    plt.tight_layout()
    file_name_psf_tbs = 'psf_tbs.png'
    plt.savefig(file_name_psf_tbs)

    #import ipdb; ipdb.set_trace()

    #-----
    #tbs_psf_model = np.zeros((size, size))

    # FWHM = 2.35 * sigma (Gaussian)
    #sigma_x = 1.733 * upsampling / 2.35 # (DIRAC pix) * upsampling / 2.35
    #sigma_y = 1.733 * upsampling / 2.35

    # upsampling
    tbs_sigma_x_pix_dirac *= tbs_sigma_x_pix_dirac * upsampling
    tbs_sigma_y_pix_dirac *= tbs_sigma_y_pix_dirac * upsampling
    tbs_fwhm_x_pix_dirac *= tbs_fwhm_x_pix_dirac * upsampling
    tbs_fwhm_y_pix_dirac *= tbs_fwhm_y_pix_dirac * upsampling
    #import ipdb; ipdb.set_trace()

    # make grid
    x = np.linspace(-int(0.5*size), int(0.5*size), size)
    y = np.linspace(-int(0.5*size), int(0.5*size), size)

    x, y = np.meshgrid(x, y)
    # 2D Gaussian
    z = (1/(2 * np.pi * tbs_sigma_x_pix_dirac * tbs_sigma_y_pix_dirac) * np.exp(-(x**2/(2*tbs_sigma_x_pix_dirac**2) + y**2/(2*tbs_sigma_y_pix_dirac**2))))

    if upsampling != 1:
        print('WARNING: UPSAMPLING != 1; SIGMA AND FWHM VALS NEED TO BE RESCALED')

    #import ipdb; ipdb.set_trace()
    return z, tbs_sigma_x_pix_tbs, tbs_sigma_y_pix_tbs, tbs_fwhm_x_pix_tbs, tbs_fwhm_y_pix_tbs


def testing_gaussian(raw_cutout_size = 25, upsampling = 1, sigma_expansion = 2):
    # generate model PSF of the telescope beam simulator
    # model based on FWHM = 6 pix * (5.2 um / pix) = 31.2 um
    # and DIRAC pitch of 18 um / pix --> FWHM is 31.2 um * (pix / 18 um) = 1.733 pix on DIRAC detector

    # PARAMETERS:
    # raw_cutout_size: edge lengths of cutout from DIRAC detector (before any upsampling)
    # upsampling: upsampling we intend to apply
    # sigma_expansion: factor by which to make the Gaussian larger than the TBS PSF

    size = raw_cutout_size * upsampling
    tbs_psf_model = np.zeros((size, size))

    # FWHM = 2.35 * sigma (Gaussian)
    sigma_x = sigma_expansion * 1.733 * upsampling / 2.35 # (DIRAC pix) * upsampling / 2.35
    sigma_y = sigma_expansion * 1.733 * upsampling / 2.35

    # make grid
    x = np.linspace(-int(0.5*size), int(0.5*size), size)
    y = np.linspace(-int(0.5*size), int(0.5*size), size)

    x, y = np.meshgrid(x, y)
    # 2D Gaussian
    z = (1/(2*np.pi*sigma_x*sigma_y) * np.exp(-(x**2/(2*sigma_x**2) + y**2/(2*sigma_y**2))))

    #plt.contourf(x, y, z, cmap='Blues')
    #plt.colorbar()

    #plt.imshow(z)

    #plt.show()

    return z


# confidence ellipse from https://matplotlib.org/stable/gallery/statistics/confidence_ellipse.html
def confidence_ellipse(x, y, ax, n_std=1.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The Axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)

    return scale_x, scale_y, ax.add_patch(ellipse)


def fit_gaussian(frame, center_guess):
    y, x = np.indices(frame.shape)
    xy_mesh = (x, y)
    p0 = [np.max(frame), center_guess[0], center_guess[1], 1, 1, 0]
    popt, pcov = curve_fit(gaussian_2d, xy_mesh, frame.ravel(), p0=p0)
    fitted_array = gaussian_2d(xy_mesh, *popt).reshape(frame.shape)
    fwhm_x_pix = 2 * np.sqrt(2 * np.log(2)) * np.abs(popt[3])
    fwhm_y_pix = 2 * np.sqrt(2 * np.log(2)) * np.abs(popt[4])
    sigma_x_pix = popt[3]
    sigma_y_pix = popt[4]
    return fitted_array, fwhm_x_pix, fwhm_y_pix, sigma_x_pix, sigma_y_pix


def strehl_based_on_peak_intensities(frame, center_guess, badpix):
    # fits a Gaussian, but without constraining it with the central pixels

    #import ipdb; ipdb.set_trace()

    y, x = np.indices(frame.shape)
    xy_mesh = (x, y)
    p0 = [np.max(frame), center_guess[0], center_guess[1], 1, 1, 0]
    # find centroid
    popt, pcov = curve_fit(gaussian_2d, xy_mesh, frame.ravel(), p0=p0)

    # mask some bad pixels
    frame[badpix == 1] = np.nan

    # to avoid effect of bad pixels, only consider max within small region around spot
    buffer_size = 10
    cutout_around_psf = frame[int(popt[2])-buffer_size:int(popt[2])+buffer_size,int(popt[1])-buffer_size:int(popt[1])+buffer_size]
    strehl_simple = np.nanmax(cutout_around_psf)/popt[0]
    
    return strehl_simple


def dark_subt(raw_science_frame_file_names, dark_array):
    # dark subtracts

    hdul = fits.open(raw_science_frame_file_names)
    sci = hdul[0].data

    sci = sci - dark_array

    return median_frame


def main(data_date = '20240710'):
    # 20240517 is best data

    # upsampling for the deconvolution
    upsampling = 1
    # sizes of square cutouts from detector, in DIRAC pixels
    raw_cutout_size = 100

    # start logging
    log_file_name = 'log_nirao_14_image_quality_' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.txt'
    logging.basicConfig(filename=log_file_name, 
                        level=logging.INFO, format='%(message)s')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)
    logger = logging.getLogger()

    if data_date == '20240710':
        stem = '/Users/bandari/Documents/git.repos/dirac/vtp_scripts_data/nirao_14_image_quality/data/20240710/'
        dark_frame_file_names = glob.glob(stem + 'calibs/darks/*.fits') # darks from 20240610

    logger.info('-----------------------------------------------------')
    logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': NIRAO-14 Image Quality test')
    logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': Criterion for success: Strehl ratio measured for all PSF locations > 90%')
    logger.info('-----------------------------------------------------')

    #dark_raw_file_name = stem + 'data/tests_junk_13may/pos1_selected_cold_target_not_cold/20sec.fits'
    #bias_file_name = stem + 'data/tests_junk_13may/pos1_selected_cold_target_not_cold/100ms.fits'
    badpix_file_name = stem + 'calibs/ersatz_bad_pix.fits'

    # positions of micrometer [mm]

    #logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': Raw dark: ' + dark_raw_file_name)
    #logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': Bias: ' + bias_file_name)
    #logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': Bad pixel mask: ' + badpix_file_name)

    #simple_bias_file_names = [stem + 'DIRAC_20240515_135139.fits', stem + 'DIRAC_20240515_135152.fits']
    #cds_bias_file_names = [stem + 'DIRAC_20240515_135206.fits', stem + 'DIRAC_20240515_135223.fits']

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
    #dark_median[badpix == 1] = np.nan

    # total number of pixels 
    N_pix_tot = dark_median.size
    # how many pixels are finite?
    N_pix_finite = np.nansum(np.isfinite(dark_median))
    # fraction of good pixels within science region of detector (i.e., 4-pixel-wide overscan region of 16320 pixels removed)
    frac_finite = N_pix_finite/(N_pix_tot - 16320)

    # read in dark
    '''
    hdul_dark_raw = fits.open(dark_raw_file_name)
    dark_raw = hdul_dark_raw[0].data.astype(np.float32) # convert to float to allow NaNs
    dark_raw[badpix == 1] = np.nan
    '''

    # Read in spot coordinate guesses
    df_coord_guesses = pd.read_csv(stem + 'filenames_coord_guesses.txt', delimiter=',')
    # make file names absolute
    df_coord_guesses['filename'] = stem + df_coord_guesses['filename']

    # TBS PSF, as projected onto the DIRAC pixels
    psf_tbs, tbs_sigma_x_pix_tbs, tbs_sigma_y_pix_tbs, tbs_fwhm_x_pix_tbs, tbs_fwhm_y_pix_tbs = get_model_tbs_psf_based_on_empirical(raw_cutout_size = raw_cutout_size, upsampling = upsampling)
    
    # TBS 5.2 um/pix; camera magnification of 2x
    #fwhm_tbs_um = 0.5 * (tbs_fwhm_x_pix_tbs + tbs_fwhm_y_pix_tbs) * 5.2 * 2 # average FWHM of TBS in DIRAC pixels
    fwhm_tbs_um = (np.min([tbs_fwhm_x_pix_tbs,tbs_fwhm_y_pix_tbs])) * 5.2 * 2 # average FWHM of TBS in DIRAC pixels
    #import ipdb; ipdb.set_trace()


    # Read the text file with coord guesses into a Pandas DataFrame
    df_coord_guesses = pd.read_csv(stem + 'filenames_coord_guesses.txt', delimiter=',')
    # make file names absolute
    df_coord_guesses['filename'] = stem + df_coord_guesses['filename']
    
    file_names = df_coord_guesses['filename'].values
    #import ipdb; ipdb.set_trace()
    # put all the (x,y) guesses of the spot centers into a list
    # x, y convention
    coord_guess = []
    for i in range(len(df_coord_guesses)):
        x_guess = df_coord_guesses['x_guess'].iloc[i]
        y_guess = df_coord_guesses['y_guess'].iloc[i]
        coord_guess.append(np.array([x_guess, y_guess]))
    #import ipdb; ipdb.set_trace()
    # read/process set of frames corresponding to upper left (of micrometer space; coords are flipped on readout)
    df = pd.DataFrame(columns=['spot number', 'fwhm_x_pix', 'fwhm_y_pix', 'x_pos_pix', 'y_pos_pix', 'fwhm_tbs_um']) # initialize
    #import ipdb; ipdb.set_trace()
    # loop over all frames
    for i in range(0,len(df_coord_guesses['filename'].values)):

        file_name_this = df_coord_guesses['filename'].values[i]

        hdul = fits.open(file_name_this)
        sci_this = hdul[0].data
        #import ipdb; ipdb.set_trace()


        #frame_this = dark_subt_take_median(raw_science_frame_file_names=frame_name, dark_array=dark_simple)
        frame_this = sci_this - dark_median
        #import ipdb; ipdb.set_trace()

        cookie_edge_size = raw_cutout_size
        x_pos_pix, y_pos_pix = centroid_sources(data=frame_this, xpos=coord_guess[i][0], ypos=coord_guess[i][1], box_size=21, centroid_func=centroid_com)
        cookie_cut_out_sci = frame_this[int(y_pos_pix[0]-0.5*cookie_edge_size):int(y_pos_pix[0]+0.5*cookie_edge_size), int(x_pos_pix[0]-0.5*cookie_edge_size):int(x_pos_pix[0]+0.5*cookie_edge_size)]
        #import ipdb; ipdb.set_trace()

        fit_result, fwhm_x_pix, fwhm_y_pix, sigma_x_pix, sigma_y_pix = fit_gaussian(frame_this, coord_guess[i])
        #import ipdb; ipdb.set_trace()

        cookie_cut_out_best_fit = fit_result[int(y_pos_pix[0]-0.5*cookie_edge_size):int(y_pos_pix[0]+0.5*cookie_edge_size), int(x_pos_pix[0]-0.5*cookie_edge_size):int(x_pos_pix[0]+0.5*cookie_edge_size)]
        #import ipdb; ipdb.set_trace()
        resids = cookie_cut_out_best_fit - cookie_cut_out_sci

        # try fit with Gaussian
        strehl_peak_intensity = strehl_based_on_peak_intensities(frame_this, coord_guess[i], badpix)
        #print('test_strehl',test_strehl)


        ################
        # BEGIN METHOD OF TAKING MTF

    

        # END METHOD OF TAKING MTF
        ################


        #import ipdb; ipdb.set_trace()

        # consider spot on detector to be a PSF_overall that is a convolution of PSF_TBS and PSF_DIRAC
        # Take the 2D Fourier transform of overall PSF to get the OTF
        otf_overall_empirical = np.fft.fftshift(np.fft.fft2(cookie_cut_out_sci))
        # FFT of TBS PSF
        otf_tbs = np.fft.fftshift(np.fft.fft2(psf_tbs))
        otf_dirac_empirical = otf_overall_empirical / otf_tbs
        mtf_dirac_empirical = np.abs(np.fft.fftshift(otf_dirac_empirical))
        mtf_dirac_empirical_zero_freq = mtf_dirac_empirical[mtf_dirac_empirical.shape[0] // 2, mtf_dirac_empirical.shape[1] // 2]
        otf_overall_bestfit = np.fft.fftshift(np.fft.fft2(cookie_cut_out_best_fit))
        otf_dirac_ideal = otf_overall_bestfit / otf_tbs
        mtf_dirac_ideal = np.abs(np.fft.fftshift(otf_dirac_ideal))
        mtf_dirac_ideal_zero_freq = mtf_dirac_ideal[mtf_dirac_empirical.shape[0] // 2, mtf_dirac_ideal.shape[1] // 2]

        # PADDING
        # Padding
        # import ipdb; ipdb.set_trace()
        pad_factor = 4
        padded_size = cookie_cut_out_sci.shape[0] * pad_factor
        pad_width = (padded_size - cookie_cut_out_sci.shape[0]) // 2
        padded_cookie_cut_out_sci = np.pad(cookie_cut_out_sci, pad_width=pad_width, mode='median')
        padded_psf_tbs = np.pad(psf_tbs, pad_width=pad_width, mode='median')
        padded_cookie_cut_out_best_fit = np.pad(cookie_cut_out_best_fit, pad_width=pad_width, mode='median')
        # import ipdb; ipdb.set_trace()

        # Perform FFT and calculations on padded arrays
        otf_overall_empirical = np.fft.fftshift(np.fft.fft2(padded_cookie_cut_out_sci))
        otf_tbs = np.fft.fftshift(np.fft.fft2(padded_psf_tbs))

        otf_dirac_empirical = otf_overall_empirical / otf_tbs
        # import ipdb; ipdb.set_trace()

        #otf_dirac_empirical = otf_dirac_empirical[20:-20, 20:-20]


        mtf_dirac_empirical = np.abs(otf_dirac_empirical)

        mtf_dirac_empirical_zero_freq = mtf_dirac_empirical[mtf_dirac_empirical.shape[0] // 2, mtf_dirac_empirical.shape[1] // 2]
        otf_overall_bestfit = np.fft.fftshift(np.fft.fft2(padded_cookie_cut_out_best_fit))
        otf_dirac_ideal = otf_overall_bestfit / otf_tbs
        mtf_dirac_ideal = np.abs(otf_dirac_ideal)
        mtf_dirac_ideal_zero_freq = mtf_dirac_ideal[mtf_dirac_empirical.shape[0] // 2, mtf_dirac_ideal.shape[1] // 2]
        # import ipdb; ipdb.set_trace()

        
        test = (mtf_dirac_empirical/mtf_dirac_empirical_zero_freq) / (mtf_dirac_ideal/mtf_dirac_ideal_zero_freq)  
        test_2 = np.sum((mtf_dirac_empirical/mtf_dirac_empirical_zero_freq)) / np.sum((mtf_dirac_ideal/mtf_dirac_ideal_zero_freq))  

        test = (mtf_dirac_empirical/np.nanmax(mtf_dirac_empirical)) / (mtf_dirac_ideal/np.nanmax(mtf_dirac_ideal))

        ## import ipdb; ipdb.set_trace

        # Plot the Fourier transform
        #plt.figure()
        #plt.imshow(np.abs(fft_sci), cmap='gray')
        #plt.title('Fourier Transform')
        #plt.colorbar()
        #plt.show()

        plt.clf()
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))

        # Plot cookie_cut_out_sci
        im0 = axs[0].imshow(cookie_cut_out_sci, cmap='gray')
        axs[0].set_title('image')

        # Plot cookie_cut_out_best_fit
        im1 = axs[1].imshow(cookie_cut_out_best_fit, cmap='gray')
        axs[1].set_title('best fit\nFWHM_x (pix): {:.2f}\nFWHM_y (pix): {:.2f}'.format(fwhm_x_pix, fwhm_y_pix))

        # Plot resids
        im2 = axs[2].imshow(resids, cmap='gray')
        axs[2].set_title('residuals')

        # Set the same color scale for first two subplots
        vmin = min(im0.get_array().min(), im1.get_array().min(), im2.get_array().min())
        vmax = max(im0.get_array().max(), im1.get_array().max(), im2.get_array().max())
        im0.set_clim(vmin, vmax)
        im1.set_clim(vmin, vmax)
        #im2.set_clim(vmin, vmax) 

        plt.tight_layout()
        plt.savefig(f'figure_{i:02d}.png')
        plt.close()

        # append the values i, fwhm_x_pix, and fwhm_y_pix in the pandas dataframe
        df = df.append({'spot number': int(i), 'fwhm_x_pix': fwhm_x_pix, 'fwhm_y_pix': fwhm_y_pix, 'x_pos_pix': x_pos_pix[0], 'y_pos_pix': y_pos_pix[0], 'fwhm_tbs_um': fwhm_tbs_um, 'strehl_via_peak_intensity': strehl_peak_intensity}, ignore_index=True)
        #import ipdb; ipdb.set_trace()
        # mask bad pixels
        #frame_this[badpix == 1] = np.nan

    '''
    # dark-subtract each frame, centroid on the spot
    x_cen_array = []
    y_cen_array = []
    for i in range(0,len(df_coord_guesses['filename'].values)):

        file_name_this = df_coord_guesses['filename'].values[i]

        hdul = fits.open(file_name_this)
        sci_this = hdul[0].data
        
        # dark subtraction
        sci = sci_this - dark_median

        #plt.imshow(sci)
        #plt.show()

        # find centers
        x_cen, y_cen = centroid_sources(data=sci, xpos=coord_guess[i][0], ypos=coord_guess[i][1], box_size=21, centroid_func=centroid_com)
        x_cen_array.append(x_cen[0])
        y_cen_array.append(y_cen[0])



        # METHOD 4: deconvolve to find PSF width
        '''
        #cutout = sci[int(y_cen-0.5*raw_cutout_size):int(y_cen+0.5*raw_cutout_size),
        #    int(x_cen-0.5*raw_cutout_size):int(x_cen+0.5*raw_cutout_size)] # cut out star
        #cutout_image = Image.fromarray(cutout)
        #cutout_upsampled = cutout_image.resize((cutout.shape[1]*upsampling, cutout.shape[0]*upsampling))
        #cutout_upsampled = np.array(cutout_upsampled)
        #star_deconv = deconvolve(cutout_upsampled, psf_tbs)
    '''
        # import ipdb; ipdb.set_trace()

        # etc...
    '''

    # add in FWHM values in microns 
    df['fwhm_x_um'] = 18. * df['fwhm_x_pix'] # 18 um per pixel in DIRAC
    df['fwhm_y_um'] = 18. * df['fwhm_y_pix']

    # add fwhm avgs
    df['fwhm_avg_um'] = 0.5 * (df['fwhm_x_um'] + df['fwhm_y_um'])
    df['fwhm_avg_pix'] = 0.5 * (df['fwhm_x_pix'] + df['fwhm_y_pix'])

    # approximate Strehls
    # the true, diffraction-limited PSF after removal of TBS and camera magnification
    df['fwhm_true_um'] = np.sqrt( df['fwhm_avg_um'] ** 2 - df['fwhm_tbs_um'] ** 2 )
    ## import ipdb; ipdb.set_trace()
    df['strehl_via_fwhm'] = np.power( 30.40 / df['fwhm_true_um'], 2)

    # write FWHM info to file
    df.to_csv('junk_output.csv', index=False)




    # make a plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plot scatter plot on the first subplot
    scatter1 = ax1.scatter(df['x_pos_pix'], df['y_pos_pix'], c=df['fwhm_x_pix'])
    ax1.set_xlim([0, 1024])
    ax1.set_ylim([0, 1024])
    ax1.set_aspect('equal', adjustable='box')
    ax1.set_title('FWHM_x (pix)')
    ax1.set_xlabel('x_pos_pix')
    ax1.set_ylabel('y_pos_pix')

    # Plot scatter plot on the second subplot
    scatter2 = ax2.scatter(df['x_pos_pix'], df['y_pos_pix'], c=df['fwhm_y_pix'])
    ax2.set_xlim([0, 1024])
    ax2.set_ylim([0, 1024])
    ax2.set_aspect('equal', adjustable='box')
    ax2.set_title('FWHM_y (pix)')
    ax2.set_xlabel('x_pos_pix')
    ax2.set_ylabel('y_pos_pix')

    # Add colorbars to each subplot
    cbar1 = plt.colorbar(scatter1, ax=ax1, aspect=40)
    cbar1.set_label('FWHM_x (pix)')
    cbar2 = plt.colorbar(scatter2, ax=ax2, aspect=40)
    cbar2.set_label('FWHM_y (pix)')

    plt.tight_layout()
    plt.show()

    plt.clf()
    # make a scatter plot of df['x_pos_pix'] and df['y_pos_pix'], where each point is labeled on the plot with the string df['fwhm_true_um'], with a small offset from the marker
    plt.scatter(df['x_pos_pix'], df['y_pos_pix'])
    for i, row in df.iterrows():
        plt.text(row['x_pos_pix'], row['y_pos_pix'] + 10, f"{row['fwhm_true_um']:.2f}", ha='center', va='bottom', fontsize=8)
    plt.title('True FWHM (um), after removing effect of TBS\n(Ideal is 30.40 um)' )
    plt.xlabel('x_pos_pix')
    plt.ylabel('y_pos_pix')
    plt.savefig('fwhm_plot_um.png')

    plt.clf()
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Plot scatter plot of Strehls based on FWHM measurements
    scatter1 = axs[0].scatter(df['x_pos_pix'], df['y_pos_pix'])
    for i, row in df.iterrows():
        axs[0].text(row['x_pos_pix'], row['y_pos_pix'] + 10, f"{row['strehl_via_fwhm']:.2f}", ha='center', va='bottom', fontsize=8)
    import ipdb; ipdb.set_trace()
    axs[0].set_title('Approx. Strehl, after removing effect of TBS\nS_avg={:.2f}'.format(np.mean(df['strehl_via_fwhm'])) + ', std={:.2f}'.format(np.std(df['strehl_via_fwhm'])) + ' (Perfect is S=1.0)')
    axs[0].set_xlabel('x_pos_pix')
    axs[0].set_ylabel('y_pos_pix')
    # Plot scatter plot of lower bounds to Strehls based on Gaussian fits
    scatter2 = axs[1].scatter(df['x_pos_pix'], df['y_pos_pix'])
    for i, row in df.iterrows():
        axs[1].text(row['x_pos_pix'], row['y_pos_pix'] + 10, f"{row['strehl_via_peak_intensity']:.2f}", ha='center', va='bottom', fontsize=8)
    axs[1].set_title('Lower bound to Strehl, based on Gaussian fits\nto PSF_total = PSF_TBS * PSF_DIRAC\nS_avg={:.2f}'.format(np.mean(df['strehl_via_peak_intensity'])) + ', std={:.2f}'.format(np.std(df['strehl_via_peak_intensity'])) + ' (Perfect is S=1.0)')
    axs[1].set_xlabel('x_pos_pix')
    axs[1].set_ylabel('y_pos_pix')

    plt.tight_layout()
    plt.savefig('strehl_plots.png')
    plt.close()

    # criterion for success:
    # Plate scale 32.7 mas/pix
    logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': -----------------------------------------------------')
    logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': -----------------------------------------------------')
    logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': Fraction of bad pixels: {:.5f}'.format(1. - frac_finite))
    logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': ----------')
    logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': Criterion for success: S > 0.9 at all positions')
    #logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': Measured stdev, x [pix, abs coords]: {:.3f}'.format(sigma_x))
    #logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': Measured stdev, y [pix, abs coords]: {:.3f}'.format(sigma_y))
    logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': ----------')
    #logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': Wrote plot ' + plot_file_name)
    logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': --------------------------------------------------')

    if np.all(df['strehl_via_peak_intensity'] > 0.9):
        logger.info('######   NIRAO-14 Image quality result: PASS   ######')
    elif np.all(df['strehl_via_peak_intensity'] > 0.9-np.std(df['strehl_via_peak_intensity'])):
        logger.info('######   NIRAO-14 Image quality result: CONDITIONAL PASS (lowest Strehl within error around 0.9)  ######')
    else:
        logger.info('######   NIRAO-14 Image quality result: FAIL   ######')

    logger.info('--------------------------------------------------')
    # import ipdb; ipdb.set_trace()


if __name__ == "__main__":
    main()