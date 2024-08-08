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
from astropy.convolution import interpolate_replace_nans
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from skimage.transform import resize
from scipy.optimize import curve_fit
from PIL import Image
from skimage import color, data, restoration
import matplotlib.colors as colors
import warnings
import os
import ipdb


def gaussian_2d(xy_mesh, amplitude, xo, yo, sigma_x_pix, sigma_y_pix, theta):
    x, y = xy_mesh
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta)**2) / (2 * sigma_x_pix**2) + (np.sin(theta)**2) / (2 * sigma_y_pix**2)
    b = -(np.sin(2 * theta)) / (4 * sigma_x_pix**2) + (np.sin(2 * theta)) / (4 * sigma_y_pix**2)
    c = (np.sin(theta)**2) / (2 * sigma_x_pix**2) + (np.cos(theta)**2) / (2 * sigma_y_pix**2)
    g = amplitude * np.exp(-(a * ((x - xo)**2) + 2 * b * (x - xo) * (y - yo) + c * ((y - yo)**2)))
    return g.ravel()

def expected_sigma(lambda_observation, FWHM_tbs_avg):
    # Ideal DIRAC PSF:
    #     r_Airy = 1.22 * (lambda/D) * F = 1.22 * lambda * F#
    # where F# = 29 for DIRAC. So, for Y-band (1.02 um),
    #     r_Airy = 1.22 * 1.02 um * 29 = 36.09 um = 36.09 um / (18 um/pix ) = 2.00 pixels
    # Now,
    #     FWHM_Airy = 1.028 * lambda/D
    # so
    #     r_Airy/FWHM_Airy = 1.187
    # Thus, for Y-band, we expect
    #     FWHM_DIRAC_ideal = FWHM_Airy = r_Airy / 1.187 = 36.09 um / 1.187 = 30.40 um
    # Relation between sigma and FWHM of Gaussian (consider FWHM_Airy and FWHM_Gaussian to be the same)
    #     FWHM_Gauss =2√2ln2σ≈2.355σ, where FWHM_DIRAC_ideal = 30.40 um / (18 um/pix) = 1.69 pix
    # Thus, 
    #     σ = FWHM_Gauss/2.355 = 0.717 IF it's only PSF_DIRAC (Y-band)
    # For PSF_total, 
    #     FWHM_tot_ideal = np.sqrt( FWHM_TBS**2 + FWHM_DIRAC_ideal**2 ) = np.sqrt( (45 um)**2 + (30.40 um)**2 ) = 54.31 um
    # then 
    #     FWHM_tot_ideal = 54.31 um / (18 um/pix) = 3.01 pix
    #     σ_ideal = FWHM/2.355 = 3.01/2.355 = 1.278 pix

    # The same reasoning for H-band (1.63 um) leads to
    #     r_Airy = 57.67 um = 3.20 pixels
    #     FWHM_DIRAC_ideal = FWHM_Airy = 48.58 um
    #     σ = 1.15 pix
    #     FWHM_tot_ideal = 66.06 um = 3.67 pix
    #     σ_ideal = FWHM_tot_ideal / 2.355 = 28.05 um = 1.56 pix

    r_Airy = 1.22 * lambda_observation * 29 # um
    FWHM_Airy = r_Airy / 1.187 # um
    FWHM_DIRAC_ideal = FWHM_Airy
    sigma_gauss = (FWHM_Airy / 2.355) / 18. # pix (DIRAC; 18 um per pix)
    FWHM_tot_ideal_um = np.sqrt( FWHM_tbs_avg ** 2 + FWHM_DIRAC_ideal ** 2 )
    FWHM_tot_ideal_pix = FWHM_tot_ideal_um / 18.
    
    sigma_expected = sigma_gauss

    #sigma_x_pix = 1.278
    #sigma_y_pix = 1.278

    return sigma_expected


def gaussian_2d_fixed_sigmas(xy_mesh, amplitude, xo, yo, theta):
    # Ideal DIRAC PSF:
    #     r_Airy = 1.22 * (lambda/D) * F = 1.22 * lambda * F#
    # where F# = 29 for DIRAC. So, for Y-band (1.02 um),
    #     r_Airy = 1.22 * 1.02 um * 29 = 36.09 um = 36.09 um / (18 um/pix ) = 2.00 pixels
    # Now,
    #     FWHM_Airy = 1.028 * lambda/D
    # so
    #     r_Airy/FWHM_Airy = 1.187
    # Thus, for Y-band, we expect
    #     FWHM_Airy = r_Airy / 1.187 = 36.09 um / 1.187 = 30.40 um
    # Relation between sigma and FWHM of Gaussian:
    #     FWHM_Gauss =2√2ln2σ≈2.355σ, where FWHM_DIRAC_ideal = 30.40 um / (18 um/pix) = 1.69 pix
    # Thus, 
    #     σ = FWHM_Gauss/2.355 = 0.717 IF it's only PSF_DIRAC (Y-band)
    # For PSF_total, FWHM_tot_ideal = np.sqrt( FWHM_TBS**2 + FWHM_DIRAC_ideal**2 ) = np.sqrt( (45 um)**2 + (30.40 um)**2 ) = 54.31 um TODO: make FWHM_tbs_avg get passed to here
    # then 
    #     FWHM_tot_ideal = 54.31 um / (18 um/pix) = 3.01
    #     σ_ideal = FWHM/2.355 = 3.01/2.355 = 1.278

    # The same reasoning for H-band (1.63 um) leads to
    #     r_Airy = 57.67 um = 3.20 pixels
    #     FWHM_Airy = 
    #     σ = 
    #     FWHM_tot_ideal = 
    #     σ_ideal = 

    sigma_x_pix = 1.278
    sigma_y_pix = 1.278

    return gaussian_2d(xy_mesh, amplitude, xo, yo, sigma_x_pix, sigma_y_pix, theta)


def gen_model_tbs_psf(raw_cutout_size = 25, upsampling = 1):
    # generate model PSF (Gaussian approximation) of the telescope beam simulator
    # based on FWHM = 6 pix * (5.2 um / pix) = 31.2 um
    # and DIRAC pitch of 18 um / pix --> FWHM is 31.2 um * (pix / 18 um) = 1.733 pix on DIRAC detector

    # PARAMETERS:
    # raw_cutout_size: edge lengths of cutout from DIRAC detector (before any upsampling)
    # upsampling: upsampling we intend to apply

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

    return z


def get_model_tbs_psf_based_on_empirical(raw_cutout_size = 100, upsampling = 1):
    # generate model of telescope beam simulator (TBS), based on a Gaussian fit to what was measured

    size = raw_cutout_size * upsampling

    # PSF of the TBS, as measured (email from RZ, 2024 04 12)
    # Note the pixels are the TBS pixels (not DIRAC)
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

    # convert to um with factors of 5.2 um/pix (TBS pixel pitch) and 2 (DIRAC camera magnification) (email from RZ, 2024 04 12)
    tbs_fwhm_x_um = tbs_fwhm_x_pix_tbs * 5.2 * 2
    tbs_fwhm_y_um = tbs_fwhm_y_pix_tbs * 5.2 * 2
    tbs_sigma_x_um = tbs_sigma_x_pix_tbs * 5.2 * 2
    tbs_sigma_y_um = tbs_sigma_y_pix_tbs * 5.2 * 2
    # convert to DIRAC pixels: DIRAC pixel pitch is 18 um/pix
    tbs_fwhm_x_pix_dirac = tbs_fwhm_x_um / 18.
    tbs_fwhm_y_pix_dirac = tbs_fwhm_y_um / 18.
    tbs_sigma_x_pix_dirac = tbs_sigma_x_um / 18.
    tbs_sigma_y_pix_dirac = tbs_sigma_y_um / 18.

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
    file_name_psf_tbs = 'psf_tbs_' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.png'
    plt.savefig(file_name_psf_tbs)
    print(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': Saved plot of TBS PSF: '+str(file_name_psf_tbs))

    # upsampling
    tbs_sigma_x_pix_dirac *= tbs_sigma_x_pix_dirac * upsampling
    tbs_sigma_y_pix_dirac *= tbs_sigma_y_pix_dirac * upsampling
    tbs_fwhm_x_pix_dirac *= tbs_fwhm_x_pix_dirac * upsampling
    tbs_fwhm_y_pix_dirac *= tbs_fwhm_y_pix_dirac * upsampling

    # make grid
    x = np.linspace(-int(0.5*size), int(0.5*size), size)
    y = np.linspace(-int(0.5*size), int(0.5*size), size)
    x, y = np.meshgrid(x, y)

    # 2D Gaussian
    z = (1/(2 * np.pi * tbs_sigma_x_pix_dirac * tbs_sigma_y_pix_dirac) * np.exp(-(x**2/(2*tbs_sigma_x_pix_dirac**2) + y**2/(2*tbs_sigma_y_pix_dirac**2))))

    if upsampling != 1:
        print('UPSAMPLING != 1; SIGMA AND FWHM VALS NEED TO BE RESCALED')

    return z, tbs_sigma_x_pix_tbs, tbs_sigma_y_pix_tbs, tbs_fwhm_x_pix_tbs, tbs_fwhm_y_pix_tbs


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
    """
    Fit a 2D Gaussian function to a given frame.

    Parameters:
    frame (ndarray): 2D array representing the frame.
    center_guess (list): List containing the initial guess for the center coordinates.

    Returns:
    fitted_array (ndarray): 2D array representing the fitted Gaussian function.
    fwhm_x_pix (float): Full Width at Half Maximum (FWHM) in the x-direction.
    fwhm_y_pix (float): Full Width at Half Maximum (FWHM) in the y-direction.
    sigma_x_pix (float): Standard deviation in the x-direction.
    sigma_y_pix (float): Standard deviation in the y-direction.
    """

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


def strehl_based_on_peak_intensities_w_variable_gaussian(frame, center_guess, badpix):
    """
    Calculate the Strehl ratio based on peak intensities using a variable Gaussian fit.

    Parameters:
    frame (ndarray): 2D array representing the frame.
    center_guess (list): List containing the initial guess for the center coordinates.
    badpix (float): Value representing bad pixels.

    Returns:
    strehl_simple (float): Strehl ratio based on peak intensities.
    """

    y, x = np.indices(frame.shape)
    xy_mesh = (x, y)
    p0 = [np.max(frame), center_guess[0], center_guess[1], 1, 1, 0]
    # find centroid
    popt, pcov = curve_fit(gaussian_2d, xy_mesh, frame.ravel(), p0=p0)

    # to avoid effect of bad pixels, only consider max within small region around spot
    buffer_size = 10
    cutout_around_psf = frame[int(popt[2])-buffer_size:int(popt[2])+buffer_size,int(popt[1])-buffer_size:int(popt[1])+buffer_size]
    strehl_simple = np.nanmax(cutout_around_psf)/popt[0]
    
    return strehl_simple


def strehl_based_on_peak_intensities_w_fixed_gaussian(frame, center_guess, badpix, sigma_x_fixed_pix_dirac, sigma_y_fixed_pix_dirac):
    """
    Calculate the Strehl ratio based on peak intensities using a fixed-width Gaussian fit.

    Parameters:
    frame (ndarray): 2D array representing the frame.
    center_guess (list): List containing the initial guess for the center coordinates.
    badpix (float): Value representing bad pixels.
    sigma_x_fixed_pix_dirac (float): Fixed standard deviation in the x-direction.
    sigma_y_fixed_pix_dirac (float): Fixed standard deviation in the y-direction.

    Returns:
    strehl_simple (float): Strehl ratio based on peak intensities.
    """

    y, x = np.indices(frame.shape)
    xy_mesh = (x, y)
    p0 = [np.max(frame), center_guess[0], center_guess[1], 0]
    # find centroid; this uses a wrapper function to fix sigma_x and sigma_y
    popt, pcov = curve_fit(gaussian_2d_fixed_sigmas, xy_mesh, frame.ravel(), p0=p0)

    # best-fit Gaussian with fixed sigma (or FWHM)
    model_fit_fixed_sigmas = gaussian_2d(xy_mesh, amplitude=popt[0], xo=popt[1], yo=popt[2], sigma_x_pix=sigma_x_fixed_pix_dirac, sigma_y_pix=sigma_y_fixed_pix_dirac, theta=popt[3]).reshape(frame.shape)

    # to avoid effect of bad pixels, only consider max within small region around spot
    buffer_size = 10
    cutout_around_psf = frame[int(popt[2])-buffer_size:int(popt[2])+buffer_size,int(popt[1])-buffer_size:int(popt[1])+buffer_size]
    strehl_simple = np.nanmax(cutout_around_psf)/popt[0]

    # FYI quantities
    cutout_around_model_fit_fixed_sigmas = model_fit_fixed_sigmas[int(popt[2])-buffer_size:int(popt[2])+buffer_size,int(popt[1])-buffer_size:int(popt[1])+buffer_size]
    illum_of_empirical = np.nansum(cutout_around_psf) # FYI
    illum_of_model_fit_fixed_sigmas = np.nansum(cutout_around_model_fit_fixed_sigmas) # FYI
    
    return strehl_simple


def main(data_date):
    # 20240710 is Y-band
    # 20240709 is H-band

    # upsampling (keep 1 for now)
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
    

    if data_date == '20240807':
        stem = '/Users/bandari/Documents/git.repos/dirac/vtp_scripts_data/nirao_14_image_quality/data/20240807/'
        dark_frame_file_names = glob.glob(stem + 'calibs/darks/*.fits')
        #sigma_x_fixed_pix_dirac = sigma_x_fixed_pix_dirac_y_band # Y-band
        lambda_observation = 1.02 # um
    elif data_date == '20240710':
        stem = '/Users/bandari/Documents/git.repos/dirac/vtp_scripts_data/nirao_14_image_quality/data/20240710/'
        dark_frame_file_names = glob.glob(stem + 'calibs/darks/*.fits')
        #sigma_x_fixed_pix_dirac = sigma_x_fixed_pix_dirac_y_band # Y-band
        lambda_observation = 1.02 # um
    elif data_date == '20240709':
        stem = '/Users/bandari/Documents/git.repos/dirac/vtp_scripts_data/nirao_14_image_quality/data/20240709/'
        dark_frame_file_names = glob.glob(stem + 'calibs/darks/*.fits')
        #sigma_x_fixed_pix_dirac = sigma_x_fixed_pix_dirac_h_band # H-band
        lambda_observation = 1.63 # um
        logger.warning('!!! Science data is in H-band; TBS PSF is in Y-band, so fixed-width Gaussians will not give accurate Strehl !!!')
    elif data_date == '20240715_fake_data_perfect':
        stem = '/Users/bandari/Documents/git.repos/dirac/vtp_scripts_data/nirao_14_image_quality/data/20240715_fake_data/'
        dark_frame_file_names = glob.glob(stem + 'calibs/darks/*.fits')
        #sigma_x_fixed_pix_dirac = sigma_x_fixed_pix_dirac_h_band # H-band
        lambda_observation = 1.02 # um
        logger.warning('!!! Fake data test !!!')
    elif data_date == '20240715_fake_data_noise_lower_strehl':
        stem = '/Users/bandari/Documents/git.repos/dirac/vtp_scripts_data/nirao_14_image_quality/data/20240715_fake_data_noise_lower_strehl/'
        dark_frame_file_names = glob.glob(stem + 'calibs/darks/*.fits')
        #sigma_x_fixed_pix_dirac = sigma_x_fixed_pix_dirac_h_band # H-band
        lambda_observation = 1.02 # um
        logger.warning('!!! Fake data test !!!')

    logger.info('-----------------------------------------------------')
    logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': NIRAO-14 Image Quality test')
    logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': Criterion for success: Strehl ratio measured for all PSF locations > 90%')
    logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': Using data from ' + data_date)
    logger.info('-----------------------------------------------------')

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

    # total number of pixels 
    N_pix_tot = dark_median.size
    # how many pixels are finite?
    N_pix_finite = np.nansum(np.isfinite(dark_median))
    # fraction of good pixels within science region of detector (i.e., 4-pixel-wide overscan region of 16320 pixels removed)
    frac_finite = N_pix_finite/(N_pix_tot - 16320)

    # Read in spot coordinate guesses
    df_coord_guesses = pd.read_csv(stem + 'filenames_coord_guesses.txt', delimiter=',')
    # make file names absolute
    df_coord_guesses['filename'] = stem + df_coord_guesses['filename']

    # TBS PSF
    psf_tbs, tbs_sigma_x_pix_tbs, tbs_sigma_y_pix_tbs, tbs_fwhm_x_pix_tbs, tbs_fwhm_y_pix_tbs = get_model_tbs_psf_based_on_empirical(raw_cutout_size = raw_cutout_size, upsampling = upsampling)
    # TBS pixel pitch is 5.2 um/pix; DIRAC camera magnification is 2x
    fwhm_tbs_um = 0.5 * (tbs_fwhm_x_pix_tbs + tbs_fwhm_y_pix_tbs) * 5.2 * 2 # average FWHM of TBS in DIRAC pixels

    logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': TBS FWHM, xy avg. (um): {:.2f}'.format(fwhm_tbs_um))
    sigma_fixed_pix_dirac = expected_sigma(lambda_observation, FWHM_tbs_avg=fwhm_tbs_um)
    #fwhm_tbs_um = (np.min([tbs_fwhm_x_pix_tbs,tbs_fwhm_y_pix_tbs])) * 5.2 * 2 # option: min FWHM of TBS in DIRAC pixels

    # Read the text file with coord guesses into a Pandas DataFrame
    df_coord_guesses = pd.read_csv(stem + 'filenames_coord_guesses.txt', delimiter=',')
    # make file names absolute
    df_coord_guesses['filename'] = stem + df_coord_guesses['filename']
    file_names = df_coord_guesses['filename'].values
    # put all the (x,y) guesses of the spot centers into a list
    # x, y convention
    coord_guess = []
    for i in range(len(df_coord_guesses)):
        x_guess = df_coord_guesses['x_guess'].iloc[i]
        y_guess = df_coord_guesses['y_guess'].iloc[i]
        coord_guess.append(np.array([x_guess, y_guess]))

    # initialize
    df = pd.DataFrame(columns=['spot number', 'fwhm_x_pix', 'fwhm_y_pix', 'x_pos_pix', 'y_pos_pix', 'fwhm_tbs_um'])

    # instantiate bad pixel fixing
    #do_fixpix = FixPixSingle(config)
    #pool.map(do_fixpix, darksubt_01_name_array)

    # loop over all frames
    for i in range(0,len(df_coord_guesses['filename'].values)):

        # suppress Python FutureWarnings
        warnings.simplefilter(action='ignore', category=Warning)

        file_name_this = df_coord_guesses['filename'].values[i]

        logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': ----------')
        logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': Reading in frame ' + file_name_this)

        # read in science frame
        hdul = fits.open(file_name_this)
        sci_this = hdul[0].data

        sci_this = sci_this.astype(float)
        sci_this[badpix == 1] = np.nan

        # bad pixel correction
        kernel_square = np.ones((3,3))
        image_fixpixed = interpolate_replace_nans(array=sci_this, kernel=kernel_square) #.astype(np.int32)
        # replace remaining NaNs with median (these are likely overscan pixels)
        image_fixpixed[np.isnan(image_fixpixed)] = np.nanmedian(image_fixpixed)

        # reassign
        sci_this = image_fixpixed

        # dark subtract
        frame_this = sci_this - dark_median

        # inject noise 
        # FOR TESTING ONLY
        '''
        logger.info('!!!!!!!!! ----------- Testing with FAKE NOISE ----------- !!!!!!!!!')
        frame_this += np.random.normal(scale=np.std(frame_this), size=np.shape(frame_this))
        '''

        # find spot center
        x_pos_pix, y_pos_pix = centroid_sources(data=frame_this, xpos=coord_guess[i][0], ypos=coord_guess[i][1], box_size=21, centroid_func=centroid_com)

        # make cutout around spot
        cookie_edge_size = raw_cutout_size
        cookie_cut_out_sci = frame_this[int(y_pos_pix[0]-0.5*cookie_edge_size):int(y_pos_pix[0]+0.5*cookie_edge_size), int(x_pos_pix[0]-0.5*cookie_edge_size):int(x_pos_pix[0]+0.5*cookie_edge_size)]

        # make best fit Gaussian to empirical; all fit parameters are free
        fit_result, fwhm_x_pix, fwhm_y_pix, sigma_x_pix, sigma_y_pix = fit_gaussian(frame_this, coord_guess[i])

        # make cutout around the model (for plot)
        cookie_cut_out_best_fit = fit_result[int(y_pos_pix[0]-0.5*cookie_edge_size):int(y_pos_pix[0]+0.5*cookie_edge_size), int(x_pos_pix[0]-0.5*cookie_edge_size):int(x_pos_pix[0]+0.5*cookie_edge_size)]

        # residuals with empirical (for plot)
        resids = cookie_cut_out_best_fit - cookie_cut_out_sci

        # Strehl ratio based on comparison of Gaussian fit (all fit parameters are free) with empirical
        strehl_via_peak_intensity_variable_gaussian = strehl_based_on_peak_intensities_w_variable_gaussian(frame_this, coord_guess[i], badpix)
        logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': Found Strehl using variable Gaussian: {:.2f}'.format(strehl_via_peak_intensity_variable_gaussian))

        # Strehl ratio based on comparison of Gaussian fit (parameters are free except for fixed, ideal FWHM in x and y) with empirical
        strehl_peak_intensity_fixed_gaussian = strehl_based_on_peak_intensities_w_fixed_gaussian(frame_this, coord_guess[i], badpix, sigma_x_fixed_pix_dirac=sigma_fixed_pix_dirac, sigma_y_fixed_pix_dirac=sigma_fixed_pix_dirac)
        #logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': Found Strehl using fixed-width Gaussian: {:.2f}'.format(strehl_peak_intensity_fixed_gaussian))

        # Strehl ratio based on comparison of Gaussian fit (all fit parameters are free) with empirical
        strehl_via_peak_intensity_variable_gaussian = strehl_based_on_peak_intensities_w_variable_gaussian(frame_this, coord_guess[i], badpix)
        #logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': Found Strehl using FWHM: {:.2f}'.format(strehl_via_peak_intensity_variable_gaussian))


        # plot empirical spots, the best Gaussian fits, and the residuals
        plt.clf()
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        # empirical spot
        im0 = axs[0].imshow(cookie_cut_out_sci, cmap='gray')
        axs[0].set_title('image')
        # best-fit model
        im1 = axs[1].imshow(cookie_cut_out_best_fit, cmap='gray')
        axs[1].set_title('best fit, '+ os.path.basename(file_name_this) +'\nFWHM_x (pix): {:.2f}\nFWHM_y (pix): {:.2f}'.format(fwhm_x_pix, fwhm_y_pix))
        # residuals
        im2 = axs[2].imshow(resids, cmap='gray')
        axs[2].set_title('residuals')
        # set the same color scale for first two subplots
        vmin = min(im0.get_array().min(), im1.get_array().min(), im2.get_array().min())
        vmax = max(im0.get_array().max(), im1.get_array().max(), im2.get_array().max())
        im0.set_clim(vmin, vmax)
        im1.set_clim(vmin, vmax)
        #im2.set_clim(vmin, vmax) 
        plt.tight_layout()
        file_name_psf_best_fit = 'psf_best_fit_' + os.path.basename(file_name_this).split('.')[0] + '_' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.png'
        plt.savefig(file_name_psf_best_fit)
        plt.close()
        logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': Wrote PSF best-fit plot ' + file_name_psf_best_fit)

        # append the values i, fwhm_x_pix, and fwhm_y_pix etc. in the pandas dataframe
        df = df.append({'spot number': int(i), 'fwhm_x_pix': fwhm_x_pix, 'fwhm_y_pix': fwhm_y_pix, 'x_pos_pix': x_pos_pix[0], 'y_pos_pix': y_pos_pix[0], 
                        'fwhm_tbs_um': fwhm_tbs_um, 'strehl_via_peak_intensity_variable_gaussian': strehl_via_peak_intensity_variable_gaussian, 
                        'strehl_via_peak_intensity_fixed_gaussian': strehl_peak_intensity_fixed_gaussian}, ignore_index=True)


    # add in FWHM values in microns 
    df['fwhm_x_um'] = 18. * df['fwhm_x_pix'] # 18 um per pixel in DIRAC
    df['fwhm_y_um'] = 18. * df['fwhm_y_pix']

    # add fwhm avgs
    df['fwhm_avg_um'] = 0.5 * (df['fwhm_x_um'] + df['fwhm_y_um'])
    df['fwhm_avg_pix'] = 0.5 * (df['fwhm_x_pix'] + df['fwhm_y_pix'])

    # approximate Strehls
    # the true, diffraction-limited PSF after removal of TBS and camera magnification
    df['fwhm_true_um'] = np.sqrt( df['fwhm_avg_um'] ** 2 - df['fwhm_tbs_um'] ** 2 )
    df['strehl_via_fwhm'] = np.power( 30.40 / df['fwhm_true_um'], 2)

    # write FWHM info to file
    file_name_output_csv = 'nirao_14_output_' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.csv'
    df.to_csv(file_name_output_csv, index=False)
    logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': ----------')
    logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': Wrote FWHM information to '+file_name_output_csv)

    # make a plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    plt.clf()
    # make a scatter plot of df['x_pos_pix'] and df['y_pos_pix'], where each point is labeled on the plot with the string df['fwhm_true_um'], with a small offset from the marker
    plt.scatter(df['x_pos_pix'], df['y_pos_pix'])
    for i, row in df.iterrows():
        plt.text(row['x_pos_pix'], row['y_pos_pix'] + 10, f"{row['fwhm_true_um']:.2f}", ha='center', va='bottom', fontsize=8)
    plt.title('True FWHM (um), after removing effect of TBS\n(Ideal is 30.40 um)' )
    plt.xlabel('x_pos_pix')
    plt.ylabel('y_pos_pix')
    file_name_plot_fwhm_um = 'nirao_14_plot_true_fwhm_spots_' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.png'
    plt.axis('equal')
    plt.savefig(file_name_plot_fwhm_um)
    logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': Wrote plot of FWHM of spots to '+file_name_plot_fwhm_um)

    plt.clf()
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Plot scatter plot of lower bounds to Strehls based on variable Gaussian fits
    scatter1 = axs[0].scatter(df['x_pos_pix'], df['y_pos_pix'])
    for i, row in df.iterrows():
        axs[0].text(row['x_pos_pix'], row['y_pos_pix'] + 10, f"{row['strehl_via_peak_intensity_variable_gaussian']:.2f}", ha='center', va='bottom', fontsize=8)
    axs[0].set_title('Lower bound to Strehl, based on variable Gaussian fits\nto PSF_total = PSF_TBS * PSF_DIRAC\nS_avg={:.2f}'.format(np.mean(df['strehl_via_peak_intensity_variable_gaussian'])) + ', std={:.2f}'.format(np.std(df['strehl_via_peak_intensity_variable_gaussian'])) + ' (Perfect is S=1.0)')
    axs[0].set_xlabel('x_pos_pix')
    axs[0].set_ylabel('y_pos_pix')

    # Plot scatter plot of lower bounds to Strehls based on fixed-width Gaussian fits
    '''
    scatter2 = axs[1].scatter(df['x_pos_pix'], df['y_pos_pix'])
    for i, row in df.iterrows():
        axs[1].text(row['x_pos_pix'], row['y_pos_pix'] + 10, f"{row['strehl_via_peak_intensity_fixed_gaussian']:.2f}", ha='center', va='bottom', fontsize=8)
    axs[1].set_title('Lower bound to Strehl, based on fixed-width Gaussian fits\nto PSF_total = PSF_TBS * PSF_DIRAC\nS_avg={:.2f}'.format(np.mean(df['strehl_via_peak_intensity_fixed_gaussian'])) + ', std={:.2f}'.format(np.std(df['strehl_via_peak_intensity_fixed_gaussian'])) + ' (Perfect is S=1.0)')
    axs[1].set_xlabel('x_pos_pix')
    axs[1].set_ylabel('y_pos_pix')
    '''

    # Plot scatter plot of Strehls based on FWHM measurements
    scatter3 = axs[1].scatter(df['x_pos_pix'], df['y_pos_pix'])
    for i, row in df.iterrows():
        axs[1].text(row['x_pos_pix'], row['y_pos_pix'] + 10, f"{row['strehl_via_fwhm']:.2f}", ha='center', va='bottom', fontsize=8)
    axs[1].set_title('Approx. Strehl, after removing effect of TBS\nS_avg={:.2f}'.format(np.mean(df['strehl_via_fwhm'])) + ', std={:.2f}'.format(np.std(df['strehl_via_fwhm'])) + ' (Perfect is S=1.0)')
    axs[1].set_xlabel('x_pos_pix')
    axs[1].set_ylabel('y_pos_pix')

    plt.tight_layout()
    file_name_plot_strehl = 'nirao_14_plot_strehl_spots_' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.png'
    plt.savefig(file_name_plot_strehl)
    plt.close()
    logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': Wrote plot of Strehls to '+file_name_plot_strehl)

    # criterion for success: Strehls > 0.9
    logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': -----------------------------------------------------')
    logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': -----------------------------------------------------')
    #logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': Fraction of bad pixels: {:.5f}'.format(1. - frac_finite))
    logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': Criterion for success: S > 0.9 at all positions')
    logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': --------------------------------------------------')

    if np.all(df['strehl_via_peak_intensity_variable_gaussian'] > 0.9):
        logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': ######   NIRAO-14 Image quality result via Gaussian fit: PASS   ######')
    elif np.all(df['strehl_via_peak_intensity_variable_gaussian'] > 0.9-np.std(df['strehl_via_peak_intensity_variable_gaussian'])):
        logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': ######   NIRAO-14 Image quality result via Gaussian fit: PASS WITHIN ERROR (lowest Strehl within error around 0.9)  ######')
    else:
        logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': ######   NIRAO-14 Image quality result via Gaussian fit: FAIL   ######')

    '''
    if np.all(df['strehl_via_fwhm'] > 0.9):
        logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': ######   NIRAO-14 Image quality result via FWHM: PASS   ######')
    elif np.all(df['strehl_via_fwhm'] > 0.9-np.std(df['strehl_via_fwhm'])):
        logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': ######   NIRAO-14 Image quality result via FWHM: PASS WITHIN ERROR (lowest Strehl within error around 0.9)  ######')
    else:
        logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': ######   NIRAO-14 Image quality result via FWHM: FAIL   ######')
    '''
        
    logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': --------------------------------------------------')


if __name__ == "__main__":
    # 20240710 is Y-band
    # 20240709 is H-band (different band from TBS PSF!)
    main(data_date = '20240807')