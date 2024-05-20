# calculates the dark current

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
import logging
import datetime
import glob
import scipy
from scipy import fftpack
from photutils.centroids import centroid_sources, centroid_com
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from skimage.transform import resize
from PIL import Image
from skimage import color, data, restoration


def deconvolve(star, psf):
    star_fft = fftpack.fftshift(fftpack.fftn(star))
    psf_fft = fftpack.fftshift(fftpack.fftn(psf))
    return fftpack.fftshift(fftpack.ifftn(fftpack.ifftshift(star_fft/psf_fft)))


def gen_model_tbs_psf(raw_cutout_size = 25, upsampling = 4):
    # generate model PSF of the telescope beam simulator
    # model based on FWHM = 6 pix * (5.2 um / pix) = 31.2 um
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

    #plt.contourf(x, y, z, cmap='Blues')
    #plt.colorbar()

    #plt.imshow(z)

    #plt.show()

    return z


def testing_gaussian(raw_cutout_size = 25, upsampling = 4, sigma_expansion = 2):
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


def dark_subt(raw_science_frame_file_names, dark_array):
    # dark subtracts

    hdul = fits.open(raw_science_frame_file_names)
    sci = hdul[0].data

    sci = sci - dark_array

    return median_frame


def main(data_date = '20240516'):
    # 20240516 is best data

    # start logging
    log_file_name = 'log_nirao_02_vibration_' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.txt'
    logging.basicConfig(filename=log_file_name, 
                        level=logging.INFO, format='%(message)s')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)
    logger = logging.getLogger()

    if data_date == '20240516':
        stem = '/Users/bandari/Documents/git.repos/dirac/vtp_scripts_data/nirao_02_vibrations/data/20240516/'
        dark_frame_file_names = glob.glob(stem + '../calibs/darks/*.fits') # darks from 20240517

    logger.info('-----------------------------------------------------')
    logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': NIRAO-02 Vibration test')
    logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': Criterion for success: FWHM of individual ' +\
                'images, and standard deviation of centroid between separate images, will not exceed 0.1 λ/D @ 900 nm. '+\
                'Standard deviation of any blurring visible on pupil edges during single exposures, and standard deviation '+\
                'of fixed position on pupil edge between exposures, will not exceed 1/50 of pupil diameter.')
    logger.info('-----------------------------------------------------')

    #dark_raw_file_name = stem + 'data/tests_junk_13may/pos1_selected_cold_target_not_cold/20sec.fits'
    #bias_file_name = stem + 'data/tests_junk_13may/pos1_selected_cold_target_not_cold/100ms.fits'
    badpix_file_name = stem + '../calibs/ersatz_bad_pix.fits'

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
    dark_median[badpix == 1] = np.nan

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

    if data_date == '20240516': 
        # read/process set of frames corresponding to upper left (of micrometer space; coords are flipped on readout)
        sci_frame_file_names = glob.glob(stem + 'DIRAC_20240516_17033[2-9].fits')
        sci_frame_file_names += glob.glob(stem + 'DIRAC_20240516_1703[4-9]*.fits')
        sci_frame_file_names += glob.glob(stem + 'DIRAC_20240516_170[4-5]*.fits')
        sci_frame_file_names += glob.glob(stem + 'DIRAC_20240516_1706[0-2]*.fits')
        sci_frame_file_names += glob.glob(stem + 'DIRAC_20240516_17063[0-3]*.fits')
        sci_frame_file_names += glob.glob(stem + 'DIRAC_20240516_17111[7-9]*.fits')
        sci_frame_file_names += glob.glob(stem + 'DIRAC_20240516_1711[2-9]*.fits')
        sci_frame_file_names += glob.glob(stem + 'DIRAC_20240516_1712*.fits')
        sci_frame_file_names += glob.glob(stem + 'DIRAC_20240516_1713[0-3]*.fits')
        sci_frame_file_names += glob.glob(stem + 'DIRAC_20240516_17134[0-5]*.fits')
        sci_frame_file_names += glob.glob(stem + 'DIRAC_20240516_17150[7-9]*.fits')
        sci_frame_file_names += glob.glob(stem + 'DIRAC_20240516_17151*.fits')
        sci_frame_file_names += glob.glob(stem + 'DIRAC_20240516_17165[2-9]*.fits')
        sci_frame_file_names += glob.glob(stem + 'DIRAC_20240516_1716[6-9]*.fits')
        sci_frame_file_names += glob.glob(stem + 'DIRAC_20240516_171[6-9]*.fits')
        sci_frame_file_names += glob.glob(stem + 'DIRAC_20240516_172[0-8]*.fits')
        #ul = dark_subt_take_median(raw_science_frame_file_names=ul_raw_frame_file_names, bias_array=bias_simple)

    # dark-subtract each frame, centroid on the spot
    x_cen_array = []
    y_cen_array = []
    for i in range(0,len(sci_frame_file_names)):
        hdul = fits.open(sci_frame_file_names[i])
        sci_this = hdul[0].data
        
        sci = sci_this - dark_median

        # find centers
        x_cen, y_cen = centroid_sources(data=sci, xpos=[544], ypos=[500], box_size=21, centroid_func=centroid_com)
        x_cen_array.append(x_cen[0])
        y_cen_array.append(y_cen[0])

        # FYI
        plt.clf()
        plt.imshow(sci)
        plt.scatter([x_cen], [y_cen], color='red')
        plt.show()

        # deconvolve to find PSF width
        raw_cutout_size = 250
        upsampling = 1
        cutout = sci[int(y_cen-0.5*raw_cutout_size):int(y_cen+0.5*raw_cutout_size),
            int(x_cen-0.5*raw_cutout_size):int(x_cen+0.5*raw_cutout_size)] # cut out star
        cutout_image = Image.fromarray(cutout)
        cutout_upsampled = cutout_image.resize((cutout.shape[1]*upsampling, cutout.shape[0]*upsampling))
        cutout_upsampled = np.array(cutout_upsampled)
        #import ipdb; ipdb.set_trace()

        psf_tbs = gen_model_tbs_psf(raw_cutout_size = raw_cutout_size, upsampling = upsampling)
        star_deconv = deconvolve(cutout_upsampled, psf_tbs)

        # begin test
        '''
        cutout_upsampled = testing_gaussian(raw_cutout_size = raw_cutout_size, upsampling = upsampling, sigma_expansion = 4)
        #cutout_upsampled[:,125] = 0
        psf_tbs = gen_model_tbs_psf(raw_cutout_size = raw_cutout_size, upsampling = upsampling)
        star_deconv = deconvolve(cutout_upsampled, psf_tbs)

        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(cutout_upsampled, cmap='gray')
        axs[0].set_title('cutout_upsampled')
        axs[1].imshow(psf_tbs, cmap='gray')
        axs[1].set_title('psf_tbs')
        axs[2].imshow(np.real(star_deconv), cmap='gray')
        axs[2].set_title('deconv')
        plt.tight_layout()
        plt.show()
        '''
        # end test


    # sanity check: compare with euclidean distances from the sample mean
    # find the sample mean of the points
    x_cen_mean = np.mean(x_cen_array)
    y_cen_mean = np.mean(y_cen_array)
    # find euclidean distances from the sample mean
    d_array = np.sqrt( np.power((x_cen_array-x_cen_mean), 2) + np.power((y_cen_array-y_cen_mean), 2) )
    logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': Average of Euclidean distances from the mean [pix]: {:.3f}'.format(np.mean(d_array)))

    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    sigma_x, sigma_y, _ = confidence_ellipse(x = x_cen_array, y = y_cen_array, ax=ax, n_std=1.0, edgecolor='red', facecolor='none')
    ax.scatter(x_cen_array, y_cen_array)
    plt.show()

    # find the distances between them

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
    logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': -----------------------------------------------------')
    logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': -----------------------------------------------------')
    logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': Fraction of bad pixels: {:.5f}'.format(1. - frac_finite))
    logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': ----------')
    # 0.1 * lambda/D @ 900 nm is equivalent to
    # 0.1 * (0.9e-6 m) * 206265" /( 4 m * 32.7e-3 "/pix) = 0.14 pix
    logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': Criterion for success: Stdev of PSF coordinate is < 0.1 * lambda/D @ 900 nm (0.14 pix)')
    logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': Measured stdev, x [pix, abs coords]: {:.3f}'.format(sigma_x))
    logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': Measured stdev, y [pix, abs coords]: {:.3f}'.format(sigma_y))
    logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': ----------')
    #logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': Wrote plot ' + plot_file_name)
    logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': --------------------------------------------------')

    '''
    # TBD: incorporate other criteria too
    if sigma_x < 0.14 and sigma_y < 0.14:
        logger.info('######   NIRAO-02 Vibrations result: PASS   ######')
    else:
        logger.info('######   NIRAO-02 Vibrations result: FAIL   ######')

    logger.info('--------------------------------------------------')
    '''


if __name__ == "__main__":
    main()