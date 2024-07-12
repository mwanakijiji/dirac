'''
Prepares the data: bad pixel masking, background-subtraction, etc.
## ## This is descended from
## ## find_background_limit.ipynb
## ## subtract stray y illumination gradient parallel.py
## ## make_pca_basis_cube_altair.py
## ## pca_background_subtraction.ipynb
'''

import multiprocessing
import configparser
import glob
import time
from astropy.io import fits
from astropy.convolution import convolve, Gaussian1DKernel, interpolate_replace_nans
from regions import PixCoord, CircleSkyRegion, CirclePixelRegion, PolygonPixelRegion
from sklearn.decomposition import PCA
from modules import *

import matplotlib
matplotlib.use('agg') # avoids some crashes when multiprocessing
import matplotlib.pyplot as plt

# configuration data
#config = configparser.ConfigParser() # for parsing values in .init file
#config.read("config.ini")

class FixPixSingle:
    '''
    Interpolates over bad pixels
    '''

    def __init__(self, config_data=config):

        self.config_data = config_data

        # read in bad pixel mask
        # (altair 180507 bpm has convention 0=good, 1=bad)
        abs_badpixmask_name = str(self.config_data["data_dirs"]["DIR_CALIB_FRAMES"]) + \
          "master_bpm.fits"
        self.badpix, self.header_badpix = fits.getdata(abs_badpixmask_name, 0, header=True)

        # (altair 180507 bpm requires a top row)
        self.badpix = np.vstack([self.badpix,np.zeros(np.shape(self.badpix)[1])])

        # turn 1->nan (bad), 0->1 (good) for interpolate_replace_nans
        self.ersatz = np.nan*np.ones(np.shape(self.badpix))
        self.ersatz[self.badpix == 0] = 1.
        self.badpix = self.ersatz # rename
        del self.ersatz

        # define the convolution kernel (normalized by default)
        self.kernel = np.ones((3,3)) # just a patch around the kernel

    def __call__(self, abs_sci_name):
        '''
        Bad pix fixing, for a single frame so as to parallelize job

        INPUTS:
        sci_name: science array filename
        '''

        # read in the science frame from raw data directory
        sci, header_sci = fits.getdata(abs_sci_name, 0, header=True)

        # fix bad pixels (note conversion to 32-bit signed ints)
        sci_badnan = np.multiply(sci,self.badpix)
        image_fixpixed = interpolate_replace_nans(array=sci_badnan, kernel=self.kernel).astype(np.int32)

        # add a line to the header indicating last reduction step
        header_sci["RED_STEP"] = "bad-pixel-fixed"

        # write file out
        abs_image_fixpixed_name = str(self.config_data["data_dirs"]["DIR_PIXL_CORRTD"] + \
                                        os.path.basename(abs_sci_name))
        fits.writeto(filename=abs_image_fixpixed_name,
                     data=image_fixpixed,
                     header=header_sci,
                     overwrite=True)
        print("Writing out bad-pixel-fixed frame " + os.path.basename(abs_image_fixpixed_name))



def main():
    '''
    Carry out the basic data-preparation steps
    '''

    # configuration data
    config = configparser.ConfigParser() # for parsing values in .init file
    config.read("/modules/config.ini")

    # multiprocessing instance
    pool = multiprocessing.Pool(ncpu)

    '''
    COMMENTED OUT BECAUSE I ALREADY HAVE RAMP-SUBTRACTED FRAMES AND WANT TO OPTIMIZE DOWNSTREAM --19 JUNE 2019
    # make a list of the raw files
    raw_00_directory = str(config["data_dirs"]["DIR_RAW_DATA"])
    raw_00_name_array = list(glob.glob(os.path.join(raw_00_directory, "*.fits")))

    # subtract darks in parallel
    print("Subtracting darks with " + str(ncpu) + " CPUs...")
    do_dark_subt = DarkSubtSingle(config)
    pool.map(do_dark_subt, raw_00_name_array)

    # make a list of the dark-subtracted files
    darksubt_01_directory = str(config["data_dirs"]["DIR_DARK_SUBTED"])
    darksubt_01_name_array = list(glob.glob(os.path.join(darksubt_01_directory, "*.fits")))

    # fix bad pixels in parallel
    print("Fixing bad pixels with " + str(ncpu) + " CPUs...")
    do_fixpix = FixPixSingle(config)
    pool.map(do_fixpix, darksubt_01_name_array)

    # make a list of the bad-pix-fixed files
    fixpixed_02_directory = str(config["data_dirs"]["DIR_PIXL_CORRTD"])
    fixpixed_02_name_array = list(glob.glob(os.path.join(fixpixed_02_directory, "*.fits")))

    # subtract ramps in parallel
    print("Subtracting artifact ramps with " + str(ncpu) + " CPUs...")
    do_ramp_subt = RemoveStrayRamp(config)
    pool.map(do_ramp_subt, fixpixed_02_name_array)

    # make lists of the ramp-removed files
    ramp_subted_03_directory = str(config["data_dirs"]["DIR_RAMP_REMOVD"])
    # all files in directory
    ramp_subted_03_name_array = list(glob.glob(os.path.join(ramp_subted_03_directory, "*.fits")))
    '''

    # begin handy snippet of code if I need to take file basenames from one directory and attach them
    # to a different path stem
    #kludge_dir = str(config["data_dirs"]["DIR_HOME"]) + "/pipeline_04_pcab_subted/escrow_old/"
    #kludge_glob = list(glob.glob(kludge_dir + "*.fits"))
    #names = [str(kludge_dir)+str(os.path.basename(y)) for y in kludge_glob]
    #ramp_subted_03_name_array = names
    # end handy snippet of code

    '''
    COMMENTED OUT TO SAVE TIME
    # make a PCA vector involving channel variations only
    pca_backg_maker_channels_only(abs_pca_cube_file_name = str(config["data_dirs"]["DIR_OTHER_FITS"] + \
                                                               "background_PCA_vector_channel_vars_only.fits"))

    # PCA-based background subtraction in parallel
    print("Subtracting backgrounds with " + str(ncpu) + " CPUs...")

    # set up parameters of PCA background-subtraction:
    # [0]: starting frame number of the PCA component cube
    # [1]: stopping frame number (inclusive)  "  "
    # [2]: total number of PCA components to reconstruct the background with
    #      (usually 32 channel elements + 100 noise/sky PCA elements)
    # [3]: background quadrant choice (2 or 3)

    # ALL science frames, using background subtraction involving just channel variations
    param_array = [-9999, -9999, 32, -9999]
    do_pca_back_subt = BackgroundPCASubtSingle(param_array, \
        simple_channel_file = str(config["data_dirs"]["DIR_OTHER_FITS"] + \
        "background_PCA_vector_channel_vars_only.fits", config, simple_channel = True)
    pool.map(do_pca_back_subt, ramp_subted_03_name_array)

    '''
    # make a list of the PCA-background-subtracted files
    pcab_subted_04_directory = str(config["data_dirs"]["DIR_PCAB_SUBTED"])
    pcab_subted_04_names = list(glob.glob(os.path.join(pcab_subted_04_directory, "*.fits")))

    # make cookie cutouts of the PSFs
    ## ## might add functionality to override the found 'center' of the PSF
    make_cookie_cuts = CookieCutout(quad_choice = -9999)
    pool.map(make_cookie_cuts, pcab_subted_04_names)
