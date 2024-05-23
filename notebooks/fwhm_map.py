# calculates the dark current

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
import logging
import datetime
import glob
from photutils.centroids import centroid_sources, centroid_com
from scipy.optimize import curve_fit
import pandas as pd


# (See emails from R. Zhelem 2024 05 22 for log)

def gaussian_2d(xy_mesh, amplitude, xo, yo, sigma_x, sigma_y, theta):
    x, y = xy_mesh
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta)**2) / (2 * sigma_x**2) + (np.sin(theta)**2) / (2 * sigma_y**2)
    b = -(np.sin(2 * theta)) / (4 * sigma_x**2) + (np.sin(2 * theta)) / (4 * sigma_y**2)
    c = (np.sin(theta)**2) / (2 * sigma_x**2) + (np.cos(theta)**2) / (2 * sigma_y**2)
    g = amplitude * np.exp(-(a * ((x - xo)**2) + 2 * b * (x - xo) * (y - yo) + c * ((y - yo)**2)))
    return g.ravel()


def fit_gaussian(frame, center_guess):
    y, x = np.indices(frame.shape)
    xy_mesh = (x, y)
    p0 = [np.max(frame), center_guess[0], center_guess[1], 1, 1, 0]
    popt, pcov = curve_fit(gaussian_2d, xy_mesh, frame.ravel(), p0=p0)
    fitted_array = gaussian_2d(xy_mesh, *popt).reshape(frame.shape)
    fwhm_x = 2 * np.sqrt(2 * np.log(2)) * np.abs(popt[3])
    fwhm_y = 2 * np.sqrt(2 * np.log(2)) * np.abs(popt[4])
    sigma_x = popt[3]
    sigma_y = popt[4]
    return fitted_array, fwhm_x, fwhm_y, sigma_x, sigma_y
 

def dark_subt_take_median(raw_science_frame_file_names, dark_array):
    # reads in list of raw frames, subtracts dark, and returns median of residuals

    test_array = []
    for i in range(0,len(raw_science_frame_file_names)):

        hdul = fits.open(raw_science_frame_file_names[i])
        sci = hdul[0].data

        sci = sci - dark_array

        test_array.append(sci)

    median_frame = np.median(test_array, axis=0)

    return median_frame


def main():
    # 20240517 is best data

    stem = '/Users/bandari/Documents/git.repos/dirac/notebooks/fwhm_data_20240521/'

    #badpix_file_name = stem + 'ersatz_bad_pix.fits'

    dark_file_names = glob.glob(stem + 'darks/*fits')

    # make net dark
    dark_array = []
    for file_name in dark_file_names:
        hdul = fits.open(file_name)
        dark_this = hdul[0].data
        dark_array.append(dark_this)
    dark_simple = np.median(dark_array, axis=0)

    file_names = [stem + 'sci/DIRAC_20240521_194458.fits', 
                  stem + 'sci/DIRAC_20240521_194642.fits', 
                  stem + 'sci/DIRAC_20240521_194830.fits', 
                  stem + 'sci/DIRAC_20240521_194940.fits',
                  stem + 'sci/DIRAC_20240521_195121.fits',
                  stem + 'sci/DIRAC_20240521_195239.fits',
                  stem + 'sci/DIRAC_20240521_195407.fits',
                  stem + 'sci/DIRAC_20240521_195532.fits',
                  stem + 'sci/DIRAC_20240521_195717.fits',
                  stem + 'sci/DIRAC_20240521_195835.fits',
                  stem + 'sci/DIRAC_20240521_200250.fits',
                  stem + 'sci/DIRAC_20240521_200612.fits',
                  stem + 'sci/DIRAC_20240521_200719.fits',
                  stem + 'sci/DIRAC_20240521_200903.fits',
                  stem + 'sci/DIRAC_20240521_201002.fits',
                  stem + 'sci/DIRAC_20240521_201154.fits',
                  stem + 'sci/DIRAC_20240521_201308.fits',
                  stem + 'sci/DIRAC_20240521_201408.fits',
                  stem + 'sci/DIRAC_20240521_201525.fits',
                  stem + 'sci/DIRAC_20240521_201644.fits',
                  stem + 'sci/DIRAC_20240521_201751.fits',
                  stem + 'sci/DIRAC_20240521_202002.fits',
                  stem + 'sci/DIRAC_20240521_202108.fits',
                  stem + 'sci/DIRAC_20240521_202300.fits',
                  stem + 'sci/DIRAC_20240521_202349.fits']
    
    # x, y convention
    coord_guess = [np.array([46, 52]),  
                   np.array([279, 51]),
                   np.array([527, 51]),
                   np.array([757, 51]),
                   np.array([988, 51]),
                   np.array([990, 271]),
                   np.array([757, 273]),
                   np.array([527, 275]),
                   np.array([310, 273]),
                   np.array([49, 275]),
                   np.array([50, 519]),
                   np.array([281, 517]),
                   np.array([532, 517]),
                   np.array([754, 517]),
                   np.array([975, 517]),
                   np.array([992, 761]),
                   np.array([755, 761]), 
                   np.array([534, 761]),
                   np.array([230, 763]),
                   np.array([35, 763]),
                   np.array([36, 984]),
                   np.array([271, 982]),
                   np.array([547, 979]),
                   np.array([971, 979]),
                   np.array([992, 979])]


    # read/process set of frames corresponding to upper left (of micrometer space; coords are flipped on readout)
    df = pd.DataFrame(columns=['spot number', 'fwhm_x', 'fwhm_y', 'x_pos', 'y_pos']) # initialize
    for i in range(0,len(file_names)):

        try: 

            frame_name = glob.glob(file_names[i])
            frame_this = dark_subt_take_median(raw_science_frame_file_names=frame_name, dark_array=dark_simple)

            cookie_edge_size = 20
            x_pos, y_pos = centroid_sources(data=frame_this, xpos=coord_guess[i][0], ypos=coord_guess[i][1], box_size=21, centroid_func=centroid_com)
            cookie_cut_out_sci = frame_this[int(y_pos[0]-0.5*cookie_edge_size):int(y_pos[0]+0.5*cookie_edge_size), int(x_pos[0]-0.5*cookie_edge_size):int(x_pos[0]+0.5*cookie_edge_size)]

            fit_result, fwhm_x, fwhm_y, sigma_x, sigma_y = fit_gaussian(frame_this, coord_guess[i])

            cookie_cut_out_best_fit = fit_result[int(y_pos[0]-0.5*cookie_edge_size):int(y_pos[0]+0.5*cookie_edge_size), int(x_pos[0]-0.5*cookie_edge_size):int(x_pos[0]+0.5*cookie_edge_size)]

            resids = cookie_cut_out_best_fit - cookie_cut_out_sci

            fig, axs = plt.subplots(1, 3, figsize=(12, 4))

            # Plot cookie_cut_out_sci
            im0 = axs[0].imshow(cookie_cut_out_sci, cmap='gray')
            axs[0].set_title('image')

            # Plot cookie_cut_out_best_fit
            im1 = axs[1].imshow(cookie_cut_out_best_fit, cmap='gray')
            axs[1].set_title('best fit\nFWHM_x: {:.2f}\nFWHM_y: {:.2f}'.format(fwhm_x, fwhm_y))

            # Plot resids
            im2 = axs[2].imshow(resids, cmap='gray')
            axs[2].set_title('residuals')

            # Set the same color scale for all subplots
            vmin = min(im0.get_array().min(), im1.get_array().min(), im2.get_array().min())
            vmax = max(im0.get_array().max(), im1.get_array().max(), im2.get_array().max())
            im0.set_clim(vmin, vmax)
            im1.set_clim(vmin, vmax)
            im2.set_clim(vmin, vmax)

            plt.tight_layout()
            plt.savefig(f'figure_{i:02d}.png')
            plt.close()

            # append the values i, fwhm_x, and fwhm_y in the pandas dataframe
            df = df.append({'spot number': int(i), 'fwhm_x': fwhm_x, 'fwhm_y': fwhm_y, 'x_pos': x_pos[0], 'y_pos': y_pos[0]}, ignore_index=True)

        except: 
            print('Failed on '+str(i))

    df.to_csv('output.csv', index=False)


if __name__ == "__main__":
    main()