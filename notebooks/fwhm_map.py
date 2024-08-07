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

def gaussian_2d(xy_mesh, amplitude, xo, yo, sigma_x_pix, sigma_y_pix, theta):
    x, y = xy_mesh
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta)**2) / (2 * sigma_x_pix**2) + (np.sin(theta)**2) / (2 * sigma_y_pix**2)
    b = -(np.sin(2 * theta)) / (4 * sigma_x_pix**2) + (np.sin(2 * theta)) / (4 * sigma_y_pix**2)
    c = (np.sin(theta)**2) / (2 * sigma_x_pix**2) + (np.cos(theta)**2) / (2 * sigma_y_pix**2)
    g = amplitude * np.exp(-(a * ((x - xo)**2) + 2 * b * (x - xo) * (y - yo) + c * ((y - yo)**2)))
    return g.ravel()


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
    # 20240910 is best data

    stem = '/Users/bandari/Documents/git.repos/dirac/vtp_scripts_data/nirao_14_image_quality/data/20240710/'

    #badpix_file_name = stem + 'ersatz_bad_pix.fits'

    dark_file_names = glob.glob(stem + 'darks/*fits')

    # make net dark
    dark_array = []
    for file_name in dark_file_names:
        hdul = fits.open(file_name)
        dark_this = hdul[0].data
        dark_array.append(dark_this)
    dark_simple = np.median(dark_array, axis=0)

    # Read the text file into a Pandas DataFrame
    df_coord_guesses = pd.read_csv(stem + 'filenames_coord_guesses.txt', delimiter=',')
    # make file names absolute
    df_coord_guesses['filename'] = stem + df_coord_guesses['filename']

    # PSF of the TBS, as measured Apr. 12, 2024 (email from RZ)
    psf_tbs = np.array([[4, 4,  4,  4,  4],
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
    tbs_fit_result, tbs_fwhm_x_pix, tbs_fwhm_y_pix, tbs_sigma_x_pix, tbs_sigma_y_pix = fit_gaussian(psf_tbs, np.array([2.5, 5]))
    # residuals of TBS PSF and its best fit model
    tbs_resids = tbs_fit_result - psf_tbs
    # net instrument FWHM: take average of x and y
    # factors of 5.2 um/pix (TBS pixel pitch) and 2 (DIRAC camera magnification) from email from RZ, 2024 04 12
    fwhm_tbs_um = 0.5 * (tbs_fwhm_x_pix + tbs_fwhm_y_pix) * 5.2 * 2 # i.e., this is the instrumental PSF FWHM inside of DIRAC

    # Make 3D plot of TBS PSF: Create a meshgrid for x and y coordinates
    x = np.arange(psf_tbs.shape[1])
    y = np.arange(psf_tbs.shape[0])
    X, Y = np.meshgrid(x, y)
    # Create a figure and a 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Plot the psf_tbs surface
    ax.plot_surface(X, Y, psf_tbs, cmap='Oranges_r', alpha=0.8)
    # Plot the tbs_fit_result surface
    ax.plot_surface(X, Y, tbs_fit_result, cmap='Purples_r', alpha=0.8)
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Intensity')
    ax.set_title('PSF Comparison')
    # Show the plot
    plt.savefig(f'tbs_psf_3d_plot.png')

    import ipdb; ipdb.set_trace()

    # Make 2D plot of TBS PSF
    plt.clf()
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    im0 = axs[0].imshow(psf_tbs, cmap='gray')
    axs[0].set_title('image')
    im1 = axs[1].imshow(tbs_fit_result, cmap='gray')
    axs[1].set_title('best fit\nFWHM_x (pix): {:.2f}\nFWHM_y (pix): {:.2f}'.format(tbs_fwhm_x_pix, tbs_fwhm_y_pix))
    im2 = axs[2].imshow(tbs_resids, cmap='gray')
    axs[2].set_title('residuals')
    # Set the same color scale for first two subplots
    vmin = min(im0.get_array().min(), im1.get_array().min(), im2.get_array().min())
    vmax = max(im0.get_array().max(), im1.get_array().max(), im2.get_array().max())
    im0.set_clim(vmin, vmax)
    im1.set_clim(vmin, vmax)
    im2.set_clim(vmin, vmax)
    plt.savefig(f'tbs_psf_2d_plot_resids.png')


    import ipdb; ipdb.set_trace()

    file_names = df_coord_guesses['filename'].values
    
    # put all the (x,y) guesses of the spot centers into a list
    # x, y convention
    coord_guess = []
    for i in range(len(df_coord_guesses)):
        x_guess = df_coord_guesses['x_guess'].iloc[i]
        y_guess = df_coord_guesses['y_guess'].iloc[i]
        coord_guess.append(np.array([x_guess, y_guess]))

    # read/process set of frames corresponding to upper left (of micrometer space; coords are flipped on readout)
    df = pd.DataFrame(columns=['spot number', 'fwhm_x_pix', 'fwhm_y_pix', 'x_pos_pix', 'y_pos_pix', 'fwhm_tbs_um']) # initialize
    for i in range(0,len(file_names)):

        try: 

            frame_name = glob.glob(file_names[i])
            frame_this = dark_subt_take_median(raw_science_frame_file_names=frame_name, dark_array=dark_simple)

            cookie_edge_size = 20
            x_pos_pix, y_pos_pix = centroid_sources(data=frame_this, xpos=coord_guess[i][0], ypos=coord_guess[i][1], box_size=21, centroid_func=centroid_com)
            cookie_cut_out_sci = frame_this[int(y_pos_pix[0]-0.5*cookie_edge_size):int(y_pos_pix[0]+0.5*cookie_edge_size), int(x_pos_pix[0]-0.5*cookie_edge_size):int(x_pos_pix[0]+0.5*cookie_edge_size)]

            fit_result, fwhm_x_pix, fwhm_y_pix, sigma_x_pix, sigma_y_pix = fit_gaussian(frame_this, coord_guess[i])

            cookie_cut_out_best_fit = fit_result[int(y_pos_pix[0]-0.5*cookie_edge_size):int(y_pos_pix[0]+0.5*cookie_edge_size), int(x_pos_pix[0]-0.5*cookie_edge_size):int(x_pos_pix[0]+0.5*cookie_edge_size)]

            resids = cookie_cut_out_best_fit - cookie_cut_out_sci

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
            df = df.append({'spot number': int(i), 'fwhm_x_pix': fwhm_x_pix, 'fwhm_y_pix': fwhm_y_pix, 'x_pos_pix': x_pos_pix[0], 'y_pos_pix': y_pos_pix[0], 'fwhm_tbs_um': fwhm_tbs_um}, ignore_index=True)

        except: 
            print('Failed on '+str(i))

    # add in FWHM values in microns 
    df['fwhm_x_um'] = 18. * df['fwhm_x_pix'] # 18 um per pixel in DIRAC
    df['fwhm_y_um'] = 18. * df['fwhm_y_pix']

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


if __name__ == "__main__":
    main()