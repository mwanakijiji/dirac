# calculates the dark current

import matplotlib.pyplot as plt
import numpy as np
import logging
import datetime
import glob
import matplotlib.colors as colors
import warnings
import os
import ipdb
import pandas as pd


def main(dir_csvs):

    # start logging
    log_file_name = 'log_nirao_14b_error_bars_' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.txt'
    logging.basicConfig(filename=log_file_name, 
                        level=logging.INFO, format='%(message)s')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)
    logger = logging.getLogger()


    logger.info('-----------------------------------------------------')
    logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': NIRAO-14 Image Quality test, error bars')
    logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': Criterion for success: Strehl ratio measured for all PSF locations > 90%')
    logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': Using data from ' + dir_csvs)
    logger.info('-----------------------------------------------------')

    # list the csvs
    #file_list = glob.glob(dir_csvs + '*csv')

    def read_files(file_pattern):
        # Read all files matching the pattern into a list of DataFrames
        files = glob.glob(file_pattern)
        dataframes = [pd.read_csv(file) for file in files]
        return dataframes

    file_pattern = dir_csvs + '*csv'

    dataframes = read_files(file_pattern)

    # Step 1: Initialize an empty dictionary to hold the strehl_via_fwhm columns
    strehl_fwhm_columns = pd.DataFrame()
    strehl_gaussian_columns = pd.DataFrame()

    # Step 2: Loop through each dataframe and extract the strehl_via_fwhm column
    for i, df in enumerate(dataframes):
        strehl_fwhm_columns[f'strehl_via_fwhm_{i}'] = df['strehl_via_fwhm']
        strehl_gaussian_columns[f'strehl_via_peak_intensity_variable_gaussian_{i}'] = df['strehl_via_peak_intensity_variable_gaussian']
    
    strehl_fwhm_avg = strehl_fwhm_columns.mean(axis=1)
    strehl_fwhm_std = strehl_fwhm_columns.std(axis=1)
    strehl_gaussian_avg = strehl_gaussian_columns.mean(axis=1)
    strehl_gaussian_std = strehl_gaussian_columns.std(axis=1)


    df_dummy = dataframes[0]
    df_dummy['strehl_fwhm_avg'] = strehl_fwhm_avg
    df_dummy['strehl_fwhm_std_across_arrays'] = strehl_fwhm_std # std for each individual PSF, across arrays
    df_dummy['strehl_fwhm_std_tot'] = np.sqrt(np.power(strehl_fwhm_std,2) + np.power(np.std(df_dummy['strehl_fwhm_avg']),2)) # std across arrays
    df_dummy['strehl_via_peak_intensity_variable_gaussian_avg'] = strehl_gaussian_avg
    df_dummy['strehl_via_peak_intensity_variable_gaussian_std'] = strehl_gaussian_std

    # Plot scatter plot of lower bounds to Strehls based on variable Gaussian fits
    plt.clf()
    fig, axs = plt.subplots(1, 1, figsize=(6, 6))
    scatter1 = axs.scatter(df_dummy['x_pos_pix'], df_dummy['y_pos_pix'])
    for i, row in df_dummy.iterrows():
        text_string = f"{row['strehl_via_peak_intensity_variable_gaussian_avg']:.2f}" + "±" f"{row['strehl_via_peak_intensity_variable_gaussian_std']:.2f}" + "±" f"{np.std(df_dummy['strehl_via_peak_intensity_variable_gaussian_avg']):.2f}"
        axs.text(row['x_pos_pix'], row['y_pos_pix'] + 10, text_string, ha='center', va='bottom', fontsize=8)
    axs.set_title('Strehl based on Gaussian fits\nto PSF_total = PSF_TBS * PSF_DIRAC\nS_avg={:.2f}'.format(np.mean(df_dummy['strehl_via_peak_intensity_variable_gaussian_avg'])) + ', std={:.2f}'.format(np.std(df_dummy['strehl_via_peak_intensity_variable_gaussian_avg'])) + ' between spots' + ' (Perfect is S=1.0)')
    axs.set_xlabel('x_pos_pix')
    axs.set_ylabel('y_pos_pix')
    plt.tight_layout()
    file_name_plot_strehl = 'nirao_14_plot_strehl_spots_gaussian_errors_' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.png'
    plt.savefig(file_name_plot_strehl)
    plt.close()
    logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': Wrote plot of Strehls to '+file_name_plot_strehl)

    # Plot scatter plot of lower bounds to Strehls based on variable Gaussian fits
    plt.clf()
    fig, axs = plt.subplots(1, 1, figsize=(6, 6))
    scatter2 = axs.scatter(df_dummy['x_pos_pix'], df_dummy['y_pos_pix'])
    for i, row in df_dummy.iterrows():
        text_string = f"{row['strehl_fwhm_avg']:.2f}" + "±" f"{row['strehl_fwhm_std_across_arrays']:.2f}" + "±" f"{np.std(df_dummy['strehl_fwhm_avg']):.2f}"
        axs.text(row['x_pos_pix'], row['y_pos_pix'] + 10, text_string, ha='center', va='bottom', fontsize=8)
    axs.set_title('Strehl based on FWHM\nS_avg={:.2f}'.format(np.mean(df_dummy['strehl_fwhm_avg'])) + ', std={:.2f}'.format(np.std(df_dummy['strehl_fwhm_avg'])) + ' between spots' + ' (Perfect is S=1.0)')
    axs.set_xlabel('x_pos_pix')
    axs.set_ylabel('y_pos_pix')
    plt.tight_layout()
    file_name_plot_strehl = 'nirao_14_plot_strehl_spots_fwhm_errors_' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.png'
    plt.savefig(file_name_plot_strehl)
    plt.close()
    logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': Wrote plot of Strehls to '+file_name_plot_strehl)
   
    logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': --------------------------------------------------')


if __name__ == "__main__":
    
    # dir containing all the csvs we want to use
    dir_csvs = '/Users/bandari/Documents/git.repos/dirac/vtp_scripts_data/nirao_14_image_quality/error_bars_2sec_20240809/'

    main(dir_csvs = dir_csvs)