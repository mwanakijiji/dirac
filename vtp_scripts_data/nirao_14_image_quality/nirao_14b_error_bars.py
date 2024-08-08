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
    # 20240710 is Y-band
    # 20240709 is H-band

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

    def find_matches_within_distance(df1, df2, distance_threshold=5):
        matches = []
        for index1, row1 in df1.iterrows():
            x1, y1 = row1['x_pos_pix'], row1['y_pos_pix']
            for index2, row2 in df2.iterrows():
                x2, y2 = row2['x_pos_pix'], row2['y_pos_pix']
                distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                if distance <= distance_threshold:
                    matches.append((index1, index2))
        return matches

    def match_coordinates_across_files(file_pattern):
        dataframes = read_files(file_pattern)
        all_matches = []
        for i in range(len(dataframes)):
            for j in range(i + 1, len(dataframes)):
                matches = find_matches_within_distance(dataframes[i], dataframes[j])
                all_matches.append((i, j, matches))
        return all_matches
    
    def round_to_nearest_even(value):
        return int(np.round(value / 2) * 2)

    def add_approx_columns(dataframes):
        for df in dataframes:
            df['x_pos_pix_approx'] = df['x_pos_pix'].apply(round_to_nearest_even)
            df['y_pos_pix_approx'] = df['y_pos_pix'].apply(round_to_nearest_even)
            #df['xy_pos_pix'] = df.apply(lambda row: (row['x_pos_pix_approx'], row['y_pos_pix_approx']), axis=1)
        return dataframes
    
    def merge_dataframes_on_approx(dataframes):
        # Start with the first dataframe
        merged_df = dataframes[0]
        for df in dataframes[1:]:
            merged_df = pd.merge(merged_df, df, on=['x_pos_pix_approx', 'y_pos_pix_approx'], how='inner', suffixes=('_l', '_r'))
            # Drop duplicate columns after merge
            #for col in merged_df.columns:
            #    if '_dup' in col:
            #        merged_df.drop(col, axis=1, inplace=True)
        return merged_df
    

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

    ipdb.set_trace()


    strehl_fwhm_columns_subset = [col for col in new_dataframe.columns if col.startswith('strehl_via_fwhm_')]
    strehl_fwhm_columns_subset = [col for col in new_dataframe.columns if col.startswith('strehl_via_fwhm_')]




    vals_mean = new_dataframe.mean(axis=1)


    # Step 3: Convert the dictionary to a new dataframe
    new_dataframe = pd.DataFrame(strehl_columns)





    ipdb.set_trace()
    #merged_df = merge_dataframes_on_approx(dataframes)




    ipdb.set_trace()

    matches = match_coordinates_across_files(file_pattern)

    ipdb.set_trace()

    # Print matches
    for file1_index, file2_index, match_list in matches:
        print(f"Matches between file {file1_index} and file {file2_index}:")
        for index1, index2 in match_list:
            print(f"File {file1_index} index {index1} matches with File {file2_index} index {index2}")

        '''
        # loop over all frames
        for file_name in file_list:

            df = pd.read_csv(file_name)
            ipdb.set_trace()

            # suppress Python FutureWarnings
            warnings.simplefilter(action='ignore', category=Warning)

        
            logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': ----------')
            logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': Reading in frame ' + file_name_this)
        '''



        '''
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
        '''


    # add in FWHM values in microns 
 
    '''
    if np.all(df['strehl_via_peak_intensity_variable_gaussian'] > 0.9):
        logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': ######   NIRAO-14 Image quality result via Gaussian fit: PASS   ######')
    elif np.all(df['strehl_via_peak_intensity_variable_gaussian'] > 0.9-np.std(df['strehl_via_peak_intensity_variable_gaussian'])):
        logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': ######   NIRAO-14 Image quality result via Gaussian fit: PASS WITHIN ERROR (lowest Strehl within error around 0.9)  ######')
    else:
        logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': ######   NIRAO-14 Image quality result via Gaussian fit: FAIL   ######')
    '''


        
    logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': --------------------------------------------------')


if __name__ == "__main__":
    
    # dir containing all the csvs we want to use
    dir_csvs = '/Users/bandari/Documents/git.repos/dirac/vtp_scripts_data/nirao_14_image_quality/testing/'

    main(dir_csvs = dir_csvs)