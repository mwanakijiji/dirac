# calculates the dark current

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
import logging
import datetime
import glob
from photutils.centroids import centroid_sources, centroid_com


def main(data_date = '20240515'):
    # 20240515 is best data

    # start logging
    log_file_name = 'log_nirao_13_field_of_view_' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.txt'
    logging.basicConfig(filename=log_file_name, 
                        level=logging.INFO, format='%(message)s')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)
    logger = logging.getLogger(__name__)

    if data_date == '20240517':
        stem = '/Users/eckhartspalding/Documents/git.repos/dirac/vtp_scripts_data/nirao_13_fov/data/20240517/'
    elif data_date == '20240515':
        stem = '/Users/eckhartspalding/Documents/git.repos/dirac/vtp_scripts_data/nirao_13_fov/data/20240515/'

    logger.info('-----------------------------------------------------')
    logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': NIRAO-13 Field of View test')
    logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': Criterion for success: Field of view measured to be 33.2” by 33.2, according to Optical Design Doc”.')
    logger.info('-----------------------------------------------------')


    if data_date == '20240517': 
    
        # See images: spots were placed so that the centers were at the very corner of the science region of the array
        
        # spot locations based on micrometer (y,x), as recorded in .docx
        locations_spots_mm = {'ul': np.array([0.762, 10.785]), 
                           'll': np.array([8.732, 10.760]), 
                           'ur': np.array([0.738, 2.062]),
                           'lr': np.array([8.712, 2.033])}

    elif data_date == '20240515': 
    
        # See images: spots were placed so that the centers were at the very corner of the science region of the array

        # spot locations based on micrometer (y,x), as recorded in .docx
        locations_spots_mm = {'ul': np.array([1.875, 10.943]), 
                           'll': np.array([10.570, 10.905]), 
                           'ur': np.array([1.850, 2.190]),
                           'lr': np.array([10.580, 2.205])}

    # find the distances between them

    # UL del_x, del_y from center
    del_y_left = (locations_spots_mm['ul'][0] - locations_spots_mm['ll'][0])
    del_x_bottom = (locations_spots_mm['lr'][1] - locations_spots_mm['ll'][1])
    del_y_right = (locations_spots_mm['ur'][0] - locations_spots_mm['lr'][0])
    del_x_top = (locations_spots_mm['ur'][1] - locations_spots_mm['ul'][1])

    fov_left = np.abs( del_y_left * 3.689 )
    fov_right = np.abs( del_y_right * 3.689 )
    fov_top = np.abs( del_x_top * 3.689 )
    fov_bottom = np.abs( del_x_bottom * 3.689 )

    logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': -----------------------------------------------------')
    logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': -----------------------------------------------------')
    #logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': Fraction of bad pixels: {:.5f}'.format(1. - frac_finite))
    logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': Criterion for success: Field of view 33.2 arcsec on each side')
    logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': Measured length of left side [arcsec]: {:.2f}'.format(fov_left))
    logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': Percent variation from ideal: {:.1f}%'.format(100 * np.abs(np.mean(fov_left)-33.2)/33.2))
    logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': Measured length of right side [arcsec]: {:.2f}'.format(fov_right))
    logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': Percent variation from ideal: {:.1f}%'.format(100 * np.abs(np.mean(fov_right)-33.2)/33.2))
    logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': Measured length of top side [arcsec]: {:.2f}'.format(fov_top))
    logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': Percent variation from ideal: {:.1f}%'.format(100 * np.abs(np.mean(fov_top)-33.2)/33.2))
    logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': Measured length of bottom side [arcsec]: {:.2f}'.format(fov_bottom))
    logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': Percent variation from ideal: {:.1f}%'.format(100 * np.abs(np.mean(fov_bottom)-33.2)/33.2))
    logger.info(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ': --------------------------------------------------')

    '''
    tolerance: TBD
    if fov_left and fov_right and fov_top and fov_bottom < tolerance:
        logger.info('######   NIRAO-13 Field of View result: PASS   ######')
    else:
        logger.info('######   NIRAO-13 Field of View result: FAIL   ######')

    logger.info('--------------------------------------------------')
    '''

if __name__ == "__main__":
    main(data_date = '20240515')