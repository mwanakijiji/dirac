import numpy as np
from scipy import ndimage

def replace_nans_with_median(arr):
    """
    Custom function to replace given value (assumed to be temporary value for NaNs)
    with the median of the surrounding values in the array.
    """
    # Replace the placeholder value with np.nan to calculate median correctly
    arr[arr == 0] = np.nan
    # Use nanmedian to ignore NaNs in the calculation
    median = np.nanmedian(arr)
    # If the median is still NaN, it means all values are NaN, so return 0 or any other placeholder
    return median if not np.isnan(median) else 0

def remove_nans_with_ndimage_filter(arr):
    """
    Removes NaNs from a 2D array by replacing them with the median of their non-NaN neighbors.
    """
    # Create a mask of where the NaNs are
    nan_mask = np.isnan(arr)
    
    # Temporarily replace NaNs with a placeholder value, e.g., 0
    arr[nan_mask] = 0
    
    # Apply the ndimage filter
    filtered_arr = ndimage.generic_filter(arr, replace_nans_with_median, size=3, mode='constant', cval=np.nan)
    
    # Optionally, restore original NaNs if needed
    # filtered_arr[nan_mask] = np.nan
    
    return filtered_arr

# Example usage
arr = np.array([[1, np.nan, 3], [4, 5, np.nan], [np.nan, 7, 9]])
print(arr)
filtered_arr = remove_nans_with_ndimage_filter(arr)
print(filtered_arr)
import ipdb; ipdb.set_trace()