"""
Physics Data Processors
=======================

Provides utilities for preprocessing and post-processing physics-related data.
This includes functionalities like normalization, scaling, filtering, and
other transformations commonly applied to experimental or simulated physics data.
"""

import numpy as np
from typing import Tuple


class DataProcessor:
    """
    A utility class for various data processing operations relevant to physics data.
    """

    def __init__(self):
        pass # No specific initialization needed for a utility class, but can be extended.

    def normalize_data(self, data: np.ndarray, feature_range: Tuple[float, float] = (0, 1)) -> np.ndarray:
        """
        Normalizes numerical data to a specified feature range (e.g., [0, 1] or [-1, 1]).

        Args:
            data: The input NumPy array to normalize.
            feature_range: A tuple (min_val, max_val) specifying the target range.

        Returns:
            The normalized NumPy array.
        """
        min_val, max_val = feature_range
        data_min = np.min(data, axis=0)
        data_max = np.max(data, axis=0)
        
        # Avoid division by zero if a feature has no variance
        range_diff = data_max - data_min
        range_diff[range_diff == 0] = 1.0 # Set to 1.0 to avoid NaNs for constant columns

        normalized_data = min_val + (data - data_min) * (max_val - min_val) / range_diff
        return normalized_data

    def standardize_data(self, data: np.ndarray) -> np.ndarray:
        """
        Standardizes numerical data to have a mean of 0 and a standard deviation of 1.

        Args:
            data: The input NumPy array to standardize.

        Returns:
            The standardized NumPy array.
        """
        mean = np.mean(data, axis=0)
        std_dev = np.std(data, axis=0)
        
        # Avoid division by zero if std_dev is 0
        std_dev[std_dev == 0] = 1.0 # Set to 1.0 to avoid NaNs for constant columns

        standardized_data = (data - mean) / std_dev
        return standardized_data

    def smooth_trajectory(self, trajectory: np.ndarray, window_size: int = 5) -> np.ndarray:
        """
        Applies a simple moving average filter to smooth a trajectory.

        Args:
            trajectory: A 1D or 2D NumPy array representing a trajectory (time_steps, features).
            window_size: The size of the moving average window. Must be an odd integer.

        Returns:
            The smoothed trajectory NumPy array.
        """
        if window_size % 2 == 0:
            raise ValueError("Window size must be an odd integer for symmetric window.")
        if window_size < 1:
            raise ValueError("Window size must be at least 1.")

        if trajectory.ndim == 1:
            return np.convolve(trajectory, np.ones(window_size)/window_size, mode='valid')
        elif trajectory.ndim == 2:
            smoothed_trajectory = np.copy(trajectory)
            for col_idx in range(trajectory.shape[1]):
                smoothed_trajectory[:, col_idx] = np.convolve(trajectory[:, col_idx], np.ones(window_size)/window_size, mode='valid')
            
            # Pad the smoothed data back to original length if mode='valid' truncated it
            pad_width = (trajectory.shape[0] - smoothed_trajectory.shape[0]) // 2
            if pad_width > 0:
                smoothed_trajectory = np.pad(smoothed_trajectory, ((pad_width, pad_width), (0,0)), mode='edge')
                # If padding is uneven due to odd length, adjust last pad
                if smoothed_trajectory.shape[0] < trajectory.shape[0]:
                    smoothed_trajectory = np.pad(smoothed_trajectory, ((0,1), (0,0)), mode='edge')
            return smoothed_trajectory
        else:
            raise ValueError("Trajectory must be 1D or 2D.")

    def remove_outliers_iqr(self, data: np.ndarray, iqr_factor: float = 1.5) -> np.ndarray:
        """
        Removes outliers from data using the Interquartile Range (IQR) method.
        Outliers are replaced with the median value of their respective feature column.

        Args:
            data: The input NumPy array.
            iqr_factor: The multiplier for the IQR to define outlier bounds.

        Returns:
            A NumPy array with outliers replaced.
        """
        processed_data = np.copy(data)
        for col_idx in range(data.shape[1]):
            column = data[:, col_idx]
            Q1 = np.percentile(column, 25)
            Q3 = np.percentile(column, 75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - iqr_factor * IQR
            upper_bound = Q3 + iqr_factor * IQR
            
            outlier_mask = (column < lower_bound) | (column > upper_bound)
            
            if np.any(outlier_mask):
                median_val = np.median(column[~outlier_mask])
                processed_data[outlier_mask, col_idx] = median_val
        return processed_data


if __name__ == "__main__":
    print("--- Testing Physics Data Processors ---")
    processor = DataProcessor()

    # Test data
    test_data = np.array([
        [1.0, 10.0, 100.0],
        [2.0, 20.0, 200.0],
        [3.0, 30.0, 300.0],
        [4.0, 40.0, 400.0],
        [5.0, 50.0, 500.0]
    ])

    print("\nOriginal Data:\n", test_data)

    # Test normalization
    normalized_data = processor.normalize_data(test_data, feature_range=(-1, 1))
    print("\nNormalized Data (-1 to 1):\n", normalized_data)
    assert np.all(normalized_data >= -1) and np.all(normalized_data <= 1)

    # Test standardization
    standardized_data = processor.standardize_data(test_data)
    print("\nStandardized Data (mean 0, std 1):\n", standardized_data)
    assert np.allclose(np.mean(standardized_data, axis=0), 0)
    assert np.allclose(np.std(standardized_data, axis=0), 1)

    # Test smoothing
    trajectory_1d = np.array([1, 2, 10, 3, 4, 11, 5, 6])
    smoothed_1d = processor.smooth_trajectory(trajectory_1d, window_size=3)
    print("\nSmoothed 1D Trajectory (window 3):\n", smoothed_1d)
    assert len(smoothed_1d) <= len(trajectory_1d) # Valid mode truncates

    trajectory_2d = np.array([
        [1, 10],
        [2, 20],
        [10, 100], # Outlier-like for smoothing effect
        [3, 30],
        [4, 40],
        [11, 110],
        [5, 50]
    ])
    smoothed_2d = processor.smooth_trajectory(trajectory_2d, window_size=3)
    print("\nSmoothed 2D Trajectory (window 3):\n", smoothed_2d)
    assert smoothed_2d.shape[1] == trajectory_2d.shape[1]

    # Test outlier removal
    data_with_outliers = np.array([
        [1.0, 10.0],
        [2.0, 20.0],
        [3.0, 1000.0], # Outlier
        [4.0, 40.0],
        [5.0, 50.0],
        [100.0, 60.0] # Outlier
    ])
    cleaned_data = processor.remove_outliers_iqr(data_with_outliers, iqr_factor=1.5)
    print("\nData with Outliers:\n", data_with_outliers)
    print("Cleaned Data (outliers replaced):\n", cleaned_data)
    # Check if 1000 and 100 (in first column) are replaced
    assert np.allclose(cleaned_data[2, 1], np.median(data_with_outliers[[0,1,3,4,5], 1]))
    assert np.allclose(cleaned_data[5, 0], np.median(data_with_outliers[[0,1,2,3,4], 0]))

    print("\nAll DataProcessor tests completed.")

