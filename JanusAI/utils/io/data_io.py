"""
Data I/O Utilities
==================

Provides functions for loading and saving various data formats common in
scientific and machine learning projects (e.g., NumPy arrays, Pandas DataFrames).
"""

import numpy as np
import pandas as pd
import json
import os
from typing import Any, Dict, List, Union, Optional


class DataIO:
    """
    A utility class for common data input/output operations.
    Supports NumPy arrays, Pandas DataFrames, and JSON.
    """

    def __init__(self):
        pass # No specific initialization needed for static methods, but can be instantiated.

    def load_numpy(self, file_path: str) -> Optional[np.ndarray]:
        """Loads data from a .npy file."""
        if not os.path.exists(file_path):
            print(f"Error: .npy file not found at {file_path}")
            return None
        try:
            return np.load(file_path)
        except Exception as e:
            print(f"Error loading .npy file {file_path}: {e}")
            return None

    def save_numpy(self, data: np.ndarray, file_path: str):
        """Saves data to a .npy file."""
        try:
            np.save(file_path, data)
            # print(f"Data saved to {file_path}") # Optional: verbose feedback
        except Exception as e:
            print(f"Error saving .npy file to {file_path}: {e}")

    def load_csv(self, file_path: str, **kwargs: Any) -> Optional[pd.DataFrame]:
        """Loads data from a .csv file into a Pandas DataFrame."""
        if not os.path.exists(file_path):
            print(f"Error: .csv file not found at {file_path}")
            return None
        try:
            return pd.read_csv(file_path, **kwargs)
        except Exception as e:
            print(f"Error loading .csv file {file_path}: {e}")
            return None

    def save_csv(self, df: pd.DataFrame, file_path: str, **kwargs: Any):
        """Saves a Pandas DataFrame to a .csv file."""
        try:
            df.to_csv(file_path, index=False, **kwargs)
            # print(f"DataFrame saved to {file_path}")
        except Exception as e:
            print(f"Error saving .csv file to {file_path}: {e}")

    def load_json(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Loads data from a .json file."""
        if not os.path.exists(file_path):
            print(f"Error: .json file not found at {file_path}")
            return None
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading .json file {file_path}: {e}")
            return None

    def save_json(self, data: Dict[str, Any], file_path: str, indent: Optional[int] = 4):
        """Saves data to a .json file."""
        try:
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=indent)
            # print(f"Data saved to {file_path}")
        except Exception as e:
            print(f"Error saving .json file to {file_path}: {e}")


if __name__ == "__main__":
    print("--- Testing DataIO Utilities ---")
    data_io = DataIO()

    # Create a temporary directory for testing
    test_dir = "temp_data_io_test"
    os.makedirs(test_dir, exist_ok=True)

    # --- Test NumPy Array I/O ---
    test_numpy_array = np.random.rand(10, 5)
    numpy_file = os.path.join(test_dir, "test_array.npy")
    data_io.save_numpy(test_numpy_array, numpy_file)
    loaded_numpy_array = data_io.load_numpy(numpy_file)
    print("\nNumPy Array Test:")
    print("Original shape:", test_numpy_array.shape)
    print("Loaded shape:", loaded_numpy_array.shape if loaded_numpy_array is not None else "None")
    assert loaded_numpy_array is not None and np.allclose(test_numpy_array, loaded_numpy_array)
    print("NumPy array I/O: SUCCESS")

    # --- Test CSV DataFrame I/O ---
    test_df_data = {'col1': np.arange(5), 'col2': np.random.rand(5)}
    test_dataframe = pd.DataFrame(test_df_data)
    csv_file = os.path.join(test_dir, "test_dataframe.csv")
    data_io.save_csv(test_dataframe, csv_file)
    loaded_dataframe = data_io.load_csv(csv_file)
    print("\nCSV DataFrame Test:")
    print("Original DataFrame:\n", test_dataframe)
    print("Loaded DataFrame:\n", loaded_dataframe)
    assert loaded_dataframe is not None and test_dataframe.equals(loaded_dataframe)
    print("CSV DataFrame I/O: SUCCESS")

    # --- Test JSON I/O ---
    test_json_data = {"name": "Janus", "version": 1.0, "metrics": {"loss": 0.01, "accuracy": 0.95}}
    json_file = os.path.join(test_dir, "test_data.json")
    data_io.save_json(test_json_data, json_file)
    loaded_json_data = data_io.load_json(json_file)
    print("\nJSON Test:")
    print("Original JSON:", test_json_data)
    print("Loaded JSON:", loaded_json_data)
    assert loaded_json_data is not None and test_json_data == loaded_json_data
    print("JSON I/O: SUCCESS")

    # --- Test Error Handling (Non-existent files) ---
    print("\nError Handling Test (expect warnings/errors):")
    non_existent_file = os.path.join(test_dir, "non_existent.npy")
    loaded_none = data_io.load_numpy(non_existent_file)
    assert loaded_none is None
    print("Loading non-existent .npy: Handled (returns None)")

    # Clean up the temporary directory
    import shutil
    shutil.rmtree(test_dir)
    print(f"\nCleaned up test directory: {test_dir}")

    print("\nAll DataIO tests completed.")

