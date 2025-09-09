"""Tests for SWMR workflows and error handling."""

import os
import tempfile

import h5py
import numpy as np
import pandas as pd
import pytest

from pandas2hdf import (
    assert_swmr_on,
    load_frame,
    load_series,
    preallocate_series_layout,
    save_frame_append,
    save_frame_new,
    save_series_append,
    save_series_new,
)
from pandas2hdf.core import SWMRModeError, ValidationError


@pytest.fixture
def temp_hdf5_file():
    """Create a temporary HDF5 file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        temp_path = f.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


class TestSWMRWorkflows:
    """Test complete SWMR workflows."""

    def test_swmr_workflow_series(self, temp_hdf5_file):
        """Test complete SWMR workflow for Series."""
        # Initial data
        initial = pd.Series([1, 2], index=["a", "b"], name="timeseries")
        append1 = pd.Series([3, 4], index=["c", "d"], name="timeseries")
        append2 = pd.Series([5], index=["e"], name="timeseries")

        # Writer process
        with h5py.File(temp_hdf5_file, "w", libver="latest") as writer_file:
            writer_file.swmr_mode = True
            group = writer_file.create_group("series")

            # Preallocate and write initial data
            preallocate_series_layout(
                group, initial, preallocate=100, require_swmr=True
            )
            save_series_new(group, initial, require_swmr=True)

            # Append more data
            save_series_append(group, append1, require_swmr=True)
            save_series_append(group, append2, require_swmr=True)

            # Ensure data is flushed
            writer_file.flush()

            final_length = group.attrs["len"]
            assert final_length == 5

        # Reader process (simulate concurrent read)
        with h5py.File(temp_hdf5_file, "r", swmr=True) as reader_file:
            group = reader_file["series"]
            loaded = load_series(group, require_swmr=False)

            expected = pd.Series(
                [1.0, 2.0, 3.0, 4.0, 5.0],
                index=["a", "b", "c", "d", "e"],
                name="timeseries",
            )
            pd.testing.assert_series_equal(loaded, expected)

    def test_swmr_workflow_dataframe(self, temp_hdf5_file):
        """Test complete SWMR workflow for DataFrame."""
        initial_df = pd.DataFrame({"col1": [1], "col2": ["a"]})
        append_df = pd.DataFrame({"col1": [2], "col2": ["b"]})

        # Writer
        with h5py.File(temp_hdf5_file, "w", libver="latest") as writer_file:
            writer_file.swmr_mode = True
            group = writer_file.create_group("frame")

            save_frame_new(group, initial_df, require_swmr=True)
            save_frame_append(group, append_df, require_swmr=True)
            writer_file.flush()

        # Reader
        with h5py.File(temp_hdf5_file, "r", swmr=True) as reader_file:
            group = reader_file["frame"]
            loaded = load_frame(group, require_swmr=False)

            expected = pd.DataFrame({"col1": [1.0, 2.0], "col2": ["a", "b"]})
            # Compare values and structure (index types will differ due to string storage)
            np.testing.assert_array_equal(loaded.values, expected.values)
            assert list(loaded.columns) == list(expected.columns)
            assert loaded.shape == expected.shape

    def test_swmr_incremental_growth(self, temp_hdf5_file):
        """Test incremental growth pattern common in SWMR scenarios."""
        # Simulate time-series data being written incrementally
        time_series = []

        with h5py.File(temp_hdf5_file, "w", libver="latest") as writer_file:
            writer_file.swmr_mode = True
            group = writer_file.create_group("series")

            # Initial batch
            batch1 = pd.Series(
                [1.0, 2.0], index=["2023-01-01", "2023-01-02"], name="data"
            )
            save_series_new(group, batch1, preallocate=1000, require_swmr=True)
            time_series.extend(batch1.tolist())

            # Incremental appends
            for i in range(3, 11):  # Add 8 more data points
                new_data = pd.Series(
                    [float(i)], index=[f"2023-01-{i:02d}"], name="data"
                )
                save_series_append(group, new_data, require_swmr=True)
                time_series.append(float(i))
                writer_file.flush()

        # Verify final result
        with h5py.File(temp_hdf5_file, "r", swmr=True) as reader_file:
            group = reader_file["series"]
            loaded = load_series(group, require_swmr=False)

            assert len(loaded) == 10
            assert loaded.name == "data"
            np.testing.assert_array_equal(loaded.values, time_series)


class TestRequireSWMRParameter:
    """Test require_swmr parameter behavior."""

    def test_functions_work_without_swmr_when_not_required(self, temp_hdf5_file):
        """Test that require_swmr=False allows operations without SWMR."""
        series = pd.Series([1, 2, 3], name="test")
        df = pd.DataFrame({"col": [1, 2]})

        with h5py.File(temp_hdf5_file, "w") as f:  # No SWMR mode
            series_group = f.create_group("series")
            frame_group = f.create_group("frame")

            # All these should work with require_swmr=False
            preallocate_series_layout(series_group, series, require_swmr=False)
            save_series_new(series_group, series, require_swmr=False)
            loaded_series = load_series(series_group, require_swmr=False)

            # Verify series round-trip (index becomes string type when stored)
            np.testing.assert_array_equal(
                loaded_series.values, series.astype(np.float64).values
            )
            assert loaded_series.name == series.name

            # Test frame operations
            save_frame_new(frame_group, df, require_swmr=False)
            loaded_frame = load_frame(frame_group, require_swmr=False)

            # Verify frame round-trip (index becomes string type when stored)
            expected_df = df.astype(np.float64)
            np.testing.assert_array_equal(loaded_frame.values, expected_df.values)
            assert list(loaded_frame.columns) == list(expected_df.columns)

    def test_functions_raise_when_swmr_required_but_not_enabled(self, temp_hdf5_file):
        """Test that require_swmr=True raises when SWMR is not enabled."""
        series = pd.Series([1, 2, 3], name="test")

        with h5py.File(temp_hdf5_file, "w") as f:  # No SWMR mode
            group = f.create_group("series")

            # Should raise for write operations
            with pytest.raises(SWMRModeError):
                preallocate_series_layout(group, series, require_swmr=True)

            # Create layout without SWMR requirement for testing other functions
            preallocate_series_layout(group, series, require_swmr=False)

            with pytest.raises(SWMRModeError):
                save_series_new(group, series, require_swmr=True)

            # Add some data for read test
            save_series_new(group, series, require_swmr=False)

            with pytest.raises(SWMRModeError):
                load_series(group, require_swmr=True)


class TestErrorHandling:
    """Test error handling and validation."""

    def test_empty_series_validation(self, temp_hdf5_file):
        """Test validation of empty series."""
        empty_series = pd.Series([], dtype=np.float64, name="empty")

        with h5py.File(temp_hdf5_file, "w", libver="latest") as f:
            f.swmr_mode = True
            group = f.create_group("series")

            with pytest.raises(ValidationError, match="Cannot save empty series"):
                save_series_new(group, empty_series, require_swmr=True)

    def test_empty_dataframe_validation(self, temp_hdf5_file):
        """Test validation of empty DataFrame."""
        empty_df = pd.DataFrame()

        with h5py.File(temp_hdf5_file, "w", libver="latest") as f:
            f.swmr_mode = True
            group = f.create_group("frame")

            with pytest.raises(ValidationError, match="Cannot save empty DataFrame"):
                save_frame_new(group, empty_df, require_swmr=True)

    def test_no_columns_dataframe_validation(self, temp_hdf5_file):
        """Test validation of DataFrame with no columns."""
        no_cols_df = pd.DataFrame(index=[0, 1, 2])  # Has rows but no columns

        with h5py.File(temp_hdf5_file, "w", libver="latest") as f:
            f.swmr_mode = True
            group = f.create_group("frame")

            with pytest.raises(
                ValidationError,
                match="Cannot preallocate layout for DataFrame with no columns",
            ):
                save_frame_new(group, no_cols_df, require_swmr=True)

    def test_series_name_preservation(self, temp_hdf5_file):
        """Test that Series names are preserved correctly."""
        # Test with None name
        unnamed_series = pd.Series([1, 2, 3])
        assert unnamed_series.name is None

        with h5py.File(temp_hdf5_file, "w", libver="latest") as f:
            f.swmr_mode = True
            group1 = f.create_group("unnamed")

            save_series_new(group1, unnamed_series, require_swmr=True)
            loaded = load_series(group1, require_swmr=True)
            assert loaded.name is None

        # Test with string name
        named_series = pd.Series([1, 2, 3], name="my_series")

        with h5py.File(temp_hdf5_file, "a", libver="latest") as f:
            f.swmr_mode = True
            group2 = f.create_group("named")

            save_series_new(group2, named_series, require_swmr=True)
            loaded = load_series(group2, require_swmr=True)
            assert loaded.name == "my_series"

    def test_large_unicode_strings(self, temp_hdf5_file):
        """Test handling of large Unicode strings."""
        large_strings = [
            "x" * 10000,  # Large ASCII
            "🚀" * 5000,  # Large Unicode emoji
            "Testing with émojis and ǘnicøde ⚡" * 100,  # Mixed Unicode
            None,  # Missing value
            "",  # Empty string
        ]
        series = pd.Series(large_strings, name="large_unicode")

        with h5py.File(temp_hdf5_file, "w", libver="latest") as f:
            f.swmr_mode = True
            group = f.create_group("series")

            save_series_new(group, series, require_swmr=True)
            loaded = load_series(group, require_swmr=True)

            # Check that all strings round-trip correctly (index becomes string type)
            np.testing.assert_array_equal(loaded.values, series.values)
            assert loaded.name == series.name
            assert loaded.name == "large_unicode"
