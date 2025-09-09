"""Tests for different write modes (preallocate, new, update, append)."""

import os
import tempfile

import h5py
import numpy as np
import pandas as pd
import pytest

from pandas2hdf import (
    load_series,
    preallocate_series_layout,
    save_series_append,
    save_series_new,
    save_series_update,
)
from pandas2hdf.core import SchemaMismatchError, ValidationError


@pytest.fixture
def temp_hdf5_file():
    """Create a temporary HDF5 file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        temp_path = f.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


class TestSeriesWriteModes:
    """Test different Series write modes."""

    def test_preallocate_layout(self, temp_hdf5_file):
        """Test preallocating Series layout."""
        series = pd.Series([1, 2, 3], index=["a", "b", "c"], name="test")

        with h5py.File(temp_hdf5_file, "w", libver="latest") as f:
            f.swmr_mode = True
            group = f.create_group("series")

            preallocate_series_layout(group, series, preallocate=100, require_swmr=True)

            # Check preallocated structure
            assert group.attrs["len"] == 0
            assert group["values"].shape == (100,)
            assert group["index/values"].shape == (100,)
            assert group["index/index_mask"].shape == (100,)

            # Should be able to load empty series
            loaded = load_series(group, require_swmr=True)
            assert len(loaded) == 0
            assert loaded.name == "test"

    def test_save_new_with_preallocated(self, temp_hdf5_file):
        """Test save_new with existing preallocated layout."""
        series = pd.Series([1, 2, 3], index=["a", "b", "c"], name="test")

        with h5py.File(temp_hdf5_file, "w", libver="latest") as f:
            f.swmr_mode = True
            group = f.create_group("series")

            # Preallocate first
            preallocate_series_layout(group, series, preallocate=100, require_swmr=True)

            # Then save data (should reuse layout)
            save_series_new(group, series, require_swmr=True)

            # Check data was written
            assert group.attrs["len"] == 3
            loaded = load_series(group, require_swmr=True)
            pd.testing.assert_series_equal(loaded, series.astype(np.float64))

    def test_save_update(self, temp_hdf5_file):
        """Test updating Series data."""
        initial = pd.Series([1, 2, 3], index=["a", "b", "c"], name="test")
        update = pd.Series([10, 20], index=["x", "y"], name="test")

        with h5py.File(temp_hdf5_file, "w", libver="latest") as f:
            f.swmr_mode = True
            group = f.create_group("series")

            # Save initial data
            save_series_new(group, initial, require_swmr=True)

            # Update at position 1
            save_series_update(group, update, start=1, require_swmr=True)

            # Check result
            assert group.attrs["len"] == 3  # max(3, 1+2)
            loaded = load_series(group, require_swmr=True)
            expected = pd.Series([1.0, 10.0, 20.0], index=["a", "x", "y"], name="test")
            pd.testing.assert_series_equal(loaded, expected)

    def test_save_append(self, temp_hdf5_file):
        """Test appending to Series data."""
        initial = pd.Series([1, 2], index=["a", "b"], name="test")
        append_data = pd.Series([3, 4], index=["c", "d"], name="test")

        with h5py.File(temp_hdf5_file, "w", libver="latest") as f:
            f.swmr_mode = True
            group = f.create_group("series")

            # Save initial data
            save_series_new(group, initial, require_swmr=True)

            # Append more data
            save_series_append(group, append_data, require_swmr=True)

            # Check result
            assert group.attrs["len"] == 4
            loaded = load_series(group, require_swmr=True)
            expected = pd.Series(
                [1.0, 2.0, 3.0, 4.0], index=["a", "b", "c", "d"], name="test"
            )
            pd.testing.assert_series_equal(loaded, expected)

    def test_non_contiguous_update_error(self, temp_hdf5_file):
        """Test that non-contiguous updates raise errors."""
        initial = pd.Series([1, 2], index=["a", "b"], name="test")
        update = pd.Series([10], index=["x"], name="test")

        with h5py.File(temp_hdf5_file, "w", libver="latest") as f:
            f.swmr_mode = True
            group = f.create_group("series")

            save_series_new(group, initial, require_swmr=True)

            # Try to update at position 5 (non-contiguous)
            with pytest.raises(ValidationError, match="Non-contiguous update"):
                save_series_update(group, update, start=5, require_swmr=True)

    def test_schema_mismatch_error(self, temp_hdf5_file):
        """Test schema mismatch detection."""
        numeric_series = pd.Series([1, 2, 3], name="test")
        string_series = pd.Series(["a", "b"], name="test")

        with h5py.File(temp_hdf5_file, "w", libver="latest") as f:
            f.swmr_mode = True
            group = f.create_group("series")

            # Save numeric series first
            save_series_new(group, numeric_series, require_swmr=True)

            # Try to update with string data
            with pytest.raises(SchemaMismatchError, match="Values kind mismatch"):
                save_series_update(group, string_series, start=0, require_swmr=True)

    def test_multiple_appends(self, temp_hdf5_file):
        """Test multiple append operations."""
        series1 = pd.Series([1, 2], index=["a", "b"], name="test")
        series2 = pd.Series([3], index=["c"], name="test")
        series3 = pd.Series([4, 5], index=["d", "e"], name="test")

        with h5py.File(temp_hdf5_file, "w", libver="latest") as f:
            f.swmr_mode = True
            group = f.create_group("series")

            # Initial data
            save_series_new(group, series1, require_swmr=True)
            assert group.attrs["len"] == 2

            # First append
            save_series_append(group, series2, require_swmr=True)
            assert group.attrs["len"] == 3

            # Second append
            save_series_append(group, series3, require_swmr=True)
            assert group.attrs["len"] == 5

            # Verify final result
            loaded = load_series(group, require_swmr=True)
            expected = pd.Series(
                [1.0, 2.0, 3.0, 4.0, 5.0], index=["a", "b", "c", "d", "e"], name="test"
            )
            pd.testing.assert_series_equal(loaded, expected)

    def test_update_at_end(self, temp_hdf5_file):
        """Test update operation at the end (equivalent to append)."""
        initial = pd.Series([1, 2], index=["a", "b"], name="test")
        update = pd.Series([3, 4], index=["c", "d"], name="test")

        with h5py.File(temp_hdf5_file, "w", libver="latest") as f:
            f.swmr_mode = True
            group = f.create_group("series")

            save_series_new(group, initial, require_swmr=True)
            save_series_update(group, update, start=2, require_swmr=True)

            loaded = load_series(group, require_swmr=True)
            expected = pd.Series(
                [1.0, 2.0, 3.0, 4.0], index=["a", "b", "c", "d"], name="test"
            )
            pd.testing.assert_series_equal(loaded, expected)
