"""Comprehensive tests for pandas2hdf library."""

import json
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
    preallocate_frame_layout,
    preallocate_series_layout,
    save_frame_append,
    save_frame_new,
    save_frame_update,
    save_series_append,
    save_series_new,
    save_series_update,
)
from pandas2hdf.core import (
    SchemaMismatchError,
    SWMRModeError,
    ValidationError,
)


@pytest.fixture
def temp_hdf5_file():
    """Create a temporary HDF5 file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        temp_path = f.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def hdf5_file_swmr(temp_hdf5_file):
    """Create an HDF5 file with SWMR mode enabled."""
    with h5py.File(temp_hdf5_file, "w", libver="latest") as f:
        f.swmr_mode = True
        yield f


@pytest.fixture
def hdf5_file_no_swmr(temp_hdf5_file):
    """Create an HDF5 file without SWMR mode."""
    with h5py.File(temp_hdf5_file, "w") as f:
        yield f


class TestSWMRAssertions:
    """Test SWMR mode assertions."""

    def test_assert_swmr_on_success(self, hdf5_file_swmr):
        """Test that assert_swmr_on passes when SWMR is enabled."""
        group = hdf5_file_swmr.create_group("test")
        assert_swmr_on(group)  # Should not raise

    def test_assert_swmr_on_failure(self, hdf5_file_no_swmr):
        """Test that assert_swmr_on raises when SWMR is disabled."""
        group = hdf5_file_no_swmr.create_group("test")
        with pytest.raises(SWMRModeError, match="SWMR mode is required"):
            assert_swmr_on(group)


class TestSeriesIO:
    """Test Series persistence functionality."""

    @pytest.mark.parametrize(
        "dtype,expected_kind",
        [
            (np.float32, "numeric_float64"),
            (np.float64, "numeric_float64"),
            (bool, "numeric_float64"),
            (str, "string_utf8_vlen"),
            (object, "string_utf8_vlen"),
        ],
    )
    def test_series_round_trip_dtypes(self, temp_hdf5_file, dtype, expected_kind):
        """Test round-trip for different data types."""
        # Create test data
        if dtype in [str, object]:
            data = ["apple", "banana", "cherry", None, "date"]
        elif dtype in [bool]:
            data = [
                True,
                False,
                True,
                False,
                False,
            ]  # Can't have None with regular bool
        else:
            data = [
                1.0,
                2.0,
                3.0,
                None,
                5.0,
            ]  # Use floats to avoid integer + None issues

        series = pd.Series(data, dtype=dtype, name="test_series")

        with h5py.File(temp_hdf5_file, "w", libver="latest") as f:
            f.swmr_mode = True
            group = f.create_group("series")

            # Save and load
            save_series_new(group, series, require_swmr=True)
            loaded = load_series(group, require_swmr=True)

            # Check values kind attribute
            values_kind_attr = group.attrs["values_kind"]
            if isinstance(values_kind_attr, bytes):
                values_kind_attr = values_kind_attr.decode("utf-8")
            assert values_kind_attr == expected_kind

            # For numeric/boolean data, check round-trip
            if expected_kind == "numeric_float64":
                # Boolean and numeric data becomes float64 with NaN for missing
                expected_values = pd.Series(data, dtype=np.float64, name="test_series")
                # Index becomes string type when stored in HDF5
                np.testing.assert_array_equal(loaded.values, expected_values.values)
                assert loaded.name == expected_values.name
            else:
                # String data preserves None values, but index becomes string type
                np.testing.assert_array_equal(loaded.values, series.values)
                assert loaded.name == series.name

    def test_series_with_index(self, temp_hdf5_file):
        """Test Series with custom Index."""
        series = pd.Series([1, 2, 3, 4], index=["a", "b", "c", "d"], name="indexed")

        with h5py.File(temp_hdf5_file, "w", libver="latest") as f:
            f.swmr_mode = True
            group = f.create_group("series")

            save_series_new(group, series, require_swmr=True)
            loaded = load_series(group, require_swmr=True)

            pd.testing.assert_series_equal(loaded, series.astype(np.float64))
            assert loaded.name == "indexed"

    def test_series_with_multiindex(self, temp_hdf5_file):
        """Test Series with MultiIndex."""
        index = pd.MultiIndex.from_tuples(
            [("A", 1), ("A", 2), ("B", 1), ("B", 2), ("C", None)],
            names=["level1", "level2"],
        )
        series = pd.Series([10, 20, 30, 40, 50], index=index, name="multi_series")

        with h5py.File(temp_hdf5_file, "w", libver="latest") as f:
            f.swmr_mode = True
            group = f.create_group("series")

            save_series_new(group, series, require_swmr=True)
            loaded = load_series(group, require_swmr=True)

            # Check attributes
            assert group.attrs["index_is_multiindex"] == 1
            assert group.attrs["index_levels"] == 2
            index_names_attr = group.attrs["index_names"]
            if isinstance(index_names_attr, bytes):
                index_names_attr = index_names_attr.decode("utf-8")
            assert json.loads(index_names_attr) == ["level1", "level2"]

            # Check that values are correct (index types will differ due to string storage)
            np.testing.assert_array_equal(
                loaded.values, series.astype(np.float64).values
            )
            assert loaded.name == series.name
            assert isinstance(loaded.index, pd.MultiIndex)
            assert loaded.index.names == series.index.names


if __name__ == "__main__":
    pytest.main([__file__])
