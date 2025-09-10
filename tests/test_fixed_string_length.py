"""Tests for fixed-length string behavior and padding/truncation."""

import os
import tempfile

import h5py
import numpy as np
import pandas as pd
import pytest

from pandas2hdf import (
    load_frame,
    load_series,
    save_frame_new,
    save_series_new,
)
from pandas2hdf.core import SWMRModeError


@pytest.fixture
def temp_hdf5_file():
    """Create a temporary HDF5 file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        temp_path = f.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


class TestFixedLengthStrings:
    """Test fixed-length string behavior."""

    def test_series_string_padding_truncation(self, temp_hdf5_file):
        """Test padding/truncation behavior for Series with fixed-length strings."""
        # Test data with strings of various lengths
        data = [
            "abc",  # Shorter than fixed length (5)
            "12345",  # Exactly fixed length (5)
            "abcdefghij",  # Longer than fixed length (10 chars, will be truncated to 5)
            None,  # Missing value
            "",  # Empty string
            "  spaces  ",  # String with trailing spaces
        ]
        series = pd.Series(data, name="test_strings")

        with h5py.File(temp_hdf5_file, "w", libver="latest") as f:
            group = f.create_group("series")

            # Save with fixed length of 5 characters
            save_series_new(group, series, string_fixed_length=5, require_swmr=False)
            loaded = load_series(group, require_swmr=False)

            # Check attributes
            assert group.attrs["values_kind"] == "string_utf8_fixed"
            assert group.attrs["string_fixed_length"] == 5

            # Expected values after padding/truncation and trimming
            expected_values = [
                "abc",  # No trailing spaces after trimming
                "12345",  # Exact length
                "abcde",  # Truncated to 5 characters
                None,  # Missing value preserved
                "",  # Empty string
                "  spa",  # Truncated and internal spaces preserved
            ]

            np.testing.assert_array_equal(loaded.values, expected_values)
            assert loaded.name == series.name

    def test_series_index_string_padding_truncation(self, temp_hdf5_file):
        """Test padding/truncation behavior for Series index with fixed-length strings."""
        # Test index with strings of various lengths
        index = ["abc", "12345", "abcdefghij", "  pad  ", ""]
        data = [1, 2, 3, 4, 5]
        series = pd.Series(data, index=index, name="test_data")

        with h5py.File(temp_hdf5_file, "w", libver="latest") as f:
            group = f.create_group("series")

            # Save with fixed length of 5 characters
            save_series_new(group, series, string_fixed_length=5, require_swmr=False)
            loaded = load_series(group, require_swmr=False)

            # Check attributes
            assert group.attrs["index_kind"] == "string_utf8_fixed"
            assert group.attrs["index_string_fixed_length"] == 5

            # Expected index values after padding/truncation and trimming
            expected_index = ["abc", "12345", "abcde", "  pad", ""]

            # Check values are correct (numeric data)
            np.testing.assert_array_equal(
                loaded.values, series.astype(np.float64).values
            )
            # Check index values are correctly processed
            np.testing.assert_array_equal(loaded.index.values, expected_index)
            assert loaded.name == series.name

    def test_dataframe_string_padding_truncation(self, temp_hdf5_file):
        """Test padding/truncation behavior for DataFrame with fixed-length strings."""
        # Test DataFrame with mixed column types
        df = pd.DataFrame(
            {
                "string_col": ["short", "exactly5", "verylongstring", None, ""],
                "numeric_col": [1, 2, 3, 4, 5],
            }
        )

        with h5py.File(temp_hdf5_file, "w", libver="latest") as f:
            group = f.create_group("frame")

            # Save with fixed length of 5 characters
            save_frame_new(group, df, string_fixed_length=5, require_swmr=False)
            loaded = load_frame(group, require_swmr=False)

            # Check that string column is properly processed
            expected_string_col = ["short", "exact", "veryl", None, ""]

            # Numeric column should be unchanged (converted to float64)
            expected_numeric_col = [1.0, 2.0, 3.0, 4.0, 5.0]

            np.testing.assert_array_equal(
                loaded["string_col"].values, expected_string_col
            )
            np.testing.assert_array_equal(
                loaded["numeric_col"].values, expected_numeric_col
            )
            assert list(loaded.columns) == list(df.columns)

    def test_series_multiindex_string_padding_truncation(self, temp_hdf5_file):
        """Test padding/truncation behavior for Series with MultiIndex."""
        # Create MultiIndex with strings of various lengths
        index = pd.MultiIndex.from_tuples(
            [
                ("levelA", "short"),
                ("levelB", "exactly5"),
                ("levelC", "verylongstring"),
                ("levelD", None),
            ],
            names=["level1", "level2"],
        )

        data = [10, 20, 30, 40]
        series = pd.Series(data, index=index, name="multi_data")

        with h5py.File(temp_hdf5_file, "w", libver="latest") as f:
            group = f.create_group("series")

            # Save with fixed length of 5 characters
            save_series_new(group, series, string_fixed_length=5, require_swmr=False)
            loaded = load_series(group, require_swmr=False)

            # Check attributes
            assert group.attrs["index_is_multiindex"] == 1
            assert group.attrs["index_kind"] == "string_utf8_fixed"
            assert group.attrs["index_string_fixed_length"] == 5

            # Check that MultiIndex is preserved
            assert isinstance(loaded.index, pd.MultiIndex)
            assert loaded.index.names == ["level1", "level2"]

            # Check values
            np.testing.assert_array_equal(
                loaded.values, series.astype(np.float64).values
            )
            assert loaded.name == series.name

    def test_string_round_trip_with_trailing_spaces(self, temp_hdf5_file):
        """Test that trailing spaces are properly handled in round-trip."""
        # Test data with intentional trailing spaces
        data = [
            "abc",
            "def   ",  # Trailing spaces
            "ghi\t",  # Trailing tab
            "jkl \n",  # Trailing space and newline
            "mno",
        ]
        series = pd.Series(data, name="spaces_test")

        with h5py.File(temp_hdf5_file, "w", libver="latest") as f:
            group = f.create_group("series")

            # Save with fixed length of 10 characters (longer than any input)
            save_series_new(group, series, string_fixed_length=10, require_swmr=False)
            loaded = load_series(group, require_swmr=False)

            # Expected values: trailing whitespace should be trimmed on load
            expected_values = [
                "abc",
                "def",  # Trailing spaces trimmed
                "ghi",  # Trailing tab trimmed
                "jkl",  # Trailing space and newline trimmed
                "mno",
            ]

            np.testing.assert_array_equal(loaded.values, expected_values)
            assert loaded.name == series.name


class TestSWMRModelEnforcement:
    """Test SWMR programming model enforcement."""

    def test_cannot_create_datasets_under_swmr(self, temp_hdf5_file):
        """Test that attempting to create datasets under SWMR raises error."""
        series = pd.Series([1, 2, 3], name="test")

        with h5py.File(temp_hdf5_file, "w", libver="latest") as f:
            # Start SWMR mode before creating any objects
            f.swmr_mode = True
            group = f.create_group("series")

            # Should raise error when trying to create new datasets under SWMR
            with pytest.raises(
                SWMRModeError,
                match="Datasets must be created before starting SWMR mode",
            ):
                save_series_new(group, series, require_swmr=True)

    def test_cannot_write_vlen_strings_under_swmr(self, temp_hdf5_file):
        """Test that writing to variable-length string datasets under SWMR raises error."""
        # This test simulates an old file with variable-length strings
        series = pd.Series(["a", "b", "c"], name="test")

        # Step 1: Create a mock file layout with variable-length strings (simulating old behavior)
        with h5py.File(temp_hdf5_file, "w", libver="latest") as f:
            group = f.create_group("series")

            # Create datasets manually to simulate old vlen format
            string_dtype = h5py.string_dtype("utf-8")  # Variable-length

            # Create values dataset
            values_dataset = group.create_dataset(
                "values", shape=(3,), maxshape=(None,), dtype=string_dtype
            )
            values_dataset[:] = ["a", "b", "c"]

            # Create values mask
            mask_dataset = group.create_dataset(
                "values_mask", shape=(3,), maxshape=(None,), dtype=np.uint8
            )
            mask_dataset[:] = [1, 1, 1]

            # Create index datasets
            index_group = group.create_group("index")
            index_values_dataset = index_group.create_dataset(
                "values", shape=(3,), maxshape=(None,), dtype=string_dtype
            )
            index_values_dataset[:] = ["0", "1", "2"]

            index_mask_dataset = index_group.create_dataset(
                "index_mask", shape=(3,), maxshape=(None,), dtype=np.uint8
            )
            index_mask_dataset[:] = [1, 1, 1]

            # Set attributes to simulate old vlen format
            group.attrs["series_name"] = "test"
            group.attrs["len"] = 3
            group.attrs["values_kind"] = "string_utf8_vlen"
            group.attrs["index_kind"] = "string_utf8_vlen"
            group.attrs["orig_values_dtype"] = "object"
            group.attrs["orig_index_dtype"] = "int64"
            group.attrs["index_is_multiindex"] = 0
            group.attrs["index_levels"] = 1
            group.attrs["index_names"] = '["__index__"]'

        # Step 2: Reopen in SWMR mode and try to write
        with h5py.File(temp_hdf5_file, "a", libver="latest") as f:
            f.swmr_mode = True
            group = f["series"]

            # Step 3: Attempt to write to variable-length datasets under SWMR
            new_series = pd.Series(["x", "y", "z"], name="test")
            from pandas2hdf import save_series_update

            with pytest.raises(
                SWMRModeError,
                match="Cannot write to variable-length string datasets under SWMR",
            ):
                save_series_update(group, new_series, start=0, require_swmr=True)

    def test_correct_swmr_workflow_with_fixed_strings(self, temp_hdf5_file):
        """Test that the correct SWMR workflow works with fixed-length strings."""
        series = pd.Series(["a", "b", "c"], name="test")
        append_series = pd.Series(["d", "e"], name="test")

        with h5py.File(temp_hdf5_file, "w", libver="latest") as f:
            group = f.create_group("series")

            # Step 1: Create all objects with fixed-length strings BEFORE enabling SWMR
            save_series_new(group, series, string_fixed_length=5, require_swmr=False)

            # Verify fixed-length attributes are set
            assert group.attrs["values_kind"] == "string_utf8_fixed"
            assert group.attrs["string_fixed_length"] == 5

            # Step 2: Start SWMR mode
            f.swmr_mode = True

            # Step 3: Write more data under SWMR (should work with fixed-length)
            from pandas2hdf import save_series_append

            save_series_append(group, append_series, require_swmr=True)

            # Verify final length
            assert group.attrs["len"] == 5

        # Verify data can be read correctly
        with h5py.File(temp_hdf5_file, "r") as f:
            group = f["series"]
            loaded = load_series(group, require_swmr=False)

            expected_values = ["a", "b", "c", "d", "e"]
            np.testing.assert_array_equal(loaded.values, expected_values)
            assert loaded.name == "test"

    def test_dataframe_multiindex_with_fixed_strings(self, temp_hdf5_file):
        """Test DataFrame with MultiIndex using fixed-length strings."""
        index = pd.MultiIndex.from_tuples(
            [
                ("levelA", "short"),
                ("levelB", "exactly5"),
                ("levelC", "verylongstring"),
                ("levelD", None),
            ],
            names=["level1", "level2"],
        )

        df = pd.DataFrame(
            {
                "string_col": ["short", "exactly5", "verylongstring", None],
                "numeric_col": [1, 2, 3, 4],
            },
            index=index,
        )

        with h5py.File(temp_hdf5_file, "w", libver="latest") as f:
            group = f.create_group("frame")

            # Save with fixed length of 5 characters
            save_frame_new(group, df, string_fixed_length=5, require_swmr=False)
            loaded = load_frame(group, require_swmr=False)

            # Check that MultiIndex is preserved
            assert isinstance(loaded.index, pd.MultiIndex)
            assert loaded.index.names == ["level1", "level2"]

            # Check string column is properly processed
            expected_string_col = ["short", "exact", "veryl", None]

            # Numeric column should be unchanged (converted to float64)
            expected_numeric_col = [1.0, 2.0, 3.0, 4.0]

            np.testing.assert_array_equal(
                loaded["string_col"].values, expected_string_col
            )
            np.testing.assert_array_equal(
                loaded["numeric_col"].values, expected_numeric_col
            )
            assert list(loaded.columns) == list(df.columns)

    def test_preallocate_with_fixed_strings(self, temp_hdf5_file):
        """Test preallocate with fixed-length strings."""
        series = pd.Series(["a", "b", "c"], name="test")

        with h5py.File(temp_hdf5_file, "w", libver="latest") as f:
            group = f.create_group("series")

            # Step 1: Create all objects BEFORE enabling SWMR
            from pandas2hdf import preallocate_series_layout

            preallocate_series_layout(
                group,
                series,
                preallocate=100,
                string_fixed_length=10,
                require_swmr=False,
            )

            # Check attributes are set correctly
            assert group.attrs["values_kind"] == "string_utf8_fixed"
            assert group.attrs["string_fixed_length"] == 10

            # Step 2: Start SWMR mode
            f.swmr_mode = True

            # Step 3: Write data under SWMR
            save_series_new(group, series, require_swmr=True)

            # Verify final result
            loaded = load_series(group, require_swmr=True)
            np.testing.assert_array_equal(loaded.values, series.values)
            assert loaded.name == series.name

    def test_update_operations_with_fixed_strings(self, temp_hdf5_file):
        """Test update operations with fixed-length strings."""
        initial = pd.Series(["abc", "def", "ghi"], name="test")
        update = pd.Series(["verylongstring", "xyz"], name="test")

        with h5py.File(temp_hdf5_file, "w", libver="latest") as f:
            group = f.create_group("series")

            # Step 1: Create all objects BEFORE enabling SWMR
            save_series_new(group, initial, string_fixed_length=5, require_swmr=False)

            # Step 2: Start SWMR mode
            f.swmr_mode = True

            # Step 3: Update data under SWMR
            from pandas2hdf import save_series_update

            save_series_update(group, update, start=1, require_swmr=True)

            # Verify result
            loaded = load_series(group, require_swmr=True)

            # Expected: first value unchanged, second and third updated and truncated
            expected_values = ["abc", "veryl", "xyz"]
            np.testing.assert_array_equal(loaded.values, expected_values)
            assert loaded.name == "test"


if __name__ == "__main__":
    pytest.main([__file__])
