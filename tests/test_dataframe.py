"""Tests for DataFrame functionality."""

import json
import os
import tempfile

import h5py
import numpy as np
import pandas as pd
import pytest

from pandas2hdf import (
    load_frame,
    preallocate_frame_layout,
    save_frame_append,
    save_frame_new,
    save_frame_update,
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


class TestDataFrameIO:
    """Test DataFrame persistence functionality."""

    def test_frame_round_trip_basic(self, temp_hdf5_file):
        """Test basic DataFrame round-trip."""
        df = pd.DataFrame(
            {
                "int_col": [1, 2, 3, None],
                "float_col": [1.1, 2.2, 3.3, 4.4],
                "str_col": ["a", "b", None, "d"],
                "bool_col": [True, False, True, None],
            }
        )

        with h5py.File(temp_hdf5_file, "w", libver="latest") as f:
            f.swmr_mode = True
            group = f.create_group("frame")

            save_frame_new(group, df, require_swmr=True)
            loaded = load_frame(group, require_swmr=True)

            # Check column order preserved
            assert list(loaded.columns) == list(df.columns)
            assert group.attrs["len"] == len(df)

            # Check data (numeric columns become float64, index becomes string)
            expected = df.copy()
            expected["int_col"] = expected["int_col"].astype(np.float64)
            expected["bool_col"] = expected["bool_col"].astype(np.float64)

            # Compare values and structure (index types will differ due to string storage)
            # Use pandas testing for better NaN handling
            for col in loaded.columns:
                if col == "str_col":
                    # String column with None values
                    pd.testing.assert_series_equal(
                        loaded[col].reset_index(drop=True),
                        expected[col].reset_index(drop=True),
                        check_dtype=False,
                    )
                else:
                    # Numeric columns
                    pd.testing.assert_series_equal(
                        loaded[col].reset_index(drop=True),
                        expected[col].reset_index(drop=True),
                        check_dtype=False,
                    )
            assert list(loaded.columns) == list(expected.columns)
            assert loaded.shape == expected.shape

    def test_frame_with_multiindex(self, temp_hdf5_file):
        """Test DataFrame with MultiIndex."""
        index = pd.MultiIndex.from_tuples(
            [("A", 1), ("A", 2), ("B", 1)], names=["level1", "level2"]
        )

        df = pd.DataFrame(
            {
                "col1": [10, 20, 30],
                "col2": ["x", "y", "z"],
            },
            index=index,
        )

        with h5py.File(temp_hdf5_file, "w", libver="latest") as f:
            f.swmr_mode = True
            group = f.create_group("frame")

            save_frame_new(group, df, require_swmr=True)
            loaded = load_frame(group, require_swmr=True)

            # Check MultiIndex preserved
            assert isinstance(loaded.index, pd.MultiIndex)
            assert loaded.index.names == ["level1", "level2"]

            expected = df.copy()
            expected["col1"] = expected["col1"].astype(np.float64)

            # Compare values and structure (index types will differ due to string storage)
            np.testing.assert_array_equal(loaded.values, expected.values)
            assert list(loaded.columns) == list(expected.columns)
            assert loaded.shape == expected.shape
            assert len(loaded.index.levels) == len(expected.index.levels)

    def test_frame_empty_error(self, temp_hdf5_file):
        """Test empty DataFrame handling."""
        empty_df = pd.DataFrame()

        with h5py.File(temp_hdf5_file, "w", libver="latest") as f:
            f.swmr_mode = True
            group = f.create_group("frame")

            with pytest.raises(ValidationError, match="Cannot save empty DataFrame"):
                save_frame_new(group, empty_df, require_swmr=True)

    def test_frame_single_column(self, temp_hdf5_file):
        """Test DataFrame with single column."""
        df = pd.DataFrame({"single_col": [1, 2, 3, 4, 5]})

        with h5py.File(temp_hdf5_file, "w", libver="latest") as f:
            f.swmr_mode = True
            group = f.create_group("frame")

            save_frame_new(group, df, require_swmr=True)
            loaded = load_frame(group, require_swmr=True)

            expected = df.astype(np.float64)
            # Index becomes string type when stored in HDF5
            np.testing.assert_array_equal(loaded.values, expected.values)
            assert list(loaded.columns) == list(expected.columns)

    def test_frame_mixed_dtypes(self, temp_hdf5_file):
        """Test DataFrame with mixed data types."""
        df = pd.DataFrame(
            {
                "int": [1, 2, 3],
                "float": [1.1, 2.2, 3.3],
                "str": ["a", "b", "c"],
                "bool": [True, False, True],
            }
        )

        with h5py.File(temp_hdf5_file, "w", libver="latest") as f:
            f.swmr_mode = True
            group = f.create_group("frame")

            save_frame_new(group, df, require_swmr=True)
            loaded = load_frame(group, require_swmr=True)

            # Numeric columns become float64
            expected = df.copy()
            expected["int"] = expected["int"].astype(np.float64)
            expected["bool"] = expected["bool"].astype(np.float64)

            # Compare values and structure (index types will differ due to string storage)
            np.testing.assert_array_equal(loaded.values, expected.values)
            assert list(loaded.columns) == list(expected.columns)
            assert loaded.shape == expected.shape


class TestDataFrameWriteModes:
    """Test DataFrame write modes."""

    def test_frame_preallocate(self, temp_hdf5_file):
        """Test DataFrame preallocation."""
        df = pd.DataFrame(
            {
                "col1": [1, 2],
                "col2": ["a", "b"],
            }
        )

        with h5py.File(temp_hdf5_file, "w", libver="latest") as f:
            f.swmr_mode = True
            group = f.create_group("frame")

            preallocate_frame_layout(group, df, preallocate=100, require_swmr=True)

            # Check structure
            assert group.attrs["len"] == 0
            assert "index" in group
            assert "columns" in group
            assert "col1" in group["columns"]
            assert "col2" in group["columns"]

            # Check column order
            column_order_attr = group.attrs["column_order"]
            if isinstance(column_order_attr, bytes):
                column_order_attr = column_order_attr.decode("utf-8")
            column_order = json.loads(column_order_attr)
            assert column_order == ["col1", "col2"]

    def test_frame_append(self, temp_hdf5_file):
        """Test DataFrame append functionality."""
        initial_df = pd.DataFrame(
            {
                "col1": [1, 2],
                "col2": ["a", "b"],
            }
        )

        append_df = pd.DataFrame(
            {
                "col1": [3, 4],
                "col2": ["c", "d"],
            }
        )

        with h5py.File(temp_hdf5_file, "w", libver="latest") as f:
            f.swmr_mode = True
            group = f.create_group("frame")

            save_frame_new(group, initial_df, require_swmr=True)
            save_frame_append(group, append_df, require_swmr=True)

            loaded = load_frame(group, require_swmr=True)

            expected = pd.concat([initial_df, append_df], ignore_index=True)
            expected["col1"] = expected["col1"].astype(np.float64)

            # Compare values and structure (index types will differ due to string storage)
            np.testing.assert_array_equal(loaded.values, expected.values)
            assert list(loaded.columns) == list(expected.columns)
            assert loaded.shape == expected.shape

    def test_frame_update(self, temp_hdf5_file):
        """Test DataFrame update functionality."""
        initial_df = pd.DataFrame(
            {
                "col1": [1, 2, 3],
                "col2": ["a", "b", "c"],
            }
        )

        update_df = pd.DataFrame(
            {
                "col1": [10, 20],
                "col2": ["x", "y"],
            }
        )

        with h5py.File(temp_hdf5_file, "w", libver="latest") as f:
            f.swmr_mode = True
            group = f.create_group("frame")

            save_frame_new(group, initial_df, require_swmr=True)
            save_frame_update(group, update_df, start=1, require_swmr=True)

            loaded = load_frame(group, require_swmr=True)

            expected = pd.DataFrame(
                {
                    "col1": [1.0, 10.0, 20.0],
                    "col2": ["a", "x", "y"],
                }
            )

            # Compare values and structure (index types will differ due to string storage)
            np.testing.assert_array_equal(loaded.values, expected.values)
            assert list(loaded.columns) == list(expected.columns)
            assert loaded.shape == expected.shape

    def test_frame_column_order_mismatch(self, temp_hdf5_file):
        """Test DataFrame column order validation."""
        initial_df = pd.DataFrame({"col1": [1], "col2": [2]})
        wrong_order_df = pd.DataFrame({"col2": [3], "col1": [4]})

        with h5py.File(temp_hdf5_file, "w", libver="latest") as f:
            f.swmr_mode = True
            group = f.create_group("frame")

            save_frame_new(group, initial_df, require_swmr=True)

            with pytest.raises(SchemaMismatchError, match="Column order mismatch"):
                save_frame_update(group, wrong_order_df, start=0, require_swmr=True)

    def test_frame_with_preallocated_layout(self, temp_hdf5_file):
        """Test using preallocated layout for DataFrame."""
        df = pd.DataFrame(
            {
                "col1": [1, 2, 3],
                "col2": ["a", "b", "c"],
            }
        )

        with h5py.File(temp_hdf5_file, "w", libver="latest") as f:
            f.swmr_mode = True
            group = f.create_group("frame")

            # Preallocate first
            preallocate_frame_layout(group, df, preallocate=100, require_swmr=True)

            # Then save (should reuse layout)
            save_frame_new(group, df, require_swmr=True)

            loaded = load_frame(group, require_swmr=True)

            expected = df.copy()
            expected["col1"] = expected["col1"].astype(np.float64)

            # Compare values and structure (index types will differ due to string storage)
            np.testing.assert_array_equal(loaded.values, expected.values)
            assert list(loaded.columns) == list(expected.columns)
            assert loaded.shape == expected.shape
            assert group.attrs["len"] == 3
