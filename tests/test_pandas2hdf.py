"""Tests for pandas2hdf functionality."""

import os
import random
import string
import tempfile
from pathlib import Path
from typing import Any

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
from pandas2hdf.core import H5DataError, H5SchemaError, H5SWMRError


def assert_series_equal_values(left: pd.Series, right: pd.Series) -> None:
    """Compare Series values and metadata, ignoring index differences."""
    assert left.name == right.name
    assert len(left) == len(right)
    np.testing.assert_array_equal(left.values, right.values)


def assert_frame_equal_values(left: pd.DataFrame, right: pd.DataFrame) -> None:
    """Compare DataFrame values and metadata, ignoring index differences."""
    assert list(left.columns) == list(right.columns)
    assert len(left) == len(right)
    for col in left.columns:
        np.testing.assert_array_equal(left[col].values, right[col].values)


# Test data generators
def random_unicode_string(length: int = 50) -> str:
    """Generate random unicode string for fuzzing."""
    # Mix ASCII, Latin-1, and some other unicode
    chars = (
        string.ascii_letters
        + string.digits
        + "àáäâèéëêìíïîòóöôùúüûñç"
        + "中文字符тестТЕСТ😀🎉"
    )
    return "".join(random.choice(chars) for _ in range(length))


class TestSWMRAssertions:
    """Test SWMR mode assertions."""

    def test_assert_swmr_on_raises_when_not_swmr(self, tmp_path):
        """Test that assert_swmr_on raises when file not in SWMR mode."""
        file_path = tmp_path / "test.h5"
        with h5py.File(file_path, "w") as f:
            g = f.create_group("test")
            with pytest.raises(H5SWMRError, match="not in SWMR mode"):
                assert_swmr_on(g)

    def test_assert_swmr_on_passes_when_swmr(self, tmp_path):
        """Test that assert_swmr_on passes when file in SWMR mode."""
        file_path = tmp_path / "test.h5"
        with h5py.File(file_path, "w", libver="latest") as f:
            g = f.create_group("test")
            f.swmr_mode = True  # Enable SWMR mode
            assert_swmr_on(g)  # Should not raise


class TestSeriesRoundTrip:
    """Test Series save/load functionality."""

    @pytest.mark.parametrize(
        "dtype,expected_dtype",
        [
            (np.int64, np.float64),
            (np.int32, np.float64),
            (np.float32, np.float64),
            (np.float64, np.float64),
            (bool, np.float64),
            ("boolean", np.float64),
            (str, object),
            ("string", object),
            ("category", object),
        ],
    )
    def test_series_dtype_roundtrip(self, tmp_path, dtype, expected_dtype):
        """Test round-trip for various dtypes."""
        file_path = tmp_path / "test.h5"

        # Create test data
        if dtype == "category":
            data = pd.Series(["a", "b", "c", "a", "b"], dtype="category")
        elif dtype == "boolean":
            data = pd.Series([True, False, None, True], dtype="boolean")
        elif dtype in (str, "string"):
            data = pd.Series(["hello", "world", None, "test"], dtype=dtype)
        else:
            data = pd.Series([1, 2, 3, 4, 5], dtype=dtype)

        data.name = "test_series"
        data.index.name = "test_index"

        # Save
        with h5py.File(file_path, "w") as f:
            g = f.create_group("series")
            save_series_new(g, data, require_swmr=False)

        # Load
        with h5py.File(file_path, "r") as f:
            loaded = load_series(f["series"])

        # Verify
        assert loaded.name == data.name
        assert loaded.index.name == data.index.name
        assert len(loaded) == len(data)
        assert loaded.dtype == expected_dtype

        # Check values (note: index will always be strings after round-trip)
        if expected_dtype == np.float64:
            if dtype in (bool, "boolean"):
                # Boolean: True=1.0, False=0.0
                expected = data.astype(float)
                np.testing.assert_array_equal(loaded.values, expected.values)
            else:
                np.testing.assert_array_equal(
                    loaded.values, data.astype(expected_dtype).values
                )
        else:
            # String types - compare values (handle NA vs None difference)
            if dtype == "string":
                # pandas StringDtype uses pd.NA, we convert to None
                expected_values = data.astype(object).values
                expected_values = np.array(
                    [v if pd.notna(v) else None for v in expected_values], dtype=object
                )
                np.testing.assert_array_equal(loaded.values, expected_values)
            else:
                np.testing.assert_array_equal(loaded.values, data.values)

    def test_series_with_missing_values(self, tmp_path):
        """Test handling of missing values."""
        file_path = tmp_path / "test.h5"

        # Numeric with NaN
        numeric_data = pd.Series([1.0, np.nan, 3.0, np.nan, 5.0], name="numeric")

        # String with None
        string_data = pd.Series(["a", None, "c", None, "e"], name="strings")

        # Save
        with h5py.File(file_path, "w") as f:
            save_series_new(f.create_group("numeric"), numeric_data, require_swmr=False)
            save_series_new(f.create_group("string"), string_data, require_swmr=False)

        # Load
        with h5py.File(file_path, "r") as f:
            loaded_numeric = load_series(f["numeric"])
            loaded_string = load_series(f["string"])

        # Verify
        assert_series_equal_values(loaded_numeric, numeric_data)
        assert_series_equal_values(loaded_string, string_data)

    def test_series_empty_raises(self, tmp_path):
        """Test that saving empty series raises error."""
        file_path = tmp_path / "test.h5"
        empty_series = pd.Series([], dtype=float)

        with h5py.File(file_path, "w") as f:
            g = f.create_group("series")
            with pytest.raises(H5DataError, match="empty series"):
                save_series_new(g, empty_series, require_swmr=False)

    def test_series_with_datetime_index(self, tmp_path):
        """Test series with datetime index."""
        file_path = tmp_path / "test.h5"

        # Create series with datetime index
        dates = pd.date_range("2023-01-01", periods=5, freq="D")
        data = pd.Series([1, 2, 3, 4, 5], index=dates, name="with_dates")
        data.index.name = "date"

        # Save
        with h5py.File(file_path, "w") as f:
            g = f.create_group("series")
            save_series_new(g, data, require_swmr=False)

        # Load
        with h5py.File(file_path, "r") as f:
            loaded = load_series(f["series"])

        # Verify
        assert loaded.name == data.name
        assert loaded.index.name == data.index.name
        assert len(loaded) == len(data)
        # Index should be strings now
        expected_index = dates.strftime("%Y-%m-%dT%H:%M:%S.%f")
        pd.testing.assert_index_equal(
            loaded.index, pd.Index(expected_index, name="date")
        )
        np.testing.assert_array_equal(loaded.values, data.values)


class TestSWMRWorkflow:
    """Test SWMR-specific workflows."""

    def test_preallocate_series_layout(self, tmp_path):
        """Test preallocation creates proper layout with no data."""
        file_path = tmp_path / "test.h5"
        test_series = pd.Series([1, 2, 3], index=["a", "b", "c"], name="test")

        with h5py.File(file_path, "w", libver="latest") as f:
            g = f.create_group("series")
            f.swmr_mode = True

            preallocate_series_layout(g, test_series, preallocate=100)

            # Check layout
            assert "values" in g
            assert "index" in g
            assert "index_mask" in g
            assert g["values"].shape == (100,)
            assert g["values"].maxshape == (None,)
            assert g["index"].shape == (100,)
            assert g.attrs["len"] == 0
            assert g.attrs["series_name"] == "test"
            assert g.attrs["values_kind"] == "numeric_float64"

    def test_save_update_contiguous(self, tmp_path):
        """Test series update with contiguous data."""
        file_path = tmp_path / "test.h5"

        # Initial data
        data1 = pd.Series([1, 2, 3], index=["a", "b", "c"])
        data2 = pd.Series([4, 5, 6], index=["d", "e", "f"])

        with h5py.File(file_path, "w", libver="latest") as f:
            g = f.create_group("series")
            f.swmr_mode = True

            # Save initial
            save_series_new(g, data1)
            assert g.attrs["len"] == 3

            # Update at end (contiguous)
            save_series_update(g, data2, start=3)
            assert g.attrs["len"] == 6

        # Verify combined data
        with h5py.File(file_path, "r") as f:
            loaded = load_series(f["series"])
            expected = pd.concat([data1, data2])
            assert_series_equal_values(loaded, expected)
            # Verify the index contains the expected string values
            expected_index = ["a", "b", "c", "d", "e", "f"]
            pd.testing.assert_index_equal(loaded.index, pd.Index(expected_index))

    def test_save_update_non_contiguous_raises(self, tmp_path):
        """Test that non-contiguous update raises error."""
        file_path = tmp_path / "test.h5"

        data1 = pd.Series([1, 2, 3])
        data2 = pd.Series([4, 5, 6])

        with h5py.File(file_path, "w", libver="latest") as f:
            g = f.create_group("series")
            f.swmr_mode = True

            save_series_new(g, data1)

            # Try non-contiguous update
            with pytest.raises(H5DataError, match="Non-contiguous update"):
                save_series_update(g, data2, start=5)  # Gap at positions 3,4

    def test_save_append(self, tmp_path):
        """Test series append functionality."""
        file_path = tmp_path / "test.h5"

        data1 = pd.Series([1, 2, 3], name="test")
        data2 = pd.Series([4, 5, 6], name="test")
        data3 = pd.Series([7, 8, 9], name="test")

        with h5py.File(file_path, "w", libver="latest") as f:
            g = f.create_group("series")
            f.swmr_mode = True

            # Initial save
            save_series_new(g, data1, preallocate=5)
            assert g.attrs["len"] == 3

            # First append
            save_series_append(g, data2)
            assert g.attrs["len"] == 6

            # Second append (should trigger resize)
            save_series_append(g, data3)
            assert g.attrs["len"] == 9

        # Verify
        with h5py.File(file_path, "r") as f:
            loaded = load_series(f["series"])
            expected = pd.concat([data1, data2, data3])
            assert_series_equal_values(loaded, expected)

    def test_schema_mismatch_error(self, tmp_path):
        """Test that schema mismatch raises error."""
        file_path = tmp_path / "test.h5"

        numeric_data = pd.Series([1, 2, 3])
        string_data = pd.Series(["a", "b", "c"])

        with h5py.File(file_path, "w", libver="latest") as f:
            g = f.create_group("series")
            f.swmr_mode = True

            save_series_new(g, numeric_data)

            # Try to update with different type
            with pytest.raises(H5SchemaError, match="different value type"):
                save_series_update(g, string_data, start=3)


class TestDataFrameRoundTrip:
    """Test DataFrame save/load functionality."""

    def test_basic_dataframe_roundtrip(self, tmp_path):
        """Test basic DataFrame round-trip."""
        file_path = tmp_path / "test.h5"

        df = pd.DataFrame(
            {
                "int_col": [1, 2, 3, 4, 5],
                "float_col": [1.1, 2.2, 3.3, 4.4, 5.5],
                "str_col": ["a", "b", "c", "d", "e"],
                "bool_col": [True, False, True, False, True],
            }
        )
        df.index = pd.Index(["r1", "r2", "r3", "r4", "r5"], name="row_id")

        # Save
        with h5py.File(file_path, "w") as f:
            g = f.create_group("frame")
            save_frame_new(g, df, require_swmr=False)

        # Load
        with h5py.File(file_path, "r") as f:
            loaded = load_frame(f["frame"])

        # Verify structure
        assert list(loaded.columns) == list(df.columns)
        assert loaded.index.name == df.index.name
        # Index values should match even though they're now strings
        pd.testing.assert_index_equal(
            loaded.index, pd.Index(["r1", "r2", "r3", "r4", "r5"], name="row_id")
        )

        # Verify data (with type conversions)
        assert loaded["int_col"].dtype == np.float64
        assert loaded["bool_col"].dtype == np.float64
        np.testing.assert_array_equal(
            loaded["int_col"].values, df["int_col"].astype(float).values
        )
        np.testing.assert_array_equal(
            loaded["float_col"].values, df["float_col"].values
        )
        np.testing.assert_array_equal(loaded["str_col"].values, df["str_col"].values)
        np.testing.assert_array_equal(
            loaded["bool_col"].values, df["bool_col"].astype(float).values
        )

    def test_dataframe_with_categorical(self, tmp_path):
        """Test DataFrame with categorical columns."""
        file_path = tmp_path / "test.h5"

        df = pd.DataFrame(
            {
                "cat_col": pd.Categorical(["a", "b", "c", "a", "b"]),
                "num_col": [1, 2, 3, 4, 5],
            }
        )

        # Save
        with h5py.File(file_path, "w") as f:
            g = f.create_group("frame")
            save_frame_new(g, df, require_swmr=False)

            # Check categories were saved
            cat_attrs = g["columns/cat_col"].attrs["categories"]
            assert cat_attrs == '["a", "b", "c"]'

        # Load
        with h5py.File(file_path, "r") as f:
            loaded = load_frame(f["frame"])

        # Categorical comes back as strings
        expected_cat = df["cat_col"].astype(str)
        np.testing.assert_array_equal(loaded["cat_col"].values, expected_cat.values)
        np.testing.assert_array_equal(
            loaded["num_col"].values, df["num_col"].astype(float).values
        )

    def test_dataframe_with_missing_values(self, tmp_path):
        """Test DataFrame with missing values."""
        file_path = tmp_path / "test.h5"

        df = pd.DataFrame(
            {
                "num_col": [1.0, np.nan, 3.0, np.nan, 5.0],
                "str_col": ["a", None, "c", None, "e"],
            }
        )

        # Save
        with h5py.File(file_path, "w") as f:
            g = f.create_group("frame")
            save_frame_new(g, df, require_swmr=False)

        # Load
        with h5py.File(file_path, "r") as f:
            loaded = load_frame(f["frame"])

        # Verify
        assert_frame_equal_values(loaded, df)

    def test_dataframe_column_order_preserved(self, tmp_path):
        """Test that column order is preserved."""
        file_path = tmp_path / "test.h5"

        # Create DataFrame with specific column order
        columns = ["z_col", "a_col", "m_col", "b_col"]
        df = pd.DataFrame({col: range(5) for col in columns}, columns=columns)

        # Save
        with h5py.File(file_path, "w") as f:
            g = f.create_group("frame")
            save_frame_new(g, df, require_swmr=False)

        # Load
        with h5py.File(file_path, "r") as f:
            loaded = load_frame(f["frame"])

        # Verify column order
        assert list(loaded.columns) == columns


class TestDataFrameSWMR:
    """Test DataFrame SWMR operations."""

    def test_frame_append(self, tmp_path):
        """Test DataFrame append functionality."""
        file_path = tmp_path / "test.h5"

        df1 = pd.DataFrame(
            {
                "a": [1, 2, 3],
                "b": ["x", "y", "z"],
            }
        )

        df2 = pd.DataFrame(
            {
                "a": [4, 5, 6],
                "b": ["p", "q", "r"],
            }
        )

        with h5py.File(file_path, "w", libver="latest") as f:
            g = f.create_group("frame")
            f.swmr_mode = True

            # Initial save
            save_frame_new(g, df1)

            # Append
            save_frame_append(g, df2)

        # Verify
        with h5py.File(file_path, "r") as f:
            loaded = load_frame(f["frame"])
            expected = pd.concat([df1, df2], ignore_index=True)
            expected["a"] = expected["a"].astype(float)  # int -> float conversion
            assert_frame_equal_values(loaded, expected)

    def test_frame_update_column_mismatch(self, tmp_path):
        """Test that column mismatch raises error."""
        file_path = tmp_path / "test.h5"

        df1 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        df2 = pd.DataFrame({"a": [7, 8, 9], "c": [10, 11, 12]})  # Different columns

        with h5py.File(file_path, "w", libver="latest") as f:
            g = f.create_group("frame")
            f.swmr_mode = True

            save_frame_new(g, df1)

            with pytest.raises(H5DataError, match="Column mismatch"):
                save_frame_update(g, df2, start=3)


class TestFuzzingAndEdgeCases:
    """Test edge cases and fuzzing."""

    def test_random_unicode_strings(self, tmp_path):
        """Test with random unicode strings."""
        file_path = tmp_path / "test.h5"

        # Generate random unicode data
        data = [random_unicode_string(random.randint(10, 100)) for _ in range(20)]
        series = pd.Series(data, name="unicode_test")

        # Save and load
        with h5py.File(file_path, "w") as f:
            g = f.create_group("series")
            save_series_new(g, series, require_swmr=False)

        with h5py.File(file_path, "r") as f:
            loaded = load_series(f["series"])

        assert_series_equal_values(loaded, series)

    def test_very_long_strings(self, tmp_path):
        """Test with very long strings."""
        file_path = tmp_path / "test.h5"

        # Create series with very long strings
        long_string = "x" * 10000
        series = pd.Series(
            [long_string, "short", long_string + "y"], name="long_strings"
        )

        # Save and load
        with h5py.File(file_path, "w") as f:
            g = f.create_group("series")
            save_series_new(g, series, require_swmr=False)

        with h5py.File(file_path, "r") as f:
            loaded = load_series(f["series"])

        assert_series_equal_values(loaded, series)

    def test_duplicate_index_values(self, tmp_path):
        """Test with duplicate index values."""
        file_path = tmp_path / "test.h5"

        # Series with duplicate index
        series = pd.Series([1, 2, 3, 4, 5], index=["a", "b", "a", "b", "c"])

        # Save and load
        with h5py.File(file_path, "w") as f:
            g = f.create_group("series")
            save_series_new(g, series, require_swmr=False)

        with h5py.File(file_path, "r") as f:
            loaded = load_series(f["series"])

        assert_series_equal_values(loaded, series.astype(float))
        # Verify index preserved duplicate values
        pd.testing.assert_index_equal(loaded.index, pd.Index(["a", "b", "a", "b", "c"]))

    @pytest.mark.parametrize("compression", [None, "gzip", "lzf"])
    def test_different_compression(self, tmp_path, compression):
        """Test with different compression settings."""
        if compression == "lzf" and not h5py.h5z.filter_avail(h5py.h5z.FILTER_LZF):
            pytest.skip("LZF filter not available")

        file_path = tmp_path / "test.h5"
        series = pd.Series(range(100), name="test")

        # Save with specified compression
        with h5py.File(file_path, "w") as f:
            g = f.create_group("series")
            save_series_new(g, series, compression=compression, require_swmr=False)

        # Load and verify
        with h5py.File(file_path, "r") as f:
            loaded = load_series(f["series"])

        assert_series_equal_values(loaded, series.astype(float))


class TestSWMRConcurrency:
    """Test SWMR concurrent read/write scenarios."""

    def test_swmr_reader_sees_appends(self, tmp_path):
        """Test that SWMR reader can see appended data after flush."""
        file_path = tmp_path / "test.h5"

        # Initial data
        data1 = pd.Series([1, 2, 3], name="test")
        data2 = pd.Series([4, 5, 6], name="test")

        # Create file and write initial data with preallocation
        with h5py.File(file_path, "w", libver="latest") as f:
            g = f.create_group("series")
            f.swmr_mode = True
            save_series_new(g, data1, preallocate=10)

        # Append more data
        with h5py.File(file_path, "r+", libver="latest") as f:
            f.swmr_mode = True
            save_series_append(f["series"], data2, require_swmr=True)

        # Now open as reader and verify all data is present
        with h5py.File(file_path, "r", swmr=True) as f:
            loaded = load_series(f["series"], require_swmr=True)
            assert len(loaded) == 6

            expected = pd.concat([data1, data2])
            assert_series_equal_values(loaded, expected)


class TestMultiIndex:
    """Test MultiIndex functionality for both row and column indexes."""

    def test_series_with_multiindex(self, tmp_path):
        """Test Series with MultiIndex."""
        file_path = tmp_path / "test.h5"

        # Create MultiIndex
        arrays = [
            ["A", "A", "B", "B"],
            ["one", "two", "one", "two"],
        ]
        index = pd.MultiIndex.from_arrays(arrays, names=["first", "second"])
        
        # Create series with MultiIndex
        series = pd.Series([1, 2, 3, 4], index=index, name="values")

        # Save
        with h5py.File(file_path, "w") as f:
            g = f.create_group("series")
            save_series_new(g, series, require_swmr=False)

        # Load
        with h5py.File(file_path, "r") as f:
            loaded = load_series(f["series"])

        # Verify
        assert loaded.name == series.name
        assert isinstance(loaded.index, pd.MultiIndex)
        assert loaded.index.nlevels == 2
        assert loaded.index.names == ["first", "second"]
        np.testing.assert_array_equal(loaded.values, series.values)
        
        # Verify index structure
        pd.testing.assert_index_equal(loaded.index, series.index)

    def test_dataframe_with_multiindex_rows(self, tmp_path):
        """Test DataFrame with MultiIndex rows."""
        file_path = tmp_path / "test.h5"

        # Create MultiIndex for rows
        arrays = [
            ["A", "A", "B", "B"],
            ["one", "two", "one", "two"],
        ]
        index = pd.MultiIndex.from_arrays(arrays, names=["first", "second"])
        
        df = pd.DataFrame(
            {
                "col1": [1, 2, 3, 4],
                "col2": [10.1, 20.2, 30.3, 40.4],
            },
            index=index,
        )

        # Save
        with h5py.File(file_path, "w") as f:
            g = f.create_group("frame")
            save_frame_new(g, df, require_swmr=False)

        # Load
        with h5py.File(file_path, "r") as f:
            loaded = load_frame(f["frame"])

        # Verify
        assert isinstance(loaded.index, pd.MultiIndex)
        assert loaded.index.nlevels == 2
        assert loaded.index.names == ["first", "second"]
        pd.testing.assert_index_equal(loaded.index, df.index)
        assert list(loaded.columns) == list(df.columns)
        np.testing.assert_array_equal(loaded["col1"].values, df["col1"].astype(float).values)
        np.testing.assert_array_equal(loaded["col2"].values, df["col2"].values)

    def test_dataframe_with_multiindex_columns(self, tmp_path):
        """Test DataFrame with MultiIndex columns."""
        file_path = tmp_path / "test.h5"

        # Create MultiIndex for columns
        arrays = [
            ["A", "A", "B", "B"],
            ["one", "two", "one", "two"],
        ]
        columns = pd.MultiIndex.from_arrays(arrays, names=["level1", "level2"])
        
        df = pd.DataFrame(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
            columns=columns,
        )

        # Save
        with h5py.File(file_path, "w") as f:
            g = f.create_group("frame")
            save_frame_new(g, df, require_swmr=False)

        # Load
        with h5py.File(file_path, "r") as f:
            loaded = load_frame(f["frame"])

        # Verify
        assert isinstance(loaded.columns, pd.MultiIndex)
        assert loaded.columns.nlevels == 2
        assert loaded.columns.names == ["level1", "level2"]
        pd.testing.assert_index_equal(loaded.columns, df.columns)
        assert len(loaded) == len(df)
        
        # Verify data for each column
        for col in df.columns:
            np.testing.assert_array_equal(
                loaded[col].values, df[col].astype(float).values
            )

    def test_dataframe_with_both_multiindex(self, tmp_path):
        """Test DataFrame with MultiIndex for both rows and columns."""
        file_path = tmp_path / "test.h5"

        # Create MultiIndex for rows
        row_arrays = [
            ["X", "X", "Y", "Y"],
            ["alpha", "beta", "alpha", "beta"],
        ]
        row_index = pd.MultiIndex.from_arrays(row_arrays, names=["group", "subgroup"])
        
        # Create MultiIndex for columns
        col_arrays = [
            ["A", "A", "B"],
            ["one", "two", "one"],
        ]
        col_index = pd.MultiIndex.from_arrays(col_arrays, names=["level1", "level2"])
        
        df = pd.DataFrame(
            [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
            index=row_index,
            columns=col_index,
        )

        # Save
        with h5py.File(file_path, "w") as f:
            g = f.create_group("frame")
            save_frame_new(g, df, require_swmr=False)

        # Load
        with h5py.File(file_path, "r") as f:
            loaded = load_frame(f["frame"])

        # Verify structure
        assert isinstance(loaded.index, pd.MultiIndex)
        assert isinstance(loaded.columns, pd.MultiIndex)
        assert loaded.index.nlevels == 2
        assert loaded.columns.nlevels == 2
        assert loaded.index.names == ["group", "subgroup"]
        assert loaded.columns.names == ["level1", "level2"]
        
        # Verify indexes
        pd.testing.assert_index_equal(loaded.index, df.index)
        pd.testing.assert_index_equal(loaded.columns, df.columns)
        
        # Verify data
        for col in df.columns:
            np.testing.assert_array_equal(
                loaded[col].values, df[col].astype(float).values
            )

    def test_multiindex_with_missing_values(self, tmp_path):
        """Test MultiIndex with missing values in levels."""
        file_path = tmp_path / "test.h5"

        # Create MultiIndex with some None values
        arrays = [
            ["A", None, "B", "B"],
            ["one", "two", None, "two"],
        ]
        index = pd.MultiIndex.from_arrays(arrays, names=["first", "second"])
        
        series = pd.Series([1, 2, 3, 4], index=index, name="test")

        # Save
        with h5py.File(file_path, "w") as f:
            g = f.create_group("series")
            save_series_new(g, series, require_swmr=False)

        # Load
        with h5py.File(file_path, "r") as f:
            loaded = load_series(f["series"])

        # Verify
        assert isinstance(loaded.index, pd.MultiIndex)
        pd.testing.assert_index_equal(loaded.index, series.index)
        np.testing.assert_array_equal(loaded.values, series.values)

    def test_multiindex_append_update(self, tmp_path):
        """Test MultiIndex with append and update operations."""
        file_path = tmp_path / "test.h5"

        # Initial data
        arrays1 = [
            ["A", "A"],
            ["one", "two"],
        ]
        index1 = pd.MultiIndex.from_arrays(arrays1, names=["first", "second"])
        series1 = pd.Series([1, 2], index=index1, name="test")

        # Additional data for append
        arrays2 = [
            ["B", "B"],
            ["one", "two"],
        ]
        index2 = pd.MultiIndex.from_arrays(arrays2, names=["first", "second"])
        series2 = pd.Series([3, 4], index=index2, name="test")

        with h5py.File(file_path, "w", libver="latest") as f:
            g = f.create_group("series")
            f.swmr_mode = True
            
            # Save initial
            save_series_new(g, series1)
            
            # Append
            save_series_append(g, series2)

        # Verify
        with h5py.File(file_path, "r") as f:
            loaded = load_series(f["series"])
            expected = pd.concat([series1, series2])
            
            assert isinstance(loaded.index, pd.MultiIndex)
            assert len(loaded) == 4
            np.testing.assert_array_equal(loaded.values, expected.values)
            pd.testing.assert_index_equal(loaded.index, expected.index)

    def test_multiindex_datetime_levels(self, tmp_path):
        """Test MultiIndex with datetime levels."""
        file_path = tmp_path / "test.h5"

        # Create MultiIndex with datetime
        dates = pd.date_range("2023-01-01", periods=2, freq="D")
        arrays = [
            dates,
            ["morning", "evening"],
        ]
        index = pd.MultiIndex.from_arrays(arrays, names=["date", "time"])
        
        series = pd.Series([100, 200], index=index, name="temperature")

        # Save
        with h5py.File(file_path, "w") as f:
            g = f.create_group("series")
            save_series_new(g, series, require_swmr=False)

        # Load
        with h5py.File(file_path, "r") as f:
            loaded = load_series(f["series"])

        # Verify
        assert isinstance(loaded.index, pd.MultiIndex)
        assert loaded.index.nlevels == 2
        assert loaded.index.names == ["date", "time"]
        
        # Datetime level should be converted to strings
        expected_dates = dates.strftime("%Y-%m-%dT%H:%M:%S.%f")
        assert all(loaded.index.get_level_values(0) == expected_dates)
        assert all(loaded.index.get_level_values(1) == ["morning", "evening"])


class TestErrorHandling:
    """Test error conditions and edge cases."""

    def test_unsupported_dtype(self, tmp_path):
        """Test that unsupported dtypes raise appropriate errors."""
        file_path = tmp_path / "test.h5"

        # Complex numbers are not supported
        series = pd.Series([1 + 2j, 3 + 4j, 5 + 6j])

        with h5py.File(file_path, "w") as f:
            g = f.create_group("series")
            with pytest.raises(H5DataError, match="Unsupported dtype"):
                save_series_new(g, series, require_swmr=False)

    def test_missing_required_datasets(self, tmp_path):
        """Test loading from incomplete group raises error."""
        file_path = tmp_path / "test.h5"

        # Create incomplete group
        with h5py.File(file_path, "w") as f:
            g = f.create_group("incomplete")
            g.attrs["len"] = 5
            # Missing values dataset

        with h5py.File(file_path, "r") as f:
            with pytest.raises(H5DataError, match="Missing 'values_kind' attribute"):
                load_series(f["incomplete"])

    def test_missing_len_attribute(self, tmp_path):
        """Test loading without len attribute raises error."""
        file_path = tmp_path / "test.h5"

        with h5py.File(file_path, "w") as f:
            g = f.create_group("no_len")
            g.create_dataset("values", data=[1, 2, 3])
            g.create_dataset("index", data=["a", "b", "c"])
            # Missing len attribute

        with h5py.File(file_path, "r") as f:
            with pytest.raises(H5DataError, match="Missing 'len' attribute"):
                load_series(f["no_len"])
