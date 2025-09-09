"""Core functionality for pandas-to-HDF5 persistence with SWMR support."""

import json
from datetime import datetime
from typing import Any

import h5py
import numpy as np
import pandas as pd


class Pandas2HDFError(Exception):
    """Base exception for pandas2hdf operations."""


class SWMRModeError(Pandas2HDFError):
    """Raised when SWMR mode is required but not enabled."""


class SchemaMismatchError(Pandas2HDFError):
    """Raised when trying to write data that doesn't match existing schema."""


class ValidationError(Pandas2HDFError):
    """Raised when data validation fails."""


def assert_swmr_on(g: h5py.Group) -> None:
    """Assert that SWMR mode is enabled on the group's file.

    Args:
        g: HDF5 group to check.

    Raises:
        SWMRModeError: If SWMR mode is not enabled.
    """
    if not g.file.swmr_mode:
        raise SWMRModeError(
            f"SWMR mode is required but not enabled on file {g.file.filename}"
        )


def _get_string_dtype() -> h5py.special_dtype:
    """Get UTF-8 variable-length string dtype for HDF5."""
    return h5py.string_dtype("utf-8")


def _encode_values_for_hdf5(
    values: pd.Series,
) -> tuple[
    np.ndarray[Any, np.dtype[Any]], np.ndarray[Any, np.dtype[Any]] | None, str, str
]:
    """Encode pandas Series values for HDF5 storage.

    Args:
        values: pandas Series to encode.

    Returns:
        Tuple of (encoded_values, mask, values_kind, orig_dtype):
        - encoded_values: numpy array ready for HDF5 storage
        - mask: optional mask array (uint8, 1=valid, 0=missing) for strings
        - values_kind: "numeric_float64" or "string_utf8_vlen"
        - orig_dtype: string representation of original dtype
    """
    orig_dtype = str(values.dtype)

    if pd.api.types.is_numeric_dtype(values.dtype) or pd.api.types.is_bool_dtype(
        values.dtype
    ):
        # Convert to float64, with NaN for missing values
        encoded = values.astype(np.float64).values
        mask = None
        values_kind = "numeric_float64"
    elif pd.api.types.is_object_dtype(values.dtype):
        # Check if object dtype contains boolean-like values
        non_null_values = values.dropna()
        if len(non_null_values) > 0 and all(
            isinstance(v, bool | np.bool_) or v in (True, False)
            for v in non_null_values
        ):
            # Treat as boolean/numeric data
            encoded = values.astype(np.float64).values
            mask = None
            values_kind = "numeric_float64"
        else:
            # Treat as string data
            str_values = values.astype(str)
            mask = (~values.isna()).astype(np.uint8)
            # Replace NaN string representations with empty strings
            str_array = str_values.values
            str_array[np.asarray(values.isna())] = ""
            encoded = str_array.astype(_get_string_dtype())
            values_kind = "string_utf8_vlen"
    elif pd.api.types.is_string_dtype(values.dtype):
        # Convert to UTF-8 strings with mask for missing values
        str_values = values.astype(str)
        mask = (~values.isna()).astype(np.uint8)
        # Replace NaN string representations with empty strings
        str_array = str_values.values
        str_array[np.asarray(values.isna())] = ""
        encoded = str_array.astype(_get_string_dtype())
        values_kind = "string_utf8_vlen"
    else:
        raise ValidationError(f"Unsupported dtype: {values.dtype}")

    # Ensure we return numpy arrays, not ExtensionArrays
    encoded_array: np.ndarray[Any, np.dtype[Any]] = np.asarray(encoded)
    mask_array: np.ndarray[Any, np.dtype[Any]] | None = (
        np.asarray(mask) if mask is not None else None
    )
    return encoded_array, mask_array, values_kind, orig_dtype


def _encode_index_for_hdf5(
    index: pd.Index,
) -> tuple[
    np.ndarray[Any, np.dtype[Any]] | list[np.ndarray[Any, np.dtype[Any]]],
    np.ndarray[Any, np.dtype[Any]] | list[np.ndarray[Any, np.dtype[Any]]],
    dict[str, Any],
    str,
]:
    """Encode pandas Index/MultiIndex for HDF5 storage.

    Args:
        index: pandas Index or MultiIndex to encode.

    Returns:
        Tuple of (encoded_arrays, mask_arrays, metadata, orig_dtype):
        - encoded_arrays: array or list of arrays for MultiIndex levels
        - mask_arrays: mask array or list of mask arrays
        - metadata: dict with index metadata
        - orig_dtype: string representation of original dtype
    """
    orig_dtype = (
        str(index.dtype) if not isinstance(index, pd.MultiIndex) else "MultiIndex"
    )

    if isinstance(index, pd.MultiIndex):
        # Handle MultiIndex
        encoded_arrays = []
        mask_arrays = []

        for level_values in index.to_frame().values.T:
            level_series = pd.Series(level_values)
            str_values = level_series.astype(str)
            mask = (~level_series.isna()).astype(np.uint8)
            # Replace NaN representations
            str_array = str_values.values
            str_array[np.asarray(level_series.isna())] = ""
            encoded_arrays.append(str_array.astype(_get_string_dtype()))
            mask_arrays.append(np.asarray(mask))

        metadata = {
            "index_is_multiindex": 1,
            "index_levels": index.nlevels,
            "index_names": json.dumps(
                [str(name) if name is not None else None for name in index.names]
            ),
        }
    else:
        # Handle regular Index
        index_series = pd.Series(index)
        str_values = index_series.astype(str)
        mask = (~index_series.isna()).astype(np.uint8)
        # Replace NaN representations
        str_array = str_values.values
        str_array[np.asarray(index_series.isna())] = ""
        encoded_arrays = str_array.astype(_get_string_dtype())  # type: ignore[assignment]
        mask_arrays = mask  # type: ignore[assignment]

        metadata = {
            "index_is_multiindex": 0,
            "index_levels": 1,
            "index_names": json.dumps(
                [str(index.name) if index.name is not None else None]
            ),
        }

    return encoded_arrays, mask_arrays, metadata, orig_dtype


def _decode_values_from_hdf5(
    group: h5py.Group,
    dataset_name: str = "values",
    length: int | None = None,
) -> tuple[np.ndarray[Any, np.dtype[Any]], str]:
    """Decode values from HDF5 storage back to numpy array.

    Args:
        group: HDF5 group containing the datasets.
        dataset_name: Name of the values dataset.
        length: Logical length to read (respects preallocated space).

    Returns:
        Tuple of (decoded_values, values_kind).
    """
    values_kind = group.attrs["values_kind"]
    if isinstance(values_kind, bytes):
        values_kind = values_kind.decode("utf-8")
    logical_length = length if length is not None else group.attrs["len"]

    if values_kind == "numeric_float64":
        values = group[dataset_name][:logical_length]
    elif values_kind == "string_utf8_vlen":
        values = group[dataset_name][:logical_length]
        mask = group[f"{dataset_name}_mask"][:logical_length]
        # Convert back to object array with proper NaN handling
        result = np.empty(logical_length, dtype=object)
        result[mask == 1] = [
            s.decode("utf-8") if isinstance(s, bytes) else str(s)
            for s in values[mask == 1]
        ]
        result[mask == 0] = None
        values = result
    else:
        raise ValidationError(f"Unknown values_kind: {values_kind}")

    return values, values_kind


def _decode_index_from_hdf5(
    group: h5py.Group,
    index_dataset_name: str = "index",
    length: int | None = None,
) -> pd.Index:
    """Decode index from HDF5 storage back to pandas Index/MultiIndex.

    Args:
        group: HDF5 group containing the index datasets.
        index_dataset_name: Base name for index datasets.
        length: Logical length to read.

    Returns:
        Reconstructed pandas Index or MultiIndex.
    """
    logical_length = length if length is not None else group.attrs["len"]
    is_multiindex = bool(group.attrs["index_is_multiindex"])
    index_names_attr = group.attrs["index_names"]
    if isinstance(index_names_attr, bytes):
        index_names_attr = index_names_attr.decode("utf-8")
    index_names = json.loads(index_names_attr)

    if is_multiindex:
        levels = []
        for i in range(group.attrs["index_levels"]):
            level_data = group[f"{index_dataset_name}/levels/L{i}"][:logical_length]
            level_mask = group[f"{index_dataset_name}/levels/L{i}_mask"][
                :logical_length
            ]

            # Reconstruct level with proper NaN handling
            level_values = np.empty(logical_length, dtype=object)
            level_values[level_mask == 1] = [
                s.decode("utf-8") if isinstance(s, bytes) else str(s)
                for s in level_data[level_mask == 1]
            ]
            level_values[level_mask == 0] = None
            levels.append(level_values)

        return pd.MultiIndex.from_arrays(levels, names=index_names)
    else:
        index_data = group[f"{index_dataset_name}/values"][:logical_length]
        index_mask = group[f"{index_dataset_name}/index_mask"][:logical_length]

        # Reconstruct index with proper NaN handling
        index_values = np.empty(logical_length, dtype=object)
        index_values[index_mask == 1] = [
            s.decode("utf-8") if isinstance(s, bytes) else str(s)
            for s in index_data[index_mask == 1]
        ]
        index_values[index_mask == 0] = None

        return pd.Index(index_values, name=index_names[0])


def _create_resizable_dataset(
    group: h5py.Group,
    name: str,
    dtype: Any,
    shape: tuple[int, ...],
    maxshape: tuple[int | None, ...],
    chunks: tuple[int, ...],
    compression: str,
) -> h5py.Dataset:
    """Create a resizable, chunked, compressed dataset."""
    return group.create_dataset(
        name,
        shape=shape,
        maxshape=maxshape,
        dtype=dtype,
        chunks=chunks,
        compression=compression,
    )


def preallocate_series_layout(
    group: h5py.Group,
    series: pd.Series,
    *,
    dataset: str = "values",
    index_dataset: str = "index",
    chunks: tuple[int, ...] = (25,),
    compression: str = "gzip",
    preallocate: int = 100,
    require_swmr: bool = True,
) -> None:
    """Preallocate HDF5 layout for a pandas Series without writing data.

    Creates resizable, chunked, compressed datasets with initial shape (preallocate,)
    and maxshape (None,). Initializes masks to zeros and sets len=0.

    Args:
        group: HDF5 group to write to.
        series: pandas Series to create layout for (used for schema).
        dataset: Name for the values dataset.
        index_dataset: Name for the index dataset.
        chunks: Chunk shape for datasets.
        compression: Compression algorithm.
        preallocate: Initial allocation size.
        require_swmr: If True, assert SWMR mode is enabled.

    Raises:
        SWMRModeError: If require_swmr=True and SWMR mode not enabled.
        ValidationError: If series validation fails.
    """
    if require_swmr:
        assert_swmr_on(group)

    # Encode series for schema information
    encoded_values, values_mask, values_kind, orig_values_dtype = (
        _encode_values_for_hdf5(series)
    )
    encoded_index, index_masks, index_metadata, orig_index_dtype = (
        _encode_index_for_hdf5(series.index)
    )

    # Create values dataset
    if values_kind == "numeric_float64":
        _create_resizable_dataset(
            group, dataset, np.float64, (preallocate,), (None,), chunks, compression
        )
    else:  # string_utf8_vlen
        _create_resizable_dataset(
            group,
            dataset,
            _get_string_dtype(),
            (preallocate,),
            (None,),
            chunks,
            compression,
        )
        # Create values mask
        mask_dataset = _create_resizable_dataset(
            group,
            f"{dataset}_mask",
            np.uint8,
            (preallocate,),
            (None,),
            chunks,
            compression,
        )
        mask_dataset[:] = 0  # Initialize to all missing

    # Create index datasets
    if index_metadata["index_is_multiindex"]:
        # Create index group and level datasets
        index_group = group.create_group(index_dataset)
        levels_group = index_group.create_group("levels")

        for i in range(index_metadata["index_levels"]):
            _create_resizable_dataset(
                levels_group,
                f"L{i}",
                _get_string_dtype(),
                (preallocate,),
                (None,),
                chunks,
                compression,
            )
            mask_dataset = _create_resizable_dataset(
                levels_group,
                f"L{i}_mask",
                np.uint8,
                (preallocate,),
                (None,),
                chunks,
                compression,
            )
            mask_dataset[:] = 0  # Initialize to all missing
    else:
        # Create index group
        index_group = group.create_group(index_dataset)
        _create_resizable_dataset(
            index_group,
            "values",
            _get_string_dtype(),
            (preallocate,),
            (None,),
            chunks,
            compression,
        )
        mask_dataset = _create_resizable_dataset(
            index_group,
            "index_mask",
            np.uint8,
            (preallocate,),
            (None,),
            chunks,
            compression,
        )
        mask_dataset[:] = 0  # Initialize to all missing

    # Set attributes
    group.attrs["series_name"] = str(series.name) if series.name is not None else ""
    group.attrs["len"] = 0  # Logical length is 0
    group.attrs["values_kind"] = values_kind
    group.attrs["index_kind"] = "string_utf8_vlen"
    group.attrs["orig_values_dtype"] = orig_values_dtype
    group.attrs["orig_index_dtype"] = orig_index_dtype
    group.attrs["created_at_iso"] = datetime.now().isoformat()
    group.attrs["version"] = "1.0"

    # Add index metadata
    for key, value in index_metadata.items():
        if isinstance(value, str):
            group.attrs[key] = value
        else:
            group.attrs[key] = value

    if require_swmr:
        group.file.flush()


def save_series_new(
    group: h5py.Group,
    series: pd.Series,
    *,
    dataset: str = "values",
    index_dataset: str = "index",
    chunks: tuple[int, ...] = (25,),
    compression: str = "gzip",
    preallocate: int = 100,
    require_swmr: bool = True,
) -> None:
    """Create datasets and write a pandas Series to HDF5.

    Creates new datasets or reuses existing preallocated layout.
    Writes the first len(series) elements and sets logical length.

    Args:
        group: HDF5 group to write to.
        series: pandas Series to persist.
        dataset: Name for the values dataset.
        index_dataset: Name for the index dataset.
        chunks: Chunk shape for new datasets.
        compression: Compression algorithm for new datasets.
        preallocate: Initial allocation size for new datasets.
        require_swmr: If True, assert SWMR mode is enabled.

    Raises:
        SWMRModeError: If require_swmr=True and SWMR mode not enabled.
        ValidationError: If series validation fails.
    """
    if require_swmr:
        assert_swmr_on(group)

    if len(series) == 0:
        raise ValidationError("Cannot save empty series")

    # Check if layout already exists (preallocated)
    if dataset in group and group.attrs.get("len", -1) == 0:
        # Use existing preallocated layout
        save_series_update(
            group,
            series,
            start=0,
            dataset=dataset,
            index_dataset=index_dataset,
            require_swmr=require_swmr,
        )
        return

    # Create new layout
    preallocate_series_layout(
        group,
        series,
        dataset=dataset,
        index_dataset=index_dataset,
        chunks=chunks,
        compression=compression,
        preallocate=max(preallocate, len(series)),
        require_swmr=require_swmr,
    )

    # Write the data
    save_series_update(
        group,
        series,
        start=0,
        dataset=dataset,
        index_dataset=index_dataset,
        require_swmr=require_swmr,
    )


def save_series_update(
    group: h5py.Group,
    series: pd.Series,
    *,
    start: int = 0,
    dataset: str = "values",
    index_dataset: str = "index",
    require_swmr: bool = True,
) -> None:
    """Update a pandas Series in HDF5 at specified position.

    Overwrites [start:start+len(series)] and updates logical length
    to the largest contiguous written extent.

    Args:
        group: HDF5 group containing existing datasets.
        series: pandas Series to write.
        start: Starting position for the update.
        dataset: Name of the values dataset.
        index_dataset: Name of the index dataset.
        require_swmr: If True, assert SWMR mode is enabled.

    Raises:
        SWMRModeError: If require_swmr=True and SWMR mode not enabled.
        ValidationError: If validation fails or schema mismatch.
    """
    if require_swmr:
        assert_swmr_on(group)

    if len(series) == 0:
        raise ValidationError("Cannot update with empty series")

    current_len = group.attrs["len"]
    end_pos = start + len(series)

    # Validate that this is a contiguous update
    if start > current_len:
        raise ValidationError(
            f"Non-contiguous update: start={start}, current_len={current_len}"
        )

    # Encode data
    encoded_values, values_mask, values_kind, _ = _encode_values_for_hdf5(series)
    encoded_index, index_masks, index_metadata, _ = _encode_index_for_hdf5(series.index)

    # Validate schema compatibility
    stored_values_kind = group.attrs["values_kind"]
    if isinstance(stored_values_kind, bytes):
        stored_values_kind = stored_values_kind.decode("utf-8")
    if stored_values_kind != values_kind:
        raise SchemaMismatchError(
            f"Values kind mismatch: expected {stored_values_kind}, got {values_kind}"
        )

    expected_multiindex = bool(group.attrs["index_is_multiindex"])
    actual_multiindex = bool(index_metadata["index_is_multiindex"])
    if expected_multiindex != actual_multiindex:
        raise SchemaMismatchError(
            f"Index type mismatch: expected multiindex={expected_multiindex}, got {actual_multiindex}"
        )

    # Resize datasets if needed
    values_dataset = group[dataset]
    if end_pos > values_dataset.shape[0]:
        values_dataset.resize((end_pos,))
        if values_mask is not None:
            group[f"{dataset}_mask"].resize((end_pos,))

    # Write values
    values_dataset[start:end_pos] = encoded_values
    if values_mask is not None:
        group[f"{dataset}_mask"][start:end_pos] = values_mask

    # Write index
    if expected_multiindex:
        levels_group = group[f"{index_dataset}/levels"]
        for i, (level_data, level_mask) in enumerate(
            zip(encoded_index, index_masks, strict=False)
        ):
            level_dataset = levels_group[f"L{i}"]
            if end_pos > level_dataset.shape[0]:
                level_dataset.resize((end_pos,))
                levels_group[f"L{i}_mask"].resize((end_pos,))
            level_dataset[start:end_pos] = level_data
            levels_group[f"L{i}_mask"][start:end_pos] = level_mask
    else:
        index_group = group[index_dataset]
        index_values_dataset = index_group["values"]
        if end_pos > index_values_dataset.shape[0]:
            index_values_dataset.resize((end_pos,))
            index_group["index_mask"].resize((end_pos,))
        index_values_dataset[start:end_pos] = encoded_index
        index_group["index_mask"][start:end_pos] = index_masks

    # Update logical length
    group.attrs["len"] = max(current_len, end_pos)

    if require_swmr:
        group.file.flush()


def save_series_append(
    group: h5py.Group,
    series: pd.Series,
    *,
    dataset: str = "values",
    index_dataset: str = "index",
    require_swmr: bool = True,
) -> None:
    """Append a pandas Series to existing HDF5 datasets.

    Appends at the end using current logical length.
    Resizes datasets if needed and updates logical length.

    Args:
        group: HDF5 group containing existing datasets.
        series: pandas Series to append.
        dataset: Name of the values dataset.
        index_dataset: Name of the index dataset.
        require_swmr: If True, assert SWMR mode is enabled.

    Raises:
        SWMRModeError: If require_swmr=True and SWMR mode not enabled.
        ValidationError: If validation fails or schema mismatch.
    """
    if require_swmr:
        assert_swmr_on(group)

    current_len = group.attrs["len"]
    save_series_update(
        group,
        series,
        start=current_len,
        dataset=dataset,
        index_dataset=index_dataset,
        require_swmr=require_swmr,
    )


def load_series(
    group: h5py.Group,
    *,
    dataset: str = "values",
    index_dataset: str = "index",
    require_swmr: bool = False,
) -> pd.Series:
    """Load a pandas Series from HDF5 storage.

    Reconstructs the Series with original name, index names, order,
    and missingness. Respects logical length from attributes.

    Args:
        group: HDF5 group containing the Series data.
        dataset: Name of the values dataset.
        index_dataset: Name of the index dataset.
        require_swmr: If True, assert SWMR mode is enabled.

    Returns:
        Reconstructed pandas Series.

    Raises:
        SWMRModeError: If require_swmr=True and SWMR mode not enabled.
        ValidationError: If data validation fails.
    """
    if require_swmr:
        assert_swmr_on(group)

    logical_length = group.attrs["len"]
    if logical_length == 0:
        # Return empty series with proper schema
        series_name = group.attrs["series_name"]
        if isinstance(series_name, bytes):
            series_name = series_name.decode("utf-8")
        series_name = series_name if series_name else None
        return pd.Series([], name=series_name, dtype=object)

    # Decode values and index
    values, _ = _decode_values_from_hdf5(group, dataset, logical_length)
    index = _decode_index_from_hdf5(group, index_dataset, logical_length)

    # Get series name
    series_name = group.attrs["series_name"]
    if isinstance(series_name, bytes):
        series_name = series_name.decode("utf-8")
    series_name = series_name if series_name else None

    return pd.Series(values, index=index, name=series_name)


# DataFrame functions - wrappers around Series functionality


def preallocate_frame_layout(
    group: h5py.Group,
    dataframe: pd.DataFrame,
    *,
    chunks: tuple[int, ...] = (25,),
    compression: str = "gzip",
    preallocate: int = 100,
    require_swmr: bool = True,
) -> None:
    """Preallocate HDF5 layout for a pandas DataFrame without writing data.

    Creates layout for shared index and column series using Series preallocation.

    Args:
        group: HDF5 group to write to.
        dataframe: pandas DataFrame to create layout for.
        chunks: Chunk shape for datasets.
        compression: Compression algorithm.
        preallocate: Initial allocation size.
        require_swmr: If True, assert SWMR mode is enabled.

    Raises:
        SWMRModeError: If require_swmr=True and SWMR mode not enabled.
        ValidationError: If DataFrame validation fails.
    """
    if require_swmr:
        assert_swmr_on(group)

    if len(dataframe.columns) == 0:
        raise ValidationError("Cannot preallocate layout for DataFrame with no columns")

    # Set frame attributes
    group.attrs["column_order"] = json.dumps(list(dataframe.columns))
    group.attrs["len"] = 0

    # Preallocate index layout
    index_group = group.create_group("index")
    # Create a dummy series with string values to match the schema
    dummy_series = pd.Series([], dtype=str, index=dataframe.index[:0], name="__index__")
    preallocate_series_layout(
        index_group,
        dummy_series,
        dataset="values",
        index_dataset="index",
        chunks=chunks,
        compression=compression,
        preallocate=preallocate,
        require_swmr=require_swmr,
    )

    # Preallocate column layouts
    columns_group = group.create_group("columns")
    for col_name in dataframe.columns:
        col_group = columns_group.create_group(str(col_name))
        # Create dummy series - use actual data to determine schema if available
        if len(dataframe) > 0:
            # Use first few values to determine the proper schema
            col_data = dataframe[col_name]
            dummy_col_series = pd.Series(
                [col_data.iloc[0]] if not col_data.isna().iloc[0] else [None],
                dtype=col_data.dtype,
                index=dataframe.index[:1],
                name=col_name,
            )
        else:
            # Fallback to dtype for empty dataframe
            col_dtype = dataframe[col_name].dtype
            dummy_col_series = pd.Series(
                [], dtype=col_dtype, index=dataframe.index[:0], name=col_name
            )

        preallocate_series_layout(
            col_group,
            dummy_col_series,
            dataset="values",
            index_dataset="index",
            chunks=chunks,
            compression=compression,
            preallocate=preallocate,
            require_swmr=require_swmr,
        )

    if require_swmr:
        group.file.flush()


def save_frame_new(
    group: h5py.Group,
    dataframe: pd.DataFrame,
    *,
    chunks: tuple[int, ...] = (25,),
    compression: str = "gzip",
    preallocate: int = 100,
    require_swmr: bool = True,
) -> None:
    """Create datasets and write a pandas DataFrame to HDF5.

    Args:
        group: HDF5 group to write to.
        dataframe: pandas DataFrame to persist.
        chunks: Chunk shape for new datasets.
        compression: Compression algorithm for new datasets.
        preallocate: Initial allocation size for new datasets.
        require_swmr: If True, assert SWMR mode is enabled.

    Raises:
        SWMRModeError: If require_swmr=True and SWMR mode not enabled.
        ValidationError: If DataFrame validation fails.
    """
    if require_swmr:
        assert_swmr_on(group)

    if len(dataframe) == 0:
        raise ValidationError("Cannot save empty DataFrame")

    # Check if layout already exists (preallocated)
    if "columns" in group and group.attrs.get("len", -1) == 0:
        save_frame_update(group, dataframe, start=0, require_swmr=require_swmr)
        return

    # Create new layout
    preallocate_frame_layout(
        group,
        dataframe,
        chunks=chunks,
        compression=compression,
        preallocate=max(preallocate, len(dataframe)),
        require_swmr=require_swmr,
    )

    # Write the data
    save_frame_update(group, dataframe, start=0, require_swmr=require_swmr)


def save_frame_update(
    group: h5py.Group,
    dataframe: pd.DataFrame,
    *,
    start: int = 0,
    require_swmr: bool = True,
) -> None:
    """Update a pandas DataFrame in HDF5 at specified position.

    Args:
        group: HDF5 group containing existing datasets.
        dataframe: pandas DataFrame to write.
        start: Starting position for the update.
        require_swmr: If True, assert SWMR mode is enabled.

    Raises:
        SWMRModeError: If require_swmr=True and SWMR mode not enabled.
        ValidationError: If validation fails or schema mismatch.
    """
    if require_swmr:
        assert_swmr_on(group)

    if len(dataframe) == 0:
        raise ValidationError("Cannot update with empty DataFrame")

    current_len = group.attrs["len"]
    end_pos = start + len(dataframe)

    # Validate contiguous update
    if start > current_len:
        raise ValidationError(
            f"Non-contiguous update: start={start}, current_len={current_len}"
        )

    # Validate column order matches
    column_order_attr = group.attrs["column_order"]
    if isinstance(column_order_attr, bytes):
        column_order_attr = column_order_attr.decode("utf-8")
    stored_columns = json.loads(column_order_attr)
    if list(dataframe.columns) != stored_columns:
        raise SchemaMismatchError(
            f"Column order mismatch: expected {stored_columns}, got {list(dataframe.columns)}"
        )

    # Update index - create a dummy series to represent the actual index
    # We need to store the index structure, so we create a dummy series where the
    # index is the actual DataFrame index and values are just placeholders
    index_series = pd.Series(
        ["dummy"] * len(dataframe), index=dataframe.index, name="__index__"
    )
    save_series_update(
        group["index"],
        index_series,
        start=start,
        dataset="values",
        index_dataset="index",
        require_swmr=require_swmr,
    )

    # Update each column
    columns_group = group["columns"]
    for col_name in dataframe.columns:
        col_series = dataframe[col_name]
        col_series.name = col_name
        save_series_update(
            columns_group[str(col_name)],
            col_series,
            start=start,
            dataset="values",
            index_dataset="index",
            require_swmr=require_swmr,
        )

    # Update frame length
    group.attrs["len"] = max(current_len, end_pos)

    if require_swmr:
        group.file.flush()


def save_frame_append(
    group: h5py.Group,
    dataframe: pd.DataFrame,
    *,
    require_swmr: bool = True,
) -> None:
    """Append a pandas DataFrame to existing HDF5 datasets.

    Args:
        group: HDF5 group containing existing datasets.
        dataframe: pandas DataFrame to append.
        require_swmr: If True, assert SWMR mode is enabled.

    Raises:
        SWMRModeError: If require_swmr=True and SWMR mode not enabled.
        ValidationError: If validation fails or schema mismatch.
    """
    if require_swmr:
        assert_swmr_on(group)

    current_len = group.attrs["len"]
    save_frame_update(group, dataframe, start=current_len, require_swmr=require_swmr)


def load_frame(
    group: h5py.Group,
    *,
    require_swmr: bool = False,
) -> pd.DataFrame:
    """Load a pandas DataFrame from HDF5 storage.

    Args:
        group: HDF5 group containing the DataFrame data.
        require_swmr: If True, assert SWMR mode is enabled.

    Returns:
        Reconstructed pandas DataFrame.

    Raises:
        SWMRModeError: If require_swmr=True and SWMR mode not enabled.
        ValidationError: If data validation fails.
    """
    if require_swmr:
        assert_swmr_on(group)

    logical_length = group.attrs["len"]
    if logical_length == 0:
        # Return empty DataFrame with proper schema
        column_order_attr = group.attrs["column_order"]
        if isinstance(column_order_attr, bytes):
            column_order_attr = column_order_attr.decode("utf-8")
        column_order = json.loads(column_order_attr)
        return pd.DataFrame(columns=column_order)

    # Load index
    index = _decode_index_from_hdf5(group["index"], "index", logical_length)

    # Load columns in order
    column_order_attr = group.attrs["column_order"]
    if isinstance(column_order_attr, bytes):
        column_order_attr = column_order_attr.decode("utf-8")
    column_order = json.loads(column_order_attr)
    columns_group = group["columns"]

    columns_data = {}
    for col_name in column_order:
        col_series = load_series(
            columns_group[str(col_name)],
            dataset="values",
            index_dataset="index",
            require_swmr=require_swmr,
        )
        columns_data[col_name] = col_series.values

    # Reconstruct DataFrame
    dataframe = pd.DataFrame(columns_data, index=index)
    result: pd.DataFrame = dataframe[column_order]  # Ensure column order
    return result
