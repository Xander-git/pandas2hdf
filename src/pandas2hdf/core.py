"""Core functionality for pandas to HDF5 round-trip persistence with SWMR support."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Optional, cast

import h5py
import numpy as np
import pandas as pd


class H5SWMRError(Exception):
    """Raised when SWMR mode is required but not enabled."""

    pass


class H5SchemaError(Exception):
    """Raised when there's a schema mismatch during writes."""

    pass


class H5DataError(Exception):
    """Raised for data validation errors."""

    pass


def assert_swmr_on(g: h5py.Group) -> None:
    """Assert that the file is in SWMR mode.

    Args:
        g: An h5py Group object whose file should be in SWMR mode.

    Raises:
        H5SWMRError: If the file is not in SWMR mode.
    """
    if not g.file.swmr_mode:
        raise H5SWMRError(
            f"File {g.file.filename} is not in SWMR mode. "
            "Open with swmr=True or enable SWMR mode before writing."
        )


def _get_value_encoding(series: pd.Series) -> tuple[str, str]:
    """Determine the encoding type for Series values.

    Args:
        series: The pandas Series to analyze.

    Returns:
        Tuple of (values_kind, orig_values_dtype) strings.
    """
    dtype = series.dtype
    orig_dtype = str(dtype)

    # Check for complex numbers early
    if pd.api.types.is_complex_dtype(dtype):
        raise H5DataError(f"Unsupported dtype: {dtype} (complex numbers not supported)")

    if pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_bool_dtype(dtype):
        return ("numeric_float64", orig_dtype)
    elif pd.api.types.is_string_dtype(dtype) or pd.api.types.is_object_dtype(dtype):
        # Check if all non-null values are strings
        non_null = series.dropna()
        if len(non_null) == 0 or all(isinstance(x, str) for x in non_null):
            return ("string_utf8_vlen", orig_dtype)
    elif isinstance(dtype, pd.CategoricalDtype):
        return ("string_utf8_vlen", orig_dtype)

    raise H5DataError(f"Unsupported dtype: {dtype}")


def _encode_values(
    series: pd.Series,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Encode Series values for HDF5 storage.

    Args:
        series: The pandas Series to encode.

    Returns:
        Tuple of (values_array, mask_array or None).
    """
    values_kind, _ = _get_value_encoding(series)

    if values_kind == "numeric_float64":
        # Convert to float64, NaN for missing
        if pd.api.types.is_bool_dtype(series.dtype):
            # Convert boolean to float: True=1.0, False=0.0
            values = series.astype(float).to_numpy()
        else:
            values = series.astype(np.float64).to_numpy()
        return (values, None)

    elif values_kind == "string_utf8_vlen":
        # Convert to object array of strings
        if isinstance(series.dtype, pd.CategoricalDtype):
            # Convert categorical to strings
            str_series = series.astype(str)
        else:
            str_series = series

        # Create mask: 1=valid, 0=missing
        mask = pd.notna(str_series).astype(np.uint8).to_numpy()

        # Replace NaN/None with empty string for storage
        values = str_series.fillna("").astype(str).to_numpy()

        return (values, mask)

    raise H5DataError(f"Unexpected values_kind: {values_kind}")


def _encode_index(
    index: pd.Index,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode pandas Index as UTF-8 strings with mask.
    
    Args:
        index: The pandas Index to encode.
        
    Returns:
        Tuple of (index_array, mask_array).
    """
    # Convert to strings
    if isinstance(index, pd.DatetimeIndex):
        # Convert datetime to ISO format strings
        str_index = index.strftime("%Y-%m-%dT%H:%M:%S.%f")
    else:
        # Convert any index to strings
        str_index = index.astype(str)
    
    # Create mask for any NaN/None values in the original index
    mask = pd.notna(pd.Series(index.to_numpy())).astype(np.uint8).to_numpy()
    
    # Convert to numpy array
    values = np.array(str_index, dtype=str)
    
    return (values, mask)


def _encode_multiindex(index: pd.MultiIndex) -> dict[str, Any]:
    """Encode pandas MultiIndex preserving all level information.
    
    Args:
        index: The pandas MultiIndex to encode.
        
    Returns:
        Dictionary containing level data, codes, names, and metadata.
    """
    result = {
        "nlevels": index.nlevels,
        "level_names": [name if name is not None else "" for name in index.names],
        "level_dtypes": [str(level.dtype) for level in index.levels],
        "levels": {},
        "codes": index.codes,
    }
    
    # Encode each level
    for i, level in enumerate(index.levels):
        level_values, level_mask = _encode_level_values(level)
        result["levels"][f"level_{i}"] = {
            "values": level_values,
            "mask": level_mask,
        }
    
    return result


def _encode_level_values(level: pd.Index) -> tuple[np.ndarray, np.ndarray]:
    """Encode level values similar to regular index encoding."""
    if isinstance(level, pd.DatetimeIndex):
        str_level = level.strftime("%Y-%m-%dT%H:%M:%S.%f")
    else:
        str_level = level.astype(str)
    
    # Create mask for any NaN/None values
    mask = pd.notna(pd.Series(level.to_numpy())).astype(np.uint8).to_numpy()
    
    # Convert to object array to ensure compatibility with h5py
    values = np.array([str(x) for x in str_level], dtype=object)
    
    return (values, mask)


def _decode_multiindex(
    index_data: dict[str, Any], logical_len: int
) -> pd.MultiIndex:
    """Decode MultiIndex from stored data.
    
    Args:
        index_data: Dictionary containing MultiIndex data.
        logical_len: The logical length of the index.
        
    Returns:
        Reconstructed pandas MultiIndex.
    """
    nlevels = index_data["nlevels"]
    level_names = index_data["level_names"]
    
    # Decode levels
    levels = []
    for i in range(nlevels):
        level_info = index_data["levels"][f"level_{i}"]
        level_values = level_info["values"]
        level_mask = level_info["mask"]
        
        # Decode level values
        decoded_values = []
        for val, valid in zip(level_values, level_mask):
            if valid:
                if isinstance(val, bytes):
                    decoded_values.append(val.decode("utf-8"))
                else:
                    decoded_values.append(val)
            else:
                decoded_values.append(None)
        
        levels.append(pd.Index(decoded_values))
    
    # Get codes for the logical length
    codes = [code_array[:logical_len] for code_array in index_data["codes"]]
    
    # Convert empty names back to None
    names = [name if name else None for name in level_names]
    
    return pd.MultiIndex(levels=levels, codes=codes, names=names)


def _write_multiindex_to_hdf5(
    group: h5py.Group,
    index: pd.MultiIndex,
    base_name: str,
    chunks: tuple[int, ...],
    compression: str,
    preallocate: int,
) -> None:
    """Write MultiIndex data to HDF5 group.
    
    Args:
        group: HDF5 group to write to.
        index: MultiIndex to write.
        base_name: Base name for the index datasets (e.g., 'index').
        chunks: Chunk shape for datasets.
        compression: Compression type.
        preallocate: Size to preallocate.
    """
    # Encode the MultiIndex
    index_data = _encode_multiindex(index)
    
    # Create levels group
    levels_group = group.create_group(f"{base_name}_levels")
    
    # Store metadata as attributes
    levels_group.attrs["nlevels"] = index_data["nlevels"]
    levels_group.attrs["level_names"] = json.dumps(index_data["level_names"])
    levels_group.attrs["level_dtypes"] = json.dumps(index_data["level_dtypes"])
    levels_group.attrs["is_multiindex"] = True
    
    # Write each level
    for level_key, level_info in index_data["levels"].items():
        level_group = levels_group.create_group(level_key)
        
        # Create datasets for level values and mask
        # Convert to bytes for h5py compatibility
        string_data = [s.encode('utf-8') if s is not None else b'' for s in level_info["values"]]
        level_group.create_dataset(
            "values",
            data=string_data,
            dtype=h5py.string_dtype(encoding="utf-8"),
            compression=compression,
        )
        level_group.create_dataset(
            "mask",
            data=level_info["mask"],
            dtype=np.uint8,
            compression=compression,
        )
    
    # Create codes dataset - shape (nlevels, preallocate)
    codes_shape = (index_data["nlevels"], preallocate)
    codes_data = np.full(codes_shape, -1, dtype=np.int32)  # -1 for missing
    
    # Fill in actual codes for the current data
    for i, code_array in enumerate(index_data["codes"]):
        codes_data[i, : len(code_array)] = code_array
    
    group.create_dataset(
        f"{base_name}_codes",
        data=codes_data,
        maxshape=(index_data["nlevels"], None),
        chunks=(index_data["nlevels"], chunks[0]),
        compression=compression,
    )


def _write_simple_index_to_hdf5(
    group: h5py.Group,
    index: pd.Index,
    base_name: str,
    chunks: tuple[int, ...],
    compression: str,
    preallocate: int,
) -> None:
    """Write simple Index data to HDF5 group."""
    # Mark as simple index
    group.attrs[f"{base_name}_is_multiindex"] = False
    
    # Create index datasets
    group.create_dataset(
        base_name,
        shape=(preallocate,),
        maxshape=(None,),
        dtype=h5py.string_dtype(encoding="utf-8"),
        chunks=chunks,
        compression=compression,
    )
    
    group.create_dataset(
        f"{base_name}_mask",
        shape=(preallocate,),
        maxshape=(None,),
        dtype=np.uint8,
        chunks=chunks,
        compression=compression,
        fillvalue=0,
    )


def _read_multiindex_from_hdf5(
    group: h5py.Group, base_name: str, logical_len: int
) -> pd.MultiIndex:
    """Read MultiIndex data from HDF5 group."""
    levels_group = group[f"{base_name}_levels"]
    
    # Read metadata
    nlevels = levels_group.attrs["nlevels"]
    level_names = json.loads(levels_group.attrs["level_names"])
    
    # Read level data
    levels = []
    for i in range(nlevels):
        level_group = levels_group[f"level_{i}"]
        level_values = level_group["values"][:]
        level_mask = level_group["mask"][:]
        
        # Decode level values
        decoded_values = []
        for val, valid in zip(level_values, level_mask, strict=False):
            if valid:
                if isinstance(val, bytes):
                    decoded_values.append(val.decode("utf-8"))
                else:
                    decoded_values.append(val)
            else:
                decoded_values.append(None)
        
        levels.append(pd.Index(decoded_values))
    
    # Read codes
    codes_data = group[f"{base_name}_codes"][:, :logical_len]
    codes = [codes_data[i] for i in range(nlevels)]
    
    # Convert empty names back to None
    names = [name if name else None for name in level_names]
    
    return pd.MultiIndex(levels=levels, codes=codes, names=names)


def _update_multiindex_data(
    group: h5py.Group,
    index: pd.MultiIndex,
    base_name: str,
    start: int,
    end: int,
) -> None:
    """Update MultiIndex codes in existing HDF5 datasets."""
    index_data = _encode_multiindex(index)
    
    # Update codes
    codes_dataset = group[f"{base_name}_codes"]
    
    # Resize if needed
    if codes_dataset.shape[1] < end:
        new_size = max(end, codes_dataset.shape[1] * 2)
        codes_dataset.resize((codes_dataset.shape[0], new_size))
    
    # Write codes
    for i, code_array in enumerate(index_data["codes"]):
        codes_dataset[i, start:end] = code_array


def _update_simple_index_data(
    group: h5py.Group,
    index: pd.Index,
    base_name: str,
    start: int,
    end: int,
) -> None:
    """Update simple Index data in existing HDF5 datasets."""
    index_values, index_mask = _encode_index(index)
    
    # Resize if needed
    if group[base_name].shape[0] < end:
        new_size = max(end, group[base_name].shape[0] * 2)
        group[base_name].resize((new_size,))
        group[f"{base_name}_mask"].resize((new_size,))
    
    # Write data
    group[base_name][start:end] = index_values
    group[f"{base_name}_mask"][start:end] = index_mask


def preallocate_series_layout(
    g: h5py.Group,
    s: pd.Series,
    *,
    dataset: str = "values",
    index_dataset: str = "index",
    chunks: tuple[int, ...] = (25,),
    compression: str = "gzip",
    preallocate: int = 100,
    require_swmr: bool = True,
) -> None:
    """Preallocate HDF5 layout for a Series without writing data.

    Creates resizable, chunked datasets with initial shape=(preallocate,) and
    maxshape=(None,). Sets len=0 and all schema attributes. This is pure layout
    preallocation with no user data written.

    Args:
        g: HDF5 group to write to.
        s: Series providing schema information.
        dataset: Name for values dataset.
        index_dataset: Name for index dataset.
        chunks: Chunk shape for datasets.
        compression: Compression type.
        preallocate: Initial dataset size.
        require_swmr: Whether to require SWMR mode.

    Raises:
        H5SWMRError: If require_swmr=True and file not in SWMR mode.
    """
    if require_swmr:
        assert_swmr_on(g)

    # Determine encoding
    values_kind, orig_values_dtype = _get_value_encoding(s)

    # Create value dataset
    if values_kind == "numeric_float64":
        dtype = np.float64
    else:  # string_utf8_vlen
        dtype = h5py.string_dtype(encoding="utf-8")

    g.create_dataset(
        dataset,
        shape=(preallocate,),
        maxshape=(None,),
        dtype=dtype,
        chunks=chunks,
        compression=compression,
    )

    # Create mask if needed
    if values_kind == "string_utf8_vlen":
        g.create_dataset(
            f"{dataset}_mask",
            shape=(preallocate,),
            maxshape=(None,),
            dtype=np.uint8,
            chunks=chunks,
            compression=compression,
            fillvalue=0,
        )

    # Create index datasets (handle both simple and MultiIndex)
    if isinstance(s.index, pd.MultiIndex):
        _write_multiindex_to_hdf5(g, s.index, index_dataset, chunks, compression, preallocate)
        g.attrs["index_kind"] = "multiindex"
    else:
        _write_simple_index_to_hdf5(g, s.index, index_dataset, chunks, compression, preallocate)
        g.attrs["index_kind"] = "string_utf8_vlen"

    # Set attributes
    g.attrs["series_name"] = s.name if s.name is not None else ""
    g.attrs["len"] = 0  # No data written yet
    g.attrs["values_kind"] = values_kind
    g.attrs["index_name"] = (
        s.index.name if s.index.name is not None and not isinstance(s.index, pd.MultiIndex) else ""
    )
    g.attrs["created_at_iso"] = datetime.utcnow().isoformat()
    g.attrs["version"] = "1.0"
    g.attrs["orig_values_dtype"] = orig_values_dtype
    g.attrs["orig_index_dtype"] = str(s.index.dtype)

    if require_swmr:
        g.file.flush()


def save_series_new(
    g: h5py.Group,
    s: pd.Series,
    *,
    dataset: str = "values",
    index_dataset: str = "index",
    chunks: tuple[int, ...] = (25,),
    compression: str = "gzip",
    preallocate: int = 100,
    require_swmr: bool = True,
) -> None:
    """Save a new Series to HDF5, creating datasets or reusing preallocated layout.

    Args:
        g: HDF5 group to write to.
        s: Series to save.
        dataset: Name for values dataset.
        index_dataset: Name for index dataset.
        chunks: Chunk shape for datasets.
        compression: Compression type.
        preallocate: Initial dataset size.
        require_swmr: Whether to require SWMR mode.

    Raises:
        H5SWMRError: If require_swmr=True and file not in SWMR mode.
        H5DataError: If series is empty or has mismatched lengths.
    """
    if require_swmr:
        assert_swmr_on(g)

    if len(s) == 0:
        raise H5DataError("Cannot save empty series")

    # Check if layout exists
    if dataset not in g:
        # Create layout
        preallocate_series_layout(
            g,
            s,
            dataset=dataset,
            index_dataset=index_dataset,
            chunks=chunks,
            compression=compression,
            preallocate=max(preallocate, len(s)),
            require_swmr=require_swmr,
        )

    # Encode data
    values, values_mask = _encode_values(s)

    n = len(values)

    # Resize if needed
    if g[dataset].shape[0] < n:
        new_size = max(n, preallocate)
        g[dataset].resize((new_size,))
        if f"{dataset}_mask" in g:
            g[f"{dataset}_mask"].resize((new_size,))

    # Write data
    g[dataset][:n] = values
    if values_mask is not None:
        g[f"{dataset}_mask"][:n] = values_mask

    # Handle index data (MultiIndex or simple)
    index_kind = g.attrs.get("index_kind", "string_utf8_vlen")
    if index_kind == "multiindex":
        if not isinstance(s.index, pd.MultiIndex):
            raise H5SchemaError(
                "Cannot update MultiIndex series with simple index"
            )
        _update_multiindex_data(g, s.index, index_dataset, 0, n)
    else:
        if isinstance(s.index, pd.MultiIndex):
            raise H5SchemaError(
                "Cannot update simple index series with MultiIndex"
            )
        _update_simple_index_data(g, s.index, index_dataset, 0, n)

    # Update length
    g.attrs["len"] = n

    if require_swmr:
        g.file.flush()


def save_series_update(
    g: h5py.Group,
    s: pd.Series,
    *,
    start: int = 0,
    dataset: str = "values",
    index_dataset: str = "index",
    require_swmr: bool = True,
) -> None:
    """Update existing Series data starting at a specific position.

    Args:
        g: HDF5 group containing the series.
        s: Series data to write.
        start: Starting position for the update.
        dataset: Name of values dataset.
        index_dataset: Name of index dataset.
        require_swmr: Whether to require SWMR mode.

    Raises:
        H5SWMRError: If require_swmr=True and file not in SWMR mode.
        H5SchemaError: If attempting to write incompatible data type.
        H5DataError: If update would create non-contiguous data.
    """
    if require_swmr:
        assert_swmr_on(g)

    if dataset not in g:
        raise H5DataError(f"Dataset '{dataset}' not found in group")

    current_len = g.attrs.get("len", 0)

    # Encode data
    values, values_mask = _encode_values(s)

    n = len(values)
    end = start + n

    # Check schema compatibility
    values_kind = g.attrs.get("values_kind")
    new_values_kind, _ = _get_value_encoding(s)
    if values_kind != new_values_kind:
        raise H5SchemaError(
            f"Cannot update with different value type. "
            f"Expected {values_kind}, got {new_values_kind}"
        )

    # Check index compatibility
    index_kind = g.attrs.get("index_kind", "string_utf8_vlen")
    if index_kind == "multiindex":
        if not isinstance(s.index, pd.MultiIndex):
            raise H5SchemaError(
                "Cannot update MultiIndex series with simple index"
            )
    else:
        if isinstance(s.index, pd.MultiIndex):
            raise H5SchemaError(
                "Cannot update simple index series with MultiIndex"
            )

    # Resize if needed
    if g[dataset].shape[0] < end:
        new_size = max(end, g[dataset].shape[0] * 2)  # Double size
        g[dataset].resize((new_size,))
        if f"{dataset}_mask" in g:
            g[f"{dataset}_mask"].resize((new_size,))

    # Write data
    g[dataset][start:end] = values
    if values_mask is not None:
        g[f"{dataset}_mask"][start:end] = values_mask

    # Handle index data (MultiIndex or simple)
    if index_kind == "multiindex":
        _update_multiindex_data(g, s.index, index_dataset, start, end)
    else:
        _update_simple_index_data(g, s.index, index_dataset, start, end)

    # Update length to largest contiguous extent
    # Only allow contiguous writes unless appending at the end
    if start > current_len:
        raise H5DataError(
            f"Non-contiguous update: start={start} but current length={current_len}. "
            "Updates must be contiguous."
        )

    new_len = max(current_len, end)
    g.attrs["len"] = new_len

    if require_swmr:
        g.file.flush()


def save_series_append(
    g: h5py.Group,
    s: pd.Series,
    *,
    dataset: str = "values",
    index_dataset: str = "index",
    require_swmr: bool = True,
) -> None:
    """Append Series data at the end of existing data.

    Args:
        g: HDF5 group containing the series.
        s: Series data to append.
        dataset: Name of values dataset.
        index_dataset: Name of index dataset.
        require_swmr: Whether to require SWMR mode.

    Raises:
        H5SWMRError: If require_swmr=True and file not in SWMR mode.
        H5SchemaError: If attempting to append incompatible data type.
    """
    if require_swmr:
        assert_swmr_on(g)

    if dataset not in g:
        raise H5DataError(f"Dataset '{dataset}' not found in group")

    current_len = g.attrs.get("len", 0)

    # Use update starting at current length
    save_series_update(
        g,
        s,
        start=current_len,
        dataset=dataset,
        index_dataset=index_dataset,
        require_swmr=require_swmr,
    )


def load_series(
    g: h5py.Group,
    *,
    dataset: str = "values",
    index_dataset: str = "index",
    require_swmr: bool = False,
) -> pd.Series:  # type: ignore[type-arg]
    """Load a Series from HDF5.

    Args:
        g: HDF5 group containing the series.
        dataset: Name of values dataset.
        index_dataset: Name of index dataset.
        require_swmr: Whether to require SWMR mode for reading.

    Returns:
        The loaded pandas Series.

    Raises:
        H5SWMRError: If require_swmr=True and file not in SWMR mode.
        H5DataError: If required datasets or attributes are missing.
    """
    if require_swmr:
        assert_swmr_on(g)

    # Get metadata
    logical_len = g.attrs.get("len")
    if logical_len is None:
        raise H5DataError("Missing 'len' attribute")

    series_name = g.attrs.get("series_name", "")
    if isinstance(series_name, (list, tuple, np.ndarray)):
        series_name = series_name[0] if len(series_name) > 0 else ""
    series_name = series_name if series_name else None

    index_name = g.attrs.get("index_name", "")
    if isinstance(index_name, (list, tuple, np.ndarray)):
        index_name = index_name[0] if len(index_name) > 0 else ""
    index_name = index_name if index_name else None

    values_kind = g.attrs.get("values_kind")
    if not values_kind:
        raise H5DataError("Missing 'values_kind' attribute")

    # Read values
    if values_kind == "numeric_float64":
        values = g[dataset][:logical_len]
        # No mask for numeric values, NaN represents missing
    else:  # string_utf8_vlen
        raw_values = g[dataset][:logical_len]
        mask = g[f"{dataset}_mask"][:logical_len]
        # Convert to object array and apply mask, decoding bytes to strings
        values = np.array(
            [
                v.decode("utf-8") if isinstance(v, bytes) and m else (v if m else None)
                for v, m in zip(raw_values, mask, strict=False)
            ],
            dtype=object,
        )

    # Read index (handle both simple and MultiIndex)
    index_kind = g.attrs.get("index_kind", "string_utf8_vlen")
    if index_kind == "multiindex":
        index = _read_multiindex_from_hdf5(g, index_dataset, logical_len)
    else:
        raw_index = g[index_dataset][:logical_len]
        index_mask = g[f"{index_dataset}_mask"][:logical_len]

        # Decode index
        index_values: list[str | None] = []
        for idx_str, valid in zip(raw_index, index_mask, strict=False):
            if valid:
                # Decode bytes to string if needed
                if isinstance(idx_str, bytes):
                    index_values.append(idx_str.decode("utf-8"))
                else:
                    index_values.append(idx_str)
            else:
                index_values.append(None)

        index = pd.Index(index_values, name=index_name)

    # Create series
    series = pd.Series(values, index=index, name=series_name)

    return series


def preallocate_frame_layout(
    g: h5py.Group,
    df: pd.DataFrame,
    *,
    chunks: tuple[int, ...] = (25,),
    compression: str = "gzip",
    preallocate: int = 100,
    require_swmr: bool = True,
) -> None:
    """Preallocate HDF5 layout for a DataFrame without writing data.

    Args:
        g: HDF5 group to write to.
        df: DataFrame providing schema information.
        chunks: Chunk shape for datasets.
        compression: Compression type.
        preallocate: Initial dataset size.
        require_swmr: Whether to require SWMR mode.

    Raises:
        H5SWMRError: If require_swmr=True and file not in SWMR mode.
    """
    if require_swmr:
        assert_swmr_on(g)

    # Store column information (handle both simple and MultiIndex columns)
    if isinstance(df.columns, pd.MultiIndex):
        # Store MultiIndex column information
        column_data = _encode_multiindex(df.columns)
        g.attrs["columns_is_multiindex"] = True
        g.attrs["columns_nlevels"] = column_data["nlevels"]
        g.attrs["columns_level_names"] = json.dumps(column_data["level_names"])
        g.attrs["columns_level_dtypes"] = json.dumps(column_data["level_dtypes"])
        
        # Store level data
        columns_levels_group = g.create_group("columns_levels")
        for level_key, level_info in column_data["levels"].items():
            level_group = columns_levels_group.create_group(level_key)
            # Convert to bytes for h5py compatibility
            string_data = [s.encode('utf-8') if s is not None else b'' for s in level_info["values"]]
            level_group.create_dataset(
                "values",
                data=string_data,
                dtype=h5py.string_dtype(encoding="utf-8"),
                compression=compression,
            )
            level_group.create_dataset(
                "mask",
                data=level_info["mask"],
                dtype=np.uint8,
                compression=compression,
            )
        
        # Store codes
        g.create_dataset(
            "columns_codes",
            data=np.array(column_data["codes"]),
            dtype=np.int32,
            compression=compression,
        )
    else:
        # Simple column index
        g.attrs["columns_is_multiindex"] = False
        g.attrs["column_order"] = json.dumps(list(df.columns))

    # Create index group
    index_group = g.create_group("index")
    
    # Handle index data (MultiIndex or simple)
    if isinstance(df.index, pd.MultiIndex):
        _write_multiindex_to_hdf5(index_group, df.index, "index", chunks, compression, preallocate)
        index_group.attrs["index_kind"] = "multiindex"
        index_group.attrs["len"] = 0
        
        # Create a dummy values dataset for compatibility
        index_group.create_dataset(
            "values",
            shape=(preallocate,),
            maxshape=(None,),
            dtype=h5py.string_dtype(encoding="utf-8"),
            chunks=chunks,
            compression=compression,
        )
        index_group.create_dataset(
            "values_mask",
            shape=(preallocate,),
            maxshape=(None,),
            dtype=np.uint8,
            chunks=chunks,
            compression=compression,
            fillvalue=0,
        )
    else:
        # Simple index
        _write_simple_index_to_hdf5(index_group, df.index, "values", chunks, compression, preallocate)
        index_group.attrs["index_kind"] = "string_utf8_vlen"
        index_group.attrs["len"] = 0
        index_group.attrs["index_name"] = df.index.name if df.index.name is not None else ""

    # Create columns group
    columns_group = g.create_group("columns")

    # Preallocate each column
    for i, col in enumerate(df.columns):
        # For MultiIndex columns, use index as group name since tuples can't be group names
        col_group_name = f"col_{i}" if isinstance(df.columns, pd.MultiIndex) else str(col)
        col_group = columns_group.create_group(col_group_name)
        
        preallocate_series_layout(
            col_group,
            df[col][:0],  # Empty slice for schema
            chunks=chunks,
            compression=compression,
            preallocate=preallocate,
            require_swmr=require_swmr,
        )

        # Store categories for categorical columns
        if isinstance(df[col].dtype, pd.CategoricalDtype):
            categories = df[col].cat.categories.tolist()
            col_group.attrs["categories"] = json.dumps([str(c) for c in categories])
        
        # Store column information for reference
        if isinstance(df.columns, pd.MultiIndex):
            col_group.attrs["column_tuple"] = json.dumps([str(x) for x in col])
        else:
            col_group.attrs["column_name"] = str(col)

    if require_swmr:
        g.file.flush()


def save_frame_new(
    g: h5py.Group,
    df: pd.DataFrame,
    *,
    chunks: tuple[int, ...] = (25,),
    compression: str = "gzip",
    preallocate: int = 100,
    require_swmr: bool = True,
) -> None:
    """Save a new DataFrame to HDF5.

    Args:
        g: HDF5 group to write to.
        df: DataFrame to save.
        chunks: Chunk shape for datasets.
        compression: Compression type.
        preallocate: Initial dataset size.
        require_swmr: Whether to require SWMR mode.

    Raises:
        H5SWMRError: If require_swmr=True and file not in SWMR mode.
        H5DataError: If dataframe is empty.
    """
    if require_swmr:
        assert_swmr_on(g)

    if len(df) == 0:
        raise H5DataError("Cannot save empty dataframe")

    # Preallocate if needed
    if "columns" not in g:
        preallocate_frame_layout(
            g,
            df,
            chunks=chunks,
            compression=compression,
            preallocate=max(preallocate, len(df)),
            require_swmr=require_swmr,
        )

    # Save index data
    index_group = g["index"]
    n = len(df)

    # Handle index data (MultiIndex or simple)
    index_kind = index_group.attrs.get("index_kind", "string_utf8_vlen")
    if index_kind == "multiindex":
        _update_multiindex_data(index_group, df.index, "index", 0, n)
    else:
        _update_simple_index_data(index_group, df.index, "values", 0, n)  # Use "values" not "index"
    
    index_group.attrs["len"] = n

    # Save each column
    for i, col in enumerate(df.columns):
        col_group_name = f"col_{i}" if isinstance(df.columns, pd.MultiIndex) else str(col)
        col_group = g["columns"][col_group_name]
        save_series_new(
            col_group,
            df[col],
            chunks=chunks,
            compression=compression,
            preallocate=preallocate,
            require_swmr=require_swmr,
        )

    if require_swmr:
        g.file.flush()


def save_frame_update(
    g: h5py.Group, df: pd.DataFrame, *, start: int = 0, require_swmr: bool = True
) -> None:
    """Update existing DataFrame data starting at a specific position.

    Args:
        g: HDF5 group containing the dataframe.
        df: DataFrame data to write.
        start: Starting position for the update.
        require_swmr: Whether to require SWMR mode.

    Raises:
        H5SWMRError: If require_swmr=True and file not in SWMR mode.
        H5DataError: If columns don't match.
    """
    if require_swmr:
        assert_swmr_on(g)

    # Validate columns match
    columns_is_multiindex = g.attrs.get("columns_is_multiindex", False)
    if columns_is_multiindex:
        if not isinstance(df.columns, pd.MultiIndex):
            raise H5DataError(
                "Cannot update DataFrame with MultiIndex columns using simple columns"
            )
        # For MultiIndex, validate the structure matches
        stored_nlevels = g.attrs.get("columns_nlevels")
        if df.columns.nlevels != stored_nlevels:
            raise H5DataError(
                f"Column MultiIndex level mismatch. Expected {stored_nlevels} levels, got {df.columns.nlevels}"
            )
    else:
        if isinstance(df.columns, pd.MultiIndex):
            raise H5DataError(
                "Cannot update DataFrame with simple columns using MultiIndex columns"
            )
        stored_columns = json.loads(g.attrs.get("column_order", "[]"))
        if list(df.columns) != stored_columns:
            raise H5DataError(
                f"Column mismatch. Expected {stored_columns}, got {list(df.columns)}"
            )

    # Update index
    index_group = g["index"]
    n = len(df)
    end = start + n

    # Handle index data (MultiIndex or simple)
    index_kind = index_group.attrs.get("index_kind", "string_utf8_vlen")
    if index_kind == "multiindex":
        _update_multiindex_data(index_group, df.index, "index", start, end)
    else:
        _update_simple_index_data(index_group, df.index, "values", start, end)  # Use "values" not "index"

    # Update length
    current_len = index_group.attrs.get("len", 0)
    if start > current_len:
        raise H5DataError(
            f"Non-contiguous update: start={start} but current length={current_len}"
        )
    index_group.attrs["len"] = max(current_len, end)

    # Update each column
    for i, col in enumerate(df.columns):
        col_group_name = f"col_{i}" if isinstance(df.columns, pd.MultiIndex) else str(col)
        col_group = g["columns"][col_group_name]
        save_series_update(col_group, df[col], start=start, require_swmr=require_swmr)

    if require_swmr:
        g.file.flush()


def save_frame_append(
    g: h5py.Group, df: pd.DataFrame, *, require_swmr: bool = True
) -> None:
    """Append DataFrame data at the end of existing data.

    Args:
        g: HDF5 group containing the dataframe.
        df: DataFrame data to append.
        require_swmr: Whether to require SWMR mode.

    Raises:
        H5SWMRError: If require_swmr=True and file not in SWMR mode.
    """
    if require_swmr:
        assert_swmr_on(g)

    # Get current length from index
    current_len = g["index"].attrs.get("len", 0)

    # Use update starting at current length
    save_frame_update(g, df, start=current_len, require_swmr=require_swmr)


def load_frame(g: h5py.Group, *, require_swmr: bool = False) -> pd.DataFrame:
    """Load a DataFrame from HDF5.

    Args:
        g: HDF5 group containing the dataframe.
        require_swmr: Whether to require SWMR mode for reading.

    Returns:
        The loaded pandas DataFrame.

    Raises:
        H5SWMRError: If require_swmr=True and file not in SWMR mode.
        H5DataError: If required groups or attributes are missing.
    """
    if require_swmr:
        assert_swmr_on(g)

    # Load index
    index_group = g["index"]
    logical_len = index_group.attrs.get("len", 0)

    # Handle index (MultiIndex or simple)
    index_kind = index_group.attrs.get("index_kind", "string_utf8_vlen")
    if index_kind == "multiindex":
        index = _read_multiindex_from_hdf5(index_group, "index", logical_len)
    else:
        raw_index = index_group["values"][:logical_len]
        index_mask = index_group["values_mask"][:logical_len]

        # Decode index
        index_values: list[str | None] = []
        for idx_str, valid in zip(raw_index, index_mask, strict=False):
            if valid:
                # Decode bytes to string if needed
                if isinstance(idx_str, bytes):
                    index_values.append(idx_str.decode("utf-8"))
                else:
                    index_values.append(idx_str)
            else:
                index_values.append(None)

        index = pd.Index(index_values, name=index_group.attrs.get("index_name", None))

    # Load columns
    columns_is_multiindex = g.attrs.get("columns_is_multiindex", False)
    
    if columns_is_multiindex:
        # Reconstruct MultiIndex columns
        nlevels = g.attrs["columns_nlevels"]
        level_names = json.loads(g.attrs["columns_level_names"])
        
        # Read level data
        levels = []
        columns_levels_group = g["columns_levels"]
        for i in range(nlevels):
            level_group = columns_levels_group[f"level_{i}"]
            level_values = level_group["values"][:]
            level_mask = level_group["mask"][:]
            
            # Decode level values
            decoded_values = []
            for val, valid in zip(level_values, level_mask, strict=False):
                if valid:
                    if isinstance(val, bytes):
                        decoded_values.append(val.decode("utf-8"))
                    else:
                        decoded_values.append(val)
                else:
                    decoded_values.append(None)
            
            levels.append(pd.Index(decoded_values))
        
        # Read codes
        codes_data = g["columns_codes"][:]
        codes = [codes_data[i] for i in range(nlevels)]
        
        # Convert empty names back to None
        names = [name if name else None for name in level_names]
        
        columns = pd.MultiIndex(levels=levels, codes=codes, names=names)
        
        # Load column data
        data = {}
        for i, col in enumerate(columns):
            col_group = g["columns"][f"col_{i}"]
            series = load_series(col_group, require_swmr=require_swmr)
            data[col] = series.values
    else:
        # Simple columns
        column_order = json.loads(g.attrs.get("column_order", "[]"))
        if not column_order:
            raise H5DataError("Missing or empty 'column_order' attribute")
        
        columns = column_order
        
        # Load column data
        data = {}
        for col in column_order:
            col_group = g["columns"][str(col)]
            series = load_series(col_group, require_swmr=require_swmr)
            data[col] = series.values

    # Create dataframe
    df = pd.DataFrame(data, index=index, columns=columns)

    return df
