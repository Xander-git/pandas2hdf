# pandas2hdf

Robust round-trip persistence between pandas Series/DataFrame and HDF5 with SWMR (Single Writer Multiple Reader) support.

## Features

- **Complete round-trip fidelity**: Preserves data types, index structure, names, and missing values
- **SWMR support**: Enables concurrent reading while writing with HDF5's Single Writer Multiple Reader mode
- **Flexible write modes**: Preallocate, new, update, and append operations
- **MultiIndex support**: Full support for pandas MultiIndex with proper reconstruction
- **Type safety**: Comprehensive type hints and strict mypy compliance
- **Comprehensive testing**: Extensive test suite covering edge cases and real-world scenarios

## Installation

```bash
pip install pandas2hdf
```

## Quick Start

### Basic Series Operations

```python
import pandas as pd
import h5py
from pandas2hdf import save_series_new, load_series

# Create a pandas Series
series = pd.Series([1, 2, 3, None, 5], 
                  index=['a', 'b', 'c', 'd', 'e'], 
                  name='my_data')

# Save to HDF5 with SWMR support
with h5py.File('data.h5', 'w', libver='latest') as f:
    f.swmr_mode = True
    group = f.create_group('my_series')
    save_series_new(group, series, require_swmr=True)

# Load from HDF5
with h5py.File('data.h5', 'r', swmr=True) as f:
    group = f['my_series']
    loaded_series = load_series(group)

print(loaded_series)
# Output preserves original data, index, and name
```

### DataFrame Operations

```python
import pandas as pd
import h5py
from pandas2hdf import save_frame_new, load_frame

# Create a DataFrame with mixed types
df = pd.DataFrame({
    'integers': [1, 2, 3, None],
    'floats': [1.1, 2.2, 3.3, 4.4],
    'strings': ['apple', 'banana', None, 'date'],
    'booleans': [True, False, True, None]
})

# Save DataFrame
with h5py.File('dataframe.h5', 'w', libver='latest') as f:
    f.swmr_mode = True
    group = f.create_group('my_dataframe')
    save_frame_new(group, df, require_swmr=True)

# Load DataFrame
with h5py.File('dataframe.h5', 'r', swmr=True) as f:
    group = f['my_dataframe']
    loaded_df = load_frame(group)

print(loaded_df)
```

### SWMR Workflow with Incremental Updates

```python
import pandas as pd
import h5py
from pandas2hdf import (
    preallocate_series_layout, 
    save_series_new, 
    save_series_append,
    load_series
)

# Writer process
with h5py.File('timeseries.h5', 'w', libver='latest') as f:
    f.swmr_mode = True
    group = f.create_group('data')
    
    # Preallocate space for efficient appending
    initial_data = pd.Series([1.0, 2.0], name='measurements')
    preallocate_series_layout(group, initial_data, preallocate=10000)
    
    # Write initial data
    save_series_new(group, initial_data, require_swmr=True)
    
    # Append new data incrementally
    for i in range(10):
        new_data = pd.Series([float(i + 3)], name='measurements')
        save_series_append(group, new_data, require_swmr=True)
        f.flush()  # Make data visible to readers

# Concurrent reader process
with h5py.File('timeseries.h5', 'r', swmr=True) as f:
    group = f['data']
    current_data = load_series(group)
    print(f"Current length: {len(current_data)}")
```

## API Reference

### Series Functions

- `preallocate_series_layout()`: Create resizable datasets without writing data
- `save_series_new()`: Create new datasets and write Series data  
- `save_series_update()`: Update Series data at specified position
- `save_series_append()`: Append Series data to end of existing datasets
- `load_series()`: Load Series from HDF5 storage

### DataFrame Functions

- `preallocate_frame_layout()`: Create resizable layout for DataFrame
- `save_frame_new()`: Create new datasets and write DataFrame  
- `save_frame_update()`: Update DataFrame data at specified position
- `save_frame_append()`: Append DataFrame data to existing datasets
- `load_frame()`: Load DataFrame from HDF5 storage

### Utility Functions

- `assert_swmr_on()`: Assert that SWMR mode is enabled on a file

## Data Type Handling

### Values
- **Numeric types** (int, float): Stored as float64 with NaN for missing values
- **Boolean**: Converted to float64 (True=1.0, False=0.0) with NaN for missing
- **Strings**: Stored as UTF-8 variable-length strings with separate mask for missing values

### Index
- **All index types**: Converted to UTF-8 strings for consistent storage
- **MultiIndex**: Each level stored separately with proper reconstruction metadata
- **Missing values**: Handled via mask arrays for all index levels

## SWMR (Single Writer Multiple Reader) Support

pandas2hdf is designed for SWMR workflows where one process writes data while multiple processes read concurrently:

```python
# Writer process
with h5py.File('data.h5', 'w', libver='latest') as f:
    f.swmr_mode = True  # Enable SWMR mode
    # ... write operations with require_swmr=True

# Reader processes  
with h5py.File('data.h5', 'r', swmr=True) as f:
    # ... read operations (automatically see new data after writer flushes)
```

### SWMR Requirements
- Use `libver='latest'` when creating files
- Set `swmr_mode = True` on writer file handle
- Use `require_swmr=True` for write operations (validates SWMR is enabled)
- Call `file.flush()` after writes to make data visible to readers
- Open reader files with `swmr=True`

## Error Handling

The library provides specific exception types:

- `SWMRModeError`: SWMR mode required but not enabled
- `SchemaMismatchError`: Data doesn't match existing schema
- `ValidationError`: General data validation errors

## Performance Considerations

- **Chunking**: Default chunk size is (25,) - adjust based on access patterns
- **Compression**: gzip compression enabled by default
- **Preallocation**: Specify expected size to avoid frequent resizing
- **SWMR**: Minimal overhead for concurrent reading

## Testing

Run the comprehensive test suite:

```bash
pytest tests/
```

The tests cover:
- Round-trip fidelity for all supported data types
- MultiIndex handling
- All write modes (preallocate, new, update, append)  
- SWMR workflows and concurrent access
- Error conditions and edge cases
- Performance with large datasets

## Requirements

- Python ≥ 3.10
- pandas ≥ 1.5.0
- h5py ≥ 3.7.0
- numpy ≥ 1.21.0

## License

This project is licensed under the MIT License - see the LICENSE file for details.
