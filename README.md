# pandas2hdf

Robust round-trip persistence between pandas Series/DataFrame and HDF5 with SWMR (Single Writer Multiple Reader) support.

## Features

- **Complete round-trip fidelity**: Preserves data types, index structure, names, and missing values
- **SWMR support**: Enables concurrent reading while writing with HDF5's Single Writer Multiple Reader mode
- **Fixed-length strings**: SWMR-compatible string storage with configurable character length
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
    group = f.create_group('my_series')
    # Step 1: Create all objects BEFORE enabling SWMR
    save_series_new(group, series, require_swmr=False)
    # Step 2: Enable SWMR mode
    f.swmr_mode = True

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
    group = f.create_group('my_dataframe')
    # Step 1: Create all objects BEFORE enabling SWMR
    save_frame_new(group, df, require_swmr=False)
    # Step 2: Enable SWMR mode
    f.swmr_mode = True

# Load DataFrame
with h5py.File('dataframe.h5', 'r', swmr=True) as f:
    group = f['my_dataframe']
    loaded_df = load_frame(group)

print(loaded_df)
```

### Fixed-Length String Configuration

```python
import pandas as pd
import h5py
from pandas2hdf import save_series_new, load_series

# Series with varying string lengths
series = pd.Series(['short', 'this is a very long string that will be truncated', 'mid'], 
                  name='text_data')

# Save with custom string length
with h5py.File('strings.h5', 'w', libver='latest') as f:
    group = f.create_group('strings')
    # Configure fixed-length to 20 characters for this dataset
    save_series_new(group, series, string_fixed_length=20, require_swmr=False)

# Load and see the results
with h5py.File('strings.h5', 'r') as f:
    group = f['strings']
    loaded = load_series(group)
    print(loaded)
    # Output: ['short', 'this is a very long', 'mid']
    # Note: long string truncated to 20 chars, trailing whitespace trimmed
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
    group = f.create_group('data')
    
    # Step 1: Create all objects BEFORE enabling SWMR
    initial_data = pd.Series([1.0, 2.0], name='measurements')
    preallocate_series_layout(group, initial_data, preallocate=10000, require_swmr=False)
    
    # Write initial data
    save_series_new(group, initial_data, require_swmr=False)
    
    # Step 2: Enable SWMR mode
    f.swmr_mode = True
    
    # Step 3: Append new data incrementally under SWMR
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

- `preallocate_series_layout(group, series, *, string_fixed_length=100, require_swmr=False, ...)`: Create resizable datasets without writing data
- `save_series_new(group, series, *, string_fixed_length=100, require_swmr=False, ...)`: Create new datasets and write Series data  
- `save_series_update(group, series, start, *, require_swmr=False)`: Update Series data at specified position
- `save_series_append(group, series, *, require_swmr=False)`: Append Series data to end of existing datasets
- `load_series(group, *, require_swmr=False)`: Load Series from HDF5 storage

### DataFrame Functions

- `preallocate_frame_layout(group, dataframe, *, string_fixed_length=100, require_swmr=False, ...)`: Create resizable layout for DataFrame
- `save_frame_new(group, dataframe, *, string_fixed_length=100, require_swmr=False, ...)`: Create new datasets and write DataFrame  
- `save_frame_update(group, dataframe, start, *, require_swmr=False)`: Update DataFrame data at specified position
- `save_frame_append(group, dataframe, *, require_swmr=False)`: Append DataFrame data to existing datasets
- `load_frame(group, *, require_swmr=False)`: Load DataFrame from HDF5 storage

### Utility Functions

- `assert_swmr_on()`: Assert that SWMR mode is enabled on a file

## Data Type Handling

### Values
- **Numeric types** (int, float): Stored as float64 with NaN for missing values
- **Boolean**: Converted to float64 (True=1.0, False=0.0) with NaN for missing
- **Strings**: Stored as UTF-8 fixed-length strings (default 100 chars) with separate mask for missing values
  - Fixed-length strings are required for SWMR compatibility
  - Longer strings are truncated, shorter strings are padded with spaces
  - Trailing whitespace is trimmed when loading data

### Index
- **All index types**: Converted to UTF-8 fixed-length strings for consistent storage
- **MultiIndex**: Each level stored separately with proper reconstruction metadata
- **Missing values**: Handled via mask arrays for all index levels
- **String length**: Configurable via `string_fixed_length` parameter (default: 100 characters)

## SWMR (Single Writer Multiple Reader) Support

pandas2hdf is designed for SWMR workflows where one process writes data while multiple processes read concurrently:

```python
# Writer process - CRITICAL: Follow this 3-step pattern
with h5py.File('data.h5', 'w', libver='latest') as f:
    # Step 1: Create all datasets BEFORE enabling SWMR
    group = f.create_group('data')
    save_series_new(group, initial_data, require_swmr=False)
    
    # Step 2: Enable SWMR mode
    f.swmr_mode = True
    
    # Step 3: Write/update data under SWMR
    save_series_append(group, new_data, require_swmr=True)
    f.flush()  # Make data visible to readers

# Reader processes  
with h5py.File('data.h5', 'r', swmr=True) as f:
    # ... read operations (automatically see new data after writer flushes)
```

### SWMR Requirements
- Use `libver='latest'` when creating files
- **Critical**: Create all datasets BEFORE enabling SWMR mode (datasets cannot be created under SWMR)
- Set `swmr_mode = True` on writer file handle AFTER creating datasets
- Use fixed-length strings for SWMR compatibility (variable-length strings cannot be written under SWMR)
- Use `require_swmr=True` for write operations under SWMR (validates SWMR is enabled)
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
- **String length**: Choose appropriate `string_fixed_length` based on your data (default: 100 chars)
- **SWMR**: Minimal overhead for concurrent reading

## Testing

Run the comprehensive test suite:

```bash
pytest tests/
```

The tests cover:
- Round-trip fidelity for all supported data types
- Fixed-length string behavior (padding, truncation, Unicode handling)
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
