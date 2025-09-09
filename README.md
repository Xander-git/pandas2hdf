# pandas2hdf

Robust round-trip persistence between pandas Series/DataFrame and HDF5 with SWMR (Single Writer Multiple Readers) support.

## Features

- **Round-trip persistence**: Save and load pandas Series and DataFrames to/from HDF5 files while preserving data types, order, and missing values
- **SWMR support**: Full support for HDF5's Single Writer Multiple Readers mode for concurrent access
- **Type safety**: Comprehensive type hints with mypy strict mode compatibility
- **Flexible encoding**:
  - Numeric and boolean types → float64
  - String and categorical types → UTF-8 variable-length strings
  - Missing values handled via NaN (numeric) or mask arrays (strings)
- **Efficient storage**: Chunked, compressed datasets with preallocation support
- **Schema validation**: Ensures data consistency across write operations

## Installation

```bash
pip install pandas2hdf
```

For development:
```bash
pip install pandas2hdf[dev]
```

## Quick Start

### Basic Series Round-trip

```python
import pandas as pd
import h5py
from pandas2hdf import save_series_new, load_series

# Create a pandas Series
series = pd.Series([1, 2, 3, 4, 5], index=['a', 'b', 'c', 'd', 'e'], name='my_data')

# Save to HDF5
with h5py.File('data.h5', 'w') as f:
    group = f.create_group('series1')
    save_series_new(group, series, require_swmr=False)

# Load from HDF5
with h5py.File('data.h5', 'r') as f:
    loaded_series = load_series(f['series1'])
    print(loaded_series)
```

### DataFrame Operations

```python
import pandas as pd
import h5py
from pandas2hdf import save_frame_new, load_frame

# Create a DataFrame
df = pd.DataFrame({
    'integers': [1, 2, 3, 4, 5],
    'floats': [1.1, 2.2, 3.3, 4.4, 5.5],
    'strings': ['a', 'b', 'c', 'd', 'e'],
    'booleans': [True, False, True, False, True]
})

# Save to HDF5
with h5py.File('data.h5', 'w') as f:
    group = f.create_group('dataframe1')
    save_frame_new(group, df, require_swmr=False)

# Load from HDF5
with h5py.File('data.h5', 'r') as f:
    loaded_df = load_frame(f['dataframe1'])
    print(loaded_df)
```

### SWMR Workflow

```python
import pandas as pd
import h5py
from pandas2hdf import save_series_new, save_series_append, load_series

# Create file with SWMR mode
with h5py.File('swmr_data.h5', 'w', libver='latest') as f:
    group = f.create_group('timeseries')
    f.swmr_mode = True  # Enable SWMR mode
    
    # Initial data
    initial_data = pd.Series([1, 2, 3], name='values')
    save_series_new(group, initial_data, require_swmr=True)

# Append data (writer)
with h5py.File('swmr_data.h5', 'r+', swmr=True) as f:
    new_data = pd.Series([4, 5, 6], name='values')
    save_series_append(f['timeseries'], new_data, require_swmr=True)

# Concurrent reader
with h5py.File('swmr_data.h5', 'r', swmr=True) as f:
    series = load_series(f['timeseries'], require_swmr=True)
    print(series)  # Shows all 6 values
```

## API Reference

### Core Functions

#### Series Operations

- `preallocate_series_layout(g, s, *, dataset="values", index_dataset="index", chunks=(25,), compression="gzip", preallocate=100, require_swmr=True)` - Preallocate HDF5 layout without writing data
- `save_series_new(g, s, *, ...)` - Save a new Series, creating datasets or reusing preallocated layout
- `save_series_update(g, s, *, start=0, ...)` - Update existing Series data at specific position
- `save_series_append(g, s, *, ...)` - Append Series data at the end
- `load_series(g, *, dataset="values", index_dataset="index", require_swmr=False)` - Load a Series from HDF5

#### DataFrame Operations

- `preallocate_frame_layout(g, df, *, ...)` - Preallocate layout for DataFrame
- `save_frame_new(g, df, *, ...)` - Save a new DataFrame
- `save_frame_update(g, df, *, start=0, ...)` - Update existing DataFrame data
- `save_frame_append(g, df, *, ...)` - Append DataFrame data
- `load_frame(g, *, require_swmr=False)` - Load a DataFrame from HDF5

#### Utilities

- `assert_swmr_on(g)` - Assert that the HDF5 file is in SWMR mode

## Data Type Handling

| pandas dtype | HDF5 storage | Round-trip dtype |
|-------------|--------------|------------------|
| int32/64    | float64      | float64          |
| float32/64  | float64      | float64          |
| bool        | float64      | float64          |
| object/string | UTF-8 vlen  | object           |
| category    | UTF-8 vlen   | object (as strings) |
| datetime    | UTF-8 vlen   | object (ISO format) |

## HDF5 Layout

### Series Layout
```
/my_series
  attrs: {series_name, len, values_kind, index_kind, ...}
  /values        (float64 or vlen str)
  /values_mask   (uint8, only for strings)
  /index         (vlen str)
  /index_mask    (uint8)
```

### DataFrame Layout
```
/my_frame
  attrs: {column_order: JSON list}
  /index/values
  /index/values_mask
  /columns/
    /column1/...  (series layout)
    /column2/...  (series layout)
```

## Development

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/pandas2hdf.git
cd pandas2hdf

# Install with development dependencies
pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=pandas2hdf --cov-report=html

# Run specific test file
pytest tests/test_pandas2hdf.py::TestSeriesRoundTrip
```

### Code Quality

```bash
# Format code
black src tests

# Lint
ruff check src tests

# Type checking
mypy src
```

## Requirements

- Python ≥ 3.10
- pandas ≥ 1.5.0
- numpy ≥ 1.21.0
- h5py ≥ 3.7.0

## License

MIT License - see LICENSE file for details.
