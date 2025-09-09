"""Robust round-trip persistence between pandas and HDF5 with SWMR support."""

from .core import (
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

__version__ = "1.0.0"

__all__ = [
    "assert_swmr_on",
    "preallocate_series_layout",
    "save_series_new",
    "save_series_update",
    "save_series_append",
    "load_series",
    "preallocate_frame_layout",
    "save_frame_new",
    "save_frame_update",
    "save_frame_append",
    "load_frame",
]
