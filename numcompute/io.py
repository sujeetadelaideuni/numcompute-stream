"""
io.py — CSV data loading and saving utilities.

This module is the data entry point for the NumCompute toolkit. It wraps
numpy.genfromtxt to provide a clean, consistent interface for loading tabular
CSV data into NumPy arrays, with support for custom delimiters, missing value
handling, streaming/chunked reads, and CSV export.

Author: Shaun D'souza (Person 1)
Module: numcompute.io
"""

import os
import io as _io
from typing import Generator

import numpy as np


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_csv(
    filepath: str,
    delimiter: str = ',',
    dtype: type = float,
    missing_values: str = '',
    filling_values: float = np.nan,
) -> np.ndarray:
    """
    Load a CSV file into a 2-D NumPy array.

    Uses ``numpy.genfromtxt`` under the hood. All missing cells (empty strings
    or the token specified by ``missing_values``) are filled with
    ``filling_values`` (default ``np.nan``).

    Parameters
    ----------
    filepath : str
        Path to the CSV file to load.
    delimiter : str, optional
        Column separator character. Default ``','``.
    dtype : data-type, optional
        NumPy dtype for the output array. Default ``float``.
    missing_values : str, optional
        Token that marks a missing cell in the file. Default ``''``
        (empty string — i.e. blank cells are treated as missing).
    filling_values : float, optional
        Scalar used to fill missing entries. Default ``np.nan``.

    Returns
    -------
    np.ndarray
        2-D array of shape ``(n_rows, n_cols)``.

    Raises
    ------
    FileNotFoundError
        If ``filepath`` does not exist on disk.
    ValueError
        If the file is empty (zero bytes or only whitespace).

    Notes
    -----
    Time complexity: O(n) where n is the total number of cells.
    Space complexity: O(n).

    Examples
    --------
    >>> X = load_csv("data.csv")
    >>> X = load_csv("data.tsv", delimiter="\\t")
    >>> X = load_csv("data.csv", missing_values="NA", filling_values=0.0)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"No such file or directory: '{filepath}'"
        )

    if os.path.getsize(filepath) == 0:
        raise ValueError(
            f"File is empty: '{filepath}'"
        )

    data = np.genfromtxt(
        filepath,
        delimiter=delimiter,
        dtype=dtype,
        missing_values=missing_values,
        filling_values=filling_values,
        autostrip=True,
    )

    # genfromtxt returns a 0-D or 1-D array for single-row/single-cell files
    if data.ndim == 0:
        raise ValueError(
            f"File is empty or could not be parsed: '{filepath}'"
        )
    if data.ndim == 1:
        # Single row — reshape to (1, n_cols)
        data = data.reshape(1, -1)

    return data


def load_csv_chunked(
    filepath: str,
    chunksize: int,
    delimiter: str = ',',
    dtype: type = float,
    missing_values: str = '',
    filling_values: float = np.nan,
) -> Generator[np.ndarray, None, None]:
    """
    Stream a large CSV file in fixed-size row chunks.

    Reads the file line-by-line so that the entire file never has to reside
    in memory at once. Each yielded chunk is a 2-D NumPy array containing at
    most ``chunksize`` rows.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.
    chunksize : int
        Maximum number of rows per yielded chunk. Must be >= 1.
    delimiter : str, optional
        Column separator character. Default ``','``.
    dtype : data-type, optional
        NumPy dtype for the output arrays. Default ``float``.
    missing_values : str, optional
        Token that marks a missing cell. Default ``''``.
    filling_values : float, optional
        Scalar used to fill missing entries. Default ``np.nan``.

    Yields
    ------
    np.ndarray
        2-D array of shape ``(≤chunksize, n_cols)``.

    Raises
    ------
    FileNotFoundError
        If ``filepath`` does not exist on disk.
    ValueError
        If ``chunksize`` is less than 1.

    Notes
    -----
    Time complexity: O(chunksize × n_cols) per chunk.
    Space complexity: O(chunksize × n_cols) — only one chunk in memory at a time.

    Examples
    --------
    >>> for chunk in load_csv_chunked("big_data.csv", chunksize=1000):
    ...     process(chunk)   # chunk.shape == (≤1000, n_cols)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"No such file or directory: '{filepath}'"
        )
    if chunksize < 1:
        raise ValueError(
            f"chunksize must be >= 1, got {chunksize}"
        )

    with open(filepath, 'r', encoding='utf-8') as fh:
        buffer: list[str] = []
        for line in fh:
            buffer.append(line)
            if len(buffer) == chunksize:
                yield _parse_lines(
                    buffer, delimiter, dtype, missing_values, filling_values
                )
                buffer = []
        # Yield any remaining lines
        if buffer:
            yield _parse_lines(
                buffer, delimiter, dtype, missing_values, filling_values
            )


def save_csv(
    array: np.ndarray,
    filepath: str,
    delimiter: str = ',',
    header: str = None,
) -> None:
    """
    Write a NumPy array to a CSV file.

    Uses ``numpy.savetxt`` under the hood. Optionally writes a header row as
    the first line of the file.

    Parameters
    ----------
    array : np.ndarray
        1-D or 2-D array to save. 3-D or higher arrays are not supported.
    filepath : str
        Destination file path. Parent directories must already exist.
    delimiter : str, optional
        Column separator character. Default ``','``.
    header : str, optional
        Header row string written as the first line
        (e.g. ``'col1,col2,col3'``). Default ``None`` (no header).

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If ``array`` is not 1-D or 2-D.

    Notes
    -----
    Time complexity: O(n) where n is the total number of cells.
    Space complexity: O(1) additional — numpy.savetxt streams rows to disk.

    Examples
    --------
    >>> save_csv(X, "output.csv")
    >>> save_csv(X, "output.csv", header="age,height,weight")
    >>> save_csv(X, "output.tsv", delimiter="\\t")
    """
    if array.ndim not in (1, 2):
        raise ValueError(
            f"array must be 1-D or 2-D, got {array.ndim}-D array "
            f"with shape {array.shape}."
        )

    np.savetxt(
        filepath,
        array,
        delimiter=delimiter,
        header=header if header is not None else '',
        comments='',
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _parse_lines(
    lines: list[str],
    delimiter: str,
    dtype: type,
    missing_values: str,
    filling_values: float,
) -> np.ndarray:
    """
    Parse a list of raw CSV text lines into a 2-D NumPy array.

    Parameters
    ----------
    lines : list of str
        Raw text lines from the CSV file (including newline characters).
    delimiter : str
        Column separator.
    dtype : data-type
        NumPy dtype for the output array.
    missing_values : str
        Token that marks a missing cell.
    filling_values : float
        Scalar used to fill missing entries.

    Returns
    -------
    np.ndarray
        2-D array of shape ``(len(lines), n_cols)``.
    """
    block = ''.join(lines)
    arr = np.genfromtxt(
        _io.StringIO(block),
        delimiter=delimiter,
        dtype=dtype,
        missing_values=missing_values,
        filling_values=filling_values,
        autostrip=True,
    )
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr
