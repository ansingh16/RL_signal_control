"""Shared fixtures: a small synthetic parquet shaped like the raw TfL files."""

import numpy as np
import pandas as pd
import pytest


def _make_raw_frame(n=500):
    """Build a raw-shaped frame with deterministic, zero-free signals.

    Mirrors the columns the processor reads from the real TfL parquet:
    ``Report Date`` (``YYYY-MM-DDT00:00:00``), ``Time Period Ending``
    (``HH:MM:SS``), and the string-typed ``Total Volume`` / ``Avg mph``. Values
    are deterministic so feature-engineering results are exactly predictable.
    """
    ts = pd.date_range("2022-01-01 00:14:00", periods=n, freq="15min")

    # Volume: a simple repeating ramp, never zero (so interpolation is a no-op).
    volume = 100 + (np.arange(n) % 50)
    # Speed: every third interval is congested (< 30 mph), the rest free-flowing.
    speed = np.where(np.arange(n) % 3 == 0, 20, 50)

    return pd.DataFrame(
        {
            "Site Name": "TEST/M25/SYN",
            "Report Date": ts.normalize().strftime("%Y-%m-%dT00:00:00"),
            "Time Period Ending": ts.strftime("%H:%M:%S"),
            "Time Interval": np.arange(n) % 96,
            "Avg mph": speed.astype(str),
            "Total Volume": volume.astype(str),
            "site_id": 9999,
        }
    )


@pytest.fixture
def raw_parquet(tmp_path):
    """Write the synthetic raw frame to a parquet and return its path."""
    path = tmp_path / "df_test.parquet"
    _make_raw_frame().to_parquet(path)
    return str(path)
