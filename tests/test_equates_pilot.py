"""Tests for the EQUATES pilot dataset (configs/data/equates_2019_07_ca12km.yaml).

Most of these need either the extracted pilot files (outside the repo, on OSC project
storage) or live network access to the public EQUATES S3 buckets, so they skip cleanly
when that's not available instead of failing CI.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from aerocast.config import load_config
from aerocast.data import available_days, load_hourly
from aerocast.grid_proj import IoapiGrid

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "data" / "equates_2019_07_ca12km.yaml"

CMAQ_12US1 = IoapiGrid(
    p_alp=33.0, p_bet=45.0, p_gam=-97.0, xcent=-97.0, ycent=40.0,
    xorig=-2556000.0, yorig=-1728000.0, xcell=12000.0, ycell=12000.0,
)


def _load_pilot_cfg():
    cfg = load_config(CONFIG_PATH)
    return cfg, cfg["equates"]["box"]


def test_box_rowcol_lonlat_round_trip():
    """The pilot box's row/col indices round-trip through lat/lon (the "compute row/col
    indices" step, and a check that configs/data/equates_2019_07_ca12km.yaml's box block
    still matches the grid math it was derived from)."""
    _, box = _load_pilot_cfg()
    row0, col0, size, ring = box["row0"], box["col0"], box["size"], box["ring"]
    for row, col in [(row0, col0), (row0 + size - 1, col0 + size - 1),
                      (row0 - ring, col0 - ring), (row0 + size - 1 + ring, col0 + size - 1 + ring)]:
        lon, lat = CMAQ_12US1.rowcol_to_lonlat(row, col)
        row2, col2 = CMAQ_12US1.lonlat_to_rowcol(lon, lat)
        assert np.isclose(row, row2, atol=1e-6)
        assert np.isclose(col, col2, atol=1e-6)


def test_los_angeles_clearly_outside_read_area():
    """LA basin must be clearly outside the 72x72 read area, not cut by its edge."""
    _, box = _load_pilot_cfg()
    row0, col0, size, ring = box["row0"], box["col0"], box["size"], box["ring"]
    ring_row0, ring_col0 = row0 - ring, col0 - ring
    ring_row1, ring_col1 = row0 + size + ring, col0 + size + ring

    # Santa Clarita: the LA basin's northernmost extent.
    la_row, la_col = CMAQ_12US1.lonlat_to_rowcol(-118.54, 34.42)
    inside = (ring_row0 <= la_row < ring_row1) and (ring_col0 <= la_col < ring_col1)
    clearance_rows = ring_row0 - la_row
    # Not inside, and outside with a real margin (>= 5 cells, 60 km), not right on the edge.
    assert not inside
    assert clearance_rows >= 5, f"LA basin only {clearance_rows:.1f} cells from the read-area edge"


@pytest.mark.skipif(not CONFIG_PATH.exists(), reason="pilot config not found")
def test_loader_two_day_subset():
    """The real Phase 1-2 loader reads the EQUATES config end to end (meteo + conc, no
    emis) over the first two extracted days, with per-cell local solar time."""
    cfg = load_config(CONFIG_PATH)
    data_cfg, features_cfg = cfg["data"], cfg["features"]
    if not Path(data_cfg["data_dir"]).exists():
        pytest.skip(f"extracted data not found at {data_cfg['data_dir']}")
    days = available_days(data_cfg)[:2]
    if len(days) < 2:
        pytest.skip("fewer than 2 extracted days available")

    hd = load_hourly(data_cfg, features_cfg, days)
    assert hd.state.shape[0] == 48  # 2 days x 24 h
    assert hd.state.shape[2:] == (72, 72)
    assert set(hd.targets) == {"NO2", "O3", "Ox"}
    assert np.allclose(hd.targets["Ox"], hd.targets["NO2"] + hd.targets["O3"])
    # local solar time must vary across columns (~8 deg of longitude across the box).
    hour_sin = hd.forcing[0, hd.channels["forcing"].index("time:HOUR_SIN")]
    assert not np.allclose(hour_sin[:, 0], hour_sin[:, -1])


@pytest.mark.slow
def test_conc_subset_matches_full_field_read():
    """One hour's full layer-1 O3 field, read directly from the source S3 object, must
    exactly match the read-area subset (validates the kerchunk byte-range read, not just
    its speed). Hits the public epa-equates-v1 bucket; needs network access."""
    pytest.importorskip("s3fs")
    pytest.importorskip("kerchunk")
    from extract_equates_pilot import build_conc_zarr_view, validate_conc_subset

    _, box = _load_pilot_cfg()
    row0 = box["row0"] - box["ring"]
    row1 = box["row0"] + box["size"] + box["ring"]
    col0 = box["col0"] - box["ring"]
    col1 = box["col0"] + box["size"] + box["ring"]

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        ds = build_conc_zarr_view(Path(tmp) / "refs.json")
        validate_conc_subset(ds, row0, row1, col0, col1, hour_index=0)
