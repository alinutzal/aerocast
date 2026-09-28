import numpy as np

from aerocast.grid_proj import IoapiGrid

# CMAQ_12US1 attrs, from the EQUATES CONC3D file header.
CONUS_12US1 = IoapiGrid(
    p_alp=33.0, p_bet=45.0, p_gam=-97.0, xcent=-97.0, ycent=40.0,
    xorig=-2556000.0, yorig=-1728000.0, xcell=12000.0, ycell=12000.0,
)


def test_rowcol_lonlat_round_trip():
    for row, col in [(0, 0), (148.5, 38.2), (298, 458), (157, 38)]:
        lon, lat = CONUS_12US1.rowcol_to_lonlat(row, col)
        row2, col2 = CONUS_12US1.lonlat_to_rowcol(lon, lat)
        assert np.isclose(row, row2, atol=1e-6)
        assert np.isclose(col, col2, atol=1e-6)


def test_lonlat_rowcol_round_trip():
    for lon, lat in [(-121.3, 38.0), (-118.24, 34.05), (-97.0, 40.0)]:
        row, col = CONUS_12US1.lonlat_to_rowcol(lon, lat)
        lon2, lat2 = CONUS_12US1.rowcol_to_lonlat(row, col)
        assert np.isclose(lon, lon2, atol=1e-6)
        assert np.isclose(lat, lat2, atol=1e-6)


def test_projection_origin_is_known_point():
    # XCENT/YCENT is the projection origin; XORIG/YORIG place it relative to the grid.
    lon, lat = CONUS_12US1.rowcol_to_lonlat(row=(0 - CONUS_12US1.yorig / CONUS_12US1.ycell) - 0.5,
                                             col=(0 - CONUS_12US1.xorig / CONUS_12US1.xcell) - 0.5)
    assert np.isclose(lon, CONUS_12US1.p_gam, atol=1e-6)
    assert np.isclose(lat, CONUS_12US1.ycent, atol=1e-6)


def test_subgrid_origin_matches_rowcol_to_lonlat():
    row0, col0 = 121, 2
    new_xorig, new_yorig = CONUS_12US1.subgrid_origin(row0, col0)
    sub = IoapiGrid(CONUS_12US1.p_alp, CONUS_12US1.p_bet, CONUS_12US1.p_gam,
                     CONUS_12US1.xcent, CONUS_12US1.ycent, new_xorig, new_yorig,
                     CONUS_12US1.xcell, CONUS_12US1.ycell)
    lon_full, lat_full = CONUS_12US1.rowcol_to_lonlat(row0 + 5, col0 + 7)
    lon_sub, lat_sub = sub.rowcol_to_lonlat(5, 7)
    assert np.isclose(lon_full, lon_sub, atol=1e-9)
    assert np.isclose(lat_full, lat_sub, atol=1e-9)
