"""Lat/lon <-> row/col for an IOAPI Lambert Conformal Conic grid.

IOAPI stores the projection as global attributes (GDTYP, P_ALP, P_BET, P_GAM, XCENT,
YCENT, XORIG, YORIG, XCELL, YCELL); see the CMAQ I/O API docs. CMAQ/WRF use a spherical
earth of radius 6370000 m rather than WGS84, so the CRS is built with +R= rather than a
named datum.
"""
from dataclasses import dataclass

import pyproj

EARTH_RADIUS_M = 6370000.0


@dataclass(frozen=True)
class IoapiGrid:
    p_alp: float   # first standard parallel
    p_bet: float   # second standard parallel
    p_gam: float   # central meridian
    xcent: float   # projection origin longitude
    ycent: float   # projection origin latitude
    xorig: float   # meters, west edge of column 0 relative to the projection origin
    yorig: float   # meters, south edge of row 0 relative to the projection origin
    xcell: float   # meters per column
    ycell: float   # meters per row

    @classmethod
    def from_attrs(cls, attrs):
        """From an IOAPI/CMAQ NetCDF file's global attributes (as a dict or xarray .attrs)."""
        if int(attrs["GDTYP"]) != 2:
            raise ValueError(f"Only Lambert Conformal Conic (GDTYP=2) is supported, got GDTYP={attrs['GDTYP']}")
        return cls(
            p_alp=float(attrs["P_ALP"]), p_bet=float(attrs["P_BET"]), p_gam=float(attrs["P_GAM"]),
            xcent=float(attrs["XCENT"]), ycent=float(attrs["YCENT"]),
            xorig=float(attrs["XORIG"]), yorig=float(attrs["YORIG"]),
            xcell=float(attrs["XCELL"]), ycell=float(attrs["YCELL"]),
        )

    @property
    def crs(self):
        return pyproj.CRS.from_proj4(
            f"+proj=lcc +lat_1={self.p_alp} +lat_2={self.p_bet} +lat_0={self.ycent} "
            f"+lon_0={self.p_gam} +x_0=0 +y_0=0 +R={EARTH_RADIUS_M} +units=m +no_defs"
        )

    def _to_proj(self):
        return pyproj.Transformer.from_crs("EPSG:4326", self.crs, always_xy=True)

    def _to_lonlat(self):
        return pyproj.Transformer.from_crs(self.crs, "EPSG:4326", always_xy=True)

    def lonlat_to_rowcol(self, lon, lat):
        """Fractional (row, col); cell (row, col)'s center is at integer (row, col)."""
        x, y = self._to_proj().transform(lon, lat)
        col = (x - self.xorig) / self.xcell - 0.5
        row = (y - self.yorig) / self.ycell - 0.5
        return row, col

    def rowcol_to_lonlat(self, row, col):
        """(lon, lat) of the center of cell (row, col) (may be fractional)."""
        x = self.xorig + (col + 0.5) * self.xcell
        y = self.yorig + (row + 0.5) * self.ycell
        return self._to_lonlat().transform(x, y)

    def subgrid_origin(self, row0, col0):
        """XORIG/YORIG for a cropped grid whose row/col 0 is this grid's (row0, col0)."""
        return self.xorig + col0 * self.xcell, self.yorig + row0 * self.ycell
