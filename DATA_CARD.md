# Data card: EQUATES pilot (July 2019, central California, 12 km)

A pilot dataset for AeroCast built from EPA's public EQUATES v1 CMAQ run: July 2019, surface
layer, a 72x72 read area (a 64x64 core box plus a 4-cell boundary ring) on the CMAQ_12US1
Lambert Conformal Conic grid, over the Bay Area, Sacramento Valley and San Joaquin Valley.
Config: `configs/data/equates_2019_07_ca12km.yaml`. Extracted by
`scripts/extract_equates_pilot.py`. Data lives outside the repo, on OSC project storage
(`/fs/ess/PLS0144/data/aerocast_equates/pilot/files/`), not in git.

**Output:** 31 days x 2 streams (`equates_conc_<date>.nc`, `equates_meteo_<date>.nc`) +
`static.nc`, 63 files, 110 MB total (49 MB concentrations, 62 MB meteorology, 80 KB static;
zlib-compressed NetCDF4-classic). Extraction wall time: ~29 min end to end.

## Sources

| Stream | Bucket / key | Format | Access |
|---|---|---|---|
| Concentrations (CONC3D) | `s3://epa-equates-v1/CMAQ_12US1/OUTPUT/CONC3D/COMBINE_CONC3D_v532_cb6r3_ae7_aq_WR413_MYR_STAGE_2019_12US1_201907.nc` | netCDF3 (64-bit offset, magic `CDF\x02`), 386 GB | public, `aws s3 --no-sign-request` or `s3fs(anon=True)` |
| Meteorology (MCIP) | `s3://cmas-equates/CMAQ_12US1/INPUT/2019/met/mcip_v51_wrf_v411_noltng/07/MCIPv51_WRFv411_noltng_12US1.35L.<YYYYMMDD>.tar` | tar of netCDF4/HDF5 members, ~7 GB/day | public, anonymous |
| Gridded emissions | `s3://cmas-equates/CMAQ_12US1/INPUT/2019/emis/cb6r3_ae6_20200131_MYR/premerged/nonpt/` | netCDF4, ~245 MB/day | public, anonymous, but **not used** - see Known gaps |

- Access date: 2026-09-28.
- Extraction method: never downloaded either source file whole.
  - CONC3D: header and variable byte offsets parsed via `kerchunk.netCDF3.NetCDF3ToZarr`
    (~1.2 s), then read as a lazy zarr/xarray view over HTTP range requests. Validated by
    comparing one hour's full-CONUS layer-1 O3 field (read directly, no subsetting) against
    the read-area subset for the same hour: exact match (`tests/test_equates_pilot.py::
    test_conc_subset_matches_full_field_read`).
  - MCIP: each day's ~7 GB tar was never downloaded; `tarfile` was pointed at a seekable
    `s3fs` file handle, which reads only the tar headers (512-byte blocks) until it finds the
    wanted member (`METCRO2D_<date>.nc4` or `GRIDCRO2D_<date>.nc4`), then `h5netcdf` reads
    that member's HDF5 structure directly from its byte range within the tar.
  - Per-file transfer was well under the 50 GB "stop and ask" threshold throughout: CONC3D
    reads pulled a few MB per variable-day (header + the read-area's chunks out of the
    386 GB object, 31 days x 4 variables in 5-55 s each). MCIP's 2D fields are stored as one
    HDF5 chunk per variable covering the full 459x299 grid, not spatially chunked, so a
    72x72-cell read still requires that variable's whole grid for that day - 6 of
    METCRO2D's 41 variables, out of a ~330 MB/day member, so on the order of 1-2 GB total
    for the month's meteorology, well under the threshold either way.

## Grid and box

CMAQ_12US1 grid: Lambert Conformal Conic, `GDTYP=2`, standard parallels 33N/45N
(`P_ALP`/`P_BET`), central meridian 97W (`P_GAM`), projection origin 97W/40N (`XCENT`/
`YCENT`), spherical earth radius 6,370,000 m (not WGS84 - the CMAQ/WRF convention), 12 km
cells (`XCELL`/`YCELL`), grid origin `XORIG=-2,556,000 m`, `YORIG=-1,728,000 m`, full grid
459 columns x 299 rows. Implemented in `src/aerocast/grid_proj.py` (`IoapiGrid`).

- **Core box:** rows 125-188, cols 6-69 (64 x 64 cells), of the 299 x 459 CMAQ_12US1 grid.
- **Read area (box + 4-cell ring):** rows 121-192, cols 2-73 (72 x 72 cells). This is the
  grid stored in every output file - there is no separate crop step back to 64 x 64; a
  model that only wants the core box can slice `[4:-4, 4:-4]` itself.
- **Center:** row 157, col 38 -> **38.883N, 121.629W**. This is shifted north from the
  literal 38.0N/121.3W in the original request: at that literal center, the Los Angeles
  basin's northern edge (Santa Clarita, ~34.42N/118.54W) fell almost exactly on the
  read-area boundary (~7 km clearance) rather than clearly outside it. The shifted center
  gives ~115 km (~9.6 cells) of clearance while still comfortably covering the Bay Area,
  Sacramento (and Redding, at the valley's north end) and Fresno; Bakersfield (San Joaquin
  Valley's south end) falls just outside the read area. See
  `data/pilot_checks/box_preview_v2.png` for the box plotted over state lines, and
  `tests/test_equates_pilot.py::test_los_angeles_clearly_outside_read_area`.
- **Corners of the read area** (cell centers, via `IoapiGrid.rowcol_to_lonlat`):

  | Corner | Row, col | Lat | Lon |
  |---|---|---|---|
  | SW | 121, 2 | 34.0301N | 124.8980W |
  | SE | 121, 73 | 35.9864N | 115.8181W |
  | NW | 192, 2 | 41.3288N | 127.9971W |
  | NE | 192, 73 | 43.5026N | 117.9958W |

  (The read area is a parallelogram in lat/lon, not a rectangle - it's a rectangle in the
  LCC projection. Matches `static.nc`'s LAT/LON min/max, which span the same four corners.)

## Variables and units

| Stream | File pattern | Variables | Units | Source var name |
|---|---|---|---|---|
| Concentrations | `equates_conc_<YYYYMMDD>.nc` | NO, NO2, O3, ATOTIJ | ppbV (NO/NO2/O3), ug/m3 (ATOTIJ) | same |
| Meteorology | `equates_meteo_<YYYYMMDD>.nc` | TEMP2, WSPD10, WDIR10, PBL, RGRND, Q2 | K, m/s, deg, m, W/m2, kg/kg | same (MCIP METCRO2D) |
| Static | `static.nc` | LAT, LON, HT, LWMASK | deg, deg, m, fraction | GRIDCRO2D |

- **ATOTIJ** (total Aitken + accumulation-mode aerosol mass, ug/m3) is EQUATES/CB6r3's
  PM2.5 analog - the counterpart to BAAQMD's SAPRC `PM25_CL`, but **not the same species
  definition**; the two should not be compared directly as PM2.5.
- Surface layer only (`LAY=0` of the source's 35 layers) is extracted; all output files have
  `NLAYS=1`.
- **Time convention:** hourly. All 27 CONC3D variables are documented in the EQUATES data
  dictionary (Table 10) as "Hourly 3D concentrations", read from a file named `CONC3D` (not
  `ACONC3D`) - by standard CMAQ CCTM convention this is an instantaneous snapshot at the top
  of each hour, not an hourly average. The data dictionary does not state this explicitly
  for this file, so treat it as inferred from naming convention, not confirmed metadata.
- **Time range:** July 2019, UTC. CONC3D's SDATE/STIME give July 1 00Z as record 0; the
  trailing record (Aug 1 00Z) is dropped, leaving 744 = 31 x 24 hourly records. MCIP's daily
  tars each hold 25 hourly records (the extra one is the next day's 00Z, for interpolation);
  only the first 24 are kept per day, so both streams cover exactly July 1 00Z - July 31 23Z
  with no relative time offset (`data.time_offset_hours: {meteo: 0, conc: 0}`).
- **Local solar time:** the box spans about 8 degrees of longitude (roughly a 32-minute
  difference in solar time edge to edge), so the pipeline's hour-of-day feature
  (`time:HOUR_SIN`/`time:HOUR_COS`) is computed per cell as UTC + longitude/15, not with a
  single grid-wide UTC offset (`data.local_solar_time: true`, `data.static_file: static.nc`,
  general in `src/aerocast/data.py`, not EQUATES-specific).

## Known gaps

- **Resolution:** 12 km cells. No near-road NOx titration or urban-plume structure below
  that scale - this run cannot represent street-level gradients the way a 1 km run
  (e.g. BAAQMD) can.
- **Emissions: not included in this pilot.** The task's assumed monthly gridded-emissions
  tars (`model_ready_emis_<year>_merged_nobeis_norwc_<month>_EQUATES_v1.0.tar`) do not exist
  for 2019 on either public bucket. What 2019 gridded emissions do exist
  (`s3://cmas-equates/CMAQ_12US1/INPUT/2019/emis/cb6r3_ae6_20200131_MYR/premerged/nonpt/`,
  individual daily netCDF4 files, not tarred) cover only 10 scattered July days (7/04-06,
  7/08-14), not the full month - so no complete forcing series can be built from them for
  this pilot. Point-source and residential-wood-combustion (RWC) emissions were not found
  on either bucket at all for 2019. Biogenic emissions are computed online inside the CMAQ
  run and were never archived separately. `configs/data/equates_2019_07_ca12km.yaml` has no
  `emis` entry (`files.emis: null`); the forcing channels use meteorology only.
- **Species mechanism:** CB6r3 (this run), not SAPRC (BAAQMD's mechanism) - emission and
  even some concentration species names and definitions differ (e.g. ATOTIJ vs PM25_CL,
  above) and should not be assumed equivalent across the two datasets. Emission channel
  lists are kept per-dataset in each config, as instructed, not mapped between mechanisms.
- **One month, one small domain:** July 2019 only, 72x72 cells. Not a substitute for a
  multi-year, full-domain evaluation.

## Citation

Per the EPA data dictionary's "How to Cite EQUATES Data" section
(`cmas-equates.s3.amazonaws.com/EQUATES_Data_Dictionary_for_CMAS_Data_Warehouses.html`):

- **DOI:** https://doi.org/10.15139/S3/F2KJSK
- **Website:** https://www.epa.gov/cmaq/EQUATES
- **Reference for EQUATES emissions:** https://doi.org/10.1016/j.dib.2023.109022
