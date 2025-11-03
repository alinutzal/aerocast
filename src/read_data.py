import xarray as xr
import numpy as np
import pandas as pd

# Load files
ds_conc = xr.open_dataset("out.combine_20181113.nc")
ds_meteo = xr.open_dataset("METCRO2D_20181113.nc")
ds_emis  = xr.open_dataset("egts_l.20181113.1.1km.baaqmd2018_newngc2.ncf")

# Helper to flatten variables
def flatten(var): return var.values.reshape(-1)

# Select 1st layer only if there are multiple (e.g., surface layer)
def get_surface(var):
    if 'LAY' in var.dims:
        return var.isel(LAY=0)
    return var

# Ensure consistent time steps
common_tsteps = min(
    ds_meteo.dims['TSTEP'],
    ds_conc.dims['TSTEP'],
    ds_emis.dims['TSTEP']
)

# Select shared region and time
def trim_and_flatten(var):
    var = get_surface(var)
    var = var.isel(TSTEP=slice(0, common_tsteps))
    return var.values.reshape(-1)

df = pd.DataFrame({
    'wind_speed': trim_and_flatten(ds_meteo['WSPD10']),
    'wind_dir':   trim_and_flatten(ds_meteo['WDIR10']),
    'temp':       trim_and_flatten(ds_meteo['TEMP2']),
    'no':         trim_and_flatten(ds_conc['NO']),
    'no2':        trim_and_flatten(ds_conc['NO2']),
    'pm25':       trim_and_flatten(ds_conc['PM25_CL']),
    'o3':         trim_and_flatten(ds_conc['O3']),

    # VOCs
    'alk1':       trim_and_flatten(ds_emis['ALK1']),
    'ole1':       trim_and_flatten(ds_emis['OLE1']),
    'aro1':       trim_and_flatten(ds_emis['ARO1']),
    'aro2':       trim_and_flatten(ds_emis['ARO2']),
    'terp':       trim_and_flatten(ds_emis['TERP']),
    'isop':       trim_and_flatten(ds_emis['ISOP']),
})


df.dropna(inplace=True)

print(df)