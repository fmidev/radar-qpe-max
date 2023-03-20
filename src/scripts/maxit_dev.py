import os
from glob import glob

import xarray as xr
import rioxarray
import pyart
import numpy as np
import matplotlib.pyplot as plt

from radproc.aliases import zh, lwe
from radproc.visual import plot_ppi
from radproc.radar import z_r_qpe


def basic_gatefilter(radar, field=zh):
    gatefilter = pyart.filters.GateFilter(radar)
    gatefilter.exclude_transition()
    gatefilter.exclude_masked(field)
    return gatefilter


def select_center(da, size=2048):
    xstart = int((da.x.size-size)/2)
    ystart = int((da.y.size-size)/2)
    return da[:, ystart:ystart+size, xstart:xstart+size].copy()


def precip_grid(radar, size=2048, resolution=250):
    gf = basic_gatefilter(radar)
    intermsize = int(size*2)
    grid_shape = (1, intermsize, intermsize)
    r_m = size*resolution*0.625 # includes reprojection margin
    grid_limits = ((0, 5000), (-r_m, r_m), (-r_m, r_m))
    grid = pyart.map.grid_from_radars(radar, gatefilters=gf,
                                      grid_shape=grid_shape,
                                      grid_limits=grid_limits,
                                      fields=[lwe])
    rds = grid.to_xarray().isel(z=0).set_index(x='lon', y='lat').reset_coords(drop=True)
    rda = rds.lwe_precipitation_rate
    rda.rio.write_crs(4326, inplace=True)
    rda3067 = rda.rio.reproject("epsg:3067", resolution=resolution, nodata=np.nan)
    rda3067 = select_center(rda3067, size=size)
    return rda3067#.reset_coords(drop=True)


if __name__ == '__main__':
    plt.close('all')
    win = 12
    datadir = os.path.expanduser('~/data/alakulma')
    #fpath0 = os.path.join(datadir, '202206030010_radar.raw.fmi_SITE=fivih_TASKS=PPI3-A.h5')
    #fpath1 = os.path.join(datadir, '202206030015_radar.raw.fmi_SITE=fivih_TASKS=PPI1-A.h5')
    #fpath2 = os.path.join(datadir, '202206030020_radar.raw.fmi_SITE=fivih_TASKS=PPI2-A.h5')
    fpaths = sorted(glob(os.path.join(datadir, '*-A.h5')))
    rdas = []
    for fpath in fpaths: #[fpath0, fpath1, fpath2]:
        radar = pyart.aux_io.read_odim_h5(fpath)
        z_r_qpe(radar)
        #axr = plot_ppi(radar, sweep=0, what=lwe)
        rdas.append(precip_grid(radar, size=512, resolution=1000))
    rda = xr.concat(rdas, 'time').fillna(0)
    del(rdas)
    accums = rda.rolling({'time': win}).sum()[win-1:]/12
    dat = accums.max('time')
    tstamp = accums.time[-1].item().strftime('%Y%m%d')
    dat.rio.to_raster(f"/tmp/{tstamp}_max.tif")
