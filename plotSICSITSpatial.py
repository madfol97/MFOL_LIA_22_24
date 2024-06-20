# %% Import packages
import numpy as np
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import xarray
import time
from matplotlib.gridspec import GridSpec
import os
import xesmf as xe
import pandas as pd
import cmocean
from piomass_hi import readPiomas
#  %%  Compute SI thickness average
start_time = time.time()

# Hist
allDatesHist = ['198001-198912','199001-199912','200001-200611']
allDatesHist2 = [str(year) for year in range(1981, 2001)]
allDatesHist3 = [str(year) for year in range(1981, 2001)]

i = 0
aiceSum = np.zeros((2400,3600))
hiSum = np.zeros((2400,3600))

for datesString in allDatesHist :
    cesm_a = xarray.open_dataset(r'/mnt/qanik/iHESP/EM1/hist/aice_hist/B.E.13.BHISTC5.ne120_t12.sehires38.003.sunway.cice.h.aice.' + datesString + '.nc', decode_times=True)
    cesm_hi = xarray.open_dataset(r'/mnt/qanik/iHESP/EM1/hist/hi_hist/B.E.13.BHISTC5.ne120_t12.sehires38.003.sunway.cice.h.hi.' + datesString + '.nc', decode_times=True)
    time_bounds = cesm_a['aice'].time[0].dt.year.item(), cesm_a['aice'].time[-1].dt.year.item()
    for year in range(time_bounds[0], time_bounds[1]):
        if year > 1980 and year < 2001:
            aiceSum += np.nan_to_num(cesm_a['aice'].sel(time=f'{year}-{10:02d}')[0,:,:], nan=0)
            hiSum += np.nan_to_num(cesm_hi['hi'].sel(time=f'{year}-{6:02d}')[0,:,:], nan=0)
            i += 1
    cesm_a.close()
    cesm_hi.close()

aiceSum2 = np.zeros((2400,3600))
hiSum2 = np.zeros((2400,3600))
i2 = 0
for datesString in allDatesHist2 :
    cesm_a = xarray.open_dataset(r'/mnt/qanik/iHESP/EM2/hist/b.e13.BHISTC5.ne120_t12.cesm-ihesp-hires1.0.30-1920-2100.002.cice.h.' + datesString + '-09.nc', decode_times=True)
    time_bounds = cesm_a['aice'].time[0].dt.year.item(), cesm_a['aice'].time[-1].dt.year.item()
    if time_bounds[0] == time_bounds[1] and cesm_a['aice'].time.shape[0] == 1:
        aiceSum2 += np.nan_to_num(cesm_a['aice'][0,:,:], nan=0)
        cesm_hi = xarray.open_dataset(r'/mnt/qanik/iHESP/EM2/hist/b.e13.BHISTC5.ne120_t12.cesm-ihesp-hires1.0.30-1920-2100.002.cice.h.' + datesString + '-05.nc', decode_times=True)
        hiSum2 += np.nan_to_num(cesm_hi['hi'][0,:,:], nan=0)
        i2 += 1
    cesm_a.close()
    cesm_hi.close()

aiceSum3 = np.zeros((2400,3600))
hiSum3 = np.zeros((2400,3600))
i3 = 0
for datesString in allDatesHist3 :
    cesm_a = xarray.open_dataset(r'/mnt/qanik/iHESP/EM3/hist/b.e13.BHISTC5.ne120_t12.cesm-ihesp-hires1.0.30-1920-2100.003.cice.h.' + datesString + '-09.nc', decode_times=True)
    time_bounds = cesm_a['aice'].time[0].dt.year.item(), cesm_a['aice'].time[-1].dt.year.item()
    if time_bounds[0] == time_bounds[1] and cesm_a['aice'].time.shape[0] == 1:
        aiceSum3 += np.nan_to_num(cesm_a['aice'][0,:,:], nan=0)
        cesm_hi = xarray.open_dataset(r'/mnt/qanik/iHESP/EM3/hist/b.e13.BHISTC5.ne120_t12.cesm-ihesp-hires1.0.30-1920-2100.003.cice.h.' + datesString + '-05.nc', decode_times=True)
        hiSum3 += np.nan_to_num(cesm_hi['hi'][0,:,:], nan=0)
        i3 += 1
    cesm_a.close()
    cesm_hi.close()

# EM mean
aice_hist_19802006 = aiceSum + aiceSum2 + aiceSum3
aice_hist_19802006 = aice_hist_19802006[:,:]/(i+i2+i3)
hi_hist_19802006 = hiSum + hiSum2 + hiSum3
hi_hist_19802006 = hi_hist_19802006[:,:]/(i+i2+i3)

print("Elapsed Time:", time.time() - start_time, "seconds")

# For proj
start_time = time.time()
allDatesHist2000 = ['200001-200611']
allDatesHist20003 = [str(year) for year in range(2000, 2006)]
allDatesProj = ['200601-201512','201601-202512','202601-203512','203601-204512','204601-205512','205601-206512','206601-207512','207601-208512','208601-209512','209601-210202']
allDatesProj3 = [str(year) for year in range(2006, 2101)]

i_20002020 = 0
aiceSumProj_20002020 = np.zeros((2400,3600))
hiSumProj_20002020 = np.zeros((2400,3600))
i_20202040 = 0
aiceSumProj_20202040 = np.zeros((2400,3600))
hiSumProj_20202040 = np.zeros((2400,3600))
i_20402060 = 0
aiceSumProj_20402060 = np.zeros((2400,3600))
hiSumProj_20402060 = np.zeros((2400,3600))
i_20802100 = 0
aiceSumProj_20802100 = np.zeros((2400,3600))
hiSumProj_20802100 = np.zeros((2400,3600))

# EM 1
for datesString in allDatesProj :
    cesm_a = xarray.open_dataset(r'/mnt/qanik/iHESP/EM1/proj/aice_proj/B.E.13.BRCP85C5CN.ne120_t12.sehires38.003.sunway.CN_OFF.cice.h.aice.' + datesString + '.nc', decode_times=True)
    cesm_hi = xarray.open_dataset(r'/mnt/qanik/iHESP/EM1/proj/hi_proj/B.E.13.BRCP85C5CN.ne120_t12.sehires38.003.sunway.CN_OFF.cice.h.hi.' + datesString + '.nc',decode_times=True)
    time_bounds = cesm_a['aice'].time[0].dt.year.item(), cesm_a['aice'].time[-1].dt.year.item()
    for year in range(time_bounds[0], time_bounds[1]):
        if year > 2000 and year < 2021:
            aiceSumProj_20002020 += np.nan_to_num(cesm_a['aice'].sel(time=f'{year}-{10:02d}')[0,:,:], nan=0)
            hiSumProj_20002020 += np.nan_to_num(cesm_hi['hi'].sel(time=f'{year}-{6:02d}')[0,:,:], nan=0)
            i_20002020 += 1
        elif year > 2020 and year < 2041:
            aiceSumProj_20202040 += np.nan_to_num(cesm_a['aice'].sel(time=f'{year}-{10:02d}')[0,:,:], nan=0)
            hiSumProj_20202040 += np.nan_to_num(cesm_hi['hi'].sel(time=f'{year}-{6:02d}')[0,:,:], nan=0)
            i_20202040 += 1
        elif year > 2040 and year < 2061:
            aiceSumProj_20402060 += np.nan_to_num(cesm_a['aice'].sel(time=f'{year}-{10:02d}')[0,:,:], nan=0)
            hiSumProj_20402060 += np.nan_to_num(cesm_hi['hi'].sel(time=f'{year}-{6:02d}')[0,:,:], nan=0)
            i_20402060 += 1
        elif year > 2080 and year < 2101:
            aiceSumProj_20802100 += np.nan_to_num(cesm_a['aice'].sel(time=f'{year}-{10:02d}')[0,:,:], nan=0)
            hiSumProj_20802100 += np.nan_to_num(cesm_hi['hi'].sel(time=f'{year}-{6:02d}')[0,:,:], nan=0)
            i_20802100 += 1
    cesm_a.close()
    cesm_hi.close()
for datesString in allDatesHist2000 :
    cesm_a = xarray.open_dataset(r'/mnt/qanik/iHESP/EM1/hist/aice_hist/B.E.13.BHISTC5.ne120_t12.sehires38.003.sunway.cice.h.aice.' + datesString + '.nc', decode_times=True)
    cesm_hi = xarray.open_dataset(r'/mnt/qanik/iHESP/EM1/hist/hi_hist/B.E.13.BHISTC5.ne120_t12.sehires38.003.sunway.cice.h.hi.' + datesString + '.nc', decode_times=True)
    time_bounds = cesm_a['aice'].time[0].dt.year.item(), cesm_a['aice'].time[-1].dt.year.item()
    for year in range(time_bounds[0], time_bounds[1]):
        if year > 2000:
            aiceSumProj_20002020 += np.nan_to_num(cesm_a['aice'].sel(time=f'{year}-{10:02d}')[0,:,:], nan=0)
            hiSumProj_20002020 += np.nan_to_num(cesm_hi['hi'].sel(time=f'{year}-{6:02d}')[0,:,:], nan=0)
            i_20002020 += 1
    cesm_a.close()
    cesm_hi.close()

i_20002020_2 = 0
aiceSumProj_20002020_2 = np.zeros((2400,3600))
hiSumProj_20002020_2 = np.zeros((2400,3600))
i_20202040_2 = 0
aiceSumProj_20202040_2 = np.zeros((2400,3600))
hiSumProj_20202040_2 = np.zeros((2400,3600))
i_20402060_2 = 0
aiceSumProj_20402060_2 = np.zeros((2400,3600))
hiSumProj_20402060_2 = np.zeros((2400,3600))
i_20802100_2 = 0
aiceSumProj_20802100_2 = np.zeros((2400,3600))
hiSumProj_20802100_2 = np.zeros((2400,3600))

# EM 2 - only one file
cesm_a = xarray.open_dataset(r'/mnt/qanik/iHESP/EM2/proj/aice_proj/b.e13.BRCP85C5.ne120_t12.cesm-ihesp-hires1.0.30.002.cice.h.aice.200601-210012.nc', decode_times=True)
cesm_hi = xarray.open_dataset(r'/mnt/qanik/iHESP/EM2/proj/hi_proj/b.e13.BRCP85C5.ne120_t12.cesm-ihesp-hires1.0.30.002.cice.h.hi.200601-210012.nc', decode_times=True)
time_bounds = cesm_a['aice'].time[0].dt.year.item(), cesm_a['aice'].time[-1].dt.year.item()
for year in range(time_bounds[0], time_bounds[1]):
    if year > 2000 and year < 2021:
        aiceSumProj_20002020_2 += np.nan_to_num(cesm_a['aice'].sel(time=f'{year}-{10:02d}')[0,:,:], nan=0)
        hiSumProj_20002020_2 += np.nan_to_num(cesm_hi['hi'].sel(time=f'{year}-{6:02d}')[0,:,:], nan=0)
        i_20002020_2 += 1
    elif year > 2020 and year < 2041:
        aiceSumProj_20202040_2 += np.nan_to_num(cesm_a['aice'].sel(time=f'{year}-{10:02d}')[0,:,:], nan=0)
        hiSumProj_20202040_2 += np.nan_to_num(cesm_hi['hi'].sel(time=f'{year}-{6:02d}')[0,:,:], nan=0)
        i_20202040_2 += 1
    elif year > 2040 and year < 2061:
        aiceSumProj_20402060_2 += np.nan_to_num(cesm_a['aice'].sel(time=f'{year}-{10:02d}')[0,:,:], nan=0)
        hiSumProj_20402060_2 += np.nan_to_num(cesm_hi['hi'].sel(time=f'{year}-{6:02d}')[0,:,:], nan=0)
        i_20402060_2 += 1
    elif year > 2080 and year < 2101:
        aiceSumProj_20802100_2 += np.nan_to_num(cesm_a['aice'].sel(time=f'{year}-{10:02d}')[0,:,:], nan=0)
        hiSumProj_20802100_2 += np.nan_to_num(cesm_hi['hi'].sel(time=f'{year}-{6:02d}')[0,:,:], nan=0)
        i_20802100_2 += 1
cesm_a.close()
cesm_hi.close()
for datesString in allDatesHist20003 :
    cesm_a = xarray.open_dataset(r'/mnt/qanik/iHESP/EM2/hist/b.e13.BHISTC5.ne120_t12.cesm-ihesp-hires1.0.30-1920-2100.002.cice.h.' + datesString + '-09.nc', decode_times=True)
    time_bounds = cesm_a['aice'].time[0].dt.year.item(), cesm_a['aice'].time[-1].dt.year.item()
    if time_bounds[0] == time_bounds[1] and cesm_a['aice'].time.shape[0] == 1:
        aiceSumProj_20002020_2 += np.nan_to_num(cesm_a['aice'][0,:,:], nan=0)
        cesm_hi = xarray.open_dataset(r'/mnt/qanik/iHESP/EM2/hist/b.e13.BHISTC5.ne120_t12.cesm-ihesp-hires1.0.30-1920-2100.002.cice.h.' + datesString + '-05.nc', decode_times=True)
        hiSumProj_20002020_2 += np.nan_to_num(cesm_hi['hi'][0,:,:], nan=0)
        i_20002020_2 += 1
    cesm_a.close()
    cesm_hi.close()

i_20002020_3 = 0
aiceSumProj_20002020_3 = np.zeros((2400,3600))
hiSumProj_20002020_3 = np.zeros((2400,3600))
i_20202040_3 = 0
aiceSumProj_20202040_3 = np.zeros((2400,3600))
hiSumProj_20202040_3 = np.zeros((2400,3600))
i_20402060_3 = 0
aiceSumProj_20402060_3 = np.zeros((2400,3600))
hiSumProj_20402060_3 = np.zeros((2400,3600))
i_20802100_3 = 0
aiceSumProj_20802100_3 = np.zeros((2400,3600))
hiSumProj_20802100_3 = np.zeros((2400,3600))

for datesString in allDatesProj3 :
    cesm_a = xarray.open_dataset(r'/mnt/qanik/iHESP/EM3/proj/b.e13.BRCP85C5.ne120_t12.cesm-ihesp-hires1.0.31.003.cice.h.' + datesString + '-09.nc', decode_times=True)
    cesm_hi = xarray.open_dataset(r'/mnt/qanik/iHESP/EM3/proj/b.e13.BRCP85C5.ne120_t12.cesm-ihesp-hires1.0.31.003.cice.h.' + datesString + '-05.nc', decode_times=True)
    time_bounds = cesm_a['aice'].time[0].dt.year.item(), cesm_a['aice'].time[-1].dt.year.item()
    if time_bounds[0] == time_bounds[1] and cesm_a['aice'].time.shape[0] == 1:
        if int(datesString)> 2000 and int(datesString) < 2021:
            aiceSumProj_20002020_3 += np.nan_to_num(cesm_a['aice'][0,:,:], nan=0)
            hiSumProj_20002020_3 += np.nan_to_num(cesm_hi['hi'][0,:,:], nan=0)
            i_20002020_3 += 1
        elif int(datesString) > 2020 and int(datesString) < 2041:
            aiceSumProj_20202040_3 += np.nan_to_num(cesm_a['aice'][0,:,:], nan=0)
            hiSumProj_20202040_3 += np.nan_to_num(cesm_hi['hi'][0,:,:], nan=0)
            i_20202040_3 += 1
        elif int(datesString) > 2040 and int(datesString) < 2061:
            aiceSumProj_20402060_3 += np.nan_to_num(cesm_a['aice'][0,:,:], nan=0)
            hiSumProj_20402060_3 += np.nan_to_num(cesm_hi['hi'][0,:,:], nan=0)
            i_20402060_3 += 1
        elif int(datesString) > 2080 and int(datesString) < 2101:
            aiceSumProj_20802100_3 += np.nan_to_num(cesm_a['aice'][0,:,:], nan=0)
            hiSumProj_20802100_3 += np.nan_to_num(cesm_hi['hi'][0,:,:], nan=0)
            i_20802100_3 += 1
    cesm_a.close()
    cesm_hi.close()

for datesString in allDatesHist20003 :
    cesm_a = xarray.open_dataset(r'/mnt/qanik/iHESP/EM3/hist/b.e13.BHISTC5.ne120_t12.cesm-ihesp-hires1.0.30-1920-2100.003.cice.h.' + datesString + '-09.nc', decode_times=True)
    cesm_hi = xarray.open_dataset(r'/mnt/qanik/iHESP/EM3/hist/b.e13.BHISTC5.ne120_t12.cesm-ihesp-hires1.0.30-1920-2100.003.cice.h.' + datesString + '-05.nc', decode_times=True)
    time_bounds = cesm_a['aice'].time[0].dt.year.item(), cesm_a['aice'].time[-1].dt.year.item()
    if time_bounds[0] == time_bounds[1] and cesm_a['aice'].time.shape[0] == 1:
        aiceSumProj_20002020_3 += np.nan_to_num(cesm_a['aice'][0,:,:], nan=0)
        hiSumProj_20002020_3 += np.nan_to_num(cesm_hi['hi'][0,:,:], nan=0)
        i_20002020_3 += 1
    cesm_a.close()
    cesm_hi.close()

# Compute average by decades
aice_hist_20002020 = aiceSumProj_20002020 + aiceSumProj_20002020_2 + aiceSumProj_20002020_3
aice_hist_20002020 = aice_hist_20002020[:,:]/(i_20002020+i_20002020_2+i_20002020_3)
hi_hist_20002020  = hiSumProj_20002020 + hiSumProj_20002020_2 + hiSumProj_20002020_3
hi_hist_20002020 = hi_hist_20002020[:,:]/(i_20002020+i_20002020_2+i_20002020_3)

aice_hist_20202040 = aiceSumProj_20202040 + aiceSumProj_20202040_2 + aiceSumProj_20202040_2
aice_hist_20202040 = aice_hist_20202040[:,:]/(i_20202040+i_20202040_2+i_20202040_3)
hi_hist_20202040  = hiSumProj_20202040 + hiSumProj_20202040_2 + hiSumProj_20202040_3
hi_hist_20202040 = hi_hist_20202040[:,:]/(i_20202040+i_20202040_2+i_20202040_3)

aice_hist_20402060 = aiceSumProj_20402060 + aiceSumProj_20402060_2 + aiceSumProj_20402060_2
aice_hist_20402060 = aice_hist_20402060[:,:]/(i_20402060+i_20402060_2+i_20402060_3)
hi_hist_20402060  = hiSumProj_20402060 + hiSumProj_20402060 + hiSumProj_20402060_3
hi_hist_20402060 = hi_hist_20402060[:,:]/(i_20402060+i_20402060_2+i_20402060_3)

aice_hist_20802100 = aiceSumProj_20802100 + aiceSumProj_20802100 + aiceSumProj_20802100_3
aice_hist_20802100 = aice_hist_20802100[:,:]/(i_20802100+i_20802100_2+i_20802100_3)
hi_hist_20802100  = hiSumProj_20802100 + hiSumProj_20802100_2 + hiSumProj_20802100_3
hi_hist_20802100 = hi_hist_20802100[:,:]/(i_20802100+i_20802100_2+i_20802100_3)


print("Elapsed Time:", time.time() - start_time, "seconds")

# %% Observations nsdic
allDatesObs1 = [str(year) for year in range(1981, 2001)]
allDatesObs2 = [str(year) for year in range(2001, 2021)]
i_obs_19812000 = 0
i_obs_20012020 = 0
aiceSumObs_19812000 = np.zeros((448,304))
aiceSumObs_20012020 = np.zeros((448,304))

grid = xarray.open_dataset( "/storage/mfol/obs/nsidc/NSIDC0771_LatLon_PS_N25km_v1.0.nc")
suffixesmarch = ['n07','n07','n07','n07','n07','n07','n07','n07','n07','f08','f08','f08','f08','f11','f11','f11','f11','f13','f13','f13','f13','f13','f13','f13','f13','f13','f13','f13','f13','f17','f17','f17','f17','f17','f17','f17','f17','f17','f17','f17','f17','f17','f17','f17','f17']
suffixessept = ['n07','n07','n07','n07','n07','n07','n07','n07','f08','f08','f08','f08','f08','f11','f11','f11','f11','f13','f13','f13','f13','f13','f13','f13','f13','f13','f13','f13','f13','f17','f17','f17','f17','f17','f17','f17','f17','f17','f17','f17','f17','f17','f17','f17','f17']
k = 2
for year in allDatesObs1 :
    nsidc = xarray.open_dataset(r'/storage/mfol/obs/nsidccdr/sept/seaice_conc_monthly_nh_'+str(year)+'09_'+ suffixessept[k] + '_v04r00.nc', decode_times=True)
    aiceseptVar = nsidc['cdr_seaice_conc_monthly'][0,:,:]
    aiceseptVar2 = np.where(grid['latitude'] > 85, 1.0, aiceseptVar)
    aiceseptVar3 = np.where(aiceseptVar2 > 1.0, np.nan, aiceseptVar2)
    aiceSumObs_19812000 += np.nan_to_num(aiceseptVar3)
    i_obs_19812000 += 1
    nsidc.close()
    k += 1

for year in allDatesObs2:
    nsidc2 = xarray.open_dataset(r'/storage/mfol/obs/nsidccdr/sept/seaice_conc_monthly_nh_'+str(year)+'09_'+ suffixessept[k] + '_v04r00.nc', decode_times=True)
    aiceseptVar4 = nsidc2['cdr_seaice_conc_monthly'][0,:,:]
    aiceseptVar5 = np.where(grid['latitude'] > 85, 1.0, aiceseptVar4)
    aiceseptVar6 = np.where(aiceseptVar5 > 1.0, np.nan, aiceseptVar5)
    aiceSumObs_20012020 += np.nan_to_num(aiceseptVar6)
    i_obs_20012020 += 1
    nsidc2.close()
    k += 1

# Compute average by decades
aiceObs_19812000 = aiceSumObs_19812000[:,:]/i_obs_19812000
aiceObs_20012020 = aiceSumObs_20012020[:,:]/i_obs_20012020

# %% Add PIOMAS MEAN OBS THICKNESS

directorydata = '/storage/mfol/obs/PIOMAS/'
years1 = [year for year in range(1981,2001)]
years2 = [year for year in range(2001,2021)]
# sit has a shape of (41 years ,12 months,120,360)
lats,lons,sitObs1 = readPiomas(directorydata,'thick',years1,0)
lats,lons,sitObs2 = readPiomas(directorydata,'thick',years2,0)
sitObs1 = np.nan_to_num(sitObs1,0)
sitObs2 = np.nan_to_num(sitObs2,0)
sitObs1Mean = np.nansum(sitObs1[:,4,:,:], axis=(0))/20
sitObs2Mean = np.nansum(sitObs2[:,4,:,:], axis=(0))/20

years = [year for year in range(1978,2021)]
lats,lons,sit = readPiomas(directorydata,'thick',years,0)
lats,lons,sic = readPiomas(directorydata,'sic',years,0)
# Already NH upper than 40N
sitMasked = np.where(sic[:,4,:,:] > 0.15, sit[:,4,:,:], np.nan)

# Get grid
gridPio = np.genfromtxt(directorydata + 'Thickness/' + 'grid.dat')
gridPio = np.reshape(gridPio,(gridPio.size))  
lonsPio = np.reshape(gridPio[:gridPio.size//2],(120,360))
latsPio = np.reshape(gridPio[gridPio.size//2:],(120,360))
gridPiomas = xarray.Dataset({
    'x': (('lat', 'lon'), lonsPio),
    'y': (('lat', 'lon'), latsPio)
})
gridPiomas = gridPiomas.rename({'y': 'lat', 'x': 'lon'})

# %% Figure 4 Plot 20 years average SIC and SIT for model and observation 

# Function to create subplot with Basemap
def create_subplot(ax, cmap, data, vmin, vmax, proj='npstere', extent=[-250, 250, 65, 90], i=0, obs=None, isObs=False):
    m = Basemap(ax=ax, projection=proj, boundinglat=extent[2], lon_0=-90, lat_0=80, resolution='l',
                width=2500000, height=2500000)
    x, y = m(cesm_grid.lon.values, cesm_grid.lat.values)
    xObs, yObs = m(grid['longitude'], grid['latitude'])
    xObsThickness, yObsThickness = m(lons,lats)
    if data.shape == xObs.shape: 
        x, y = xObs, yObs 
    elif data.shape == xObsThickness.shape: 
        x, y = xObsThickness, yObsThickness 

    # Draw MIZ if SIC
    if i==0 or i==1:
        levels = [15, 85]
        if isObs:
            m.contour(xObs, yObs, obs, levels=levels, colors=['red', 'cyan'], linestyles='solid', linewidths=2)
        else:
            m.contour(x, y, data, levels=levels, colors=['red', 'cyan'], linestyles='solid', linewidths=2)

    m.drawcoastlines(color='gray', linewidth=0.3)

    scat = m.pcolormesh(x, y, data, cmap=cmap, vmin=vmin, vmax=vmax)
    m.fillcontinents(color='lightgray')
    m.drawmapboundary(fill_color='lightgray')
    m.drawcountries(linewidth=0.5, linestyle='dotted', color='gray')
    
    return scat

# Load data
cesm_grid = xarray.open_dataset(r'/aos/home/mfol/Data/CESM/domain.ocn.tx0.1v2_090218.nc', decode_times=False)
cesm = xarray.open_dataset(r'/mnt/qanik/iHESP/EM1/hist/aice_hist/B.E.13.BHISTC5.ne120_t12.sehires38.003.sunway.cice.h.aice.185001-185912.nc', decode_times=True)
tarea = cesm['tarea'][:, :]
aice = cesm['aice'][0, :, :]

# Define data and parameters
aice_data = [aice_hist_19802006,aice_hist_20002020, aice_hist_20202040, aice_hist_20402060]
hi_data = [hi_hist_19802006, hi_hist_20002020, hi_hist_20202040, hi_hist_20402060]
observations_aice = [aiceObs_19812000[:,:]*100, aiceObs_20012020[:,:]*100]
observations_hi = [sitObs1Mean, sitObs2Mean]
cmaps = cmocean.cm.thermal
cmapsObs = cmocean.cm.balance
xlabels = ['Sept. A (%) - CESM1.3-HR', 'Sept. A (%) - CDR','May h (m) - CESM1.3-HR','May h (m) - PIOMAS']
ylabels = ['1981-2000','2001-2020', '2021-2040', '2041-2060']

cesm_grid = cesm_grid.rename({"xc": "lon", "yc": "lat"})

# Create subplot
fig = plt.figure(figsize=(16, 15))
plt.rc('font', size=14)
gs = GridSpec(4, 4)

for i in range(4):
    for j in range(4):
        ax = plt.subplot(gs[i, j])

        if j == 0:
            ax.set_ylabel(xlabels[i], fontsize=16)
        if i == 0:
            ax_cb1 = fig.add_axes([0.99, 0.5, 0.02, 0.5]) 
            if j < 3:
                scat = create_subplot(ax, cmaps, aice_data[j], 0, 100, proj='npstere', i=i, isObs=(False))
            else:
                scat = create_subplot(ax, cmaps, aice_data[j], 0, 100, proj='stere', extent=[-122, 120, 66, 90], i=i, isObs=(False))
            if j == 3:
                c = plt.colorbar(scat, ax=ax_cb1, orientation='vertical',fraction=1.00)
                c.set_label('Sept. A (%)', fontsize=16)
            ax.set_title(ylabels[j], fontsize=16)
            ax_cb1.yaxis.set_ticks([])
            ax_cb1.xaxis.set_ticks([])
            ax_cb1.spines['left'].set_color('white')
            ax_cb1.spines['right'].set_color('white')
            ax_cb1.spines['top'].set_color('white')
            ax_cb1.spines['bottom'].set_color('white')
        if i == 2:
            ax_cb2 = fig.add_axes([0.99, 0, 0.02, 0.5]) 
            if j < 3 and j > 1:
                print('Oui')
                scat = create_subplot(ax, cmaps, hi_data[j], 0,5, proj='npstere', i=i,  isObs=(False))
            elif j <2:
                scat = create_subplot(ax, cmaps, observations_hi[j], 0,5, proj='npstere', i=i,  isObs=(False))
            else:
                scat = create_subplot(ax, cmaps, hi_data[j], 0,5, proj='stere', extent=[-122, 120, 66, 90], i=i,  isObs=(False))
            
            if j== 3:
                c = plt.colorbar(scat, ax=ax_cb2, orientation='vertical', extend='max', fraction=1.00)
                c.set_label('May h (m)', fontsize=16)
            
            ax_cb2.yaxis.set_ticks([])
            ax_cb2.xaxis.set_ticks([])
            ax_cb2.spines['left'].set_color('white')
            ax_cb2.spines['right'].set_color('white')
            ax_cb2.spines['top'].set_color('white')
            ax_cb2.spines['bottom'].set_color('white')

        if i == 1:
            if j < 2:
                scat = create_subplot(ax, cmaps, observations_aice[j], 0,100, proj='npstere', i=i, obs=observations_aice[j], isObs=(True))
            else: 
                ax.set_axis_off() 
        if i == 3:
            if j < 2:
                scat = create_subplot(ax, cmaps, observations_hi[j], 0,5, proj='npstere', i=i, isObs=(False))
            else: 
                ax.set_axis_off() 

plt.tight_layout()
plt.show()


