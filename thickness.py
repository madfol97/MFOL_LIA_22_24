# %% Import packages
import numpy as np
import matplotlib.pyplot as plt
import xarray
import cartopy.crs as ccrs
import scipy.io.matlab as matlab
import matplotlib.path as mpath
import cartopy.feature as cfeature
from piomass_hi import readPiomas

# %% Get grids and masks
cesm_grid = xarray.open_dataset(r'/aos/home/mfol/Data/CESM/domain.ocn.tx0.1v2_090218.nc', decode_times=False)
lia_mask_QEI = xarray.open_dataarray(r'./masked_data_QEI23May.nc')
lia_mask_CAA = xarray.open_dataarray(r'./masked_data_CAA23May.nc')
lia_mask_LIAN = xarray.open_dataarray(r'/aos/home/mfol/Results/IHESP/masked_data_LIANorth.nc')

regions = [
    {'name': 'LIA-N','maskData': lia_mask_LIAN,'mask': None},
    {'name': 'QEI', 'maskData': lia_mask_QEI,'mask': None},
    {'name': 'CAA-S','maskData': lia_mask_CAA,'mask': None}
    ]
for region in regions: 
    lia = xarray.where((region['maskData'] == 0) | (region['maskData'].isnull()), 0, 1) # 0 everywhere, 1 in ROI
    liaSumValues = lia.sum() # compute number of pixels in mask
    mask = xarray.where((region['maskData'] == 0) | (region['maskData'].isnull()), 1, 0) # 1 everywhere, 0 in ROI
    region['mask'] = mask

# Define color code for each dataset
colorHR = '#2576af'
colorLR = '#37CAEC'
colorLE = '#c63e47'
colorObs = 'black'

# %%  COMPUTE THICKNESS FREQUENCY DISTRIBUTION
# From aicen001,2,3,4,5, filter out the 15% sic, then call computeThicknessDistribution
# which translates the aicen00x categories into frequencies (aicen00x/aice). 
# This gives the frequency of sea ice that is into each thickness bin. 
# Computes the frequencies for each period for each ensemble and does the em mean.

# At every grid point compute the thickness distribution
def computeThicknessDistribution(aice,aicen1,aicen2,aicen3,aicen4,aicen5):
    lat_min, lat_max = 30.98, 90
    mask = np.logical_and((cesm_grid.yc >= lat_min), (cesm_grid.yc <= lat_max))

    freqencies = np.empty((5,2400,3600))
    
    hiTotcutmsk = np.ma.masked_where(~mask, aice)
    hi_Tot_msk = np.where(hiTotcutmsk==0, np.nan, hiTotcutmsk)    
    index = 0
    for hi in [aicen1, aicen2, aicen3, aicen4, aicen5]:
        freq = []
        hi_cutmsk= np.ma.masked_where(~mask, hi)
        hi_mskd = np.where(hi_cutmsk==0, np.nan, hi_cutmsk[:,:])
        freq=(hi_mskd[:,:]/hi_Tot_msk[:,:])
        freqencies[index,:,:] = freq
        index += 1
    return freqencies


# Hist
allDatesHist = ['192001-192912','193001-193912','194001-194912','195001-195912','196001-196912','197001-197912','198001-198912','199001-199912','200001-200611']
allDatesHist3 = [str(year) for year in range(1921, 2001)]
threshold = 15

# EM 1
hiSum1 = [np.empty((2400,3600)),np.empty((2400,3600)),np.empty((2400,3600)),np.empty((2400,3600)),np.empty((2400,3600)),np.empty((2400,3600))]
i1 = 0
hiSum20C1 = [np.empty((2400,3600)),np.empty((2400,3600)),np.empty((2400,3600)),np.empty((2400,3600)),np.empty((2400,3600)),np.empty((2400,3600))]
i20C1 = 0
for datesString in allDatesHist :
    aice = xarray.open_dataset(r'/mnt/qanik/iHESP/EM1/hist/aice_hist/B.E.13.BHISTC5.ne120_t12.sehires38.003.sunway.cice.h.aice.' + datesString + '.nc', decode_times=True)
    cesm_01 = xarray.open_dataset(r'/storage/mfol/iHESP-HRCESM/EM1/hist/aicehCat/B.E.13.BHISTC5.ne120_t12.sehires38.003.sunway.cice.h.aicen001.' + datesString + '.nc', decode_times=True)
    cesm_02 = xarray.open_dataset(r'/storage/mfol/iHESP-HRCESM/EM1/hist/aicehCat/B.E.13.BHISTC5.ne120_t12.sehires38.003.sunway.cice.h.aicen002.' + datesString + '.nc', decode_times=True)
    cesm_03 = xarray.open_dataset(r'/storage/mfol/iHESP-HRCESM/EM1/hist/aicehCat/B.E.13.BHISTC5.ne120_t12.sehires38.003.sunway.cice.h.aicen003.' + datesString + '.nc', decode_times=True)
    cesm_04 = xarray.open_dataset(r'/storage/mfol/iHESP-HRCESM/EM1/hist/aicehCat/B.E.13.BHISTC5.ne120_t12.sehires38.003.sunway.cice.h.aicen004.' + datesString + '.nc', decode_times=True)
    cesm_05 = xarray.open_dataset(r'/storage/mfol/iHESP-HRCESM/EM1/hist/aicehCat/B.E.13.BHISTC5.ne120_t12.sehires38.003.sunway.cice.h.aicen005.' + datesString + '.nc', decode_times=True)
    time_bounds = aice['aice'].time[0].dt.year.item(), aice['aice'].time[-1].dt.year.item()
    for year in range(time_bounds[0], time_bounds[1]):
        if year > 1980 and year < 2001:
            aiceSel = aice.sel(time=f'{year}-{6:02d}')['aice']
            aice1Sel = cesm_01.sel(time=f'{year}-{6:02d}')['aicen001']
            aice2Sel = cesm_02.sel(time=f'{year}-{6:02d}')['aicen002']
            aice3Sel = cesm_03.sel(time=f'{year}-{6:02d}')['aicen003']
            aice4Sel = cesm_04.sel(time=f'{year}-{6:02d}')['aicen004']
            aice5Sel = cesm_05.sel(time=f'{year}-{6:02d}')['aicen005']
            hiwice = np.where(aiceSel[0,:,:] > threshold, aiceSel[0,:,:],  np.nan)
            hiwice1 = np.where(aiceSel[0,:,:] > threshold, aice1Sel[0,:,:],  np.nan)
            hiwice2 = np.where(aiceSel[0,:,:] > threshold, aice2Sel[0,:,:],  np.nan)
            hiwice3 = np.where(aiceSel[0,:,:] > threshold, aice3Sel[0,:,:],  np.nan)
            hiwice4 = np.where(aiceSel[0,:,:] > threshold, aice4Sel[0,:,:],  np.nan)
            hiwice5 = np.where(aiceSel[0,:,:] > threshold, aice5Sel[0,:,:],  np.nan)
            
            freq = computeThicknessDistribution(hiwice, hiwice1,hiwice2,hiwice3,hiwice4,hiwice5)
            
            hiSum1[0] += np.nan_to_num(freq[0,:,:], nan=0)
            hiSum1[1] += np.nan_to_num(freq[1,:,:], nan=0)
            hiSum1[2] += np.nan_to_num(freq[2,:,:], nan=0)
            hiSum1[3] += np.nan_to_num(freq[3,:,:], nan=0)
            hiSum1[4] += np.nan_to_num(freq[4,:,:], nan=0)
            # hiSum1[5] += np.nan_to_num(hiwice, nan=0)
            i1 += 1
        elif year > 1920 and year < 1981:
            aiceSel = aice.sel(time=f'{year}-{6:02d}')['aice']
            aice1Sel = cesm_01.sel(time=f'{year}-{6:02d}')['aicen001']
            aice2Sel = cesm_02.sel(time=f'{year}-{6:02d}')['aicen002']
            aice3Sel = cesm_03.sel(time=f'{year}-{6:02d}')['aicen003']
            aice4Sel = cesm_04.sel(time=f'{year}-{6:02d}')['aicen004']
            aice5Sel = cesm_05.sel(time=f'{year}-{6:02d}')['aicen005']
            hiwice = np.where(aiceSel[0,:,:] > threshold, aiceSel[0,:,:], np.nan)
            hiwice1 = np.where(aiceSel[0,:,:] > threshold, aice1Sel[0,:,:], np.nan)
            hiwice2 = np.where(aiceSel[0,:,:] > threshold, aice2Sel[0,:,:],np.nan)
            hiwice3 = np.where(aiceSel[0,:,:] > threshold, aice3Sel[0,:,:], np.nan)
            hiwice4 = np.where(aiceSel[0,:,:] > threshold, aice4Sel[0,:,:], np.nan)
            hiwice5 = np.where(aiceSel[0,:,:] > threshold, aice5Sel[0,:,:],np.nan)
            freq = computeThicknessDistribution(hiwice, hiwice1,hiwice2,hiwice3,hiwice4,hiwice5)

            hiSum20C1[0] += freq[0,:,:]
            hiSum20C1[1] += freq[1,:,:]
            hiSum20C1[2] += freq[2,:,:]
            hiSum20C1[3] += freq[3,:,:]
            hiSum20C1[4] += freq[4,:,:]
            
            # hiSum20C1[5] += np.nan_to_num(hiwice, nan=0)
            i20C1 += 1
    aice.close()
    cesm_01.close()
    cesm_02.close()
    cesm_03.close()
    cesm_04.close()
    cesm_05.close()

# # EM 3
hiSum3 = [np.empty((2400,3600)),np.empty((2400,3600)),np.empty((2400,3600)),np.empty((2400,3600)),np.empty((2400,3600)),np.empty((2400,3600))]
i3 = 0
hiSum20C3 = [np.empty((2400,3600)),np.empty((2400,3600)),np.empty((2400,3600)),np.empty((2400,3600)),np.empty((2400,3600)),np.empty((2400,3600))]
i20C3 = 0
for datesString in allDatesHist3 :
    cesm_a = xarray.open_dataset(r'/mnt/qanik/iHESP/EM3/hist/b.e13.BHISTC5.ne120_t12.cesm-ihesp-hires1.0.30-1920-2100.003.cice.h.' + datesString + '-05.nc', decode_times=True)
    time_bounds = cesm_a['hi'].time[0].dt.year.item(), cesm_a['hi'].time[-1].dt.year.item()
    year = cesm_a['hi'].time[0].dt.year.item()
    if time_bounds[0] == time_bounds[1] and cesm_a['hi'].time.shape[0] == 1:
        if year > 1980 and year < 2001:
            hiwice = np.where(cesm_a['aice'][0,:,:] > threshold, cesm_a['aice'][0,:,:], np.nan)
            hiwice1 = np.where(cesm_a['aice'][0,:,:] > threshold, cesm_a['aicen001'][0,:,:],np.nan)
            hiwice2 = np.where(cesm_a['aice'][0,:,:] > threshold, cesm_a['aicen002'][0,:,:],np.nan)
            hiwice3 = np.where(cesm_a['aice'][0,:,:] > threshold, cesm_a['aicen003'][0,:,:], np.nan)
            hiwice4 = np.where(cesm_a['aice'][0,:,:] > threshold, cesm_a['aicen004'][0,:,:],np.nan)
            hiwice5 = np.where(cesm_a['aice'][0,:,:] > threshold, cesm_a['aicen005'][0,:,:], np.nan)
            freq = computeThicknessDistribution(hiwice, hiwice1,hiwice2,hiwice3,hiwice4,hiwice5)

            hiSum3[0] +=  freq[0,:,:]
            hiSum3[1] +=  freq[1,:,:]
            hiSum3[2] += freq[2,:,:]
            hiSum3[3] +=  freq[3,:,:]
            hiSum3[4] +=  freq[4,:,:]
            # hiSum3[5] += freq[0,:,:]
            i3 += 1
        elif year > 1920 and year < 1981:
            hiwice = np.where(cesm_a['aice'][0,:,:] > threshold, cesm_a['aice'][0,:,:],np.nan)
            hiwice1 = np.where(cesm_a['aice'][0,:,:] > threshold, cesm_a['aicen001'][0,:,:],np.nan)
            hiwice2 = np.where(cesm_a['aice'][0,:,:] > threshold, cesm_a['aicen002'][0,:,:],np.nan)
            hiwice3 = np.where(cesm_a['aice'][0,:,:] > threshold, cesm_a['aicen003'][0,:,:],np.nan)
            hiwice4 = np.where(cesm_a['aice'][0,:,:] > threshold, cesm_a['aicen004'][0,:,:], np.nan)
            hiwice5 = np.where(cesm_a['aice'][0,:,:] > threshold, cesm_a['aicen005'][0,:,:], np.nan)
            freq = computeThicknessDistribution(hiwice, hiwice1,hiwice2,hiwice3,hiwice4,hiwice5)

            hiSum20C3[0] += freq[0,:,:]
            hiSum20C3[1] += freq[1,:,:]
            hiSum20C3[2] += freq[2,:,:]
            hiSum20C3[3] += freq[3,:,:]
            hiSum20C3[4] += freq[4,:,:]
            # hiSum20C3[5] += np.nan_to_num(hiwice, nan=0)
            i20C3 += 1
    cesm_a.close()

hi_hist_19802006 = []
hi_hist_19201980 = []
for j, elem in enumerate(hiSum3):
    hi_hist_19802006.append((hiSum3[j] + hiSum1[j])/(i3+i1))
for j, elem in enumerate(hiSum20C3):
    hi_hist_19201980.append((hiSum20C3[j] + hiSum20C1[j])/(i20C3+ i20C1))


# # # Proj
allDatesHist20003 = [str(year) for year in range(2000, 2006)]
allDatesProj3 = [str(year) for year in range(2006, 2101)]
allDatesProj1 = ['200601-201512','201601-202512','202601-203512','203601-204512','204601-205512','205601-206512','206601-207512','207601-208512','208601-209512','209601-210202']
allDatesHist20001 = ['200001-200611']

# # E1
hiSumProj_20002020_1 = [np.empty((2400,3600)),np.empty((2400,3600)),np.empty((2400,3600)),np.empty((2400,3600)),np.empty((2400,3600)),np.empty((2400,3600))]
i_20002020_1 = 0
hiSumProj_20202040_1 = [np.empty((2400,3600)),np.empty((2400,3600)),np.empty((2400,3600)),np.empty((2400,3600)),np.empty((2400,3600)),np.empty((2400,3600))]
i_20202040_1 = 0
hiSumProj_20402060_1 = [np.empty((2400,3600)),np.empty((2400,3600)),np.empty((2400,3600)),np.empty((2400,3600)),np.empty((2400,3600)),np.empty((2400,3600))]
i_20402060_1 = 0
hiSumProj_20602080_1 = [np.empty((2400,3600)),np.empty((2400,3600)),np.empty((2400,3600)),np.empty((2400,3600)),np.empty((2400,3600)),np.empty((2400,3600))]
i_20602080_1 = 0
hiSumProj_20802100_1 = [np.empty((2400,3600)),np.empty((2400,3600)),np.empty((2400,3600)),np.empty((2400,3600)),np.empty((2400,3600)),np.empty((2400,3600))]
i_20802100_1 = 0

for datesString in allDatesProj1 :
    aice = xarray.open_dataset(r'/mnt/qanik/iHESP/EM1/proj/aice_proj/B.E.13.BRCP85C5CN.ne120_t12.sehires38.003.sunway.CN_OFF.cice.h.aice.' + datesString + '.nc', decode_times=True)
    cesm_01 = xarray.open_dataset(r'/storage/mfol/iHESP-HRCESM/EM1/proj/aicehCat/B.E.13.BRCP85C5CN.ne120_t12.sehires38.003.sunway.CN_OFF.cice.h.aicen001.' + datesString + '.nc', decode_times=True)
    cesm_02 = xarray.open_dataset(r'/storage/mfol/iHESP-HRCESM/EM1/proj/aicehCat/B.E.13.BRCP85C5CN.ne120_t12.sehires38.003.sunway.CN_OFF.cice.h.aicen002.' + datesString + '.nc', decode_times=True)
    cesm_03 = xarray.open_dataset(r'/storage/mfol/iHESP-HRCESM/EM1/proj/aicehCat/B.E.13.BRCP85C5CN.ne120_t12.sehires38.003.sunway.CN_OFF.cice.h.aicen003.' + datesString + '.nc', decode_times=True)
    cesm_04 = xarray.open_dataset(r'/storage/mfol/iHESP-HRCESM/EM1/proj/aicehCat/B.E.13.BRCP85C5CN.ne120_t12.sehires38.003.sunway.CN_OFF.cice.h.aicen004.' + datesString + '.nc', decode_times=True)
    cesm_05 = xarray.open_dataset(r'/storage/mfol/iHESP-HRCESM/EM1/proj/aicehCat/B.E.13.BRCP85C5CN.ne120_t12.sehires38.003.sunway.CN_OFF.cice.h.aicen005.' + datesString + '.nc', decode_times=True)
    time_bounds = aice['aice'].time[0].dt.year.item(), aice['aice'].time[-1].dt.year.item()
    for year in range(time_bounds[0], time_bounds[1]): 
        aiceSel = aice.sel(time=f'{year}-{6:02d}')['aice']
        aice1Sel = cesm_01.sel(time=f'{year}-{6:02d}')['aicen001']
        aice2Sel = cesm_02.sel(time=f'{year}-{6:02d}')['aicen002']
        aice3Sel = cesm_03.sel(time=f'{year}-{6:02d}')['aicen003']
        aice4Sel = cesm_04.sel(time=f'{year}-{6:02d}')['aicen004']
        aice5Sel = cesm_05.sel(time=f'{year}-{6:02d}')['aicen005']
        hiwice = np.where(aiceSel[0,:,:] > threshold, aiceSel[0,:,:],np.nan)
        hiwice1 = np.where(aiceSel[0,:,:] > threshold, aice1Sel[0,:,:],np.nan)
        hiwice2 = np.where(aiceSel[0,:,:] > threshold, aice2Sel[0,:,:], np.nan)
        hiwice3 = np.where(aiceSel[0,:,:] > threshold, aice3Sel[0,:,:], np.nan)
        hiwice4 = np.where(aiceSel[0,:,:] > threshold, aice4Sel[0,:,:], np.nan)
        hiwice5 = np.where(aiceSel[0,:,:] > threshold, aice5Sel[0,:,:], np.nan)
        freq = computeThicknessDistribution(hiwice, hiwice1,hiwice2,hiwice3,hiwice4,hiwice5)

        if year> 2000 and year < 2021:
            hiSumProj_20002020_1[0] += freq[0,:,:]
            hiSumProj_20002020_1[1] += freq[1,:,:]
            hiSumProj_20002020_1[2] += freq[2,:,:]
            hiSumProj_20002020_1[3] += freq[3,:,:]
            hiSumProj_20002020_1[4] += freq[4,:,:]
            i_20002020_1 += 1
        elif year > 2020 and year < 2041:
            hiSumProj_20202040_1[0] += freq[0,:,:]
            hiSumProj_20202040_1[1] += freq[1,:,:]
            hiSumProj_20202040_1[2] += freq[2,:,:]
            hiSumProj_20202040_1[3] += freq[3,:,:]
            hiSumProj_20202040_1[4] += freq[4,:,:]
            i_20202040_1 += 1
        elif year > 2040 and year < 2061:
            hiSumProj_20402060_1[0] += freq[0,:,:]
            hiSumProj_20402060_1[1] += freq[1,:,:]
            hiSumProj_20402060_1[2] += freq[2,:,:]
            hiSumProj_20402060_1[3] += freq[3,:,:]
            hiSumProj_20402060_1[4] += freq[4,:,:]
            i_20402060_1 += 1
        elif year > 2060 and year < 2081:
            hiSumProj_20602080_1[0] += freq[0,:,:]
            hiSumProj_20602080_1[1] += freq[1,:,:]
            hiSumProj_20602080_1[2] += freq[2,:,:]
            hiSumProj_20602080_1[3] += freq[3,:,:]
            hiSumProj_20602080_1[4] += freq[4,:,:]
            i_20602080_1 += 1
        elif year > 2080 and year < 2101:
            hiSumProj_20802100_1[0] += freq[0,:,:]
            hiSumProj_20802100_1[1] += freq[1,:,:]
            hiSumProj_20802100_1[2] += freq[2,:,:]
            hiSumProj_20802100_1[3] += freq[3,:,:]
            hiSumProj_20802100_1[4] += freq[4,:,:]
            i_20802100_1 += 1
    aice.close()
    cesm_01.close()
    cesm_02.close()
    cesm_03.close()
    cesm_04.close()
    cesm_05.close()

for datesString in allDatesHist20001 :
    aice = xarray.open_dataset(r'/mnt/qanik/iHESP/EM1/hist/aice_hist/B.E.13.BHISTC5.ne120_t12.sehires38.003.sunway.cice.h.aice.' + datesString + '.nc', decode_times=True)
    cesm_01 = xarray.open_dataset(r'/storage/mfol/iHESP-HRCESM/EM1/hist/aicehCat/B.E.13.BHISTC5.ne120_t12.sehires38.003.sunway.cice.h.aicen001.' + datesString + '.nc', decode_times=True)
    cesm_02 = xarray.open_dataset(r'/storage/mfol/iHESP-HRCESM/EM1/hist/aicehCat/B.E.13.BHISTC5.ne120_t12.sehires38.003.sunway.cice.h.aicen002.' + datesString + '.nc', decode_times=True)
    cesm_03 = xarray.open_dataset(r'/storage/mfol/iHESP-HRCESM/EM1/hist/aicehCat/B.E.13.BHISTC5.ne120_t12.sehires38.003.sunway.cice.h.aicen003.' + datesString + '.nc', decode_times=True)
    cesm_04 = xarray.open_dataset(r'/storage/mfol/iHESP-HRCESM/EM1/hist/aicehCat/B.E.13.BHISTC5.ne120_t12.sehires38.003.sunway.cice.h.aicen004.' + datesString + '.nc', decode_times=True)
    cesm_05 = xarray.open_dataset(r'/storage/mfol/iHESP-HRCESM/EM1/hist/aicehCat/B.E.13.BHISTC5.ne120_t12.sehires38.003.sunway.cice.h.aicen005.' + datesString + '.nc', decode_times=True)
    time_bounds = aice['aice'].time[0].dt.year.item(), aice['aice'].time[-1].dt.year.item()
    for year in range(time_bounds[0], time_bounds[1]):
        if year > 2000 and year < 2020:
            aiceSel = aice.sel(time=f'{year}-{6:02d}')['aice']
            aice1Sel = cesm_01.sel(time=f'{year}-{6:02d}')['aicen001']
            aice2Sel = cesm_02.sel(time=f'{year}-{6:02d}')['aicen002']
            aice3Sel = cesm_03.sel(time=f'{year}-{6:02d}')['aicen003']
            aice4Sel = cesm_04.sel(time=f'{year}-{6:02d}')['aicen004']
            aice5Sel = cesm_05.sel(time=f'{year}-{6:02d}')['aicen005']
            hiwice = np.where(aiceSel[0,:,:] > threshold, aiceSel[0,:,:], np.nan)
            hiwice1 = np.where(aiceSel[0,:,:] > threshold, aice1Sel[0,:,:], np.nan)
            hiwice2 = np.where(aiceSel[0,:,:] > threshold, aice2Sel[0,:,:],np.nan)
            hiwice3 = np.where(aiceSel[0,:,:] > threshold, aice3Sel[0,:,:], np.nan)
            hiwice4 = np.where(aiceSel[0,:,:] > threshold, aice4Sel[0,:,:], np.nan)
            hiwice5 = np.where(aiceSel[0,:,:] > threshold, aice5Sel[0,:,:], np.nan)
            freq = computeThicknessDistribution(hiwice, hiwice1,hiwice2,hiwice3,hiwice4,hiwice5)

            hiSumProj_20002020_1[0] += freq[0,:,:]
            hiSumProj_20002020_1[1] += freq[1,:,:]
            hiSumProj_20002020_1[2] += freq[2,:,:]
            hiSumProj_20002020_1[3] += freq[3,:,:]
            hiSumProj_20002020_1[4] += freq[4,:,:]
            i_20002020_1 += 1
    aice.close()
    cesm_01.close()
    cesm_02.close()
    cesm_03.close()
    cesm_04.close()
    cesm_05.close()

# E3
hiSumProj_20002020_3 = [np.empty((2400,3600)),np.empty((2400,3600)),np.empty((2400,3600)),np.empty((2400,3600)),np.empty((2400,3600)),np.empty((2400,3600))]
i_20002020_3 = 0
hiSumProj_20202040_3 = [np.empty((2400,3600)),np.empty((2400,3600)),np.empty((2400,3600)),np.empty((2400,3600)),np.empty((2400,3600)),np.empty((2400,3600))]
i_20202040_3 = 0
hiSumProj_20402060_3 = [np.empty((2400,3600)),np.empty((2400,3600)),np.empty((2400,3600)),np.empty((2400,3600)),np.empty((2400,3600)),np.empty((2400,3600))]
i_20402060_3 = 0
hiSumProj_20602080_3 = [np.empty((2400,3600)),np.empty((2400,3600)),np.empty((2400,3600)),np.empty((2400,3600)),np.empty((2400,3600)),np.empty((2400,3600))]
i_20602080_3 = 0
hiSumProj_20802100_3 = [np.empty((2400,3600)),np.empty((2400,3600)),np.empty((2400,3600)),np.empty((2400,3600)),np.empty((2400,3600)),np.empty((2400,3600))]
i_20802100_3 = 0

for datesString in allDatesProj3 :
    cesm_a = xarray.open_dataset(r'/mnt/qanik/iHESP/EM3/proj/b.e13.BRCP85C5.ne120_t12.cesm-ihesp-hires1.0.31.003.cice.h.' + datesString + '-05.nc', decode_times=True)
    time_bounds = cesm_a['aice'].time[0].dt.year.item(), cesm_a['aice'].time[-1].dt.year.item()
    if time_bounds[0] == time_bounds[1] and cesm_a['aice'].time.shape[0] == 1:
        hiwice = np.where(cesm_a['aice'][0,:,:] > threshold, cesm_a['aice'][0,:,:],np.nan)
        hiwice1 = np.where(cesm_a['aice'][0,:,:] > threshold, cesm_a['aicen001'][0,:,:],np.nan)
        hiwice2 = np.where(cesm_a['aice'][0,:,:] > threshold, cesm_a['aicen002'][0,:,:], np.nan)
        hiwice3 = np.where(cesm_a['aice'][0,:,:] > threshold, cesm_a['aicen003'][0,:,:], np.nan)
        hiwice4 = np.where(cesm_a['aice'][0,:,:] > threshold, cesm_a['aicen004'][0,:,:], np.nan)
        hiwice5 = np.where(cesm_a['aice'][0,:,:] > threshold, cesm_a['aicen005'][0,:,:], np.nan)
        freq = computeThicknessDistribution(hiwice, hiwice1,hiwice2,hiwice3,hiwice4,hiwice5)
        
        if int(datesString)> 2000 and int(datesString) < 2021:
            hiSumProj_20002020_3[0] += freq[0,:,:]
            hiSumProj_20002020_3[1] += freq[1,:,:]
            hiSumProj_20002020_3[2] += freq[2,:,:]
            hiSumProj_20002020_3[3] += freq[3,:,:]
            hiSumProj_20002020_3[4] += freq[4,:,:]
            i_20002020_3 += 1
        elif int(datesString) > 2020 and int(datesString) < 2041:
            hiSumProj_20202040_3[0] += freq[0,:,:]
            hiSumProj_20202040_3[1] += freq[1,:,:]
            hiSumProj_20202040_3[2] += freq[2,:,:]
            hiSumProj_20202040_3[3] += freq[3,:,:]
            hiSumProj_20202040_3[4] += freq[4,:,:]
            i_20202040_3 += 1
        elif int(datesString) > 2040 and int(datesString) < 2061:
            hiSumProj_20402060_3[0] += freq[0,:,:]
            hiSumProj_20402060_3[1] += freq[1,:,:]
            hiSumProj_20402060_3[2] += freq[2,:,:]
            hiSumProj_20402060_3[3] += freq[3,:,:]
            hiSumProj_20402060_3[4] += freq[4,:,:]
            i_20402060_3 += 1
        elif int(datesString) > 2060 and int(datesString) < 2081:
            hiSumProj_20602080_3[0] += freq[0,:,:]
            hiSumProj_20602080_3[1] += freq[1,:,:]
            hiSumProj_20602080_3[2] += freq[2,:,:]
            hiSumProj_20602080_3[3] += freq[3,:,:]
            hiSumProj_20602080_3[4] += freq[4,:,:]
            i_20602080_3 += 1
        elif int(datesString) > 2080 and int(datesString) < 2101:
            hiSumProj_20802100_3[0] += freq[0,:,:]
            hiSumProj_20802100_3[1] += freq[1,:,:]
            hiSumProj_20802100_3[2] += freq[2,:,:]
            hiSumProj_20802100_3[3] += freq[3,:,:]
            hiSumProj_20802100_3[4] += freq[4,:,:]
            i_20802100_3 += 1
    cesm_a.close()

for datesString in allDatesHist20003 :
    cesm_a = xarray.open_dataset(r'/mnt/qanik/iHESP/EM3/hist/b.e13.BHISTC5.ne120_t12.cesm-ihesp-hires1.0.30-1920-2100.003.cice.h.' + datesString + '-05.nc', decode_times=True)
    time_bounds = cesm_a['aice'].time[0].dt.year.item(), cesm_a['aice'].time[-1].dt.year.item()
    if time_bounds[0] == time_bounds[1] and cesm_a['aice'].time.shape[0] == 1:
        hiwice = np.where(cesm_a['aice'][0,:,:] > threshold, cesm_a['aice'][0,:,:], np.nan)
        hiwice1 = np.where(cesm_a['aice'][0,:,:] > threshold, cesm_a['aicen001'][0,:,:], np.nan)
        hiwice2 = np.where(cesm_a['aice'][0,:,:] > threshold, cesm_a['aicen002'][0,:,:], np.nan)
        hiwice3 = np.where(cesm_a['aice'][0,:,:] > threshold, cesm_a['aicen003'][0,:,:],np.nan)
        hiwice4 = np.where(cesm_a['aice'][0,:,:] > threshold, cesm_a['aicen004'][0,:,:],np.nan)
        hiwice5 = np.where(cesm_a['aice'][0,:,:] > threshold, cesm_a['aicen005'][0,:,:], np.nan)
        freq = computeThicknessDistribution(hiwice, hiwice1,hiwice2,hiwice3,hiwice4,hiwice5)
        
        hiSumProj_20002020_3[0] += freq[0,:,:]
        hiSumProj_20002020_3[1] += freq[1,:,:]
        hiSumProj_20002020_3[2] += freq[2,:,:]
        hiSumProj_20002020_3[3] += freq[3,:,:]
        hiSumProj_20002020_3[4] += freq[4,:,:]
        i_20002020_3 += 1
    cesm_a.close()
    
# # Compute average by decades
hi_hist_20002020 = []
for k, elem in enumerate(hiSumProj_20002020_3):
    hi_hist_20002020.append((elem + hiSumProj_20002020_1[k])/(i_20002020_3+i_20002020_1))
hi_hist_20202040 = []
for k, elem in enumerate(hiSumProj_20202040_3):
    hi_hist_20202040.append((elem+ hiSumProj_20202040_1[k])/(i_20202040_3 + i_20202040_1))
hi_hist_20402060 = []
for k, elem in enumerate(hiSumProj_20402060_3):
    hi_hist_20402060.append((elem+ hiSumProj_20402060_1[k])/(i_20402060_3+i_20402060_1))
hi_hist_20602080 = []
for k, elem in enumerate(hiSumProj_20602080_3):
    hi_hist_20602080.append((elem+hiSumProj_20602080_1[k])/(i_20602080_3+i_20602080_1))
hi_hist_20802100 = []
for k, elem in enumerate(hiSumProj_20802100_3):
    hi_hist_20802100.append((elem+hiSumProj_20802100_1[k])/(i_20802100_3+i_20802100_1))

# %% compute SIT frequency on average pan arctic wise.
hi_data = [hi_hist_19201980, hi_hist_19802006, hi_hist_20002020, hi_hist_20202040, hi_hist_20402060,hi_hist_20602080, hi_hist_20802100]

bin_edges = [0,0.64,1.39,2.47,4.57,6]
bin_middle = [0.64/2,(1.39-0.64)/2 + 0.64,(2.47-1.39)/2+1.39,(4.57-2.47)/2+2.47,(6-4.57)/2 +4.57]
binwidth = [0.64,1.39-0.64,2.47-1.39,4.57-2.47, 6-4.57]


# Get PIOMAS thickness distribution from gice variable
directorydata = '/storage/mfol/obs/PIOMAS/'
years = [year for year in range(1981,2001)]
# SIT distribut5ion gice has a shape of (41 years ,12 months,120,360)
lats,lons,sit = readPiomas(directorydata,'thick',years,0)
lats,lons,sic = readPiomas(directorydata,'sic',years,0)
lats,lons,gice = readPiomas(directorydata,'gice',years,0)
sitMasked19802000 = np.empty((12,120,360))
for i in range(12):
    sitMasked19802000[i,:,:] = np.nanmean(np.where(sic[:,4,:,:] > 0.15, gice[:,4,i,:,:], np.nan),axis=(0))

years = [year for year in range(2001,2021)]
lats,lons,sit = readPiomas(directorydata,'thick',years,0)
lats,lons,sic = readPiomas(directorydata,'sic',years,0)
lats,lons,gice = readPiomas(directorydata,'gice',years,0)
sitMasked20002020 = np.empty((12,120,360))
for i in range(12):
    sitMasked20002020[i,:,:] = np.nanmean(np.where(sic[:,4,:,:] > 0.15, gice[:,4,i,:,:], np.nan),axis=(0))

# Mask distribution from PIOMAS to regions to compare with CESM1.3-HR
coordinatesQEI = [(360-117, 76.6),(360-116.1,76.66),(360-115,76.4),(360-112,75.4),(360-107,75.4),(360-106,75.3),(360-105.68,75.8),(360-103.65,75.9),(360-101.75,75.7),(360-98.7,75.8),(360-97.73,76.5),
                  (360-96.4,76.67),(360-94.38,76.43),(360-90.55,76.6),(360-89.5,76.55),
                  (360-87.11,76.58),(360-82.90,77.09),(360-83.5,78.4),(360-87.32, 78.17),(360-88.84, 78.2),(360-92.8,80.5),(360-96.3, 80.3),(360-99.05, 80.10),(360-100.0, 79.9),
                  (360-103.78, 79.35), (360-105.5, 79.2),(360-110.4, 78.75),(360-113.10, 78.3),(360-114.3, 78.08),(360-115.06, 77.95),(360-116.47, 77.56),
                  (360-117, 76.6)]
coordinatesCAA = [(360-128.19, 69.0),(360-110.58,66.0),(360-95.56,66.5),(360-86.2,67.02),(360-82.7,71.0),
                  (360-81.9, 73.7),(360-81.89, 74.52),(360-91.67,74.79),(360-91,75.55),
                   (360-91.69,76.4),(360-94.38,76.43),(360-96.4,76.67),(360-97.73,76.5),(360-98.7,75.8),(360-101.75,75.7),(360-103.65,75.9),
                   (360-105.68,75.8),(360-106,75.3),(360-107,75.4),(360-112,75.4),(360-115,76.4),(360-116.1,76.66),(360-120.2,76.6),(360-122.0,76.2),
                   (360-124.19, 74.32),(360-123.2, 73.28),(360-125.4, 72.18),(360-128.19, 70.16)]
          
# LIA North from DeRepentigny, P., L.B. Tremblay, R. Newton, and S. Pfirman, (2016), 
# Patterns of sea ice retreat in the transition to a seasonally ice-free Arctic. 
# Journal of Climate, DOI: 10.1175/JCLI-D-15-0733.1. For the SITU system.
# I used LIAIndices.mat -> XY2LatLon(blia,alia)
coordLIANewton = matlab.loadmat('/aos/home/mfol/Results/IHESP/LIALatLonEdges.mat')
coordinatesLIANorth = []
for idx, elem in enumerate(coordLIANewton['lon']): 
    coordinatesLIANorth.append((360+elem[0], coordLIANewton['lat'][idx][0]))
coordinatesLIANorth.append(coordinatesLIANorth[0])
coordinateMasks = [coordinatesLIANorth, coordinatesQEI, coordinatesCAA]
masks = []
for coord in coordinateMasks:
    # Get polygon from coordinates
    # Change the polygon to coordinatesQEI or coordinatesCAA
    lon, lat = zip(*coord)
    path_data = []
    for i in range(len(lon)):
        if i == 0:
            path_data.append((mpath.Path.MOVETO, (lon[i], lat[i])))
        else:
            path_data.append((mpath.Path.LINETO, (lon[i], lat[i])))

    path_data.append((mpath.Path.CLOSEPOLY, (lon[0], lat[0])))
    codes, verts = zip(*path_data)
    path = mpath.Path(verts, codes)
    vertices = path.vertices
    contour_path = mpath.Path(vertices, codes[:len(vertices)])

    # Create mask outside of polygon and common continent
    grid = np.genfromtxt(directorydata + 'Thickness/' + 'grid.dat')
    grid = np.reshape(grid,(grid.size)) 
     ### Define Lat/Lon
    lon = grid[:grid.size//2]   
    lons = np.reshape(lon,(120,360))
    lon_flat = np.ravel(lons)
    lat = grid[grid.size//2:]
    lats = np.reshape(lat,(120,360))
    lat_flat = np.ravel(lats)
    mask = path.contains_points(np.column_stack((lon_flat, lat_flat))).reshape((120,360))
    masks.append(mask)
    
    # See masks on PIOMAS grid
    fig, ax = plt.subplots(figsize=(3,3), subplot_kw={'projection': ccrs.NorthPolarStereo(central_longitude=-90)})
    scat = ax.pcolormesh(lons, lats,mask, transform=ccrs.PlateCarree())
    ax.set_extent([-140, -60, 65, 90], crs=ccrs.PlateCarree()) # CAA

# %% Compute per region masks and plot Figure 6
xlabels = ['LIA-N', 'QEI', 'CAA-S']
fig, axs = plt.subplots(3,1, figsize=(10, 15))
plt.rc('font', size=14)
colors=['grey','#009392','#9CCB86','#C3D791','#EEB479','#E88471','#CF597E']
labels=['1921-1980','1981-2000', '2001-2020', '2021-2040', '2041-2060', '2061-2080', '2081-2100']

obs = [sitMasked19802000, sitMasked20002020]
obsLabels = ['PIOMAS: 1981-2000','PIOMAS: 2001-2020']


bin_edges = [0,0.64,1.39,2.47,4.57,6]
bin_middle = [0.64/2,(1.39-0.64)/2 + 0.64,(2.47-1.39)/2+1.39,(4.57-2.47)/2+2.47,(6.1)]
binwidth = [0.64,1.39-0.64,2.47-1.39,4.57-2.47, 6-4.57]
# I cap the higher thickness together
bin_middleObs = [0,0.26,0.71,1.46,2.61,4.23, 6.39]#,9.10,12.39,16.24,20.62,25.49]

for j, region in enumerate(regions): 
    thickness_histogramsRegion = []
    k = 0
    # Model
    for hi in hi_data:
        freq = []
        for index in range(0,5):
            hi_cutmsk= np.where(region['mask'], np.nan,np.array(hi)[index,:,:])
            freq.append(np.nanmean(hi_cutmsk))
        thickness_histogramsRegion.append(freq)


    # PIOMAS
    frequencies_percentageObs = []
    for ob in obs:
        freq = []
        for index in range(12):
            hi_cutmsk= np.where(masks[j], ob[index,:,:] ,np.nan)
            freq.append(np.nanmean(hi_cutmsk))
        freqSumEnd = freq[1:7]
        freqSumEnd[-1] = np.sum(freq[6:12])
        freqSumEnd = freqSumEnd/sum(freqSumEnd)
        frequencies_percentageObs.append(freqSumEnd)

    for i, histogram in enumerate(thickness_histogramsRegion):
        axs[j].plot(bin_middle, histogram, linewidth=2,color=colors[i], label=labels[i])
        axs[j].fill_between(bin_middle, histogram,0, color=colors[i], alpha=0.5)
        if (i == 1) or (i ==2):
            axs[j].plot(bin_middleObs[1:7], frequencies_percentageObs[i-1], 'bo-',  markersize=10,linewidth=3,color=colors[i],label=obsLabels[i-1], linestyle='dashed')
        axs[j].set_xlim(0,6)
        axs[j].set_ylim(0,1)
        axs[j].set_xticks(bin_edges[0::])
        axs[j].tick_params(axis='x') 
        axs[j].grid()
        k+=1

axs[0].text(0.01, 0.93,'a) LIA-N',  transform=axs[0].transAxes,fontsize=14)
axs[1].text(0.01, 0.93, 'b) QEI',  transform=axs[1].transAxes,fontsize=14)
axs[2].text(0.01, 0.93, 'c) CAA-S',  transform=axs[2].transAxes,fontsize=14)

axs[0].legend(fontsize=12)
fig.text(0.02, 0.5, 'Frequency', va='center', rotation='vertical')
fig.text(0.5, 0.06, 'Edge of SIT bins (m)', va='center')
plt.show()
