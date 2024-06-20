# %% Import packages
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib.dates as mdates
import xarray
import datetime
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os


# %% Compute SIT for mean May SIT - pan Artic 
# Comment and uncomment EM in all dates. Creates one excel file per ensemble.

lat_min, lat_max = 30.98, 90
threshold = 15 # threshold to compute sea ice extent
cesm_grid = xarray.open_dataset(r'/aos/home/mfol/Data/CESM/domain.ocn.tx0.1v2_090218.nc', decode_times=False)

mask = np.logical_and((cesm_grid.yc >= lat_min), (cesm_grid.yc <= lat_max))

allDatesHist = ['185001-185912','186001-186912','187001-187912','188001-188912','189001-189912','190001-190912','191001-191912','192001-192912','193001-193912','194001-194912','195001-195912','196001-196912','197001-197912','198001-198912','199001-199912','200001-200611']
# E1 date proj 
allDatesProj = ['200601-201512','201601-202512','202601-203512','203601-204512','204601-205512','205601-206512','206601-207512','207601-208512','208601-209512','209601-210202']

# Create obj with filename toward hist and proj with respective dates
allDates = [
            # {'em': 'E1', 'fileArea':'mnt/qanik/iHESP/EM1/hist/aice_hist/B.E.13.BHISTC5.ne120_t12.sehires38.003.sunway.cice.h.aice.', 'file': 'mnt/qanik/iHESP/EM1/hist/hi_hist/B.E.13.BHISTC5.ne120_t12.sehires38.003.sunway.cice.h.hi.','endfile':'.nc', 'dates':allDatesHist },
            #{'em': 'E1', 'fileArea': 'mnt/qanik/iHESP/EM1/proj/aice_proj/B.E.13.BRCP85C5CN.ne120_t12.sehires38.003.sunway.CN_OFF.cice.h.aice.','file': 'mnt/qanik/iHESP/EM1/proj/hi_proj/B.E.13.BRCP85C5CN.ne120_t12.sehires38.003.sunway.CN_OFF.cice.h.hi.','endfile':'.nc', 'dates':allDatesProj },
             #{'em': 'E2', 'file': 'mnt/qanik/iHESP/EM2/hist/b.e13.BHISTC5.ne120_t12.cesm-ihesp-hires1.0.30-1920-2100.002.cice.h.','endfile':'-05.nc', 'dates':[str(year) for year in range(1920, 2006)]},
            # {'em': 'E2', 'fileArea':'mnt/qanik/iHESP/EM2/proj/aice_proj/b.e13.BRCP85C5.ne120_t12.cesm-ihesp-hires1.0.30.002.cice.h.aice.','file': 'mnt/qanik/iHESP/EM2/proj/hi_proj/b.e13.BRCP85C5.ne120_t12.cesm-ihesp-hires1.0.30.002.cice.h.hi.','endfile':'.nc', 'dates':['200601-210012']},
             {'em': 'E3', 'file': 'mnt/qanik/iHESP/EM3/hist/b.e13.BHISTC5.ne120_t12.cesm-ihesp-hires1.0.30-1920-2100.003.cice.h.','endfile':'-05.nc', 'dates':[str(year) for year in range(1920, 2006)]},
             {'em': 'E3', 'file': 'mnt/qanik/iHESP/EM3/proj/b.e13.BRCP85C5.ne120_t12.cesm-ihesp-hires1.0.31.003.cice.h.','endfile':'-05.nc', 'dates':[str(year) for year in range(2006, 2101)]},
            ]

hiArray=[]
dates = []
filesList = []
for obj in allDates :
    for datesString in obj['dates']:
        cesm = xarray.open_dataset(r'/'+ obj['file'] + datesString + obj['endfile'], decode_times=True)
        hi = cesm['hi']

        time_bounds = hi.time[0].dt.year.item(), hi.time[-1].dt.year.item()
        if time_bounds[0] == time_bounds[1] and hi.time.shape[0] == 1:
            aiceCut = np.logical_and((cesm['aice'][0,:,:] >= threshold),(hi[0,:,:] <= 120))
            aice_cutmsk= np.ma.masked_where(~mask, aiceCut)
            
            #aice_mskedCut = np.where(aice_cutmsk==1, cesm['aice'][0,:,:], np.nan)
            hi_mskd = np.where(aice_cutmsk==1, hi[0,:,:], np.nan)

            hiAvrg = np.nanmean((hi_mskd))
            hiArray.append(hiAvrg)
            dates.append(pd.to_datetime(f"{datesString}-{5:02d}"))
            filesList.append(datesString)
        else:
            for year in range(time_bounds[0], time_bounds[1]):
                # must put 6 for may instead of 5 when based on decode times 
                cesmArea = xarray.open_dataset(r'/'+ obj['fileArea'] + datesString + obj['endfile'], decode_times=True)
            
                aice_may = cesmArea['aice'].sel(time=f'{year}-{6:02d}')
                hi_may = hi.sel(time=f'{year}-{6:02d}')

                aiceCut = np.logical_and((aice_may[0,:,:] >= threshold),(hi_may[0,:,:] <= 120))
                aice_cutmsk= np.ma.masked_where(~mask, aiceCut)
                
                #aice_mskedCut = np.where(aice_cutmsk==1, aice_may[0,:,:], np.nan)
                hi_mskd = np.where(aice_cutmsk==1, hi_may[0,:,:], np.nan)

                hiAvrg = np.nanmean((hi_mskd))
                hiArray.append(hiAvrg)
                dates.append(pd.to_datetime(f"{year}-{5:02d}"))
                filesList.append(datesString)
        cesm.close()
output_file = 'IHESP/ThicknessAveraged/CESM_HR_SIT/CESM_HR_sitPanArctic'+ obj['em']
df = pd.DataFrame(columns=['Date','hi', 'file'])
for i in range(len(dates)):
    row = pd.Series({
            'Date': dates[i].strftime('%Y-%m-%d %H:%M:%S'),
            'hi': hiArray[i],
            'file':  filesList[i]})
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
df.to_excel(output_file+'.xlsx')

# %% Compute panArctic For CESM LR

hiArray=[]
dates = []
filesList = []

repo = '/storage/mfol/iHESP-HRCESM/LR/may/'
cesm1 = xarray.open_dataset(r'/storage/mfol/iHESP-HRCESM/LR/may/B.E.13.BHISTC5.ne30g16.sehires38.003.sunway.cice.h.1920-05.nc', decode_times=True)

tarea = cesm1['tarea']
lat_min, lat_max = 30.98, 90
threshold = 15 # threshold to compute sea ice extent

mask = np.logical_and((cesm1['aice'][0,:,:].TLAT >= lat_min), (cesm1['aice'][0,:,:].TLAT <= lat_max))

for year in range(1850, 2101):
    if year < 2006: 
        cesm = xarray.open_dataset(repo + 'B.E.13.BHISTC5.ne30g16.sehires38.003.sunway.cice.h.'+ str(year) + '-05.nc', decode_times=True)
    else: cesm = xarray.open_dataset(repo + 'B.E.13.BRCP85C5CN.ne30g16.sehires38.003.sunway.CN_OFF.cice.h.'+ str(year) + '-05.nc', decode_times=True)
    
    aice_may = cesm['aice']
    hi_may = cesm['hi']

    aiceCut = np.logical_and((aice_may[0,:,:] >= threshold),(hi_may[0,:,:] <= 120))
    aice_cutmsk= np.ma.masked_where(~mask, aiceCut)
    
    #aice_mskedCut = np.where(aice_cutmsk==1, aice_may[0,:,:], np.nan)
    hi_mskd = np.where(aice_cutmsk==1, hi_may[0,:,:], np.nan)

    hiAvrg = np.nanmean((hi_mskd))
    hiArray.append(hiAvrg)
    dates.append(pd.to_datetime(f"{year}-{5:02d}"))
    filesList.append(year)
cesm.close()

output_file = 'IHESP/ThicknessAveraged/CESM_HR_SIT/CESM_LR_sitPanArctic'
df = pd.DataFrame(columns=['Date','hi', 'file'])
for i in range(len(dates)):
    row = pd.Series({
            'Date': dates[i].strftime('%Y-%m-%d %H:%M:%S'),
            'hi': hiArray[i],
            'file':  filesList[i]})
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
df.to_excel(output_file+'.xlsx')
# %% Compute Pan Arctic SIT for CESM2LE

directory = '/mnt/qanik/CESM2-LE/aice/'
directoryhi = '/mnt/qanik/CESM2-LE/hi/'

for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    if os.path.isfile(f):
        cems_le = xarray.open_dataset(f, decode_times=True)
        endOfFileName = f.split('_')[1]
        cems_le_hi = xarray.open_dataset(directoryhi+'hi_'+endOfFileName, decode_times=True)
        
        aice = cems_le['aice']
        lat_min = 30.98
        threshold = 0.15 # threshold to compute sea ice extent
        mask = ((aice[0,:,:].TLAT >= lat_min))

        hiArray=[]
        dates = []
        filesList = []
        time_bounds = aice.time[0].dt.year.item(), aice.time[-1].dt.year.item()
        for year in range(time_bounds[0], time_bounds[1]):
            aice_may = cems_le['aice'].sel(time=f'{year}-{6:02d}')
            hi_may = cems_le_hi['hi'].sel(time=f'{year}-{6:02d}')

            aiceCut = np.logical_and((aice_may[0,:,:] >= threshold),(hi_may[0,:,:] <= 120))
            aice_cutmsk= np.ma.masked_where(~mask, aiceCut)
            
            #aice_mskedCut = np.where(aice_cutmsk==1, aice_may[0,:,:], np.nan)
            hi_mskd = np.where(aice_cutmsk==1, hi_may[0,:,:], np.nan)

            hiAvrg = np.nanmean((hi_mskd))
            hiArray.append(hiAvrg)
            dates.append(pd.to_datetime(f"{year}-{5:02d}"))
        cems_le.close()
        cems_le_hi.close()

        if len(filename.split('.')) > 2:
            filename = filename.split('.')[0] + '_' + filename.split('.')[1]
        else: filename = filename.split('.')[0]

        output_file = 'CESM_LE/PanArctic_May/CESM_LE_'+ filename
        df = pd.DataFrame(columns=['Date','hi'])
        for i in range(len(hiArray)):
            row = pd.Series({
                'Date': dates[i].strftime('%Y-%m-%d %H:%M:%S'),
                'hi': hiArray[i]})
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df.to_excel(output_file+'.xlsx')

    else: print('Not a file: '+filename)
print('Done!')