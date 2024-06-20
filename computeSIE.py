# %%
import numpy as np
import xarray
import pandas as pd
import matplotlib.path as mpath
import os
import scipy.io.matlab as matlab
import time

# %% CESM HR  
#
#
# Define grid and masks
cesm_grid = xarray.open_dataset(r'/aos/home/mfol/Data/CESM/domain.ocn.tx0.1v2_090218.nc', decode_times=False)
lia_mask_QEI = xarray.open_dataarray(r'./masked_data_QEI23May.nc')
lia_mask_CAA = xarray.open_dataarray(r'./masked_data_CAA23May.nc')
lia_mask_LIAN = xarray.open_dataarray(r'/aos/home/mfol/Results/IHESP/masked_data_LIANorth.nc')
#cesm = xarray.open_dataset(r'/mnt/qanik/iHESP/EM2/proj/aice_proj/b.e13.BRCP85C5.ne120_t12.cesm-ihesp-hires1.0.30.002.cice.h.aice.200601-210012.nc', decode_times=True)

regions = [
    {'name': 'LIA-N','maskData': lia_mask_LIAN,'mask': None},
    {'name': 'QEI', 'maskData': lia_mask_QEI,'mask': None},
    {'name': 'CAA','maskData': lia_mask_CAA,'mask': None}
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
colorObs2 = '#4F4F4F'

# %%  HR Find SIE sept for all years of all ensembles
threshold = 15 # threshold to compute sea ice extent

# E1 dates hist
allDatesHist = ['192001-192912','193001-193912','194001-194912','195001-195912','196001-196912','197001-197912','198001-198912','199001-199912','200001-200611']
# E1 date proj 
allDatesProj = ['200601-201512','201601-202512','202601-203512','203601-204512','204601-205512','205601-206512','206601-207512','207601-208512','208601-209512','209601-210202']

# Create obj with filename toward hist and proj with respective dates
# Comment and uncomment all objects from an ensemble at a time (proj and hist)
# Also change the month to september or march and the name of the saved file accordingly
# This will create one excel file per ensemble
allDates = [
            #{'em': 'E1', 'file': 'mnt/qanik/iHESP/EM1/hist/aice_hist/B.E.13.BHISTC5.ne120_t12.sehires38.003.sunway.cice.h.aice.','endfile':'.nc', 'dates':allDatesHist },
            #{'em': 'E1', 'file': 'mnt/qanik/iHESP/EM1/proj/aice_proj/B.E.13.BRCP85C5CN.ne120_t12.sehires38.003.sunway.CN_OFF.cice.h.aice.','endfile':'.nc', 'dates':allDatesProj },
            #{'em': 'E2', 'file': 'mnt/qanik/iHESP/EM3/hist/b.e13.BHISTC5.ne120_t12.cesm-ihesp-hires1.0.30-1920-2100.003.cice.h.','endfile':'-09.nc', 'dates':[str(year) for year in range(1920, 2006)]},
            #{'em': 'E2', 'file': 'mnt/qanik/iHESP/EM2/proj/aice_proj/b.e13.BRCP85C5.ne120_t12.cesm-ihesp-hires1.0.30.002.cice.h.aice.','endfile':'.nc', 'dates':['200601-210012']},
            {'em': 'E3', 'file': 'mnt/qanik/iHESP/EM3/hist/b.e13.BHISTC5.ne120_t12.cesm-ihesp-hires1.0.30-1920-2100.003.cice.h.','endfile':'-03.nc', 'dates':[str(year) for year in range(1920, 2006)]},
            {'em': 'E3', 'file': 'mnt/qanik/iHESP/EM3/proj/b.e13.BRCP85C5.ne120_t12.cesm-ihesp-hires1.0.31.003.cice.h.','endfile':'-03.nc', 'dates':[str(year) for year in range(2006, 2101)]},
            ]
for region in regions: 
    sie = []
    sia = []
    aiceArray=[]
    dates = []
    filesList = []
    for obj in allDates :
        for datesString in obj['dates']:
            cesm = xarray.open_dataset(r'/'+ obj['file'] + datesString + obj['endfile'], decode_times=True)
            aice = cesm['aice']
            tarea = cesm['tarea']

            # Apply ROI mask and threshold
            mask = region['mask']
            tarea_mskd = np.ma.masked_where(mask, tarea)

            time_bounds = aice.time[0].dt.year.item(), aice.time[-1].dt.year.item()
            if time_bounds[0] == time_bounds[1] and aice.time.shape[0] == 1:
                aice_mskd = np.ma.masked_where(mask, aice[0,:,:])
                aiceCut = aice_mskd >= threshold
                siextent = np.ma.masked_where(~aiceCut, tarea_mskd).sum(axis=(0,1))
                siarea = (aice_mskd * tarea_mskd/100).sum(axis=(0,1))
                sie.append(siextent)
                sia.append(siarea)
                dates.append(pd.to_datetime(f"{datesString}-{3:02d}"))
                filesList.append(datesString)
            else:
                for year in range(time_bounds[0], time_bounds[1]):
                    # must put 10 instead of 9 when based on decode times for September
                    # 4 for march
                    aice_sept = aice.sel(time=f'{year}-{4:02d}')
                    aice_mskd = np.ma.masked_where(mask, aice_sept[0,:,:])
                    aiceCut = aice_mskd >= threshold
                    siextent = np.ma.masked_where(~aiceCut, tarea_mskd).sum(axis=(0,1))
                    siarea = (aice_mskd * tarea_mskd/100).sum(axis=(0,1))
                    sie.append(siextent)
                    sia.append(siarea)
                    dates.append(pd.to_datetime(f"{year}-{3:02d}"))
                    filesList.append(datesString)

            cesm.close()
            
    # Save excel with results
    # Change Sept or March
    output_file = 'Results23May/CESM_HR_sieMarch'+ obj['em']+ region['name']
    df = pd.DataFrame(columns=['Date','SIE', 'SIA','file'])
    for i in range(len(dates)):
        row = pd.Series({
                'Date': dates[i].strftime('%Y-%m-%d %H:%M:%S'),
                'SIE': sie[i],
                'SIA': sia[i],
                'file':  filesList[i]})
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_excel(output_file+'.xlsx')

# %% HR Compute SIE pan-Arctic and save in excel.
# Comment and uncomment all dates for each ensemble and change name of excel file
start_time = time.time()

# E1 dates hist
allDates = ['185001-185912','186001-186912','187001-187912','188001-188912','189001-189912','190001-190912','191001-191912','192001-192912','193001-193912','194001-194912','195001-195912','196001-196912','197001-197912','198001-198912','199001-199912','200001-200611']

# E1 date proj 
# allDates = ['200601-201512','201601-202512','202601-203512','203601-204512','204601-205512','205601-206512','206601-207512','207601-208512','208601-209512','209601-210202']

# E2-E3 hist
# allDates = [str(year) for year in range(1920, 2006)]

# E2 proj
# allDates = ['200601-210012']

# E3 - proj
# allDates = [str(year) for year in range(2006, 2101)]

sie = []
aiceArray=[]
dates = []
h = []
filesList = []
for datesString in allDates :
    # E1 - hist
    cesm = xarray.open_dataset(r'/mnt/qanik/iHESP/EM1/hist/aice_hist/B.E.13.BHISTC5.ne120_t12.sehires38.003.sunway.cice.h.aice.' + datesString + '.nc', decode_times=True)
    # E1 - proj
    # cesm = xarray.open_dataset(r'/mnt/qanik/iHESP/EM1/proj/aice_proj/B.E.13.BRCP85C5CN.ne120_t12.sehires38.003.sunway.CN_OFF.cice.h.aice.' + datesString + '.nc', decode_times=True)
    # E2 - hist
    # cesm = xarray.open_dataset(r'/mnt/qanik/iHESP/EM2/hist/b.e13.BHISTC5.ne120_t12.cesm-ihesp-hires1.0.30-1920-2100.002.cice.h.' + datesString + '-09.nc', decode_times=True)
    # E2 - proj
    # cesm = xarray.open_dataset(r'/mnt/qanik/iHESP/EM2/proj/aice_proj/b.e13.BRCP85C5.ne120_t12.cesm-ihesp-hires1.0.30.002.cice.h.aice.' + datesString + '.nc', decode_times=True)
    # E3 - hist
    # cesm = xarray.open_dataset(r'/mnt/qanik/iHESP/EM3/hist/b.e13.BHISTC5.ne120_t12.cesm-ihesp-hires1.0.30-1920-2100.003.cice.h.' + datesString + '-09.nc', decode_times=True)
    # E3 - proj
    # cesm = xarray.open_dataset(r'/mnt/qanik/iHESP/EM3/proj/b.e13.BRCP85C5.ne120_t12.cesm-ihesp-hires1.0.31.003.cice.h.' + datesString + '-09.nc', decode_times=True)
    
    aice = cesm['aice']
    tarea = cesm['tarea']
    lat_min, lat_max = 30.98, 90
    threshold = 15 # threshold to compute sea ice extent

    mask = np.logical_and((aice.TLAT >= lat_min), (aice.TLAT <= lat_max))
    tarea_arctic = tarea.where(mask, drop=True)

    time_bounds = aice.time[0].dt.year.item(), aice.time[-1].dt.year.item()
    if time_bounds[0] == time_bounds[1] and aice.time.shape[0] == 1:
        aiceCut = aice[0,:,:].where(mask, drop=True) >= threshold
        siextent = tarea_arctic.where(aiceCut).sum(dim=['nj', 'ni'])
        aicesum = (aice[0,:,:].where(mask, drop=True) * tarea_arctic).sum(dim=['nj', 'ni'])
        aiceArray.append(aicesum.values)
        sie.append(siextent.values)
        dates.append(pd.to_datetime(f"{datesString}-{9:02d}"))
        filesList.append(datesString)
    else:
        for year in range(time_bounds[0], time_bounds[1]):
            # must put 10 instead of 9 when based on decode times instead of file name
            aice_sept = aice.sel(time=f'{year}-{10:02d}')
            if aice_sept.shape[0] > 1:
                print('ERROR: two files in september')
            aiceCut = aice_sept[0,:,:].where(mask, drop=True) >= threshold
            siextent = tarea_arctic.where(aiceCut).sum(dim=['nj', 'ni'])
            aicesum = (aice_sept[0,:,:].where(mask, drop=True) * tarea_arctic).sum(dim=['nj', 'ni'])
            aiceArray.append(aicesum.values)
            sie.append(siextent.values)
            dates.append(pd.to_datetime(f"{year}-{9:02d}"))
            filesList.append(datesString)
    cesm.close()

# Change name hist or proj and E1,E2,E3
output_file = 'CESM_HR/CESM_HR_septsie_hist_E1'
df = pd.DataFrame(columns=['Date','SIE','AICE', 'file'])
for i in range(len(dates)):
    row = pd.Series({
            'Date': dates[i].strftime('%Y-%m-%d %H:%M:%S'),
            'SIE': sie[i],
            'AICE': aiceArray[i],
            'file':  filesList[i]})
    df = df.append(row, ignore_index=True)
df.to_excel(output_file+'.xlsx')

print("Elapsed Time:", time.time() - start_time, "seconds")







# %% CESM LE 
#
#
# Open grids and define regions
cesm1 = xarray.open_dataset(r'/mnt/qanik/CESM2-LE/aice/EM_1001.nc', decode_times=True)
repo = '/mnt/qanik/CESM2-LE/aice/'

tarea = cesm1['uarea']
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

# Get polygon from coordinates
# Change the polygon to coordinatesQEI or coordinatesCAA and excel name
lon, lat = zip(*coordinatesQEI)
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
lon_flat = np.ravel(cesm1['TLON'])
lat_flat = np.ravel(cesm1['TLAT'])
mask = path.contains_points(np.column_stack((lon_flat, lat_flat))).reshape(cesm1['TLON'].shape)

# %% LE compute SIE and SIA per region
# Change date to march if needed
repo = '/mnt/qanik/CESM2-LE/aice/'
threshold = 0.15

# Compute SIA and SIE for each masked region
for filename in os.listdir(repo):
    dates = []
    aiceArray = []
    sieArray = []
    f = os.path.join(repo, filename)
    if os.path.isfile(f):
        cesm = xarray.open_dataset(f, decode_times=True)
        tarea = cesm['uarea'] # I dont have tarea ..
        # Apply mask of region on tarea and aice
        tarea_mskd = np.ma.masked_where(~mask, tarea)
        aice = cesm['aice']
        time_bounds = aice.time[0].dt.year.item(), aice.time[-1].dt.year.item()
        for year in range(time_bounds[0], time_bounds[1]):
            aice_sept = aice.sel(time=f'{year}-{10:02d}')[0,:,:]
            aice_mskd = np.ma.masked_where(~mask, aice_sept)
    
            # Apply SIE threshold
            aiceCut = aice_mskd >= threshold

            # Compute SIE and SIA
            siextent = np.ma.masked_where(~aiceCut, tarea_mskd).sum(axis=(0,1))
            siarea = np.nansum(np.array((aice_mskd * tarea_mskd)), axis=(0,1))
            
            aiceArray.append(siarea)
            sieArray.append(siextent)
            dates.append(pd.to_datetime(f"{year}-{9:02d}"))

        cesm.close()
        if len(filename.split('.')) > 2:
            filename = filename.split('.')[0] + '_' + filename.split('.')[1]
        else: filename = filename.split('.')[0]
        output_file = './Results23May/CESMLE/CESM_LE_sept_QEI'+ filename
        
        df = pd.DataFrame(columns=['Date','SIE','AICE'])
        for i in range(len(dates)-1):
            row = pd.Series({
                    'Date': dates[i].strftime('%Y-%m-%d %H:%M:%S'),
                    'SIE': sieArray[i],
                    'AICE': aiceArray[i]})
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df.to_excel(output_file+'.xlsx')

# %% LE pan arctic SIE : per ensemble 1850 to 2100
# Mask NH
directory = '/mnt/qanik/CESM2-LE/aice/'

for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    if os.path.isfile(f):
        cems_le = xarray.open_dataset(f, decode_times=True)
        tarea = cems_le['uarea']
        
        aice = cems_le['aice']
        lat_min = 30.98
        threshold = 0.15 # threshold to compute sea ice extent
        mask = ((aice[0,:,:].TLAT >= lat_min))
        tarea_arctic = tarea.where(mask, drop=True)

        dates = []
        sie = []
        aiceArray = []
        time_bounds = aice.time[0].dt.year.item(), aice.time[-1].dt.year.item()
        for year in range(time_bounds[0], time_bounds[1]):
                aice_sept = aice.sel(time=f'{year}-{10:02d}')
                if aice_sept.shape[0] > 1:
                    print('ERROR: two files in september')
                aiceCut = aice_sept[0,:,:].where(mask, drop=True) >= threshold
                siextent = tarea_arctic.where(aiceCut).sum(dim=['nj', 'ni'])
                aicesum = (aice_sept[0,:,:].where(mask, drop=True) * tarea_arctic).sum(dim=['nj', 'ni'])
                aiceArray.append(aicesum.values)
                sie.append(siextent.values)
                dates.append(pd.to_datetime(f"{year}-{9:02d}"))
        cems_le.close()
        if len(filename.split('.')) > 2:
            filename = filename.split('.')[0] + '_' + filename.split('.')[1]
        else: filename = filename.split('.')[0]
        output_file = 'CESM_LE_NH2/CESM_LE_'+ filename
        df = pd.DataFrame(columns=['Date','SIE', 'AICE'])
        for i in range(len(dates)):
            row = pd.Series({
                    'Date': dates[i].strftime('%Y-%m-%d %H:%M:%S'),
                    'SIE': sie[i], 
                    'AICE': aiceArray[i]})
            df = df.append(row, ignore_index=True)
        df.to_excel(output_file+'.xlsx')

    else: print('Not a file: '+filename)
print('Done!')








# %% CESM1.3-LR
#
#
# Find pan Arctic SIE and SIA for LR AICE is from 0 to 100
# Change sept to march if needed
sieArray = []
aiceArray=[]
dates = []

repo = '/storage/mfol/iHESP-HRCESM/LR/'
cesm1 = xarray.open_dataset(r'/storage/mfol/iHESP-HRCESM/LR/B.E.13.BHISTC5.ne30g16.sehires38.003.sunway.cice.h.1850-09.nc', decode_times=True)

tarea = cesm1['tarea']
lat_min, lat_max = 30.98, 90
threshold = 15 # threshold to compute sea ice extent

mask = np.logical_and((cesm1['aice'][0,:,:].TLAT >= lat_min), (cesm1['aice'][0,:,:].TLAT <= lat_max))
tarea_arctic = tarea.where(mask, drop=True)

for year in range(1850, 2101):
    if year < 2006: 
        cesm = xarray.open_dataset(repo + 'B.E.13.BHISTC5.ne30g16.sehires38.003.sunway.cice.h.'+ str(year) + '-09.nc', decode_times=True)
    else: cesm = xarray.open_dataset(repo + 'B.E.13.BRCP85C5CN.ne30g16.sehires38.003.sunway.CN_OFF.cice.h.'+ str(year) + '-09.nc', decode_times=True)
    
    aiceCut = cesm['aice'][0,:,:].where(mask, drop=True) >= threshold
    siextent = tarea_arctic.where(aiceCut).sum(dim=['nj', 'ni'])
    aicesum = (cesm['aice'][0,:,:].where(mask, drop=True) * tarea_arctic).sum(dim=['nj', 'ni'])
    aiceArray.append(aicesum.values)
    sieArray.append(siextent.values)
    dates.append(pd.to_datetime(f"{year}-{9:02d}"))
cesm.close()

output_file = 'Results23May/CESM_LR_septsie_proj'
df = pd.DataFrame(columns=['Date','SIE','AICE'])
for i in range(len(dates)-1):
    row = pd.Series({
            'Date': dates[i].strftime('%Y-%m-%d %H:%M:%S'),
            'SIE': sieArray[i],
            'AICE': aiceArray[i]})
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
df.to_excel(output_file+'.xlsx')

# %% LR find SIE and SIA per region 
# Find mask per region on LR grid
# Change to march if needed
grid = xarray.open_dataset(r'/aos/home/mfol/Data/CESM/domain.ocn.gx1v6.090206.nc', decode_times=True)
cesm1 = xarray.open_dataset(r'/storage/mfol/iHESP-HRCESM/LR/sept/B.E.13.BHISTC5.ne30g16.sehires38.003.sunway.cice.h.1850-09.nc', decode_times=True)

tarea = cesm1['tarea']
repo = '/storage/mfol/iHESP-HRCESM/LR/sept/'
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

# Get polygon from coordinates
# Change the polygon to coordinatesQEI or coordinatesCAA and excel name
lon, lat = zip(*coordinatesQEI)
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
lon_flat = np.ravel(grid.xc)
lat_flat = np.ravel(grid.yc)
mask = path.contains_points(np.column_stack((lon_flat, lat_flat))).reshape(grid.xc.shape)

dates = []
aiceArray = []
sieArray = []
threshold = 15
# Compute SIA and SIE for each masked region
for year in range(1850, 2101):
    if year < 2006: 
        cesm = xarray.open_dataset(repo + 'B.E.13.BHISTC5.ne30g16.sehires38.003.sunway.cice.h.'+ str(year) + '-09.nc', decode_times=True)
    else: cesm = xarray.open_dataset(repo + 'B.E.13.BRCP85C5CN.ne30g16.sehires38.003.sunway.CN_OFF.cice.h.'+ str(year) + '-09.nc', decode_times=True)
    
    # Apply mask of region on tarea and aice
    tarea_mskd = np.ma.masked_where(~mask, tarea)
    aice_mskd = np.ma.masked_where(~mask, cesm['aice'][0,:,:])

    # Apply SIE threshold
    aiceCut = aice_mskd >= threshold

    # Compute SIE and SIA
    siextent = np.ma.masked_where(~aiceCut, tarea_mskd).sum(axis=(0,1))
    siarea = (aice_mskd * tarea_mskd/100).sum(axis=(0,1))
    aiceArray.append(siarea)
    sieArray.append(siextent)
    dates.append(pd.to_datetime(f"{year}-{9:02d}"))

    cesm.close()

output_file = 'Results23May/CESM_LR_sept_QEI'
df = pd.DataFrame(columns=['Date','SIE','AICE'])
for i in range(len(dates)-1):
    row = pd.Series({
            'Date': dates[i].strftime('%Y-%m-%d %H:%M:%S'),
            'SIE': sieArray[i],
            'AICE': aiceArray[i]})
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
df.to_excel(output_file+'.xlsx')
