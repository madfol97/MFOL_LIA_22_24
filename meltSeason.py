# %% import packages
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import xarray
import datetime
import pandas as pd
import matplotlib.path as mpath
from matplotlib.colors import Normalize
from matplotlib.colorbar import ColorbarBase
import scipy.io.matlab as matlab
from datetime import datetime
import cmocean

# %% Get grids and masks
cesm_grid = xarray.open_dataset(r'/aos/home/mfol/Data/CESM/domain.ocn.tx0.1v2_090218.nc', decode_times=False)
lia_mask_QEI = xarray.open_dataarray(r'./masked_data_QEI23May.nc')
lia_mask_CAA = xarray.open_dataarray(r'./masked_data_CAA23May.nc')
lia_mask_LIAN = xarray.open_dataarray(r'/aos/home/mfol/Results/IHESP/masked_data_LIANorth.nc')

cesm = xarray.open_dataset(r'/mnt/qanik/iHESP/EM1/hist/aice_hist/B.E.13.BHISTC5.ne120_t12.sehires38.003.sunway.cice.h.aice.185001-185912.nc', decode_times=True)
nsdicobs = xarray.open_dataset(r'/storage/mfol/obs/nsidc/arctic_seaice_climate_indicators_nh_v01r01_1979-2017.nc', decode_times=True)
gridnsdic = xarray.open_dataset(r'/storage/mfol/obs/nsidc/NSIDC0771_LatLon_PS_N25km_v1.0.nc', decode_times=True)

regions = [
    {'name': 'QEI', 'maskData': lia_mask_QEI,'mask': None, 'nbPntsMask': None},
    {'name': 'CAA','maskData': lia_mask_CAA,'mask': None, 'nbPntsMask': None},
    {'name': 'LIA-N','maskData': lia_mask_LIAN,'mask': None, 'nbPntsMask': None}
    ]
for region in regions: 
    lia = xarray.where((region['maskData'] == 0) | (region['maskData'].isnull()), 0, 1) # 0 everywhere, 1 in ROI
    liaSumValues = lia.sum() # compute number of pixels in mask
    mask = xarray.where((region['maskData'] == 0) | (region['maskData'].isnull()), 1, 0) # 1 everywhere, 0 in ROI
    region['mask'] = mask
    region['nbPntsMask'] = liaSumValues

# %%  Find melt day with daidtt criteria
def find_threshold_crossings(array,threshold):
    crossings = []
    array = np.array(array)
    # Go over each year and find crossings with thereshold2
    for month in range(1, len(array)):
        if (array[month-1] <= threshold and array[month] > threshold) or (array[month-1] >= threshold and array[month] < threshold):
            # lin interpolation between two points over y = 0
            dx = month - (month-1)
            dy = array[month] - array[month-1]
            a = dy/dx
            b = array[month-1] - a*(month-1)
            interp = -b/a
            crossings.append(interp)
    # Validation and add a month since difference between index of month and the month number itself (ex: march = 2+1)
    if len(crossings) == 2:
        return crossings[0]+1, crossings[1]+1
    return 0, 0


threshold = 0 # threshold to compute melt season based on tendencies

# E1 dates hist
allDatesHist = ['192001-192912','193001-193912','194001-194912','195001-195912','196001-196912','197001-197912','198001-198912','199001-199912','200001-200611']
# E1 date proj 
allDatesProj = ['200601-201512','201601-202512','202601-203512','203601-204512','204601-205512','205601-206512','206601-207512','207601-208512','208601-209512','209601-210202']

# Create obj with filename toward hist and proj with respective dates
# Comment and uncomment EM1 and EM3. Will create one excel file per ensemble
allDates = [
            {'em': 'E1', 'file': 'storage/mfol/iHESP-HRCESM/EM1/hist/daidtt/B.E.13.BHISTC5.ne120_t12.sehires38.003.sunway.cice.h.daidtt.','filedyn': 'storage/mfol/iHESP-HRCESM/EM1/hist/dvidtd/B.E.13.BHISTC5.ne120_t12.sehires38.003.sunway.cice.h.dvidtd.','endfile':'.nc', 'dates':allDatesHist },
            {'em': 'E1', 'file': 'storage/mfol/iHESP-HRCESM/EM1/proj/daidtt/B.E.13.BRCP85C5CN.ne120_t12.sehires38.003.sunway.CN_OFF.cice.h.daidtt.', 'filedyn': 'storage/mfol/iHESP-HRCESM/EM1/proj/dvidtd/B.E.13.BRCP85C5CN.ne120_t12.sehires38.003.sunway.CN_OFF.cice.h.dvidtd.','endfile':'.nc', 'dates':allDatesProj },
            #{'em': 'E3', 'file': 'mnt/qanik/iHESP/EM3/hist/b.e13.BHISTC5.ne120_t12.cesm-ihesp-hires1.0.30-1920-2100.003.cice.h.','filedyn': 'mnt/qanik/iHESP/EM3/hist/b.e13.BHISTC5.ne120_t12.cesm-ihesp-hires1.0.30-1920-2100.003.cice.h.','endfile':'-01.nc', 'dates':[str(year) for year in range(1920, 2006)]},
            #{'em': 'E3', 'file': 'mnt/qanik/iHESP/EM3/proj/b.e13.BRCP85C5.ne120_t12.cesm-ihesp-hires1.0.31.003.cice.h.','filedyn': 'mnt/qanik/iHESP/EM3/proj/b.e13.BRCP85C5.ne120_t12.cesm-ihesp-hires1.0.31.003.cice.h.','endfile':'-01.nc', 'dates':[str(year) for year in range(2006, 2101)]},
            ]

allYears = range(1920, 2101)

for region in regions: 
    meltStart = []
    freezeStart = []
    meltDuration = []
    years = []
    crossingsTotal = []
    filesList = []

    for obj in allDates :
        for datesString in obj['dates']:
            cesm = xarray.open_dataset(r'/'+ obj['file'] + datesString + obj['endfile'], decode_times=True)
            daidtt = cesm['daidtt']

            # Apply ROI mask and threshold
            mask = region['mask']

            time_bounds = daidtt.time[0].dt.year.item(), daidtt.time[-1].dt.year.item()
            if time_bounds[0] == time_bounds[1] and daidtt.time.shape[0] == 1:
                yearlyBudget = []

                # For january (already opened file)
                daidttNa = daidtt[0,:,:].fillna(0.0)
                maskedaidtt = np.ma.masked_where(mask, daidttNa)
                
                yearlyBudget.append(np.sum(maskedaidtt, axis=(0,1)))
                # For Feb to Dec
                for month in range(2,13):
                    if month < 10:
                        cesmMonth = xarray.open_dataset(r'/'+ obj['file'] + datesString + '-0' + str(month) + '.nc', decode_times=True)
                    else: 
                        cesmMonth = xarray.open_dataset(r'/'+ obj['file'] + datesString + '-' + str(month) + '.nc', decode_times=True)
                    
                    #  apply mask and compute sum
                    daidttNa = cesmMonth['daidtt'][0,:,:].fillna(0.0)
                    maskedaidtt = np.ma.masked_where(mask, daidttNa)
                    yearlyBudget.append(np.sum(maskedaidtt, axis=(0,1)))
                    cesmMonth.close()

                # Find 0 crossing of daidtt
                starts, end = find_threshold_crossings(yearlyBudget, threshold)

                meltStart.append(starts)
                freezeStart.append(end)
                meltDuration.append(end-starts)
                years.append(pd.to_datetime(f"{datesString}"))
                filesList.append(datesString)

            else:
                for year in range(time_bounds[0], time_bounds[1]):
                    print(year)
                    daidttOfYear = []
                    # Get all daidtt from each  month of a year
                    for monthToSelect in range(2,13): #Jan to nov (2 - 12)
                        daidttOfYear.append(daidtt.sel(time=f'{year}-{monthToSelect:02d}')) # Does not take december
                    daidttDec = daidtt.sel(time=f'{year+1}-{1:02d}')
                    concatmonths = xarray.concat(daidttOfYear, dim='time')
                    daidttCombined = xarray.concat([concatmonths, daidttDec], dim='time')

                    yearlyBudget = []
                    idx = 0
                    #  apply mask and compute sum
                    for daidttInYear in daidttCombined:
                        daidttInYear = daidttInYear.fillna(0.0)
                        maskedaidtt = np.ma.masked_where(mask, daidttInYear)
                        yearlyBudget.append(np.sum(maskedaidtt, axis=(0,1)))
                        idx += 1
                    # Find 0 crossing of daidtt
                    starts, end = find_threshold_crossings(yearlyBudget, threshold)

                    meltStart.append(starts)
                    freezeStart.append(end)
                    meltDuration.append(end-starts)
                    years.append(pd.to_datetime(f"{year}"))
                    filesList.append(datesString)
            cesm.close()

    output_file = './Results23May/CESM_HR_meltSeason'+ obj['em']+ region['name']
    df = pd.DataFrame(columns=['Date','meltStart', 'freezeStart', 'meltDuration','file'])
    for i in range(len(years)):
        row = pd.Series({
                'Date': years[i].strftime('%Y-%m-%d %H:%M:%S'),
                'meltStart': meltStart[i],
                'freezeStart': freezeStart[i],
                'meltDuration': meltDuration[i],
                'file':  filesList[i]})
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        
    df.to_excel(output_file+'.xlsx')


# %% Compute melt season Obs
emoPerRegion = []
efoPerRegion = []
# longitude go from -180 to 180.
coordinatesQEI = [(-117, 76.6),(-116.1,76.66),(-115,76.4),(-112,75.4),(-107,75.4),(-106,75.3),(-105.68,75.8),(-103.65,75.9),(-101.75,75.7),(-98.7,75.8),(-97.73,76.5),
                  (-96.4,76.67),(-94.38,76.43),(-90.55,76.6),(-89.5,76.55),
                  (-87.11,76.58),(-82.90,77.09),(-83.5,78.4),(-87.32, 78.17),(-88.84, 78.2),(-92.8,80.5),(-96.3, 80.3),(-99.05, 80.10),(-100.0, 79.9),
                  (-103.78, 79.35), (-105.5, 79.2),(-110.4, 78.75),(-113.10, 78.3),(-114.3, 78.08),(-115.06, 77.95),(-116.47, 77.56),
                  (-117, 76.6)]
coordinatesCAA = [(-128.19, 69.0),(-110.58,66.0),(-95.56,66.5),(-86.2,67.02),(-82.7,71.0),
                  (-81.9, 73.7),(-81.89, 74.52),(-91.67,74.79),(-91,75.55),
                   (-91.69,76.4),(-94.38,76.43),(-96.4,76.67),(-97.73,76.5),(-98.7,75.8),(-101.75,75.7),(-103.65,75.9),
                   (-105.68,75.8),(-106,75.3),(-107,75.4),(-112,75.4),(-115,76.4),(-116.1,76.66),(-120.2,76.6),(-122.0,76.2),
                   (-124.19, 74.32),(-123.2, 73.28),(-125.4, 72.18),(-128.19, 70.16)]
          
# LIA North from DeRepentigny, P., L.B. Tremblay, R. Newton, and S. Pfirman, (2016), 
# Patterns of sea ice retreat in the transition to a seasonally ice-free Arctic. 
# Journal of Climate, DOI: 10.1175/JCLI-D-15-0733.1. For the SITU system.
# I used LIAIndices.mat -> XY2LatLon(blia,alia)
coordLIANewton = matlab.loadmat('/aos/home/mfol/Results/IHESP/LIALatLonEdges.mat')
coordinatesLIANorth = []
for idx, elem in enumerate(coordLIANewton['lon']): 
    coordinatesLIANorth.append((elem[0], coordLIANewton['lat'][idx][0]))
coordinatesLIANorth.append(coordinatesLIANorth[0])

for coord in [coordinatesLIANorth, coordinatesQEI, coordinatesCAA]:
    # Get polygon from coordinates
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
    lon_flat = np.ravel(gridnsdic['longitude'])
    lat_flat = np.ravel(gridnsdic['latitude'])

    mask = path.contains_points(np.column_stack((lon_flat, lat_flat))).reshape(gridnsdic['longitude'].shape)

    # Mask of region + get rid of unreaslistic values of the pole hole (-5)
    emoregion = []
    eforegion = []
    for i in range(44):
        fileName = '/aos/home/mfol/Data/Observations/Markus2009/'+ str(1979 + i) + '731smeltfreeze.nc'
        nsdicobs = xarray.open_dataset(fileName, decode_times=True)
        emoMasked = np.where(mask, nsdicobs['Melt'][:,:], np.nan)
        efoMasked = np.where(mask, nsdicobs['Freeze'][:,:], np.nan)
        
        # Replace land and water with nan
        emoMasked= np.where(emoMasked <= 0, np.nan, emoMasked)
        efoMasked= np.where(efoMasked <= 0, np.nan, efoMasked)
        emoregion.append(np.nanmean(emoMasked,axis=(0,1)))
        eforegion.append(np.nanmean(efoMasked, axis=(0,1)))
    
    emoPerRegion.append(emoregion)
    efoPerRegion.append(eforegion)
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.NorthPolarStereo(central_longitude=-90)})
    scat = ax.pcolormesh(gridnsdic['longitude'][:,:],gridnsdic['latitude'][:,:], emoMasked, transform=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE,color='gray',linewidth=0.3)
    ax.gridlines(draw_labels=True)
    ax.set_extent([-140, -60, 65, 90], crs=ccrs.PlateCarree()) # CAA


# %% Plot melt season from model and obs - Figure 7

def month_to_julian_day(month_decimal):
    if not np.isnan(month_decimal) and month_decimal:
        month_integer = int(month_decimal)
        month_fractional = month_decimal - month_integer
        if month_integer + 1 <12:
            num_days_in_month = (datetime(1970, month_integer + 1, 1) - datetime(1970, month_integer, 1)).days
        else:
            num_days_in_month = (datetime(1971, 1, 1) - datetime(1970, month_integer, 1)).days
        
        # Calculate the Julian day using the number of days in the month
        julian_day = month_integer * num_days_in_month + month_fractional * num_days_in_month
        
        return julian_day
    return np.nan

regions = [
        {'region': 'LIA-N', 'data': [
            {'em': '1','file':'./Results23May/CESM_HR_meltSeasonE1LIA-N.xlsx','meltSeason': None},
            {'em': '3','file':'./Results23May/CESM_HR_meltSeasonE3LIA-N.xlsx','meltSeason': None},
            ],'meltDuration': None,'min':None,'max':None,'freezeStart': None,'meltStart': None},
        {'region': 'QEI', 'data': [
            {'em': '1','file':'./Results23May/CESM_HR_meltSeasonE1QEI.xlsx','meltSeason': None},
            {'em': '3','file':'./Results23May/CESM_HR_meltSeasonE3QEI.xlsx','meltSeason': None},
            ],'meltDuration': None,'min':None,'max':None,'freezeStart': None,'meltStart': None},
        {'region': 'CAA', 'data': [
            {'em': '1','file':'./Results23May/CESM_HR_meltSeasonE1CAA.xlsx','meltSeason': None},
            {'em': '3','file':'./Results23May/CESM_HR_meltSeasonE3CAA.xlsx','meltSeason': None},
            ], 'meltDuration': None,'min':None,'max':None,'freezeStart': None,'meltStart': None},
]


fig, axs = plt.subplots(3,1,figsize=(12, 12), sharex=True)
plt.rc('font', size=14)
j = 0
cmap = cmocean.cm.matter
norm = Normalize(vmin=60, vmax=200)

for region in regions:
    dataOfAllEm = []
    for em in region['data']:
        dataOfAllEm.append(pd.read_excel(em['file'],  sheet_name='Sheet1', parse_dates=['Date'], date_parser=pd.to_datetime))
    
    merged_df = pd.concat(dataOfAllEm, ignore_index=True)  
    merged_df['Date'] = pd.to_datetime(merged_df['Date'])
    merged_df['meltDuration'] = pd.to_numeric(merged_df['meltDuration'], errors='coerce')
    merged_df['freezeStart'] = pd.to_numeric(merged_df['freezeStart'], errors='coerce')
    merged_df['meltStart'] = pd.to_numeric(merged_df['meltStart'], errors='coerce')
    
    region['meltDuration'] = merged_df.groupby('Date')['meltDuration'].mean().reset_index()['meltDuration']
    region['min'] = merged_df.groupby('Date')['meltDuration'].min().reset_index()['meltDuration']
    region['max'] = merged_df.groupby('Date')['meltDuration'].max().reset_index()['meltDuration']
    region['freezeStart'] = merged_df.groupby('Date')['freezeStart'].mean().reset_index()['freezeStart']
    region['meltStart'] = merged_df.groupby('Date')['meltStart'].mean().reset_index()['meltStart']
    dates = merged_df['Date'][0:181]

    for i, elem in enumerate(region['freezeStart'][0:181]):
        region['freezeStart'][i] = month_to_julian_day(elem)
        region['meltStart'][i] = month_to_julian_day(region['meltStart'][i])
    
    diff = region['freezeStart'][0:181].values - region['meltStart'][0:181].values

    axs[j].plot(dates, region['freezeStart'][0:181], label='Freeze Onset', color='grey')
    axs[j].plot(dates, region['meltStart'][0:181], label='Melt Onset', color='grey')
    for i in range(180):
        axs[j].fill_between([dates[i], dates[i+1]], [region['freezeStart'][i], region['freezeStart'][i+1]], color=cmap(norm(diff[i])))
        axs[j].fill_between([dates[i], dates[i+1]], [region['meltStart'][i], region['meltStart'][i+1]], color='white', linewidth=3)

    years = list(range(1920, 2101))
    juliandays = list(range(1, 365))
    axs[j].set_xticklabels(years[::20]) 
    axs[j].set_xlim(datetime(1920, 1, 1), datetime(2100, 1, 1))
    axs[j].set_ylim(0,365)
    axs[j].grid(True)
    axs[j].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    axs[j].yaxis.set_major_locator(MultipleLocator(50))

    j += 1

fig.subplots_adjust(right=0.8)

# Add obs
yearsObs = dates[59:59+44]
axs[0].plot(yearsObs, emoPerRegion[0],linewidth=2,color='black')
axs[0].plot(yearsObs, efoPerRegion[0],linewidth=2,color='black')
axs[1].plot(yearsObs, emoPerRegion[1],linewidth=2,color='black')
axs[1].plot(yearsObs, efoPerRegion[1],linewidth=2,color='black')
axs[2].plot(yearsObs, emoPerRegion[2],linewidth=2,color='black')
axs[2].plot(yearsObs, efoPerRegion[2],linewidth=2,color='black')
    
cbax = fig.add_axes([0.85, 0.1, 0.03, 0.78])
cb = ColorbarBase(cbax, cmap=cmap, norm=norm, orientation='vertical')
cb.set_label("Melt season length (days)", labelpad=15)
axs[0].text(0.01, 0.92, 'a) LIA-N', transform=axs[0].transAxes)
axs[1].text(0.01, 0.92, 'b) QEI', transform=axs[1].transAxes)
axs[2].text(0.01, 0.92, 'c) CAA-S', transform=axs[2].transAxes)
axs[1].set_ylabel('Julian day')
axs[2].set_xlabel('Time')

plt.show()
